from __future__ import annotations

from collections import defaultdict

from docextract.contracts import Contract, ParseSnapshot, SourceAtom, SourceRecord, UnresolvedItem, digest
from docextract.geometry import bounds
from docextract.models import FieldCandidate, WorkUnit
from docextract.schema import Schema, join, parts


class SectionBinding(Contract):
    path: str
    label: str
    field: str
    continued_label: str | None = None


class TableBinding(Contract):
    path: str
    columns: dict[str, str]
    key: str
    end_label: str = "End items"
    continuation_marker: str = "Items continue on next page"
    continuation_columns: tuple[str, ...] = ()


class BindingPlan(Contract):
    version: str = "bindings/1"
    fields: dict[str, str] = {}
    section: SectionBinding | None = None
    tables: tuple[TableBinding, ...] = ()


class PlannedWork(Contract):
    tasks: tuple[WorkUnit, ...]
    examined_line_ids: tuple[str, ...]
    claimed_line_ids: tuple[str, ...]
    unresolved: tuple[UnresolvedItem, ...]
    array_scopes: tuple[tuple[str, str | None], ...] = ()


def scalar_leaves(schema: Schema, pointer: str = "") -> list[tuple[str, dict]]:
    node = schema.node(pointer)
    if schema.kind(node) == "object":
        return [
            leaf for key in node.get("properties", {}) for leaf in scalar_leaves(schema, join(pointer, key))
        ]
    if schema.kind(node) == "array" and schema.kind(node["items"]) == "object":
        return []
    return [(pointer, node)]


def object_arrays(schema: Schema, pointer: str = "") -> list[str]:
    node = schema.node(pointer)
    if schema.kind(node) == "object":
        return [
            path for key in node.get("properties", {}) for path in object_arrays(schema, join(pointer, key))
        ]
    if schema.kind(node) == "array" and schema.kind(node["items"]) == "object":
        return [pointer, *object_arrays(schema, pointer + "/*")]
    return []


class Planner:
    def __init__(self, schema: Schema, plan: BindingPlan):
        self.schema = schema
        self.plan = plan
        self._validate_plan()

    def _validate_plan(self):
        if self.plan.version != "bindings/1":
            raise ValueError("Unsupported binding-plan version")
        for pointer in self.plan.fields:
            if self.schema.kind(self.schema.node(pointer)) == "object" or "*" in parts(pointer):
                raise ValueError(f"Bind scalar fields individually: {pointer}")
        section = self.plan.section
        if section:
            if len(parts(section.path)) != 1 or self.schema.kind(self.schema.node(section.path)) != "array":
                raise ValueError("B0 sections must be a root array")
            self.schema.node(join(f"{section.path}/*", section.field))
        for table in self.plan.tables:
            item = self.schema.node(f"{table.path}/*")
            if self.schema.kind(item) != "object" or table.key not in table.columns:
                raise ValueError("A table requires object items and a bound key column")
            if "*" in parts(table.path) and (
                not section or table.path != f"{section.path}/*/{parts(table.path)[-1]}"
            ):
                raise ValueError("B0 nested tables require one declared parent section")
            for column in table.columns:
                if column not in item["properties"]:
                    raise ValueError(f"Unknown table field: {column}")
            if not set(table.continuation_columns) <= table.columns.keys():
                raise ValueError("Unknown continuation column")

    def build(self, snapshot: ParseSnapshot, *, apply_defaults: bool = False) -> PlannedWork:
        self.snapshot = snapshot
        self.atoms = snapshot.atom_index()
        self.lines = [atom for page in snapshot.pages for atom in page.atoms if atom.kind == "line"]
        self.cells = {
            page.page_index: tuple(atom for atom in page.atoms if atom.kind == "cell")
            for page in snapshot.pages
        }
        self.claimed: set[str] = set()
        self.issues: list[UnresolvedItem] = []
        self.records: list[tuple[SourceRecord, dict[str, FieldCandidate]]] = []
        self.array_scopes: set[tuple[str, str | None]] = set()
        self.header_columns: dict[str, list[tuple[str, float]]] = {}
        self.parent: SourceRecord | None = None
        root = self._root_fields()
        active: tuple[TableBinding, list[tuple[str, float]], int] | None = None
        pending: tuple[str, int, list[tuple[str, float]]] | None = None
        for line in self.lines:
            previous_parent = self.parent
            if self._section(line):
                active = None
                if self.parent != previous_parent:
                    if pending:
                        self._unfinished_continuation(pending[0])
                    pending = None
                continue
            header = self._header(line)
            if header:
                table, columns = header
                if pending and pending[2] != columns:
                    self._unfinished_continuation(pending[0])
                    pending = None
                self.header_columns[table.path] = columns
                parent = self.parent.record_id if "*" in parts(table.path) and self.parent else None
                if "*" not in parts(table.path) or parent:
                    self.array_scopes.add((table.path, parent))
                active = (*header, line.page_index)
                self.claimed.add(line.atom_id)
                continue
            if active is None:
                continue
            table, columns, header_page = active
            if line.text == table.end_label:
                if pending:
                    self._unfinished_continuation(pending[0])
                    pending = None
                active = None
                self.claimed.add(line.atom_id)
                continue
            if line.text == table.continuation_marker:
                if self.records and self.records[-1][0].schema_pointer == table.path:
                    pending = (self.records[-1][0].record_id, line.page_index, columns)
                self.claimed.add(line.atom_id)
                active = None
                continue
            if line.page_index != header_page:
                continue
            fields = self._row_fields(line, columns, table)
            key = fields[join("", table.key)]
            if key.lexeme:
                if pending:
                    self._unfinished_continuation(pending[0])
                self._new_row(line, table, fields)
                pending = None
            elif any(field.lexeme for field in fields.values()):
                if pending and line.page_index == pending[1] + 1:
                    self._continue_row(line, table, fields, pending[0])
                    pending = None
                else:
                    self.issues.append(
                        UnresolvedItem(
                            pointer=table.path,
                            reason="unresolved keyless table row",
                            source_atom_ids=(line.atom_id,),
                        )
                    )
            self.claimed.add(line.atom_id)
        if pending:
            self._unfinished_continuation(pending[0])
        bound_arrays = {table.path for table in self.plan.tables}
        if self.plan.section:
            bound_arrays.add(self.plan.section.path)
        for pointer in object_arrays(self.schema):
            if pointer not in bound_arrays:
                self.issues.append(
                    UnresolvedItem(pointer=pointer, reason="no record discovery binding for array")
                )
        for table in self.plan.tables:
            if table.path not in self.header_columns:
                self.issues.append(
                    UnresolvedItem(pointer=table.path, reason="no declared table header found")
                )
        tasks = [WorkUnit(task_id="root", order=0, fields=tuple(root), apply_defaults=apply_defaults)]
        for index, (record, fields) in enumerate(self.records, 1):
            tasks.append(
                WorkUnit(
                    task_id=record.record_id,
                    record=record,
                    order=index,
                    fields=tuple(fields.values()),
                    apply_defaults=apply_defaults,
                )
            )
        return PlannedWork(
            tasks=tuple(tasks),
            examined_line_ids=tuple(line.atom_id for line in self.lines),
            claimed_line_ids=tuple(sorted(self.claimed)),
            unresolved=tuple(self.issues),
            array_scopes=tuple(sorted(self.array_scopes, key=str)),
        )

    def _unfinished_continuation(self, record_id):
        record = next(record for record, _ in self.records if record.record_id == record_id)
        self.issues.append(
            UnresolvedItem(
                pointer=record.schema_pointer,
                reason="unresolved continuation fragment",
                source_atom_ids=record.anchor_atom_ids,
            )
        )

    def _root_fields(self) -> list[FieldCandidate]:
        fields = []
        for pointer, node in scalar_leaves(self.schema):
            label = self.plan.fields.get(pointer)
            if label is None:
                self.issues.append(UnresolvedItem(pointer=pointer, reason="no field binding provided"))
                fields.append(FieldCandidate(pointer=pointer, schema_node=node, uncertainty="ambiguous"))
                continue
            matches = [line for line in self.lines if label and line.text.startswith(label + ":")]
            if len(matches) > 1:
                self.issues.append(
                    UnresolvedItem(
                        pointer=pointer,
                        reason="multiple competing label occurrences",
                        source_atom_ids=tuple(line.atom_id for line in matches),
                    )
                )
                fields.append(
                    FieldCandidate(
                        pointer=pointer,
                        schema_node=node,
                        uncertainty="ambiguous",
                        source_atom_ids=tuple(line.atom_id for line in matches),
                    )
                )
                continue
            line = matches[0] if matches else None
            fields.append(self._label_field(pointer, node, line, label))
        return fields

    def _label_field(self, pointer, node, line, label) -> FieldCandidate:
        if line is None:
            return FieldCandidate(pointer=pointer, schema_node=node)
        self.claimed.add(line.atom_id)
        text = line.text[len(label) + 1 :].strip()
        tokens = [self.atoms[atom_id] for atom_id in line.child_ids]
        character_ids = []
        offset = 0
        for token in tokens:
            for character_id in token.child_ids:
                character = self.atoms[character_id]
                if offset > len(label):
                    character_ids.append(character_id)
                offset += len(character.text)
            offset += 1
        return FieldCandidate(
            pointer=pointer,
            schema_node=node,
            lexeme=text or None,
            source_atom_ids=tuple(character_ids) if text else (line.atom_id,),
            uncertainty=None if text else "unreadable",
        )

    def _section(self, line: SourceAtom) -> bool:
        section = self.plan.section
        if not section:
            return False
        if section.continued_label and line.text.startswith(section.continued_label + ":"):
            name = line.text[len(section.continued_label) + 1 :].strip()
            current = next((fields for record, fields in self.records if record == self.parent), {})
            if not self.parent or current[join("", section.field)].lexeme != name:
                self.issues.append(
                    UnresolvedItem(
                        pointer=section.path,
                        reason="unresolved parent continuation",
                        source_atom_ids=(line.atom_id,),
                    )
                )
                self.parent = None
            self.claimed.add(line.atom_id)
            return True
        if not line.text.startswith(section.label + ":"):
            return False
        record_id = digest([self.snapshot.snapshot_id, section.path, line.atom_id])
        self.parent = SourceRecord(
            record_id=record_id,
            pointer="",
            schema_pointer=section.path,
            anchor_atom_ids=(line.atom_id,),
            discovery_method="declared_section",
        )
        self.array_scopes.add((section.path, None))
        fields = self._empty_record_fields(section.path)
        pointer = join("", section.field)
        fields[pointer] = self._label_field(
            pointer, self.schema.node(f"{section.path}/*{pointer}"), line, section.label
        )
        self.records.append((self.parent, fields))
        return True

    def _header(self, line: SourceAtom):
        tokens = set(line.child_ids)
        cells = [
            cell for cell in self.cells[line.page_index] if cell.child_ids and set(cell.child_ids) <= tokens
        ]
        for table in self.plan.tables:
            matched = [
                (key, next((cell for cell in cells if cell.text == label), None))
                for key, label in table.columns.items()
            ]
            if all(cell is not None for _, cell in matched):
                return table, sorted(
                    [(key, bounds(cell.native_polygon)[0]) for key, cell in matched], key=lambda pair: pair[1]
                )
        return None

    def _row_fields(self, line, columns, table):
        grouped = defaultdict(list)
        for atom_id in line.child_ids:
            token = self.atoms[atom_id]
            x = bounds(token.native_polygon)[0]
            key = next((key for key, start in reversed(columns) if x >= start - 4), None)
            if key:
                grouped[key].append(token)
        fields = self._empty_record_fields(table.path)
        for key in table.columns:
            tokens = grouped[key]
            pointer = join("", key)
            fields[pointer] = FieldCandidate(
                pointer=pointer,
                schema_node=self.schema.node(f"{table.path}/*{pointer}"),
                lexeme=" ".join(token.text for token in tokens) if tokens else None,
                source_atom_ids=self._characters(tokens),
            )
        return fields

    def _empty_record_fields(self, path):
        prefix = f"{path}/*"
        return {
            pointer[len(prefix) :]: FieldCandidate(pointer=pointer[len(prefix) :], schema_node=node)
            for pointer, node in scalar_leaves(self.schema, prefix)
        }

    def _new_row(self, line, table, fields):
        if "*" in parts(table.path) and self.parent is None:
            self.issues.append(
                UnresolvedItem(
                    pointer=table.path, reason="table has no resolved parent", source_atom_ids=(line.atom_id,)
                )
            )
            return
        record = SourceRecord(
            record_id=digest([self.snapshot.snapshot_id, table.path, line.atom_id]),
            pointer="",
            schema_pointer=table.path,
            parent_record_id=self.parent.record_id if "*" in parts(table.path) else None,
            anchor_atom_ids=(line.atom_id,),
        )
        self.records.append((record, fields))

    def _continue_row(self, line, table, fields, record_id):
        index = next(i for i, (record, _) in enumerate(self.records) if record.record_id == record_id)
        record, existing = self.records[index]
        present = {parts(pointer)[-1] for pointer, field in fields.items() if field.lexeme}
        if record.schema_pointer != table.path or not present <= set(table.continuation_columns):
            self.issues.append(
                UnresolvedItem(
                    pointer=table.path,
                    reason="incompatible continuation columns",
                    source_atom_ids=(line.atom_id,),
                )
            )
            return
        for pointer, field in fields.items():
            if field.lexeme:
                old = existing[pointer]
                existing[pointer] = old.model_copy(
                    update={
                        "lexeme": " ".join(text for text in (old.lexeme, field.lexeme) if text),
                        "source_atom_ids": (*old.source_atom_ids, *field.source_atom_ids),
                    }
                )
        self.records[index] = (
            record.model_copy(update={"fragment_atom_ids": (*record.fragment_atom_ids, line.atom_id)}),
            existing,
        )

    def _characters(self, tokens) -> tuple[str, ...]:
        return tuple(char_id for token in tokens for char_id in token.child_ids)
