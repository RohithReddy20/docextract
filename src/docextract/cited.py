"""Schema-shaped values with per-value source citations, checked against native evidence.

The model writes the user's JSON shape with {"value": ..., "source": id} at each scalar. Code types
each value against the original schema, locates it inside the cited native text, and identifies
records by the source occurrences they cite. Values the model could not support stay visible as such.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from contextlib import closing
from datetime import date
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, Protocol

from docextract.assembly import assemble
from docextract.contracts import (
    Completion,
    Contract,
    Coverage,
    ExtractionResult,
    FieldEvidence,
    SourceRecord,
    UnresolvedItem,
    Usage,
    digest,
    save_json,
)
from docextract.engine import IncompleteGeneration
from docextract.interpretation import (
    EvidenceUnit,
    UnparseableOutput,
    evidence_units,
    render_evidence,
    windows,
)
from docextract.jsonutil import model_json
from docextract.models import GenerationResult, InputTooLong, TaskOutput, WorkUnit
from docextract.parsing.pdfium import PdfiumParser
from docextract.partition import factor_notes as factor_notes_in
from docextract.partition import inline, partition
from docextract.planning import object_arrays, scalar_leaves
from docextract.routing import route
from docextract.runtime import Journal, exclusive_workspace
from docextract.schema import Schema, join, parts
from docextract.tables import discover_tables

WRAPPER_KEYS = {"value", "source", "v", "s"}
NUMBER = re.compile(r"\(?[-+−]?(?:[$€£¥]\s?)?(?:\d[\d,]*(?:\.\d+)?|\.\d+)\)?%?")
MONTHS = {
    name: index
    for index, names in enumerate(
        (
            ("january", "jan"),
            ("february", "feb"),
            ("march", "mar"),
            ("april", "apr"),
            ("may",),
            ("june", "jun"),
            ("july", "jul"),
            ("august", "aug"),
            ("september", "sep", "sept"),
            ("october", "oct"),
            ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
MONTH = "|".join(sorted(MONTHS, key=len, reverse=True))
DATES = (
    (re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"), ("y", "m", "d")),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b"), ("m", "d", "y")),
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"), ("d", "m", "y")),
    (re.compile(rf"\b({MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), ("M", "d", "y")),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH})\.?,?\s+(\d{{4}})\b", re.I), ("d", "M", "y")),
)


class Pending(Exception):
    """A deferring backend has queued this task; extraction resumes once its output is stored."""

    def __init__(self, task_ids=()):
        super().__init__(f"{len(task_ids)} generation task(s) pending")
        self.task_ids = tuple(task_ids)


class Unanswered(Exception):
    """No answer was generated for this window (for example a run stopped at its deadline)."""


class Absent(Exception):
    """A null value for a non-nullable field: the model is reporting absence, not a value."""


class CitedTask(Contract):
    task_id: str
    original_schema: dict[str, Any]
    units: tuple[EvidenceUnit, ...]
    # A self-contained field group of the original schema when the whole schema is too large to prompt.
    schema_part: dict[str, Any] | None = None
    # Images of the window's recognized (scanned) pages, for a model that reads images.
    page_images: tuple[str, ...] = ()
    # Longest side the images are shown at (smaller images cost fewer prompt tokens); None: as stored.
    page_image_side: int | None = None
    # "cited": a source reference with every value; "values": plain values that code locates.
    protocol: Literal["cited", "values"] = "cited"

    @property
    def prompt_schema(self) -> dict[str, Any]:
        return self.schema_part or self.original_schema


class CitedBackend(Protocol):
    identity: dict[str, Any]

    def generate_cited(self, task: CitedTask) -> GenerationResult: ...


class CitedLeaf(Contract):
    pointer: str
    value: Any = None
    atom_ids: tuple[str, ...] = ()
    lexeme: str | None = None
    transform: str
    support: Literal["character", "cell", "none"] = "none"
    position: int | None = None
    cited: bool = False
    sources: tuple[str, ...] = ()
    # Located without a citation at the only place in the window where the value is printed.
    distinct: bool = False


class CitedRecord(Contract):
    index: int
    path: str
    parent: int | None = None
    leaves: tuple[CitedLeaf, ...] = ()
    empty_arrays: tuple[str, ...] = ()


class CitedWindow(Contract):
    root_leaves: tuple[CitedLeaf, ...] = ()
    root_empty_arrays: tuple[str, ...] = ()
    records: tuple[CitedRecord, ...] = ()
    rejected: tuple[str, ...] = ()
    copied_rows: int = 0


def parse_number(text: str) -> int | float:
    cleaned = text.strip().replace("−", "-")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = re.sub(r"[()$€£¥,%\s]", "", cleaned)
    try:
        number = Decimal(cleaned)
    except InvalidOperation as error:
        raise ValueError(f"Not a number: {text!r}") from error
    if not number.is_finite():
        raise ValueError("Non-finite number")
    number = -number if negative else number
    return int(number) if number == number.to_integral_value() and "." not in cleaned else float(number)


def nullable(node: dict) -> bool:
    types = node.get("type")
    return "null" in (types if isinstance(types, list) else [types])


def coerce(value: Any, node: dict, schema: Schema) -> Any:
    """Type a model value against the original schema node; never guesses dates or locales."""
    if value is None:
        if nullable(node):
            return None
        raise ValueError("null for a non-nullable field")
    kind = schema.kind(node)
    if kind == "array":
        if not isinstance(value, list):
            raise ValueError("expected a list")
        return [coerce(item, schema.resolve(node["items"]), schema) for item in value]
    if kind == "string":
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("expected a string")
        typed = value if isinstance(value, str) else str(value)
    elif kind in ("integer", "number"):
        if isinstance(value, bool):
            raise ValueError("expected a number")
        typed = parse_number(value) if isinstance(value, str) else value
        if not isinstance(typed, (int, float)):
            raise ValueError("expected a number")
        if kind == "integer":
            if isinstance(typed, float) and not typed.is_integer():
                raise ValueError("expected an integer")
            typed = int(typed)
    elif kind == "boolean":
        if isinstance(value, str) and value.strip().casefold() in ("true", "yes", "false", "no"):
            typed = value.strip().casefold() in ("true", "yes")
        elif isinstance(value, bool):
            typed = value
        else:
            raise ValueError("expected a boolean")
    elif kind == "null":
        raise ValueError("expected null")
    else:
        raise ValueError(f"unsupported scalar kind {kind}")
    choices = node.get("enum", [node["const"]] if "const" in node else None)
    if choices is not None and typed not in choices:
        folded = [
            c
            for c in choices
            if isinstance(c, str) and isinstance(typed, str) and c.casefold() == typed.casefold()
        ]
        if len(folded) != 1:
            raise ValueError("value outside the schema enum")
        typed = folded[0]
    return typed


def _virtual(units: list[EvidenceUnit]) -> tuple[str, tuple[str | None, ...], tuple[int, ...]]:
    """Joined cell text with each character's source atom and a document-order position."""
    text, offsets, positions = "", [], []
    for unit in units:
        base = int(unit.alias[1:]) * 100_000
        if text:
            text += " "
            offsets.append(None)
            positions.append(positions[-1])
        text += unit.text
        offsets.extend(unit.offset_atom_ids)
        positions.extend(base + index for index in range(len(unit.text)))
    return text, tuple(offsets), tuple(positions)


def _dates(text: str):
    for pattern, order in DATES:
        for match in pattern.finditer(text):
            fields = dict(zip(order, match.groups(), strict=True))
            try:
                year = int(fields["y"])
                year += 2000 if len(fields["y"]) == 2 else 0
                month = MONTHS[fields["M"].casefold()] if "M" in fields else int(fields["m"])
                yield date(year, month, int(fields["d"])).isoformat(), match.start(), match.end()
            except (ValueError, KeyError):
                continue


def find_all(value: Any, kind: str, text: str):
    """Every printed occurrence of a typed value, in text order: (start, end, transform)."""
    if isinstance(value, bool) or value is None or isinstance(value, list):
        return
    if kind in ("integer", "number") and isinstance(value, (int, float)):
        for match in NUMBER.finditer(text):
            lexeme = match.group().strip()
            try:
                printed = parse_number(lexeme)
            except ValueError:
                continue
            if printed == value or (lexeme.endswith("%") and abs(printed / 100 - value) < 1e-12):
                yield match.start(), match.end(), "verbatim" if lexeme == str(value) else "number_normalized"
        return
    if not isinstance(value, str) or not value.strip():
        return
    exact = [match.start() for match in re.finditer(re.escape(value), text)]
    for begin in exact:
        yield begin, begin + len(value), "verbatim"
    if exact:
        return
    tokens = value.split()
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens), re.I)
    found = False
    for match in pattern.finditer(text):
        found = True
        yield match.start(), match.end(), "space_or_case_normalized"
    if not found and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        for iso, begin, end in _dates(text):
            if iso == value:
                yield begin, end, "date_normalized"


def find_value(value: Any, kind: str, text: str) -> tuple[int, int, str] | None:
    """Locate the first printed occurrence of a typed value; returns (start, end, transform)."""
    return next(find_all(value, kind, text), None)


def _atoms(offsets, start, end) -> tuple[str, ...]:
    return tuple(dict.fromkeys(atom for atom in offsets[start:end] if atom))


class _Window:
    """Parse one model response against the schema and the window's evidence units."""

    def __init__(self, schema: Schema, units: tuple[EvidenceUnit, ...], complete_tables=False):
        self.schema = schema
        self.units = {unit.alias: unit for unit in units}
        self.ordered = units
        self.rejected: list[str] = []
        self.records: list[CitedRecord | None] = []
        self.complete_tables = complete_tables

    def parse(self, data) -> CitedWindow:
        if not isinstance(data, dict):
            raise ValueError("Expected one JSON object")
        leaves, empties = self._object(data, "", "", None)
        pending = {None: leaves, **{record.index: list(record.leaves) for record in self.records}}
        self._resolve(pending)
        copied = self._complete_tables(pending) if self.complete_tables else 0
        records = tuple(
            record.model_copy(update={"leaves": tuple(pending[record.index])}) for record in self.records
        )
        return CitedWindow(
            root_leaves=tuple(pending[None]),
            root_empty_arrays=tuple(empties),
            records=records,
            rejected=tuple(self.rejected),
            copied_rows=copied,
        )

    def _complete_tables(self, pending: dict[int | None, list[CitedLeaf]]) -> int:
        """Copy regular-table rows the model skipped, using the column mapping its own cited rows show.

        At least two records must cite cells of exactly one row each with a consistent field-to-column
        mapping and a shared parent. Rows already cited by anything are never copied; a row whose
        numeric cells do not parse is left unclaimed rather than guessed.
        """
        tables = discover_tables(self.ordered)
        if not tables:
            return 0
        cell_at, alias_of = {}, {}
        for unit in self.ordered:
            for atom in unit.offset_atom_ids:
                if atom:
                    alias_of[atom] = unit.alias
        for table_index, table in enumerate(tables):
            for row_index, row in enumerate((table.header_aliases, *table.rows)):
                for column, alias in enumerate(row):
                    cell_at[alias] = (table_index, row_index, column)
        covered = {
            cell_at[alias_of[atom]][:2]
            for leaves in pending.values()
            for leaf in leaves
            for atom in leaf.atom_ids
            if atom in alias_of and alias_of[atom] in cell_at
        }
        examples: dict[tuple, list] = {}
        for record in self.records:
            places = {}
            for leaf in pending[record.index]:
                cells = {cell_at.get(alias_of.get(atom)) for atom in leaf.atom_ids}
                if len(cells) != 1 or None in cells:
                    break
                places[leaf.pointer] = (next(iter(cells)), leaf.transform)
            else:
                rows = {place[0][:2] for place in places.values()}
                if len(rows) == 1 and places:
                    table_index = next(iter(rows))[0]
                    key = (table_index, record.path, record.parent)
                    examples.setdefault(key, []).append(
                        {pointer: (place[0][2], place[1]) for pointer, place in places.items()}
                    )
        copied = 0
        for (table_index, path, parent), shown in examples.items():
            if len(shown) < 2:
                continue
            mapping: dict[str, int] = {}
            transforms: dict[str, list[str]] = {}
            consistent = True
            for example in shown:
                for pointer, (column, transform) in example.items():
                    consistent &= mapping.setdefault(pointer, column) == column
                    transforms.setdefault(pointer, []).append(transform)
            if not consistent or len(set(mapping.values())) != len(mapping):
                continue
            table = tables[table_index]
            for row_index, row in enumerate(table.rows, start=1):
                if (table_index, row_index) in covered:
                    continue
                leaves = self._copied_row(path, row, mapping, transforms)
                if leaves is None:
                    continue
                index = len(self.records)
                self.records.append(CitedRecord(index=index, path=path, parent=parent, leaves=()))
                pending[index] = leaves
                covered.add((table_index, row_index))
                copied += 1
        return copied

    def _copied_row(self, path, row, mapping, transforms) -> list[CitedLeaf] | None:
        leaves = []
        for pointer, column in mapping.items():
            unit = self.units[row[column]]
            text = unit.text.strip()
            if not text:
                continue
            kind = self.schema.kind(self.schema.node(path + "/*" + pointer))
            usual = max(set(transforms[pointer]), key=transforms[pointer].count)
            try:
                if kind in ("integer", "number"):
                    value = parse_number(text)
                    if kind == "integer":
                        if isinstance(value, float) and not value.is_integer():
                            return None
                        value = int(value)
                elif kind == "string" and usual == "date_normalized":
                    value = next(iso for iso, _, _ in _dates(text))
                elif kind == "string":
                    value = text
                else:
                    continue  # booleans and other kinds are never guessed from a copied cell
            except (ValueError, StopIteration):
                return None
            atoms = _atoms(unit.offset_atom_ids, 0, len(unit.offset_atom_ids))
            leaves.append(
                CitedLeaf(
                    pointer=pointer,
                    value=value,
                    atom_ids=atoms,
                    lexeme=unit.text,
                    transform="copied_table_row",
                    support="character",
                    position=int(unit.alias[1:]),
                    cited=True,
                    sources=(unit.alias,),
                )
            )
        return leaves or None

    def _object(self, obj: dict, schema_pointer: str, relative: str, owner: int | None):
        leaves, empties = [], []
        properties = self.schema.node(schema_pointer).get("properties", {})
        for key, value in obj.items():
            here = join(relative, key)
            if key not in properties:
                self.rejected.append(f"{here}: not a schema property")
                continue
            child_pointer = join(schema_pointer, key)
            child = self.schema.node(child_pointer)
            kind = self.schema.kind(child)
            if kind == "object":
                if isinstance(value, dict) and not (value.keys() <= WRAPPER_KEYS and "value" in value):
                    inner, inner_empty = self._object(value, child_pointer, here, owner)
                    leaves += inner
                    empties += inner_empty
                elif value is not None and value != {"value": None}:
                    self.rejected.append(f"{here}: expected an object")
            elif kind == "array" and self.schema.kind(child["items"]) == "object":
                if value is None:
                    continue
                if not isinstance(value, list):
                    self.rejected.append(f"{here}: expected a list of records")
                elif not value:
                    empties.append(here)
                for item in value if isinstance(value, list) else ():
                    if not isinstance(item, dict):
                        self.rejected.append(f"{here}: record is not an object")
                        continue
                    index = len(self.records)
                    self.records.append(None)
                    inner, inner_empty = self._object(item, child_pointer + "/*", "", index)
                    self.records[index] = CitedRecord(
                        index=index,
                        path=child_pointer,
                        parent=owner,
                        leaves=tuple(inner),
                        empty_arrays=tuple(inner_empty),
                    )
            else:
                try:
                    leaves.append(self._leaf(value, child, kind, here))
                except Absent:
                    continue  # the model reported the field as not present
                except ValueError as error:
                    self.rejected.append(f"{here}: {error}")
        return leaves, empties

    def _leaf(self, raw: Any, node: dict, kind: str, pointer: str) -> CitedLeaf:
        def wrapped(item):
            return (
                isinstance(item, dict) and item.keys() <= WRAPPER_KEYS and bool(item.keys() & {"value", "v"})
            )

        def refs(item):
            source = item.get("source", item.get("s"))
            return [source] if isinstance(source, str) else source if isinstance(source, list) else []

        sources: list[str] = []
        if kind == "array" and isinstance(raw, list) and len(raw) == 1 and wrapped(raw[0]):
            if isinstance(raw[0].get("value", raw[0].get("v")), list):
                raw = raw[0]  # one wrapper holding the whole primitive list
        if wrapped(raw):
            value, sources = raw.get("value", raw.get("v")), refs(raw)
        elif (
            kind == "array" and isinstance(raw, list) and raw and all(isinstance(item, dict) for item in raw)
        ):
            if not all(wrapped(item) for item in raw):
                raise ValueError("malformed list item")
            value = [item.get("value", item.get("v")) for item in raw]
            sources = [ref for item in raw for ref in refs(item)]
        else:
            value = raw
        if value is None and not nullable(node):
            raise Absent(pointer)
        typed = coerce(value, node, self.schema)
        if (
            isinstance(typed, str)
            and typed in self.units
            and not any(typed in unit.text for unit in self.ordered)
        ):
            raise ValueError("a source reference was returned as the value")
        valid = [source for source in sources if isinstance(source, str) and source in self.units]
        if len(valid) != len(sources):
            self.rejected.append(f"{pointer}: citation to an unknown source ref ignored")
        return CitedLeaf(
            pointer=pointer, value=typed, transform="uncited", sources=tuple(dict.fromkeys(valid))
        )

    def _options(self, leaf: CitedLeaf, kind: str):
        """Candidate occurrences: inside the cited cells, or anywhere in the window when uncited."""
        groups = (
            [[self.units[alias] for alias in leaf.sources]] if leaf.sources else [[u] for u in self.ordered]
        )
        options = []
        for units in groups:
            text, offsets, positions = _virtual(units)
            for start, end, transform in find_all(leaf.value, kind, text):
                atoms = _atoms(offsets, start, end)
                if atoms:
                    options.append((positions[start], atoms, text[start:end], transform))
        return options

    def _resolve(self, pending: dict[int | None, list[CitedLeaf]]) -> None:
        """Choose one occurrence per value: unused by sibling records first, then nearest the record."""
        used: dict[tuple[str, str], set[str]] = {}
        for owner, leaves in pending.items():
            path = self.records[owner].path if owner is not None else ""
            choices, counts = [], []
            for leaf in leaves:
                kind = self.schema.kind(self.schema.node(self._schema_pointer(owner, leaf.pointer)))
                options = self._options(leaf, kind)
                taken = used.get((path, leaf.pointer), set())
                choices.append([option for option in options if not set(option[1]) & taken] or options)
                counts.append(len(options))
            unique = sorted(options[0][0] for options in choices if len(options) == 1)
            center = unique[len(unique) // 2] if unique else None
            for index, (leaf, options, count) in enumerate(zip(leaves, choices, counts, strict=True)):
                if options:
                    position, atoms, lexeme, transform = min(
                        options,
                        key=lambda option: (abs(option[0] - center) if center is not None else 0, option[0]),
                    )
                    used.setdefault((path, leaf.pointer), set()).update(atoms)
                    leaves[index] = leaf.model_copy(
                        update={
                            "atom_ids": atoms,
                            "lexeme": lexeme,
                            "transform": transform if leaf.sources else "located_without_citation",
                            "support": "character",
                            "position": position // 100_000,
                            "cited": bool(leaf.sources),
                            "distinct": not leaf.sources and count == 1,
                        }
                    )
                elif leaf.sources:
                    units = [self.units[alias] for alias in leaf.sources]
                    text, offsets, positions = _virtual(units)
                    leaves[index] = leaf.model_copy(
                        update={
                            "atom_ids": _atoms(offsets, 0, len(offsets)),
                            "lexeme": text,
                            "transform": "model_value_unverified",
                            "support": "cell",
                            "position": positions[0] // 100_000 if positions else None,
                            "cited": True,
                        }
                    )
                else:
                    leaves[index] = leaf.model_copy(update={"transform": "model_value_unsupported"})

    def _schema_pointer(self, owner, pointer):
        base = ""
        if owner is not None:
            base = self.records[owner].path + "/*"
        return base + pointer


def cited_output_schema(schema: dict) -> dict:
    """JSON Schema of a cited answer, for constrained decoding: user shape, value/source leaves.

    Leaves keep the user's type, enum and const; other constraints stay with the original-schema
    validation. Properties are optional, so absent facts can be omitted.
    """
    flat = inline(schema)
    source = {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}

    def kinds(node):
        value = node.get("type", [])
        return value if isinstance(value, list) else [value]

    def nullable_of(node, shape):
        return {"anyOf": [shape, {"type": "null"}]} if "null" in kinds(node) else shape

    def leaf(node):
        value = {key: node[key] for key in ("type", "enum", "const") if key in node}
        if "enum" in value and "null" in kinds(node) and None not in value["enum"]:
            value["enum"] = [*value["enum"], None]
        return {
            "type": "object",
            "properties": {"value": value, "source": source},
            "required": ["value", "source"],
            "additionalProperties": False,
        }

    def convert(node):
        if "object" in kinds(node) and node.get("properties"):
            shape = {
                "type": "object",
                "properties": {key: convert(child) for key, child in node["properties"].items()},
                "additionalProperties": False,
            }
            return nullable_of(node, shape)
        if "array" in kinds(node):
            items = node.get("items", {})
            nested = "object" in kinds(items) and items.get("properties")
            return nullable_of(node, {"type": "array", "items": convert(items) if nested else leaf(items)})
        return leaf(node)

    return convert(flat)


def parse_cited(
    data: Any, schema: Schema, units: tuple[EvidenceUnit, ...], complete_tables=False
) -> CitedWindow:
    return _Window(schema, units, complete_tables).parse(data)


def _agree(left, right) -> bool:
    """Two values of one field name the same thing: equal numbers, or text equal without case,
    spacing or punctuation, or nearly equal (OCR noise)."""
    if isinstance(left, str) and isinstance(right, str):
        a, b = (re.sub(r"[\W_]+", "", text.casefold()) for text in (left, right))
        return a == b or (min(len(a), len(b)) >= 8 and SequenceMatcher(None, a, b).ratio() >= 0.85)
    return left == right


class _Node:
    def __init__(self, record_id, path, parent, position, serial=0):
        self.record_id, self.path, self.parent, self.position = record_id, path, parent, position
        self.serial = serial
        self.window = -1
        self.anchor: set[str] = set()
        self.leaves: dict[str, list[tuple[CitedLeaf, int]]] = {}
        self.empty_arrays: set[str] = set()


class CitedState:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.nodes: dict[str, _Node] = {}
        # Records by the source atoms they cite, so matching a record is not a scan of all records.
        self.by_atom: dict[str, set[str]] = {}
        self.root = _Node(None, "", None, (-1, 0))
        self.root_empty: dict[str, int] = {}
        self.windows = 0
        self.failed_windows = 0
        self.examined: set[str] = set()
        self.claimed: set[str] = set()
        self.unresolved: list[UnresolvedItem] = []
        self.rejected = 0
        self.copied = 0

    def apply(self, task: CitedTask, window: CitedWindow) -> None:
        number = self.windows
        self.windows += 1
        self.copied += window.copied_rows
        self.examined.update(unit.line_id for unit in task.units)
        lines = {atom: unit.line_id for unit in task.units for atom in unit.offset_atom_ids if atom}
        for reason in window.rejected:
            self.rejected += 1
            self.unresolved.append(UnresolvedItem(pointer="", reason="rejected value: " + reason))
        for path in window.root_empty_arrays:
            self.root_empty[path] = self.root_empty.get(path, 0) + 1
        self._leaves(self.root, window.root_leaves, number, lines)
        start = min((int(unit.alias[1:]) for unit in task.units), default=0)
        local: dict[int, _Node] = {}
        merged: set[str] = set()
        for record in window.records:
            parent = local[record.parent] if record.parent is not None else None
            parent_id = parent.record_id if parent else None
            # Cited characters identify a record; so do values printed only once in the window.
            anchor = {atom for leaf in record.leaves if leaf.cited or leaf.distinct for atom in leaf.atom_ids}
            positions = [leaf.position for leaf in record.leaves if leaf.position is not None]
            # Records in one response are distinct; only an earlier window's record can be the same one.
            candidates = {record_id for atom in anchor for record_id in self.by_atom.get(atom, ())}
            match = next(
                (
                    node
                    for node in sorted((self.nodes[i] for i in candidates), key=lambda node: node.serial)
                    if node.window < number
                    and node.record_id not in merged
                    and node.path == record.path
                    and node.parent == parent_id
                ),
                None,
            )
            if match is None:
                position = (min(positions) if positions else start, number * 100000 + record.index)
                identity = sorted(anchor)[:1] + [task.task_id, record.index]
                record_id = digest({"path": record.path, "parent": parent_id, "anchor": identity})
                if record_id not in self.nodes:
                    serial = len(self.nodes)
                    self.nodes[record_id] = _Node(record_id, record.path, parent_id, position, serial)
                match = self.nodes[record_id]
                match.window = number
            merged.add(match.record_id)
            match.anchor |= anchor
            for atom in anchor:
                self.by_atom.setdefault(atom, set()).add(match.record_id)
            match.empty_arrays.update(record.empty_arrays)
            self._leaves(match, record.leaves, number, lines)
            local[record.index] = match

    def _leaves(self, node: _Node, leaves, number, lines):
        for leaf in leaves:
            node.leaves.setdefault(leaf.pointer, []).append((leaf, number))
            self.claimed.update(lines[atom] for atom in leaf.atom_ids if atom in lines)

    def _choose(self, node: _Node, pointer: str) -> CitedLeaf | None:
        entries = node.leaves.get(pointer)
        if not entries:
            return None
        rank = {"character": 0, "cell": 1, "none": 2}
        values = {digest(leaf.value) for leaf, _ in entries}
        best = min(entries, key=lambda entry: (rank[entry[0].support], entry[1]))[0]
        if len(values) > 1:
            self.unresolved.append(
                UnresolvedItem(
                    pointer=pointer,
                    reason="conflicting values across windows; best-supported kept",
                    source_atom_ids=best.atom_ids,
                )
            )
        return best

    def _values(self, node: _Node) -> dict[str, Any]:
        """The best-supported non-null value of each field of a record (no side effects)."""
        rank = {"character": 0, "cell": 1, "none": 2}
        found = {}
        for pointer, entries in node.leaves.items():
            best = min(entries, key=lambda entry: (rank[entry[0].support], entry[1]))[0]
            if best.value is not None:
                found[pointer] = best.value
        return found

    def merge_complementary(self) -> int:
        """Merge records that are one entity described in two places of a document.

        A record from a later window is merged into an earlier one of the same array and parent when
        their field sets differ (one is a partial view of the other, or each adds fields), they share a
        non-boolean value, and the shared fields agree (text compared without case, spacing or
        punctuation, near-identical text allowed; one disagreement tolerated when at least four agree).
        A summary table and a later detail section then yield one record per entity, while identical
        printed rows (same fields) stay separate.
        """
        families: dict[tuple, list[_Node]] = {}
        for node in self.nodes.values():
            families.setdefault((node.path, node.parent), []).append(node)
        absorbed: dict[str, str] = {}
        for family in families.values():
            family.sort(key=lambda node: node.serial)
            taken: set[tuple[str, int]] = set()
            for later in family:
                mine = self._values(later)
                for earlier in family:
                    if earlier.serial >= later.serial or earlier.window == later.window:
                        continue
                    if earlier.record_id in absorbed or (earlier.record_id, later.window) in taken:
                        continue
                    theirs = self._values(earlier)
                    shared = mine.keys() & theirs.keys()
                    # One record must add fields: a partial view (strict subset) or complementary halves.
                    if mine.keys() == theirs.keys():
                        continue
                    if not any(not isinstance(mine[key], bool) for key in shared):
                        continue
                    agree = sum(_agree(mine[key], theirs[key]) for key in shared)
                    # One misread field is tolerated when several others agree.
                    if len(shared) - agree <= (1 if agree >= 4 else 0):
                        for pointer, entries in later.leaves.items():
                            earlier.leaves.setdefault(pointer, []).extend(entries)
                        earlier.anchor |= later.anchor
                        earlier.empty_arrays |= later.empty_arrays
                        absorbed[later.record_id] = earlier.record_id
                        taken.add((earlier.record_id, later.window))
                        break
        for node in self.nodes.values():
            while node.parent in absorbed:
                node.parent = absorbed[node.parent]
        for record_id in absorbed:
            del self.nodes[record_id]
        return len(absorbed)

    def ordered(self) -> list[_Node]:
        result: list[_Node] = []
        families: dict[str | None, list[_Node]] = {}
        for node in self.nodes.values():
            families.setdefault(node.parent, []).append(node)

        def children(parent):
            for node in sorted(families.get(parent, ()), key=lambda node: (node.position, node.path)):
                result.append(node)
                children(node.record_id)

        children(None)
        return result

    def outputs(
        self, apply_defaults: bool, absent_nullable: str, document_strings=False, nulls_if_partial=False
    ):
        tasks = []
        complete = not self.failed_windows or nulls_if_partial
        for order, node in enumerate([self.root, *self.ordered()]):
            record = None
            base = ""
            if node.record_id:
                atoms = tuple(sorted(node.anchor))
                record = SourceRecord(
                    record_id=node.record_id,
                    pointer="",
                    schema_pointer=node.path,
                    parent_record_id=node.parent,
                    anchor_atom_ids=atoms,
                    discovery_method="cited_values",
                )
                base = node.path + "/*"
            fields = []
            for pointer, schema_node in scalar_leaves(self.schema, base):
                relative = pointer[len(base) :]
                leaf = self._choose(node, relative)
                common = {"pointer": relative, "record_id": node.record_id}
                if leaf is not None:
                    value = leaf.value
                    printed = re.sub(r"\s+", " ", leaf.lexeme or "").strip()
                    if (
                        document_strings
                        and isinstance(value, str)
                        and printed
                        and printed != value
                        and printed.casefold() == re.sub(r"\s+", " ", value).strip().casefold()
                    ):
                        # The model changed case or spacing of located text; the document's own text is
                        # returned (cited or found by search alike).
                        value = printed
                    fields.append(
                        FieldEvidence(
                            **common,
                            value=value,
                            status="observed",
                            source_atom_ids=leaf.atom_ids,
                            source_lexeme=leaf.lexeme,
                            transform=leaf.transform,
                            support="character"
                            if leaf.support == "character"
                            else "cell"
                            if leaf.support == "cell"
                            else "none",
                        )
                    )
                elif apply_defaults and "default" in schema_node:
                    fields.append(
                        FieldEvidence(
                            **common,
                            value=schema_node["default"],
                            status="schema_default",
                            transform="schema_default",
                        )
                    )
                elif complete or node.record_id:
                    fields.append(
                        FieldEvidence(
                            **common,
                            value=None,
                            status="absent_after_search",
                            transform="null_projection"
                            if nullable(schema_node)
                            and (
                                absent_nullable == "null"
                                or (absent_nullable == "root" and not node.record_id)
                            )
                            else None,
                        )
                    )
                else:
                    fields.append(FieldEvidence(**common, value=None, status="ambiguous"))
            tasks.append(
                (
                    WorkUnit(task_id=node.record_id or "root", record=record, order=order, fields=()),
                    TaskOutput(fields=tuple(fields)),
                )
            )
        return tasks


def _array_scopes(schema: Schema, state: CitedState) -> tuple[tuple[str, str | None], ...]:
    """Empty arrays are emitted only when the model said so or the schema requires the root key."""
    nodes = list(state.nodes.values())
    required = set(schema.resolve(schema.original).get("required", ()))
    scopes = []
    for path in object_arrays(schema):
        if "*" in path or any(node.path == path for node in nodes):
            continue
        if state.root_empty.get(path) or parts(path)[0] in required:
            scopes.append((path, None))
    for node in nodes:
        for relative in sorted(node.empty_arrays):
            path = node.path + "/*" + relative
            if not any(child.path == path and child.parent == node.record_id for child in nodes):
                scopes.append((path, node.record_id))
    return tuple(scopes)


def _process(journal, run_id, task, backend, state, fault, schema, complete_tables=False):
    journal.ensure(run_id, task.task_id)
    status, raw, saved = journal.get(run_id, task.task_id)
    if status == "committed":
        window = CitedWindow.model_validate_json(saved)
    else:
        if raw is None:
            journal.start(run_id, task.task_id)
            try:
                generation = backend.generate_cited(task)
            except InputTooLong:
                generation = GenerationResult(raw_text="", finish_reason="input_limit")
            except Pending:
                journal.release(run_id, task.task_id)
                raise
            fault("generated", task)
            journal.save_raw(run_id, task.task_id, generation)
            fault("raw_saved", task)
        else:
            generation = GenerationResult.model_validate_json(raw)
        if generation.finish_reason == "not_stored":
            raise Unanswered("No answer was generated for this window")
        if generation.finish_reason == "input_limit":
            raise InputTooLong("Task exceeds the backend input limit; no source truncation")
        if generation.finish_reason != "stop":
            raise IncompleteGeneration("Unfinished response; raw output retained, no prefix salvage")
        try:
            window = parse_cited(model_json(generation.raw_text), schema, task.units, complete_tables)
        except ValueError as error:
            raise UnparseableOutput(str(error)) from error
    # Applied in place: every failure path above raises before the state changes.
    state.apply(task, window)
    updated = state
    if status != "committed":
        journal.validated(run_id, task.task_id)
        fault("validated", task)
        journal.commit(run_id, task.task_id, window)
        fault("committed", task)
    return updated


def extract_cited(
    document: Path,
    schema: dict,
    *,
    workspace: Path,
    backend: CitedBackend,
    parser=None,
    max_window_chars=16000,
    overlap_lines=2,
    max_schema_chars=24000,
    max_window_units=None,
    complete_tables=False,
    apply_defaults=False,
    absent_nullable: Literal["null", "omit", "root"] = "null",
    fault=None,
    prompt_code: dict[str, str] | None = None,
    document_strings: bool = False,
    page_images: Path | None = None,
    max_window_pages: int = 4,
    nulls_if_partial: bool = False,
    protocol: Literal["cited", "values"] = "cited",
    route_groups: bool = False,
    prompt_budget_chars: int | None = None,
    factor_notes: bool = False,
    merge_entities: bool = False,
    page_image_side: int | None = None,
    page_image_side_split: int | None = None,
) -> ExtractionResult:
    compiled = Schema(schema)
    compiled.require_supported()
    fault = fault or (lambda stage, task: None)
    with exclusive_workspace(workspace):
        snapshot = (parser or PdfiumParser()).parse(document)
        units = evidence_units(snapshot)
        scanned = {
            page.page_index
            for page in snapshot.pages
            if {"ocr_text", "sparse_native_text", "scanned_text_layer"} & set(page.quality_flags)
        }
        # Page images change nothing for a document without scanned pages, so its windows and answer
        # identities stay those of the text-only variant and stored answers are shared, not regenerated.
        if not scanned:
            page_images = None
        # The model may read a shortened copy of a large schema (repeated guidance written once as
        # notes); answers are still validated against the original.
        source, factored = schema, False
        if factor_notes:
            shortened = factor_notes_in(schema)
            if len(json.dumps(shortened, separators=(",", ":"))) <= 0.9 * len(
                json.dumps(schema, separators=(",", ":"))
            ):
                source, factored = shortened, True
        groups = partition(source, max_schema_chars) or [source if factored else None]
        # A window is asked once per schema group, each time with its page images: a schema split into
        # more than two groups gets smaller images.
        if page_image_side_split and len(groups) > 2:
            page_image_side = page_image_side_split
        if prompt_budget_chars:
            # Windows take the prompt room the largest schema group leaves. Characters outside ASCII
            # (for example Japanese) count about 2.8 times, being closer to one token each.
            text = "".join(unit.text for unit in units) or " "
            weight = 1 + 1.8 * sum(not char.isascii() for char in text) / len(text)
            largest = max(len(json.dumps(group or schema, separators=(",", ":"))) for group in groups)
            max_window_chars = max(max_window_chars, int((prompt_budget_chars - largest) / weight))
        chunks = tuple(
            windows(
                units,
                max_chars=max_window_chars,
                overlap_lines=overlap_lines,
                unit_overhead=12,
                max_units=max_window_units,
                max_pages=max_window_pages if page_images else None,
                # Page images are shown once each: windows hold whole pages without overlap.
                align_pages=bool(page_images),
            )
        )

        def images(chunk) -> tuple[str, ...]:
            if page_images is None:
                return ()
            from docextract.parsing.ocr import page_image_path

            paths = (
                page_image_path(page_images, snapshot.snapshot_id, index)
                for index in sorted({unit.page_index for unit in chunk} & scanned)
            )
            return tuple(str(path) for path in paths if path.exists())

        def hashes(*names):
            return {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in names
            }

        # Generation identity covers only what changes a prompt; stored answers can be re-assembled
        # under different deterministic options without generating again.
        generation_id = digest(
            {
                "snapshot": snapshot.snapshot_id,
                "schema": schema,
                "backend": backend.identity,
                "protocol": "cited-values/3",
                "max_chars": max_window_chars,
                "overlap_lines": overlap_lines,
                "max_schema_chars": max_schema_chars,
                # A replay of an older run passes that run's prompt-code hashes explicitly.
                "prompt_code": prompt_code or hashes("interpretation.py", "partition.py"),
                # Present only when set, so earlier stored answers keep their identity.
                **({"max_units": max_window_units} if max_window_units else {}),
                **({"page_images": {"max_pages": max_window_pages, "format": "jpeg"}} if page_images else {}),
                **({"protocol": protocol} if protocol != "cited" else {}),
                **({"prompt_budget_chars": prompt_budget_chars} if prompt_budget_chars else {}),
                **({"factor_notes": True} if factor_notes else {}),
            }
        )
        run_id = digest(
            {
                "generation": generation_id,
                "complete_tables": complete_tables,
                "apply_defaults": apply_defaults,
                "absent_nullable": absent_nullable,
                **({"document_strings": True} if document_strings else {}),
                **({"nulls_if_partial": True} if nulls_if_partial else {}),
                **({"route_groups": True} if route_groups else {}),
                **({"merge_entities": True} if merge_entities else {}),
                "implementation": hashes(
                    "cited.py", "tables.py", "models.py", "assembly.py", "schema.py", "jsonutil.py"
                ),
            }
        )
        run_dir = workspace / run_id
        if not (run_dir / "snapshot.json").exists():
            save_json(run_dir / "snapshot.json", snapshot)
        state = CitedState(compiled)
        with closing(Journal(workspace / "cited.sqlite")) as journal:
            pending: list[str] = []

            def interpret(chunk, part=None):
                nonlocal state
                identity = {"generation": generation_id, "sources": [unit.alias for unit in chunk]}
                if part is not None:
                    identity["schema_part"] = digest(part)
                shown = images(chunk)
                if shown:
                    # The answer depends on the exact image bytes the model saw, and their size.
                    identity["images"] = [
                        hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in shown
                    ]
                    if page_image_side:
                        identity["image_side"] = page_image_side
                task = CitedTask(
                    task_id=digest(identity),
                    original_schema=schema,
                    units=chunk,
                    schema_part=part,
                    page_images=shown,
                    page_image_side=page_image_side if shown else None,
                    protocol=protocol,
                )
                try:
                    state = _process(journal, run_id, task, backend, state, fault, compiled, complete_tables)
                except Pending:
                    # Windows are independent, so later windows can still be queued in this pass.
                    pending.append(task.task_id)
                except (IncompleteGeneration, InputTooLong, UnparseableOutput, Unanswered) as error:
                    lines = tuple(dict.fromkeys(unit.line_id for unit in chunk))
                    # A missing answer is not a size problem, so the window is not subdivided.
                    if len(lines) > 1 and not isinstance(error, Unanswered):
                        cut = lines[len(lines) // 2]
                        middle = next(i for i, unit in enumerate(chunk) if unit.line_id == cut)
                        interpret(chunk[:middle], part)
                        interpret(chunk[middle:], part)
                        return
                    state.failed_windows += 1
                    state.unresolved.append(
                        UnresolvedItem(
                            pointer="",
                            reason=f"source line not interpreted: {type(error).__name__}",
                            source_atom_ids=lines,
                        )
                    )

            if route_groups and len(groups) > 1:
                plan = route([render_evidence(chunk, refs=False) for chunk in chunks], groups)
                pairs = [
                    (chunk, groups[index]) for chunk, keep in zip(chunks, plan, strict=True) for index in keep
                ]
            else:
                pairs = [(chunk, part) for chunk in chunks for part in groups]
            for chunk, part in pairs:
                interpret(chunk, part)
            if pending:
                raise Pending(pending)
            usage = Usage(backend=backend.identity["backend"], **journal.usage(run_id), cost_usd=None)
        if merge_entities:
            state.merge_complementary()
        outputs = state.outputs(apply_defaults, absent_nullable, document_strings, nulls_if_partial)
        data, fields, records = assemble(outputs, _array_scopes(compiled, state))
        unresolved = list(state.unresolved)
        unresolved.extend(
            UnresolvedItem(
                pointer=field.pointer,
                reason="value not found in the cited or nearby source",
                source_atom_ids=field.source_atom_ids,
            )
            for field in fields
            if field.status == "observed" and field.transform in ("model_value_unsupported", "uncited")
        )
        unresolved.extend(
            UnresolvedItem(pointer="", reason=f"page {page.page_index + 1}: {flag}")
            for page in snapshot.pages
            for flag in page.quality_flags
        )
        validation = compiled.validate(data)
        unresolved.extend(
            UnresolvedItem(pointer="", reason="schema validation: " + error) for error in validation.errors
        )
        all_lines = tuple(
            atom.atom_id for page in snapshot.pages for atom in page.atoms if atom.kind == "line"
        )
        result = ExtractionResult(
            snapshot_id=snapshot.snapshot_id,
            schema_sha256=digest(schema),
            run_id=run_id,
            data=data,
            field_evidence=fields,
            source_records=records,
            unresolved=tuple(unresolved),
            validation=validation,
            usage=usage,
            snapshot=snapshot,
            coverage=Coverage(
                total_lines=len(all_lines),
                examined_lines=len(state.examined),
                unexamined_region_ids=tuple(x for x in all_lines if x not in state.examined),
                unclaimed_region_ids=tuple(x for x in all_lines if x not in state.claimed),
                rejected_proposals=state.rejected,
                materialized_table_rows=state.copied,
            ),
            completion=Completion(
                execution_complete=True,
                status="complete" if validation.valid and not unresolved else "partial",
            ),
        )
        save_json(run_dir / "result.json", result)
        temporary = workspace / "result.json.tmp"
        shutil.copyfile(run_dir / "result.json", temporary)
        temporary.replace(workspace / "result.json")
        return result
