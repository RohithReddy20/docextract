"""Schema-only discovery followed by checked source copying and durable assembly."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from docextract.assembly import assemble
from docextract.contracts import (
    Completion,
    Coverage,
    ExtractionResult,
    SourceRecord,
    UnresolvedItem,
    Usage,
    digest,
    save_json,
)
from docextract.engine import IncompleteGeneration
from docextract.interpretation import (
    CommittedWindow,
    FieldProposal,
    InterpretationBackend,
    InterpretationTask,
    UnparseableOutput,
    evidence_units,
    normalize_span,
    occurrence_id,
    parse_batch,
    resolve_span,
    windows,
)
from docextract.jsonutil import model_json
from docextract.models import (
    DeterministicBackend,
    FieldCandidate,
    GenerationResult,
    InputTooLong,
    TaskOutput,
    WorkUnit,
)
from docextract.parsing.pdfium import PdfiumParser
from docextract.planning import object_arrays, scalar_leaves
from docextract.runtime import Journal, exclusive_workspace
from docextract.schema import Schema, join, parts
from docextract.tables import discover_tables, expand_tables


class DiscoveryState:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.arrays = set(object_arrays(schema))
        self.records: dict[str, SourceRecord] = {}
        self.positions: dict[str, tuple[int, int]] = {}
        self.last_seen: dict[str, int] = {}
        self.anchors: dict[str, tuple[str, int]] = {}
        self.fields: dict[tuple[str | None, str], list[tuple[FieldProposal, tuple, str]]] = defaultdict(list)
        self.absence_votes: dict[tuple[str | None, str], set[str]] = defaultdict(set)
        self.examined: set[str] = set()
        self.claimed: set[str] = set()
        self.unresolved: list[UnresolvedItem] = []
        self.boundaries: set[tuple[str, ...]] = set()
        self.table_rows: set[str] = set()
        self.rejected = 0
        self.windows = 0
        self.failed_windows = 0

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.unresolved.append(UnresolvedItem(pointer="", reason="rejected proposal: " + reason))

    def apply(self, task: InterpretationTask, window: CommittedWindow) -> None:
        """Commit every valid proposal; invalid items are rejected individually and reported."""
        self.windows += 1
        for reason in window.rejected:
            self.reject(reason)
        item_fields = {
            array: {pointer[len(array) + 2 :] for pointer, _ in scalar_leaves(self.schema, array + "/*")}
            for array in self.arrays
        }
        batch, table_rows, rejected_tables = expand_tables(task, window.batch, item_fields)
        for reason in rejected_tables:
            self.reject(reason)
        mapped_keys = {key for rows in table_rows.values() for key in rows}
        units = {unit.alias: unit for unit in task.units}
        visible = {record.record_id for record in task.known_records}
        counts = Counter(proposal.key for proposal in batch.records)
        keys: dict[str, str] = {}
        failed: set[str] = set()
        pending = []
        for proposal in batch.records:
            if not proposal.key or counts[proposal.key] > 1 or proposal.key in visible:
                self.reject(f"record {proposal.key!r}: keys must be unique window-local names")
                failed.add(proposal.key)
            else:
                pending.append(proposal)
        local = {proposal.key for proposal in pending}
        while pending:
            progressed = False
            for proposal in pending[:]:
                if proposal.parent in local and proposal.parent not in keys and proposal.parent not in failed:
                    continue
                pending.remove(proposal)
                progressed = True
                if proposal.parent in failed:
                    self.reject(f"record {proposal.key!r}: parent proposal was rejected")
                    failed.add(proposal.key)
                    continue
                try:
                    mapped = proposal.key in mapped_keys
                    keys[proposal.key] = self._record(proposal, keys, visible, units, mapped)
                except ValueError as error:
                    self.reject(f"record {proposal.key!r}: {error}")
                    failed.add(proposal.key)
            if not progressed:
                for proposal in pending:
                    self.reject(f"record {proposal.key!r}: cyclic parent proposals")
                    failed.add(proposal.key)
                break
        self.table_rows.update(keys[key] for key in mapped_keys if key in keys)
        for proposal in batch.fields:
            if proposal.record in failed:
                self.reject(f"field {proposal.pointer}: its record proposal was rejected")
                continue
            try:
                self._field(proposal, keys, visible, units, task.task_id)
            except ValueError as error:
                self.reject(f"field {proposal.pointer}: {error}")
        self.examined.update(unit.line_id for unit in task.units)
        self.unresolved.extend(UnresolvedItem(pointer="", reason=reason) for reason in batch.unresolved)
        for boundary in batch.unresolved_boundaries:
            try:
                sources = resolve_span(boundary, units)
            except ValueError as error:
                self.reject(f"unresolved boundary: {error}")
                continue
            self.boundaries.add(sources)
            self.unresolved.append(
                UnresolvedItem(
                    pointer="", reason="unresolved record boundary or continuation", source_atom_ids=sources
                )
            )

    def _record(self, proposal, keys, visible, units, mapped) -> str:
        """Validate one record proposal completely before changing state."""
        parent = keys.get(proposal.parent, proposal.parent)
        if parent is not None and parent not in visible | set(keys.values()):
            raise ValueError("parent is outside supplied context")
        if proposal.array_pointer not in self.arrays:
            raise ValueError("unknown object array")
        segments = parts(proposal.array_pointer)
        stars = [i for i, part in enumerate(segments) if part == "*"]
        expected_parent = ""
        if stars:
            for segment in segments[: stars[-1]]:
                expected_parent = join(expected_parent, segment)
        actual_parent = self.records[parent].schema_pointer if parent else ""
        if expected_parent != actual_parent:
            raise ValueError("parent does not match the original schema hierarchy")
        anchor = normalize_span(proposal.anchor, units)
        source_ids = resolve_span(anchor, units)
        unit = units[anchor.alias]
        if proposal.continuation_of:
            record_id = proposal.continuation_of
            if record_id not in visible:
                raise ValueError("continuation must name a record supplied in context")
            record = self.records[record_id]
            if record.schema_pointer != proposal.array_pointer or record.parent_record_id != parent:
                raise ValueError("continuation changes record type or parent")
            self.records[record_id] = record.model_copy(
                update={"fragment_atom_ids": tuple(dict.fromkeys((*record.fragment_atom_ids, *source_ids)))}
            )
        else:
            record_id = occurrence_id(proposal.array_pointer, anchor, unit, parent)
            self.records.setdefault(
                record_id,
                SourceRecord(
                    record_id=record_id,
                    pointer="",
                    schema_pointer=proposal.array_pointer,
                    parent_record_id=parent,
                    anchor_atom_ids=source_ids,
                    discovery_method="learned_table_mapping" if mapped else "learned_source_span",
                ),
            )
            self.positions[record_id] = (int(unit.alias[1:]), anchor.start)
        self.last_seen[record_id] = max(self.last_seen.get(record_id, -1), int(unit.alias[1:]))
        self.anchors.setdefault(record_id, (proposal.anchor.quote, unit.page_index))
        self.claimed.add(unit.line_id)
        return record_id

    def _field(self, proposal: FieldProposal, keys, visible, units, task_id) -> None:
        """Validate one field proposal completely before changing state."""
        proposal = proposal.model_copy(
            update={"spans": tuple(normalize_span(span, units) for span in proposal.spans)}
        )
        record_id = keys.get(proposal.record, proposal.record)
        if record_id is not None and record_id not in visible | set(keys.values()):
            raise ValueError("record is outside supplied context")
        base = f"{self.records[record_id].schema_pointer}/*" if record_id else ""
        allowed = {pointer[len(base) :] for pointer, _ in scalar_leaves(self.schema, base)}
        if proposal.pointer not in allowed:
            raise ValueError("outside the original schema scope")
        if proposal.status == "observed" and not proposal.spans:
            raise ValueError("an observed proposal needs native source spans")
        if proposal.status == "absent_after_search" and (proposal.spans or proposal.operation != "set"):
            raise ValueError("an absent field cannot have source spans or append content")
        if proposal.operation == "append" and (not record_id or proposal.status != "observed"):
            raise ValueError("continuation append requires an observed record field")
        source_ids = tuple(
            dict.fromkeys(atom_id for span in proposal.spans for atom_id in resolve_span(span, units))
        )
        key = (record_id, proposal.pointer)
        if proposal.status == "absent_after_search":
            self.absence_votes[key].add(task_id)
        entry = (
            proposal.model_copy(update={"record": record_id}),
            source_ids,
            proposal.separator.join(span.quote for span in proposal.spans),
        )
        if entry not in self.fields[key]:
            self.fields[key].append(entry)
        if record_id and proposal.spans:
            self.last_seen[record_id] = max(
                self.last_seen.get(record_id, -1), *(int(span.alias[1:]) for span in proposal.spans)
            )
            if proposal.operation == "append":
                record = self.records[record_id]
                self.records[record_id] = record.model_copy(
                    update={
                        "fragment_atom_ids": tuple(dict.fromkeys((*record.fragment_atom_ids, *source_ids)))
                    }
                )
        self.claimed.update(units[span.alias].line_id for span in proposal.spans)

    def context(self, limit=8):
        selected = set(sorted(self.records, key=self.last_seen.get)[-limit:])
        for record_id in tuple(selected):
            parent = self.records[record_id].parent_record_id
            while parent:
                selected.add(parent)
                parent = self.records[parent].parent_record_id
        records = tuple(self.records[key] for key in sorted(selected, key=self.positions.get))
        summaries = []
        for record in records:
            quote, page = self.anchors[record.record_id]
            fields, omitted = {}, []
            for (owner, pointer), entries in self.fields.items():
                if owner != record.record_id:
                    continue
                values = [
                    text
                    for proposal, _, text in entries
                    if proposal.status == "observed" and proposal.operation == "set"
                ]
                if len(values) == 1 and len(values[0]) <= 160 and len(fields) < 4:
                    fields[pointer] = values[0]
                else:
                    omitted.append(pointer)
            summaries.append(
                {
                    "id": record.record_id,
                    "array_pointer": record.schema_pointer,
                    "parent": record.parent_record_id,
                    "page": page + 1,
                    "anchor_quote": quote if len(quote) <= 256 else None,
                    "anchor_quote_omitted": len(quote) > 256,
                    "fields": fields,
                    "omitted_field_pointers": omitted,
                }
            )
        return records, tuple(summaries)

    def _candidate(self, record_id, pointer, node, total_tasks):
        entries = self.fields.get((record_id, pointer), [])
        sources = tuple(dict.fromkeys(atom for _, ids, _ in entries for atom in ids))
        observed = [entry for entry in entries if entry[0].status == "observed"]
        sets = [entry for entry in observed if entry[0].operation == "set"]
        appends = [entry for entry in observed if entry[0].operation == "append"]
        uncertainty = None
        if any(entry[0].status == "ambiguous" for entry in entries) or len(sets) > 1:
            uncertainty = "ambiguous"
        if appends and (not sets or any(entry[0].separator != sets[0][0].separator for entry in appends)):
            uncertainty = "ambiguous"
        if sum(len(entry[1]) for entry in observed) != len(
            set(atom for entry in observed for atom in entry[1])
        ):
            uncertainty = "ambiguous"
        if observed and not uncertainty:
            # Source order, never value equality, determines continuation materialization.
            fragments = sorted(
                observed, key=lambda entry: (int(entry[0].spans[0].alias[1:]), entry[0].spans[0].start)
            )
            if fragments[0][0].operation != "set":
                uncertainty = "ambiguous"
            else:
                lexeme = sets[0][0].separator.join(entry[2] for entry in fragments)
                return FieldCandidate(
                    pointer=pointer, schema_node=node, lexeme=lexeme, source_atom_ids=sources
                )
        if not observed and any(entry[0].status == "unreadable" for entry in entries):
            uncertainty = "unreadable"
        absent = self.absence_votes.get((record_id, pointer), set())
        if uncertainty or not absent or (record_id is None and len(absent) != total_tasks):
            return FieldCandidate(
                pointer=pointer,
                schema_node=node,
                source_atom_ids=sources,
                uncertainty=uncertainty or "ambiguous",
            )
        return FieldCandidate(pointer=pointer, schema_node=node)

    def work(self, total_tasks, apply_defaults):
        ordered: list[SourceRecord | None] = [None]

        def children(parent):
            for record in sorted(
                (record for record in self.records.values() if record.parent_record_id == parent),
                key=lambda record: (*self.positions[record.record_id], record.schema_pointer),
            ):
                ordered.append(record)
                children(record.record_id)

        children(None)
        tasks = []
        for order, record in enumerate(ordered):
            base = f"{record.schema_pointer}/*" if record else ""
            fields = tuple(
                self._candidate(record.record_id if record else None, pointer[len(base) :], node, total_tasks)
                for pointer, node in scalar_leaves(self.schema, base)
            )
            tasks.append(
                WorkUnit(
                    task_id=record.record_id if record else "root",
                    record=record,
                    order=order,
                    fields=fields,
                    apply_defaults=apply_defaults,
                )
            )
        return tasks


def _process(journal, run_id, task, backend, state, fault):
    journal.ensure(run_id, task.task_id)
    status, raw, saved = journal.get(run_id, task.task_id)
    if status == "committed":
        window = CommittedWindow.model_validate_json(saved)
    else:
        if raw is None:
            journal.start(run_id, task.task_id)
            try:
                generation = backend.generate(task)
            except InputTooLong:
                # The refusal is durable, so recovery subdivides without asking the backend again.
                generation = GenerationResult(raw_text="", finish_reason="input_limit")
            fault("generated", task)
            journal.save_raw(run_id, task.task_id, generation)
            fault("raw_saved", task)
        else:
            generation = GenerationResult.model_validate_json(raw)
        if generation.finish_reason == "input_limit":
            raise InputTooLong("Task exceeds the backend input limit; no source truncation")
        if generation.finish_reason != "stop":
            raise IncompleteGeneration(
                "Interpretation unfinished; saved raw output is not a complete inventory"
            )
        try:
            window = parse_batch(model_json(generation.raw_text))
        except ValueError as error:
            raise UnparseableOutput(str(error)) from error
    updated = deepcopy(state)
    updated.apply(task, window)
    if status != "committed":
        journal.validated(run_id, task.task_id)
        fault("validated", task)
        journal.commit(run_id, task.task_id, window)
        fault("committed", task)
    return updated


def extract_learned(
    document: Path,
    schema: dict,
    *,
    workspace: Path,
    backend: InterpretationBackend,
    parser=None,
    max_window_chars=6000,
    overlap_lines=2,
    apply_defaults=False,
    fault=None,
) -> ExtractionResult:
    compiled = Schema(schema)
    compiled.require_supported()

    def check_names(node):
        node = compiled.resolve(node)
        for name, child in node.get("properties", {}).items():
            if name == "*":
                raise ValueError("Learned protocol reserves '*' for schema array scopes")
            check_names(child)
        if "items" in node:
            check_names(node["items"])

    check_names(schema)
    fault = fault or (lambda stage, task: None)
    with exclusive_workspace(workspace):
        snapshot = (parser or PdfiumParser()).parse(document)
        units = evidence_units(snapshot)
        chunks = tuple(windows(units, max_chars=max_window_chars, overlap_lines=overlap_lines))
        run_id = digest(
            {
                "snapshot": snapshot.snapshot_id,
                "schema": schema,
                "backend": backend.identity,
                "engine": "learned-source/3",
                "max_chars": max_window_chars,
                "overlap_lines": overlap_lines,
                "apply_defaults": apply_defaults,
                "implementation": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in (
                        "learned.py",
                        "interpretation.py",
                        "models.py",
                        "assembly.py",
                        "schema.py",
                        "jsonutil.py",
                        "tables.py",
                    )
                },
            }
        )
        run_dir = workspace / run_id
        save_json(run_dir / "snapshot.json", snapshot)
        state = DiscoveryState(compiled)
        with closing(Journal(workspace / "interpretation.sqlite")) as journal:

            def interpret(chunk):
                """An unfinished, oversized or unparseable window is split; one failing line is reported."""
                nonlocal state
                context, context_data = state.context()
                tables = discover_tables(chunk)
                task_id = digest(
                    {
                        "run": run_id,
                        "sources": [unit.alias for unit in chunk],
                        "context": [record.model_dump() for record in context],
                        "context_data": context_data,
                        "tables": [table.model_dump() for table in tables],
                    }
                )
                task = InterpretationTask(
                    task_id=task_id,
                    original_schema=schema,
                    units=chunk,
                    known_records=context,
                    record_context=context_data,
                    tables=tables,
                )
                try:
                    state = _process(journal, run_id, task, backend, state, fault)
                except (IncompleteGeneration, InputTooLong, UnparseableOutput) as error:
                    lines = tuple(dict.fromkeys(unit.line_id for unit in chunk))
                    if len(lines) > 1:
                        cut = lines[len(lines) // 2]
                        middle = next(i for i, unit in enumerate(chunk) if unit.line_id == cut)
                        interpret(chunk[:middle])
                        interpret(chunk[middle:])
                        return
                    state = deepcopy(state)
                    state.failed_windows += 1
                    state.unresolved.append(
                        UnresolvedItem(
                            pointer="",
                            reason=f"source line not interpreted: {type(error).__name__}",
                            source_atom_ids=lines,
                        )
                    )

            for chunk in chunks:
                interpret(chunk)
            usage = Usage(backend=backend.identity["backend"], **journal.usage(run_id), cost_usd=None)
        copier = DeterministicBackend()
        copied = [
            (task, TaskOutput.model_validate_json(copier.generate(task).raw_text))
            for task in state.work(state.windows + state.failed_windows, apply_defaults)
        ]
        data, fields, records = assemble(copied)
        atoms = snapshot.atom_index()
        support_order = {"character": 0, "cell": 1, "line": 2, "region": 3}
        levels = {
            "character": "character",
            "cell": "cell",
            "token": "line",
            "line": "line",
            "checkbox": "region",
            "region": "region",
        }
        fields = tuple(
            field.model_copy(
                update={
                    "support": max(
                        (levels[atoms[source].kind] for source in field.source_atom_ids),
                        key=support_order.get,
                    )
                }
            )
            if field.source_atom_ids
            else field
            for field in fields
        )
        unresolved = list(state.unresolved)
        unresolved.extend(
            UnresolvedItem(
                pointer=field.pointer, reason=f"field {field.status}", source_atom_ids=field.source_atom_ids
            )
            for field in fields
            if field.status in ("ambiguous", "unreadable", "absent_after_search")
        )
        for task, output in copied:
            prefix = next(
                (
                    record.pointer
                    for record in records
                    if task.record and record.record_id == task.record.record_id
                ),
                "",
            )
            unresolved.extend(
                issue.model_copy(update={"pointer": prefix + issue.pointer}) for issue in output.unresolved
            )
        # No-record output never proves that a source array is empty.
        final_records = {record.record_id: record for record in records}
        for path in sorted(state.arrays):
            segments = parts(path)
            stars = [index for index, segment in enumerate(segments) if segment == "*"]
            if stars:
                parent_path = ""
                for segment in segments[: stars[-1]]:
                    parent_path = join(parent_path, segment)
                parents = [record.record_id for record in records if record.schema_pointer == parent_path]
            else:
                parents = [None]
            for parent in parents:
                if not any(
                    record.schema_pointer == path and record.parent_record_id == parent for record in records
                ):
                    prefix = final_records[parent].pointer if parent else ""
                    suffix = path[len(final_records[parent].schema_pointer) + 2 :] if parent else path
                    unresolved.append(
                        UnresolvedItem(
                            pointer=prefix + suffix,
                            reason="no anchored record discovered; emptiness unverified",
                        )
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
                unresolved_boundaries=len(state.boundaries),
                inventoried_table_rows=len(state.table_rows),
                materialized_table_rows=len(state.table_rows & state.records.keys()),
                rejected_proposals=state.rejected,
            ),
            completion=Completion(
                execution_complete=True,
                status="complete" if validation.valid and not unresolved else "partial",
            ),
        )
        save_json(run_dir / "result.json", result)
        save_json(workspace / "result.json", result)
        return result
