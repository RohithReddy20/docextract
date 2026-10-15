from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from docextract.assembly import assemble
from docextract.contracts import (
    Completion,
    Coverage,
    ExtractionResult,
    UnresolvedItem,
    Usage,
    digest,
    save_json,
)
from docextract.models import (
    DeterministicBackend,
    GenerationResult,
    ModelBackend,
    TaskOutput,
    WorkUnit,
)
from docextract.parsing.pdfium import PdfiumParser
from docextract.parsing.protocol import ParserAdapter
from docextract.planning import BindingPlan, Planner
from docextract.runtime import Journal, exclusive_workspace
from docextract.schema import Schema


class IncompleteGeneration(ValueError):
    pass


def _validated_output(task: WorkUnit, generation: GenerationResult) -> TaskOutput:
    if generation.finish_reason != "stop":
        raise IncompleteGeneration("Task unfinished; raw output retained, no prefix salvage or commit")
    output = TaskOutput.model_validate_json(generation.raw_text)
    candidates = {field.pointer: field for field in task.fields}
    if (
        len(output.fields) != len(candidates)
        or {field.pointer for field in output.fields} != candidates.keys()
    ):
        raise ValueError("Backend must return each planned field exactly once")
    for field in output.fields:
        candidate = candidates[field.pointer]
        expected_record = task.record.record_id if task.record else None
        if field.record_id != expected_record or field.source_atom_ids != candidate.source_atom_ids:
            raise ValueError("Backend returned an unbound record or source reference")
        if field.source_lexeme != candidate.lexeme:
            raise ValueError("Backend changed the bound source lexeme")
        if field.status == "observed" and not field.source_atom_ids:
            raise ValueError("Observed field has no source evidence")
        if field.status == "schema_default" and (
            not task.apply_defaults
            or "default" not in candidate.schema_node
            or field.value != candidate.schema_node["default"]
            or candidate.lexeme is not None
        ):
            raise ValueError("Backend applied an unauthorized schema default")
    return output


def _process(journal, run_id, task, backend, fault):
    journal.ensure(run_id, task.task_id)
    state, raw, saved = journal.get(run_id, task.task_id)
    if state == "committed":
        return TaskOutput.model_validate_json(saved)
    if raw is None:
        journal.start(run_id, task.task_id)
        generation = backend.generate(task)
        fault("generated", task)
        journal.save_raw(run_id, task.task_id, generation)
        fault("raw_saved", task)
    else:
        generation = GenerationResult.model_validate_json(raw)
    output = _validated_output(task, generation)
    journal.validated(run_id, task.task_id)
    fault("validated", task)
    journal.commit(run_id, task.task_id, output)
    fault("committed", task)
    return output


def extract(
    document: Path,
    schema: dict,
    *,
    workspace: Path,
    plan: BindingPlan,
    parser: ParserAdapter | None = None,
    backend: ModelBackend | None = None,
    apply_defaults: bool = False,
    window_size: int = 8,
    overlap: int = 1,
    fault: Callable[[str, WorkUnit], None] | None = None,
) -> ExtractionResult:
    compiled = Schema(schema)
    compiled.require_supported()
    planner = Planner(compiled, plan)
    if window_size < 1 or not 0 <= overlap < window_size:
        raise ValueError("Require 0 <= overlap < window_size")
    parser = parser or PdfiumParser()
    backend = backend or DeterministicBackend()
    fault = fault or (lambda stage, task: None)
    with exclusive_workspace(workspace):
        snapshot = parser.parse(document)
        work = planner.build(snapshot, apply_defaults=apply_defaults)
        run_id = digest(
            {
                "snapshot": snapshot.snapshot_id,
                "schema": schema,
                "plan": plan.model_dump(),
                "backend": backend.identity,
                "apply_defaults": apply_defaults,
                "engine": "b0/3",
            }
        )
        run_dir = workspace / run_id
        save_json(run_dir / "snapshot.json", snapshot)
        save_json(run_dir / "tasks.json", work)
        outputs = {}
        with closing(Journal(workspace / "journal.sqlite")) as journal:
            for start in range(0, len(work.tasks), window_size - overlap):
                for task in work.tasks[start : start + window_size]:
                    outputs[task.task_id] = (task, _process(journal, run_id, task, backend, fault))
            usage = Usage(
                backend=backend.identity["backend"],
                **journal.usage(run_id),
                cost_usd=0.0 if isinstance(backend, DeterministicBackend) else None,
            )
        data, fields, records = assemble(list(outputs.values()), work.array_scopes)
        unresolved = list(work.unresolved)
        for task, output in outputs.values():
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
        validation = compiled.validate(data)
        for error in validation.errors:
            unresolved.append(UnresolvedItem(pointer="", reason="schema validation: " + error))
        for field in fields:
            if field.status == "absent_after_search":
                unresolved.append(
                    UnresolvedItem(pointer=field.pointer, reason="field absent after label/row search")
                )
            elif field.status in ("unreadable", "ambiguous"):
                unresolved.append(
                    UnresolvedItem(
                        pointer=field.pointer,
                        reason=f"field {field.status}",
                        source_atom_ids=field.source_atom_ids,
                    )
                )
        for page in snapshot.pages:
            if page.quality_flags:
                unresolved.append(
                    UnresolvedItem(
                        pointer="", reason=f"page {page.page_index + 1}: " + ", ".join(page.quality_flags)
                    )
                )
        unclaimed = tuple(
            line_id for line_id in work.examined_line_ids if line_id not in work.claimed_line_ids
        )
        complete = validation.valid and not unresolved
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
            coverage=Coverage(
                total_lines=len(work.examined_line_ids),
                examined_lines=len(work.examined_line_ids),
                unclaimed_region_ids=unclaimed,
                unresolved_boundaries=sum(
                    "continuation" in issue.reason or "keyless" in issue.reason for issue in unresolved
                ),
            ),
            completion=Completion(execution_complete=True, status="complete" if complete else "partial"),
            snapshot=snapshot,
        )
        save_json(run_dir / "result.json", result)
        save_json(workspace / "result.json", result)
        return result
