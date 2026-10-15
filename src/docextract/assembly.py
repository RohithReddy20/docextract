from __future__ import annotations

from collections import defaultdict

from docextract.contracts import FieldEvidence, SourceRecord
from docextract.models import TaskOutput, WorkUnit
from docextract.schema import join, parts


def put(document: dict, pointer: str, value) -> None:
    segments = parts(pointer)
    target = document
    for segment in segments[:-1]:
        target = target.setdefault(segment, {})
    target[segments[-1]] = value


def assemble(outputs: list[tuple[WorkUnit, TaskOutput]], array_scopes=()):
    data: dict = {}
    evidence: list[FieldEvidence] = []
    records: list[SourceRecord] = []
    objects: dict[str | None, dict] = {None: data}
    pointers: dict[str | None, str] = {None: ""}
    source_records = {task.record.record_id: task.record for task, _ in outputs if task.record}

    def relative_path(path, parent):
        segments = parts(path)
        if parent:
            prefix = [*parts(source_records[parent].schema_pointer), "*"]
            if segments[: len(prefix)] != prefix:
                raise ValueError("Child array is outside its parent schema")
            segments = segments[len(prefix) :]
        return segments

    array_counts: dict[tuple[str | None, str], int] = defaultdict(int)
    for task, output in sorted(outputs, key=lambda pair: pair[0].order):
        record = task.record
        if record:
            parent = record.parent_record_id
            relative = relative_path(record.schema_pointer, parent)
            key = (parent, record.schema_pointer)
            index = array_counts[key]
            array_counts[key] += 1
            target = objects[parent]
            for part in relative[:-1]:
                target = target.setdefault(part, {})
            target = target.setdefault(relative[-1], [])
            target.append({})
            objects[record.record_id] = target[-1]
            pointer = pointers[parent]
            for part in relative:
                pointer = join(pointer, part)
            pointers[record.record_id] = join(pointer, index)
            records.append(record.model_copy(update={"pointer": pointers[record.record_id]}))
        record_id = record.record_id if record else None
        for field in output.fields:
            if field.status in ("observed", "schema_default", "computed") or (
                field.status == "absent_after_search" and field.transform == "null_projection"
            ):
                put(objects[record_id], field.pointer, field.value)
            evidence.append(field.model_copy(update={"pointer": pointers[record_id] + field.pointer}))
    for path, parent in array_scopes:
        segments = relative_path(path, parent)
        target = objects[parent]
        for segment in segments[:-1]:
            target = target.setdefault(segment, {})
        target.setdefault(segments[-1], [])
    return data, tuple(evidence), tuple(records)
