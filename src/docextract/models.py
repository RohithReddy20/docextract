from __future__ import annotations

import re
from typing import Any, Literal, Protocol

from docextract.contracts import Contract, FieldEvidence, SourceRecord, UnresolvedItem
from docextract.jsonutil import strict_json


class InputTooLong(ValueError):
    """The backend refused a task above its measured input limit; source is never truncated."""


class FieldCandidate(Contract):
    pointer: str
    schema_node: dict[str, Any]
    lexeme: str | None = None
    source_atom_ids: tuple[str, ...] = ()
    uncertainty: Literal["ambiguous", "unreadable"] | None = None


class WorkUnit(Contract):
    task_id: str
    record: SourceRecord | None = None
    order: int
    fields: tuple[FieldCandidate, ...]
    apply_defaults: bool = False


class TaskOutput(Contract):
    fields: tuple[FieldEvidence, ...]
    unresolved: tuple[UnresolvedItem, ...] = ()


class GenerationResult(Contract):
    raw_text: str
    finish_reason: str = "stop"
    input_tokens: int = 0
    output_tokens: int = 0


class ModelBackend(Protocol):
    identity: dict[str, Any]

    def generate(self, task: WorkUnit) -> GenerationResult: ...

    def cancel(self, request_id: str) -> bool: ...


class DeterministicBackend:
    """Typed copying of pre-bound native evidence, not semantic model inference."""

    identity = {"backend": "deterministic-bindings", "version": "1"}

    def generate(self, task: WorkUnit) -> GenerationResult:
        fields: list[FieldEvidence] = []
        unresolved: list[UnresolvedItem] = []
        for candidate in task.fields:
            field, issue = self._field(candidate, task)
            fields.append(field)
            if issue:
                unresolved.append(issue)
        output = TaskOutput(fields=tuple(fields), unresolved=tuple(unresolved))
        return GenerationResult(raw_text=output.model_dump_json())

    def cancel(self, request_id: str) -> bool:
        return False

    def _field(self, candidate: FieldCandidate, task: WorkUnit):
        record_id = task.record.record_id if task.record else None
        common = dict(
            pointer=candidate.pointer,
            record_id=record_id,
            source_atom_ids=candidate.source_atom_ids,
            source_lexeme=candidate.lexeme,
        )
        if candidate.uncertainty:
            return FieldEvidence(
                **common,
                value=None,
                status=candidate.uncertainty,
                support="line" if candidate.source_atom_ids else "none",
            ), None
        if candidate.lexeme is None:
            if task.apply_defaults and "default" in candidate.schema_node:
                return FieldEvidence(
                    **common,
                    value=candidate.schema_node["default"],
                    status="schema_default",
                    transform="schema_default",
                ), None
            return FieldEvidence(**common, value=None, status="absent_after_search"), None
        try:
            value, transform = self._convert(candidate.lexeme, candidate.schema_node)
            return FieldEvidence(
                **common, value=value, status="observed", transform=transform, support="character"
            ), None
        except ValueError as error:
            return FieldEvidence(
                **common, value=None, status="ambiguous", support="character"
            ), UnresolvedItem(
                pointer=candidate.pointer, reason=str(error), source_atom_ids=candidate.source_atom_ids
            )

    @staticmethod
    def _convert(text: str, schema: dict) -> tuple[Any, str]:
        types = schema["type"]
        types = [types] if isinstance(types, str) else types
        if text == "NULL" and "null" in types:
            return None, "literal_NULL"
        kind = next((kind for kind in types if kind != "null"), "null")
        if kind == "string":
            return text, "join_source_tokens"
        if kind == "boolean" and text in ("true", "false"):
            return text == "true", "literal_boolean"
        if kind == "integer" and re.fullmatch(r"-?\d+", text):
            return int(text), "parse_integer"
        if kind == "number" and re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            return strict_json(text), "parse_decimal_json"
        if kind == "array":
            value = strict_json(text)
            if isinstance(value, list):
                return value, "parse_json_array"
        raise ValueError(f"Cannot copy {text!r} as {types}; no locale or date guessing")
