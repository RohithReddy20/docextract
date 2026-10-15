from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceAtom(Contract):
    atom_id: str
    page_index: int
    kind: Literal["character", "token", "line", "cell", "checkbox", "region"]
    text: str
    polygon: tuple[tuple[float, float], ...]
    native_polygon: tuple[tuple[float, float], ...]
    coordinate_frame: str = "display_top_left_points"
    source_channel: str = "native_pdf"
    child_ids: tuple[str, ...] = ()
    native_char_index: int | None = None
    quality_flags: tuple[str, ...] = ()
    raw_label: str | None = None
    checkbox_state: Literal["selected", "unselected", "uncertain"] | None = None


class PageEvidence(Contract):
    page_index: int
    width: float
    height: float
    crop_box: tuple[float, float, float, float]
    rotation: int
    native_to_display: tuple[float, float, float, float, float, float]
    native_text: str
    atoms: tuple[SourceAtom, ...]
    quality_flags: tuple[str, ...] = ()


class ParseSnapshot(Contract):
    contract_version: Literal["parse-snapshot/1"] = "parse-snapshot/1"
    snapshot_id: str
    document_sha256: str
    parser_identity: dict[str, Any]
    pages: tuple[PageEvidence, ...]
    events: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_sources(self):
        expected = digest({"document": self.document_sha256, "parser": self.parser_identity})
        if self.snapshot_id != expected:
            raise ValueError("Snapshot identity does not match document and parser configuration")
        atoms = self.atom_index()
        if len(atoms) != sum(len(page.atoms) for page in self.pages):
            raise ValueError("Duplicate source atom identity")
        for index, page in enumerate(self.pages):
            if page.page_index != index or min(page.width, page.height) <= 0:
                raise ValueError("Invalid page sequence or dimensions")
            for atom in page.atoms:
                if atom.page_index != index or not atom.atom_id.startswith(self.snapshot_id + "/"):
                    raise ValueError("Source identity belongs to a different snapshot or page")
                if any(child not in atoms or atoms[child].page_index != index for child in atom.child_ids):
                    raise ValueError("Source atom has an unknown or cross-page child")
                if not all(
                    math.isfinite(value)
                    for polygon in (atom.polygon, atom.native_polygon)
                    for point in polygon
                    for value in point
                ):
                    raise ValueError("Source coordinates must be finite")
        return self

    def atom_index(self) -> dict[str, SourceAtom]:
        return {atom.atom_id: atom for page in self.pages for atom in page.atoms}


class FieldEvidence(Contract):
    pointer: str
    value: Any
    status: Literal[
        "observed", "absent_after_search", "unreadable", "ambiguous", "schema_default", "computed"
    ]
    record_id: str | None = None
    source_atom_ids: tuple[str, ...] = ()
    source_lexeme: str | None = None
    transform: str | None = None
    support: Literal["character", "cell", "line", "region", "page", "none"] = "none"


class SourceRecord(Contract):
    record_id: str
    pointer: str
    schema_pointer: str
    parent_record_id: str | None = None
    anchor_atom_ids: tuple[str, ...]
    fragment_atom_ids: tuple[str, ...] = ()
    discovery_method: str = "declared_table"


class UnresolvedItem(Contract):
    pointer: str
    reason: str
    source_atom_ids: tuple[str, ...] = ()


class Coverage(Contract):
    total_lines: int
    examined_lines: int
    unexamined_region_ids: tuple[str, ...] = ()
    unclaimed_region_ids: tuple[str, ...] = ()
    unresolved_boundaries: int = 0
    inventory_completeness: str = "not_independently_verified"
    inventoried_table_rows: int = 0
    materialized_table_rows: int = 0
    rejected_proposals: int = 0


class ValidationReport(Contract):
    valid: bool
    errors: tuple[str, ...] = ()


class Usage(Contract):
    backend: str
    generated_attempts: int = 0
    committed_tasks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0


class Completion(Contract):
    execution_complete: bool
    status: Literal["complete", "partial", "unsupported"]
    record_completeness: str = "not_independently_verified"


class ExtractionResult(Contract):
    contract_version: Literal["extraction-result/1"] = "extraction-result/1"
    snapshot_id: str
    schema_sha256: str
    run_id: str
    data: dict[str, Any]
    field_evidence: tuple[FieldEvidence, ...]
    source_records: tuple[SourceRecord, ...]
    coverage: Coverage
    unresolved: tuple[UnresolvedItem, ...]
    validation: ValidationReport
    usage: Usage
    completion: Completion
    snapshot: ParseSnapshot


def save_json(path: Path, value: BaseModel | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, BaseModel):
        # Pydantic's native encoder; snapshots carry every character atom and are large.
        temporary.write_bytes(value.model_dump_json(indent=2).encode() + b"\n")
    else:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
