"""Learned discovery contracts; this module never imports or loads a model."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from docextract.contracts import Contract, ParseSnapshot, SourceRecord, digest
from docextract.geometry import bounds
from docextract.jsonutil import strict_json as strict_json
from docextract.models import GenerationResult


class ModelOutput(Contract):
    """Model output may spell an omitted optional value as null; required values stay required."""

    @model_validator(mode="before")
    @classmethod
    def _omit_optional_nulls(cls, data):
        if isinstance(data, dict):
            fields = cls.model_fields
            data = {
                key: value
                for key, value in data.items()
                if value is not None or key not in fields or fields[key].is_required()
            }
        return data


class UnparseableOutput(ValueError):
    """Complete model output that is not one JSON object."""


class EvidenceUnit(Contract):
    alias: str
    atom_id: str
    line_id: str
    page_index: int
    text: str
    bbox: tuple[float, float, float, float]
    offset_atom_ids: tuple[str | None, ...]

    def prompt_data(self) -> dict:
        return {"id": self.alias, "page": self.page_index + 1, "text": self.text, "xyxy": self.bbox}


class Span(ModelOutput):
    alias: str
    start: int | None = Field(default=None, ge=0, strict=True)
    end: int | None = Field(default=None, gt=0, strict=True)
    quote: str


class RecordProposal(ModelOutput):
    key: str
    array_pointer: str
    anchor: Span
    parent: str | None = None
    continuation_of: str | None = None


class FieldProposal(ModelOutput):
    pointer: str
    record: str | None = None
    spans: tuple[Span, ...] = ()
    status: Literal["observed", "absent_after_search", "unreadable", "ambiguous"] = "observed"
    operation: Literal["set", "append"] = "set"
    separator: Literal["", " ", "\n"] = " "


class TableCandidate(Contract):
    table_id: str
    header_aliases: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    ambiguous_header_rows: tuple[str, ...] = ()


class ColumnBinding(ModelOutput):
    pointer: str
    header_alias: str


class TableProposal(ModelOutput):
    table_id: str
    array_pointer: str
    parent: str | None = None
    columns: tuple[ColumnBinding, ...]


class ProposalBatch(ModelOutput):
    records: tuple[RecordProposal, ...] = ()
    fields: tuple[FieldProposal, ...] = ()
    unresolved: tuple[str, ...] = ()
    unresolved_boundaries: tuple[Span, ...] = ()
    tables: tuple[TableProposal, ...] = ()


class CommittedWindow(Contract):
    """Journal disposition: the checked proposal items plus every item rejected while parsing."""

    batch: ProposalBatch
    rejected: tuple[str, ...] = ()


def parse_batch(data) -> CommittedWindow:
    """Validate each proposal item separately so one malformed item cannot discard the others."""
    if not isinstance(data, dict):
        raise ValueError("Expected one JSON object")
    items, rejected = {}, []
    for key, value in data.items():
        if key not in ProposalBatch.model_fields:
            rejected.append(f"unknown top-level key {key!r} ignored")
            continue
        if value is None:
            continue
        if not isinstance(value, list):
            rejected.append(f"{key}: expected a list")
            continue
        adapter = TypeAdapter(ProposalBatch.model_fields[key].annotation.__args__[0])
        accepted = []
        for index, item in enumerate(value):
            try:
                accepted.append(adapter.validate_python(item))
            except ValidationError as error:
                rejected.append(f"{key}[{index}] malformed: {error.errors()[0]['msg']}")
        items[key] = tuple(accepted)
    return CommittedWindow(batch=ProposalBatch(**items), rejected=tuple(rejected))


class InterpretationTask(Contract):
    task_id: str
    original_schema: dict[str, Any]
    units: tuple[EvidenceUnit, ...]
    known_records: tuple[SourceRecord, ...] = ()
    record_context: tuple[dict[str, Any], ...] = ()
    tables: tuple[TableCandidate, ...] = ()


class InterpretationBackend(Protocol):
    identity: dict[str, Any]

    def generate(self, task: InterpretationTask) -> GenerationResult: ...


def evidence_units(snapshot: ParseSnapshot) -> tuple[EvidenceUnit, ...]:
    atoms = snapshot.atom_index()

    def characters(atom_id):
        atom = atoms[atom_id]
        if atom.kind == "character":
            return atom.text, tuple(atom.atom_id for _ in atom.text)
        separator = "" if atom.kind == "token" else " "
        text, positions = "", []
        for child in atom.child_ids:
            child_text, child_positions = characters(child)
            if text:
                text += separator
                positions.extend(None for _ in separator)
            text += child_text
            positions.extend(child_positions)
        if not atom.child_ids:
            return atom.text, tuple(atom.atom_id for _ in atom.text)
        return text, tuple(positions)

    units = []
    for page in snapshot.pages:
        lines = [atom for atom in page.atoms if atom.kind == "line"]
        cells = [atom for atom in page.atoms if atom.kind == "cell"]
        for line in lines:
            line_tokens = set(line.child_ids)
            selected = [cell for cell in cells if set(cell.child_ids) & line_tokens] or [line]
            for atom in selected:
                text, offsets = characters(atom.atom_id)
                units.append(
                    EvidenceUnit(
                        alias=f"s{len(units)}",
                        atom_id=atom.atom_id,
                        line_id=line.atom_id,
                        page_index=page.page_index,
                        text=text,
                        bbox=bounds(atom.polygon),
                        offset_atom_ids=offsets,
                    )
                )
    return tuple(units)


def render_evidence(units: tuple[EvidenceUnit, ...], refs=True) -> str:
    """Compact line-oriented evidence: one row per native line, cells separated by tabs, each cell
    prefixed by its reference unless `refs` is false."""
    rows, page, line = [], None, None
    for unit in units:
        if unit.page_index != page:
            page, line = unit.page_index, None
            rows.append(f"[page {page + 1}]")
        cell = f"{unit.alias} {unit.text}" if refs else unit.text
        if unit.line_id == line:
            rows[-1] += f"\t{cell}"
        else:
            line = unit.line_id
            rows.append(cell)
    return "\n".join(rows)


def windows(
    units: tuple[EvidenceUnit, ...],
    *,
    max_chars=6000,
    overlap_lines=2,
    unit_overhead=80,
    max_units=None,
    max_pages=None,
    align_pages=False,
):
    """Line-aligned windows; unit_overhead approximates each unit's prompt formatting cost.

    `max_units` also caps the cells per window: answer length grows with the cells a window holds,
    so dense tables get smaller windows than prose. `max_pages` caps the pages a window spans (each
    page may bring its image). With `align_pages`, windows hold whole pages and do not overlap, so a
    page shown as an image appears in exactly one window; only a page too large for one window is
    split, and then by lines.
    """
    if max_chars < 1 or overlap_lines < 0 or (max_units is not None and max_units < 1):
        raise ValueError("Require positive max_chars and max_units and nonnegative overlap_lines")
    if max_pages is not None and not align_pages:
        raise ValueError("A page limit needs page-aligned windows (overlap would cross pages)")
    lines: list[list[EvidenceUnit]] = []
    for unit in units:
        if not lines or lines[-1][0].line_id != unit.line_id:
            lines.append([])
        lines[-1].append(unit)
    start = 0
    while start < len(lines):
        stop, size, count, pages = start, 0, 0, set()
        while stop < len(lines):
            line_size = sum(len(unit.text) + unit_overhead for unit in lines[stop])
            over_units = max_units is not None and count + len(lines[stop]) > max_units
            page = lines[stop][0].page_index
            over_pages = max_pages is not None and page not in pages and len(pages) >= max_pages
            if stop > start and (size + line_size > max_chars or over_units or over_pages):
                break
            size += line_size
            count += len(lines[stop])
            pages.add(page)
            stop += 1
        if align_pages and stop < len(lines) and lines[stop][0].page_index == lines[stop - 1][0].page_index:
            # The window would cut its last page in two: end before that page instead, unless the
            # window holds nothing else.
            first = stop - 1
            while first > start and lines[first - 1][0].page_index == lines[stop - 1][0].page_index:
                first -= 1
            if first > start:
                stop = first
        yield tuple(unit for line in lines[start:stop] for unit in line)
        if stop == len(lines):
            break
        start = stop if align_pages else max(start + 1, stop - overlap_lines)


def normalize_span(span: Span, units: dict[str, EvidenceUnit]) -> Span:
    if span.alias not in units:
        raise ValueError("Source alias is outside the supplied evidence window")
    unit = units[span.alias]
    if span.start is None and span.end is None:
        first = unit.text.find(span.quote) if span.quote else -1
        if first < 0 or unit.text.find(span.quote, first + 1) >= 0:
            raise ValueError("Quote is absent or ambiguous within its selected source unit; offsets required")
        span = span.model_copy(update={"start": first, "end": first + len(span.quote)})
    if span.start is None or span.end is None:
        raise ValueError("Provide both source offsets or neither")
    if not span.start < span.end <= len(unit.text) or unit.text[span.start : span.end] != span.quote:
        raise ValueError("Source quote or character offsets do not match native evidence")
    return span


def resolve_span(span: Span, units: dict[str, EvidenceUnit]) -> tuple[str, ...]:
    span = normalize_span(span, units)
    unit = units[span.alias]
    sources = tuple(dict.fromkeys(x for x in unit.offset_atom_ids[span.start : span.end] if x))
    if not sources:
        raise ValueError("Source span selects no original characters")
    return sources


def occurrence_id(array_pointer: str, span: Span, unit: EvidenceUnit, parent: str | None) -> str:
    span = normalize_span(span, {unit.alias: unit})
    return digest(
        {"array": array_pointer, "atom": unit.atom_id, "start": span.start, "end": span.end, "parent": parent}
    )
