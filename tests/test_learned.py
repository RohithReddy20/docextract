"""Scripted proposals exercise contracts/recovery, never learned extraction quality."""

import json
import sqlite3

import pytest
from reportlab.pdfgen.canvas import Canvas

from docextract.interpretation import evidence_units, strict_json, windows
from docextract.learned import extract_learned
from docextract.models import GenerationResult, InputTooLong
from docextract.parsing.pdfium import PdfiumParser


def make_pdf(tmp_path, pages):
    path = tmp_path / "source.pdf"
    canvas = Canvas(str(path), invariant=1)
    for lines in pages:
        for i, text in enumerate(lines):
            canvas.drawString(48, 740 - i * 24, text)
        canvas.showPage()
    canvas.save()
    return path


def span(unit, quote=None):
    quote = quote if quote is not None else unit.text
    start = unit.text.index(quote)
    return {"alias": unit.alias, "start": start, "end": start + len(quote), "quote": quote}


class ScriptedProposals:
    identity = {"backend": "scripted-test-proposals", "version": "1"}

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def generate(self, task):
        self.calls += 1
        return GenerationResult(raw_text=json.dumps(self.script(task)), input_tokens=10, output_tokens=20)


ROWS = {
    "type": "object",
    "required": ["rows"],
    "properties": {
        "rows": {
            "type": "array",
            "items": {"type": "object", "required": ["value"], "properties": {"value": {"type": "integer"}}},
        }
    },
}


def row_script(task):
    records, fields = [], []
    for index, unit in enumerate(task.units):
        if unit.text.startswith("Row: "):
            key = f"local-{index}"
            records.append({"key": key, "array_pointer": "/rows", "anchor": span(unit)})
            fields.append({"record": key, "pointer": "/value", "spans": [span(unit, unit.text[5:])]})
    return {"records": records, "fields": fields}


def test_duplicate_rows_overlap_and_retry_use_source_occurrences(tmp_path):
    pdf = make_pdf(tmp_path, [["Row: 2", "Row: 2", "Row: 3", "Row: 4"]])
    backend = ScriptedProposals(row_script)
    first = extract_learned(
        pdf, ROWS, workspace=tmp_path / "run", backend=backend, max_window_chars=180, overlap_lines=1
    )
    calls = backend.calls
    second = extract_learned(
        pdf, ROWS, workspace=tmp_path / "run", backend=backend, max_window_chars=180, overlap_lines=1
    )
    assert calls > 1 and calls == backend.calls
    assert first == second
    assert first.data == {"rows": [{"value": 2}, {"value": 2}, {"value": 3}, {"value": 4}]}
    assert len({record.record_id for record in first.source_records}) == 4
    assert first.validation.valid and first.completion.execution_complete
    assert first.coverage.examined_lines == 4
    assert first.completion.record_completeness == "not_independently_verified"


def test_arbitrary_nested_records_are_assembled_under_their_parents(tmp_path):
    pdf = make_pdf(tmp_path, [["Team Ops", "Item A", "Alias X", "Item A", "Alias X"]])
    schema = {
        "type": "object",
        "properties": {
            "teams": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": {"type": "string"},
                                    "aliases": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {"name": {"type": "string"}},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            }
        },
    }

    def script(task):
        records, fields = [], []
        parent = None
        for index, unit in enumerate(task.units):
            key = f"r{index}"
            if unit.text.startswith("Team"):
                records.append({"key": key, "array_pointer": "/teams", "anchor": span(unit)})
                fields.append({"record": key, "pointer": "/name", "spans": [span(unit, "Ops")]})
                team = key
            elif unit.text.startswith("Item"):
                records.append(
                    {"key": key, "array_pointer": "/teams/*/items", "anchor": span(unit), "parent": team}
                )
                fields.append({"record": key, "pointer": "/sku", "spans": [span(unit, "A")]})
                parent = key
            else:
                records.append(
                    {
                        "key": key,
                        "array_pointer": "/teams/*/items/*/aliases",
                        "anchor": span(unit),
                        "parent": parent,
                    }
                )
                fields.append({"record": key, "pointer": "/name", "spans": [span(unit, "X")]})
        return {"records": records, "fields": fields}

    result = extract_learned(pdf, schema, workspace=tmp_path / "run", backend=ScriptedProposals(script))
    item = {"sku": "A", "aliases": [{"name": "X"}]}
    assert result.data == {"teams": [{"name": "Ops", "items": [item, item]}]}
    assert len(result.source_records) == 5
    assert "/teams/0/items/1/aliases/0/name" in {field.pointer for field in result.field_evidence}


@pytest.mark.parametrize("stage", ["generated", "raw_saved", "validated", "committed"])
def test_interpretation_recovers_at_every_durable_boundary(tmp_path, stage):
    pdf = make_pdf(tmp_path, [["Row: 2", "Row: 2"]])
    backend = ScriptedProposals(row_script)

    def interrupt(actual, task):
        if actual == stage:
            raise InterruptedError(stage)

    with pytest.raises(InterruptedError):
        extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=backend, fault=interrupt)
    result = extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=backend)
    assert result.data == {"rows": [{"value": 2}, {"value": 2}]}
    assert backend.calls == (2 if stage == "generated" else 1)


def test_explicit_cross_page_append_keeps_both_pages(tmp_path):
    pdf = make_pdf(tmp_path, [["Record Extended"], ["support"]])
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"description": {"type": "string"}}},
            }
        },
    }

    def script(task):
        unit = task.units[0]
        if unit.page_index == 0:
            return {
                "records": [{"key": "r", "array_pointer": "/rows", "anchor": span(unit)}],
                "fields": [{"pointer": "/description", "record": "r", "spans": [span(unit, "Extended")]}],
            }
        record = task.known_records[0].record_id
        return {
            "records": [
                {
                    "key": "continued",
                    "array_pointer": "/rows",
                    "anchor": span(unit),
                    "continuation_of": record,
                }
            ],
            "fields": [
                {
                    "pointer": "/description",
                    "record": "continued",
                    "operation": "append",
                    "spans": [span(unit)],
                }
            ],
        }

    result = extract_learned(
        pdf,
        schema,
        workspace=tmp_path / "run",
        backend=ScriptedProposals(script),
        max_window_chars=100,
        overlap_lines=0,
    )
    assert result.data == {"rows": [{"description": "Extended support"}]}
    assert len(result.source_records) == 1
    pages = {
        result.snapshot.atom_index()[source].page_index for source in result.field_evidence[0].source_atom_ids
    }
    assert pages == {0, 1}
    assert result.source_records[0].fragment_atom_ids


def test_missing_null_zero_default_and_omission_remain_distinct(tmp_path):
    pdf = make_pdf(tmp_path, [["NULL", "0"]])
    schema = {
        "type": "object",
        "properties": {
            "null": {"type": ["string", "null"]},
            "zero": {"type": "number"},
            "default": {"type": "string", "default": "USD"},
            "missing": {"type": "string"},
        },
    }

    def script(task):
        return {
            "fields": [
                {"pointer": "/null", "spans": [span(task.units[0])]},
                {"pointer": "/zero", "spans": [span(task.units[1])]},
                {"pointer": "/default", "status": "absent_after_search"},
            ]
        }

    result = extract_learned(
        pdf, schema, workspace=tmp_path / "run", backend=ScriptedProposals(script), apply_defaults=True
    )
    assert result.data == {"null": None, "zero": 0, "default": "USD"}
    fields = {field.pointer: field for field in result.field_evidence}
    assert fields["/null"].status == "observed"
    assert fields["/default"].status == "schema_default" and not fields["/default"].source_atom_ids
    assert fields["/missing"].status == "ambiguous"
    assert result.completion.status == "partial"


@pytest.mark.parametrize("failure", ["unknown_alias", "bad_quote", "unknown_field"])
def test_invalid_proposals_are_rejected_individually_and_reported(tmp_path, failure):
    pdf = make_pdf(tmp_path, [["Row: 2", "Row: 3"]])

    def script(task):
        data = row_script(task)
        if failure == "unknown_alias":
            data["records"][0]["anchor"]["alias"] = "invented"
        if failure == "bad_quote":
            data["fields"][0]["spans"][0]["quote"] = "9"
        if failure == "unknown_field":
            data["fields"][0]["pointer"] = "/made_up"
        return data

    result = extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=ScriptedProposals(script))
    rows = result.data["rows"]
    assert rows[-1] == {"value": 3} and {"value": 2} not in rows
    assert result.coverage.rejected_proposals >= 1
    assert any(item.reason.startswith("rejected proposal:") for item in result.unresolved)
    assert result.completion.status == "partial"
    with sqlite3.connect(tmp_path / "run/interpretation.sqlite") as db:
        state, raw, output = db.execute("SELECT state, raw, output FROM tasks").fetchone()
    assert state == "committed" and raw and json.loads(output)["rejected"] == []


def test_unfinished_single_line_window_is_reported_without_salvage(tmp_path):
    pdf = make_pdf(tmp_path, [["Row: 2"]])

    class Truncated(ScriptedProposals):
        def generate(self, task):
            return GenerationResult(raw_text=json.dumps(row_script(task)), finish_reason="length")

    result = extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=Truncated(row_script))
    assert result.data == {} and result.coverage.examined_lines == 0
    assert any("not interpreted: IncompleteGeneration" in item.reason for item in result.unresolved)
    with sqlite3.connect(tmp_path / "run/interpretation.sqlite") as db:
        state, raw, output = db.execute("SELECT state, raw, output FROM tasks").fetchone()
    assert state == "raw_saved" and raw and output is None


@pytest.mark.parametrize("limit", ["length", "input", "unparseable"])
def test_oversized_windows_are_subdivided_until_they_finish(tmp_path, limit):
    pdf = make_pdf(tmp_path, [["Row: 1", "Row: 2", "Row: 3", "Row: 4", "Row: 5"]])

    class Limited(ScriptedProposals):
        def generate(self, task):
            self.calls += 1
            if len(task.units) > 2:
                if limit == "input":
                    raise InputTooLong("too long")
                if limit == "unparseable":
                    return GenerationResult(raw_text='{"records": [')
                return GenerationResult(raw_text="{}", finish_reason="length")
            return GenerationResult(raw_text=json.dumps(row_script(task)))

    backend = Limited(row_script)
    result = extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=backend, overlap_lines=0)
    assert result.data == {"rows": [{"value": value} for value in range(1, 6)]}
    assert result.coverage.examined_lines == 5 and backend.calls > 3
    again = extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=backend, overlap_lines=0)
    assert again == result


def test_observed_model_output_quirks_keep_valid_table_and_field_proposals(tmp_path):
    """Replays the formatting/duplication failure modes seen in the first Kaggle retest."""
    pdf = tmp_path / "table.pdf"
    canvas = Canvas(str(pdf), invariant=1)
    canvas.drawString(48, 760, "Invoice ID: 0017")
    rows = [("Item", "Quantity", "Amount USD"), ("A0017", "2", "12.50"), ("A0017", "2", "12.50")]
    rows.append(("B0017", "1", "5.00"))
    for index, row in enumerate(rows):
        for x, text in zip((48, 226, 448), row, strict=True):
            canvas.drawString(x, 720 - index * 28, text)
    canvas.save()
    schema = {
        "type": "object",
        "properties": {
            "invoice_id": {"type": "string"},
            "transactions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                },
            },
        },
    }

    def script(task):
        table = task.tables[0]
        first, second, third = (row[0] for row in table.rows)
        invoice = next(unit for unit in task.units if unit.text.startswith("Invoice"))
        nulls = {"record": None, "separator": None, "status": None, "operation": None}
        invoice_span = {"alias": invoice.alias, "quote": "0017"}
        return {
            "fields": [{"pointer": "/invoice_id", "spans": [invoice_span], **nulls}]
            + [
                {"pointer": "/code", "record": "A0017", "spans": [{"alias": alias, "quote": "A0017"}]}
                for alias in (first, second)
            ]
            + [{"pointer": "/code", "record": "B0017", "spans": [{"alias": third, "quote": "B0017"}]}],
            "records": [
                {"key": key, "array_pointer": "/transactions", "anchor": {"alias": alias, "quote": key}}
                for key, alias in (("A0017", first), ("A0017", second), ("B0017", third))
            ],
            "tables": [
                {
                    "table_id": table.table_id,
                    "array_pointer": "/transactions",
                    "parent": None,
                    "columns": [
                        {"pointer": pointer, "header_alias": alias}
                        for pointer, alias in zip(
                            ("/code", "/quantity", "/amount"), table.header_aliases, strict=True
                        )
                    ],
                }
            ],
            "notes": "ignored",
        }

    class Fenced(ScriptedProposals):
        def generate(self, task):
            return GenerationResult(raw_text="```json\n" + json.dumps(self.script(task)) + "\n```")

    result = extract_learned(pdf, schema, workspace=tmp_path / "run", backend=Fenced(script))
    repeated = {"code": "A0017", "quantity": 2, "amount": 12.5}
    assert result.data == {
        "invoice_id": "0017",
        "transactions": [repeated, repeated, {"code": "B0017", "quantity": 1, "amount": 5}],
    }
    assert len(result.source_records) == 3 and result.validation.valid
    # Two records shared one local key; they and their fields are rejected, then the
    # table mapping supplies both physical rows by occurrence identity.
    assert result.coverage.rejected_proposals == 5


def test_two_records_in_one_source_line_have_distinct_offsets(tmp_path):
    pdf = make_pdf(tmp_path, [["2; 2"]])

    def script(task):
        unit = task.units[0]
        spans = [{"alias": unit.alias, "start": start, "end": start + 1, "quote": "2"} for start in (0, 3)]
        return {
            "records": [
                {"key": str(i), "array_pointer": "/rows", "anchor": source} for i, source in enumerate(spans)
            ],
            "fields": [
                {"record": str(i), "pointer": "/value", "spans": [source]} for i, source in enumerate(spans)
            ],
        }

    result = extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=ScriptedProposals(script))
    assert result.data == {"rows": [{"value": 2}, {"value": 2}]}
    assert len(set(record.record_id for record in result.source_records)) == 2


def test_conflicting_mentions_remain_unresolved(tmp_path):
    pdf = make_pdf(tmp_path, [["10", "20"]])
    schema = {"type": "object", "properties": {"amount": {"type": "number"}}}
    backend = ScriptedProposals(
        lambda task: {"fields": [{"pointer": "/amount", "spans": [span(unit)]} for unit in task.units]}
    )
    result = extract_learned(pdf, schema, workspace=tmp_path / "run", backend=backend)
    assert result.data == {} and result.field_evidence[0].status == "ambiguous"


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_corruption_is_not_silently_normalized(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_evidence_projection_retains_original_character_bindings(tmp_path):
    pdf = make_pdf(tmp_path, [["Code: 0007", "Another line"]])
    snapshot = PdfiumParser().parse(pdf)
    units = evidence_units(snapshot)
    assert all(len(unit.text) == len(unit.offset_atom_ids) for unit in units)
    assert all(
        snapshot.atom_index()[source].kind == "character"
        for unit in units
        for source in unit.offset_atom_ids
        if source
    )
    covered = {unit.alias for chunk in windows(units, max_chars=1) for unit in chunk}
    assert covered == {unit.alias for unit in units}


def test_empty_proposals_do_not_assert_empty_array_inventory(tmp_path):
    pdf = make_pdf(tmp_path, [["Nothing recognized"]])
    result = extract_learned(
        pdf, ROWS, workspace=tmp_path / "run", backend=ScriptedProposals(lambda task: {})
    )
    assert result.data == {} and not result.validation.valid
    assert result.completion.status == "partial" and result.completion.execution_complete
    assert result.unresolved and result.coverage.unclaimed_region_ids


def test_unique_local_quotes_resolve_to_same_occurrences_as_explicit_offsets(tmp_path):
    pdf = make_pdf(tmp_path, [["Row: 2", "Row: 2"]])

    def quotes(task):
        output = row_script(task)
        for record in output["records"]:
            record["anchor"].pop("start")
            record["anchor"].pop("end")
        for field in output["fields"]:
            field["spans"][0].pop("start")
            field["spans"][0].pop("end")
        return output

    explicit = extract_learned(
        pdf, ROWS, workspace=tmp_path / "explicit", backend=ScriptedProposals(row_script)
    )
    quoted = extract_learned(pdf, ROWS, workspace=tmp_path / "quoted", backend=ScriptedProposals(quotes))
    assert quoted.data == explicit.data
    assert quoted.source_records == explicit.source_records
    assert quoted.field_evidence == explicit.field_evidence


def test_ambiguous_quote_inside_one_cell_requires_offsets(tmp_path):
    pdf = make_pdf(tmp_path, [["2; 2"]])
    backend = ScriptedProposals(
        lambda task: {
            "records": [
                {"key": "r", "array_pointer": "/rows", "anchor": {"alias": task.units[0].alias, "quote": "2"}}
            ]
        }
    )
    result = extract_learned(pdf, ROWS, workspace=tmp_path / "run", backend=backend)
    assert result.data == {} and result.coverage.rejected_proposals == 1
    assert any("ambiguous" in item.reason for item in result.unresolved)


@pytest.mark.parametrize(
    "text", ['{"x":1e999}', '{"x":0.1234567890123456789012345}', '{"x":9007199254740993.00}']
)
def test_numeric_overflow_and_precision_loss_are_rejected(text):
    with pytest.raises(ValueError):
        strict_json(text)


def test_typed_copy_does_not_round_large_printed_decimal():
    from docextract.models import DeterministicBackend

    with pytest.raises(ValueError, match="faithfully"):
        DeterministicBackend._convert("9007199254740993.00", {"type": "number"})
    assert DeterministicBackend._convert("12.50", {"type": "number"})[0] == 12.5
    with pytest.raises(ValueError):
        DeterministicBackend._convert("[NaN]", {"type": "array", "items": {"type": "number"}})


def test_learned_header_mapping_materializes_every_duplicate_source_row(tmp_path):
    pdf = tmp_path / "table.pdf"
    canvas = Canvas(str(pdf), invariant=1)
    rows = [
        ("Code", "Quantity", "Amount"),
        ("A001", "2", "12.50"),
        ("A001", "2", "12.50"),
        ("B001", "1", "5.00"),
    ]
    for index, row in enumerate(rows):
        for x, text in zip((48, 226, 448), row, strict=True):
            canvas.drawString(x, 740 - index * 28, text)
    canvas.save()
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["code", "quantity", "amount"],
                    "properties": {
                        "code": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                },
            }
        },
    }

    def script(task):
        assert len(task.tables) == 1
        table = task.tables[0]
        return {
            "tables": [
                {
                    "table_id": table.table_id,
                    "array_pointer": "/rows",
                    "columns": [
                        {"pointer": pointer, "header_alias": alias}
                        for pointer, alias in zip(
                            ("/code", "/quantity", "/amount"), table.header_aliases, strict=True
                        )
                    ],
                }
            ]
        }

    backend = ScriptedProposals(script)
    first = extract_learned(pdf, schema, workspace=tmp_path / "run", backend=backend)
    second = extract_learned(pdf, schema, workspace=tmp_path / "run", backend=backend)
    repeated = {"code": "A001", "quantity": 2, "amount": 12.5}
    assert first.data == {"rows": [repeated, repeated, {"code": "B001", "quantity": 1, "amount": 5}]}
    assert first == second and backend.calls == 1
    assert first.coverage.inventoried_table_rows == first.coverage.materialized_table_rows == 3
    assert {record.discovery_method for record in first.source_records} == {"learned_table_mapping"}
    assert len(first.field_evidence) == 9 and len(first.source_records) == 3
    assert first.validation.valid and first.completion.record_completeness == "not_independently_verified"
