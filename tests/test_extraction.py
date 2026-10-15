from copy import deepcopy

from reportlab.pdfgen.canvas import Canvas

from docextract.engine import extract
from docextract.example import EXPECTED
from docextract.planning import BindingPlan


def test_real_multipage_pdf_has_expected_nested_values_and_duplicate_occurrences(result):
    assert result.data == EXPECTED
    assert result.validation.valid
    records = [record for record in result.source_records if record.schema_pointer.endswith("/items")]
    assert len(records) == 4
    assert records[0].record_id != records[1].record_id
    assert records[0].anchor_atom_ids != records[1].anchor_atom_ids
    assert records[0].parent_record_id == records[2].parent_record_id
    assert records[3].parent_record_id != records[0].parent_record_id
    assert records[2].pointer == "/departments/0/items/2"
    assert len(records[2].fragment_atom_ids) == 1


def test_continued_value_keeps_all_pages_and_unmodified_numeric_lexeme(result):
    fields = {field.pointer: field for field in result.field_evidence}
    description = fields["/departments/0/items/2/description"]
    atoms = result.snapshot.atom_index()
    assert {atoms[atom_id].page_index for atom_id in description.source_atom_ids} == {0, 1}
    amount = fields["/departments/0/items/0/amount"]
    assert amount.source_lexeme == "12.50"
    assert amount.value == 12.5
    assert amount.transform == "parse_decimal_json"
    assert "".join(atoms[atom_id].text for atom_id in amount.source_atom_ids) == "12.50"


def test_missing_null_zero_and_defaults_are_separate(result):
    fields = {field.pointer: field for field in result.field_evidence}
    assert fields["/missing_contact"].status == "absent_after_search"
    assert "missing_contact" not in result.data
    assert fields["/approved"].status == "observed" and result.data["approved"] is None
    assert fields["/zero"].status == "observed" and result.data["zero"] == 0
    assert fields["/currency"].status == "schema_default" and result.data["currency"] == "USD"
    assert not fields["/currency"].source_atom_ids
    assert result.completion.execution_complete
    assert result.completion.status == "partial"
    assert result.completion.record_completeness == "not_independently_verified"
    assert result.coverage.unclaimed_region_ids
    assert result.coverage.examined_lines == result.coverage.total_lines


def test_validator_does_not_populate_defaults_without_policy(authored, tmp_path):
    pdf, schema, plan = authored
    result = extract(pdf, schema, workspace=tmp_path, plan=plan)
    assert result.validation.valid
    assert "currency" not in result.data
    assert schema["properties"]["currency"]["default"] == "USD"


def test_required_absence_is_invalid_and_never_fabricated(authored, tmp_path):
    pdf, schema, plan = authored
    schema = deepcopy(schema)
    schema["required"].append("missing_contact")
    result = extract(pdf, schema, workspace=tmp_path, plan=plan)
    assert not result.validation.valid
    assert "missing_contact" not in result.data
    assert any("required property" in error for error in result.validation.errors)


def test_identically_named_parent_sections_remain_distinct(tmp_path):
    pdf = tmp_path / "two-groups.pdf"
    canvas = Canvas(str(pdf), pagesize=(612, 792))
    canvas.setFont("Helvetica", 10)
    for y in (700, 500):
        canvas.drawString(50, y, "Group: Same")
        canvas.drawString(50, y - 25, "Code")
        canvas.drawString(200, y - 25, "Value")
        canvas.drawString(50, y - 50, "0001")
        canvas.drawString(200, y - 50, "7")
        canvas.drawString(50, y - 75, "End items")
    canvas.save()
    schema = {
        "type": "object",
        "properties": {
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}, "value": {"type": "integer"}},
                            },
                        },
                    },
                },
            }
        },
    }
    plan = BindingPlan.model_validate(
        {
            "section": {"path": "/groups", "label": "Group", "field": "name"},
            "tables": [
                {"path": "/groups/*/rows", "key": "code", "columns": {"code": "Code", "value": "Value"}}
            ],
        }
    )
    result = extract(pdf, schema, workspace=tmp_path / "run", plan=plan)
    assert result.data["groups"] == [{"name": "Same", "rows": [{"code": "0001", "value": 7}]}] * 2
    assert len({record.record_id for record in result.source_records}) == 4
    assert result.source_records[1].parent_record_id != result.source_records[3].parent_record_id


def test_keyless_row_without_continuation_permission_is_unresolved(authored, tmp_path):
    pdf, schema, plan = authored
    table = plan.tables[0].model_copy(update={"continuation_marker": "No permission"})
    plan = plan.model_copy(update={"tables": (table,)})
    result = extract(pdf, schema, workspace=tmp_path, plan=plan)
    assert result.coverage.unresolved_boundaries > 0
    assert any("keyless" in item.reason for item in result.unresolved)
    assert result.completion.status == "partial"


def test_header_with_no_data_materializes_an_empty_array(tmp_path):
    pdf = tmp_path / "empty-table.pdf"
    canvas = Canvas(str(pdf))
    canvas.setFont("Helvetica", 10)
    canvas.drawString(50, 700, "Code")
    canvas.drawString(200, 700, "Value")
    canvas.drawString(50, 675, "End items")
    canvas.save()
    schema = {
        "type": "object",
        "required": ["rows"],
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}, "value": {"type": "integer"}},
                },
            }
        },
    }
    plan = BindingPlan.model_validate(
        {"tables": [{"path": "/rows", "key": "code", "columns": {"code": "Code", "value": "Value"}}]}
    )
    result = extract(pdf, schema, workspace=tmp_path / "run", plan=plan)
    assert result.data == {"rows": []}
    assert result.validation.valid


def test_optional_array_without_discovery_binding_is_not_silently_complete(authored, tmp_path):
    pdf, schema, plan = authored
    schema = deepcopy(schema)
    schema["properties"]["extra"] = {
        "type": "array",
        "items": {"type": "object", "properties": {"code": {"type": "string"}}},
    }
    result = extract(pdf, schema, workspace=tmp_path, plan=plan)
    assert any(item.pointer == "/extra" and "binding" in item.reason for item in result.unresolved)
    assert result.completion.status == "partial"


def test_unfinished_cross_page_marker_is_not_complete(authored, tmp_path):
    from pypdf import PdfReader, PdfWriter

    pdf, schema, plan = authored
    writer = PdfWriter()
    writer.add_page(PdfReader(pdf).pages[0])
    first_page = tmp_path / "cut-short.pdf"
    writer.write(first_page)
    result = extract(first_page, schema, workspace=tmp_path / "run", plan=plan, apply_defaults=True)
    assert any(item.reason == "unresolved continuation fragment" for item in result.unresolved)
    assert result.coverage.unresolved_boundaries == 1


def test_competing_or_blank_labels_do_not_apply_defaults(tmp_path):
    pdf = tmp_path / "ambiguity.pdf"
    canvas = Canvas(str(pdf))
    canvas.setFont("Helvetica", 12)
    for y, text in [(700, "Country: India"), (675, "Country: Canada"), (650, "Empty:"), (625, "Code:00042")]:
        canvas.drawString(50, y, text)
    canvas.save()
    schema = {
        "type": "object",
        "properties": {
            "country": {"type": "string", "default": "US"},
            "empty": {"type": "string", "default": "fallback"},
            "code": {"type": "string"},
        },
    }
    plan = BindingPlan.model_validate({"fields": {"/country": "Country", "/empty": "Empty", "/code": "Code"}})
    result = extract(pdf, schema, workspace=tmp_path / "run", plan=plan, apply_defaults=True)
    assert result.data == {"code": "00042"}
    fields = {field.pointer: field for field in result.field_evidence}
    assert fields["/country"].status == "ambiguous"
    assert fields["/empty"].status == "unreadable"
    atoms = result.snapshot.atom_index()
    assert "".join(atoms[atom_id].text for atom_id in fields["/code"].source_atom_ids) == "00042"
