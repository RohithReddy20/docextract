"""Merging answers to page batches of one document."""

from docextract.oneshot import merge_answers


def test_scalars_take_the_first_value_found_in_page_order():
    parts = [{"invoice": None, "total": None}, {"invoice": "A-1", "total": "10"}, {"invoice": "B-2"}]
    assert merge_answers(parts) == {"invoice": "A-1", "total": "10"}


def test_lists_concatenate_and_drop_records_repeated_from_an_earlier_batch():
    header = {"name": "Fund", "id": 7}
    parts = [
        {"rows": [{"n": 1}, {"n": 2}], "funds": [header]},
        {"rows": [{"n": 2}, {"n": 3}], "funds": [header]},
    ]
    assert merge_answers(parts) == {"rows": [{"n": 1}, {"n": 2}, {"n": 3}], "funds": [header]}


def test_identical_rows_within_one_batch_are_kept():
    parts = [{"rows": [{"qty": 1}, {"qty": 1}]}, {"rows": []}]
    assert merge_answers(parts) == {"rows": [{"qty": 1}, {"qty": 1}]}


def test_nested_objects_merge_key_by_key_and_empty_answers_are_ignored():
    parts = [{}, {"payer": {"name": "ACME", "id": None}}, {"payer": {"id": "9", "name": "Other"}}, None]
    assert merge_answers(parts) == {"payer": {"name": "ACME", "id": "9"}}
    assert merge_answers([None, {}]) is None


def test_conform_types_values_and_fills_two_state_checkboxes():
    from docextract.oneshot import conform

    schema = {
        "type": "object",
        "properties": {
            "amount": {"type": ["string", "null"]},
            "count": {"type": "integer"},
            "final": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "description": "TWO-STATE: true only if marked; false otherwise incl. blank. Never null.",
            },
            "maybe": {"type": ["boolean", "null"], "description": "True if checked."},
            "kind": {"type": "string", "enum": ["Oil", "Gas"]},
            "rows": {"type": "array", "items": {"$ref": "#/$defs/row"}},
        },
        "$defs": {
            "row": {"type": "object", "properties": {"qty": {"type": "number"}, "ok": {"type": "boolean"}}}
        },
    }
    answer = {
        "amount": 41000.0,
        "count": "1,200",
        "kind": "gas",
        "rows": [{"qty": "2.5", "ok": "Yes"}],
        "maybe": None,
    }
    assert conform(answer, schema) == {
        "amount": "41000",
        "count": 1200,
        "kind": "Gas",
        "rows": [{"qty": 2.5, "ok": True}],
        "maybe": None,
        "final": False,
    }
    assert conform({"count": "about 3"}, schema)["count"] == "about 3"


def test_located_evidence_cites_each_printed_value_by_page_and_box(tmp_path):
    from reportlab.pdfgen.canvas import Canvas

    from docextract.oneshot import locate_evidence
    from docextract.parsing.pdfium import PdfiumParser

    path = tmp_path / "invoice.pdf"
    canvas = Canvas(str(path), invariant=1)
    canvas.drawString(48, 740, "Invoice A-17")
    canvas.drawString(48, 712, "Total 1,250.00")
    canvas.showPage()
    canvas.drawString(48, 740, "Approved by Dana Reyes")
    canvas.showPage()
    canvas.save()
    schema = {
        "type": "object",
        "properties": {
            "invoice": {"type": "string"},
            "total": {"type": "number"},
            "approver": {"type": "string"},
            "paid": {"type": "boolean"},
            "missing": {"type": ["string", "null"]},
        },
    }
    answer = {
        "invoice": "A-17",
        "total": 1250.0,
        "approver": "Dana Reyes",
        "paid": True,
        "missing": "nowhere",
    }
    citations = {c["field_path"]: c for c in locate_evidence(answer, schema, PdfiumParser().parse(path))}
    assert set(citations) == {"invoice", "total", "approver"}
    assert citations["invoice"]["page"] == 1 and citations["approver"]["page"] == 2
    assert citations["total"]["reference_text"] == "1,250.00"
    x, y, width, height = citations["approver"]["bbox"]
    assert 0 < x < 0.5 and 0 < y < 0.2 and 0 < width < 1 and 0 < height < 0.1


def test_answers_cut_off_at_the_output_limit_keep_their_complete_elements():
    from docextract.oneshot import parse_answer

    assert parse_answer('<think>\n\n</think>\n\n{"a": 1, "rows": [{"n": 1}, {"n": 2}]}') == {
        "a": 1,
        "rows": [{"n": 1}, {"n": 2}],
    }
    # The half-written value is dropped; the fields the last record completed are kept.
    assert parse_answer('{"a": "x", "rows": [{"n": 1, "m": "y"}, {"n": 2, "m": "unfini') == {
        "a": "x",
        "rows": [{"n": 1, "m": "y"}, {"n": 2}],
    }
    assert parse_answer('```json\n{"a": [1, 2, 3')["a"] == [1, 2]
    assert parse_answer("no json here") == {}


def test_the_prepared_schema_closes_objects_and_promotes_repeated_structures():
    from docextract.oneshot import prepare_schema

    schema = {
        "type": "object",
        "properties": {"id": {"type": "string"}},
        "repeated_structure": {"rows": {"type": "array", "items": {"type": "object", "properties": {}}}},
    }
    prepared = prepare_schema(schema)
    assert prepared["additionalProperties"] is False
    assert prepared["properties"]["rows"]["items"]["additionalProperties"] is False
    assert "repeated_structure" not in prepared and "repeated_structure" in schema


def _invoice_pdf(tmp_path):
    from reportlab.pdfgen.canvas import Canvas

    path = tmp_path / "invoice.pdf"
    canvas = Canvas(str(path), invariant=1)
    canvas.drawString(48, 740, "Invoice A-17")
    canvas.drawString(48, 712, "Total 1,250.00")
    canvas.showPage()
    canvas.save()
    return path


INVOICE = {
    "type": "object",
    "properties": {
        "invoice": {"type": "string"},
        "total": {"type": "number"},
        "paid": {
            "type": ["boolean", "null"],
            "description": "True if the PAID box is ticked, false if blank.",
        },
        "note": {"type": ["string", "null"]},
    },
}


class OneShotModel:
    identity = {"backend": "scripted-oneshot", "version": "1"}

    def __init__(self, answers):
        self.answers, self.tasks = list(answers), []

    def generate_cited(self, task):
        from docextract.models import GenerationResult

        self.tasks.append(task)
        raw, finish = self.answers.pop(0)
        return GenerationResult(raw_text=raw, finish_reason=finish, input_tokens=100, output_tokens=20)


def test_whole_document_answer_is_typed_and_located(tmp_path):
    from docextract.cited import extract_cited
    from docextract.remote import cited_messages

    model = OneShotModel(
        [('<think>\n\n</think>\n\n{"invoice": "A-17", "total": "1,250.00", "note": null}', "stop")]
    )
    result = extract_cited(
        _invoice_pdf(tmp_path), INVOICE, workspace=tmp_path / "run", backend=model, protocol="oneshot"
    )
    assert result.data == {"invoice": "A-17", "total": 1250.0, "note": None, "paid": False}
    evidence = {field.pointer: field for field in result.field_evidence}
    assert evidence["/invoice"].source_atom_ids and evidence["/total"].source_lexeme == "1,250.00"
    assert evidence["/note"].status == "absent_after_search"
    assert evidence["/paid"].transform == "image_read" and not evidence["/paid"].source_atom_ids
    assert (tmp_path / "run" / "result.json").exists()
    task = model.tasks[0]
    assert task.protocol == "oneshot" and len(task.page_images) == 1 and "Invoice A-17" in task.page_texts[0]
    messages, _ = cited_messages(task, "a", "oneshot")
    assert messages[0]["role"] == "system" and messages[1]["content"][0] == {"type": "image"}
    assert '"additionalProperties": false' in messages[1]["content"][-1]["text"]


def test_too_long_requests_shrink_their_images_then_fall_back(tmp_path):
    from docextract.oneshot import extract_oneshot

    model = OneShotModel([("", "input_limit"), ("", "input_limit"), ("", "input_limit")])
    outcome = extract_oneshot(
        _invoice_pdf(tmp_path),
        INVOICE,
        workspace=tmp_path / "run",
        backend=model,
        dpi=150,
        fallback=lambda: "windowed",
    )
    assert outcome == "windowed"
    sides = [__import__("PIL.Image").Image.open(task.page_images[0]).width for task in model.tasks]
    assert sides[0] > sides[1] > sides[2]


def test_a_document_without_a_stored_answer_stays_empty_and_partial(tmp_path):
    from docextract.oneshot import extract_oneshot

    model = OneShotModel([("", "not_stored")])
    result = extract_oneshot(_invoice_pdf(tmp_path), INVOICE, workspace=tmp_path / "run", backend=model)
    assert result.data == {} and result.completion.status == "partial"
    assert any("not answered" in item.reason for item in result.unresolved)
