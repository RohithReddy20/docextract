"""Merging answers to page batches of one document."""

import json

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


def _ledger_pdf(tmp_path):
    from reportlab.pdfgen.canvas import Canvas

    path = tmp_path / "ledger.pdf"
    canvas = Canvas(str(path), invariant=1)
    canvas.drawString(48, 760, "Ledger")
    canvas.drawString(48, 730, "A1 10")
    canvas.drawString(48, 710, "A2 20")
    canvas.showPage()
    canvas.drawString(48, 730, "B1 30")
    canvas.showPage()
    canvas.drawString(48, 730, "Total 60")
    canvas.showPage()
    canvas.save()
    return path


LEDGER = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "rows": {
            "type": "array",
            "items": {"$ref": "#/$defs/Row"},
        },
        "total": {"type": "number"},
    },
    "$defs": {
        "Row": {"type": "object", "properties": {"name": {"type": "string"}, "amount": {"type": "number"}}}
    },
}
CUT_OFF = '{"title":"Ledger","rows":[{"name":"A1","amount":10},{"name":"A2","amou'


def test_a_table_cut_off_at_the_output_limit_is_read_page_by_page(tmp_path):
    from docextract.oneshot import extract_oneshot
    from docextract.remote import cited_messages

    model = OneShotModel(
        [
            (CUT_OFF, "length"),
            ('{"rows":[{"name":"A1","amount":10},{"name":"A2","amount":20}]}', "stop"),
            ('{"rows":[{"name":"B1","amount":30}]}', "stop"),
            ('{"rows":[]}', "stop"),
            ('{"total":60}', "stop"),
        ]
    )
    result = extract_oneshot(
        _ledger_pdf(tmp_path), LEDGER, workspace=tmp_path / "run", backend=model, fallback=lambda: "windowed"
    )
    assert result.data == {
        "title": "Ledger",
        "rows": [{"name": "A1", "amount": 10}, {"name": "A2", "amount": 20}, {"name": "B1", "amount": 30}],
        "total": 60,
    }
    document, *pages, rest = model.tasks
    assert [task.first_page for task in pages] == [1, 2, 3] and {task.page_count for task in pages} == {3}
    assert all(
        len(task.page_images) == 1 and list(task.original_schema["properties"]) == ["rows"] for task in pages
    )
    assert "B1" in pages[1].page_texts[0] and "A2" not in pages[1].page_texts[0]
    assert list(rest.original_schema["properties"]) == ["total"] and len(rest.page_images) == 3
    assert rest.max_tokens == 2048 and document.max_tokens is None  # a few values, not a table
    text = cited_messages(pages[1], "c", "oneshot")[0][1]["content"][-1]["text"]
    assert "The image is page 2 of a 3-page document" in text and "--- page 2 ---" in text
    assert not result.unresolved and result.usage.input_tokens == 500 and result.usage.committed_tasks == 5
    evidence = {field.pointer: field for field in result.field_evidence}
    assert evidence["/rows/2/name"].source_lexeme == "B1" and evidence["/total"].source_lexeme == "60"


def test_the_cut_off_table_is_the_list_being_written():
    from docextract.oneshot import cut_off_table, parse_answer

    assert cut_off_table(parse_answer(CUT_OFF), LEDGER) == "rows"
    assert cut_off_table(parse_answer('{"title":"Ledg'), LEDGER) is None  # not inside a table
    assert cut_off_table({"title": "x"}, LEDGER) is None


def test_a_batch_run_queues_every_page_of_a_cut_off_table_at_once(tmp_path):
    from docextract.cited import Pending
    from docextract.models import GenerationResult
    from docextract.oneshot import extract_oneshot

    class Store:
        identity = {"backend": "store", "version": "1"}

        def __init__(self):
            self.saved, self.queued = {}, {}

        def generate_cited(self, task):
            if task.task_id in self.saved:
                return self.saved[task.task_id]
            self.queued[task.task_id] = task
            raise Pending([task.task_id])

    def answer(task):
        if task.page_count:
            # Pages 2 and 3 print identical rows; both are kept.
            row = {"name": "P1", "amount": 1} if task.first_page == 1 else {"name": "Same", "amount": 2}
            raw = json.dumps({"rows": [row]})
            return GenerationResult(raw_text=raw, finish_reason="stop")
        if list(task.original_schema["properties"]) == ["total"]:
            return GenerationResult(raw_text='{"total":6}', finish_reason="stop")
        return GenerationResult(raw_text=CUT_OFF, finish_reason="length")

    store, pdf = Store(), _ledger_pdf(tmp_path)
    rounds = []
    while True:
        store.queued.clear()
        try:
            result = extract_oneshot(pdf, LEDGER, workspace=tmp_path / "run", backend=store)
            break
        except Pending as waiting:
            rounds.append(len(waiting.task_ids))
            store.saved.update({task_id: answer(task) for task_id, task in store.queued.items()})
    assert rounds == [1, 4]  # the document, then its three pages and the remaining field together
    assert [row["name"] for row in result.data["rows"]] == ["P1", "Same", "Same"] and result.data[
        "total"
    ] == 6


def test_rows_continued_across_a_page_break_keep_their_group_label():
    from docextract.oneshot import carry_forward

    texts = [
        "Listing\nR1003 Placebo Week 0 2009-02-01 08:00\nWeek 6 2009-03-12 08:30\nWeek 10 2009-04-20 08:40\n"
        "R1004 Active Week 0 2009-03-01 08:10\nWeek 6 2009-04-12 09:00\n",
        "Listing\nWeek 10 2009-05-20 10:30\nWeek 16 2009-07-01 11:00\nR1005 Placebo Week 0 2009-03-10 09:47\n"
        "Week 6 2009-04-20 11:00\n",
    ]

    def row(subject, product, week, date, time):
        return {"subject": subject, "product": product, "week": week, "date": date, "time": time}

    pages = [
        [
            row("R1003", "Placebo", "Week 0", "2009-02-01", "08:00"),
            row("R1003", "Placebo", "Week 6", "2009-03-12", "08:30"),
            row("R1003", "Placebo", "Week 10", "2009-04-20", "08:40"),
            row("R1004", "Active", "Week 0", "2009-03-01", "08:10"),
            row("R1004", "Active", "Week 6", "2009-04-12", "09:00"),
        ],
        [
            # Read alone, the continued rows took the label printed below them, or none.
            row("R1005", "Placebo", "Week 10", "2009-05-20", "10:30"),
            row(None, "Placebo", "Week 16", "2009-07-01", "11:00"),
            row("R1005", "Placebo", "Week 0", "2009-03-10", "09:47"),
            row("R1005", "Placebo", "Week 6", "2009-04-20", "11:00"),
        ],
    ]
    fixed = carry_forward(pages, texts)
    assert [(r["subject"], r["product"]) for r in fixed[1]] == [
        ("R1004", "Active"),
        ("R1004", "Active"),
        ("R1005", "Placebo"),
        ("R1005", "Placebo"),
    ]
    assert carry_forward(fixed, texts) == fixed  # correct labels are left alone
    assert pages[1][0]["subject"] == "R1005"  # the input is not modified
    # A label the schema allows to be null is filled in only while no label is printed above the row.
    nullable = frozenset({"subject"})
    assert carry_forward(pages, texts, nullable)[1][1]["subject"] == "R1004"
    later = [
        row("R1006", "Active", "Week 0", "2009-03-12", "07:00"),
        row(None, "Active", "Week 6", "2009-04-22", "07:30"),
    ]
    texts.append("Listing\nR1006 Active Week 0 2009-03-12 07:00\nWeek 6 2009-04-22 07:30\n")
    assert carry_forward([*pages, later], texts, nullable)[2][1]["subject"] is None


def test_a_page_cut_off_at_its_text_sized_cap_is_asked_again_uncapped(tmp_path):
    from reportlab.pdfgen.canvas import Canvas

    from docextract.oneshot import extract_oneshot

    path = tmp_path / "list.pdf"
    canvas = Canvas(str(path), invariant=1)
    for line in range(12):
        canvas.drawString(48, 760 - 20 * line, f"Item {line:02d} described at some length for the listing")
    canvas.showPage()
    canvas.save()
    schema = {
        "type": "object",
        "properties": {"rows": LEDGER["properties"]["rows"]},
        "$defs": LEDGER["$defs"],
    }
    model = OneShotModel(
        [
            ('{"rows":[{"name":"Item 00"', "length"),
            ('{"rows":[{"name":"Item 00","amount":1},{"name":"It', "length"),
            ('{"rows":[{"name":"Item 00","amount":1},{"name":"Item 01","amount":2}]}', "stop"),
        ]
    )
    result = extract_oneshot(path, schema, workspace=tmp_path / "run", backend=model)
    capped, uncapped = model.tasks[1:]
    assert capped.max_tokens and capped.max_tokens >= 1024 and uncapped.max_tokens is None
    assert [row["name"] for row in result.data["rows"]] == ["Item 00", "Item 01"]
    assert not result.unresolved and result.usage.committed_tasks == 3
