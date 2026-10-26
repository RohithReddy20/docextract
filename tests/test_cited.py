"""Scripted cited-value responses exercise contracts and recovery, never learned extraction quality."""

import json

import pytest
from reportlab.pdfgen.canvas import Canvas

from docextract.cited import CitedTask, extract_cited, find_value, parse_number
from docextract.models import GenerationResult

TABLE = {
    "type": "object",
    "required": ["invoice_id", "rows"],
    "properties": {
        "invoice_id": {"type": "string"},
        "note": {"type": ["string", "null"]},
        "currency": {"type": "string"},
        "rows": {
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


def make_pdf(tmp_path, pages, name="source.pdf"):
    path = tmp_path / name
    canvas = Canvas(str(path), invariant=1)
    for lines in pages:
        for index, line in enumerate(lines):
            cells = line if isinstance(line, tuple) else (line,)
            for x, text in zip((48, 226, 448), cells, strict=False):
                canvas.drawString(x, 740 - index * 28, text)
        canvas.showPage()
    canvas.save()
    return path


class Scripted:
    identity = {"backend": "scripted-cited", "version": "1"}

    def __init__(self, script, wrap=lambda text: text):
        self.script, self.wrap, self.calls = script, wrap, 0

    def generate_cited(self, task: CitedTask):
        self.calls += 1
        return GenerationResult(raw_text=self.wrap(json.dumps(self.script(task))))


def cell(task, text):
    return next(unit.alias for unit in task.units if unit.text == text)


def cells(task, text):
    return [unit.alias for unit in task.units if unit.text == text]


def table_script(task):
    rows = []
    for alias in cells(task, "A1") + cells(task, "B2"):
        index = next(i for i, unit in enumerate(task.units) if unit.alias == alias)
        code, quantity, amount = task.units[index : index + 3]
        rows.append(
            {
                "code": {"value": code.text, "source": code.alias},
                "quantity": {"value": int(quantity.text), "source": quantity.alias},
                "amount": {"value": float(amount.text.replace("$", "")), "source": amount.alias},
            }
        )
    rows.sort(key=lambda row: int(row["code"]["source"][1:]))
    output = {"rows": rows}
    invoice = [unit for unit in task.units if unit.text.startswith("Invoice")]
    if invoice:
        output["invoice_id"] = {"value": "0017", "source": invoice[0].alias}
    return output


def test_identical_rows_stay_separate_and_values_cite_their_own_characters(tmp_path):
    pdf = make_pdf(
        tmp_path,
        [
            [
                "Invoice ID: 0017",
                ("Code", "Qty", "Amount"),
                ("A1", "2", "$12.50"),
                ("A1", "2", "$12.50"),
                ("B2", "1", "$5.00"),
            ]
        ],
    )
    result = extract_cited(pdf, TABLE, workspace=tmp_path / "run", backend=Scripted(table_script))
    row = {"code": "A1", "quantity": 2, "amount": 12.5}
    assert result.data == {
        "invoice_id": "0017",
        "note": None,
        "rows": [row, row, {"code": "B2", "quantity": 1, "amount": 5.0}],
    }
    assert len({record.record_id for record in result.source_records}) == 3
    fields = {field.pointer: field for field in result.field_evidence}
    assert fields["/invoice_id"].source_lexeme == "0017" and fields["/invoice_id"].transform == "verbatim"
    assert fields["/rows/0/amount"].transform == "number_normalized"
    assert fields["/rows/0/amount"].source_atom_ids != fields["/rows/1/amount"].source_atom_ids
    assert fields["/note"].status == "absent_after_search"
    assert "currency" not in result.data  # absent and non-nullable: omitted, not invented
    assert result.validation.valid


def test_overlapping_windows_merge_by_cited_occurrence_not_by_value(tmp_path):
    lines = [("A1", "2", "$12.50") for _ in range(6)]
    pdf = make_pdf(tmp_path, [[("Code", "Qty", "Amount"), *lines]])
    backend = Scripted(table_script)
    result = extract_cited(
        pdf, TABLE, workspace=tmp_path / "run", backend=backend, max_window_chars=150, overlap_lines=2
    )
    assert backend.calls > 1
    assert len(result.data["rows"]) == 6 and len(result.source_records) == 6
    again = extract_cited(
        pdf, TABLE, workspace=tmp_path / "run", backend=backend, max_window_chars=150, overlap_lines=2
    )
    assert again == result and backend.calls == backend.calls


def test_value_spanning_pages_cites_both_fragments(tmp_path):
    pdf = make_pdf(tmp_path, [[("Code", "Description"), ("A1", "Extended support")], ["through December"]])
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"description": {"type": "string"}}},
            }
        },
    }

    def script(task):
        sources = [cell(task, "Extended support"), cell(task, "through December")]
        return {"items": [{"description": {"value": "Extended support through December", "source": sources}}]}

    result = extract_cited(pdf, schema, workspace=tmp_path / "run", backend=Scripted(script))
    assert result.data == {"items": [{"description": "Extended support through December"}]}
    evidence = result.field_evidence[0]
    pages = {result.snapshot.atom_index()[atom].page_index for atom in evidence.source_atom_ids}
    assert pages == {0, 1} and evidence.support == "character"


def test_uncited_duplicates_take_distinct_occurrences_and_unsupported_values_are_flagged(tmp_path):
    pdf = make_pdf(tmp_path, [[("Code", "Qty"), ("A1", "2"), ("A1", "2")]])
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}, "quantity": {"type": "integer"}},
                },
            },
            "total": {"type": "number"},
        },
    }
    script = {"rows": [{"code": "A1", "quantity": 2}, {"code": "A1", "quantity": 2}], "total": 99}
    result = extract_cited(pdf, schema, workspace=tmp_path / "run", backend=Scripted(lambda task: script))
    fields = {field.pointer: field for field in result.field_evidence}
    assert fields["/rows/0/code"].transform == "located_without_citation"
    assert set(fields["/rows/0/code"].source_atom_ids).isdisjoint(fields["/rows/1/code"].source_atom_ids)
    assert fields["/total"].transform == "model_value_unsupported" and not fields["/total"].source_atom_ids
    assert any(item.pointer == "/total" for item in result.unresolved)


def test_types_enums_fences_and_unknown_keys(tmp_path):
    pdf = make_pdf(tmp_path, [["Status: approved", "Count: 7", "Date: March 5, 2024"]])
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["Approved", "Rejected"]},
            "count": {"type": "integer"},
            "date": {"type": "string", "description": "ISO date"},
        },
    }

    def script(task):
        line = {unit.text.split(":")[0]: unit.alias for unit in task.units}
        return {
            "status": {"value": "approved", "source": line["Status"]},
            "count": {"value": "7", "source": line["Count"]},
            "date": {"value": "2024-03-05", "source": line["Date"]},
            "extra": {"value": 1},
        }

    result = extract_cited(
        pdf,
        schema,
        workspace=tmp_path / "run",
        backend=Scripted(script, wrap=lambda text: "```json\n" + text + "\n```"),
    )
    assert result.data == {"status": "Approved", "count": 7, "date": "2024-03-05"}
    fields = {field.pointer: field for field in result.field_evidence}
    assert fields["/date"].transform == "date_normalized" and fields["/date"].source_lexeme == "March 5, 2024"
    assert fields["/status"].transform == "space_or_case_normalized"
    assert result.coverage.rejected_proposals == 1


@pytest.mark.parametrize("stage", ["generated", "raw_saved", "validated", "committed"])
def test_cited_extraction_recovers_at_every_durable_boundary(tmp_path, stage):
    pdf = make_pdf(tmp_path, [[("Code", "Qty", "Amount"), ("A1", "2", "$12.50"), ("A1", "2", "$12.50")]])
    backend = Scripted(table_script)

    def interrupt(actual, task):
        if actual == stage:
            raise InterruptedError(stage)

    with pytest.raises(InterruptedError):
        extract_cited(pdf, TABLE, workspace=tmp_path / "run", backend=backend, fault=interrupt)
    result = extract_cited(pdf, TABLE, workspace=tmp_path / "run", backend=backend)
    assert len(result.data["rows"]) == 2
    assert backend.calls == (2 if stage == "generated" else 1)


def test_unparseable_single_line_is_reported_and_other_lines_survive(tmp_path):
    pdf = make_pdf(tmp_path, [[("Code", "Qty", "Amount"), ("A1", "2", "$12.50"), ("B2", "1", "$5.00")]])

    class Flaky(Scripted):
        def generate_cited(self, task):
            self.calls += 1
            if any(unit.text == "B2" for unit in task.units):
                return GenerationResult(raw_text='{"rows": [')
            return GenerationResult(raw_text=json.dumps(table_script(task)))

    result = extract_cited(
        pdf, TABLE, workspace=tmp_path / "run", backend=Flaky(table_script), overlap_lines=0
    )
    assert result.data["rows"] == [{"code": "A1", "quantity": 2, "amount": 12.5}]
    assert any("not interpreted: UnparseableOutput" in item.reason for item in result.unresolved)
    assert result.completion.status == "partial"


def test_number_and_date_location_rules():
    assert parse_number("$1,234.50") == 1234.5 and parse_number("(500)") == -500
    assert find_value(1234.5, "number", "Total: $1,234.50 due")[2] == "number_normalized"
    assert find_value(12, "integer", "Rate 12%")[:2] == (5, 8)
    assert find_value("2024-03-05", "string", "on 5 March 2024")[2] == "date_normalized"
    assert find_value(7, "integer", "Count: 17") is None
    assert find_value("Smith", "string", "J. SMITH")[2] == "space_or_case_normalized"


def test_batch_runner_queues_every_window_and_matches_sequential_results(tmp_path):
    from docextract.batch import run_batch

    rows = [("Code", "Qty", "Amount"), *[("A1", "2", "$12.50") for _ in range(5)], ("B2", "1", "$5.00")]
    first = make_pdf(tmp_path, [rows], name="first.pdf")
    second = make_pdf(tmp_path, [rows[:3]], name="second.pdf")

    class Engine:
        identity = {"backend": "scripted-batch", "version": "1"}

        def __init__(self):
            self.batches = []

        def generate_batch(self, tasks):
            self.batches.append(len(tasks))
            outputs = []
            for task in tasks:
                if len(task.units) > 9:  # force a subdivision round for the larger windows
                    outputs.append(GenerationResult(raw_text="{}", finish_reason="length"))
                else:
                    outputs.append(GenerationResult(raw_text=json.dumps(table_script(task))))
            return outputs

    engine = Engine()
    cases = [("first", first, TABLE), ("second", second, TABLE)]
    outcomes = run_batch(cases, root=tmp_path / "batch", engine=engine, max_window_chars=200, overlap_lines=1)
    assert engine.batches[0] >= 3 and len(engine.batches) >= 2
    assert len(outcomes["first"].data["rows"]) == 6 and len(outcomes["second"].data["rows"]) == 2
    again = run_batch(cases, root=tmp_path / "batch", engine=engine, max_window_chars=200, overlap_lines=1)
    assert again == outcomes and len(engine.batches) == len(engine.batches)


def test_a_source_reference_is_never_accepted_as_a_value(tmp_path):
    pdf = make_pdf(tmp_path, [[("Reference", "Units"), ("0019", "1")]])
    schema = {
        "type": "object",
        "properties": {
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "units": {"type": "integer"}},
                },
            }
        },
    }

    def script(task):
        ref, units = cell(task, "0019"), cell(task, "1")
        return {"lines": [{"id": {"value": ref, "source": ref}, "units": {"value": 1, "source": units}}]}

    result = extract_cited(pdf, schema, workspace=tmp_path / "run", backend=Scripted(script))
    assert result.data == {"lines": [{"units": 1}]}
    assert any("source reference was returned" in item.reason for item in result.unresolved)


def test_table_completion_copies_skipped_rows_with_the_demonstrated_mapping(tmp_path):
    rows = [("Code", "Qty", "Amount")] + [(f"C{i}", str(i), f"${i}.50") for i in range(1, 7)]
    pdf = make_pdf(tmp_path, [rows])
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                },
            }
        },
    }

    def first_two(task):
        output = []
        for code in ("C1", "C2"):
            index = next(i for i, unit in enumerate(task.units) if unit.text == code)
            code_unit, quantity, amount = task.units[index : index + 3]
            output.append(
                {
                    "code": {"value": code, "source": code_unit.alias},
                    "quantity": {"value": int(quantity.text), "source": quantity.alias},
                    "amount": {"value": float(amount.text[1:]), "source": amount.alias},
                }
            )
        return {"rows": output}

    plain = extract_cited(pdf, schema, workspace=tmp_path / "plain", backend=Scripted(first_two))
    assert len(plain.data["rows"]) == 2
    completed = extract_cited(
        pdf, schema, workspace=tmp_path / "completed", backend=Scripted(first_two), complete_tables=True
    )
    assert [row["code"] for row in completed.data["rows"]] == [f"C{i}" for i in range(1, 7)]
    assert completed.data["rows"][5] == {"code": "C6", "quantity": 6, "amount": 6.5}
    assert completed.coverage.materialized_table_rows == 4
    copied = [field for field in completed.field_evidence if field.transform == "copied_table_row"]
    assert len(copied) == 12 and all(field.source_atom_ids for field in copied)


def test_replay_reassembles_stored_answers_under_new_options_without_generating(tmp_path):
    from docextract.batch import CachedParser, ReplayEngine, run_batch

    rows = [("Code", "Qty", "Amount")] + [(f"C{i}", str(i), f"${i}.50") for i in range(1, 6)]
    pdf = make_pdf(tmp_path, [rows])
    schema = TABLE

    class Engine:
        identity = {"backend": "scripted-batch", "version": "2"}

        def generate_batch(self, tasks):
            outputs = []
            for task in tasks:
                answer = table_script(task)  # cites only rows starting with A1 or B2: none here
                rows_out = []
                for code in ("C1", "C2"):
                    index = next(i for i, unit in enumerate(task.units) if unit.text == code)
                    c, q, a = task.units[index : index + 3]
                    rows_out.append(
                        {
                            "code": {"value": code, "source": c.alias},
                            "quantity": {"value": int(q.text), "source": q.alias},
                            "amount": {"value": float(a.text[1:]), "source": a.alias},
                        }
                    )
                answer["rows"] = rows_out
                outputs.append(GenerationResult(raw_text=json.dumps(answer)))
            return outputs

    cases = [("doc", pdf, schema)]
    first = run_batch(
        cases,
        root=tmp_path / "run",
        engine=Engine(),
        parser=CachedParser(store=tmp_path / "run" / "snapshots"),
    )
    assert len(first["doc"].data["rows"]) == 2
    replay = ReplayEngine(Engine.identity)
    (tmp_path / "replay").mkdir()
    (tmp_path / "replay" / "generations.jsonl").write_bytes(
        (tmp_path / "run" / "generations.jsonl").read_bytes()
    )
    second = run_batch(
        cases,
        root=tmp_path / "replay",
        engine=replay,
        parser=CachedParser(store=tmp_path / "run" / "snapshots", parse_missing=False),
        complete_tables=True,
    )
    assert replay.missing == 0
    assert [row["code"] for row in second["doc"].data["rows"]] == [f"C{i}" for i in range(1, 6)]


def test_streaming_runner_stores_each_answer_and_assembles_documents_as_they_finish(tmp_path):
    from docextract.batch import run_batch

    rows = [("Code", "Qty", "Amount"), *[("A1", "2", "$12.50") for _ in range(5)], ("B2", "1", "$5.00")]
    first = make_pdf(tmp_path, [rows], name="first.pdf")
    second = make_pdf(tmp_path, [rows[:3]], name="second.pdf")
    log = tmp_path / "batch" / "generations.jsonl"

    class Engine:
        identity = {"backend": "scripted-stream", "version": "1"}

        def __init__(self):
            self.queue, self.events = [], []

        def submit(self, tasks, first=False):
            self.queue.extend(tasks)

        def busy(self):
            return bool(self.queue)

        def poll(self):
            # One answer per step, newest first, so completion order differs from submission order.
            if not self.queue:
                return []
            task = self.queue.pop()
            stored = len(log.read_text().splitlines()) if log.exists() else 0
            self.events.append(("answer", stored))
            if len(task.units) > 9:
                return [(task.task_id, GenerationResult(raw_text="{}", finish_reason="length"))]
            return [(task.task_id, GenerationResult(raw_text=json.dumps(table_script(task))))]

    engine = Engine()
    finished = []
    cases = [("first", first, TABLE), ("second", second, TABLE)]
    outcomes = run_batch(
        cases,
        root=tmp_path / "batch",
        engine=engine,
        on_result=lambda case_id, outcome: finished.append(case_id),
        max_window_chars=200,
        overlap_lines=1,
    )
    # Every answer is on disk before the next one is requested.
    assert [stored for _, stored in engine.events] == list(range(len(engine.events)))
    assert finished == ["second", "first"]
    assert len(outcomes["first"].data["rows"]) == 6 and len(outcomes["second"].data["rows"]) == 2
    sequential = run_batch(
        cases, root=tmp_path / "again", engine=engine, max_window_chars=200, overlap_lines=1
    )
    assert sequential["first"].data == outcomes["first"].data


def test_replay_assembles_a_partly_answered_document_without_subdividing(tmp_path):
    from docextract.batch import CachedParser, ReplayEngine, run_batch

    rows = [("Code", "Qty", "Amount"), *[("A1", "2", "$12.50") for _ in range(5)], ("B2", "1", "$5.00")]
    pdf = make_pdf(tmp_path, [rows])

    class Engine:
        identity = {"backend": "scripted-batch", "version": "3"}

        def generate_batch(self, tasks):
            return [GenerationResult(raw_text=json.dumps(table_script(task))) for task in tasks]

    cases = [("doc", pdf, TABLE)]
    run_batch(
        cases,
        root=tmp_path / "run",
        engine=Engine(),
        parser=CachedParser(store=tmp_path / "run" / "snapshots"),
        max_window_chars=120,
        overlap_lines=0,
    )
    answers = (tmp_path / "run" / "generations.jsonl").read_text().splitlines()
    assert len(answers) >= 2
    for name in ("replay", "replay-nulls"):
        (tmp_path / name).mkdir()
        # A deadline stopped the run after the first answer.
        (tmp_path / name / "generations.jsonl").write_text(answers[0] + "\n")
    replay = ReplayEngine(Engine.identity)
    outcome = run_batch(
        cases,
        root=tmp_path / "replay",
        engine=replay,
        parser=CachedParser(store=tmp_path / "run" / "snapshots", parse_missing=False),
        max_window_chars=120,
        overlap_lines=0,
    )["doc"]
    assert replay.missing == len(answers) - 1
    assert outcome.completion.status == "partial"
    assert sum("Unanswered" in item.reason for item in outcome.unresolved) == len(answers) - 1
    assert 0 < len(outcome.data["rows"]) < 6
    # Absence is not claimed for a partly read document unless asked for.
    assert "note" not in outcome.data
    nulls = run_batch(
        cases,
        root=tmp_path / "replay-nulls",
        engine=ReplayEngine(Engine.identity),
        parser=CachedParser(store=tmp_path / "run" / "snapshots", parse_missing=False),
        max_window_chars=120,
        overlap_lines=0,
        nulls_if_partial=True,
    )["doc"]
    assert nulls.data["note"] is None and nulls.data["rows"] == outcome.data["rows"]


def test_local_replay_reassembles_an_exported_run_from_stored_snapshots(tmp_path):
    import hashlib

    from docextract.batch import CachedParser, run_batch
    from tools.replay_local import replay

    rows = [("Code", "Qty", "Amount"), *[("A1", "2", "$12.50") for _ in range(5)], ("B2", "1", "$5.00")]
    pdf = make_pdf(tmp_path, [rows])

    class Engine:
        identity = {"backend": "scripted-batch", "version": "4"}

        def generate_batch(self, tasks):
            return [GenerationResult(raw_text=json.dumps(table_script(task))) for task in tasks]

    run = tmp_path / "run"
    remote = run_batch(
        [("case1", pdf, TABLE)],
        root=run,
        engine=Engine(),
        parser=CachedParser(store=run / "snapshots"),
        max_window_chars=120,
    )["case1"]
    (run / "runtime.json").write_text(json.dumps({"identity": Engine.identity}))
    packet = tmp_path / "packet"
    (packet / "inference").mkdir(parents=True)
    (packet / "inference" / "case1.schema.json").write_text(json.dumps(TABLE))
    case = {
        "id": "case1",
        "schema": "case1.schema.json",
        "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
    }
    (packet / "inference" / "documents.json").write_text(json.dumps({"cases": [case]}))
    summary = replay(packet, run, tmp_path / "local", max_window_chars=120)
    assert summary == {"cases": 1, "failed": {}, "unanswered_windows": 0}
    local = json.loads((tmp_path / "local" / "case1" / remote.run_id / "result.json").read_text())
    assert local["data"] == remote.data
    with pytest.raises(ValueError):
        replay(packet, run, packet, max_window_chars=120)


def test_windows_sharing_a_schema_group_are_submitted_together():
    from docextract.batch import adjacent_prompts

    def task(name, part):
        return CitedTask(task_id=name, original_schema={"type": "object"}, units=(), schema_part=part)

    first, second = {"type": "object", "properties": {"a": {}}}, {"type": "object", "properties": {"b": {}}}
    tasks = [task("w1g1", first), task("w1g2", second), task("w2g1", first), task("w2g2", second)]
    assert [t.task_id for t in adjacent_prompts(tasks)] == ["w1g1", "w2g1", "w1g2", "w2g2"]
    plain = [task("w1", None), task("w2", None)]
    assert adjacent_prompts(plain) == plain


def test_document_strings_returns_printed_text_when_the_model_recased_it(tmp_path):
    pdf = make_pdf(tmp_path, [["Payer: ACME TRUST BANK  W&IM"]])
    schema = {"type": "object", "properties": {"payer": {"type": "string"}}}

    def script(task):
        unit = next(unit for unit in task.units if "ACME" in unit.text)
        return {"payer": {"value": "Acme Trust Bank W&IM", "source": unit.alias}}

    model = extract_cited(pdf, schema, workspace=tmp_path / "model", backend=Scripted(script))
    assert model.data == {"payer": "Acme Trust Bank W&IM"}
    printed = extract_cited(
        pdf, schema, workspace=tmp_path / "printed", backend=Scripted(script), document_strings=True
    )
    assert printed.data == {"payer": "ACME TRUST BANK W&IM"}
    assert printed.field_evidence[0].transform == "space_or_case_normalized"


def test_plain_values_are_located_and_overlapping_windows_do_not_duplicate_rows(tmp_path):
    from docextract.remote import cited_messages

    rows = [("Code", "Qty", "Amount")] + [(f"C{i}", str(i), f"${i}.50") for i in range(1, 9)]
    pdf = make_pdf(tmp_path, [rows])
    schema = {
        "type": "object",
        "properties": {
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "quantity": {"type": "integer"},
                        "amount": {"type": "number"},
                    },
                },
            }
        },
    }
    prompts = []

    def plain(task):
        prompts.append(cited_messages(task, "a", "values"))
        found = []
        for index, unit in enumerate(task.units):
            if unit.text.startswith("C") and unit.text[1:].isdigit():
                quantity, amount = task.units[index + 1], task.units[index + 2]
                found.append(
                    {"code": unit.text, "quantity": int(quantity.text), "amount": float(amount.text[1:])}
                )
        return {"rows": found}

    result = extract_cited(
        pdf,
        schema,
        workspace=tmp_path / "run",
        backend=Scripted(plain),
        max_window_chars=150,
        overlap_lines=2,
    )
    assert len(prompts) > 1  # overlapping windows saw some rows twice
    assert [row["code"] for row in result.data["rows"]] == [f"C{i}" for i in range(1, 9)]
    assert result.data["rows"][4] == {"code": "C5", "quantity": 5, "amount": 5.5}
    located = [field for field in result.field_evidence if field.pointer.endswith("/code")]
    assert all(field.transform == "located_without_citation" and field.source_atom_ids for field in located)
    messages, _ = prompts[0]
    assert "Write plain JSON values" in messages[0]["content"]
    assert "\ts" not in messages[1]["content"] and "C1\t1\t$1.50" in messages[1]["content"]


def test_root_null_policy_writes_nulls_only_outside_records(tmp_path):
    pdf = make_pdf(tmp_path, [[("Code", "Qty", "Amount"), ("A1", "2", "$12.50")]])
    schema = {
        "type": "object",
        "properties": {
            "note": {"type": ["string", "null"]},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}, "memo": {"type": ["string", "null"]}},
                },
            },
        },
    }

    def script(task):
        return {"rows": [{"code": {"value": "A1", "source": cell(task, "A1")}}]}

    both = extract_cited(pdf, schema, workspace=tmp_path / "both", backend=Scripted(script))
    root = extract_cited(
        pdf, schema, workspace=tmp_path / "root", backend=Scripted(script), absent_nullable="root"
    )
    assert both.data == {"note": None, "rows": [{"code": "A1", "memo": None}]}
    assert root.data == {"note": None, "rows": [{"code": "A1"}]}


def test_one_engine_runs_several_prompt_variants_of_the_same_documents(tmp_path):
    from docextract.batch import run_batch

    rows = [("Code", "Qty", "Amount"), ("A1", "2", "$12.50"), ("B2", "1", "$5.00")]
    pdf = make_pdf(tmp_path, [rows])

    class Engine:
        identity = {"backend": "scripted-variants", "version": "1"}

        def __init__(self):
            self.protocols = []

        def generate_batch(self, tasks):
            answers = []
            for task in tasks:
                self.protocols.append(task.protocol)
                cited = table_script(task)
                if task.protocol == "values":
                    cited = {"rows": [{k: v["value"] for k, v in row.items()} for row in cited["rows"]]}
                answers.append(GenerationResult(raw_text=json.dumps(cited)))
            return answers

    engine = Engine()
    jobs = [
        ("cited/doc", pdf, TABLE, {"protocol": "cited"}),
        ("values/doc", pdf, TABLE, {"protocol": "values"}),
    ]
    outcomes = run_batch(jobs, root=tmp_path / "run", engine=engine)
    assert sorted(engine.protocols) == ["cited", "values"]
    assert outcomes["cited/doc"].data["rows"] == outcomes["values/doc"].data["rows"]
    assert (tmp_path / "run" / "values" / "doc" / "result.json").exists()
    transforms = {
        field.transform for field in outcomes["values/doc"].field_evidence if "/rows/" in field.pointer
    }
    assert transforms == {"located_without_citation"}


def test_remote_runner_runs_variants_then_replays_them_without_a_model(tmp_path, monkeypatch):
    import sys

    import tools.extract_remote as runner

    rows = [("Code", "Qty", "Amount"), ("A1", "2", "$12.50"), ("B2", "1", "$5.00")]
    packet = tmp_path / "packet"
    packet.mkdir()
    for name in ("one", "two"):
        make_pdf(packet, [rows], name=f"{name}.pdf")
        (packet / f"{name}.schema.json").write_text(json.dumps(TABLE))
    cases = [{"id": name, "pdf": f"{name}.pdf", "schema": f"{name}.schema.json"} for name in ("one", "two")]
    (packet / "cases.json").write_text(json.dumps({"version": "inference-cases/1", "cases": cases}))

    class Engine:
        identity = {"backend": "scripted-runner", "version": "1"}
        telemetry = []

        def generate_batch(self, tasks):
            answers = []
            for task in tasks:
                cited = table_script(task)
                if task.protocol == "values":
                    cited = {"rows": [{k: v["value"] for k, v in row.items()} for row in cited["rows"]]}
                answers.append(GenerationResult(raw_text=json.dumps(cited)))
            return answers

    monkeypatch.setattr("docextract.vllm_engine.require_remote", lambda: "modal")
    real_engine_for = runner.engine_for
    monkeypatch.setattr(
        runner, "engine_for", lambda args: Engine() if args.engine == "vllm" else real_engine_for(args)
    )
    variants = json.dumps([{"name": "cited"}, {"name": "values", "protocol": "values"}])
    common = ["--cases", str(packet / "cases.json"), "--model-manifest", "unused.json", "--arm", "a"]
    common += ["--cache-root", str(tmp_path / "cache"), "--variants", variants, "--max-window-chars", "16000"]
    monkeypatch.setattr(sys, "argv", ["x", *common, "--output", str(tmp_path / "run"), "--engine", "vllm"])
    runner.main()
    (tmp_path / "run" / "runtime.json").write_text(json.dumps({"identity": Engine.identity}))
    for variant in ("cited", "values"):
        summary = json.loads((tmp_path / "run" / variant / "summary.json").read_text())
        assert [row["result"] for row in summary["cases"]] == ["complete", "complete"]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            *common,
            "--output",
            str(tmp_path / "replay"),
            "--engine",
            "replay",
            "--replay-from",
            str(tmp_path / "run"),
        ],
    )
    runner.main()
    for variant in ("cited", "values"):
        for case in ("one", "two"):
            first = json.loads((tmp_path / "run" / variant / case / "result.json").read_text())
            again = json.loads((tmp_path / "replay" / variant / case / "result.json").read_text())
            assert again["data"] == first["data"] and len(first["data"]["rows"]) == 2


def test_routing_asks_each_page_only_about_its_schema_groups(tmp_path):
    pdf = make_pdf(
        tmp_path,
        [
            ["SCHEDULE A medical dental expenses 500"],
            ["SCHEDULE B interest dividends 40"],
        ],
    )
    big = "x" * 300
    schema = {
        "type": "object",
        "properties": {
            "medical": {"type": "integer", "description": f"Schedule A medical and dental expenses {big}"},
            "interest": {"type": "integer", "description": f"Schedule B interest and dividends {big}"},
        },
    }
    asked = []

    def script(task):
        asked.append((task.units[0].page_index, tuple(task.prompt_schema["properties"])))
        text = " ".join(unit.text for unit in task.units)
        answer = {}
        if "medical" in task.prompt_schema["properties"] and "500" in text:
            answer["medical"] = {"value": 500, "source": task.units[0].alias}
        if "interest" in task.prompt_schema["properties"] and "40" in text:
            answer["interest"] = {"value": 40, "source": task.units[0].alias}
        return answer

    common = {"max_window_chars": 40, "overlap_lines": 0, "max_schema_chars": 400}
    plain = extract_cited(pdf, schema, workspace=tmp_path / "plain", backend=Scripted(script), **common)
    assert len(asked) == 4
    asked.clear()
    routed = extract_cited(
        pdf, schema, workspace=tmp_path / "routed", backend=Scripted(script), route_groups=True, **common
    )
    assert sorted(asked) == [(0, ("medical",)), (1, ("interest",))]
    assert routed.data == plain.data == {"medical": 500, "interest": 40}


def test_a_long_repetitive_schema_is_prompted_with_notes_but_validated_as_given(tmp_path):
    pdf = make_pdf(tmp_path, [["Line 3 total 42"]])
    boiler = "Whole-dollar amount as a JSON number; an amount in parentheses is negative; null if blank."
    schema = {
        "type": "object",
        "properties": {
            f"line_{n}": {"type": ["integer", "null"], "description": f"Line {n}. {boiler}"}
            for n in range(20)
        },
    }
    seen = []

    def script(task):
        seen.append(task.prompt_schema)
        return {"line_3": {"value": 42, "source": cell(task, "Line 3 total 42")}}

    result = extract_cited(
        pdf, schema, workspace=tmp_path / "run", backend=Scripted(script), factor_notes=True
    )
    assert "[N1]" in json.dumps(seen[0]) and "[N1]" in seen[0]["description"]
    assert result.data["line_3"] == 42 and result.validation.valid


def test_one_entity_described_in_two_sections_becomes_one_record(tmp_path):
    pdf = make_pdf(
        tmp_path,
        [
            ["Comparables", ("1", "2009 SEDAN DELUXE", "158 mi"), ("2", "2009 SEDAN BASIC", "40 mi")],
            ["Details", ("1", "VIN 1EXAMPLE0000000A1", ""), ("2", "VIN 1EXAMPLE0000000B2", "")],
            ["Repeated", ("7", "SAME ROW", "9 mi"), ("7", "SAME ROW", "9 mi")],
        ],
    )
    schema = {
        "type": "object",
        "properties": {
            "cars": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "row": {"type": "integer"},
                        "description": {"type": "string"},
                        "distance": {"type": "string"},
                        "vin": {"type": "string"},
                    },
                },
            }
        },
    }

    def script(task):
        # Report what the window shows, line by line, as a reader would.
        texts = [unit.text for unit in task.units]
        cars = []
        for index, text in enumerate(texts[:-1]):
            following = texts[index + 1]
            if text in ("1", "2") and following.startswith("2009"):
                cars.append({"row": int(text), "description": following, "distance": texts[index + 2]})
            elif text in ("1", "2") and following.startswith("VIN "):
                cars.append({"row": int(text), "vin": following[4:]})
            elif text == "7":
                cars.append({"row": 7, "description": following, "distance": texts[index + 2]})
        return {"cars": cars}

    common = {"max_window_chars": 90, "overlap_lines": 0, "protocol": "values"}
    plain = extract_cited(pdf, schema, workspace=tmp_path / "plain", backend=Scripted(script), **common)
    assert len(plain.data["cars"]) == 6
    merged = extract_cited(
        pdf, schema, workspace=tmp_path / "merged", backend=Scripted(script), merge_entities=True, **common
    )
    assert merged.data["cars"][:2] == [
        {"row": 1, "description": "2009 SEDAN DELUXE", "distance": "158 mi", "vin": "1EXAMPLE0000000A1"},
        {"row": 2, "description": "2009 SEDAN BASIC", "distance": "40 mi", "vin": "1EXAMPLE0000000B2"},
    ]
    assert merged.data["cars"][2:] == [{"row": 7, "description": "SAME ROW", "distance": "9 mi"}] * 2


def test_a_higher_output_limit_reuses_windows_answered_under_the_lower_one(tmp_path):
    from docextract.batch import run_batch

    rows = [("Code", "Qty", "Amount"), ("A1", "2", "$12.50"), ("B2", "1", "$5.00")]
    pdf = make_pdf(tmp_path, [rows], name="table.pdf")

    class Engine:
        def __init__(self, limit):
            self.identity = {"backend": "scripted-batch", "generation": {"max_new_tokens": limit}}
            self.batches = []

        def generate_batch(self, tasks):
            self.batches.append(len(tasks))
            return [GenerationResult(raw_text=json.dumps(table_script(task))) for task in tasks]

    old, new = Engine(16384), Engine(32768)
    cases = [("table", pdf, TABLE)]
    first = run_batch(cases, root=tmp_path / "batch", engine=old, max_window_chars=200, overlap_lines=1)
    again = run_batch(
        cases,
        root=tmp_path / "batch",
        engine=new,
        max_window_chars=200,
        overlap_lines=1,
        legacy_identities=(old.identity,),
    )
    assert old.batches and not new.batches  # every window answered from the store
    assert again["table"].data == first["table"].data
