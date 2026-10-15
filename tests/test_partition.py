import json

from reportlab.pdfgen.canvas import Canvas

from docextract.cited import extract_cited
from docextract.models import GenerationResult
from docextract.partition import inline, leaf_pointers, partition


def big_schema():
    note = "Detailed instruction text. " * 12
    boxes = {f"box_{i}": {"type": ["string", "null"], "description": note} for i in range(12)}
    return {
        "type": "object",
        "$defs": {"Box": {"type": ["number", "null"], "description": note}},
        "properties": {
            "taxpayer": {"type": "string", "description": note},
            "form": {"type": "object", "properties": {**boxes, "total": {"$ref": "#/$defs/Box"}}},
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}, "amount": {"$ref": "#/$defs/Box"}},
                },
            },
        },
    }


def test_partition_covers_every_field_without_splitting_arrays():
    schema = big_schema()
    parts = partition(schema, 2500)
    assert parts and len(parts) > 2
    assert all(
        len(json.dumps(part, separators=(",", ":"))) <= 2500
        for part in parts
        if "lines" not in part["properties"]
    )
    covered = set().union(*(leaf_pointers(part) for part in parts))
    assert covered == leaf_pointers(inline(schema))
    assert sum("lines" in part["properties"] for part in parts) == 1
    assert partition(schema, 10**6) is None


def test_partitioned_extraction_merges_groups_into_one_document(tmp_path):
    pdf = tmp_path / "form.pdf"
    canvas = Canvas(str(pdf), invariant=1)
    for index, text in enumerate(["Name: Ada", "Box 3: alpha", "Total: 12.5"]):
        canvas.drawString(48, 740 - index * 24, text)
    canvas.save()
    schema = big_schema()

    class Backend:
        identity = {"backend": "scripted", "version": "1"}
        calls = 0

        def generate_cited(self, task):
            self.calls += 1
            units = {unit.text.split(":")[0]: unit.alias for unit in task.units}
            answer = {}
            fields = task.prompt_schema["properties"]
            if "taxpayer" in fields:
                answer["taxpayer"] = {"value": "Ada", "source": units["Name"]}
            form = fields.get("form", {}).get("properties", {})
            if form:
                answer["form"] = {}
                if "box_3" in form:
                    answer["form"]["box_3"] = {"value": "alpha", "source": units["Box 3"]}
                if "total" in form:
                    answer["form"]["total"] = {"value": 12.5, "source": units["Total"]}
            return GenerationResult(raw_text=json.dumps(answer))

    backend = Backend()
    result = extract_cited(pdf, schema, workspace=tmp_path / "run", backend=backend, max_schema_chars=2500)
    assert backend.calls > 2
    assert result.data["taxpayer"] == "Ada" and result.data["form"]["box_3"] == "alpha"
    assert result.data["form"]["total"] == 12.5 and result.data["form"]["box_0"] is None
    assert result.validation.valid


def test_cited_output_schema_accepts_cited_answers_and_rejects_invented_keys():
    from jsonschema import Draft202012Validator

    from docextract.cited import cited_output_schema

    schema = big_schema()
    schema["properties"]["status"] = {"type": ["string", "null"], "enum": ["open", "closed"]}
    constraint = cited_output_schema(schema)
    validator = Draft202012Validator(constraint)
    answer = {
        "taxpayer": {"value": "Ada", "source": "s1"},
        "status": {"value": None, "source": "s2"},
        "form": {"total": {"value": 12.5, "source": ["s3", "s4"]}},
        "lines": [{"code": {"value": "A", "source": "s5"}}],
    }
    assert not list(validator.iter_errors(answer))
    for bad in (
        {"taxpayer": "Ada"},
        {"invented": {"value": 1, "source": "s1"}},
        {"status": {"value": "pending", "source": "s1"}},
    ):
        assert list(validator.iter_errors(bad))


def test_repeated_description_sentences_become_numbered_notes():
    from docextract.partition import factor_notes

    boiler = "Whole-dollar amount as a JSON number; null if the entry box is blank."
    schema = {
        "type": "object",
        "description": "A tax form.",
        "properties": {
            f"line_{n}": {"type": ["number", "null"], "description": f"Line {n}. {boiler}"} for n in range(10)
        },
    }
    factored = factor_notes(schema)
    assert factored["properties"]["line_3"]["description"] == "Line 3. [N1] [N2]"
    assert factored["description"].startswith("A tax form. Notes cited as [N#]")
    assert "[N1] Whole-dollar amount as a JSON number;" in factored["description"]
    assert {key: value["type"] for key, value in factored["properties"].items()} == {
        key: value["type"] for key, value in schema["properties"].items()
    }
    assert factor_notes({"type": "object", "properties": {"a": {"description": "x"}}}) == {
        "type": "object",
        "properties": {"a": {"description": "x"}},
    }
