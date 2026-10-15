import pytest

from docextract.schema import Schema, UnsupportedSchema


@pytest.mark.parametrize(
    "node",
    [
        {"type": ["string", "integer"]},
        {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        {"type": "object", "patternProperties": {".*": {"type": "string"}}},
        {"type": "object", "additionalProperties": {"type": "string"}},
        {"$ref": "https://example.invalid/schema.json"},
        {"type": "array", "prefixItems": [{"type": "string"}]},
        {"type": "string", "format": "custom-unregistered-format"},
        {"type": "integer", "if": {"minimum": 10}, "then": {"maximum": 20}},
    ],
)
def test_unsupported_schema_keywords_are_rejected_before_work(node):
    schema = Schema({"type": "object", "properties": {"value": node}})
    assert not schema.support.supported
    assert schema.support.issues
    with pytest.raises(UnsupportedSchema):
        schema.require_supported()


def test_recursive_local_refs_are_rejected():
    schema = Schema(
        {
            "type": "object",
            "properties": {"child": {"$ref": "#/$defs/Self"}},
            "$defs": {"Self": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Self"}}}},
        }
    )
    assert not schema.support.supported
    assert "recursive" in schema.support.issues[0]


def test_nullable_local_refs_and_constraints_preserve_original_schema():
    original = {
        "type": "object",
        "required": ["x"],
        "additionalProperties": False,
        "properties": {"x": {"$ref": "#/$defs/Value"}},
        "$defs": {"Value": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}], "default": None}},
    }
    schema = Schema(original)
    assert schema.support.supported
    assert schema.node("/x")["type"] == ["integer", "null"]
    assert schema.validate({"x": None}).valid
    assert not schema.validate({"x": 0}).valid
    assert not schema.validate({}).valid
    assert original == schema.original


def test_original_constraints_and_default_annotations():
    schema = Schema(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "code": {"type": "string", "enum": ["001"], "default": "001"},
                "date": {"type": "string", "format": "date"},
            },
        }
    )
    data = {}
    assert schema.validate(data).valid
    assert data == {}
    assert not schema.validate({"code": "1"}).valid
    assert not schema.validate({"date": "not-a-date"}).valid
    assert not schema.validate({"extra": 1}).valid


@pytest.mark.parametrize(
    "dialect",
    [None, "http://json-schema.org/draft-07/schema#", "https://json-schema.org/draft/2020-12/schema"],
)
def test_declared_dialect_preserves_constraints_and_missing_defaults(dialect):
    from jsonschema import Draft7Validator, Draft202012Validator

    original = {
        "type": "object",
        "required": ["count"],
        "properties": {
            "count": {"type": "integer", "exclusiveMinimum": 0},
            "note": {"type": ["string", "null"], "default": None},
        },
    }
    if dialect:
        original["$schema"] = dialect
    schema = Schema(original)
    assert schema.validator_class is (
        Draft7Validator if dialect and "draft-07" in dialect else Draft202012Validator
    )
    assert schema.support.supported and schema.original == original
    data = {"count": 1}
    assert schema.validate(data).valid and data == {"count": 1}
    assert schema.validate({"count": 1, "note": None}).valid
    assert not schema.validate({"count": 0}).valid
    assert not schema.validate({}).valid


def test_unknown_or_nested_dialect_cannot_silently_use_default_validator():
    schema = Schema({"$schema": "https://example.invalid/schema", "type": "object"})
    with pytest.raises(UnsupportedSchema):
        schema.validate({})
    nested = Schema(
        {
            "type": "object",
            "properties": {"value": {"$schema": "http://json-schema.org/draft-07/schema#", "type": "string"}},
        }
    )
    assert not nested.support.supported
    with pytest.raises(UnsupportedSchema):
        Schema(False)


def test_unsupported_schema_fails_before_parsing_or_backend_work(tmp_path):
    from docextract.engine import extract
    from docextract.planning import BindingPlan

    class NoWork:
        def parse(self, document):
            pytest.fail("parser should not run")

        def generate(self, task):
            pytest.fail("backend should not run")

    with pytest.raises(UnsupportedSchema):
        extract(
            tmp_path / "not-a-file.pdf",
            {"type": "object", "allOf": [{}]},
            workspace=tmp_path,
            plan=BindingPlan(),
            parser=NoWork(),
            backend=NoWork(),
        )
