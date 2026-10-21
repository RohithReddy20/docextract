import subprocess
import sys
from collections import deque

import pytest

from docextract.remote import UnsupportedProjection, native_template


def test_local_model_backend_fails_before_heavy_imports(tmp_path):
    code = """
import sys
from pathlib import Path
from docextract.remote import KaggleTransformersBackend, RemoteOnlyError
try:
    KaggleTransformersBackend('a', manifest=Path('never-read.json'))
except RemoteOnlyError:
    pass
else:
    raise AssertionError('Local model loading was permitted')
assert not any(name in sys.modules for name in ('torch', 'transformers', 'huggingface_hub'))
"""
    process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr


def test_native_projection_keeps_original_constraints_for_validation():
    from docextract.schema import Schema

    schema = {
        "type": "object",
        "required": ["id", "count"],
        "properties": {
            "id": {"type": ["string", "null"], "description": "Printed identifier"},
            "count": {"type": "integer", "minimum": 1},
            "rows": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"category": {"type": "string", "enum": ["A", "B"]}},
                },
            },
        },
    }
    assert native_template(schema) == {
        "id": "verbatim-string",
        "count": "integer",
        "rows": [{"category": ["A", "B"]}],
    }
    assert not Schema(schema).validate({"id": None, "count": 0}).valid
    assert schema["properties"]["count"]["minimum"] == 1


@pytest.mark.parametrize("node", [{"type": "null"}, {"type": "integer", "enum": [1, 2]}])
def test_native_projection_rejects_unsupported_shapes(node):
    with pytest.raises(UnsupportedProjection):
        native_template({"type": "object", "properties": {"value": node}})


def test_proposal_choices_cover_nested_schema_without_binding_to_values():
    from docextract.interpretation import InterpretationTask
    from docextract.remote import proposal_contract
    from docextract.schema import Schema

    original = {
        "type": "object",
        "properties": {
            "due/amount": {"type": "number"},
            "groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "members": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"name": {"type": "string"}}},
                        },
                    },
                },
            },
        },
    }
    response, catalog = proposal_contract(
        InterpretationTask(task_id="test", original_schema=original, units=())
    )
    assert catalog == {
        "root_fields": ["/due~1amount"],
        "item_fields_by_array_path": {"/groups": ["/name"], "/groups/*/members": ["/name"]},
    }
    assert Schema(response).support.supported
    assert native_template(response, verbatim_strings=False)["records"][0]["array_pointer"] == [
        "/groups",
        "/groups/*/members",
    ]
    assert original["properties"]["due/amount"] == {"type": "number"}


def test_vllm_streaming_maps_suffixed_request_ids_and_refuses_oversized_windows():
    from types import SimpleNamespace

    from docextract.cited import CitedTask
    from docextract.vllm_engine import VllmCitedEngine

    class FakeEngine:
        def __init__(self):
            self.running, self.steps = [], 0

        def add_request(self, request_id, prompt, params):
            internal = f"{request_id}-1a2b3c4d"
            self.running.append((internal, prompt["prompt_token_ids"]))
            self.most = max(getattr(self, "most", 0), len(self.running))
            return internal

        def has_unfinished_requests(self):
            return bool(self.running)

        def step(self):
            self.steps += 1
            internal, ids = self.running.pop(0)
            completion = SimpleNamespace(text='{"a": 1}', finish_reason="stop", token_ids=[1, 2, 3])
            yield SimpleNamespace(
                request_id=internal, finished=True, outputs=[completion], prompt_token_ids=ids
            )

    engine = object.__new__(VllmCitedEngine)
    engine.llm = SimpleNamespace(llm_engine=FakeEngine())
    engine.max_model_len, engine.max_new_tokens, engine.max_images = 10, 4, 0
    engine.ready, engine.requests, engine.telemetry = [], {}, []
    # One request in the engine at a time: the rest wait in the backlog (page images use memory).
    engine.backlog, engine.active, engine.in_flight = deque(), set(), 1
    engine.params = lambda task: None
    lengths = {"short": 3, "long": 8, "other": 2}
    engine.prompt_ids = lambda task: list(range(lengths[task.task_id]))
    tasks = [CitedTask(task_id=name, original_schema={"type": "object"}, units=()) for name in lengths]
    answers = engine.generate_batch(tasks)
    assert [answer.finish_reason for answer in answers] == ["stop", "input_limit", "stop"]
    assert answers[0].input_tokens == 3 and answers[0].output_tokens == 3
    assert answers[1].input_tokens == 8 and engine.llm.llm_engine.steps == 2
    assert engine.llm.llm_engine.most == 1 and not engine.backlog and not engine.active
    assert engine.requests == {} and [row["task_id"] for row in engine.telemetry] == ["short", "other"]


def test_prompt_schema_drops_only_redundant_text():
    import json

    from docextract.remote import prompt_schema

    schema = {
        "title": "Report",
        "type": "object",
        "$defs": {"Row": {"title": "Row", "type": "object", "properties": {"code": {"type": "string"}}}},
        "properties": {
            "lease_name": {
                "title": "Lease Name",
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            },
            "district": {"title": "FILE IN DUPLICATE WITH DISTRICT No", "type": "string"},
            "rows": {"anyOf": [{"$ref": "#/$defs/Row"}, {"type": "null"}]},
            "total": {"anyOf": [{"type": "number", "minimum": 0}, {"type": "null"}], "default": 0},
        },
    }
    compact = json.loads(prompt_schema(schema))
    assert compact["title"] == "Report" and "title" not in compact["$defs"]["Row"]
    assert compact["properties"]["lease_name"] == {"type": ["string", "null"]}
    assert compact["properties"]["district"]["title"] == "FILE IN DUPLICATE WITH DISTRICT No"
    assert compact["properties"]["rows"] == schema["properties"]["rows"]
    assert compact["properties"]["total"] == {"type": ["number", "null"], "minimum": 0, "default": 0}
    assert ", " not in prompt_schema(schema)


def test_vllm_sends_page_images_with_a_text_prompt(tmp_path):
    from types import SimpleNamespace

    from PIL import Image

    from docextract.cited import CitedTask
    from docextract.vllm_engine import VllmCitedEngine

    image_path = tmp_path / "p0000.jpg"
    Image.new("RGB", (320, 400), "white").save(image_path)
    requests = []

    class FakeEngine:
        def add_request(self, request_id, prompt, params):
            requests.append(prompt)
            return request_id

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, **options):
            assert tokenize is False
            return "rendered:" + messages[1]["content"][-1]["text"]

        def __call__(self, text):
            return {"input_ids": list(range(len(text)))}

    engine = object.__new__(VllmCitedEngine)
    engine.llm = SimpleNamespace(llm_engine=FakeEngine())
    engine.tokenizer, engine.arm, engine.protocol = Tokenizer(), "a", "cited"
    engine.max_model_len, engine.max_new_tokens, engine.max_images = 1000, 100, 1
    engine.ready, engine.requests, engine.telemetry = [], {}, []
    engine.backlog, engine.active, engine.in_flight = deque(), set(), 8
    engine.params = lambda task: None
    task = CitedTask(
        task_id="t1", original_schema={"type": "object"}, units=(), page_images=(str(image_path),)
    )
    engine.submit([task])
    (prompt,) = requests
    assert prompt["prompt"].startswith("rendered:Document:")
    assert [image.size for image in prompt["multi_modal_data"]["image"]] == [(320, 400)]
    engine.max_model_len = 150  # 320*400/1024 = 125 image tokens alone exceed the room left
    engine.submit([task.model_copy(update={"task_id": "t2"})])
    assert engine.ready[0][1].finish_reason == "input_limit" and len(requests) == 1
    # More images than the engine was started for: refused before vLLM would raise.
    engine.max_model_len, engine.ready = 1000, []
    engine.submit([task.model_copy(update={"task_id": "t3", "page_images": (str(image_path),) * 2})])
    assert engine.ready[0][1].finish_reason == "input_limit" and len(requests) == 1


def test_models_run_only_on_kaggle_modal_or_colab(monkeypatch):
    from docextract.remote import RemoteOnlyError
    from docextract.vllm_engine import require_remote

    for name in ("KAGGLE_KERNEL_RUN_TYPE", "MODAL_TASK_ID", "COLAB_RELEASE_TAG"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RemoteOnlyError):
        require_remote()
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release")  # without Colab's /content folder: not a Colab VM
    with pytest.raises(RemoteOnlyError):
        require_remote()
    monkeypatch.setattr("docextract.vllm_engine.Path.is_dir", lambda self: str(self) == "/content")
    assert require_remote() == "colab"
