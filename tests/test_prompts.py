"""The prompts of requests whose answers may be stored.

A stored answer is found by its request's identity, which names the prompt version
(`docextract.vllm_engine.PROMPT_VERSION`) rather than hashing the prompt code. These digests pin the
prompts: a change to an existing prompt must change PROMPT_VERSION (stored answers would otherwise be
reused for a prompt they were not generated from) and then these digests.
"""

import hashlib
import json

from docextract.cited import CitedTask
from docextract.interpretation import EvidenceUnit
from docextract.remote import cited_messages

SCHEMA = {
    "type": "object",
    "properties": {
        "invoice": {"type": "string", "description": "Invoice number."},
        "rows": {
            "anyOf": [{"type": "array", "items": {"$ref": "#/$defs/Row"}}, {"type": "null"}],
        },
    },
    "$defs": {
        "Row": {"type": "object", "properties": {"code": {"type": "string"}, "qty": {"type": "number"}}}
    },
}
UNITS = tuple(
    EvidenceUnit(
        alias=f"u{index}",
        atom_id=f"a{index}",
        line_id=f"l{index}",
        page_index=0,
        text=text,
        bbox=(0.1, 0.1 * index, 0.9, 0.1 * index + 0.05),
        offset_atom_ids=(),
    )
    for index, text in enumerate(["Invoice A-17", "Code Qty", "X1 2"], start=1)
)


def digest(task: CitedTask, arm: str, protocol: str) -> str:
    messages, options = cited_messages(task, arm, protocol)
    return hashlib.sha256(json.dumps([messages, options], sort_keys=True).encode()).hexdigest()[:16]


def test_prompts_are_unchanged():
    whole = CitedTask(
        task_id="t",
        original_schema=SCHEMA,
        units=(),
        page_images=("p1.png", "p2.png"),
        protocol="oneshot",
        page_texts=("Invoice A-17", "Code Qty\nX1 2"),
    )
    page = whole.model_copy(
        update={
            "page_images": ("p2.png",),
            "page_texts": ("Code Qty\nX1 2",),
            "first_page": 2,
            "page_count": 2,
        }
    )
    edges = whole.model_copy(update={"page_numbers": (1, 2), "page_count": 9})
    window = CitedTask(task_id="w", original_schema=SCHEMA, units=UNITS, protocol="values")
    scanned = window.model_copy(update={"page_images": ("p1.jpg",)})
    found = {
        "oneshot": digest(whole, "c", "oneshot"),
        "oneshot page": digest(page, "c", "oneshot"),
        "oneshot pages": digest(edges, "c", "oneshot"),
        "values": digest(window, "c", "values"),
        "values with images": digest(scanned, "c", "values"),
        "cited": digest(window, "c", "cited"),
    }
    # The prompts full-1's stored answers were generated from (checked at its commit, df1a16a).
    assert found == {
        "oneshot": "6956c89c5a5ee81f",
        "oneshot page": "d2b48aac3b699999",
        "oneshot pages": "c113611a8f721af2",
        "values": "9489cc25ee8ee6dc",
        "values with images": "1ba205f386c14f40",
        "cited": "2e65e6c4b6f667f7",
    }, json.dumps(found, indent=1)
