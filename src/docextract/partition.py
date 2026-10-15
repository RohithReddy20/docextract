"""Split an oversized schema into field groups that fit a prompt, without dropping any field.

Local references are inlined so each group is self-contained. Root properties are packed in order;
an object property that alone exceeds the budget is split by its own properties. Arrays and scalar
leaves are never split, so a record's fields are always requested together. The union of the groups
covers every property of the original schema; constraints are still validated on the original.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from docextract.schema import parts


def inline(schema: dict) -> dict:
    """Replace local $defs references with their targets; recursive references are unsupported."""

    def resolve(node, trail=()):
        if isinstance(node, list):
            return [resolve(item, trail) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            reference = node["$ref"]
            if reference in trail or not reference.startswith("#/$defs/"):
                raise ValueError(f"Cannot inline reference {reference}")
            target = schema
            for part in parts(reference[1:]):
                target = target[part]
            merged = {**deepcopy(target), **{k: v for k, v in node.items() if k != "$ref"}}
            return resolve(merged, (*trail, reference))
        return {key: resolve(value, trail) for key, value in node.items() if key not in ("$defs", "$schema")}

    return resolve(schema)


def _is_object(node: dict) -> bool:
    types = node.get("type")
    return "object" in (types if isinstance(types, list) else [types]) and bool(node.get("properties"))


def _size(node: dict) -> int:
    return len(json.dumps(node, ensure_ascii=False, separators=(",", ":")))


def _split(node: dict, budget: int) -> list[dict]:
    shell = {key: value for key, value in node.items() if key not in ("properties", "required")}
    required = set(node.get("required", ()))
    groups: list[dict] = []
    current: dict = {}

    def sized(properties):
        return _size({**shell, "properties": properties})

    for key, child in node["properties"].items():
        if sized({key: child}) > budget and _is_object(child):
            if current:
                groups.append(current)
                current = {}
            overhead = sized({key: {}})
            groups.extend({key: part} for part in _split(child, max(1, budget - overhead)))
            continue
        if current and sized({**current, key: child}) > budget:
            groups.append(current)
            current = {}
        current[key] = child
    if current:
        groups.append(current)
    return [
        {
            **shell,
            "properties": group,
            **({"required": [k for k in group if k in required]} if required else {}),
        }
        for group in groups
    ]


def partition(schema: dict, max_chars: int) -> list[dict] | None:
    """None when the original schema already fits; otherwise self-contained field-group schemas."""
    if _size(schema) <= max_chars:
        return None
    flat = inline(schema)
    if not _is_object(flat):
        return None
    return _split(flat, max_chars)


def leaf_pointers(node: dict, prefix: str = "") -> set[str]:
    """Property paths covered by a (partial) inlined schema, used to check that groups cover all."""
    covered = set()
    for key, child in node.get("properties", {}).items():
        pointer = f"{prefix}/{key}"
        if _is_object(child):
            covered |= leaf_pointers(child, pointer)
        else:
            covered.add(pointer)
    return covered


def factor_notes(schema: dict, min_count: int = 8, min_length: int = 30) -> dict:
    """Write a description sentence that many fields repeat once, as a numbered note.

    Large generated schemas repeat the same guidance in hundreds of field descriptions. Each sentence
    (or ';'-separated clause) used at least `min_count` times is defined once in the root description
    as `[N1] ...` and cited as `[N1]` where it appeared. Structure and constraints are untouched; only
    description text changes, and the schema used for validation is always the original.
    """
    pieces: dict[str, int] = {}

    def split(text: str) -> list[str]:
        return [part for part in re.split(r"(?<=[.;])\s+", text) if part]

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    for part in split(value):
                        pieces[part] = pieces.get(part, 0) + 1
                else:
                    collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(schema)
    common = [
        part
        for part, count in sorted(pieces.items(), key=lambda kv: -kv[1])
        if count >= min_count and len(part) >= min_length
    ]
    if not common:
        return schema
    codes = {part: f"[N{number}]" for number, part in enumerate(common, start=1)}

    def rewrite(node):
        if isinstance(node, dict):
            return {
                key: " ".join(codes.get(part, part) for part in split(value))
                if key == "description" and isinstance(value, str)
                else rewrite(value)
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [rewrite(item) for item in node]
        return node

    factored = rewrite(schema)
    notes = "Notes cited as [N#] in field descriptions: " + " ".join(
        f"{code} {part}" for part, code in codes.items()
    )
    root = factored.get("description")
    factored["description"] = f"{root} {notes}" if isinstance(root, str) else notes
    return factored
