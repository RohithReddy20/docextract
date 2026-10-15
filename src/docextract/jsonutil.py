"""Reject JSON corruption and numeric changes before values enter saved contracts."""

import json
import math
import re
from decimal import Decimal, DecimalException
from typing import Any

FENCE = re.compile(r"\A\s*```(?:json)?[ \t]*\n(.*)\n[ \t]*```\s*\Z", re.DOTALL)


def strict_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("Non-finite JSON number")
        try:
            faithful = Decimal(text) == Decimal(str(value))
        except DecimalException as error:
            raise ValueError("Unsupported JSON decimal magnitude") from error
        if not faithful:
            raise ValueError(
                "JSON decimal cannot round-trip faithfully in the supported number representation"
            )
        return value

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid, parse_float=number)


def model_json(text: str) -> Any:
    """Strict JSON after removing one complete Markdown fence; truncated output is never repaired."""
    match = FENCE.match(text)
    return strict_json(match.group(1) if match else text)
