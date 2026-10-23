"""Image-first extraction: the whole document in one request, evidence located afterwards.

The model sees every page as an image, the PDF's own text layer and the full schema, and answers with
one JSON object (`extract_oneshot`). The answer is typed to the schema (`conform`) and each printed value
is then located in the parsed page text (`locate_evidence`), so values keep field-level evidence without
the model having to cite anything. On ExtractBench samples this beat the windowed cited pipeline on
short, medium and long documents (docs/model-trials.md). A table too long for one answer is read again
one page at a time, and group labels of rows continued across a page break are repaired
(`carry_forward`); the windowed pipeline remains for schemas too large for one request.

Answers to batches of consecutive pages can be merged in page order (`merge_answers`): a scalar takes
the first value any batch found, objects merge key by key, and lists are concatenated, except that a
record repeated verbatim from an earlier batch (a header or a row printed again on the next page) is
dropped. Records repeated within one batch are kept: a page can list identical rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

THINK = re.compile(r"^\s*<think>.*?(?:</think>\s*|\Z)", re.DOTALL)

# The request text of ExtractBench's one-shot baseline for self-hosted models (run-llama/ExtractBench
# at 0e76479, Apache-2.0: direct_model_utils.DEFAULT_SYSTEM_PROMPT and DEFAULT_USER_INSTRUCTION, and
# vllm_extract._build_user_content), which the measurements start from.
SYSTEM_PROMPT = (
    "You are extracting structured data from a document according to the "
    "provided JSON schema. Return only the JSON that matches the schema. Use "
    "null for fields not present in the document. When the schema includes a "
    "list field, populate every relevant row visible in the document - do not "
    "return an empty list when rows are present."
)
USER_INSTRUCTION = (
    "Extract every field from the attached document according to the schema. "
    "Return JSON only. Use null for fields not present in the document. "
    "Whenever the schema declares a list field, enumerate every row visible in "
    "the document - do not collapse rows or return an empty list when rows are "
    "present."
)


def prepare_schema(schema: dict) -> dict:
    """The schema as the model reads it: LlamaExtract's `repeated_structure` entries become ordinary
    properties and every object is closed (`additionalProperties: false`)."""
    out = dict(schema)
    repeated = out.pop("repeated_structure", None)
    if isinstance(repeated, dict):
        out["properties"] = {**repeated, **(out.get("properties") or {})}

    def close(node):
        if isinstance(node, dict):
            node = dict(node)
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
            for key, value in node.items():
                if key in ("properties", "$defs", "definitions") and isinstance(value, dict):
                    node[key] = {name: close(child) for name, child in value.items()}
                elif key in ("items", "anyOf", "oneOf", "allOf"):
                    node[key] = close(value)
            return node
        if isinstance(node, list):
            return [close(item) for item in node]
        return node

    return close(out)


def user_text(
    schema: dict,
    page_texts: list[str] | None = None,
    *,
    first_page: int = 1,
    page_count: int | None = None,
    pages: list[int] | None = None,
    after: str | None = None,
) -> str:
    """The request text: the prepared schema, the instructions and, when given, the PDF's text layer.
    A request for part of a longer document (`page_count`) says which pages it shows: the one page
    `first_page`, or the listed `pages`. A request reading on a list begun on earlier pages gives the
    last record read so far (`after`)."""
    text = (
        "Extract every field from the attached document page image(s) according to the JSON schema below.\n\n"
        "JSON schema:\n"
        f"{json.dumps(prepare_schema(schema), indent=2)}\n\n"
        "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
        f"{USER_INSTRUCTION}"
    )
    if page_count and pages:
        shown = ", ".join(map(str, pages[:-1])) + f" and {pages[-1]}" if len(pages) > 1 else str(pages[0])
        text += (
            f"\n\nThe images are pages {shown} of a {page_count}-page document; the other pages are not "
            "shown. Extract what these pages show; use null (or an empty list) for anything not on them."
        )
    elif page_count:
        text += (
            f"\n\nThe image is page {first_page} of a {page_count}-page document. Extract what this page "
            "shows; use null (or an empty list) for anything not on it."
        )
    if after:
        text += (
            "\n\nThese pages continue a list begun on earlier pages, which were read up to and including "
            f"this record:\n{after}\nList only the records printed after it, in the order printed. A "
            "record at the top of these pages that continues its group (a heading printed on an earlier "
            "page) keeps that record's group fields."
        )
    if page_texts:
        numbers = pages or range(first_page, first_page + len(page_texts))
        pairs = zip(numbers, page_texts, strict=True)
        layer = "\n\n".join(f"--- page {number} ---\n{page.strip()}" for number, page in pairs)
        text += f"\n\nText layer of the document (may be incomplete; trust the images):\n{layer}"
    return text


def split_schema(schema: dict, names: list[str]) -> dict:
    """The schema restricted to the top-level properties `names` (definitions kept)."""
    return {
        **schema,
        "properties": {key: value for key, value in schema["properties"].items() if key in names},
        "required": [key for key in schema.get("required", []) if key in names],
    }


def cut_off_table(partial: dict, schema: dict, raw: str = "") -> str | None:
    """The top-level list of records an answer was writing when the output limit cut it off: the last
    key of the salvaged answer (or, when nothing was salvaged, the list the raw answer opens), when the
    schema declares it an array of objects."""
    opened = re.match(r'\s*\{\s*"([^"]+)"\s*:\s*\[', THINK.sub("", raw, count=1))
    if not partial and not opened:
        return None
    name = list(partial)[-1] if partial else opened.group(1)
    return name if name in record_lists(schema) else None


def record_lists(schema: dict) -> list[str]:
    """The top-level properties the schema declares lists of records (arrays of objects), null allowed
    or not."""
    names = []
    for name, node in (prepare_schema(schema).get("properties") or {}).items():
        array = _array(node, schema) if isinstance(node, dict) else None
        items = _resolve((array or {}).get("items") or {}, schema)
        if array is not None and ("object" in _types(items) or "properties" in items):
            names.append(name)
    return names


def _array(node: dict, schema: dict) -> dict | None:
    """The array a property declares: the property itself, or the array it may be instead of null
    (`anyOf`/`oneOf` options, as generated for an optional list)."""
    node = _resolve(node, schema)
    for option in [node, *(node.get("anyOf") or node.get("oneOf") or [])]:
        option = _resolve(option, schema) if isinstance(option, dict) else {}
        if "array" in _types(option):
            return option
    return None


# Pages at each end of a document too long for one request that are shown for its fields outside
# tables (titles, parties, dates and totals are usually printed there).
EDGE_PAGES = 2


def edge_pages(count: int) -> list[int]:
    """The page numbers of the first and last EDGE_PAGES pages of a `count`-page document."""
    return sorted({*range(1, min(EDGE_PAGES, count) + 1), *range(max(1, count - EDGE_PAGES + 1), count + 1)})


def _types(node: dict) -> set[str]:
    kind = node.get("type")
    return set(kind) if isinstance(kind, list) else {kind} if kind else set()


def _resolve(node: dict, schema: dict) -> dict:
    """A local `$ref` ("#/$defs/Name") followed to its definition."""
    reference = node.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return node
    target: Any = schema
    for part in reference[2:].split("/"):
        target = target.get(part, {}) if isinstance(target, dict) else {}
    return target if isinstance(target, dict) else {}


def page_dpi(pages: int) -> int:
    """Short documents are read at 200 DPI; longer ones at 150 so the whole document fits one request."""
    return 200 if pages <= 10 else 150


def _closing(text: str) -> tuple[bool, list[str]]:
    """Whether `text` ends inside a string, and the brackets still open at its end."""
    in_string = escaped = False
    stack: list[str] = []
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]" and stack:
            stack.pop()
    return in_string, stack


def parse_answer(raw: str) -> dict:
    """The JSON object in a model answer. A reply cut off at the output limit keeps everything it
    completed; only the half-written last value is dropped. Not a JSON object: an empty answer."""
    text = THINK.sub("", raw, count=1).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.removesuffix("```").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
        # Cut back to the end of the last complete element and close what is still open.
        cuts = [match.start() for match in re.finditer(r"[,\]}]", text)]
        for cut in reversed(cuts[-512:]):
            head = text[: cut + 1] if text[cut] in "]}" else text[:cut]
            in_string, stack = _closing(head)
            if in_string:
                continue
            candidate = head.rstrip().rstrip(",") + "".join(reversed(stack))
            try:
                data = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
    return data if isinstance(data, dict) else {}


def _empty(value: Any) -> bool:
    return value is None or value == [] or value == {}


def _printed(value: Any) -> str | None:
    """A scalar value as it would be printed, when long enough to recognize in a line of text."""
    if value is None or isinstance(value, (bool, dict, list)):
        return None
    text = str(value).strip()
    return text if len(text) >= 2 else None


def _group_fields(pages: list[list], texts: list[str]) -> set[str]:
    """Fields naming a group of consecutive records, such as an identifier printed once per group or a
    section heading: the same text on consecutive records (at least three times, on a fifth of the
    pairs), printed in the document for most of its values, and printed less often on a page than
    records use it (a value read from a code, or printed on every row, is not a group label)."""
    records = [record for page in pages for record in page if isinstance(record, dict)]
    pairs: dict[str, list[bool]] = {}
    for before, after in zip(records, records[1:], strict=False):
        for key, value in after.items():
            if isinstance(value, str) and value.strip() and isinstance(before.get(key), str):
                pairs.setdefault(key, []).append(value == before[key])
    candidates = {key for key, same in pairs.items() if sum(same) >= 3 and sum(same) >= len(same) / 5}
    everything = "\n".join(texts)
    group = set()
    for key in candidates:
        values = {text for record in records if (text := _printed(record.get(key)))}
        if not values or sum(1 for text in values if text in everything) < 0.8 * len(values):
            continue
        used = printed = 0
        for page, text in zip(pages, texts, strict=False):
            counts: dict[str, int] = {}
            for record in page:
                if isinstance(record, dict) and (value := _printed(record.get(key))):
                    counts[value] = counts.get(value, 0) + 1
            lines = text.splitlines()
            for value, count in counts.items():
                used += count
                printed += sum(1 for line in lines if value in line)
        if used and printed < 0.75 * used:
            group.add(key)
    return group


def _forms(value: Any) -> tuple[str, ...]:
    """Ways a value may be printed: as written, and numbers also with thousands separators."""
    text = _printed(value)
    if text is None:
        return ()
    if isinstance(value, int):
        return (text, f"{value:,}")
    if isinstance(value, float):
        return (text, f"{value:,.2f}", f"{value:.2f}")
    return (text,)


def _record_line(record: dict, lines: list[str], start: int, group: set[str]) -> int | None:
    """The line (from `start`) holding most of a record's own printed values: None unless it holds at
    least two of them and half (a record printed over several lines has no line of its own)."""
    values = [forms for key, value in record.items() if key not in group and (forms := _forms(value))]
    best, most = None, max(1, (len(values) - 1) // 2)
    for index in range(start, len(lines)):
        count = sum(1 for forms in values if any(form in lines[index] for form in forms))
        if count > most:
            best, most = index, count
    return best


def nullable_fields(schema: dict, table: str) -> frozenset[str]:
    """Fields of a top-level list's records that may be null (typed null, or not required)."""
    prepared = prepare_schema(schema)
    array = _array((prepared.get("properties") or {}).get(table, {}), prepared) or {}
    items = _resolve(array.get("items") or {}, prepared)
    required = set(items.get("required") or [])
    return frozenset(
        key
        for key, node in (items.get("properties") or {}).items()
        if key not in required or "null" in _types(_resolve(node, prepared))
    )


def carry_forward(pages: list[list], texts: list[str], nullable: frozenset[str] = frozenset()) -> list[list]:
    """Records of a table read page by page, with group labels of continued rows repaired.

    Rows at the top of a page can continue a group begun on an earlier page (a subject's remaining
    visits, a check's further lines, a section running on). A model reading only that page labels
    them with the group printed below them, or leaves the label empty. A group's label is printed on
    its first row, so a row whose label is first printed below its own lines (those up to the next
    record) cannot belong to it: such a row takes the label of the previous page's last record. So
    does a row left unlabelled, when the schema does not allow the label to be null or when no group
    label of any kind is printed above it on its page (fields in `nullable` may be null). Repair stops
    for a field at the first row whose label is printed on its own lines or above them.
    """
    group = _group_fields(pages, texts)
    if not group:
        return pages
    labels = {
        text
        for page in pages
        for record in page
        if isinstance(record, dict)
        for key in group
        if (text := _printed(record.get(key)))
    }
    previous = None
    repaired = []
    for index, page in enumerate(pages):
        page = [dict(record) if isinstance(record, dict) else record for record in page]
        if previous is not None and index < len(texts):
            lines = texts[index].splitlines()
            # The first line of this page that prints a group label (a heading, a group's first row).
            heading = next(
                (i for i, other in enumerate(lines) if any(label in other for label in labels)), None
            )
            pending = {key for key in group if previous.get(key) not in (None, "")}
            records = [record for record in page if isinstance(record, dict)]
            line = _record_line(records[0], lines, 0, group) if records else None
            for position, record in enumerate(records):
                if not pending or line is None:
                    break
                following = (
                    _record_line(records[position + 1], lines, line, group)
                    if position + 1 < len(records)
                    else None
                )
                end = max(following if following is not None else len(lines), line + 1)
                for key in sorted(pending):
                    value, text = record.get(key), _printed(record.get(key))
                    first = next((i for i, other in enumerate(lines) if text and text in other), None)
                    if first is not None and first < end:
                        pending.discard(key)  # labelled on this record's lines or by a heading above
                    elif first is not None:
                        record[key] = previous[key]  # took a label printed below it
                    elif value is None and (key not in nullable or heading is None or heading > line):
                        # Unlabelled, and no group label is printed above it on this page: the group
                        # (a section, a subject) continues from the previous page.
                        record[key] = previous[key]
                line = following
        repaired.append(page)
        previous = next((record for record in reversed(page) if isinstance(record, dict)), previous)
    return repaired


# A whole-document answer can list part of a long table and stop (answers to two 13F filings listed 25
# and 32 of their 1,168 and 1,223 holdings). Lines printed like the list's own rows but on pages none of
# its rows come from, at least LAZY_RATIO times its rows and LAZY_LINES in all, have the list read
# again page by page. (On 36 benchmark documents answered in one request, the two filings showed over
# 1,000 such lines, 33 and 43 times their rows; complete lists showed at most 5.6 times.)
LAZY_RATIO = 10
LAZY_LINES = 50
NUMBER_TOKEN = re.compile(r"[-+(]?\$?\d[\d,]*(?:\.\d+)?%?\)?")


def _line_shape(line: str) -> tuple[str, int]:
    """The kinds of a line's tokens in order (n a number, c letters with digits, w a run of words) and
    how many numbers it prints."""
    kinds: list[str] = []
    for token in line.split():
        if NUMBER_TOKEN.fullmatch(token):
            kind = "n"
        elif re.search(r"\d", token) and re.search(r"[^\W\d_]", token):
            kind = "c"
        else:
            kind = "w"
        if not (kind == "w" and kinds and kinds[-1] == "w"):
            kinds.append(kind)
    return "".join(kinds), kinds.count("n")


def _scalars(record: dict) -> list:
    """A record's own values, those of objects inside it included (lists inside it are not)."""
    values = []
    for value in record.values():
        if isinstance(value, dict):
            values += _scalars(value)
        elif not isinstance(value, list):
            values.append(value)
    return values


class _PrintedLines:
    """The lines of a document's page texts, and which of them print a record's values."""

    def __init__(self, texts):
        from itertools import accumulate

        self.lines = [(page, line) for page, text in enumerate(texts) for line in text.splitlines()]
        self.text = "\n".join(line for _, line in self.lines)
        self.starts = list(accumulate((len(line) + 1 for _, line in self.lines[:-1]), initial=0))
        self.found: dict[tuple, set[int]] = {}

    def where(self, forms: tuple) -> set[int]:
        """The lines printing a value in any of its forms."""
        import bisect

        if forms not in self.found:
            self.found[forms] = set()
            for form in forms:
                at = self.text.find(form)
                while at != -1:
                    self.found[forms].add(bisect.bisect_right(self.starts, at) - 1)
                    at = self.text.find(form, at + 1)
        return self.found[forms]

    def own(self, records: list[dict]) -> list[set[int]]:
        """Each record's own lines: those printing two of its distinctive values (or the only one it
        has), printed values of three characters or more that at most two of the `records` use."""
        from collections import Counter

        printed = [
            {_forms(value) for value in _scalars(record) if len(_printed(value) or "") >= 3}
            for record in records
        ]
        uses = Counter(forms for values in printed for forms in values)
        own = []
        for values in printed:
            distinctive = [forms for forms in values if uses[forms] <= 2]
            hits = Counter(index for forms in distinctive for index in self.where(forms))
            own.append({index for index, count in hits.items() if count >= min(2, len(distinctive))})
        return own


def record_pages(records: list[dict], texts) -> list[set[int]]:
    """The pages (indexes into `texts`) each record is printed on, as `_PrintedLines.own` finds them."""
    printed = _PrintedLines(texts)
    return [{printed.lines[index][0] for index in own} for own in printed.own(records)]


def partial_lists(data: dict, schema: dict, texts: list[str] | tuple[str, ...]) -> list[str]:
    """Top-level lists of records an answer listed only part of, judged from the PDF's page `texts`.

    A line comes from a record when it prints two of the record's distinctive values (or the only one
    it has): printed values of three characters or more used by at most two records of the answer's
    lists. A list's own lines are those coming from its records; lines printed like them (the same
    kinds of tokens in the same order, as on three of its lines, with two numbers or more) that come
    from no record, on pages none of its records come from, are rows the answer left out.
    """
    from collections import Counter

    printed = _PrintedLines(texts)
    lines = printed.lines
    tables = [name for name in record_lists(schema) if isinstance(data.get(name), list)]
    records = [(name, record) for name in tables for record in data[name] if isinstance(record, dict)]
    if not lines or not records:
        return []
    own: dict[str, set[int]] = {name: set() for name in tables}
    for (name, _), lines_of in zip(records, printed.own([record for _, record in records]), strict=True):
        own[name].update(lines_of)
    explained = set().union(*own.values())
    shapes = [_line_shape(line) for _, line in lines]
    partial = []
    for name in tables:
        rows = sum(1 for record in data[name] if isinstance(record, dict))
        pages = {lines[index][0] for index in own[name]}
        seen = Counter(shapes[index][0] for index in own[name] if shapes[index][1] >= 2)
        alike = {shape for shape, count in seen.items() if count >= 3}
        left = [
            index
            for index, (page, _) in enumerate(lines)
            if page not in pages
            and index not in explained
            and shapes[index][0] in alike
            and shapes[index][1] >= 2
        ]
        if len(left) >= max(LAZY_LINES, LAZY_RATIO * rows) and len({lines[index][0] for index in left}) >= 2:
            partial.append(name)
    return partial


def merge_answers(parts: list[Any]) -> Any:
    """Merge answers to consecutive page batches, given in page order."""
    present = [part for part in parts if not _empty(part)]
    if not present:
        return parts[0] if parts else None
    if all(isinstance(part, dict) for part in present):
        keys = list(dict.fromkeys(key for part in present for key in part))
        return {key: merge_answers([part.get(key) for part in present]) for key in keys}
    if all(isinstance(part, list) for part in present):
        merged: list[Any] = []
        earlier: set[str] = set()
        for part in present:
            own = [json.dumps(item, sort_keys=True) for item in part]
            merged.extend(item for item, key in zip(part, own, strict=True) if key not in earlier)
            earlier.update(own)
        return merged
    return present[0]


# A boolean whose description says an unmarked box is false (not unknown).
TWO_STATE = re.compile(
    r"never null|two[- ]state|false (?:if|when|otherwise)\b[^.]{0,40}?"
    r"(?:blank|unmarked|unchecked|empty|not (?:marked|checked|ticked))|false otherwise|false if blank",
    re.IGNORECASE,
)


def _number(text: str) -> int | float | None:
    cleaned = text.strip().replace(",", "").replace("$", "")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        return None
    value = float(cleaned)
    return int(value) if value.is_integer() and "." not in cleaned else value


def _scalar(value: Any, node: dict, kind: str) -> Any:
    if isinstance(value, str) and not value.strip():
        value = None
    if value is None:
        if kind == "boolean" and TWO_STATE.search(node.get("description", "")):
            return False
        return None
    if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)
    if kind in ("integer", "number") and isinstance(value, str):
        number = _number(value)
        if number is None:
            return value
        return int(number) if kind == "integer" and float(number).is_integer() else number
    if (
        kind == "boolean"
        and isinstance(value, str)
        and value.strip().casefold() in ("true", "yes", "false", "no")
    ):
        return value.strip().casefold() in ("true", "yes")
    choices = node.get("enum")
    if isinstance(choices, list) and isinstance(value, str) and value not in choices:
        folded = [c for c in choices if isinstance(c, str) and c.casefold() == value.strip().casefold()]
        if len(folded) == 1:
            return folded[0]
    return value


def conform(data: Any, schema: dict) -> Any:
    """Give a model answer the schema's types, leniently; never drops or invents structure.

    Numbers for string fields become their printed form (41000.0 -> "41000"), numeric strings for
    number fields become numbers, "true"/"no" strings for booleans become booleans, enum values take the
    schema's spelling, and a missing value of a boolean the description defines as two-state ("false if
    blank", "never null") becomes false. A blank string is no value (null), and a property the answer
    leaves out is written as null: the benchmark's keys hold null for what a document does not show,
    and scored a left-out field as wrong (finished answers of a full run: 4 of 14 documents with
    left-out fields gained 4-18 points, none lost). Anything else that does not fit is left as the
    model gave it.
    """
    from docextract.schema import Schema

    try:
        compiled = Schema(schema)
    except Exception:
        return data

    def walk(value: Any, node: dict) -> Any:
        try:
            node = compiled.resolve(node)
            kind = compiled.kind(node)
        except Exception:
            return value
        if kind == "object" and isinstance(value, dict):
            properties = node.get("properties", {})
            out = {
                key: walk(item, properties[key]) if key in properties else item for key, item in value.items()
            }
            for key, child in properties.items():
                if key not in out:
                    out[key] = walk(None, child)
            return out
        if kind == "array" and isinstance(value, list) and isinstance(node.get("items"), dict):
            return [walk(item, node["items"]) for item in value]
        if kind in ("object", "array"):
            return value
        return _scalar(value, node, kind)

    return walk(data, schema)


def _leaves(data: Any, schema: dict, with_nodes: bool = False):
    """(JSON pointer, value, schema kind[, schema node]) of every scalar in an answer, in answer order."""
    from docextract.schema import Schema, escape

    compiled = Schema(schema)

    def walk(value, node, pointer):
        try:
            node = compiled.resolve(node)
            kind = compiled.kind(node)
        except Exception:
            kind, node = None, {}
        if isinstance(value, dict):
            properties = node.get("properties", {})
            for key, item in value.items():
                yield from walk(item, properties.get(key, {}), f"{pointer}/{escape(key)}")
        elif isinstance(value, list):
            items = node.get("items", {}) if isinstance(node.get("items"), dict) else {}
            for index, item in enumerate(value):
                yield from walk(item, items, f"{pointer}/{index}")
        else:
            kind = kind or ("number" if isinstance(value, (int, float)) else "string")
            yield (pointer, value, kind, node) if with_nodes else (pointer, value, kind)

    yield from walk(data, schema, "")


def field_path(pointer: str, data: Any) -> str:
    """ExtractBench's field path for a JSON pointer into `data` (a.b[2].c)."""
    from docextract.schema import parts

    path, current = "", data
    for part in parts(pointer):
        if isinstance(current, list):
            path += f"[{int(part)}]"
            current = current[int(part)]
        else:
            path += ("." if path else "") + part
            current = current[part]
    return path


# Characters after the previous located value searched first for the next one.
NEAR = 4000


class _Occurrences:
    """Every printed occurrence of a value in one document text, in text order, found once per
    distinct value; numbers come from a single pass over the text instead of one pass per value."""

    def __init__(self, text: str):
        self.text = text
        self.numbers: dict[Any, list[tuple[int, int, str]]] | None = None
        self.cache: dict[tuple, tuple[list, list[int]]] = {}

    def _number_index(self) -> dict[Any, list[tuple[int, int, str]]]:
        from docextract.cited import NUMBER, parse_number

        index: dict[Any, list[tuple[int, int, str]]] = {}
        for match in NUMBER.finditer(self.text):
            lexeme = match.group().strip()
            try:
                printed = parse_number(lexeme)
            except ValueError:
                continue
            index.setdefault(printed, []).append((match.start(), match.end(), lexeme))
            if lexeme.endswith("%"):  # a percentage may stand for its fraction
                index.setdefault(round(printed / 100, 12), []).append((match.start(), match.end(), lexeme))
        return index

    def get(self, value: Any, kind: str) -> tuple[list, list[int]]:
        """(occurrences as (start, end, transform), their start offsets)."""
        from docextract.cited import find_all

        key = (kind, type(value).__name__, value)
        if key not in self.cache:
            if kind in ("integer", "number") and isinstance(value, (int, float)):
                if self.numbers is None:
                    self.numbers = self._number_index()
                hits = self.numbers.get(value, []) + (
                    self.numbers.get(round(value, 12), []) if round(value, 12) != value else []
                )
                found = sorted(
                    (start, end, "verbatim" if lexeme == str(value) else "number_normalized")
                    for start, end, lexeme in hits
                )
            else:
                found = list(find_all(value, kind, self.text))
            self.cache[key] = (found, [start for start, _, _ in found])
        return self.cache[key]


def locate_values(data: Any, schema: dict, snapshot):
    """Where each printed value of an answer appears in the parsed page text (native text or OCR):
    (pointer, value, atom ids, printed text, transform) per located value, in answer order.

    Values are searched in document order from the previous located value, so repeated rows find their
    own lines; a value printed nowhere (or a check box) is not yielded rather than guessed. (Preferring
    the occurrence printed after the field's label was tried and located fewer values correctly.)
    """
    import bisect

    from docextract.cited import _atoms, _virtual, find_all
    from docextract.interpretation import evidence_units

    units = list(evidence_units(snapshot))
    if not units:
        return
    text, offsets, _ = _virtual(units)
    every = _Occurrences(text)
    cursor = 0
    for pointer, value, kind in _leaves(data, schema):
        if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
            continue
        numeric = kind in ("integer", "number") and isinstance(value, (int, float))
        # The next value is usually printed just after the previous one (a table's rows in order), so
        # the text right after the cursor is searched before the whole document: long tables stay fast.
        near = None if numeric else next(find_all(value, kind, text[cursor : cursor + NEAR]), None)
        if near is not None:
            start, end, transform = near[0] + cursor, near[1] + cursor, near[2]
        else:
            occurrences, starts = every.get(value, kind)
            if not occurrences:
                continue
            position = bisect.bisect_left(starts, cursor)
            start, end, transform = occurrences[position if position < len(occurrences) else 0]
        cursor = start
        yield pointer, value, _atoms(offsets, start, end), text[start:end], transform


def locate_evidence(data: Any, schema: dict, snapshot) -> list[dict]:
    """ExtractBench-style citations (field path, page, normalized box) for `locate_values`."""
    from docextract.geometry import envelope, normalized_coco

    atoms = snapshot.atom_index()
    pages = {page.page_index: page for page in snapshot.pages}
    citations = []
    for pointer, _, atom_ids, printed, transform in locate_values(data, schema, snapshot):
        by_page: dict[int, list] = {}
        for atom_id in atom_ids:
            atom = atoms[atom_id]
            by_page.setdefault(atom.page_index, []).append(atom)
        for page_index, located in by_page.items():
            page = pages[page_index]
            try:
                box = normalized_coco(envelope([atom.polygon for atom in located]), page.width, page.height)
            except ValueError:
                continue
            citations.append(
                {
                    "field_path": field_path(pointer, data),
                    "page": page_index + 1,
                    "bbox": box,
                    "reference_text": printed,
                    "source": located[0].source_channel,
                    "transform": transform,
                }
            )
    return citations


def render_page_images(document: Path, directory: Path, dpi: int) -> list[Path]:
    """Every page as a PNG at `dpi`, rendered once into `directory`."""
    from contextlib import closing

    import pypdfium2 as pdfium

    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    with closing(pdfium.PdfDocument(str(document))) as pdf:
        for index in range(len(pdf)):
            path = directory / f"p{index:04d}.png"
            if not path.exists():
                bitmap = pdf[index].render(scale=dpi / 72)
                bitmap.to_pil().convert("RGB").save(path, format="PNG")
            paths.append(path)
    return paths


def pdf_page_texts(document: Path) -> list[str]:
    """Each page's own PDF text in the order the PDF stores it (PDFium's text page), which keeps the
    columns of a multi-column page apart; text recognized by OCR and form-field values are not
    included, since the model reads those from the page image. Measured on ExtractBench samples
    (docs/model-trials.md) with this text; text regrouped into lines across the page width mixes the
    columns of newspaper-style listings and two-column layouts."""
    from contextlib import closing

    import pypdfium2 as pdfium

    with closing(pdfium.PdfDocument(str(document))) as pdf:
        texts = []
        for index in range(len(pdf)):
            with closing(pdf[index].get_textpage()) as text:
                texts.append(text.get_text_bounded())
        return texts


# Output allowed for one page's records: about two tokens per character of the page's own text were
# measured (JSON repeats every field name), so twice that leaves room and ends a runaway answer early.
PAGE_TOKENS_PER_CHAR = 4


def _recognized(page) -> bool:
    """Whether a parsed page's text was read by OCR (a scan without enough text of its own)."""
    return bool({"ocr_text", "sparse_native_text"} & set(page.quality_flags))


def recognized_texts(snapshot, texts: tuple[str, ...]) -> tuple[str, ...]:
    """The page `texts` with each scanned page's text replaced by its parsed lines (OCR, in reading
    order, with whatever text of its own the page has)."""
    return tuple(
        "\n".join(atom.text for atom in page.atoms if atom.kind == "line")
        if index < len(snapshot.pages) and _recognized(page := snapshot.pages[index])
        else text
        for index, text in enumerate(texts)
    )


def page_output_cap(text: str) -> int | None:
    """The output limit of a page request; None (the engine's) for a page with little or no text."""
    chars = len(text.strip())
    return max(1024, PAGE_TOKENS_PER_CHAR * chars) if chars >= 200 else None


def _aliases(identity: dict, backend) -> tuple[str, ...]:
    """The ids a request identified by `identity` had under the store's legacy engines (the same
    engine with a lower output limit), whose stored answers may stand for it."""
    from docextract.contracts import digest

    return tuple(digest({**identity, "backend": other}) for other in getattr(backend, "legacy", ()))


def beyond_limits(images: list, texts: tuple, identity: dict) -> bool:
    """Whether the engine named by `identity` would refuse a request showing `images` and `texts`
    whatever their resolution: more page images than it accepts, or text alone longer than its window
    (taking at most eight characters a token)."""
    limits = identity.get("generation") or {}
    pages, window = limits.get("max_images"), limits.get("max_model_len")
    return bool(pages and len(images) > pages) or bool(window and sum(map(len, texts)) > 8 * window)


def _ask_document(
    document, schema, snapshot, texts, store, dpi, backend, max_tokens=None, pages=None, ocr_text=False
):
    """One request for the whole document, or for the listed `pages` (numbers from 1), with output
    limited to `max_tokens` when given. A request too long for the model's window is retried with
    smaller images (two thirds of the resolution, down to 72 DPI); one the engine would refuse at any
    resolution is refused here, without building its prompt (a run crashed in native code right after
    building one of 466,296 tokens). Last, the pages with a text layer are shown by their text alone,
    except the first and last ones (`edge_pages`, where a form's boxed fields are printed), at the
    first resolution: a long contract then fits one request, whose model sees which of its tables the
    schema means (read one page at a time, clause lists and item numbers repeated in other sections
    became line items; with no page image at all, a purchase order's boxed fields were misread). With
    `ocr_text`, a scanned page's text is its OCR lines (`recognized_texts`); it keeps its image."""
    import hashlib

    from docextract.cited import CitedTask
    from docextract.contracts import digest
    from docextract.models import GenerationResult

    read = recognized_texts(snapshot, texts) if texts and ocr_text else texts
    text_only, first_dpi = False, dpi
    while True:
        images = render_page_images(document, store / snapshot.snapshot_id / f"dpi-{dpi}", dpi)
        count = len(images)
        numbers = pages or range(1, count + 1)
        own = tuple(texts[number - 1] for number in numbers) if texts else ()
        shown = tuple(read[number - 1] for number in numbers) if texts else ()
        if pages:
            images = [images[number - 1] for number in pages]
        if text_only:
            edges = edge_pages(count)
            images = [
                image
                for number, image, text in zip(numbers, images, own, strict=True)
                if number in edges or len(text.strip()) < 200
            ]
        identity = {
            "protocol": "oneshot/1",
            "schema": schema,
            "backend": backend.identity,
            "images": [hashlib.sha256(path.read_bytes()).hexdigest() for path in images],
            "texts": digest(shown),
            **({"max_tokens": max_tokens} if max_tokens else {}),
            **({"pages": list(pages)} if pages else {}),
        }
        task = CitedTask(
            task_id=digest(identity),
            original_schema=schema,
            units=(),
            page_images=tuple(str(path) for path in images),
            protocol="oneshot",
            page_texts=shown,
            max_tokens=max_tokens,
            page_numbers=tuple(pages or ()),
            page_count=count if pages else None,
            aliases=_aliases(identity, backend),
        )
        refused = beyond_limits(images, shown, backend.identity)
        if refused:
            generation = GenerationResult(raw_text="", finish_reason="input_limit")
        else:
            generation = backend.generate_cited(task)  # a batch run's store raises Pending here
        if generation.finish_reason != "input_limit":
            return task, generation
        if dpi > 72 and not refused:
            dpi = max(72, dpi * 2 // 3)
        elif not text_only and any(
            len(text.strip()) >= 200 and number not in edge_pages(count)
            for number, text in zip(numbers, own, strict=True)
        ):
            text_only, dpi = True, first_dpi
        else:
            return task, generation


# Resolution of a page read on its own when it has no text layer (single short pages read better at
# 200 than at 150 DPI on ExtractBench samples).
SCAN_PAGE_DPI = 200

# Output allowed for the fields a cut-off answer never reached, when none of them is a list: a few
# values, while a whole-document request left unbounded can go on to write the table again.
REST_MAX_TOKENS = 2048


def _answer(
    document,
    schema,
    snapshot,
    texts,
    store,
    dpi,
    backend,
    label="document",
    depth=0,
    max_tokens=None,
    pages=None,
    ocr_text=False,
):
    """The answer to `schema`, every request it took as (label, generation, read by pages), and the
    whole-document request's task (scanned pages shown with their OCR text when `ocr_text`).

    Tables are read page by page when one request cannot hold them: an answer cut off at the output
    limit while writing a top-level list of records completes that list, and a document too long for
    one request even with small images reads all its top-level lists of records. A list that a
    complete answer gave only part of (`partial_lists`) is read page by page too, and the pages' rows
    replace the answer's when there are more of them. Each page request shows one page image and its
    text; the records of all pages are kept in page order, identical ones included, and rows continued
    across a page break get their group's labels (`carry_forward`). (Adding the end of the previous
    page as context changed nothing but added duplicated rows.) The other fields are asked for again:
    over the whole document after a cut-off answer (which may split in turn), or, before any page is
    read, from its first and last pages when the document does not fit (`edge_pages`).
    """
    import hashlib

    from docextract.cited import CitedTask, Pending
    from docextract.contracts import digest

    task, generation = _ask_document(
        document, schema, snapshot, texts, store, dpi, backend, max_tokens, pages, ocr_text
    )
    partial = parse_answer(generation.raw_text)
    too_long = generation.finish_reason == "input_limit" and depth == 0
    lazy = False
    if too_long:
        tables = record_lists(schema)
    elif generation.finish_reason == "length" and depth < 3:
        tables = [name for name in [cut_off_table(partial, schema, generation.raw_text)] if name]
    elif generation.finish_reason == "stop" and not pages and texts:
        tables, lazy = partial_lists(partial, schema, texts), True
    else:
        tables = []
    if not tables:
        return partial, [(label, generation, False)], task
    prepared = prepare_schema(schema)
    rows = split_schema(prepared, tables)
    images = render_page_images(document, store / snapshot.snapshot_id / f"dpi-{dpi}", dpi)
    bare = [index for index in range(len(images)) if not texts or len(texts[index].strip()) < 200]
    if bare and dpi < SCAN_PAGE_DPI:
        # A page without its own text is read from the image alone: sharper for one page at a time.
        sharp = render_page_images(
            document, store / snapshot.snapshot_id / f"dpi-{SCAN_PAGE_DPI}", SCAN_PAGE_DPI
        )
        images = [sharp[index] if index in bare else image for index, image in enumerate(images)]
    pending: list[str] = []
    answered = []
    steps = [(label, generation, True)]
    data = {key: value for key, value in partial.items() if key not in tables}
    missing = [key for key in prepared["properties"] if key not in tables and key not in partial]

    def ask_rest():
        return _answer(
            document,
            split_schema(prepared, missing),
            snapshot,
            texts,
            store,
            dpi,
            backend,
            label="remaining fields",
            depth=depth + 1,
            max_tokens=None
            if any(_array(prepared["properties"][key], prepared) is not None for key in missing)
            else REST_MAX_TOKENS,
            pages=edge_pages(len(images)) if too_long else pages,
            ocr_text=ocr_text,
        )

    if too_long and missing:
        # The fields outside tables are asked first (a batch run queues the pages once they are
        # answered): a schema too long for one request even without the tables (tax returns with
        # 760K-character schemas) leaves the whole document to the fallback, unread page by page.
        rest, more, _ = ask_rest()
        if more[0][1].finish_reason == "input_limit":
            return partial, [(label, generation, False), *more], task
        data.update(rest)
        steps += more
        missing = []
    done = (
        []
        if too_long or lazy
        else [record for record in partial.get(tables[0]) or [] if isinstance(record, dict)]
    )
    start = _last_page(done, texts) if done and texts else None
    span = max(1, min(8, ((start or 0) + 1) // 2))
    if start is not None and -(-(len(images) - start) // span) <= READ_ON_REQUESTS:
        # A table cut off at the output limit keeps the records its answer completed and is read on
        # from the page the last of them is printed on, a few pages at a time, each request given the
        # last record read so far (page by page, heading levels and groups were lost at page breaks).
        if missing:
            try:
                rest, more, _ = ask_rest()
                data.update(rest)
                steps += more
            except Pending as waiting:
                pending.extend(waiting.task_ids)
        try:
            records, more = _read_on(tables[0], rows, done, start, images, texts, backend, span)
            steps += more
        except Pending as waiting:
            pending.extend(waiting.task_ids)
        if pending:
            raise Pending(pending)
        data[tables[0]] = records
        return data, steps, task
    for index, image in enumerate(images):
        own = (texts[index],) if texts else ()
        cap = page_output_cap(texts[index]) if texts else None
        # A page cut off at its text-sized cap (its table may be an image) is asked again uncapped.
        for limit in (cap, None) if cap else (None,):
            identity = {
                "protocol": "oneshot-page/1",
                "schema": rows,
                "backend": backend.identity,
                "image": hashlib.sha256(image.read_bytes()).hexdigest(),
                "texts": digest(own),
                "page": index + 1,
                "pages": len(images),
                "max_tokens": limit,
            }
            page = CitedTask(
                task_id=digest(identity),
                original_schema=rows,
                units=(),
                page_images=(str(image),),
                protocol="oneshot",
                page_texts=own,
                first_page=index + 1,
                page_count=len(images),
                max_tokens=limit,
                aliases=_aliases(identity, backend),
            )
            try:
                answer = backend.generate_cited(page)
            except Pending as waiting:  # a batch run queues every page at once
                pending.extend(waiting.task_ids)
                break
            # Asked again uncapped only when the page's own cap cut it off; a cut-off below the cap
            # came from the engine's limit, which a second request would meet again.
            reached = limit is not None and (answer.output_tokens or 0) >= limit
            if answer.finish_reason == "length" and reached:
                steps.append((f"page {index + 1} (capped)", answer, True))
                continue
            answered.append((index, answer))
            break
    if missing:
        try:
            rest, more, _ = ask_rest()
            data.update(rest)
            steps += more
        except Pending as waiting:
            pending.extend(waiting.task_ids)
    if pending:
        raise Pending(pending)
    found = {table: [[] for _ in images] for table in tables}
    for index, answer in answered:
        steps.append((f"page {index + 1}", answer, False))
        parsed = parse_answer(answer.raw_text)
        for table in tables:
            records = parsed.get(table)
            found[table][index] = records if isinstance(records, list) else []
    for table in tables:
        records = found[table]
        if texts:
            records = carry_forward(records, list(texts), nullable_fields(prepared, table))
        listed = [record for page_records in records for record in page_records]
        # A list judged partial keeps the whole-document answer's rows unless its pages list more.
        data[table] = listed if not lazy or len(listed) > len(partial[table]) else partial[table]
    return data, steps, task


# Requests one after another that a cut-off table may take to be read on; a longer table (thousands
# of rows over dozens of dense pages) is read page by page, all pages at once.
READ_ON_REQUESTS = 12


def _last_page(records: list[dict], texts) -> int | None:
    """The page (an index) the last of `records` is printed on, following them through the document
    from its first page (a record whose values are printed again further on, as in a summary, takes
    its first page at or after its predecessor's); None when none of them is found."""
    current, found = 0, False
    for pages in record_pages(records, texts):
        later = [page for page in pages if page >= current]
        if later:
            current, found = min(later), True
    return current if found else None


def _beyond(records: list, new: list) -> list:
    """`new` without the records it repeats at its start from the end of `records` (the record given
    as context, or one or two before it)."""
    tail = {json.dumps(record, sort_keys=True) for record in records[-3:]}
    skip = 0
    while skip < len(new) and json.dumps(new[skip], sort_keys=True) in tail:
        skip += 1
    return new[skip:]


def _read_on(table, rows, done, start, images, texts, backend, span):
    """The records of `table` (`rows`: the schema of that list alone) read on from page `start` (an
    index) to the end, `span` pages per request, after the records `done`; each request is given the
    last record read so far and its answer's records are added in order. A request cut off at the
    output limit is followed by one from the page its last record is printed on."""
    import hashlib

    from docextract.cited import CitedTask
    from docextract.contracts import digest

    records, steps, page, count = list(done), [], start, len(images)
    while page < count:
        chunk = list(range(page, min(count, page + span)))
        after = json.dumps(records[-1], ensure_ascii=False) if records else None
        shown = tuple(texts[index] for index in chunk) if texts else ()
        identity = {
            "protocol": "oneshot-continue/1",
            "schema": rows,
            "backend": backend.identity,
            "images": [hashlib.sha256(images[index].read_bytes()).hexdigest() for index in chunk],
            "texts": digest(shown),
            "pages": [index + 1 for index in chunk],
            "count": count,
            "after": after,
        }
        task = CitedTask(
            task_id=digest(identity),
            original_schema=rows,
            units=(),
            page_images=tuple(str(images[index]) for index in chunk),
            protocol="oneshot",
            page_texts=shown,
            page_numbers=tuple(index + 1 for index in chunk),
            page_count=count,
            after=after,
            aliases=_aliases(identity, backend),
        )
        answer = backend.generate_cited(task)  # a batch run's store raises Pending here
        listed = parse_answer(answer.raw_text).get(table)
        new = _beyond(
            records,
            [record for record in listed if isinstance(record, dict)] if isinstance(listed, list) else [],
        )
        cut = answer.finish_reason == "length" and bool(new)
        steps.append((f"pages {chunk[0] + 1}-{chunk[-1] + 1}", answer, cut))
        records += new
        if cut:
            last = _last_page(new, shown) if shown else None
            page = chunk[last] if last is not None else chunk[-1]
        else:
            page = chunk[-1] + 1
    return records, steps


def extract_oneshot(
    document: Path,
    schema: dict,
    *,
    workspace: Path,
    backend,
    parser=None,
    dpi: int | None = None,
    text_layer: bool = True,
    ocr_text: bool = False,
    page_images: Path | None = None,
    fallback=None,
):
    """Extract a whole document in one request: every page image, the native text layer and the full
    schema; the answer is typed to the schema and its printed values are located for evidence. With
    `ocr_text`, whole-document requests show a scanned page's OCR lines as its text (on 24 scanned
    documents 0.911 -> 0.926; page-by-page reading keeps the PDF's own text, untested with OCR).

    A request too long for the model's window is retried with smaller images (two thirds of the
    resolution, down to 72 DPI). An answer cut off at the output limit while writing a top-level list
    of records is completed page by page, and a document that still does not fit has its lists of
    records read page by page (`_answer`). If it has none, or an answer cut off elsewhere cannot be
    completed that way, `fallback()` (the windowed pipeline) answers instead when given.
    """
    import shutil

    from docextract.contracts import (
        Completion,
        Coverage,
        ExtractionResult,
        FieldEvidence,
        UnresolvedItem,
        Usage,
        ValidationReport,
        digest,
        save_json,
    )
    from docextract.parsing.pdfium import PdfiumParser
    from docextract.schema import Schema

    snapshot = (parser or PdfiumParser()).parse(document)
    texts = tuple(pdf_page_texts(document)) if text_layer else ()
    store = page_images or workspace / "page-images"
    dpi = dpi or page_dpi(len(snapshot.pages))
    data, steps, task = _answer(document, schema, snapshot, texts, store, dpi, backend, ocr_text=ocr_text)
    generation, by_pages = steps[0][1], steps[0][2]
    if not by_pages and generation.finish_reason in ("input_limit", "length") and fallback is not None:
        return fallback()
    # A replay of a run that stopped before this document's answer: nothing was read, so nothing is
    # filled in (not even check boxes the schema defines as false when blank).
    answered = by_pages or generation.finish_reason not in ("not_stored", "input_limit")
    data = conform(data, schema) if answered else {}
    compiled = Schema(schema)
    try:
        validation = compiled.validate(data)
    except Exception as error:  # an unsupported schema feature still yields the answer
        validation = ValidationReport(valid=False, errors=(f"not validated: {error}",))
    located = {
        pointer: (atom_ids, printed, transform)
        for pointer, _, atom_ids, printed, transform in locate_values(data, schema, snapshot)
    }
    evidence, unresolved = [], []
    for pointer, value, _ in _leaves(data, schema):
        if value is None:
            evidence.append(FieldEvidence(pointer=pointer, value=None, status="absent_after_search"))
        elif pointer in located:
            atom_ids, printed, transform = located[pointer]
            evidence.append(
                FieldEvidence(
                    pointer=pointer,
                    value=value,
                    status="observed",
                    source_atom_ids=atom_ids,
                    source_lexeme=printed,
                    transform=transform,
                    support="character",
                )
            )
        else:
            # A check box, or a value read from the image that the parsed text does not contain.
            evidence.append(
                FieldEvidence(pointer=pointer, value=value, status="observed", transform="image_read")
            )
            if not isinstance(value, bool):
                unresolved.append(
                    UnresolvedItem(pointer=pointer, reason="value not located in the page text")
                )
    for label, step, handled in steps:
        if step.finish_reason == "length" and not handled:
            reason = "answer cut off at the output limit"
            unresolved.append(
                UnresolvedItem(pointer="", reason=reason if step is generation else f"{label}: {reason}")
            )
        elif step.finish_reason in ("not_stored", "input_limit") and not handled:
            unresolved.append(
                UnresolvedItem(pointer="", reason=f"{label} not answered ({step.finish_reason})")
            )
    lines = [atom.atom_id for page in snapshot.pages for atom in page.atoms if atom.kind == "line"]
    result = ExtractionResult(
        snapshot_id=snapshot.snapshot_id,
        schema_sha256=digest(schema),
        run_id=task.task_id,
        data=data,
        field_evidence=tuple(evidence),
        source_records=(),
        coverage=Coverage(total_lines=len(lines), examined_lines=len(lines)),
        unresolved=tuple(unresolved),
        validation=validation,
        usage=Usage(
            backend=backend.identity.get("backend", "unknown"),
            generated_attempts=len(steps),
            committed_tasks=len(steps),
            input_tokens=sum(step.input_tokens or 0 for _, step, _ in steps),
            output_tokens=sum(step.output_tokens or 0 for _, step, _ in steps),
            cost_usd=None,
        ),
        completion=Completion(
            execution_complete=True,
            status="complete" if validation.valid and not unresolved else "partial",
        ),
        snapshot=snapshot,
    )
    run_dir = workspace / task.task_id
    save_json(run_dir / "result.json", result)
    temporary = workspace / "result.json.tmp"
    shutil.copyfile(run_dir / "result.json", temporary)
    temporary.replace(workspace / "result.json")
    return result
