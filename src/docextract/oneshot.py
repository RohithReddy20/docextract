"""Image-first extraction: the whole document in one request, evidence located afterwards.

The model sees every page as an image, the PDF's own text layer and the full schema, and answers with
one JSON object (`extract_oneshot`). The answer is typed to the schema (`conform`) and each printed value
is then located in the parsed page text (`locate_evidence`), so values keep field-level evidence without
the model having to cite anything. On ExtractBench samples this beat the windowed cited pipeline on
short, medium and long documents (docs/model-trials.md); the windowed pipeline remains for schemas or
answers too large for one request.

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


def user_text(schema: dict, page_texts: list[str] | None = None) -> str:
    """The request text: the prepared schema, the instructions and, when given, the PDF's text layer."""
    text = (
        "Extract every field from the attached document page image(s) according to the JSON schema below.\n\n"
        "JSON schema:\n"
        f"{json.dumps(prepare_schema(schema), indent=2)}\n\n"
        "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
        f"{USER_INSTRUCTION}"
    )
    if page_texts:
        layer = "\n\n".join(f"--- page {i + 1} ---\n{t.strip()}" for i, t in enumerate(page_texts))
        text += f"\n\nText layer of the document (may be incomplete; trust the images):\n{layer}"
    return text


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
    blank", "never null") becomes false. Anything that does not fit is left as the model gave it.
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
                    filled = walk(None, child)
                    if filled is not None:
                        out[key] = filled
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


def locate_values(data: Any, schema: dict, snapshot):
    """Where each printed value of an answer appears in the parsed page text (native text or OCR):
    (pointer, value, atom ids, printed text, transform) per located value, in answer order.

    Values are searched in document order from the previous located value, so repeated rows find their
    own lines; a value printed nowhere (or a check box) is not yielded rather than guessed. (Preferring
    the occurrence printed after the field's label was tried and located fewer values correctly.)
    """
    from docextract.cited import _atoms, _virtual, find_all
    from docextract.interpretation import evidence_units

    units = list(evidence_units(snapshot))
    if not units:
        return
    text, offsets, _ = _virtual(units)
    cursor = 0
    for pointer, value, kind in _leaves(data, schema):
        if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
            continue
        occurrences = list(find_all(value, kind, text))
        if not occurrences:
            continue
        start, end, transform = next((o for o in occurrences if o[0] >= cursor), occurrences[0])
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


def native_page_texts(snapshot) -> list[str]:
    """Each page's own PDF text, one line per text line; recognized (OCR) text is left out, since the
    model reads scanned pages better from the image than from a copy with recognition errors."""
    texts = []
    for page in snapshot.pages:
        lines = [
            atom.text
            for atom in page.atoms
            if atom.kind == "line" and atom.source_channel == "native_pdf" and atom.text.strip()
        ]
        texts.append("\n".join(lines))
    return texts


def extract_oneshot(
    document: Path,
    schema: dict,
    *,
    workspace: Path,
    backend,
    parser=None,
    dpi: int | None = None,
    text_layer: bool = True,
    page_images: Path | None = None,
    fallback=None,
):
    """Extract a whole document in one request: every page image, the native text layer and the full
    schema; the answer is typed to the schema and its printed values are located for evidence.

    A request too long for the model's window is retried with smaller images (two thirds of the
    resolution, down to 72 DPI). If it still does not fit, or the answer is cut off at the output limit,
    `fallback()` (the windowed pipeline) answers instead when given.
    """
    import hashlib
    import shutil

    from docextract.cited import CitedTask
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
    texts = tuple(native_page_texts(snapshot)) if text_layer else ()
    store = page_images or workspace / "page-images"
    dpi = dpi or page_dpi(len(snapshot.pages))
    while True:
        images = render_page_images(document, store / snapshot.snapshot_id / f"dpi-{dpi}", dpi)
        identity = {
            "protocol": "oneshot/1",
            "schema": schema,
            "backend": backend.identity,
            "images": [hashlib.sha256(path.read_bytes()).hexdigest() for path in images],
            "texts": digest(texts),
        }
        task = CitedTask(
            task_id=digest(identity),
            original_schema=schema,
            units=(),
            page_images=tuple(str(path) for path in images),
            protocol="oneshot",
            page_texts=texts,
        )
        generation = backend.generate_cited(task)  # a batch run's store raises Pending here
        if generation.finish_reason != "input_limit" or dpi <= 72:
            break
        dpi = max(72, dpi * 2 // 3)
    if generation.finish_reason in ("input_limit", "length") and fallback is not None:
        return fallback()
    # A replay of a run that stopped before this document's answer: nothing was read, so nothing is
    # filled in (not even check boxes the schema defines as false when blank).
    answered = generation.finish_reason not in ("not_stored", "input_limit")
    data = conform(parse_answer(generation.raw_text), schema) if answered else {}
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
    if generation.finish_reason == "length":
        unresolved.append(UnresolvedItem(pointer="", reason="answer cut off at the output limit"))
    if not answered:
        unresolved.append(
            UnresolvedItem(pointer="", reason=f"document not answered ({generation.finish_reason})")
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
            generated_attempts=1,
            committed_tasks=1,
            input_tokens=generation.input_tokens or 0,
            output_tokens=generation.output_tokens or 0,
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
