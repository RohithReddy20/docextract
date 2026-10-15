"""Measure benchmark workload on a remote worker; exports counts and timings, never document content."""

import argparse
import hashlib
import json
import resource
import time
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from docextract.contracts import digest, save_json
from docextract.interpretation import evidence_units, render_evidence, windows
from docextract.parsing.pdfium import PDFIUM_LOCK, PdfiumParser
from docextract.planning import object_arrays, scalar_leaves
from docextract.schema import Schema

REVISION = "51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc"


def counted(value) -> tuple[int, int]:
    """Non-null scalar values and array-of-object items in an output document."""
    if isinstance(value, dict):
        totals = [counted(child) for child in value.values()]
        return sum(t[0] for t in totals), sum(t[1] for t in totals)
    if isinstance(value, list):
        totals = [counted(child) for child in value]
        records = sum(isinstance(child, dict) for child in value)
        return sum(t[0] for t in totals), records + sum(t[1] for t in totals)
    return int(value is not None), 0


def schema_profile(schema: dict) -> dict:
    profile = {"schema_chars": len(json.dumps(schema, ensure_ascii=False))}

    def descriptions(node):
        if isinstance(node, dict):
            own = len(node["description"]) if isinstance(node.get("description"), str) else 0
            return own + sum(descriptions(child) for child in node.values())
        if isinstance(node, list):
            return sum(descriptions(child) for child in node)
        return 0

    profile["description_chars"] = descriptions(schema)
    compiled = Schema(schema)
    profile["supported"] = compiled.support.supported
    if not compiled.support.supported:
        return profile
    arrays = object_arrays(compiled)
    profile["object_arrays"] = len(arrays)
    profile["max_array_depth"] = max((path.count("*") + 1 for path in arrays), default=0)
    profile["root_scalar_leaves"] = len(scalar_leaves(compiled))
    profile["item_scalar_leaves"] = sum(len(scalar_leaves(compiled, path + "/*")) for path in arrays)
    return profile


def image_pages(pdf_bytes: bytes) -> tuple[int, int]:
    """Pages whose image objects cover at least half the page, and those with any image."""
    large = anything = 0
    with PDFIUM_LOCK, closing(pdfium.PdfDocument(pdf_bytes)) as pdf:
        for index in range(len(pdf)):
            with closing(pdf[index]) as page:
                width, height = page.get_size()
                area = 0.0
                seen = False
                for item in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
                    left, bottom, right, top = item.get_pos()
                    area += max(0.0, right - left) * max(0.0, top - bottom)
                    seen = True
                anything += seen
                large += area >= 0.5 * width * height
    return large, anything


def measure(pdf: Path, schema, expected, tokenizer, max_window_chars: int) -> dict:
    content = pdf.read_bytes()
    row = {"pdf_bytes": len(content)}
    started = time.perf_counter()
    snapshot = PdfiumParser().parse(pdf)
    row["parse_seconds"] = round(time.perf_counter() - started, 3)
    row["pages"] = len(snapshot.pages)
    row["pages_without_native_text"] = sum(bool(page.quality_flags) for page in snapshot.pages)
    row["native_characters"] = sum(
        atom.kind == "character" and not atom.text.isspace() for page in snapshot.pages for atom in page.atoms
    )
    row["pages_mostly_image"], row["pages_with_images"] = image_pages(content)
    units = evidence_units(snapshot)
    prompt = [unit.prompt_data() for unit in units]
    row["evidence_units"] = len(units)
    row["evidence_text_chars"] = sum(len(unit.text) for unit in units)
    row["evidence_prompt_chars"] = len(json.dumps(prompt, ensure_ascii=False))
    compact = render_evidence(units)
    row["compact_evidence_chars"] = len(compact)
    row["windows_at_max_chars"] = sum(1 for _ in windows(units, max_chars=max_window_chars, overlap_lines=2))
    row.update(schema_profile(schema))
    values, records = counted(expected)
    row["expected_scalar_values"] = values
    row["expected_records"] = records
    row["expected_json_chars"] = len(json.dumps(expected, ensure_ascii=False))
    if tokenizer:
        encode = tokenizer.encode
        row["evidence_prompt_tokens"] = len(encode(json.dumps(prompt, ensure_ascii=False)).ids)
        row["compact_evidence_tokens"] = len(encode(compact).ids)
        row["schema_tokens"] = len(encode(json.dumps(schema, ensure_ascii=False)).ids)
        row["expected_json_tokens"] = len(encode(json.dumps(expected, ensure_ascii=False)).ids)
    row["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return row


def inventory(snapshot: Path, tokenizer_file: Path | None, deadline: float, max_window_chars: int) -> dict:
    manifest = json.loads((snapshot.parent / "dataset-manifest.json").read_text())
    if manifest.get("revision") != REVISION or manifest.get("mode") != "downloaded_remote_snapshot":
        raise ValueError("Require a downloaded, pinned manifest from prepare_benchmark.py")
    tokenizer = None
    if tokenizer_file:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_file))
    rows = []
    for split in ("short", "medium", "long"):
        for line in (snapshot / f"{split}.jsonl").open("rb"):
            source = json.loads(line)
            case = {"case": digest(source.get("id") or source["pdf"])[:16], "split": split}
            if time.monotonic() > deadline:
                rows.append(case | {"error": "deadline"})
                continue
            try:
                schema = source["data_schema"]
                schema = json.loads(schema) if isinstance(schema, str) else schema
                expected = source["expected_output"]
                expected = json.loads(expected) if isinstance(expected, str) else expected
                pdf = (snapshot / source["pdf"]).resolve()
                if not pdf.is_relative_to(snapshot.resolve()):
                    raise ValueError("Document path escapes the snapshot")
                case |= measure(pdf, schema, expected, tokenizer, max_window_chars)
            except Exception as error:  # every case stays in the accounting population
                case["error"] = type(error).__name__
            rows.append(case)
            print(json.dumps({key: case.get(key) for key in ("split", "pages", "parse_seconds", "error")}))
    tokenizer_hash = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() if tokenizer_file else None
    return {
        "dataset_revision": REVISION,
        "content_exported": False,
        "tokenizer_sha256": tokenizer_hash,
        "max_window_chars": max_window_chars,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seconds", type=float, default=9000)
    parser.add_argument("--max-window-chars", type=int, default=6000)
    args = parser.parse_args()
    deadline = time.monotonic() + args.seconds
    save_json(args.output, inventory(args.snapshot, args.tokenizer, deadline, args.max_window_chars))
    print(args.output)


if __name__ == "__main__":
    main()
