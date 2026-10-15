"""Offline check of schema-group routing on development documents (no model; answers used only to
measure). For each expected non-null scalar that is printed in the document, find the windows that
contain it and report whether routing still asks such a window about the field's schema group.
"""

import argparse
import gzip
import json
import re
from pathlib import Path

from docextract.contracts import ParseSnapshot
from docextract.interpretation import evidence_units, render_evidence, windows
from docextract.partition import factor_notes, inline, leaf_pointers, partition
from docextract.routing import route

FACTOR = False


def scalars(value, pointer=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from scalars(child, f"{pointer}/{key}")
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from scalars(child, pointer + "/*" if isinstance(child, dict) else pointer)
            else:
                yield pointer, child
    else:
        yield pointer, value


def squeeze(text: str) -> str:
    return re.sub(r"[\s,$]", "", str(text)).lower()


def check(snapshot: ParseSnapshot, schema: dict, expected: dict, budget: int, schema_chars: int) -> dict:
    units = evidence_units(snapshot)
    if FACTOR:
        schema = factor_notes(schema)
    groups = partition(schema, schema_chars) or [inline(schema)]
    largest = max(len(json.dumps(group, separators=(",", ":"))) for group in groups)
    chunks = list(windows(units, max_chars=max(24000, budget - largest), overlap_lines=2, unit_overhead=12))
    texts = [squeeze(render_evidence(chunk, refs=False)) for chunk in chunks]
    plan = route([render_evidence(chunk, refs=False) for chunk in chunks], groups)
    sizes = [len(json.dumps(group, separators=(",", ":"))) for group in groups]
    prompt = sum(
        sizes[g] + len(render_evidence(chunks[w])) + 3000 for w, keep in enumerate(plan) for g in keep
    )
    covered = [leaf_pointers(group) for group in groups]
    totals = {
        "windows": len(chunks),
        "groups": len(groups),
        "requests": sum(map(len, plan)),
        "all_pairs": len(chunks) * len(groups),
        "prompt_kchars": prompt // 1000,
        "values": 0,
        "kept": 0,
        "lost": [],
    }
    for pointer, value in scalars(expected):
        if value is None or isinstance(value, bool) or len(squeeze(value)) < 2:
            continue
        field = pointer.replace("/*", "")
        owner = next(
            (
                i
                for i, cover in enumerate(covered)
                if any(
                    field == c.replace("/*", "")
                    or field.startswith(c.replace("/*", "") + "/")
                    or c.replace("/*", "").startswith(field)
                    for c in cover
                )
            ),
            None,
        )
        hits = [i for i, text in enumerate(texts) if squeeze(value) in text]
        if owner is None or not hits:
            continue
        totals["values"] += 1
        if any(owner in plan[i] for i in hits):
            totals["kept"] += 1
        else:
            totals["lost"].append(pointer)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True, help="per-case expected answers")
    parser.add_argument("--budget", type=int, default=72000)
    parser.add_argument("--schema-chars", type=int, default=48000)
    parser.add_argument("--only", nargs="*", help="case ids to check")
    parser.add_argument("--factor-notes", action="store_true", help="shorten repeated description sentences")
    args = parser.parse_args()
    global FACTOR
    FACTOR = args.factor_notes
    documents = json.loads((args.packet / "inference/documents.json").read_text())["cases"]
    for case in documents:
        path = args.snapshots / f"{case['pdf_sha256']}.json.gz"
        detail = args.details / f"{case['id']}.json"
        if not path.exists() or not detail.exists():
            continue
        with gzip.open(path, "rt") as source:
            snapshot = ParseSnapshot.model_validate_json(source.read())
        schema = json.loads((args.packet / "inference" / case["schema"]).read_text())
        expected = json.loads(detail.read_text())["expected"] or {}
        if args.only and case["id"] not in args.only:
            continue
        result = check(snapshot, schema, expected, args.budget, args.schema_chars)
        if result["groups"] > 1:
            lost = result.pop("lost")
            print(case["id"], json.dumps(result), "lost:", lost[:5])


if __name__ == "__main__":
    main()
