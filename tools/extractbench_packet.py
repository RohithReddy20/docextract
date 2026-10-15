"""Split a pinned remote ExtractBench snapshot into inference inputs and separate scoring files.

Run on a remote worker only. Inference receives PDFs and original schemas; answers, field rules and
row-identity hints are written to a separate evaluation directory that inference never reads.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from docextract.contracts import digest, save_json

SALT = "docextract-dev/1"


def ranked(rows: list[dict]) -> list[dict]:
    """A fixed, content-blind order; selection never looks at answers or results."""
    return sorted(rows, key=lambda row: hashlib.sha256(f"{SALT}/{row['id']}".encode()).hexdigest())


def select(rows: list[dict], count: int, offset: int = 0, ids: set[str] | None = None) -> list[dict]:
    """A hash-order slice, or exactly the given case ids (chosen from properties, never answers)."""
    order = ranked(rows)
    if ids:
        return [row for row in order if digest(row["id"])[:16] in ids]
    return order[offset : offset + count]


def build(
    snapshot: Path,
    output: Path,
    counts: dict[str, int],
    offset: int | dict[str, int] = 0,
    ids: set[str] | None = None,
) -> dict:
    from extract_bench.data.download import _write_sidecar

    inference, evaluation = output / "inference", output / "evaluation"
    inference.mkdir(parents=True, exist_ok=True)
    cases, index = [], []
    for split, count in counts.items():
        with (snapshot / f"{split}.jsonl").open(encoding="utf-8") as source:
            rows = [json.loads(line) for line in source if line.strip()]
        skip = offset.get(split, 0) if isinstance(offset, dict) else offset
        for row in select(rows, count, skip, ids):
            case_id = digest(row["id"])[:16]
            source = (snapshot / row["pdf"]).resolve()
            if not source.is_relative_to(snapshot.resolve()):
                raise ValueError("Document path escapes the snapshot")
            shutil.copyfile(source, inference / f"{case_id}.pdf")
            schema = json.loads(row["data_schema"])
            save_json(inference / f"{case_id}.schema.json", schema)
            cases.append({"id": case_id, "pdf": f"{case_id}.pdf", "schema": f"{case_id}.schema.json"})
            sidecar = _write_sidecar(row, evaluation / split, snapshot)
            index.append({"id": case_id, "split": split, "test_json": str(sidecar.relative_to(evaluation))})
    save_json(inference / "cases.json", {"version": "inference-cases/1", "cases": cases})
    save_json(evaluation / "index.json", {"version": "evaluation-index/1", "salt": SALT, "cases": index})
    return {"cases": len(cases), "inference": str(inference / "cases.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--short", type=int, default=0)
    parser.add_argument("--medium", type=int, default=0)
    parser.add_argument("--long", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--offsets", type=json.loads, help='per-split offsets, e.g. {"short": 16, "medium": 4}'
    )
    parser.add_argument("--ids", default="", help="comma-separated case ids; selects exactly these")
    args = parser.parse_args()
    counts = {split: getattr(args, split) for split in ("short", "medium", "long") if getattr(args, split)}
    ids = {case for case in args.ids.split(",") if case}
    if ids:
        counts = {split: 0 for split in ("short", "medium", "long")}
    offset = args.offsets if args.offsets else args.offset
    print(json.dumps(build(args.snapshot, args.output, counts, offset, ids or None)))


if __name__ == "__main__":
    main()
