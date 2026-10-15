"""Summarize a scored ExtractBench development run: per-document F1 and leaf-level differences.

Reads the scorer report and its per-case predicted/expected dumps. Development cases only; the
comparison here is a coarse, order-insensitive leaf diff for triage, not the official metric.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def leaves(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from leaves(child, f"{path}.{key}" if path else key)
    elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        for item in value:
            yield from leaves(item, f"{path}[]")
    else:
        yield path, json.dumps(value, sort_keys=True)


def diff(predicted, expected):
    got, want = Counter(leaves(predicted)), Counter(leaves(expected))
    missing, extra = want - got, got - want
    by_path = Counter()
    for (path, _), count in missing.items():
        by_path[("missing", path)] += count
    for (path, _), count in extra.items():
        by_path[("extra", path)] += count
    return by_path, sum(want.values()), sum((got & want).values())


def engine_summary(run: Path) -> dict:
    """Request counts, token totals and finish reasons from a run's streamed engine telemetry."""
    rows = [row for row in json.loads((run / "telemetry.json").read_text())["tasks"] if "task_id" in row]
    generations = [json.loads(line) for line in (run / "generations.jsonl").read_text().splitlines() if line]
    finishes = Counter(row["generation"]["finish_reason"] for row in generations)
    total = sum(row["seconds"] for row in rows)
    return {
        "answers_stored": len(generations),
        "finish_reasons": dict(finishes),
        "input_tokens": sum(row["generation"]["input_tokens"] for row in generations),
        "output_tokens": sum(row["generation"]["output_tokens"] for row in generations),
        "max_output_tokens": max((row["generation"]["output_tokens"] for row in generations), default=0),
        "mean_request_seconds": round(total / len(rows), 1) if rows else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--run", type=Path, help="exported run directory with telemetry and answers")
    parser.add_argument("--packet", type=Path, help="exported packet, to report OCR pages per case")
    parser.add_argument(
        "--workload", type=Path, help="private per-case workload inventory: answer size versus output tokens"
    )
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    if args.run:
        print(json.dumps(engine_summary(args.run), indent=1))
    pages = {}
    if args.run and args.packet:
        documents = json.loads((args.packet / "inference/documents.json").read_text())["cases"]
        for case in documents:
            path = args.run / "snapshots" / f"{case['pdf_sha256']}.json"
            if path.exists():
                flags = [page["quality_flags"] for page in json.loads(path.read_text())["pages"]]
                ocr = sum("ocr_text" in page for page in flags)
                rotated = sum(any(flag.startswith("ocr_rotated") for flag in page) for page in flags)
                pages[case["id"]] = f"ocr_pages={ocr}/{len(flags)}" + (
                    f" rotated={rotated}" if rotated else ""
                )
    if args.workload and args.run:
        expected = {
            row["case"]: row.get("expected_json_tokens")
            for row in json.loads(args.workload.read_text())["cases"]
        }
        summary = {row["id"]: row for row in json.loads((args.run / "summary.json").read_text())["cases"]}
        produced = sum(row.get("output_tokens", 0) for row in summary.values())
        answers = sum(expected.get(case_id) or 0 for case_id in summary)
        print(f"output tokens {produced} for answer keys of {answers} plain-JSON tokens")
        for case_id, row in summary.items():
            if row.get("output_tokens"):
                print(f"  {case_id} output={row['output_tokens']} answer_key={expected.get(case_id)}")
    report = json.loads(args.scores.read_text())
    print(json.dumps(report["groups"], indent=1))
    for row in sorted(report["cases"], key=lambda row: row.get("extract_unified_value_f1") or 0):
        detail = json.loads((args.details / f"{row['id']}.json").read_text())
        by_path, total, matched = diff(detail["predicted"], detail["expected"] or {})
        f1 = row.get("extract_unified_value_f1")
        print(
            f"\n{row['id']} {row['split']:6} F1={f1 if f1 is None else round(f1, 3)} "
            f"nulls_omitted={row.get('value_f1_nulls_omitted')} pages={row.get('pages')} "
            f"status={row.get('status')} result={row['result']} leaves={matched}/{total} "
            f"{pages.get(row['id'], '')}"
        )
        for (kind, path), count in by_path.most_common(args.examples * 2):
            print(f"   {kind:7} {count:4d} {path}")


if __name__ == "__main__":
    main()
