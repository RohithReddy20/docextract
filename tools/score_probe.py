"""Score the answers of one extraction probe with ExtractBench's official evaluator, locally.

Answers (`artifacts/probes/<name>/answers.json`) are parsed the way ExtractBench's vLLM provider parses
them (a leading think block removed, truncated JSON salvaged, runaway nesting pruned) and scored per
document and method. Expected outputs come from `artifacts/probes/data/<id>.test.json`; a reference run's
per-document details (for example final-v1) can be shown alongside.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

try:
    import openai  # noqa: F401
except ImportError:
    # ExtractBench's vLLM provider module imports the OpenAI client at load time; only its pure JSON
    # helpers are used here, so a placeholder stands in when the client is not installed.
    sys.modules["openai"] = types.SimpleNamespace(OpenAI=None)

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.inference.providers.extract.vllm_extract import prune_excess_depth, salvage_truncated_json
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult
from score_extractbench import load_case

from docextract.oneshot import carry_forward, conform, merge_answers, nullable_fields, pdf_page_texts

ROOT = Path(__file__).resolve().parents[1]
THINK = re.compile(r"^\s*<think>.*?(?:</think>\s*|\Z)", re.DOTALL)
METRICS = (
    "extract_unified_value_f1",
    "extract_unified_value_precision",
    "extract_unified_value_recall",
    "extract_unified_grounded_f1",
    "extract_unified_page_f1",
)


def parse(raw: str):
    text = THINK.sub("", raw, count=1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = salvage_truncated_json(text)
    if not isinstance(data, dict):
        return {}
    return prune_excess_depth(data)[0]


def assemble(answered: list[dict], texts: list[str] | None = None, schema: dict | None = None) -> dict:
    """One document's answer from its requests, in page order. A table read page by page ("rows"
    parts) keeps every page's records, identical ones included; the other fields come from the
    whole-document request ("rest"). Plain page batches are merged as before. With the PDF's page
    `texts` and the `schema`, group labels of rows continued across a page break are repaired
    (`carry_forward`, as the product does)."""
    rows = [answer for answer in answered if answer.get("part") == "rows"]
    if not rows and not any(answer.get("part") == "rest" for answer in answered):
        return merge_answers([parse(answer["raw_text"]) for answer in answered]) or {}
    rest = [answer for answer in answered if answer.get("part") == "rest"]
    data = merge_answers([parse(answer["raw_text"]) for answer in rest]) or {}
    tables: dict[str, list[list]] = {}
    for answer in rows:
        for key, value in parse(answer["raw_text"]).items():
            if isinstance(value, list):
                pages = tables.setdefault(key, [[] for _ in texts] if texts else [])
                if texts and answer["batch"] < len(pages):
                    pages[answer["batch"]] = value
                else:
                    pages.append(value)
    for key, pages in tables.items():
        if texts and schema:
            pages = carry_forward(pages, texts, nullable_fields(schema, key))
        data[key] = [*(data.get(key) or []), *(record for page in pages for record in page)]
    return data


def pdfs(case_id: str) -> list[Path]:
    """Where a probe document's PDF may be kept locally (git-ignored)."""
    return [ROOT / "artifacts" / "probes" / folder / f"{case_id}.pdf" for folder in ("data", "giant")]


def evaluate(evaluator, case_id: str, data: dict, test_case, citations=None) -> dict:
    now = datetime.now(UTC)
    cited = [
        FieldCitation(
            field_path=c["field_path"],
            page=c["page"],
            bbox=c["bbox"],
            reference_text=c["reference_text"],
            source=c["source"],
        )
        for c in citations or []
    ]
    output = ExtractOutput(
        example_id=case_id, pipeline_name="probe", extracted_data=data, field_citations=cited
    )
    inference = InferenceResult(
        request=InferenceRequest(example_id=case_id, source_file_path=case_id, product_type="extract"),
        pipeline_name="probe",
        product_type="extract",
        raw_output={},
        output=output,
        started_at=now,
        completed_at=now,
        latency_in_ms=0,
    )
    metrics = {m.metric_name: m.value for m in evaluator.evaluate(inference, test_case).metrics}
    return {name: metrics.get(name) for name in METRICS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT / "artifacts/modal/final-v1/final-details")
    parser.add_argument("--conform", action="store_true", help="give answers the schema's types first")
    parser.add_argument(
        "--fallback", action="store_true", help="refused or cut-off answers take the reference result"
    )
    parser.add_argument("--export", action="store_true", help="write the scored answers for grounding")
    parser.add_argument("--evidence", action="store_true", help="score with citations from evidence.json")
    parser.add_argument(
        "--carry", action="store_true", help="repair group labels across page breaks (needs the PDF locally)"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    data_dir = ROOT / "artifacts" / "probes" / "data"
    out = ROOT / "artifacts" / "probes" / config["name"]
    answers = json.loads((out / "answers.json").read_text())["outputs"]
    evaluator = ExtractEvaluator()
    cases = {case_id: load_case(data_dir / f"{case_id}.test.json") for case_id in config["ids"]}
    methods = [method["name"] for method in config["methods"]]
    table: dict[str, dict[str, dict]] = {case_id: {} for case_id in config["ids"]}
    # Answers to page batches of one document are merged in page order before scoring.
    exported: dict[str, dict] = {}
    citations = {}
    if args.evidence:
        citations = {
            key: value for key, value in json.loads((out / "evidence.json").read_text()).items() if value
        }
    grouped: dict[tuple[str, str], list[dict]] = {}
    for answer in answers:
        grouped.setdefault((answer["id"], answer["variant"]), []).append(answer)
    for (case_id, variant), parts in grouped.items():
        parts.sort(key=lambda answer: answer.get("batch", 0))
        answered = [
            answer
            for answer in parts
            if answer["finish_reason"] != "deadline" and not answer["finish_reason"].startswith("error")
        ]
        if not answered:
            # Never answered (stopped at the deadline, or refused): not a score of zero.
            table[case_id][variant] = {"finish": parts[0]["finish_reason"], "output_tokens": 0}
            continue
        texts = schema = None
        if args.carry:
            pdf = next((path for path in pdfs(case_id) if path.exists()), None)
            texts = pdf_page_texts(pdf) if pdf else None
            schema = json.loads((data_dir / f"{case_id}.schema.json").read_text())
        data = assemble(answered, texts, schema)
        if args.conform:
            data = conform(data, json.loads((data_dir / f"{case_id}.schema.json").read_text()))
        key = f"{case_id}|{variant}"
        exported[key] = {
            "key": key,
            "pdf": f"packets/{config['packet']}/inference/{case_id}.pdf",
            "data": data,
            "schema": json.loads((data_dir / f"{case_id}.schema.json").read_text()),
        }
        scores = evaluate(evaluator, case_id, data, cases[case_id], citations.get(key))
        capped = any(answer["finish_reason"] == "length" for answer in answered)
        scores.update(
            finish="length" if capped else ("partial" if len(answered) < len(parts) else "stop"),
            input_tokens=sum(answer["input_tokens"] for answer in answered),
            output_tokens=sum(answer["output_tokens"] for answer in answered),
            requests=len(parts),
        )
        table[case_id][variant] = scores
    if args.fallback:
        # The hybrid: a document whose answer was refused (too large) or cut off at the output limit
        # takes the reference pipeline's result instead.
        for case_id, row in table.items():
            reference = args.reference / f"{case_id}.json"
            for name, scores in row.items():
                if scores.get("finish") in ("length",) or str(scores.get("finish", "")).startswith("error"):
                    if reference.exists():
                        row[name] = {**json.loads(reference.read_text())["metrics"], "finish": "fallback"}
    for case_id in config["ids"]:
        reference = args.reference / f"{case_id}.json"
        if reference.exists():
            table[case_id]["reference"] = json.loads(reference.read_text())["metrics"]
    columns = ["reference", *methods]
    print(f"{'document':18}" + "".join(f"{name[:14]:>15}" for name in columns))
    for case_id in config["ids"]:
        row = table[case_id]
        cells = []
        for name in columns:
            value = row.get(name, {}).get("extract_unified_value_f1")
            finish = row.get(name, {}).get("finish", "")
            mark = {"length": "*", "partial": "~", "fallback": "r"}.get(finish, " ")
            missing = (
                "unfinished" if finish == "deadline" else ("error" if finish.startswith("error") else "-")
            )
            cells.append(f"{value * 100:14.1f}{mark}" if value is not None else f"{missing:>15}")
        print(f"{case_id:18}" + "".join(cells))

    def score(case_id, name):
        return table[case_id].get(name, {}).get("extract_unified_value_f1")

    means = []
    for name in columns:
        values = [score(c, name) for c in config["ids"] if score(c, name) is not None]
        means.append(f"{100 * sum(values) / len(values):15.1f}" if values else f"{'-':>15}")
    print(f"{'mean (answered)':18}" + "".join(means))
    # The fair comparison: documents every method answered.
    answered = [name for name in columns if any(score(c, name) is not None for c in config["ids"])]
    common = [c for c in config["ids"] if all(score(c, name) is not None for name in answered)]
    if common:
        line = "".join(
            f"{100 * sum(score(c, name) for c in common) / len(common):15.1f}"
            if name in answered
            else f"{'-':>15}"
            for name in columns
        )
        print(f"{f'mean, {len(common)} common':18}" + line)
    print("(* = an answer hit the output-token limit; ~ = some page batches unanswered; r = reference used)")
    tokens = {
        name: sum(table[c][name].get("output_tokens", 0) for c in config["ids"] if name in table[c])
        for name in methods
    }
    print("output tokens per method:", tokens)
    (out / ("scores-conformed.json" if args.conform else "scores.json")).write_text(
        json.dumps(table, indent=1)
    )
    if args.export:
        (out / "final-answers.json").write_text(json.dumps(list(exported.values())))
    if args.evidence:
        for name in methods:
            rows = [
                table[c][name]
                for c in config["ids"]
                if table[c].get(name, {}).get("extract_unified_grounded_f1") is not None
            ]
            if rows:
                grounded = sum(row["extract_unified_grounded_f1"] for row in rows) / len(rows)
                page = sum(row["extract_unified_page_f1"] for row in rows) / len(rows)
                print(
                    f"{name}: grounded F1 {100 * grounded:.1f}, page F1 {100 * page:.1f} ({len(rows)} docs)"
                )


if __name__ == "__main__":
    main()
