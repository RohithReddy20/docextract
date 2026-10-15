"""Score saved docextract results with the official ExtractBench evaluator, in its own process.

Every selected case stays in the population: a missing or unreadable result is scored as an empty
prediction. Means are unweighted over documents, as in the benchmark report.
"""

import argparse
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from extract_bench.evaluation.evaluators.extract import ExtractEvaluator
from extract_bench.evaluation.score_case import load_extract_test_case
from extract_bench.schemas.extract_output import ExtractOutput
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult

from docextract.contracts import ExtractionResult, save_json
from docextract.integrations.extractbench import project_result
from docextract.schema import parts

HEADLINE = (
    "extract_unified_value_f1",
    "extract_unified_value_precision",
    "extract_unified_value_recall",
    "extract_unified_grounded_f1",
    "extract_unified_page_f1",
)


def without_projected_nulls(result: ExtractionResult) -> dict:
    """The saved data minus fields that were only projected as null because nothing was found."""
    data = deepcopy(result.data)
    for field in result.field_evidence:
        if field.transform != "null_projection":
            continue
        *path, last = parts(field.pointer)
        target = data
        for part in path:
            target = target[int(part)] if isinstance(target, list) else target.get(part)
            if target is None:
                break
        if isinstance(target, dict) and target.get(last, ...) is None:
            del target[last]
    return data


def load_case(test_json: Path):
    """Scoring reads only the test file; the document beside it may be absent off the worker."""
    try:
        return load_extract_test_case(test_json)
    except ValueError:
        stem = test_json.name.removesuffix(".test.json")
        return load_extract_test_case(test_json, source_file=test_json.with_name(stem + ".pdf"))


def score(
    evaluation: Path, results: Path, pipeline: str, details: Path | None = None, seen: set[str] | None = None
) -> dict:
    index = json.loads((evaluation / "index.json").read_text())
    evaluator = ExtractEvaluator()
    now = datetime.now(UTC)
    rows = []
    for case in index["cases"]:
        test_case = load_case(evaluation / case["test_json"])
        row = {"id": case["id"], "split": case["split"], "result": "missing"}
        output = ExtractOutput(
            example_id=case["id"], pipeline_name=pipeline, extracted_data={}, field_citations=[]
        )
        path = results / case["id"] / "result.json"
        if path.exists():
            try:
                result = ExtractionResult.model_validate_json(path.read_text())
                output = project_result(result, example_id=case["id"], pipeline_name=pipeline)
                row.update(
                    result="scored",
                    status=result.completion.status,
                    schema_valid=result.validation.valid,
                    unresolved=len(result.unresolved),
                    rejected=result.coverage.rejected_proposals,
                    pages=len(result.snapshot.pages),
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    model_calls=result.usage.generated_attempts,
                )
            except (ValueError, KeyError) as error:
                row["result"] = f"unreadable: {type(error).__name__}"
        inference = InferenceResult(
            request=InferenceRequest(
                example_id=case["id"],
                source_file_path=str(test_case.source_file_path)
                if hasattr(test_case, "source_file_path")
                else case["id"],
                product_type="extract",
            ),
            pipeline_name=pipeline,
            product_type="extract",
            raw_output={},
            output=output,
            started_at=now,
            completed_at=now,
            latency_in_ms=0,
        )
        evaluated = evaluator.evaluate(inference, test_case)
        metrics = {metric.metric_name: metric.value for metric in evaluated.metrics}
        row.update({name: metrics.get(name) for name in HEADLINE})
        if row["result"] == "scored":
            # Alternative projection of the same saved result: absent nullable fields omitted, not null.
            omitted = output.model_copy(update={"extracted_data": without_projected_nulls(result)})
            variant = evaluator.evaluate(inference.model_copy(update={"output": omitted}), test_case)
            values = {metric.metric_name: metric.value for metric in variant.metrics}
            row["value_f1_nulls_omitted"] = values.get("extract_unified_value_f1")
        rows.append(row)
        if details is not None:
            save_json(
                details / f"{case['id']}.json",
                {
                    "id": case["id"],
                    "predicted": output.extracted_data,
                    "expected": test_case.expected_output,
                    "metrics": {name: metrics.get(name) for name in HEADLINE},
                },
            )

    def mean(name, cases):
        values = [row[name] for row in cases if row[name] is not None]
        return sum(values) / len(values) if values else None

    groups = {}
    selections = {
        split: (lambda row, split=split: split == "all" or row["split"] == split)
        for split in ("all", "short", "medium", "long")
    }
    if seen:
        # Documents examined during development are reported apart from the untouched remainder.
        selections["seen_in_development"] = lambda row: row["id"] in seen
        selections["unseen"] = lambda row: row["id"] not in seen
    for split, chosen in selections.items():
        cases = [row for row in rows if chosen(row)]
        if cases:
            # A missing value-F1 (unscorable shape) counts as zero so failures stay in the mean.
            groups[split] = {
                "documents": len(cases),
                "scored_results": sum(row["result"] == "scored" for row in cases),
                "value_f1": sum(row["extract_unified_value_f1"] or 0.0 for row in cases) / len(cases),
                "value_f1_nulls_omitted": sum(
                    (row.get("value_f1_nulls_omitted") if row["result"] == "scored" else 0.0) or 0.0
                    for row in cases
                )
                / len(cases),
                "grounded_f1": mean("extract_unified_grounded_f1", cases),
                "page_f1": mean("extract_unified_page_f1", cases),
            }
    return {"pipeline": pipeline, "evaluator": "ExtractEvaluator (official)", "groups": groups, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pipeline", default="docextract_trial")
    parser.add_argument("--details", type=Path, help="per-case predicted/expected dumps for dev analysis")
    parser.add_argument("--seen", type=Path, help="JSON list of case ids examined during development")
    args = parser.parse_args()
    seen = set(json.loads(args.seen.read_text())) if args.seen else None
    report = score(args.evaluation, args.results, args.pipeline, args.details, seen)
    save_json(args.output, report)
    print(json.dumps(report["groups"]))


if __name__ == "__main__":
    main()
