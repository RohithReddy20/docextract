"""Score authored value trials offline, including every missing/failed case."""

import argparse
from collections import Counter
from pathlib import Path

from docextract.contracts import save_json
from docextract.interpretation import strict_json


def canonical(value):
    if isinstance(value, dict):
        return ("object", tuple((key, canonical(child)) for key, child in sorted(value.items())))
    if isinstance(value, list):
        if value and all(isinstance(child, dict) for child in value):
            return ("records", tuple(sorted(Counter(canonical(child) for child in value).items(), key=repr)))
        return ("array", tuple(canonical(child) for child in value))
    # bool and zero must not compare equal; JSON integer and numeric 1.0 may compare equally.
    kind = (
        "number" if isinstance(value, (int, float)) and not isinstance(value, bool) else type(value).__name__
    )
    return kind, value


def top_level_records(data):
    records = Counter()

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, key))
        elif isinstance(value, list) and all(isinstance(child, dict) for child in value):
            records.update((path, canonical(child)) for child in value)
            # A record includes its complete nested subtree, preserving parent associations.

    visit(data, ())
    return records


def score(answers: Path, results: Path):
    truth = strict_json(answers.read_text())
    rows = []
    for case in truth["cases"]:
        case_id = case["id"]
        if not case_id.isalnum():
            raise ValueError("Invalid case identity")
        paths = [results / case_id / name for name in ("value-result.json", "result.json")]
        existing = [path for path in paths if path.exists()]
        data = None
        valid, success = False, False
        invalid_result = False
        if existing:
            try:
                if len(existing) > 1:
                    raise ValueError("Score one trial mode at a time")
                result = strict_json(existing[0].read_text())
                if result["contract_version"] == "value-trial/1":
                    success = result["error"] is None and result["generation"]["finish_reason"] == "stop"
                elif result["contract_version"] == "extraction-result/1":
                    success = result["completion"]["execution_complete"]
                else:
                    raise ValueError("Unknown trial result contract")
                data = result.get("data") if success else None
                if data is not None and not isinstance(data, dict):
                    raise ValueError("Trial result must have an object root")
                valid = bool(result.get("validation") and result["validation"]["valid"])
            except (ValueError, KeyError, TypeError):
                data, valid, success, invalid_result = None, False, False, True
        expected = top_level_records(case["expected"])
        predicted = top_level_records(data or {})
        matched = sum((expected & predicted).values())
        rows.append(
            {
                "id": case_id,
                "family": case["family"],
                "split": case["split"],
                "success": success,
                "schema_valid": valid,
                "invalid_saved_result": invalid_result,
                "exact_document": success and canonical(data) == canonical(case["expected"]),
                "expected_complete_subtree_records": sum(expected.values()),
                "predicted_complete_subtree_records": sum(predicted.values()),
                "correct_complete_subtree_records": matched,
            }
        )
    groups = {}
    families = sorted({row["family"] for row in rows})
    for name, cases in (
        ("all", rows),
        ("development", [row for row in rows if row["split"] == "development"]),
        ("holdout", [row for row in rows if row["split"] == "holdout"]),
        *((f"family:{family}", [row for row in rows if row["family"] == family]) for family in families),
    ):
        groups[name] = {
            "expected_documents": len(cases),
            "successful_documents": sum(row["success"] for row in cases),
            "schema_valid_documents": sum(row["schema_valid"] for row in cases),
            "exact_documents": sum(row["exact_document"] for row in cases),
            **{
                key: sum(row[key] for row in cases)
                for key in (
                    "expected_complete_subtree_records",
                    "predicted_complete_subtree_records",
                    "correct_complete_subtree_records",
                )
            },
        }
    return {
        "metric": "authored exact documents and complete top-level record subtrees; not ExtractBench F1",
        "grounding_scored": False,
        "groups": groups,
        "cases": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, score(args.answers, args.results))
