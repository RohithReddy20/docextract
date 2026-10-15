"""Inspect schemas in an already pinned remote snapshot; never persist answers or field rules."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from jsonschema import SchemaError

from docextract.contracts import digest, save_json
from docextract.schema import Schema

REVISION = "51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc"


def inventory(snapshot: Path) -> dict:
    manifest = json.loads((snapshot.parent / "dataset-manifest.json").read_text())
    if manifest.get("revision") != REVISION or manifest.get("mode") != "downloaded_remote_snapshot":
        raise ValueError("Require a downloaded, pinned manifest from prepare_benchmark.py")
    hashes = {entry["path"]: entry["sha256"] for entry in manifest["files"]}
    schemas = {}
    splits = Counter()
    supported_cases = Counter()
    dialects = Counter()
    issues = Counter()
    for split in ("short", "medium", "long"):
        checksum = hashlib.sha256()
        with (snapshot / f"{split}.jsonl").open("rb") as source:
            for line in source:
                checksum.update(line)
                row = json.loads(line)
                schema = row["data_schema"]
                schema = json.loads(schema) if isinstance(schema, str) else schema
                identity = digest(schema)
                try:
                    if not isinstance(schema, dict):
                        raise ValueError("Schema is not an object")
                    report = Schema(schema).support
                    summary = {"supported": report.supported, "issues": list(report.issues)}
                except (SchemaError, ValueError, TypeError, RecursionError) as error:
                    summary = {
                        "supported": False,
                        "issues": [f"invalid_or_uninterpretable_schema: {type(error).__name__}"],
                    }
                summary["declared_dialect"] = schema.get("$schema") if isinstance(schema, dict) else None
                dialects[str(summary["declared_dialect"])] += 1
                summary["cases"] = schemas.get(identity, {}).get("cases", 0) + 1
                schemas[identity] = summary
                issues.update(summary["issues"])
                splits[split] += 1
                supported_cases[split] += int(summary["supported"])
        if checksum.hexdigest() != hashes.get(f"{split}.jsonl"):
            raise ValueError(f"Snapshot hash mismatch: {split}")
    return {
        "dataset_revision": REVISION,
        "case_counts": dict(splits),
        "supported_case_counts": dict(supported_cases),
        "declared_dialect_case_counts": dict(dialects),
        "unique_schemas": len(schemas),
        "supported_unique_schemas": sum(item["supported"] for item in schemas.values()),
        "schemas": schemas,
        "issue_counts": dict(issues),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    save_json(args.output, inventory(args.snapshot))
    print(args.output)


if __name__ == "__main__":
    main()
