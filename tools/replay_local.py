"""Re-assemble a remote run's stored answers on a CPU and optionally score them; no model runs.

Inputs are a run exported by the Kaggle runner (stored answers, engine identity, parsed snapshots)
and the exported development packet (schemas, document hashes, scoring files; no PDFs). Prompt and
window code must be unchanged, or stored answers no longer match their windows and are reported
as unanswered.
"""

import argparse
import gzip
import json
import shutil
from pathlib import Path

from docextract.batch import ReplayEngine, run_batch
from docextract.contracts import ParseSnapshot, save_json


class StoredSnapshots:
    """Parser stand-in that returns the snapshot the stored answers were generated from."""

    def __init__(self, directory: Path, hashes: dict[str, str]):
        self.directory, self.hashes = directory, hashes

    def parse(self, document: Path) -> ParseSnapshot:
        path = self.directory / f"{self.hashes[document.stem]}.json"
        if path.exists():
            return ParseSnapshot.model_validate_json(path.read_text())
        with gzip.open(path.with_name(path.name + ".gz"), "rt") as source:
            return ParseSnapshot.model_validate_json(source.read())


def bundle_prompt_code(kernel_script: Path) -> dict[str, str]:
    """Hashes of the prompt-shaping sources inside a built Kaggle script's embedded bundle."""
    import base64
    import hashlib
    import io
    import re
    import tarfile

    payload = re.search(r'^PAYLOAD = "([A-Za-z0-9+/=]+)"$', kernel_script.read_text(), re.M).group(1)
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)), mode="r:gz") as bundle:
        for name in ("interpretation.py", "partition.py"):
            member = bundle.extractfile(f"docextract/src/docextract/{name}")
            hashes[name] = hashlib.sha256(member.read()).hexdigest()
    return hashes


def replay(packet: Path, run: Path, output: Path, ids: set[str] | None = None, **options) -> dict:
    documents = json.loads((packet / "inference/documents.json").read_text())["cases"]
    documents = [case for case in documents if ids is None or case["id"] in ids]
    cases = [
        (
            case["id"],
            packet / "inference" / f"{case['id']}.pdf",
            json.loads((packet / "inference" / case["schema"]).read_text()),
        )
        for case in documents
    ]
    if output.exists():
        # Only a previous replay's directory is ever replaced.
        if any(output.iterdir()) and not (output / "replay.json").exists():
            raise ValueError(f"{output} exists and is not a replay directory")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copyfile(run / "generations.jsonl", output / "generations.jsonl")
    engine = ReplayEngine(json.loads((run / "runtime.json").read_text())["identity"])
    parser = StoredSnapshots(run / "snapshots", {case["id"]: case["pdf_sha256"] for case in documents})
    outcomes = run_batch(cases, root=output, engine=engine, parser=parser, **options)
    failures = {case_id: outcome for case_id, outcome in outcomes.items() if isinstance(outcome, str)}
    summary = {"cases": len(cases), "failed": failures, "unanswered_windows": engine.missing}
    save_json(output / "replay.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", help="replay only these case ids")
    parser.add_argument("--max-window-chars", type=int, default=24000)
    parser.add_argument("--max-schema-chars", type=int, default=24000)
    parser.add_argument("--complete-tables", action="store_true")
    parser.add_argument("--absent-nullable", choices=("null", "omit", "root"), default="null")
    parser.add_argument("--document-strings", action="store_true", help="document text for re-cased strings")
    parser.add_argument(
        "--nulls-if-partial", action="store_true", help="null absent nullable fields even if windows failed"
    )
    parser.add_argument("--page-images", action="store_true", help="the run showed page images")
    parser.add_argument("--route-groups", action="store_true", help="ask windows only about routed groups")
    parser.add_argument("--prompt-budget-chars", type=int)
    parser.add_argument("--factor-notes", action="store_true", help="the run shortened repeated guidance")
    parser.add_argument("--merge-entities", action="store_true", help="merge one entity seen in two sections")
    parser.add_argument("--protocol", choices=("cited", "values"), default="cited")
    parser.add_argument("--max-window-pages", type=int, default=4)
    parser.add_argument("--score", action="store_true", help="score with the official evaluator")
    parser.add_argument(
        "--kernel-script", type=Path, help="the run's built Kaggle script, for its prompt-code hashes"
    )
    args = parser.parse_args()
    extra = {"prompt_code": bundle_prompt_code(args.kernel_script)} if args.kernel_script else {}
    if args.page_images:
        extra |= {"page_images": args.run / "page-images", "max_window_pages": args.max_window_pages}
    if args.route_groups:
        extra["route_groups"] = True
    if args.prompt_budget_chars:
        extra["prompt_budget_chars"] = args.prompt_budget_chars
    if args.factor_notes:
        extra["factor_notes"] = True
    if args.merge_entities:
        extra["merge_entities"] = True
    if args.protocol != "cited":
        extra["protocol"] = args.protocol
    summary = replay(
        args.packet,
        args.run,
        args.output,
        set(args.ids) if args.ids else None,
        max_window_chars=args.max_window_chars,
        max_schema_chars=args.max_schema_chars,
        complete_tables=args.complete_tables,
        absent_nullable=args.absent_nullable,
        document_strings=args.document_strings,
        nulls_if_partial=args.nulls_if_partial,
        **extra,
    )
    print(json.dumps(summary))
    if args.score:
        from tools.score_extractbench import score

        report = score(args.packet / "evaluation", args.output, "docextract_replay", args.output / "details")
        save_json(args.output / "scores.json", report)
        print(json.dumps(report["groups"], indent=1))


if __name__ == "__main__":
    main()
