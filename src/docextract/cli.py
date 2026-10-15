from __future__ import annotations

import argparse
import json
from pathlib import Path

from docextract.contracts import ExtractionResult, save_json
from docextract.engine import extract
from docextract.planning import BindingPlan
from docextract.schema import Schema, UnsupportedSchema


def main() -> None:
    parser = argparse.ArgumentParser(prog="docextract", description="Evidence-first document extraction")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="create and extract an authored two-page PDF (dev dependency)")
    demo.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    run = commands.add_parser("extract", help="extract native labels and declared tables; no model inference")
    run.add_argument("pdf", type=Path)
    run.add_argument("--schema", required=True, type=Path)
    run.add_argument("--bindings", required=True, type=Path)
    run.add_argument("--workspace", type=Path, required=True)
    run.add_argument("--apply-defaults", action="store_true")
    learned = commands.add_parser("extract-model", help="Kaggle-only learned source binding; no manual plan")
    learned.add_argument("pdf", type=Path)
    learned.add_argument("--schema", required=True, type=Path)
    learned.add_argument("--workspace", required=True, type=Path)
    learned.add_argument("--arm", choices=("a", "b"), required=True)
    learned.add_argument("--model-manifest", required=True, type=Path)
    learned.add_argument("--device", type=int, default=0)
    support = commands.add_parser("schema-support")
    support.add_argument("schema", type=Path)
    project = commands.add_parser("project-extractbench", help="offline projection; requires benchmark extra")
    project.add_argument("result", type=Path)
    project.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract-model":
        from docextract.learned import extract_learned
        from docextract.remote import KaggleTransformersBackend, RemoteOnlyError, require_kaggle

        try:
            require_kaggle()
            schema = json.loads(args.schema.read_text())
            Schema(schema).require_supported()
            backend = KaggleTransformersBackend(args.arm, manifest=args.model_manifest, device=args.device)
            result = extract_learned(args.pdf, schema, workspace=args.workspace, backend=backend)
        except (RemoteOnlyError, UnsupportedSchema) as error:
            parser.exit(2, str(error) + "\n")
        print(
            json.dumps(
                {
                    "result": str(args.workspace / "result.json"),
                    "schema_valid": result.validation.valid,
                    "status": result.completion.status,
                }
            )
        )
        return
    if args.command == "schema-support":
        print(Schema(json.loads(args.schema.read_text())).support.model_dump_json(indent=2))
        return
    if args.command == "project-extractbench":
        from docextract.integrations.extractbench import project_result

        result = ExtractionResult.model_validate_json(args.result.read_text())
        save_json(args.output, project_result(result))
        print(args.output)
        return
    try:
        if args.command == "demo":
            from docextract.example import EXPECTED, create_example

            pdf, schema, plan = create_example(args.output)
            result = extract(pdf, schema, workspace=args.output / "run", plan=plan, apply_defaults=True)
            if result.data != EXPECTED or not result.validation.valid:
                raise RuntimeError("Authored example failed expected-value/schema verification")
            destination = args.output / "run/result.json"
        else:
            result = extract(
                args.pdf,
                json.loads(args.schema.read_text()),
                workspace=args.workspace,
                plan=BindingPlan.model_validate_json(args.bindings.read_text()),
                apply_defaults=args.apply_defaults,
            )
            destination = args.workspace / "result.json"
    except UnsupportedSchema as error:
        parser.exit(2, error.report.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "result": str(destination),
                "schema_valid": result.validation.valid,
                "execution_complete": result.completion.execution_complete,
                "status": result.completion.status,
                "source_records": len(result.source_records),
                "unresolved": len(result.unresolved),
                "usage": result.usage.model_dump(),
            },
            indent=2,
        )
    )
