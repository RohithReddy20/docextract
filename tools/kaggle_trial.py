"""Remote-only supervisor. It never submits a notebook or starts cloud resources."""

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

from docextract.contracts import save_json
from docextract.remote import GenerationOptions, KaggleTransformersBackend, require_kaggle


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Case path escapes the inference packet")
    return path


def worker(args):
    from docextract.cited import extract_cited
    from docextract.learned import extract_learned
    from docextract.trial import run_value_trial

    packet = json.loads(args.cases.read_text())
    if set(packet) != {"version", "cases"} or packet["version"] != "inference-cases/1":
        raise ValueError("Expected an inference-only case manifest")
    for case in packet["cases"]:
        if set(case) != {"id", "pdf", "schema"}:
            raise ValueError("Inference manifest must not contain answers, rules or binding plans")
        if not case["id"].isalnum():
            raise ValueError("Case IDs must be alphanumeric")
    if len({case["id"] for case in packet["cases"]}) != len(packet["cases"]):
        raise ValueError("Duplicate case IDs")
    summaries = [
        {"id": case["id"], "success": False, "schema_valid": False, "error": "not_started"}
        for case in packet["cases"]
    ]
    save_json(args.output / "summary.json", {"mode": args.mode, "cases": summaries})
    backend = KaggleTransformersBackend(
        args.arm,
        manifest=args.model_manifest,
        device=args.device,
        wrapper=args.wrapper,
        options=GenerationOptions(max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens),
        offline=args.cache_only,
        cache_root=args.cache_root,
    )
    save_json(args.output / "runtime.json", {"identity": backend.identity, "preflight": backend.preflight})
    for index, case in enumerate(packet["cases"]):
        row = {"id": case["id"], "success": False, "schema_valid": False}
        workspace = args.output / case["id"]
        try:
            schema = json.loads(contained(args.cases.parent, case["schema"]).read_text())
            pdf = contained(args.cases.parent, case["pdf"])
            if args.mode == "values":
                result = run_value_trial(pdf, schema, workspace=workspace, backend=backend)
                row.update(
                    success=result.error is None,
                    schema_valid=bool(result.validation and result.validation.valid),
                    result=f"{case['id']}/value-result.json",
                    error="invalid_or_incomplete_output" if result.error else None,
                )
            else:
                extract = extract_cited if args.mode == "cited" else extract_learned
                windowing = {"max_window_chars": args.max_window_chars} if args.max_window_chars else {}
                result = extract(pdf, schema, workspace=workspace, backend=backend, **windowing)
                row.update(
                    success=True,
                    schema_valid=result.validation.valid,
                    result=f"{case['id']}/result.json",
                    status=result.completion.status,
                    rejected_proposals=result.coverage.rejected_proposals,
                    unresolved=len(result.unresolved),
                )
        except Exception as error:
            # Raw model/source content stays in the private journal, never in summary errors.
            row["error"] = type(error).__name__
            save_json(
                workspace / "diagnostic.json",
                {"error_type": type(error).__name__, "message": str(error)[:4000]},
            )
            if type(error).__name__ in ("OutOfMemoryError", "AcceleratorError"):
                summaries[index] = row
                save_json(args.output / "summary.json", {"mode": args.mode, "cases": summaries})
                raise
        summaries[index] = row
        journal = workspace / ("cited.sqlite" if args.mode == "cited" else "interpretation.sqlite")
        if journal.exists():
            with sqlite3.connect(f"file:{journal}?mode=ro", uri=True) as database:
                attempts = [
                    {"task_id": task_id, "state": state, "generation": json.loads(raw) if raw else None}
                    for task_id, state, raw in database.execute("SELECT task_id,state,raw FROM tasks")
                ]
            save_json(workspace / "proposals.json", {"private_authored_diagnostics": True, "tasks": attempts})
        save_json(args.output / "summary.json", {"mode": args.mode, "cases": summaries})
        save_json(args.output / "telemetry.json", {"tasks": backend.telemetry})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--arm", choices=("a", "b"), required=True)
    parser.add_argument("--mode", choices=("values", "bindings", "cited"), default="values")
    parser.add_argument("--wrapper", choices=("text", "multimodal"), default="text")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-window-chars", type=int, help="evidence window size; default per protocol")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--cache-only", action="store_true", help="fail if the pinned remote cache is incomplete"
    )
    parser.add_argument("--cache-root", type=Path, default=Path("/kaggle/temp/docextract-cache"))
    args = parser.parse_args()
    require_kaggle()
    if not 0 < args.timeout_seconds <= 3600:
        parser.error("Initial free-development session limit is at most one hour")
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        "Estimated total cash cost: $0 with available free Kaggle quota; no paid fallback. "
        f"Maximum worker time: {args.timeout_seconds} seconds.",
        flush=True,
    )
    process = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], start_new_session=True
    )
    try:
        code = process.wait(timeout=args.timeout_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        save_json(
            args.output / "interrupted.json",
            {
                "execution_complete": False,
                "reason": "worker stopped",
                "all_expected_cases_require_accounting": True,
            },
        )
        return 124
    return code


if __name__ == "__main__":
    raise SystemExit(main())
