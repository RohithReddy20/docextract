"""Score post-processing options on a finished run's stored answers, on this machine, for free.

Each option set re-assembles the stored answers (no model, no PDFs) and scores them with the official
evaluator; the table lists value, grounded and page F1 per option set. Prompt-shaping code must match
the run (pass the run's built Kaggle script), or answers are reported as unanswered.
"""

import argparse
import json
from pathlib import Path

from docextract.contracts import save_json
from tools.replay_local import bundle_prompt_code, replay
from tools.score_extractbench import score

OPTION_SETS = {
    "base": {},
    "omit-nulls": {"absent_nullable": "omit"},
    "root-nulls": {"absent_nullable": "root"},
    "nulls-if-partial": {"nulls_if_partial": True},
    "document-strings": {"document_strings": True},
    "tables": {"complete_tables": True},
    # Only valid on an unrouted run: re-assembles from the subset of answers routing would have asked.
    "routed": {"route_groups": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True, help="exported run (answers, snapshots)")
    parser.add_argument("--kernel-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="", help="prompt variant name inside the run, if any")
    parser.add_argument("--options", nargs="*", default=list(OPTION_SETS), choices=list(OPTION_SETS))
    parser.add_argument(
        "--window", type=json.loads, default={"max_window_chars": 24000, "max_schema_chars": 24000}
    )
    parser.add_argument("--protocol", default="cited", choices=("cited", "values"))
    args = parser.parse_args()
    prompt_code = bundle_prompt_code(args.kernel_script)
    rows = {}
    for name in args.options:
        target = args.output / name
        options = args.window | OPTION_SETS[name] | {"prompt_code": prompt_code}
        if args.protocol != "cited":
            options["protocol"] = args.protocol
        summary = replay(args.packet, args.run, target, **options)
        report = score(args.packet / "evaluation", target, f"sweep_{name}", target / "details")
        save_json(target / "scores.json", report)
        group = report["groups"]["all"]
        rows[name] = {
            "value_f1": round(group["value_f1"], 4),
            "grounded_f1": group["grounded_f1"] and round(group["grounded_f1"], 4),
            "page_f1": group["page_f1"] and round(group["page_f1"], 4),
            "unanswered_windows": summary["unanswered_windows"],
            "failed": len(summary["failed"]),
        }
        print(name, json.dumps(rows[name]), flush=True)
    save_json(args.output / "sweep.json", rows)


if __name__ == "__main__":
    main()
