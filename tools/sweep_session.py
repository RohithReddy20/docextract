"""Score assembly options on a finished small session's stored answers, on this machine (no model).

The session's GPU shards are replayed separately (each with its own engine identity and documents)
and scored together with the official evaluator, once per option set.
"""

import argparse
import json
import shutil
from pathlib import Path

from tools.replay_local import bundle_prompt_code, replay
from tools.score_extractbench import score

OPTION_SETS = {
    "base": {},
    "merge-entities": {"merge_entities": True},
    "document-strings": {"document_strings": True},
    "tables": {"complete_tables": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True, help="downloaded session output folder")
    parser.add_argument("--kernel-script", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--shards", nargs="+", default=["q0", "q1"])
    parser.add_argument("--options", type=json.loads, required=True, help="the variant's generation options")
    parser.add_argument("--sets", nargs="+", default=list(OPTION_SETS), choices=list(OPTION_SETS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    code = bundle_prompt_code(args.kernel_script)
    for name in args.sets:
        merged = args.output / name / "results"
        if merged.exists():
            shutil.rmtree(merged)
        merged.mkdir(parents=True)
        missing = 0
        for shard in args.shards:
            run = args.session / shard
            summary = run / args.variant / "summary.json"
            ids = {row["id"] for row in json.loads(summary.read_text())["cases"]}
            options = args.options | OPTION_SETS[name] | {"prompt_code": code}
            if options.pop("page_images", False):
                options |= {"page_images": run / "page-images", "max_window_pages": 4}
            target = args.output / name / shard
            missing += replay(args.packet, run, target, ids, **options)["unanswered_windows"]
            for result in target.glob("*/result.json"):
                (merged / result.parent.name).mkdir()
                shutil.copyfile(result, merged / result.parent.name / "result.json")
        report = score(args.packet / "evaluation", merged, f"sweep_{name}")
        scored = [row for row in report["cases"] if row["result"] == "scored"]
        mean = sum(row["extract_unified_value_f1"] or 0 for row in scored) / len(scored)
        print(
            f"{name:17} value_f1={mean:.4f} over {len(scored)} documents, unanswered windows={missing}",
            flush=True,
        )
        (args.output / name / "scores.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
