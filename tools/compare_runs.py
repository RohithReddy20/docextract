"""Compare scored development runs document by document (official value F1 and its variants).

Each argument is NAME=PATH to a scores JSON written by tools/score_extractbench.py. Prints the per-case
value F1 of every run, the mean per split, and for the first two runs the number of documents each one
wins. Development cases only.
"""

import argparse
import json
from pathlib import Path


def load(argument: str) -> tuple[str, dict]:
    name, path = argument.split("=", 1)
    report = json.loads(Path(path).read_text())
    return name, {row["id"]: row for row in report["cases"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="NAME=scores.json")
    parser.add_argument("--metric", default="extract_unified_value_f1")
    args = parser.parse_args()
    runs = [load(argument) for argument in args.runs]
    # Only documents every run scored are compared.
    shared = set.intersection(*(set(rows) for _, rows in runs))
    ids = sorted(shared, key=lambda case: (runs[0][1][case]["split"], case))
    header = "case              split  " + "  ".join(f"{name[:12]:>12}" for name, _ in runs)
    print(header)
    for case in ids:
        cells = []
        for _, rows in runs:
            value = rows.get(case, {}).get(args.metric)
            cells.append(f"{value:12.3f}" if value is not None else f"{'-':>12}")
        print(f"{case} {runs[0][1][case]['split']:6} " + "  ".join(cells))
    for split in ("short", "medium", "long", "all"):
        chosen = [case for case in ids if split == "all" or runs[0][1][case]["split"] == split]
        if not chosen:
            continue
        means = [
            sum(rows.get(c, {}).get(args.metric) or 0.0 for c in chosen) / len(chosen) for _, rows in runs
        ]
        print(f"mean {split:6} ({len(chosen):3d}) " + "  ".join(f"{mean:12.3f}" for mean in means))
    if len(runs) >= 2:
        (first, a), (second, b) = runs[:2]
        better = sum((a[c].get(args.metric) or 0) > (b[c].get(args.metric) or 0) + 1e-9 for c in ids)
        worse = sum((b[c].get(args.metric) or 0) > (a[c].get(args.metric) or 0) + 1e-9 for c in ids)
        print(f"{first} better on {better}, {second} better on {worse}, tied on {len(ids) - better - worse}")


if __name__ == "__main__":
    main()
