"""Modal spend of this project (apps named docextract*) versus everything else, recent hours.

Reads `modal billing report` (hourly resolution) through the Modal CLI; prints dollars. Credit
balances are not written anywhere in the repository.
"""

import argparse
import json
import subprocess


def spend(start: str = "2 days ago") -> dict[str, float]:
    """Cost per app since `start`, at hourly resolution (complete hours; at most 7 days back)."""
    report = subprocess.run(
        ["modal", "billing", "report", "--start", start, "--resolution", "h", "--json"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    totals: dict[str, float] = {}
    for row in json.loads(report):
        name = row.get("description") or row.get("object_id", "?")
        totals[name] = totals.get(name, 0.0) + float(row.get("cost") or 0)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2 days ago", help="hourly report start (at most 7 days back)")
    args = parser.parse_args()
    totals = spend(args.start)
    mine = {name: cost for name, cost in totals.items() if name.startswith("docextract")}
    for name, cost in sorted(mine.items()):
        print(f"{name:40} ${cost:.3f}")
    others = sum(totals.values()) - sum(mine.values())
    print(f"docextract total: ${sum(mine.values()):.3f}; other apps: ${others:.3f}")


if __name__ == "__main__":
    main()
