"""Summarize vLLM periodic stats lines from a Kaggle kernel log (JSON list of output records).

Reports per engine process: prompt and generation throughput, running/waiting requests, KV-cache
use and prefix-cache hit rate, as medians and maxima over the logged intervals.
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

STATS = re.compile(
    r"Avg prompt throughput: (?P<prompt>[\d.]+) tokens/s, Avg generation throughput: (?P<generation>[\d.]+)"
    r" tokens/s, Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, GPU KV cache usage: "
    r"(?P<kv>[\d.]+)%(?:, Prefix cache hit rate: (?P<hit>[\d.]+)%)?"
)
PROCESS = re.compile(r"pid=(\d+)")


def summarize(text: str) -> dict:
    series = defaultdict(lambda: defaultdict(list))
    for line in text.splitlines():
        match = STATS.search(line)
        if not match:
            continue
        # Lines without a process id come from front-end processes and may interleave between runs.
        process = PROCESS.search(line)
        pid = process.group(1) if process else "front-end"
        for key in ("prompt", "generation", "running", "waiting", "kv", "hit"):
            if match.group(key) is not None:
                series[pid][key].append(float(match.group(key)))
    report = {}
    for pid, values in series.items():
        report[pid] = {"intervals": len(values["prompt"])}
        for key, numbers in values.items():
            report[pid][key] = {"median": round(statistics.median(numbers), 1), "max": max(numbers)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    raw = args.log.read_text(errors="replace")
    try:  # a Kaggle kernel log is a JSON list of output records; a run's engine.log is plain text
        text = "".join(record.get("data", "") for record in json.loads(raw))
    except json.JSONDecodeError:
        text = raw
    print(json.dumps(summarize(text), indent=1))


if __name__ == "__main__":
    main()
