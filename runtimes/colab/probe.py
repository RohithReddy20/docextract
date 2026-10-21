"""Run a probe's requests on a Colab GPU (the notebook next to this file calls it).

Same requests and engine as runtimes/modal/probe.py; PDFs come straight from the pinned ExtractBench
dataset on Hugging Face (only the PDFs the requests name, never the answer files) and the model from
its pinned, hash-verified revision. Answers are appended to `<out>/answers.jsonl` as they finish, so a
disconnected session loses nothing: run it again with the same `--out` and it asks only what is left.
`<out>/answers.json` is the file tools/score_probe.py reads.

    python runtimes/colab/probe.py --config manifests/probes/<name>.json \\
        --out /content/drive/MyDrive/docextract/<name>
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "runtimes"), str(ROOT / "tools")]

from probe_batch import answered, load_engine, request_key, run  # noqa: E402

# Colab compute units per hour, as measured by third parties (Google publishes no table); the
# notebook's Resources panel shows the live rate.
UNITS_PER_HOUR = {"T4": 1.2, "L4": 1.7, "A100": 7.5, "H100": 15.0}


def interrupt(*_):
    raise KeyboardInterrupt


def units_per_hour(gpu: str) -> float:
    return next((rate for name, rate in UNITS_PER_HOUR.items() if name in gpu), 15.0)


def dataset_file(name: str, cache: Path) -> Path:
    """One file of the ExtractBench dataset revision pinned in manifests/references.json."""
    from huggingface_hub import hf_hub_download

    dataset = json.loads((ROOT / "manifests" / "references.json").read_text())["dataset"]
    return Path(
        hf_hub_download(
            dataset["repository"],
            name,
            repo_type="dataset",
            revision=dataset["revision"],
            local_dir=cache,
            token=False,
        )
    )


def jobs_from_config(config: dict, cache: Path) -> dict:
    """A probe's requests built here from its configuration, as tools/probe_jobs.py builds them.

    Each case's schema and PDF path are read from the dataset's split files; only those two fields
    are kept (the rows also hold the expected outputs, which are not used here).
    """
    from probe_jobs import build

    from docextract.contracts import digest

    wanted, sources = set(config["ids"]), {}
    data = cache / "data"
    data.mkdir(parents=True, exist_ok=True)
    for split in ("short", "medium", "long"):
        with dataset_file(f"{split}.jsonl", cache).open(encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line) if line.strip() else None
                case_id = digest(row["id"])[:16] if row else None
                if case_id in wanted:
                    (data / f"{case_id}.schema.json").write_text(row["data_schema"])
                    sources[case_id] = row["pdf"]
    if wanted - sources.keys():
        raise SystemExit(f"Not in the dataset: {sorted(wanted - sources.keys())}")
    jobs = build(config, data)
    for job in jobs:
        job["dataset_pdf"] = sources[job["id"]]
    return {"engine": config["engine"], "limit_minutes": config.get("limit_minutes", 15), "jobs": jobs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path, help="a probe configuration (manifests/probes/)")
    source.add_argument("--jobs", type=Path, help="written by tools/probe_jobs.py")
    parser.add_argument("--out", type=Path, required=True, help="where answers go (in the notebook, Drive)")
    parser.add_argument("--minutes", type=float, help="stop after this long (default: the probe's limit)")
    parser.add_argument(
        "--engine", type=json.loads, default={}, help='engine options to override, e.g. {"max_num_seqs": 64}'
    )
    parser.add_argument("--cache", type=Path, default=Path("/content/cache"))
    args = parser.parse_args()

    import torch

    from docextract.remote import pinned_checkpoint

    started = time.time()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.config:
        batch = jobs_from_config(json.loads(args.config.read_text()), args.cache / "extractbench")
    else:
        batch = json.loads(args.jobs.read_text())
    jobs, engine = batch["jobs"], {**batch["engine"], **args.engine}
    missing = sorted({job["id"] for job in jobs if not job.get("dataset_pdf")})
    if missing:
        raise SystemExit(f"No dataset PDF for {missing}: rebuild jobs.json after tools/probe_data.sh")
    minutes = args.minutes or batch["limit_minutes"]
    gpu = torch.cuda.get_device_name(0)
    rate = units_per_hour(gpu)
    print(f"engine: {json.dumps(engine)}")
    print(f"{gpu}: at most {minutes:.0f} minutes, about {minutes / 60 * rate:.1f} compute units")

    args.out.mkdir(parents=True, exist_ok=True)
    sink = args.out / "answers.jsonl"
    skip = answered(sink)
    print(f"{len(skip)} requests answered by earlier runs are skipped")
    names = {job["dataset_pdf"] for job in jobs}
    pdfs = {name: dataset_file(name, args.cache / "extractbench") for name in names}
    _, _, checkpoint = pinned_checkpoint(
        engine.get("arm", "a"), ROOT / "manifests" / "model-research.json", args.cache
    )
    llm = load_engine(checkpoint, engine)
    loaded = time.time()
    print(f"model loaded in {loaded - started:.0f} s")
    # The notebook's stop button ends this process with SIGTERM; what finished is kept either way.
    signal.signal(signal.SIGTERM, interrupt)
    try:
        outputs = run(
            llm,
            jobs,
            engine,
            lambda job: pdfs[job["dataset_pdf"]],
            sink,
            started + minutes * 60,
            loaded=loaded,
            skip=skip,
        )
    except KeyboardInterrupt:
        print("stopped early")
        outputs = []

    # Every finished answer of every run, plus what this run left unfinished.
    rows = {}
    for line in sink.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[request_key(row)] = row
    for row in outputs:
        rows.setdefault(request_key(row), row)
    seconds = time.time() - started
    (args.out / "answers.json").write_text(
        json.dumps(
            {
                "outputs": list(rows.values()),
                "gpu": gpu,
                "load_seconds": round(loaded - started),
                "seconds": round(seconds),
            },
            indent=1,
        )
    )
    unfinished = sum(1 for row in rows.values() if row["finish_reason"] == "deadline")
    print(
        f"{len(rows) - unfinished} answered, {unfinished} unfinished, in {seconds / 60:.1f} minutes "
        f"(about {seconds / 3600 * rate:.1f} compute units); answers in {args.out}"
    )


if __name__ == "__main__":
    main()
