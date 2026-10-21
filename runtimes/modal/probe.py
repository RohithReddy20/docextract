"""Small extraction probes on one Modal L4: a few documents, several prompt methods, one model load.

Each method is a request recipe. `baseline` reproduces ExtractBench's one-shot recipe for self-hosted
vLLM models (the leaderboard's Qwen3.5-4B entry): every page rendered at 150 DPI, the full schema
pretty-printed, JSON-object output, temperature 0. Other methods change one thing at a time.

Requests are built locally with ExtractBench's own prompt helpers (`tools/probe_jobs.py`); this
entrypoint states the worst-case cost, runs them in one GPU call and saves the raw answers next to the
request file. The GPU worker reads only PDFs from the work volume; answers are scored locally
(`tools/score_probe.py`), so expected outputs never reach it.

    uv run python tools/probe_jobs.py manifests/probes/<name>.json
    modal run runtimes/modal/probe.py::main --jobs artifacts/probes/<name>/jobs.json --confirm-usd <estimate>
    modal run runtimes/modal/probe.py::evidence --answers artifacts/probes/<name>/final-answers.json
    uv run python tools/score_probe.py manifests/probes/<name>.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

SRC = "/opt/docextract"
WORK_DIR = "/work"
L4_PER_HOUR = 1.31  # L4 with 4 cores and 40 GiB, Modal list prices (see final_run.py)
work = modal.Volume.from_name("docextract-work")
app = modal.App("docextract-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "vllm==0.30.0",
        "python-doctr==1.1.0",
        "jsonschema>=4.23.0,<5",
        "pydantic>=2.11.9,<3",
        "pypdfium2==4.30.0",
    )
    .env(
        {
            "PYTHONPATH": f"{SRC}/src",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HOME": f"{WORK_DIR}/model-cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "TRITON_CACHE_DIR": f"{WORK_DIR}/compile-cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{WORK_DIR}/compile-cache/inductor",
            "VLLM_CACHE_ROOT": f"{WORK_DIR}/compile-cache/vllm",
        }
    )
)
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    image = (
        image.add_local_dir(root / "src", f"{SRC}/src")
        .add_local_dir(root / "manifests", f"{SRC}/manifests")
        .add_local_file(root / "runtimes" / "probe_batch.py", f"{SRC}/probe_batch.py")
    )


@app.function(image=image, gpu="L4", cpu=4, memory=40 * 1024, volumes={WORK_DIR: work})
def generate(jobs: list[dict], engine: dict, name: str, seconds: float) -> dict:
    """Answer every job in one continuous batch; each answer is saved as it finishes.

    Prompts are rendered per job (its own chat-template options) and run together, so a slow answer
    holds up nothing else. Answers stream to `runs/probes/<name>/answers.jsonl` in the work volume, and
    the batch stops `seconds` after this container starts (time spent waiting for a GPU does not
    count), returning what finished.
    """
    deadline = time.time() + seconds
    import sys

    sys.path.insert(0, f"{SRC}/src")
    sys.path.insert(0, SRC)
    from docextract.remote import pinned_checkpoint

    started = time.time()
    _, _, checkpoint = pinned_checkpoint(
        engine.get("arm", "a"),
        Path(SRC) / "manifests/model-research.json",
        Path(WORK_DIR) / "model-cache",
        offline=True,
    )
    from probe_batch import load_engine, run

    llm = load_engine(checkpoint, engine)
    loaded = time.time()

    sink = Path(WORK_DIR) / "runs" / "probes" / name / "answers.jsonl"
    sink.parent.mkdir(parents=True, exist_ok=True)
    sink.write_text("")
    outputs = run(
        llm,
        jobs,
        engine,
        lambda job: Path(WORK_DIR) / job["pdf"],
        sink,
        deadline,
        loaded=loaded,
        commit=work.commit,
    )
    return {
        "outputs": outputs,
        "load_seconds": round(loaded - started),
        "seconds": round(time.time() - started),
    }


@app.local_entrypoint()
def main(jobs: str, confirm_usd: float = 0.0):
    """`jobs` is the request file written by tools/probe_jobs.py; answers are saved next to it."""
    path = Path(jobs)
    batch = json.loads(path.read_text())
    limit_minutes = batch["limit_minutes"]
    estimate = round(limit_minutes / 60 * L4_PER_HOUR, 2)
    print(
        f"{len(batch['jobs'])} requests; worst-case Modal cost ${estimate:.2f} ({limit_minutes}-minute limit)"
    )
    if confirm_usd < estimate:
        raise SystemExit("Not launched: pass --confirm-usd at least equal to the stated estimate")
    name = path.parent.name
    # The batch stops itself two minutes before the container's time limit and returns what finished.
    result = generate.with_options(timeout=limit_minutes * 60).remote(
        batch["jobs"], batch["engine"], name, limit_minutes * 60 - 120
    )
    (path.parent / "answers.json").write_text(json.dumps(result, indent=1))
    print(
        f"{len(result['outputs'])} answers in {result['seconds']} s (model load {result['load_seconds']} s)"
    )


cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "jsonschema>=4.23.0,<5", "pydantic>=2.11.9,<3", "pypdfium2==4.30.0"
)
if modal.is_local():
    cpu_image = cpu_image.add_local_dir(Path(__file__).resolve().parents[2] / "src", f"{SRC}/src")


@app.function(image=cpu_image, cpu=2, memory=16 * 1024, volumes={WORK_DIR: work}, timeout=1200)
def ground(items: list[dict], snapshots: str) -> dict:
    """Locate each answer's printed values in the stored parse of its PDF (native text or OCR) and
    return citations per answer key. Snapshots are found by the PDF's SHA-256 under `snapshots`."""
    import hashlib
    import sys

    sys.path.insert(0, f"{SRC}/src")
    from docextract.contracts import ParseSnapshot
    from docextract.oneshot import locate_evidence

    stores = list(Path(WORK_DIR).glob(snapshots))
    loaded: dict[str, object] = {}
    out = {}
    for item in items:
        pdf = Path(WORK_DIR) / item["pdf"]
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if digest not in loaded:
            path = next(
                (store / f"{digest}.json" for store in stores if (store / f"{digest}.json").exists()), None
            )
            loaded[digest] = ParseSnapshot.model_validate_json(path.read_text()) if path else None
        snapshot = loaded[digest]
        out[item["key"]] = locate_evidence(item["data"], item["schema"], snapshot) if snapshot else None
    return out


@app.local_entrypoint()
def evidence(answers: str, snapshots: str = "runs/final-v1/shard-*/snapshots"):
    """`answers` is the file `tools/score_probe.py --export` wrote; citations are saved next to it."""
    path = Path(answers)
    items = json.loads(path.read_text())
    print(f"{len(items)} answers to ground; worst-case Modal cost $0.03 (CPU, 20-minute limit)")
    result = ground.remote(items, snapshots)
    (path.parent / "evidence.json").write_text(json.dumps(result))
    print(f"grounded {sum(1 for v in result.values() if v is not None)} of {len(result)} answers")


fetch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "huggingface-hub>=0.34", "jsonschema>=4.23.0,<5", "pydantic>=2.11.9,<3", "pypdfium2==4.30.0"
    )
    .env({"HF_HOME": f"{WORK_DIR}/model-cache/huggingface", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    fetch_image = fetch_image.add_local_dir(root / "src", f"{SRC}/src").add_local_dir(
        root / "manifests", f"{SRC}/manifests"
    )


@app.function(image=fetch_image, cpu=2, memory=4 * 1024, volumes={WORK_DIR: work}, timeout=1800)
def fetch(arm: str) -> str:
    """Download a pinned checkpoint into the work volume on a CPU container and verify every hash."""
    import sys

    sys.path.insert(0, f"{SRC}/src")
    from docextract.remote import pinned_checkpoint

    name, revision, checkpoint = pinned_checkpoint(
        arm, Path(SRC) / "manifests/model-research.json", Path(WORK_DIR) / "model-cache"
    )
    work.commit()
    return f"{name}@{revision} verified at {checkpoint}"


@app.local_entrypoint()
def checkpoint(arm: str):
    """Fetch a model before a GPU probe needs it (CPU only; about $0.02)."""
    print(fetch.remote(arm))
