"""docextract on Modal: CPU containers prepare data and score, L4 GPUs extract.

One launcher for two kinds of configuration (JSON):

- a research session (`"final": false`): a few chosen documents and prompt variants, paid from the
  research budget;
- the single frozen final run (`"final": true`, `"frozen": true`) over the whole benchmark.

Print the worst-case cost without contacting Modal:

    uv run python runtimes/modal/final_run.py <config.json>

Launch (the confirmation must cover the printed estimate):

    modal run runtimes/modal/final_run.py --config <config.json> --confirm-usd <estimate>

Answers and the dataset copy live in an evaluation volume that GPU workers never mount; inference
workers read only PDFs and original schemas. Every call has a hard time limit set at launch, and GPU
shards also stop at an absolute deadline, so a restarted (preempted) container resumes from stored
answers without extending the budget. Results for local analysis are written under
`runs/<name>/export` in the work volume (`modal volume get docextract-work runs/<name>/export <dir>`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Modal list prices (modal.com/pricing, checked 2026-10-15), USD per second.
L4_PER_SECOND = 0.000222
CORE_PER_SECOND = 0.0000131
GIB_PER_SECOND = 0.00000222
GPU_CORES, GPU_MEMORY_GIB = 4, 24
CPU_CORES, CPU_MEMORY_GIB = 4, 16
PROMPT_KEYS = (
    "protocol",
    "page_images",
    "route_groups",
    "prompt_budget_chars",
    "max_window_units",
    "factor_notes",
    "merge_entities",
    "page_image_side",
    "page_image_side_split",
)


def gpu_hourly() -> float:
    return 3600 * (L4_PER_SECOND + GPU_CORES * CORE_PER_SECOND + GPU_MEMORY_GIB * GIB_PER_SECOND)


def cpu_hourly() -> float:
    return 3600 * (CPU_CORES * CORE_PER_SECOND + CPU_MEMORY_GIB * GIB_PER_SECOND)


ORCHESTRATOR_CORES, ORCHESTRATOR_MEMORY_GIB = 0.25, 1


def orchestrator_hourly() -> float:
    return 3600 * (ORCHESTRATOR_CORES * CORE_PER_SECOND + ORCHESTRATOR_MEMORY_GIB * GIB_PER_SECOND)


def total_hours(config: dict) -> float:
    return config["gpu_hours"] + config["prepare_hours"] + config["score_hours"]


def worst_case_usd(config: dict) -> float:
    """Every GPU shard and CPU stage running to its time limit, plus the small orchestrating job."""
    gpu = config["shards"] * config["gpu_hours"] * gpu_hourly()
    cpu = (config["prepare_hours"] + config["score_hours"]) * cpu_hourly()
    return round(gpu + cpu + total_hours(config) * orchestrator_hourly(), 2)


def prompt_variant(config: dict) -> dict:
    """The frozen prompt-level options as the runner's single, unnamed variant."""
    return {"name": "", **{key: config[key] for key in PROMPT_KEYS if config.get(key) is not None}}


def variants_of(config: dict) -> list[dict]:
    return config.get("variants") or [prompt_variant(config)]


def assign_shards(estimates: list[dict], shards: int) -> list[list[str]]:
    """Largest documents first to the least-loaded shard; each shard then runs short work first."""
    loads = [0.0] * shards
    members: list[list[dict]] = [[] for _ in range(shards)]
    for case in sorted(estimates, key=lambda case: -case["work"]):
        target = loads.index(min(loads))
        loads[target] += case["work"]
        members[target].append(case)
    return [[case["id"] for case in sorted(group, key=lambda case: case["work"])] for group in members]


if __name__ == "__main__":
    settings = json.loads(Path(sys.argv[1]).read_text())
    print(
        json.dumps(
            {
                "worst_case_usd": worst_case_usd(settings),
                "gpu_container_usd_per_hour": round(gpu_hourly(), 3),
                "cpu_container_usd_per_hour": round(cpu_hourly(), 3),
            }
        )
    )
    raise SystemExit(0)

import modal  # noqa: E402  (the estimate above needs no Modal client)

SRC = "/opt/docextract"
WORK_DIR, EVAL_DIR = "/work", "/eval"
work = modal.Volume.from_name("docextract-work", create_if_missing=True)
answers = modal.Volume.from_name("docextract-eval", create_if_missing=True)
app = modal.App("docextract")


def with_project(image: modal.Image) -> modal.Image:
    if not modal.is_local():
        return image
    root = Path(__file__).resolve().parents[2]
    return (
        image.add_local_dir(root / "src", f"{SRC}/src")
        .add_local_dir(root / "tools", f"{SRC}/tools")
        .add_local_dir(root / "manifests", f"{SRC}/manifests")
    )


BASE_ENV = {
    "PYTHONPATH": f"{SRC}/src:{SRC}",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "HF_HOME": f"{WORK_DIR}/model-cache/huggingface",
}
cpu_image = with_project(
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "llama-extract-bench @ git+https://github.com/run-llama/ExtractBench.git"
        "@0e7647997211867060a645ac82deb9f7bc4a3b3f",
        "parse-bench==1.0.4",
        "jsonschema>=4.23.0,<5",
        "pydantic>=2.11.9,<3",
        "pypdfium2==4.30.0",
        "huggingface-hub>=0.34",
    )
    .env(BASE_ENV)
)
gpu_image = with_project(
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
            **BASE_ENV,
            "HF_HUB_OFFLINE": "1",
            "DOCTR_CACHE_DIR": "/tmp/doctr-cache",
            # FlashInfer's sampler compiles CUDA code at start-up (needs nvcc, absent here); decoding
            # is greedy, so vLLM's own sampler is used instead.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            # Compiled GPU kernels persist in the work volume, so later sessions start faster.
            "TRITON_CACHE_DIR": f"{WORK_DIR}/compile-cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{WORK_DIR}/compile-cache/inductor",
            "VLLM_CACHE_ROOT": f"{WORK_DIR}/compile-cache/vllm",
        }
    )
)


def run(command: list[str], *, deadline: float | None = None, commit=None, log: Path | None = None) -> int:
    """Run a step; commit the work volume every few minutes so a preemption loses little."""
    sink = log.open("a") if log else None
    try:
        process = subprocess.Popen(
            command,
            cwd=SRC,
            start_new_session=True,
            stdout=sink,
            stderr=subprocess.STDOUT if sink else None,
        )
        while process.poll() is None:
            if deadline is not None and time.time() > deadline:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            if commit:
                commit()
            time.sleep(1 if commit is None else 180)
        if commit:
            commit()
        return process.wait()
    finally:
        if sink:
            sink.close()


def window_arguments(config: dict) -> list[str]:
    units = config.get("max_window_units")
    return [
        "--max-window-chars",
        str(config["max_window_chars"]),
        "--max-schema-chars",
        str(config["max_schema_chars"]),
        *(["--max-window-units", str(units)] if units else []),
    ]


@app.function(
    image=cpu_image,
    volumes={WORK_DIR: work, EVAL_DIR: answers},
    cpu=CPU_CORES,
    memory=CPU_MEMORY_GIB * 1024,
)
def prepare(config: dict) -> list[dict]:
    """Download the pinned dataset and checkpoint (once), split this run's inference inputs from its
    answers, and estimate each document's work for sharding (pages and schema size, no answers)."""
    import shutil
    from contextlib import closing

    import pypdfium2 as pdfium

    data = Path(EVAL_DIR) / "data"
    if not (data / "dataset-manifest.json").exists():
        subprocess.run(
            ["python", "tools/prepare_benchmark.py", "--download", "--output", str(data)], cwd=SRC, check=True
        )
    staging = Path("/tmp/packet")
    selection = [f"--{split}={count}" for split, count in config.get("counts", {}).items()]
    if config.get("offsets"):
        selection += ["--offsets", json.dumps(config["offsets"])]
    if config.get("ids"):
        selection += ["--ids", ",".join(config["ids"])]
    subprocess.run(
        [
            "python",
            "tools/extractbench_packet.py",
            "--snapshot",
            str(data / "snapshot"),
            "--output",
            str(staging),
            *selection,
        ],
        cwd=SRC,
        check=True,
    )
    inference = Path(WORK_DIR) / "packets" / config["name"] / "inference"
    evaluation = Path(EVAL_DIR) / "packets" / config["name"] / "evaluation"
    for source, target in ((staging / "inference", inference), (staging / "evaluation", evaluation)):
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    sys.path.insert(0, f"{SRC}/src")
    from docextract.remote import pinned_checkpoint

    cache = Path(WORK_DIR) / "model-cache"
    cache.mkdir(parents=True, exist_ok=True)
    pinned_checkpoint(config["arm"], Path(SRC) / "manifests/model-research.json", cache)
    estimates = []
    for case in json.loads((inference / "cases.json").read_text())["cases"]:
        with closing(pdfium.PdfDocument(str(inference / case["pdf"]))) as pdf:
            pages = len(pdf)
        schema_chars = len((inference / case["schema"]).read_text())
        estimates.append({"id": case["id"], "work": pages * 3000 + schema_chars})
    work.commit()
    answers.commit()
    return estimates


@app.function(
    image=gpu_image,
    gpu="L4",
    volumes={WORK_DIR: work},
    cpu=GPU_CORES,
    memory=GPU_MEMORY_GIB * 1024,
)
def extract(config: dict, shard: int, ids: list[str], deadline: float) -> dict:
    """One shard of documents through vLLM on one L4; answers stream to the work volume."""
    work.reload()
    inference = Path(WORK_DIR) / "packets" / config["name"] / "inference"
    cases = json.loads((inference / "cases.json").read_text())
    chosen = {case["id"]: case for case in cases["cases"]}
    manifest = inference / f"shard-{shard}.json"
    manifest.write_text(json.dumps({"version": cases["version"], "cases": [chosen[i] for i in ids]}))
    output = Path(WORK_DIR) / "runs" / config["name"] / f"shard-{shard}"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "tools/extract_remote.py",
        "--cases",
        str(manifest),
        "--output",
        str(output),
        "--model-manifest",
        f"{SRC}/manifests/model-research.json",
        "--engine",
        "vllm",
        "--offline",
        "--arm",
        config["arm"],
        "--cache-root",
        f"{WORK_DIR}/model-cache",
        "--dtype",
        config["dtype"],
        *window_arguments(config),
        "--max-new-tokens",
        str(config["max_new_tokens"]),
        "--max-model-len",
        str(config["max_model_len"]),
        "--max-num-seqs",
        str(config["max_num_seqs"]),
        "--gpu-memory-utilization",
        str(config["gpu_memory_utilization"]),
        "--ocr",
        config["ocr"],
        "--speculative-tokens",
        str(config.get("speculative_tokens", 0)),
        "--variants",
        json.dumps(variants_of(config)),
        *(["--quantization", config["quantization"]] if config.get("quantization") else []),
        *(["--kv-cache-dtype", config["kv_cache_dtype"]] if config.get("kv_cache_dtype") else []),
        *(
            ["--max-document-pages", str(config["max_document_pages"])]
            if config.get("max_document_pages")
            else []
        ),
    ]
    started = time.time()
    code = run(command, deadline=deadline, commit=work.commit, log=output / "engine.log")
    return {
        "shard": shard,
        "exit": code,
        "seconds": round(time.time() - started),
        "stopped_at_deadline": time.time() > deadline,
    }


@app.function(
    image=cpu_image,
    volumes={WORK_DIR: work, EVAL_DIR: answers},
    cpu=CPU_CORES,
    memory=CPU_MEMORY_GIB * 1024,
)
def score(config: dict, shards: int) -> dict:
    """Assemble every document from stored answers (partial ones included), score each prompt variant
    officially, and gather what local analysis needs under runs/<name>/export."""
    import gzip
    import shutil

    work.reload()
    name = config["name"]
    runs = Path(WORK_DIR) / "runs" / name
    inference = Path(WORK_DIR) / "packets" / name / "inference"
    evaluation = Path(EVAL_DIR) / "packets" / name / "evaluation"
    export = runs / "export"
    variants = variants_of(config)
    # Each GPU shard's stored answers are re-assembled in parallel.
    replays = {}
    for shard in range(shards):
        source = runs / f"shard-{shard}"
        if not (source / "generations.jsonl").exists():
            continue
        replays[shard] = subprocess.Popen(
            [
                "python",
                "tools/extract_remote.py",
                "--cases",
                str(inference / f"shard-{shard}.json"),
                "--output",
                f"/tmp/assembled-{shard}",
                "--model-manifest",
                f"{SRC}/manifests/model-research.json",
                "--engine",
                "replay",
                "--replay-from",
                str(source),
                "--arm",
                config["arm"],
                "--cache-root",
                f"/tmp/replay-cache-{shard}",
                *window_arguments(config),
                "--absent-nullable",
                config.get("absent_nullable", "null"),
                *(["--complete-tables"] if config.get("complete_tables") else []),
                "--variants",
                json.dumps(variants),
            ],
            cwd=SRC,
            env={**os.environ, "MODAL_TASK_ID": os.environ.get("MODAL_TASK_ID", "score")},
        )
    for shard, process in replays.items():
        if process.wait() != 0:
            raise RuntimeError(f"Re-assembly of shard {shard} failed")
    for shard in replays:
        source = runs / f"shard-{shard}"
        shard_export = export / f"shard-{shard}"
        (shard_export / "snapshots").mkdir(parents=True, exist_ok=True)
        for file in ("runtime.json", "telemetry.json", "generations.jsonl", "ocr.json", "engine.log"):
            if (source / file).exists():
                shutil.copyfile(source / file, shard_export / file)
        for path in (source / "snapshots").glob("*.json"):
            with (
                path.open("rb") as raw,
                gzip.open(shard_export / "snapshots" / (path.name + ".gz"), "wb") as sink,
            ):
                shutil.copyfileobj(raw, sink)
        for path in (source / "page-images").glob("*/*.jpg"):
            target = shard_export / "page-images" / path.relative_to(source / "page-images")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        for summary in source.glob("**/summary.json"):
            target = shard_export / summary.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(summary, target)
    seen = Path("/tmp/seen.json")
    subsets = json.loads(Path(f"{SRC}/manifests/dev-seen-cases.json").read_text())["subsets"]
    seen.write_text(json.dumps(sorted({case for subset in subsets for case in subset["cases"]})))
    groups = {}
    for variant in variants:
        label = variant["name"] or "final"
        merged = Path("/tmp/scored") / label
        for shard in range(shards):
            for result in Path(f"/tmp/assembled-{shard}", variant["name"]).glob("*/result.json"):
                target = merged / result.parent.name / "result.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(result, target)
        report = export / f"{label}-scores.json"
        subprocess.run(
            [
                "python",
                "tools/score_extractbench.py",
                "--evaluation",
                str(evaluation),
                "--results",
                str(merged),
                "--output",
                str(report),
                "--details",
                str(export / f"{label}-details"),
                "--pipeline",
                f"docextract_{name}_{label}",
                "--seen",
                str(seen),
            ],
            cwd=SRC,
            check=True,
        )
        groups[label] = json.loads(report.read_text())["groups"]
    work.commit()
    return groups


def pipeline(settings: dict) -> dict:
    """Prepare, extract on every shard (absolute deadline), then score; returns the score groups."""
    started = time.time()
    estimates = prepare.with_options(timeout=int(settings["prepare_hours"] * 3600)).remote(settings)
    groups = assign_shards(estimates, settings["shards"])
    deadline = started + settings["gpu_hours"] * 3600 - 300
    limit = int(settings["gpu_hours"] * 3600)
    results = list(
        extract.with_options(timeout=limit).starmap(
            [(settings, shard, ids, deadline) for shard, ids in enumerate(groups)],
            return_exceptions=True,
        )
    )
    print(json.dumps([str(result) for result in results]), flush=True)
    scored = score.with_options(timeout=int(settings["score_hours"] * 3600)).remote(
        settings, settings["shards"]
    )
    print(json.dumps(scored, indent=1), flush=True)
    return scored


@app.function(image=cpu_image, cpu=ORCHESTRATOR_CORES, memory=ORCHESTRATOR_MEMORY_GIB * 1024)
def orchestrate(settings: dict) -> dict:
    """The whole run driven from Modal, so a dropped connection on the launching machine stops nothing."""
    return pipeline(settings)


@app.local_entrypoint()
def main(config: str, confirm_usd: float = 0.0, stage: str = "all"):
    """stage "all" runs everything; "score" only re-assembles and scores answers already stored."""
    settings = json.loads(Path(config).read_text())
    if settings.get("final") and settings.get("frozen") is not True:
        raise SystemExit("Not launched: the final configuration is not marked frozen")
    if stage == "score":
        estimate = round(settings["score_hours"] * cpu_hourly(), 2)
        print(f"Worst-case Modal cost for scoring only: ${estimate:.2f}")
        if confirm_usd < estimate:
            raise SystemExit("Not launched: pass --confirm-usd at least equal to the stated estimate")
        scored = score.with_options(timeout=int(settings["score_hours"] * 3600)).remote(
            settings, settings["shards"]
        )
        print(json.dumps(scored, indent=1))
        return
    estimate = worst_case_usd(settings)
    print(f"Worst-case Modal cost for this run: ${estimate:.2f} (GPU ${gpu_hourly():.3f}/h per shard)")
    if confirm_usd < estimate:
        raise SystemExit("Not launched: pass --confirm-usd at least equal to the stated estimate")
    if stage == "remote":
        # Run with `modal run --detach`: the whole run is spawned as one job on Modal and this command
        # returns at once, so nothing here waits on it. (final-v1 waited on the job instead; when this
        # machine lost its connection the waiting call failed and Modal cancelled the job, its GPU
        # shards and the scoring.) Answers and scores still go to the work volume.
        call = orchestrate.with_options(timeout=int(total_hours(settings) * 3600) + 900).spawn(settings)
        print(f"Spawned {call.object_id}; results go to runs/{settings['name']}/export in docextract-work")
        return
    pipeline(settings)
