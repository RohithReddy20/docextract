"""Template for a private Kaggle ExtractBench development run; filled by build_extractbench_kernel.py.

Inference workers read only PDFs and original schemas. Answers and field rules stay in a separate
evaluation directory that only the official scorer process reads. Weights and data stay remote.
"""

import base64
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import signal
import subprocess
import tarfile
import time
from pathlib import Path

PAYLOAD = "__DOCEXTRACT_PAYLOAD__"
CHECKSUM = "__DOCEXTRACT_CHECKSUM__"
CONFIG = json.loads("""__DOCEXTRACT_CONFIG__""")


def main():
    if platform.system() != "Linux" or not Path("/kaggle/working").is_dir() or not shutil.which("uv"):
        raise RuntimeError("This bootstrap runs only in Kaggle with uv; no local fallback")
    gpus = []
    if CONFIG["runs"]:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        if len(gpus) < len(CONFIG["runs"]) or any("T4" not in name for name in gpus):
            raise RuntimeError("Expected one independent T4 per run")
    payload = base64.b64decode(PAYLOAD, validate=True)
    if hashlib.sha256(payload).hexdigest() != CHECKSUM:
        raise RuntimeError("Source archive hash mismatch")
    root = Path("/kaggle/temp") / ("docextract-" + CHECKSUM[:12])
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        bundle.extractall(root, filter="data")
    project = root / "docextract"
    runtime = project / "runtimes/kaggle"
    packet = root / "packet"
    output = Path("/kaggle/working/extractbench-dev")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + CONFIG["seconds"]
    report = {
        "cash_cost_estimate_usd": 0.0,
        "source_bundle_sha256": CHECKSUM,
        "config": CONFIG,
        "gpus_assigned": gpus,
        "stages": {},
        "complete": False,
    }
    # Compiled GPU kernels and installed wheels are reused across sessions: a later run attaches an
    # earlier run's output (kernel_sources) and restores these before setup and engine start.
    compile_cache = Path("/kaggle/temp/compile-cache")
    environment = dict(
        os.environ,
        UV_CACHE_DIR=str(root / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(root / "python"),
        HF_HOME=str(root / "hf-data"),
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TRITON_CACHE_DIR=str(compile_cache / "triton"),
        TRITON_CACHE_AUTOTUNING="1",
        TORCHINDUCTOR_CACHE_DIR=str(compile_cache / "inductor"),
        VLLM_CACHE_ROOT=str(compile_cache / "vllm"),
    )

    def restore(name: str, target: Path) -> str | None:
        inputs = Path("/kaggle/input")
        found = sorted(inputs.glob(f"*/extractbench-dev/{name}")) if inputs.is_dir() else []
        if not found:
            return None
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(found[0]) as bundle:
            bundle.extractall(target, filter="data")
        return str(found[0].parent.parent.name)

    def keep(name: str, source: Path, limit: int) -> None:
        if source.is_dir():
            size = sum(path.stat().st_size for path in source.rglob("*") if path.is_file())
            if 0 < size <= limit:
                with tarfile.open(output / name, "w") as bundle:
                    bundle.add(source, arcname=".")

    def export_parsed(base: Path, target: Path) -> None:
        """Parsed evidence (with OCR, gzip-compressed) and page images, so stored answers can be
        re-assembled and analysed off-GPU. Snapshots hold every character box, hence compression."""
        for path in sorted((base / "snapshots").glob("*.json")):
            destination = target / "snapshots" / (path.name + ".gz")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source, gzip.open(destination, "wb", compresslevel=6) as sink:
                shutil.copyfileobj(source, sink)
        images = sorted((base / "page-images").glob("*/*.jpg"))
        if sum(path.stat().st_size for path in images) <= 500_000_000:
            for path in images:
                destination = target / "page-images" / path.relative_to(base / "page-images")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)

    def checkpoint():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")

    def run(name, *commands, until=None, logs=None):
        limit = until or deadline
        report["stages"][name] = {"status": "running"}
        checkpoint()
        begin = time.monotonic()
        # With several parallel commands each one writes its own log, so engine lines stay apart.
        sinks = [open(path, "w") for path in logs] if logs else [None] * len(commands)
        processes = [
            subprocess.Popen(
                command,
                cwd=project,
                env=environment,
                start_new_session=True,
                stdout=sink,
                stderr=subprocess.STDOUT if sink else None,
            )
            for command, sink in zip(commands, sinks, strict=True)
        ]
        try:
            codes = [process.wait(timeout=max(1, limit - time.monotonic())) for process in processes]
            status = "complete" if not any(codes) else f"exit {codes}"
        except subprocess.TimeoutExpired:
            status = "deadline"
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            for sink in sinks:
                if sink:
                    sink.close()
        report["stages"][name] = {"status": status, "seconds": round(time.monotonic() - begin, 1)}
        checkpoint()
        return status

    main_env = ["uv", "run", "--frozen", "--no-dev", "--extra", "benchmark", "python"]
    runs = CONFIG["runs"]
    hf_python = str(runtime / ".venv/bin/python")
    vllm_python = str(root / "vllm-env/bin/python")
    try:
        print("Estimated total cash cost $0 on free Kaggle quota; Modal unused.", flush=True)
        report["restored"] = {
            "wheels": restore("uv-cache.tar", root / "uv-cache"),
            "compiled_kernels": restore("compile-cache.tar", compile_cache),
        }
        checkpoint()
        authored = CONFIG.get("packet") == "authored"
        if authored:
            # Embedded authored inference inputs only; their answers are scored after export.
            packet, setup = project / "authored", []
        else:
            setup = [
                ("sync-benchmark", ["uv", "sync", "--frozen", "--no-dev", "--extra", "benchmark"]),
                (
                    "download",
                    [*main_env, "tools/prepare_benchmark.py", "--download", "--output", str(root / "data")],
                ),
                (
                    "packet",
                    [
                        *main_env,
                        "tools/extractbench_packet.py",
                        "--snapshot",
                        str(root / "data/snapshot"),
                        "--output",
                        str(packet),
                        *(f"--{key}={value}" for key, value in CONFIG.get("counts", {}).items()),
                        *(["--ids", ",".join(CONFIG["ids"])] if CONFIG.get("ids") else []),
                        *(["--offsets", json.dumps(CONFIG["offsets"])] if CONFIG.get("offsets") else []),
                    ],
                ),
            ]
        if any(run_["engine"] == "hf" for run_ in runs):
            setup.append(("sync-runtime", ["uv", "sync", "--project", str(runtime), "--frozen"]))
        if any(run_.get("kernels") == "fla" for run_ in runs):
            setup.append(
                ("fla", ["uv", "pip", "install", "--python", hf_python, "--no-deps", *CONFIG["fla_packages"]])
            )
        if (
            any(run_["engine"] == "vllm" for run_ in runs)
            or CONFIG.get("build_wheel_cache")
            or CONFIG.get("parse_only")
        ):
            setup.append(("vllm-venv", ["uv", "venv", str(root / "vllm-env"), "--python", "3.12"]))
            setup.append(
                (
                    "vllm-install",
                    ["uv", "pip", "install", "--python", vllm_python, *CONFIG["vllm_packages"], "."],
                )
            )
        for name, command in setup:
            if run(name, command) != "complete":
                raise RuntimeError(f"{name} did not complete")
        if CONFIG.get("build_wheel_cache"):
            # A CPU session fills the wheel cache once; GPU sessions then install without downloading.
            keep("uv-cache.tar", root / "uv-cache", 15_000_000_000)
        for number, arguments in enumerate(CONFIG.get("tools", [])):
            # Counting tools over the pinned snapshot; they export counts, never document content.
            filled = [
                argument.replace("{snapshot}", str(root / "data/snapshot")).replace("{output}", str(output))
                for argument in arguments
            ]
            run(f"tool-{number}", [*main_env, *filled])
        if CONFIG.get("export_packet") and not authored:
            # Development cases only: schemas, document hashes and scoring files (no PDFs), so stored
            # answers can be re-assembled and scored on a CPU elsewhere.
            exported = output / "packet"
            shutil.copytree(
                packet / "evaluation",
                exported / "evaluation",
                dirs_exist_ok=True,
                ignore=lambda folder, names: [name for name in names if name.endswith(".pdf")],
            )
            cases = json.loads((packet / "inference/cases.json").read_text())
            (exported / "inference").mkdir(parents=True, exist_ok=True)
            for case in cases["cases"]:
                shutil.copyfile(
                    packet / "inference" / case["schema"], exported / "inference" / case["schema"]
                )
                pdf = (packet / "inference" / case["pdf"]).read_bytes()
                case["pdf_sha256"] = hashlib.sha256(pdf).hexdigest()
            (exported / "inference/documents.json").write_text(json.dumps(cases, indent=1) + "\n")

        if CONFIG.get("parse_only"):
            # CPU session: parse (and OCR on the CPU) the selected documents and export the snapshots.
            settings = CONFIG["parse_only"]
            run(
                "parse",
                [
                    vllm_python,
                    "tools/extract_remote.py",
                    "--cases",
                    str(packet / "inference/cases.json"),
                    "--output",
                    str(root / "parse"),
                    "--model-manifest",
                    str(project / "manifests/model-research.json"),
                    "--engine",
                    "vllm",
                    "--arm",
                    "a",
                    "--cache-root",
                    str(root / "model-cache-parse"),
                    "--ocr",
                    settings.get("ocr", "doctr"),
                    "--ocr-device",
                    "cpu",
                    "--parse-only",
                    "--snapshots",
                    str(root / "snapshots"),
                    "--page-images-dir",
                    str(root / "page-images"),
                    *(["--keep-page-images"] if settings.get("page_images") else []),
                ],
            )
            export_parsed(root, output / "parsed")
            if (root / "parse" / "ocr.json").exists():
                shutil.copyfile(root / "parse" / "ocr.json", output / "parsed" / "ocr.json")

        def window_arguments():
            return [
                "--max-window-chars",
                str(CONFIG["max_window_chars"]),
                "--max-schema-chars",
                str(CONFIG.get("max_schema_chars", 24000)),
                *(
                    ["--max-window-units", str(CONFIG["max_window_units"])]
                    if CONFIG.get("max_window_units")
                    else []
                ),
            ]

        def variants_of(run_):
            # Prompt variants share one engine per GPU; without a list the run is one flat variant.
            return run_.get("variants") or [
                {
                    "name": "",
                    "protocol": run_.get("protocol", "cited"),
                    "page_images": bool(run_.get("page_images")),
                }
            ]

        # With one OCR setting for every GPU, documents are parsed once, split across the GPUs, into a
        # shared store; every run (and replay) then reads the same snapshots and page images.
        shared = len({run_.get("ocr", "none") for run_ in runs}) == 1
        stores = (
            ["--snapshots", str(root / "snapshots"), "--page-images-dir", str(root / "page-images")]
            if shared
            else []
        )
        if shared and runs and runs[0].get("ocr", "none") != "none":
            keep_images = any(v.get("page_images") for run_ in runs for v in variants_of(run_))
            parse = [
                [
                    vllm_python,
                    "tools/extract_remote.py",
                    "--cases",
                    str(packet / "inference/cases.json"),
                    "--output",
                    str(root / "parse" / str(number)),
                    "--model-manifest",
                    str(project / "manifests/model-research.json"),
                    "--engine",
                    "vllm",
                    "--arm",
                    "a",
                    "--device",
                    str(run_["device"]),
                    "--cache-root",
                    str(root / f"model-cache-{run_['name']}"),
                    "--ocr",
                    run_["ocr"],
                    "--parse-only",
                    "--shard",
                    f"{number}/{len(runs)}",
                    *stores,
                    *(["--keep-page-images"] if keep_images else []),
                ]
                for number, run_ in enumerate(runs)
            ]
            run("parse", *parse)
        commands, logs = [], []
        for run_ in runs:
            commands.append(
                [
                    vllm_python if run_["engine"] == "vllm" else hf_python,
                    "tools/extract_remote.py",
                    "--cases",
                    str(packet / "inference/cases.json"),
                    "--output",
                    str(root / "runs" / run_["name"]),
                    "--model-manifest",
                    str(project / "manifests/model-research.json"),
                    "--engine",
                    run_["engine"],
                    "--arm",
                    run_["arm"],
                    "--kernels",
                    run_.get("kernels", "torch"),
                    "--device",
                    str(run_["device"]),
                    "--cache-root",
                    str(root / f"model-cache-{run_['name']}"),
                    *window_arguments(),
                    "--max-input-tokens",
                    str(CONFIG["max_input_tokens"]),
                    "--max-new-tokens",
                    str(CONFIG["max_new_tokens"]),
                    "--max-model-len",
                    str(CONFIG["max_model_len"]),
                    "--max-num-seqs",
                    str(CONFIG.get("max_num_seqs", 64)),
                    "--gpu-memory-utilization",
                    str(CONFIG.get("gpu_memory_utilization", 0.9)),
                    "--ocr",
                    run_.get("ocr", "none"),
                    *(["--constrained"] if run_.get("constrained") else []),
                    "--speculative-tokens",
                    str(run_.get("speculative_tokens", 0)),
                    *(["--no-prefix-caching"] if run_.get("prefix_caching") is False else []),
                    "--variants",
                    json.dumps(variants_of(run_)),
                    "--shard",
                    run_.get("shard", "0/1"),
                    *stores,
                ]
            )
            logs.append(root / f"engine-{run_['name']}.log")
        # Reserve time for assembly, official scoring and export; unfinished work is assembled partly.
        run("inference", *commands, until=deadline - CONFIG["scoring_reserve_seconds"], logs=logs)
        keep("compile-cache.tar", compile_cache, 4_000_000_000)
        for run_, log in zip(runs, logs, strict=True):
            if log.exists():
                (output / run_["name"]).mkdir(parents=True, exist_ok=True)
                (output / run_["name"] / "engine.log").write_bytes(log.read_bytes()[-4_000_000:])
        # "final" re-assembles every document from the stored answers with the run's own options,
        # including documents a deadline left part-answered. Other options are replayed off-GPU.
        for run_ in runs:
            source = root / "runs" / run_["name"]
            if not (source / "generations.jsonl").exists():
                continue
            run(
                f"final-{run_['name']}",
                [
                    vllm_python if run_["engine"] == "vllm" else hf_python,
                    "tools/extract_remote.py",
                    "--cases",
                    str(packet / "inference/cases.json"),
                    "--output",
                    str(root / "runs" / f"{run_['name']}+final"),
                    "--model-manifest",
                    str(project / "manifests/model-research.json"),
                    "--engine",
                    "replay",
                    "--replay-from",
                    str(source),
                    "--arm",
                    run_["arm"],
                    "--cache-root",
                    str(root / f"replay-cache-{run_['name']}"),
                    *window_arguments(),
                    "--variants",
                    json.dumps(variants_of(run_)),
                    "--shard",
                    run_.get("shard", "0/1"),
                    *stores,
                ],
            )
        # Runs that split one configuration's documents (shards) are scored together, per variant.
        groups: dict[str, list] = {}
        for run_ in runs:
            groups.setdefault(run_.get("group", run_["name"]), []).append(run_)
        for group, members in groups.items():
            for variant in variants_of(members[0]):
                label = f"{group}-{variant['name']}" if variant["name"] else group
                merged = root / "scored" / label
                for member in members:
                    final = root / "runs" / f"{member['name']}+final" / variant["name"]
                    for result in final.glob("*/result.json"):
                        target = merged / result.parent.name / "result.json"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(result, target)
                if not authored:
                    run(
                        f"score-{label}",
                        [
                            *main_env,
                            "tools/score_extractbench.py",
                            "--evaluation",
                            str(packet / "evaluation"),
                            "--results",
                            str(merged),
                            "--output",
                            str(output / f"{label}-scores.json"),
                            "--details",
                            str(output / f"{label}-details"),
                            "--pipeline",
                            f"docextract_{label}",
                        ],
                    )
                for result in sorted(merged.glob("*/result.json")):
                    data = json.loads(result.read_text())
                    if not authored:
                        data.pop("snapshot", None)
                    compact = json.dumps(data)
                    if len(compact) <= 3_000_000:
                        target = output / label / result.parent.name / "result.compact.json"
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(compact)
        for run_ in runs:
            name = run_["name"]
            source = root / "runs" / name
            (output / name).mkdir(parents=True, exist_ok=True)
            for file in ("telemetry.json", "runtime.json", "generations.jsonl", "ocr.json"):
                if (source / file).exists() and (source / file).stat().st_size <= 20_000_000:
                    shutil.copyfile(source / file, output / name / file)
            for summary in source.glob("**/summary.json"):
                target = output / name / summary.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(summary, target)
            export_parsed(root if shared else source, output / name)
        report["complete"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)[:500]
        raise
    finally:
        report["seconds"] = round(time.monotonic() - started, 1)
        checkpoint()


if __name__ == "__main__":
    main()
