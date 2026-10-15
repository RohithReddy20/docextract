"""Template for a private Kaggle engine probe; filled by build_probe_kernel.py. Weights stay remote."""

import base64
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
    gpus = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if len(gpus) < 2 or any("T4" not in name for name in gpus):
        raise RuntimeError("The probe expects two independent T4s")
    payload = base64.b64decode(PAYLOAD, validate=True)
    if hashlib.sha256(payload).hexdigest() != CHECKSUM:
        raise RuntimeError("Source archive hash mismatch")
    root = Path("/kaggle/temp") / ("docextract-probe-" + CHECKSUM[:12])
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        bundle.extractall(root, filter="data")
    project = root / "docextract"
    runtime = project / "runtimes/kaggle"
    output = Path("/kaggle/working/engine-probe")
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + CONFIG["seconds"]
    environment = dict(
        os.environ,
        UV_CACHE_DIR=str(root / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(root / "python"),
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        KAGGLE_KERNEL_RUN_TYPE=os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "Batch"),
    )
    probe = [
        "tools/engine_probe.py",
        "--prompts",
        "probe/prompts.jsonl",
        "--manifest",
        "manifests/model-research.json",
    ]
    hf_python = "runtimes/kaggle/.venv/bin/python"
    hf_steps = [
        ["uv", "sync", "--project", str(runtime), "--frozen"],
        [
            hf_python,
            *probe,
            "--engine",
            "hf",
            "--kernels",
            "torch",
            "--only",
            "sweep",
            "--device",
            "0",
            "--cache-root",
            str(root / "cache-hf"),
            "--output",
            str(output / "hf-torch.json"),
        ],
        ["uv", "pip", "install", "--python", hf_python, "--no-deps", *CONFIG["fla_packages"]],
        [
            hf_python,
            *probe,
            "--engine",
            "hf",
            "--kernels",
            "fla",
            "--device",
            "0",
            "--cache-root",
            str(root / "cache-hf"),
            "--output",
            str(output / "hf-fla.json"),
        ],
    ]
    vllm_python = str(root / "vllm-env/bin/python")
    vllm_steps = [
        ["uv", "venv", str(root / "vllm-env"), "--python", "3.12"],
        ["uv", "pip", "install", "--python", vllm_python, *CONFIG["vllm_packages"]],
        [
            vllm_python,
            *probe,
            "--engine",
            "vllm",
            "--device",
            "1",
            "--cache-root",
            str(root / "cache-vllm"),
            "--output",
            str(output / "vllm.json"),
        ],
    ]
    report = {"config": CONFIG, "gpus": gpus, "source_bundle_sha256": CHECKSUM, "paths": {}}

    def checkpoint():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def start(name, steps):
        """Run one path's steps in order inside a child shell; paths run concurrently."""
        script = " && ".join(" ".join(json.dumps(part) for part in step) for step in steps)
        log = (output / f"{name}.log").open("w")
        return subprocess.Popen(
            ["bash", "-c", script],
            cwd=project,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    print("Estimated total cash cost $0 on free Kaggle quota; Modal unused.", flush=True)
    processes = {"hf": start("hf", hf_steps), "vllm": start("vllm", vllm_steps)}
    try:
        for name, process in processes.items():
            try:
                report["paths"][name] = process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                report["paths"][name] = "deadline"
            checkpoint()
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        for name in ("hf", "vllm"):
            log = output / f"{name}.log"
            if log.exists() and log.stat().st_size > 400_000:
                text = log.read_text(errors="replace")
                log.write_text(text[:100_000] + "\n...\n" + text[-250_000:])
        checkpoint()


if __name__ == "__main__":
    main()
