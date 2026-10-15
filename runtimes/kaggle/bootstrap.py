"""Template filled with an allowlisted archive; run only as a private Kaggle script."""

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

PAYLOAD = "__DOCEEXTRACT_PAYLOAD__"
CHECKSUM = "__DOCEEXTRACT_CHECKSUM__"
EXPERIMENT = "__DOCEEXTRACT_EXPERIMENT__"


def main():
    if platform.system() != "Linux" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("This bootstrap runs only in Kaggle; no local fallback")
    if not shutil.which("uv"):
        raise RuntimeError("Kaggle must provide uv before the trial can run")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if not gpu or any("T4" not in name for name in gpu):
        raise RuntimeError("The trial requests T4s and will not use a different accelerator")
    payload = base64.b64decode(PAYLOAD, validate=True)
    if hashlib.sha256(payload).hexdigest() != CHECKSUM:
        raise RuntimeError("Source archive hash mismatch")
    root = Path("/kaggle/temp") / ("docextract-" + CHECKSUM[:12])
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        bundle.extractall(root, filter="data")
    project = root / "docextract"
    runtime = project / "runtimes/kaggle"
    output = Path("/kaggle/working/docextract-smoke")
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "cash_cost_estimate_usd": 0.0,
        "source_bundle_sha256": CHECKSUM,
        "active_devices_per_stage": 1,
        "gpus_assigned": gpu,
        "session_worker_limit_seconds": 3300,
        "stages": {},
        "complete": False,
        "experiment": EXPERIMENT,
    }
    deadline = time.monotonic() + 3300
    environment = dict(
        os.environ,
        UV_CACHE_DIR=str(root / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(root / "python"),
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )

    def checkpoint():
        temporary = output / "report.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "report.json")

    def run(*commands):
        """Run independent commands concurrently, one per assigned device, under the global deadline."""
        remaining = int(deadline - time.monotonic())
        if remaining < 5:
            raise TimeoutError("Global Kaggle trial limit reached")
        processes = [
            subprocess.Popen(command, cwd=project, env=environment, start_new_session=True)
            for command in commands
        ]
        try:
            codes = [process.wait(timeout=max(1, deadline - time.monotonic())) for process in processes]
        except BaseException:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
        if any(codes):
            raise RuntimeError(f"Trial subprocess exited with {codes}")

    def export(stage, source):
        target = output / stage
        for path in sorted(source.rglob("*.json")):
            if path.name not in {
                "runtime.json",
                "summary.json",
                "telemetry.json",
                "value-result.json",
                "result.json",
                "diagnostic.json",
                "proposals.json",
            }:
                continue
            if path.stat().st_size > 2_000_000:
                raise RuntimeError("Trial result exceeds the small-artifact export limit")
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)

    checkpoint()
    try:
        print(
            "Estimated total cash cost $0 on free Kaggle quota; Modal unused. "
            "All weights and caches stay in this remote worker.",
            flush=True,
        )
        run(["uv", "sync", "--project", str(runtime), "--frozen"])
        shutil.copyfile(runtime / "uv.lock", output / "runtime.uv.lock")
        packet = json.loads((project / "inference/cases.json").read_text())
        if len(packet["cases"]) != 24:
            raise RuntimeError("Expected the frozen 24-case authored packet")
        indices = {
            "bindings": (0, 8, 20),
            "bindings-all": tuple(range(24)),
            "cited-all": tuple(range(24)),
        }.get(EXPERIMENT, (0, 4, 8, 12, 16, 20))
        selected = [packet["cases"][index] for index in indices]
        smoke = project / "inference/smoke-cases.json"
        parity = project / "inference/parity-case.json"
        smoke.write_text(json.dumps({"version": packet["version"], "cases": selected}))
        parity.write_text(json.dumps({"version": packet["version"], "cases": selected[:1]}))
        report["diagnostic_case_ids"] = [case["id"] for case in selected]
        cache_a, cache_b = root / "model-cache-a", root / "model-cache-b"

        mode = EXPERIMENT.split("-")[0] if EXPERIMENT.startswith(("bindings", "cited")) else "values"

        def command(name, arm, wrapper, cases, cache, cache_only=False, device=0):
            arguments = [
                "uv",
                "run",
                "--project",
                str(runtime),
                "--no-sync",
                "python",
                "tools/kaggle_trial.py",
                "--cases",
                str(cases),
                "--output",
                str(root / "runs" / name),
                "--model-manifest",
                str(project / "manifests/model-research.json"),
                "--arm",
                arm,
                "--mode",
                mode,
                "--wrapper",
                wrapper,
                "--device",
                str(device),
                "--worker",
                "--cache-root",
                str(cache),
                "--max-input-tokens",
                {"bindings": "6144", "cited": "8192"}.get(mode, "2048"),
                "--max-new-tokens",
                {"bindings": "1536", "cited": "2048"}.get(mode, "256"),
            ]
            return arguments + (["--cache-only"] if cache_only else [])

        def stages(*specs):
            """Each spec is (name, arm, wrapper, cases, cache[, cache_only[, device]])."""
            started = time.monotonic()
            for spec in specs:
                report["stages"][spec[0]] = {
                    "status": "running",
                    "cases": 1 if spec[3] == parity else len(selected),
                }
            checkpoint()
            try:
                run(*(command(*spec) for spec in specs))
                for spec in specs:
                    report["stages"][spec[0]]["status"] = "complete"
            except BaseException:
                for spec in specs:
                    report["stages"][spec[0]]["status"] = "failed"
                raise
            finally:
                for spec in specs:
                    report["stages"][spec[0]]["seconds"] = time.monotonic() - started
                    if (root / "runs" / spec[0]).exists():
                        export(spec[0], root / "runs" / spec[0])
                checkpoint()
            return [root / "runs" / spec[0] for spec in specs]

        def stage(name, arm, wrapper, cases, cache, cache_only=False):
            return stages((name, arm, wrapper, cases, cache, cache_only, 0))[0]

        if EXPERIMENT == "bindings":
            stage("a-bindings", "a", "text", smoke, cache_a)
            shutil.rmtree(cache_a)
            stage("b-bindings", "b", "text", smoke, cache_b)
            report["complete"] = True
            return
        if EXPERIMENT in ("bindings-all", "cited-all"):
            if len(gpu) < 2:
                raise RuntimeError("The parallel A/B experiment needs two independent T4s")
            stages(
                (f"a-{mode}", "a", "text", smoke, cache_a, False, 0),
                (f"b-{mode}", "b", "text", smoke, cache_b, False, 1),
            )
            report["complete"] = True
            return
        full = stage("a-full-parity", "a", "multimodal", parity, cache_a)
        text = stage("a-text-parity", "a", "text", parity, cache_a, cache_only=True)
        full_tokens = json.loads((full / "telemetry.json").read_text())["tasks"][0]
        text_tokens = json.loads((text / "telemetry.json").read_text())["tasks"][0]
        report["text_wrapper_parity"] = {
            key: full_tokens[key] == text_tokens[key]
            for key in ("input_token_ids_sha256", "output_token_ids_sha256")
        }
        checkpoint()
        if not all(report["text_wrapper_parity"].values()):
            raise RuntimeError(
                "Text wrapper changed the controlled input or greedy output; comparison stopped"
            )
        stage("a-smoke", "a", "text", smoke, cache_a, cache_only=True)
        # Only this job's generated cache is removed, after its model processes have exited.
        shutil.rmtree(cache_a)
        report["a_cache_removed_remotely_after_use"] = True
        stage("b-smoke", "b", "text", smoke, cache_b)
        report["complete"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        checkpoint()
        raise
    finally:
        checkpoint()


if __name__ == "__main__":
    main()
