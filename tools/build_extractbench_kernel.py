"""Prepare a private Kaggle ExtractBench development kernel. Submission is a separate command."""

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

SOURCES = (
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "README.md",
    "runtimes/kaggle/pyproject.toml",
    "runtimes/kaggle/uv.lock",
    "manifests/model-research.json",
    "tools/kaggle_trial.py",
    "tools/extract_remote.py",
    "tools/prepare_benchmark.py",
    "tools/extractbench_packet.py",
    "tools/score_extractbench.py",
    "tools/form_inventory.py",
)


def archive(root: Path, authored: bool = False) -> bytes:
    paths = [*sorted((root / "src/docextract").rglob("*.py")), *sorted((root / "notices").glob("*.txt"))]
    paths += [root / name for name in SOURCES]
    if authored:
        # Inference inputs only (PDFs, schemas, manifest); answers stay local for scoring.
        inference = root / "artifacts/diagnostics/inference"
        paths += sorted(path for path in inference.iterdir() if path.suffix in (".pdf", ".json"))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for path in paths:
            if path.is_symlink() or path.suffix in (".safetensors", ".gguf", ".pt", ".pth"):
                raise ValueError(f"Refusing {path}")
            if path.name == "example.py":
                continue
            relative = str(path.relative_to(root)).replace(
                "artifacts/diagnostics/inference", "authored/inference"
            )
            info = bundle.gettarinfo(str(path), "docextract/" + relative)
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as source:
                bundle.addfile(info, source)
    return buffer.getvalue()


def build(root: Path, kernel_id: str, output: Path, config: dict) -> dict:
    owner, slug = kernel_id.split("/", 1)
    if not owner or not slug or any(not (c.isalnum() or c == "-") for c in owner + slug):
        raise ValueError("Expected a Kaggle owner/kernel-slug")
    if any(run["engine"] not in ("hf", "vllm") or run["arm"] not in ("a", "b") for run in config["runs"]):
        raise ValueError("Unsupported run configuration")
    if len({run["device"] for run in config["runs"]}) != len(config["runs"]):
        raise ValueError("Each run needs its own GPU")
    sources = config.get("caches_from", [])
    if any(source.count("/") != 1 for source in sources):
        raise ValueError("caches_from lists earlier kernels as owner/slug")
    payload = archive(root, authored=config.get("packet") == "authored")
    checksum = hashlib.sha256(payload).hexdigest()
    template = (root / "runtimes/kaggle/extractbench.py").read_text()
    # Short placeholders first: base64 text could contain a placeholder name.
    code = (
        template.replace("__DOCEXTRACT_CHECKSUM__", checksum)
        .replace("__DOCEXTRACT_CONFIG__", json.dumps(config))
        .replace("__DOCEXTRACT_PAYLOAD__", base64.b64encode(payload).decode())
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "extractbench.py").write_text(code)
    metadata = {
        "id": kernel_id,
        "title": slug.replace("-", " "),
        "code_file": "extractbench.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": bool(config["runs"]),
        "enable_internet": True,
        **({"machine_shape": "NvidiaTeslaT4"} if config["runs"] else {}),
        "dataset_sources": [],
        "competition_sources": [],
        # Earlier runs whose outputs hold reusable wheels and compiled GPU kernels (no data or answers).
        "kernel_sources": sources,
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"source_bundle_sha256": checksum, "bytes": len(payload), "estimated_cash_cost_usd": 0.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=json.loads, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[1], args.kernel_id, args.output, args.config))
