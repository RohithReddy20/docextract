"""Prepare a private Kaggle engine probe kernel. Submission is a separate command."""

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

from tools.build_probe_prompts import build as build_prompts

SOURCES = (
    "pyproject.toml",
    "uv.lock",
    "LICENSE",
    "README.md",
    "runtimes/kaggle/pyproject.toml",
    "runtimes/kaggle/uv.lock",
    "manifests/model-research.json",
    "tools/engine_probe.py",
)
CONFIG = {
    "seconds": 3300,
    "fla_packages": ["fla-core==0.5.2", "flash-linear-attention==0.5.2", "einops==0.8.1"],
    "vllm_packages": ["vllm==0.30.0"],
}


def build(root: Path, kernel_id: str, output: Path, config=CONFIG) -> dict:
    owner, slug = kernel_id.split("/", 1)
    if not owner or not slug or any(not (c.isalnum() or c == "-") for c in owner + slug):
        raise ValueError("Expected a Kaggle owner/kernel-slug")
    prompts = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in build_prompts(root / "artifacts/diagnostics")
    ).encode()
    paths = [*sorted((root / "src/docextract").rglob("*.py")), *(root / name for name in SOURCES)]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        for path in paths:
            if path.is_symlink() or path.name == "example.py":
                continue
            info = bundle.gettarinfo(str(path), "docextract/" + str(path.relative_to(root)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as source:
                bundle.addfile(info, source)
        info = tarfile.TarInfo("docextract/probe/prompts.jsonl")
        info.size = len(prompts)
        bundle.addfile(info, io.BytesIO(prompts))
    payload = buffer.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    template = (root / "runtimes/kaggle/probe.py").read_text()
    code = (
        template.replace("__DOCEXTRACT_CHECKSUM__", checksum)
        .replace("__DOCEXTRACT_CONFIG__", json.dumps(config))
        .replace("__DOCEXTRACT_PAYLOAD__", base64.b64encode(payload).decode())
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "probe.py").write_text(code)
    metadata = {
        "id": kernel_id,
        "title": slug.replace("-", " "),
        "code_file": "probe.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"source_bundle_sha256": checksum, "bytes": len(payload), "estimated_cash_cost_usd": 0.0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[1], args.kernel_id, args.output))
