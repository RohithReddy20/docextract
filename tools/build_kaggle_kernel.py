"""Prepare a private Kaggle smoke kernel. Submission is a separate explicit command."""

import argparse
import base64
import hashlib
import json
import tarfile
from pathlib import Path


def build(archive: Path, kernel_id: str, output: Path, *, experiment="values"):
    if experiment not in ("values", "bindings", "bindings-all", "cited-all"):
        raise ValueError("Unknown experiment")
    owner, slug = kernel_id.split("/", 1)
    if not owner or not slug or "/" in slug or any(not (c.isalnum() or c == "-") for c in owner + slug):
        raise ValueError("Expected a Kaggle owner/kernel-slug")
    content = archive.read_bytes()
    if len(content) > 2_000_000:
        raise ValueError("Only small source/authored-input archives may be embedded")
    manifest = json.loads(archive.with_suffix(".manifest.json").read_text())
    expected = {entry["path"]: entry for entry in manifest["files"]}
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {member.name for member in members} != expected.keys():
            raise ValueError("Archive file list differs from its allowlist")
        for member in members:
            path = Path(member.name)
            if (
                not member.isfile()
                or ".." in path.parts
                or path.is_absolute()
                or path.suffix in (".safetensors", ".gguf", ".pt", ".pth")
                or any(part in ("private", "credentials", "evaluation", ".git") for part in path.parts)
            ):
                raise ValueError("Forbidden file in the inference archive")
            data = bundle.extractfile(member).read()
            if (
                len(data) != expected[member.name]["bytes"]
                or hashlib.sha256(data).hexdigest() != expected[member.name]["sha256"]
            ):
                raise ValueError("Archive content differs from its source manifest")
    checksum = hashlib.sha256(content).hexdigest()
    template = Path(__file__).resolve().parents[1] / "runtimes/kaggle/bootstrap.py"
    code = template.read_text().replace("__DOCEEXTRACT_PAYLOAD__", base64.b64encode(content).decode())
    code = code.replace("__DOCEEXTRACT_CHECKSUM__", checksum)
    code = code.replace("__DOCEEXTRACT_EXPERIMENT__", experiment)
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke.py").write_text(code)
    metadata = {
        "id": kernel_id,
        "title": slug.replace("-", " "),
        "code_file": "smoke.py",
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
    return {
        "source_bundle_sha256": checksum,
        "private": True,
        "estimated_cash_cost_usd": 0.0,
        "submitted": False,
        "gpu_jobs_launched": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment", choices=("values", "bindings", "bindings-all", "cited-all"), default="values"
    )
    args = parser.parse_args()
    print(build(args.archive, args.kernel_id, args.output, experiment=args.experiment))
