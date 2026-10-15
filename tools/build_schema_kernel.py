"""Prepare a private, CPU-only Kaggle schema audit; no submission or model dependencies."""

import argparse
import base64
import hashlib
import io
import json
import tarfile
from pathlib import Path

BOOTSTRAP = """import base64, hashlib, io, json, os, platform, subprocess, tarfile, time
from pathlib import Path
if platform.system() != "Linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
    raise RuntimeError("Kaggle only")
payload = base64.b64decode("PAYLOAD", validate=True)
assert hashlib.sha256(payload).hexdigest() == "CHECKSUM"
root = Path("/kaggle/temp/docextract-schema-audit")
root.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
    bundle.extractall(root, filter="data")
output = Path("/kaggle/working/schema-audit")
output.mkdir(parents=True, exist_ok=True)
environment = dict(os.environ, UV_CACHE_DIR=str(root / "uv-cache"),
                   UV_PYTHON_INSTALL_DIR=str(root / "python"))
deadline = time.monotonic() + 1500
def run(arguments):
    subprocess.run(["uv", "run", "--frozen", "--no-dev", "python", *arguments], cwd=root,
                   env=environment, check=True, timeout=max(1, deadline-time.monotonic()))
run(["tools/prepare_benchmark.py", "--download", "--schemas-only", "--output", str(root / "dataset")])
run(["tools/schema_inventory.py", "--snapshot", str(root / "dataset/snapshot"),
     "--output", str(output / "schema-inventory.json")])
(output / "dataset-manifest.json").write_bytes((root / "dataset/dataset-manifest.json").read_bytes())
(output / "audit-source.json").write_text(json.dumps({
    "source_bundle_sha256":"CHECKSUM", "models_loaded":False,
    "pdfs_downloaded":False, "dataset_jsonl_exported":False, "estimated_cash_cost_usd":0}))
"""


def build(root: Path, kernel_id: str, output: Path):
    parts = kernel_id.split("/")
    if len(parts) != 2 or any(not part or not all(c.isalnum() or c == "-" for c in part) for part in parts):
        raise ValueError("Expected owner/kernel-slug")
    paths = [
        *sorted((root / "src/docextract").rglob("*.py")),
        *(root / name for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE")),
        root / "tools/prepare_benchmark.py",
        root / "tools/schema_inventory.py",
    ]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in paths:
            if path.is_symlink():
                raise ValueError("Source symlinks are not allowed")
            if path.name == "example.py":
                continue
            info = archive.gettarinfo(str(path), str(path.relative_to(root)))
            info.uid = info.gid = info.mtime = 0
            info.uname = info.gname = ""
            with path.open("rb") as source:
                archive.addfile(info, source)
    payload = buffer.getvalue()
    checksum = hashlib.sha256(payload).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    code = BOOTSTRAP.replace("PAYLOAD", base64.b64encode(payload).decode()).replace("CHECKSUM", checksum)
    (output / "audit.py").write_text(code)
    metadata = {
        "id": kernel_id,
        "title": parts[1].replace("-", " "),
        "code_file": "audit.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"source_bundle_sha256": checksum, "submitted": False, "estimated_cash_cost_usd": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build(Path(__file__).resolve().parents[1], args.kernel_id, args.output))
