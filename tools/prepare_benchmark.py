"""Pinned dataset preparation. Run --download on a remote CPU worker, not a laptop."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

from docextract.contracts import save_json

DATASET = "llamaindex/ExtractBench"
REVISION = "51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc"


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def metadata() -> dict:
    url = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}"
    with urlopen(url, timeout=30) as response:
        entries = json.load(response)
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "mode": "metadata_only",
        "files": [
            {"path": entry["path"], "bytes": entry["size"], "sha256": entry["lfs"]["oid"]}
            for entry in entries
            if entry["path"].endswith(".jsonl") and "lfs" in entry
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--download", action="store_true", help="remote worker only: download pinned corpus")
    parser.add_argument(
        "--schemas-only", action="store_true", help="download JSONL splits without PDF documents"
    )
    args = parser.parse_args()
    if args.schemas_only and not args.download:
        parser.error("--schemas-only requires --download")
    if args.download:
        from docextract.vllm_engine import require_remote

        require_remote()
    manifest = metadata()
    if args.download:
        if args.schemas_only:
            snapshot = args.output / "snapshot"
            snapshot.mkdir(parents=True, exist_ok=True)
            for entry in manifest["files"]:
                destination = snapshot / entry["path"]
                temporary = destination.with_suffix(".part")
                url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}"
                with urlopen(url, timeout=30) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if temporary.stat().st_size != entry["bytes"] or sha256(temporary) != entry["sha256"]:
                    raise ValueError(f"Dataset hash mismatch: {entry['path']}")
                temporary.replace(destination)
        else:
            from huggingface_hub import snapshot_download

            snapshot = Path(
                snapshot_download(
                    repo_id=DATASET,
                    repo_type="dataset",
                    revision=REVISION,
                    allow_patterns=["*.jsonl", "docs/**"],
                    local_dir=args.output / "snapshot",
                )
            )
        files = []
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                checksum = sha256(path)
                expected = next(
                    (
                        entry["sha256"]
                        for entry in manifest["files"]
                        if entry["path"] == path.relative_to(snapshot).as_posix()
                    ),
                    None,
                )
                if expected and checksum != expected:
                    raise ValueError(f"Dataset hash mismatch: {path.name}")
                files.append(
                    {
                        "path": path.relative_to(snapshot).as_posix(),
                        "sha256": checksum,
                        "bytes": path.stat().st_size,
                    }
                )
        manifest = manifest | {"mode": "downloaded_remote_snapshot", "files": files}
    save_json(args.output / "dataset-manifest.json", manifest)
    print(json.dumps({"revision": REVISION, "mode": manifest["mode"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
