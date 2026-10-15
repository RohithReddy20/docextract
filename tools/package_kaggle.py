"""Build an explicit source/input archive. This does not upload or launch anything."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from docextract.contracts import save_json


def package(root: Path, cases: Path, output: Path):
    packet = json.loads(cases.read_text())
    if packet.get("version") != "inference-cases/1" or set(packet) != {"version", "cases"}:
        raise ValueError("Expected inference-only cases")
    entries = {}
    paths = [
        *sorted((root / "src/docextract").rglob("*.py")),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "LICENSE",
        root / "README.md",
        root / "docs/kaggle-trial.md",
        *sorted((root / "notices").glob("*.txt")),
        root / "runtimes/kaggle/pyproject.toml",
        root / "runtimes/kaggle/uv.lock",
        root / "runtimes/kaggle/bootstrap.py",
        root / "manifests/model-research.json",
        root / "tools/kaggle_trial.py",
    ]
    for path in paths:
        if path.name == "example.py":
            continue
        if path.is_symlink():
            raise ValueError("Refusing a symlink in the source archive")
        entries["docextract/" + str(path.relative_to(root))] = path
    entries["docextract/inference/cases.json"] = cases
    for case in packet["cases"]:
        if set(case) != {"id", "pdf", "schema"}:
            raise ValueError("Inference manifest contains unexpected metadata")
        for key, extension in (("pdf", ".pdf"), ("schema", ".json")):
            path = (cases.parent / case[key]).resolve()
            if not path.is_relative_to(cases.parent.resolve()) or path.suffix != extension:
                raise ValueError("Input path escapes the packet or has the wrong type")
            relative = str(path.relative_to(cases.parent.resolve()))
            entries["docextract/inference/" + relative] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []

    def metadata(info):
        info.uid = info.gid = info.mtime = 0
        info.uname = info.gname = ""
        return info

    with tarfile.open(output, "w:gz") as archive:
        for name, path in sorted(entries.items()):
            manifest.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            archive.add(path, arcname=name, recursive=False, filter=metadata)
    save_json(
        output.with_suffix(".manifest.json"),
        {"files": manifest, "includes_answers": False, "includes_weights": False},
    )
    return {"archive": str(output), "bytes": output.stat().st_size, "files": len(manifest)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package(Path(__file__).resolve().parents[1], args.cases, args.output))
