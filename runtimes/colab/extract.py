"""Run the product on a Colab GPU: tools/extract_remote.py over chosen documents of the dataset.

A run configuration (as in manifests/modal/) gives the model, engine sizes, OCR and prompt variants,
passed to tools/extract_remote.py as runtimes/modal/final_run.py passes them to its GPU stage. PDFs
and schemas come from the pinned ExtractBench dataset (never its answers). The product stores every
answer in `<out>/run/generations.jsonl` as it finishes, so a rerun over the same folder asks only
what is left. At the end `<out>/answers.json` holds each finished document's data in the form
tools/score_probe.py reads (one answer per document, method "product").

    python runtimes/colab/extract.py --config manifests/colab/<name>.json --out /content/out/<name>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "runtimes" / "colab")]

from probe import dataset_file, interrupt  # noqa: E402  (the probe runner's dataset access)


def packet(ids: list[str], cache: Path) -> Path:
    """An inference-only packet (PDFs, schemas and cases.json) for `ids`, read from the dataset's
    split files; only each case's PDF path and schema are kept from its row."""
    from docextract.contracts import digest

    inference = cache / "packet" / "inference"
    inference.mkdir(parents=True, exist_ok=True)
    wanted, cases = set(ids), {}
    for split in ("short", "medium", "long"):
        with dataset_file(f"{split}.jsonl", cache).open(encoding="utf-8") as lines:
            for line in lines:
                row = json.loads(line) if line.strip() else None
                case_id = digest(row["id"])[:16] if row else None
                if case_id in wanted:
                    shutil.copyfile(dataset_file(row["pdf"], cache), inference / f"{case_id}.pdf")
                    (inference / f"{case_id}.schema.json").write_text(row["data_schema"])
                    names = {"pdf": f"{case_id}.pdf", "schema": f"{case_id}.schema.json"}
                    cases[case_id] = {"id": case_id, **names}
    if wanted - cases.keys():
        raise SystemExit(f"Not in the dataset: {sorted(wanted - cases.keys())}")
    manifest = inference / "cases.json"
    manifest.write_text(json.dumps({"version": "inference-cases/1", "cases": [cases[i] for i in ids]}))
    return manifest


def arguments(config: dict) -> list[str]:
    """tools/extract_remote.py's options for a run configuration (the GPU stage of final_run.py)."""
    units = config.get("max_window_units")
    return [
        "--engine",
        "vllm",
        "--arm",
        config["arm"],
        "--dtype",
        config["dtype"],
        "--max-window-chars",
        str(config["max_window_chars"]),
        "--max-schema-chars",
        str(config["max_schema_chars"]),
        *(["--max-window-units", str(units)] if units else []),
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
        json.dumps(config["variants"]),
        *(["--quantization", config["quantization"]] if config.get("quantization") else []),
        *(["--max-document-pages", str(pages)] if (pages := config.get("max_document_pages")) else []),
    ]


def answers(run: Path, ids: list[str]) -> list[dict]:
    """Each finished document's extracted data as one answer for tools/score_probe.py."""
    rows = []
    for case_id in ids:
        path = run / case_id / "result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rows.append(
            {
                "id": case_id,
                "variant": "product",
                "batch": 0,
                "batches": 1,
                "finish_reason": "stop",
                "raw_text": json.dumps(result["data"]),
                "input_tokens": result["usage"]["input_tokens"],
                "output_tokens": result["usage"]["output_tokens"],
                "status": result["completion"]["status"],
                "unresolved": [item["reason"] for item in result["unresolved"]][:20],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="a run configuration")
    parser.add_argument("--out", type=Path, required=True, help="where results go")
    parser.add_argument("--minutes", type=float, help="stop the product after this long")
    parser.add_argument("--engine", type=json.loads, default={}, help="configuration values to override")
    parser.add_argument("--cache", type=Path, default=Path("/content/cache"))
    parser.add_argument("--dry-run", action="store_true", help="build the packet only, without a GPU")
    args = parser.parse_args()

    started = time.time()
    config = {**json.loads(args.config.read_text()), **args.engine}
    manifest = packet(config["ids"], args.cache / "extractbench")
    # The product runs as its own process, from this source tree.
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])),
    }
    if args.dry_run:
        subprocess.run(
            ["python", "-c", "import docextract.batch, docextract.vllm_engine"], cwd=ROOT, env=env, check=True
        )
        print(f"dry run: packet of {len(config['ids'])} documents, options {' '.join(arguments(config))}")
        return
    run = args.out / "run"
    run.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "-u",
        "tools/extract_remote.py",
        "--cases",
        str(manifest),
        "--output",
        str(run),
        "--model-manifest",
        "manifests/model-research.json",
        "--cache-root",
        str(args.cache / "models"),
        *arguments(config),
    ]
    # Its own process group, so stopping it also stops vLLM's engine process (which otherwise keeps
    # the GPU's memory).
    product = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    # The notebook's stop button and the driver's stop end this process with SIGTERM; the product
    # is stopped too, and what finished stays stored.
    signal.signal(signal.SIGTERM, interrupt)
    try:
        product.wait(timeout=args.minutes * 60 - (time.time() - started) if args.minutes else None)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        print("stopping the product: time limit or stop", flush=True)
        os.killpg(product.pid, signal.SIGTERM)
        try:
            product.wait(timeout=120)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(product.pid, signal.SIGKILL)  # anything of it still running
        except ProcessLookupError:
            pass
    rows = answers(run, config["ids"])
    (args.out / "answers.json").write_text(json.dumps({"outputs": rows}, indent=1))
    minutes = (time.time() - started) / 60
    print(f"{len(rows)} of {len(config['ids'])} documents finished in {minutes:.1f} minutes")
    if product.returncode:
        raise SystemExit(f"the product exited with code {product.returncode}")


if __name__ == "__main__":
    main()
