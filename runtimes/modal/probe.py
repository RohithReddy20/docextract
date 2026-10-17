"""Small extraction probes on one Modal L4: a few documents, several prompt methods, one model load.

Each method is a request recipe. `baseline` reproduces ExtractBench's one-shot recipe for self-hosted
vLLM models (the leaderboard's Qwen3.5-4B entry): every page rendered at 150 DPI, the full schema
pretty-printed, JSON-object output, temperature 0. Other methods change one thing at a time.

Requests are built locally with ExtractBench's own prompt helpers (`tools/probe_jobs.py`); this
entrypoint states the worst-case cost, runs them in one GPU call and saves the raw answers next to the
request file. The GPU worker reads only PDFs from the work volume; answers are scored locally
(`tools/score_probe.py`), so expected outputs never reach it.

    uv run python tools/probe_jobs.py manifests/probes/<name>.json
    modal run runtimes/modal/probe.py::main --jobs artifacts/probes/<name>/jobs.json --confirm-usd <estimate>
    modal run runtimes/modal/probe.py::evidence --answers artifacts/probes/<name>/final-answers.json
    uv run python tools/score_probe.py manifests/probes/<name>.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

SRC = "/opt/docextract"
WORK_DIR = "/work"
L4_PER_HOUR = 1.18  # L4 with 4 cores and 24 GiB, Modal list prices (see final_run.py)
work = modal.Volume.from_name("docextract-work")
app = modal.App("docextract-probe")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "vllm==0.30.0",
        "python-doctr==1.1.0",
        "jsonschema>=4.23.0,<5",
        "pydantic>=2.11.9,<3",
        "pypdfium2==4.30.0",
    )
    .env(
        {
            "PYTHONPATH": f"{SRC}/src",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HOME": f"{WORK_DIR}/model-cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "TRITON_CACHE_DIR": f"{WORK_DIR}/compile-cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": f"{WORK_DIR}/compile-cache/inductor",
            "VLLM_CACHE_ROOT": f"{WORK_DIR}/compile-cache/vllm",
        }
    )
)
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    image = image.add_local_dir(root / "src", f"{SRC}/src").add_local_dir(
        root / "manifests", f"{SRC}/manifests"
    )


def render_pages(pdf: Path, dpi: int) -> list[str]:
    """Every page as a base64 PNG at `dpi` (the benchmark renders with poppler; PDFium here)."""
    import base64
    import io
    from contextlib import closing

    import pypdfium2 as pdfium

    pages = []
    with closing(pdfium.PdfDocument(str(pdf))) as document:
        for index in range(len(document)):
            bitmap = document[index].render(scale=dpi / 72)
            buffer = io.BytesIO()
            bitmap.to_pil().convert("RGB").save(buffer, format="PNG")
            pages.append(base64.b64encode(buffer.getvalue()).decode())
    return pages


def page_texts(pdf: Path) -> list[str]:
    from contextlib import closing

    import pypdfium2 as pdfium

    with closing(pdfium.PdfDocument(str(pdf))) as document:
        texts = []
        for index in range(len(document)):
            with closing(document[index].get_textpage()) as text:
                texts.append(text.get_text_bounded())
        return texts


@app.function(image=image, gpu="L4", cpu=4, memory=24 * 1024, volumes={WORK_DIR: work})
def generate(jobs: list[dict], engine: dict, name: str, deadline: float) -> dict:
    """Answer every job in one continuous batch; each answer is saved as it finishes.

    Prompts are rendered per job (its own chat-template options) and run together, so a slow answer
    holds up nothing else. Answers stream to `runs/probes/<name>/answers.jsonl` in the work volume, and
    the batch stops at `deadline` (epoch seconds), returning what finished.
    """
    import sys

    sys.path.insert(0, f"{SRC}/src")
    from docextract.remote import pinned_checkpoint

    started = time.time()
    _, _, checkpoint = pinned_checkpoint(
        engine.get("arm", "a"),
        Path(SRC) / "manifests/model-research.json",
        Path(WORK_DIR) / "model-cache",
        offline=True,
    )
    import base64
    import io

    from PIL import Image
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    llm = LLM(
        model=str(checkpoint),
        dtype="bfloat16",
        max_model_len=engine["max_model_len"],
        gpu_memory_utilization=0.92,
        max_num_seqs=32,
        seed=0,
        limit_mm_per_prompt={"image": engine["max_images"], "video": 0},
        enable_prefix_caching=False,
        quantization=engine.get("quantization"),
        **(
            {"speculative_config": {"method": "mtp", "num_speculative_tokens": engine["speculative_tokens"]}}
            if engine.get("speculative_tokens")
            else {}
        ),
    )
    loaded = time.time()
    tokenizer = llm.get_tokenizer()
    rendered: dict[tuple[str, int], list] = {}
    texts: dict[str, list[str]] = {}
    sink = Path(WORK_DIR) / "runs" / "probes" / name / "answers.jsonl"
    sink.parent.mkdir(parents=True, exist_ok=True)
    sink.write_text("")
    outputs, pending = [], {}
    number = 0
    for job in jobs:
        pdf = Path(WORK_DIR) / job["pdf"]
        if (job["id"], job["dpi"]) not in rendered:
            rendered[job["id"], job["dpi"]] = [
                Image.open(io.BytesIO(base64.b64decode(page))).convert("RGB")
                for page in render_pages(pdf, job["dpi"])
            ]
        every = rendered[job["id"], job["dpi"]]
        if job.get("text_layer") and job["id"] not in texts:
            texts[job["id"]] = page_texts(pdf)
        # A document too long for one request is read in batches of consecutive pages; the scorer
        # merges the batch answers. Otherwise the whole document is one batch.
        size = job.get("pages_per_request") or len(every)
        spans = [(start, min(start + size, len(every))) for start in range(0, len(every), size)]
        for batch, (first, last) in enumerate(spans):
            pages = every[first:last]
            text = job["user_text"]
            if len(spans) > 1:
                text += (
                    f"\n\nThe images are pages {first + 1}-{last} of a {len(every)}-page document. Extract "
                    "what these pages show; use null (or an empty list) for anything not on them."
                )
            if job.get("text_layer"):
                layer = "\n\n".join(
                    f"--- page {first + i + 1} ---\n{t.strip()}"
                    for i, t in enumerate(texts[job["id"]][first:last])
                )
                text = f"{text}\n\nText layer of the document (may be incomplete; trust the images):\n{layer}"
            messages = [
                {"role": "system", "content": job["system"]},
                {
                    "role": "user",
                    "content": [*({"type": "image"} for _ in pages), {"type": "text", "text": text}],
                },
            ]
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **job.get("template_options", {})
            )
            structured = (
                StructuredOutputsParams(
                    json_object=True, disable_any_whitespace=job.get("compact_json", True)
                )
                if job.get("json_object")
                else None
            )
            params = SamplingParams(
                temperature=0.0, max_tokens=job["max_tokens"], structured_outputs=structured
            )
            row = {
                "id": job["id"],
                "variant": job["variant"],
                "batch": batch,
                "batches": len(spans),
                "pages": [first + 1, last],
                "template_tail": prompt[-40:],
            }
            key = str(number)
            number += 1
            # Too long for the context window: shrink the page images (as ExtractBench's vLLM harness
            # does, by a quarter per round, at most four rounds) and try again.
            added, internal = False, None
            for rounds in range(5):
                try:
                    internal = llm.llm_engine.add_request(
                        key, {"prompt": prompt, "multi_modal_data": {"image": pages}}, params
                    )
                    row["downscaled"] = rounds
                    added = True
                    break
                except Exception as error:  # for example a prompt longer than the context window
                    too_long = any(
                        marker in str(error) for marker in ("maximum context length", "maximum model length")
                    )
                    if not too_long or rounds == 4:
                        row.update(
                            raw_text="",
                            finish_reason=f"error: {str(error)[:300]}",
                            input_tokens=0,
                            output_tokens=0,
                        )
                        break
                    pages = [
                        page.resize((max(64, page.width * 3 // 4), max(64, page.height * 3 // 4)))
                        for page in pages
                    ]
            if not added:
                outputs.append(row)
                continue
            pending[key] = row
            if isinstance(internal, str):
                pending[internal] = row
    last_commit = time.time()
    while llm.llm_engine.has_unfinished_requests() and time.time() < deadline:
        for output in llm.llm_engine.step():
            if not output.finished:
                continue
            key = output.request_id if output.request_id in pending else output.request_id.rsplit("-", 1)[0]
            row = pending[key]
            completion = output.outputs[0]
            row.update(
                raw_text=completion.text,
                finish_reason=completion.finish_reason,
                input_tokens=len(output.prompt_token_ids),
                output_tokens=len(completion.token_ids),
                seconds=round(time.time() - loaded),
            )
            outputs.append(row)
            with sink.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
        if time.time() - last_commit > 30:
            work.commit()
            last_commit = time.time()
    finished = {id(row) for row in outputs}
    for row in {id(row): row for row in pending.values()}.values():
        if id(row) not in finished:
            row.update(raw_text="", finish_reason="deadline", input_tokens=0, output_tokens=0)
            outputs.append(row)
    work.commit()
    return {
        "outputs": outputs,
        "load_seconds": round(loaded - started),
        "seconds": round(time.time() - started),
    }


@app.local_entrypoint()
def main(jobs: str, confirm_usd: float = 0.0):
    """`jobs` is the request file written by tools/probe_jobs.py; answers are saved next to it."""
    path = Path(jobs)
    batch = json.loads(path.read_text())
    limit_minutes = batch["limit_minutes"]
    estimate = round(limit_minutes / 60 * L4_PER_HOUR, 2)
    print(
        f"{len(batch['jobs'])} requests; worst-case Modal cost ${estimate:.2f} ({limit_minutes}-minute limit)"
    )
    if confirm_usd < estimate:
        raise SystemExit("Not launched: pass --confirm-usd at least equal to the stated estimate")
    name = path.parent.name
    # The batch stops itself two minutes before the hard limit and returns what finished.
    deadline = time.time() + limit_minutes * 60 - 120
    result = generate.with_options(timeout=limit_minutes * 60).remote(
        batch["jobs"], batch["engine"], name, deadline
    )
    (path.parent / "answers.json").write_text(json.dumps(result, indent=1))
    print(
        f"{len(result['outputs'])} answers in {result['seconds']} s (model load {result['load_seconds']} s)"
    )


cpu_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "jsonschema>=4.23.0,<5", "pydantic>=2.11.9,<3", "pypdfium2==4.30.0"
)
if modal.is_local():
    cpu_image = cpu_image.add_local_dir(Path(__file__).resolve().parents[2] / "src", f"{SRC}/src")


@app.function(image=cpu_image, cpu=2, memory=16 * 1024, volumes={WORK_DIR: work}, timeout=1200)
def ground(items: list[dict], snapshots: str) -> dict:
    """Locate each answer's printed values in the stored parse of its PDF (native text or OCR) and
    return citations per answer key. Snapshots are found by the PDF's SHA-256 under `snapshots`."""
    import hashlib
    import sys

    sys.path.insert(0, f"{SRC}/src")
    from docextract.contracts import ParseSnapshot
    from docextract.oneshot import locate_evidence

    stores = list(Path(WORK_DIR).glob(snapshots))
    loaded: dict[str, object] = {}
    out = {}
    for item in items:
        pdf = Path(WORK_DIR) / item["pdf"]
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if digest not in loaded:
            path = next(
                (store / f"{digest}.json" for store in stores if (store / f"{digest}.json").exists()), None
            )
            loaded[digest] = ParseSnapshot.model_validate_json(path.read_text()) if path else None
        snapshot = loaded[digest]
        out[item["key"]] = locate_evidence(item["data"], item["schema"], snapshot) if snapshot else None
    return out


@app.local_entrypoint()
def evidence(answers: str, snapshots: str = "runs/final-v1/shard-*/snapshots"):
    """`answers` is the file `tools/score_probe.py --export` wrote; citations are saved next to it."""
    path = Path(answers)
    items = json.loads(path.read_text())
    print(f"{len(items)} answers to ground; worst-case Modal cost $0.03 (CPU, 20-minute limit)")
    result = ground.remote(items, snapshots)
    (path.parent / "evidence.json").write_text(json.dumps(result))
    print(f"grounded {sum(1 for v in result.values() if v is not None)} of {len(result)} answers")


fetch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "huggingface-hub>=0.34", "jsonschema>=4.23.0,<5", "pydantic>=2.11.9,<3", "pypdfium2==4.30.0"
    )
    .env({"HF_HOME": f"{WORK_DIR}/model-cache/huggingface", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
if modal.is_local():
    root = Path(__file__).resolve().parents[2]
    fetch_image = fetch_image.add_local_dir(root / "src", f"{SRC}/src").add_local_dir(
        root / "manifests", f"{SRC}/manifests"
    )


@app.function(image=fetch_image, cpu=2, memory=4 * 1024, volumes={WORK_DIR: work}, timeout=1800)
def fetch(arm: str) -> str:
    """Download a pinned checkpoint into the work volume on a CPU container and verify every hash."""
    import sys

    sys.path.insert(0, f"{SRC}/src")
    from docextract.remote import pinned_checkpoint

    name, revision, checkpoint = pinned_checkpoint(
        arm, Path(SRC) / "manifests/model-research.json", Path(WORK_DIR) / "model-cache"
    )
    work.commit()
    return f"{name}@{revision} verified at {checkpoint}"


@app.local_entrypoint()
def checkpoint(arm: str):
    """Fetch a model before a GPU probe needs it (CPU only; about $0.02)."""
    print(fetch.remote(arm))
