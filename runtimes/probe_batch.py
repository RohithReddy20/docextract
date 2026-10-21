"""One probe batch on a loaded vLLM engine, shared by the Modal and Colab probe runners.

Requests come from `tools/probe_jobs.py`. Each answer is appended to a JSON-lines file as it finishes,
so a runner that stops early (a deadline, a disconnected notebook) keeps everything already answered,
and a rerun can skip those requests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

# Requests admitted to the engine beyond those it runs at once (max_num_seqs), so it stays busy.
SPARE_IN_FLIGHT = 8


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


def request_text(job: dict, first: int, last: int, total: int, texts: list[str]) -> str:
    """The text of one request for pages `first`..`last - 1` (0-based) of a `total`-page document."""
    text = job["user_text"]
    if last - first < total:
        where = f"image is page {last}" if last == first + 1 else f"images are pages {first + 1}-{last}"
        text += (
            f"\n\nThe {where} of a {total}-page document. Extract what "
            f"{'this page shows' if last == first + 1 else 'these pages show'}; use null (or an empty "
            "list) for anything not on "
            f"{'it' if last == first + 1 else 'them'}."
        )
    if job.get("context_chars") and first > 0 and texts:
        # Rows continued from the previous page may rely on its headings or on values printed only on
        # a group's first row, so the end of that page comes along as context.
        tail = texts[first - 1].strip()[-job["context_chars"] :]
        if tail:
            text += (
                "\n\nThe end of the previous page, for context only (headings and values carried "
                f"forward; its own records are read separately):\n{tail}"
            )
    if texts:
        layer = "\n\n".join(
            f"--- page {first + i + 1} ---\n{t.strip()}" for i, t in enumerate(texts[first:last])
        )
        text = f"{text}\n\nText layer of the document (may be incomplete; trust the images):\n{layer}"
    return text


def request_key(row: dict) -> tuple:
    """What identifies one request across runs, for skipping requests a previous run answered."""
    return row["id"], row["variant"], row.get("part"), row["batch"]


def answered(sink: Path) -> set[tuple]:
    """Requests already answered in `sink` (finished answers only, not deadline or error rows)."""
    if not sink.exists():
        return set()
    keys = set()
    for line in sink.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # a line cut off when the previous run stopped
            continue
        if row.get("finish_reason") in ("stop", "length"):
            keys.add(request_key(row))
    return keys


def load_engine(checkpoint: Path, engine: dict):
    """The vLLM engine of a probe configuration's `engine` block."""
    from vllm import LLM

    return LLM(
        model=str(checkpoint),
        dtype="bfloat16",
        max_model_len=engine["max_model_len"],
        gpu_memory_utilization=engine.get("gpu_memory_utilization", 0.92),
        # "fp8" stores twice as many tokens of running answers in the same memory.
        kv_cache_dtype=engine.get("kv_cache_dtype", "auto"),
        # Fewer sequences than the KV cache could start: when running answers outgrow it, vLLM pushes
        # the newest back to recompute them, and some page requests never finished.
        max_num_seqs=engine.get("max_num_seqs", 32),
        seed=0,
        limit_mm_per_prompt={"image": engine["max_images"], "video": 0},
        enable_prefix_caching=False,
        quantization=engine.get("quantization"),
        # Throughput, running and waiting requests in the log every few seconds.
        disable_log_stats=False,
        # JSON without whitespace is set for the whole engine: xgrammar ignores the per-request flag.
        **(
            {"structured_outputs_config": {"backend": "xgrammar", "disable_any_whitespace": True}}
            if engine.get("compact_json")
            else {}
        ),
        **(
            {"speculative_config": {"method": "mtp", "num_speculative_tokens": engine["speculative_tokens"]}}
            if engine.get("speculative_tokens")
            else {}
        ),
    )


def run(
    llm,
    jobs: list[dict],
    engine: dict,
    pdf_of: Callable[[dict], Path],
    sink: Path,
    deadline: float,
    *,
    loaded: float,
    commit: Callable[[], None] = lambda: None,
    skip: set[tuple] = frozenset(),
) -> list[dict]:
    """Answer every job in one continuous batch until `deadline`; each answer is appended to `sink`.

    Prompts are rendered per job (its own chat-template options) and run together, so a slow answer
    holds up nothing else. Requests whose key is in `skip` are not asked again. `commit` is called
    every 30 seconds (the Modal volume commit). Returns the answers of this run, with a "deadline" row
    for every request left unfinished.
    """
    import base64
    import io

    from PIL import Image
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    from docextract.oneshot import page_output_cap

    tokenizer = llm.get_tokenizer()
    rendered: dict[tuple[str, int], list] = {}
    texts: dict[str, list[str]] = {}
    outputs, pending, partial = [], {}, {}
    queue: list[tuple] = []
    number = 0
    for job in jobs:
        pdf = pdf_of(job)
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
        if job.get("max_pages"):
            spans = [(first, last) for first, last in spans if last <= job["max_pages"]]
        for batch, (first, last) in enumerate(spans):
            row = {
                "id": job["id"],
                "variant": job["variant"],
                "batch": batch,
                "batches": len(spans),
                "pages": [first + 1, last],
                "part": job.get("part"),
            }
            if request_key(row) in skip:
                continue
            pages = every[first:last]
            text = request_text(job, first, last, len(every), texts.get(job["id"], []))
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
            row["template_tail"] = prompt[-40:]
            structured = (
                StructuredOutputsParams(
                    json_object=True, disable_any_whitespace=job.get("compact_json", True)
                )
                if job.get("json_object")
                else None
            )
            limit = job["max_tokens"]
            if job.get("part") == "rows" and job.get("page_cap") and texts.get(job["id"]):
                # The product's cap for one page's records, sized by the page's own text.
                limit = min(limit, page_output_cap(texts[job["id"]][first]) or limit)
            params = SamplingParams(temperature=0.0, max_tokens=limit, structured_outputs=structured)
            queue.append((str(number), row, prompt, pages, params))
            number += 1

    def admit(key, row, prompt, pages, params):
        """Add one request; one too long for the context window is retried with page images shrunk
        (as ExtractBench's vLLM harness does, by a quarter per round, at most four rounds)."""
        internal = None
        for rounds in range(5):
            try:
                internal = llm.llm_engine.add_request(
                    key, {"prompt": prompt, "multi_modal_data": {"image": pages}}, params
                )
                row["downscaled"] = rounds
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
                    outputs.append(row)
                    return
                pages = [
                    page.resize((max(64, page.width * 3 // 4), max(64, page.height * 3 // 4)))
                    for page in pages
                ]
        pending[key] = row
        row["request"] = key
        active.add(key)
        if isinstance(internal, str):
            pending[internal] = row

    # Requests are admitted a few at a time: each holds its preprocessed page images in memory until
    # it finishes, and hundreds of page requests at once exceed the container's memory.
    active: set[str] = set()
    last_commit = time.time()
    while (queue or llm.llm_engine.has_unfinished_requests()) and time.time() < deadline:
        while queue and len(active) < engine.get("max_num_seqs", 32) + SPARE_IN_FLIGHT:
            admit(*queue.pop(0))
        if not llm.llm_engine.has_unfinished_requests():
            continue
        for output in llm.llm_engine.step():
            if not output.finished:
                partial[output.request_id] = output.outputs[0]
                continue
            key = output.request_id if output.request_id in pending else output.request_id.rsplit("-", 1)[0]
            row = pending[key]
            active.discard(row["request"])
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
            commit()
            last_commit = time.time()
    for _, row, _, _, _ in queue:
        pending[str(id(row))] = row
    finished = {id(row) for row in outputs}
    unfinished = {id(row): row for row in pending.values()}
    for row in unfinished.values():
        if id(row) not in finished:
            row.update(raw_text="", finish_reason="deadline", input_tokens=0, output_tokens=0)
            # What an unfinished answer had written by the deadline, for diagnosis only.
            so_far = next(
                (partial[name] for name, owner in pending.items() if owner is row and name in partial), None
            )
            if so_far is not None:
                row.update(partial_text=so_far.text, partial_tokens=len(so_far.token_ids))
            outputs.append(row)
    commit()
    return outputs
