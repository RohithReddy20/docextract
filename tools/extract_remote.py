"""Remote-only batched cited extraction over an inference-only packet (PDFs and original schemas).

The packet manifest may contain only case ids, PDF paths and schema paths: no answers, rules or
binding plans. Results, a per-case summary and engine telemetry are written under --output.
"""

import argparse
import json
import os
import re
from pathlib import Path

# Prompt-level options a variant may set; engine-level ones (model, decoding) are per process.
VARIANT_KEYS = {
    "name",
    "protocol",
    "page_images",
    "max_window_units",
    "max_window_chars",
    "max_schema_chars",
    "route_groups",
    "prompt_budget_chars",
    "factor_notes",
    "merge_entities",
    "page_image_side",
    "page_image_side_split",
    "max_window_pages",
    "oneshot_dpi",
    "ocr_text",
}


def contained(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Case path escapes the inference packet")
    return path


def load_cases(manifest: Path) -> list[tuple[str, Path, dict]]:
    packet = json.loads(manifest.read_text())
    if set(packet) != {"version", "cases"} or packet["version"] != "inference-cases/1":
        raise ValueError("Expected an inference-only case manifest")
    cases = []
    for case in packet["cases"]:
        if set(case) != {"id", "pdf", "schema"} or not case["id"].isalnum():
            raise ValueError("Inference manifest must contain only alphanumeric ids, PDFs and schemas")
        schema = json.loads(contained(manifest.parent, case["schema"]).read_text())
        cases.append((case["id"], contained(manifest.parent, case["pdf"]), schema))
    if len({case_id for case_id, _, _ in cases}) != len(cases):
        raise ValueError("Duplicate case ids")
    return cases


def balanced_share(cases, index: int, count: int):
    """This process's documents when `count` processes split the work: heaviest first to the
    lightest share, weighed by pages times schema groups. Every process computes the same split."""
    if count == 1:
        return cases
    import math
    from contextlib import closing

    import pypdfium2 as pdfium

    weights = []
    for _, pdf, schema in cases:
        with closing(pdfium.PdfDocument(str(pdf))) as document:
            pages = len(document)
        weights.append(pages * max(1, math.ceil(len(json.dumps(schema)) / 24000)))
    loads, owner = [0] * count, {}
    for position in sorted(range(len(cases)), key=lambda i: (-weights[i], i)):
        target = loads.index(min(loads))
        loads[target] += weights[position]
        owner[position] = target
    return [case for position, case in enumerate(cases) if owner[position] == index]


def prepare_parser(args, cases):
    """Parse every document before the model loads; OCR shares the GPU only during this step.

    Snapshots (and page images) go to a store that several runs may share: a document already in the
    store is not parsed again, and the OCR model loads only when some document still needs it.
    """
    import hashlib

    from docextract.batch import CachedParser

    store = args.snapshots
    if args.engine == "replay":
        return CachedParser(store=store, parse_missing=False)
    if args.ocr == "none":
        return CachedParser(store=store)
    import gc
    import time

    from docextract.parsing.ocr import DoctrRecognizer, OcrFallbackParser

    started = time.perf_counter()
    missing = [
        pdf
        for _, pdf, _ in cases
        if not (store / f"{hashlib.sha256(pdf.read_bytes()).hexdigest()}.json").exists()
    ]
    if not missing:
        # Everything was parsed (with OCR) earlier, for example by a shared parse step.
        return CachedParser(store=store, parse_missing=False)
    images = args.page_images_dir if args.keep_page_images else None
    recognizer = DoctrRecognizer(device=args.ocr_device)
    parser = CachedParser(OcrFallbackParser(recognizer, image_store=images), store=store)
    stats = {"documents": 0, "parsed_here": len(missing), "ocr_pages": 0}
    for _, pdf, _ in cases:
        try:
            snapshot = parser.parse(pdf)
        except Exception as error:  # extraction will record the same failure for this case
            stats.setdefault("errors", []).append(type(error).__name__)
            continue
        stats["documents"] += 1
        stats["ocr_pages"] += sum("ocr_text" in page.quality_flags for page in snapshot.pages)
    del recognizer
    parser.parser.recognizer = None
    gc.collect()
    if args.ocr_device == "cuda":
        import torch

        torch.cuda.empty_cache()
    stats["seconds"] = round(time.perf_counter() - started, 1)
    from docextract.contracts import save_json

    save_json(args.output / "ocr.json", stats)
    return parser


def engine_for(args):
    if args.engine == "replay":
        from docextract.batch import ReplayEngine

        runtime = json.loads((args.replay_from / "runtime.json").read_text())
        args.output.mkdir(parents=True, exist_ok=True)
        if not (args.output / "generations.jsonl").exists():
            (args.output / "generations.jsonl").write_bytes(
                (args.replay_from / "generations.jsonl").read_bytes()
            )
        return ReplayEngine(runtime["identity"])
    if args.engine == "vllm":
        from docextract.vllm_engine import VllmCitedEngine

        return VllmCitedEngine(
            args.arm,
            manifest=args.model_manifest,
            cache_root=args.cache_root,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            constrained=args.constrained,
            offline=args.offline,
            dtype=args.dtype,
            speculative_tokens=args.speculative_tokens,
            max_images=max(
                (
                    v.get("max_window_pages", args.max_window_pages)
                    for v in args.variants
                    if v.get("page_images")
                ),
                default=0,
            )
            # A whole-document request shows every page.
            if not any(v.get("protocol") == "oneshot" for v in args.variants)
            else args.max_document_pages,
            prefix_caching=not args.no_prefix_caching,
            quantization=args.quantization,
            kv_cache_dtype=args.kv_cache_dtype,
        )
    from docextract.remote import GenerationOptions, KaggleTransformersBackend

    return KaggleTransformersBackend(
        args.arm,
        manifest=args.model_manifest,
        device=args.device,
        cache_root=args.cache_root,
        options=GenerationOptions(max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens),
        kernels=args.kernels,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--engine", choices=("hf", "vllm", "replay"), default="hf")
    parser.add_argument(
        "--replay-from", type=Path, help="run directory whose stored answers are re-assembled"
    )
    parser.add_argument("--complete-tables", action="store_true")
    parser.add_argument("--absent-nullable", choices=("null", "omit", "root"), default="null")
    parser.add_argument("--document-strings", action="store_true", help="document text for re-cased strings")
    parser.add_argument(
        "--nulls-if-partial", action="store_true", help="null absent nullable fields even if windows failed"
    )
    parser.add_argument("--arm", choices=("a", "b", "c"), required=True)
    parser.add_argument("--quantization", choices=("fp8",), help="quantize weights at load (vLLM)")
    parser.add_argument("--kv-cache-dtype", choices=("fp8",), help="store running answers in 8 bits (vLLM)")
    parser.add_argument("--kernels", choices=("torch", "fla"), default="torch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--max-window-chars", type=int, default=16000)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--legacy-max-new-tokens",
        type=int,
        nargs="*",
        default=[],
        help="earlier output limits of this engine whose stored answers stand unless they were cut off",
    )
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--speculative-tokens", type=int, default=0, help="MTP draft tokens per step")
    parser.add_argument("--page-images", action="store_true", help="show scanned page images to the model")
    parser.add_argument(
        "--protocol",
        choices=("cited", "values", "oneshot"),
        default="cited",
        help="values: no source references; oneshot: the whole document (page images + text) at once",
    )
    parser.add_argument(
        "--variants",
        type=json.loads,
        help='JSON list of prompt variants run together on one engine, e.g. [{"name": "values", '
        '"protocol": "values"}, {"name": "images", "page_images": true}]',
    )
    parser.add_argument("--shard", default="0/1", help="i/n: this process takes every n-th document from i")
    parser.add_argument("--snapshots", type=Path, help="parsed-document store (shareable between runs)")
    parser.add_argument("--page-images-dir", type=Path, help="page-image store (shareable between runs)")
    parser.add_argument("--parse-only", action="store_true", help="parse (and OCR) documents, then stop")
    parser.add_argument("--keep-page-images", action="store_true", help="save recognized pages as images")
    parser.add_argument("--ocr-device", default="cuda", help="cuda, or cpu for a parse-only CPU session")
    parser.add_argument("--no-prefix-caching", action="store_true", help="disable vLLM prefix reuse")
    parser.add_argument("--max-window-pages", type=int, default=4, help="pages per window with --page-images")
    parser.add_argument("--offline", action="store_true", help="use only an already verified model cache")
    parser.add_argument("--ocr", choices=("none", "doctr"), default="none")
    parser.add_argument("--max-schema-chars", type=int, default=24000)
    parser.add_argument("--max-window-units", type=int, help="cap cells per window (dense tables)")
    parser.add_argument("--constrained", action="store_true", help="vLLM JSON-schema constrained decoding")
    parser.add_argument(
        "--max-document-pages", type=int, default=200, help="most page images in one whole-document request"
    )
    args = parser.parse_args()
    # Without --variants the single variant is the flat options above, written directly under --output.
    args.variants = args.variants or [
        {"name": "", "protocol": args.protocol, "page_images": args.page_images}
    ]
    names = [variant.setdefault("name", "") for variant in args.variants]
    home = args.replay_from if args.engine == "replay" else args.output
    args.snapshots = args.snapshots or home / "snapshots"
    args.page_images_dir = args.page_images_dir or home / "page-images"
    args.keep_page_images = args.keep_page_images or any(v.get("page_images") for v in args.variants)
    for variant in args.variants:
        if set(variant) - VARIANT_KEYS or variant.get("protocol", "cited") not in (
            "cited",
            "values",
            "oneshot",
        ):
            raise ValueError(f"Unsupported variant: {variant}")
    if len(set(names)) != len(names) or (
        len(names) > 1 and not all(re.fullmatch(r"[A-Za-z0-9-]+", n) for n in names)
    ):
        raise ValueError("Variants need distinct names of letters, digits and dashes")
    from docextract.vllm_engine import require_remote

    if require_remote() == "kaggle":
        # Caches are fixed before any model library (OCR or LLM) is imported in this process.
        from docextract.remote import configure_caches

        configure_caches(args.cache_root)
    if args.engine == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    from docextract.batch import run_batch
    from docextract.contracts import save_json

    index, count = (int(part) for part in args.shard.split("/"))
    cases = balanced_share(load_cases(args.cases), index, count)
    args.output.mkdir(parents=True, exist_ok=True)
    parser = prepare_parser(args, cases)
    if args.parse_only:
        return
    # Documents interleave across variants, so every variant progresses at the same pace.
    jobs = [
        (f"{variant['name']}/{case_id}" if variant["name"] else case_id, pdf, schema, variant)
        for case_id, pdf, schema in cases
        for variant in args.variants
    ]
    summary = {job[0]: {"id": job[0].split("/")[-1], "result": "not_started"} for job in jobs}
    variant_of = {job[0]: job[3]["name"] for job in jobs}

    def save_summaries():
        for name in names:
            rows = [row for job_id, row in summary.items() if variant_of[job_id] == name]
            save_json(args.output / name / "summary.json", {"cases": rows})

    save_summaries()
    engine = engine_for(args)
    save_json(args.output / "runtime.json", {"identity": engine.identity})

    def record(case_id, outcome):
        if isinstance(outcome, str):
            summary[case_id] = {
                "id": case_id.split("/")[-1],
                "result": "failed",
                "error": outcome.split(":")[0],
            }
        else:
            summary[case_id] = {
                "id": case_id.split("/")[-1],
                "result": "complete",
                "status": outcome.completion.status,
                "schema_valid": outcome.validation.valid,
                "pages": len(outcome.snapshot.pages),
                "model_calls": outcome.usage.generated_attempts,
                "input_tokens": outcome.usage.input_tokens,
                "output_tokens": outcome.usage.output_tokens,
                "rejected": outcome.coverage.rejected_proposals,
                "unresolved": len(outcome.unresolved),
            }
        save_summaries()
        save_json(args.output / "telemetry.json", {"tasks": engine.telemetry})

    options = {
        "max_window_chars": args.max_window_chars,
        "max_schema_chars": args.max_schema_chars,
        "max_window_units": args.max_window_units,
        "complete_tables": args.complete_tables,
        "absent_nullable": args.absent_nullable,
        "document_strings": args.document_strings,
        "nulls_if_partial": args.nulls_if_partial,
    }
    # A replay reads the images the answers were generated from, so their identities match.
    images = args.page_images_dir
    for variant in args.variants:
        variant.setdefault("protocol", "cited")
        if variant.pop("page_images", False):
            variant |= {
                "page_images": images,
                "max_window_pages": variant.get("max_window_pages", args.max_window_pages),
            }
    jobs = [
        (job_id, pdf, schema, {k: v for k, v in own.items() if k != "name"})
        for job_id, pdf, schema, own in jobs
    ]
    if args.engine in ("vllm", "replay"):
        # Windows stream through one continuously batched engine; answers are stored as they finish.
        # The same engine with a lower output limit: answers it finished stand (`GenerationStore`).
        legacy = tuple(
            {**engine.identity, "generation": {**engine.identity["generation"], "max_new_tokens": limit}}
            for limit in args.legacy_max_new_tokens
            if limit < args.max_new_tokens
        )
        run_batch(
            jobs,
            root=args.output,
            engine=engine,
            on_result=record,
            batch_size=256,
            parser=parser,
            legacy_identities=legacy,
            **options,
        )
    else:
        # A sequential engine completes documents one at a time, so a deadline keeps finished ones.
        from docextract.cited import extract_cited

        for case_id, pdf, schema, own in jobs:
            try:
                outcome = extract_cited(
                    pdf,
                    schema,
                    workspace=args.output / case_id,
                    backend=engine,
                    parser=parser,
                    **(options | own),
                )
            except Exception as error:  # the case stays in the population as a failure
                outcome = f"{type(error).__name__}: {str(error)[:400]}"
            record(case_id, outcome)
    save_json(args.output / "telemetry.json", {"tasks": engine.telemetry})


if __name__ == "__main__":
    # Engine worker processes can outlive an exception; results are already saved after each case,
    # so exit hard instead of waiting on them.
    code = 0
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
