"""Remote-only engine probe: one pinned checkpoint, fixed prompts, recorded speed, memory and outputs.

The vLLM path imports nothing from docextract, so it can run in its own isolated environment.
"""

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def require_kaggle():
    if platform.system() != "Linux" or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RuntimeError("Model downloads and execution are permitted only inside Kaggle")


def device_memory(device: int) -> int:
    used = subprocess.run(
        ["nvidia-smi", f"--id={device}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return int(used) * 1024 * 1024


def save(path: Path, report: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def hf(args, prompts, report):
    from docextract.models import InputTooLong
    from docextract.remote import GenerationOptions, KaggleTransformersBackend

    backend = KaggleTransformersBackend(
        "a",
        manifest=args.manifest,
        device=args.device,
        cache_root=args.cache_root,
        options=GenerationOptions(max_input_tokens=200_000, max_new_tokens=2048),
        kernels=args.kernels,
    )
    report.update(identity=backend.identity, preflight=backend.preflight)
    for prompt in prompts:
        if args.only and prompt["kind"] != args.only:
            continue
        backend.options = GenerationOptions(max_input_tokens=200_000, max_new_tokens=prompt["max_tokens"])
        row = {"id": prompt["id"], "kind": prompt["kind"]}
        started = time.perf_counter()
        try:
            generation = backend._run(prompt["messages"], {})
            row.update(
                text=generation.raw_text,
                finish=generation.finish_reason,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
                seconds=round(time.perf_counter() - started, 3),
                peak_reserved_bytes=backend.telemetry[-1]["peak_reserved_bytes"],
            )
        except (InputTooLong, RuntimeError) as error:
            row["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            backend.torch.cuda.empty_cache()
        report["results"].append(row)
        save(args.output, report)


def vllm(args, prompts, report):
    from huggingface_hub import snapshot_download

    entry = next(
        item for item in json.loads(args.manifest.read_text())["models"] if item["model"] == args.model
    )
    started = time.perf_counter()
    checkpoint = snapshot_download(
        repo_id=args.model,
        revision=entry["pinned_revision"],
        cache_dir=str(args.cache_root),
        token=False,
        allow_patterns=["*.json", "*.jinja", "*.txt", "*.safetensors", "tokenizer*", "LICENSE", "README.md"],
    )
    report["download_seconds"] = round(time.perf_counter() - started, 1)
    import torch
    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=checkpoint,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.88,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
        trust_remote_code=False,
    )
    report.update(
        load_seconds=round(time.perf_counter() - started, 1),
        engine={"vllm": __import__("vllm").__version__, "torch": torch.__version__},
        device_bytes_after_load=device_memory(args.device),
        revision=entry["pinned_revision"],
    )
    save(args.output, report)
    options = {"chat_template_kwargs": {"enable_thinking": False}}

    def record(batch, outputs, seconds, mode):
        for prompt, output in zip(batch, outputs, strict=True):
            completion = output.outputs[0]
            report["results"].append(
                {
                    "id": prompt["id"],
                    "kind": prompt["kind"],
                    "mode": mode,
                    "text": completion.text,
                    "finish": completion.finish_reason,
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(completion.token_ids),
                    "batch_seconds": round(seconds, 3),
                }
            )
        save(args.output, report)

    diagnostic = [prompt for prompt in prompts if prompt["kind"] == "diagnostic"]
    params = [SamplingParams(temperature=0.0, max_tokens=prompt["max_tokens"]) for prompt in diagnostic]
    started = time.perf_counter()
    outputs = llm.chat([prompt["messages"] for prompt in diagnostic], params, **options)
    record(diagnostic, outputs, time.perf_counter() - started, "batched")
    for prompt in [*diagnostic[:3], *(p for p in prompts if p["kind"] == "sweep")]:
        started = time.perf_counter()
        try:
            output = llm.chat(
                [prompt["messages"]],
                SamplingParams(temperature=0.0, max_tokens=prompt["max_tokens"]),
                **options,
            )
            record([prompt], output, time.perf_counter() - started, "single")
        except Exception as error:  # an oversized sweep prompt is a measurement, not a crash
            report["results"].append({"id": prompt["id"], "mode": "single", "error": str(error)[:300]})
            save(args.output, report)
    report["device_bytes_end"] = device_memory(args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("hf", "vllm"), required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--kernels", choices=("torch", "fla"), default="torch")
    parser.add_argument("--only", choices=("diagnostic", "sweep"))
    parser.add_argument("--max-model-len", type=int, default=32768)
    args = parser.parse_args()
    require_kaggle()
    if args.engine == "vllm":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    report = {"engine": args.engine, "kernels": args.kernels, "results": [], "estimated_cash_cost_usd": 0.0}
    try:
        (hf if args.engine == "hf" else vllm)(args, prompts, report)
        report["complete"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {str(error)[:1500]}"
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()
