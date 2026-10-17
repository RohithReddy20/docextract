"""Batched vLLM engine for cited extraction. Remote-only; heavy imports occur after the guard.

Prompts are the exact token IDs the Hugging Face chat template produces for the shared cited
messages, so prompt construction is identical across engines.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from docextract.cited import CitedTask, cited_output_schema
from docextract.models import GenerationResult
from docextract.remote import RemoteOnlyError, cited_messages, configure_caches, pinned_checkpoint


def require_remote() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and Path("/kaggle/working").is_dir():
        return "kaggle"
    if os.environ.get("MODAL_TASK_ID"):
        return "modal"
    raise RemoteOnlyError("Model downloads and execution are permitted only on Kaggle or Modal")


class VllmCitedEngine:
    def __init__(
        self,
        arm: str,
        *,
        manifest: Path,
        cache_root: Path,
        max_model_len=16384,
        max_new_tokens=2048,
        gpu_memory_utilization=0.9,
        max_num_seqs=64,
        constrained=False,
        offline=False,
        dtype="float16",
        speculative_tokens=0,
        max_images=0,
        prefix_caching=True,
        quantization=None,
    ):
        platform = require_remote()
        if arm not in ("a", "b", "c"):
            raise ValueError("Unknown trial arm")
        print("Estimated total cash cost is stated by the launching job; no paid fallback here.", flush=True)
        if platform == "kaggle":
            configure_caches(cache_root)
        else:
            cache_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
        started = time.perf_counter()
        name, revision, checkpoint = pinned_checkpoint(arm, manifest, cache_root, offline=offline)
        import vllm
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=str(checkpoint),
            # float16 on T4 (no bfloat16 support); the checkpoints' own bfloat16 where supported.
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            # Each running sequence holds one linear-attention state block; a T4 fits about 130.
            max_num_seqs=max_num_seqs,
            seed=0,
            # Images only when scanned pages are shown to the model; otherwise text-only memory use.
            limit_mm_per_prompt={"image": max_images, "video": 0},
            trust_remote_code=False,
            # Periodic throughput, queue and cache-hit lines in the job log.
            disable_log_stats=False,
            # Prefix reuse is unreliable for this hybrid architecture and keeps extra state copies.
            enable_prefix_caching=prefix_caching,
            # "fp8": weights quantized to 8 bits at load, so a 9B checkpoint fits a 24 GB L4.
            quantization=quantization,
            # The checkpoint's own multi-token-prediction head drafts tokens that the model then
            # verifies; greedy output is unchanged, only faster when drafts are accepted.
            **(
                {"speculative_config": {"method": "mtp", "num_speculative_tokens": speculative_tokens}}
                if speculative_tokens
                else {}
            ),
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
        self.ready: list[tuple[str, GenerationResult]] = []
        self.requests: dict[str, tuple[str, float]] = {}
        self.constrained = constrained
        self.arm, self.max_model_len, self.max_new_tokens = arm, max_model_len, max_new_tokens
        self.identity = {
            "backend": "vllm",
            "platform": platform,
            "model": name,
            "revision": revision,
            "dtype": dtype,
            "engine": vllm.__version__,
            "generation": {
                "max_model_len": max_model_len,
                "max_new_tokens": max_new_tokens,
                "max_num_seqs": max_num_seqs,
                "greedy": True,
                "constrained_json": constrained,
                "speculative_tokens": speculative_tokens,
                "max_images": max_images,
                "prefix_caching": prefix_caching,
                **({"quantization": quantization} if quantization else {}),
            },
            # The protocol (cited or plain values) is chosen per window and is part of its identity.
            "prompt_version": "cited-values/4+plain-values/2+oneshot/1",
            "engine_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "prompt_source_sha256": hashlib.sha256(
                Path(__file__).with_name("remote.py").read_bytes()
            ).hexdigest(),
        }
        self.load_seconds = time.perf_counter() - started
        self.telemetry: list[dict] = []

    def prompt_ids(self, task: CitedTask) -> list[int]:
        messages, options = cited_messages(task, self.arm, task.protocol)
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
            **options,
        )
        return [int(token) for token in encoded["input_ids"]]

    def params(self, task: CitedTask):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams

        structured = None
        if self.constrained and task.protocol == "cited":
            # One grammar per request: each window may carry a different schema field group.
            structured = StructuredOutputsParams(json=cited_output_schema(task.prompt_schema))
        elif task.protocol == "oneshot":
            # Any JSON object, without free whitespace (no runaway blank output).
            structured = StructuredOutputsParams(json_object=True, disable_any_whitespace=True)
        return SamplingParams(
            temperature=0.0,
            max_tokens=self.max_new_tokens,
            structured_outputs=structured,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

    def result(self, output) -> GenerationResult:
        completion = output.outputs[0]
        return GenerationResult(
            raw_text=completion.text,
            finish_reason="stop" if completion.finish_reason == "stop" else "length",
            input_tokens=len(output.prompt_token_ids),
            output_tokens=len(completion.token_ids),
        )

    def multimodal_prompt(self, task: CitedTask) -> tuple[dict, int]:
        """Text prompt with page images; vLLM expands each image placeholder to its patch tokens."""
        from PIL import Image

        messages, options = cited_messages(task, self.arm, task.protocol)
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False, **options
        )
        images = []
        for path in task.page_images:
            with Image.open(path) as image:
                shown = image.convert("RGB")
            if task.page_image_side:
                shown.thumbnail((task.page_image_side, task.page_image_side))
            images.append(shown)
        # About one token per 32x32 pixels after patch merging; used only for the length check.
        estimate = len(self.tokenizer(text)["input_ids"]) + sum(i.width * i.height // 1024 for i in images)
        return {"prompt": text, "multi_modal_data": {"image": images}}, estimate

    def submit(self, tasks: list[CitedTask]) -> None:
        """Queue windows in the running engine; continuous batching admits them in order."""
        for task in tasks:
            if task.page_images:
                prompt, length = self.multimodal_prompt(task)
            else:
                ids = self.prompt_ids(task)
                prompt, length = {"prompt_token_ids": ids}, len(ids)
            if length + self.max_new_tokens > self.max_model_len:
                refusal = GenerationResult(raw_text="", finish_reason="input_limit", input_tokens=length)
                self.ready.append((task.task_id, refusal))
                continue
            internal = self.llm.llm_engine.add_request(task.task_id, prompt, self.params(task))
            submitted = time.perf_counter()
            # The engine may suffix request ids internally; either spelling maps back to the task.
            self.requests[task.task_id] = (task.task_id, submitted)
            if isinstance(internal, str):
                self.requests[internal] = (task.task_id, submitted)

    def busy(self) -> bool:
        return bool(self.ready) or self.llm.llm_engine.has_unfinished_requests()

    def poll(self) -> list[tuple[str, GenerationResult]]:
        """Run one engine step; return the windows that finished in it, by task id."""
        answers, self.ready = self.ready, []
        if answers or not self.llm.llm_engine.has_unfinished_requests():
            return answers
        for output in self.llm.llm_engine.step():
            if not output.finished:
                continue
            key = output.request_id
            if key not in self.requests:
                key = key.rsplit("-", 1)[0]
            task_id, submitted = self.requests[key]
            for alias in [name for name, (owner, _) in self.requests.items() if owner == task_id]:
                del self.requests[alias]
            generation = self.result(output)
            answers.append((task_id, generation))
            self.telemetry.append(
                {
                    "task_id": task_id,
                    "seconds": round(time.perf_counter() - submitted, 2),
                    "input_tokens": generation.input_tokens,
                    "output_tokens": generation.output_tokens,
                    "finish_reason": generation.finish_reason,
                }
            )
        return answers

    def generate_cited(self, task: CitedTask) -> GenerationResult:
        return self.generate_batch([task])[0]

    def generate_batch(self, tasks: list[CitedTask]) -> list[GenerationResult]:
        """Answer a list of windows at once (diagnostic tools); runs use the streaming methods."""
        results: dict[str, GenerationResult] = {}
        self.submit(tasks)
        while self.busy():
            results.update(self.poll())
        return [results[task.task_id] for task in tasks]
