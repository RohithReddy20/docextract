# First remote model comparison

The runner, private-kernel builder, platform guard, artifact allowlist and 24 authored documents
are implemented. No checkpoint is selected. Local scripted tests do not measure learned extraction
quality. Both 4B candidates have now run on a T4; [measured results](model-trials.md) document the
small tested context, failures and remaining working-context/quality gates.

## Current runner (ExtractBench development runs)

`tools/build_extractbench_kernel.py` embeds the source and a run configuration into a private Kaggle
script (`runtimes/kaggle/extractbench.py`). Submitting it is a separate, explicit `kaggle kernels
push`. On Kaggle the script downloads the pinned dataset, writes inference inputs (PDF + original
schema only) and a separate evaluation directory, installs vLLM 0.30.0 (plus docTR when OCR is
enabled) in its own environment, runs `tools/extract_remote.py` once per GPU, re-assembles stored
answers for each configured replay variant, scores every run with the official evaluator in another
process, and exports scores, per-case dumps (development subset only) and compact results. Measured
fixed overhead on T4s: about 5 minutes to install and about 9 minutes for vLLM engine start-up.

Answers stream: each finished window is stored at once, so a run stopped at its deadline keeps its
work. A `final` replay always re-assembles every document from the stored answers (documents a
deadline left part-answered included, their missing windows reported as unanswered), and that
replay is what is scored. Runs also export parsed snapshots and engine telemetry, and with
`export_packet` the development schemas, document hashes and scoring files (never PDFs), so
`tools/replay_local.py` can re-assemble and score stored answers on a laptop CPU without a model.
`tools` entries run counting scripts (for example `tools/form_inventory.py`) over the pinned
snapshot; a configuration without GPU runs becomes a CPU-only kernel.

The first attempt (eb-dev v3) sent all 193 windows of the 20 development documents to vLLM as one
batch; at about 41 s per window per T4 nothing had finished when the deadline arrived, and nothing
was scored. Streaming replaced it.

### Small sessions (how experiments run now)

GPU time is limited (Kaggle's weekly quota, at most two GPU sessions at once; Modal credit is kept
for the single final run), so experiments are small and packed:

- **Few, chosen documents.** `ids` selects a handful of benchmark documents by their properties
  (scanned or native, form or table), never by answers; reusing development documents keeps earlier
  results comparable.
- **Many variants per model load.** Each GPU runs one engine (model, decoding); prompt variants
  (`variants`: cited or plain values, page images, window caps) are queued into it together and
  interleaved by document. A window two variants share is generated once.
- **Both GPUs busy.** Two engine configurations run side by side, or one configuration's documents
  are split across both GPUs (`shard`, scored together by `group`).
- **Parse once.** With one OCR setting, documents are parsed and recognized once, split across the
  GPUs, into a shared store that every run and replay reads.
- **Less start-up.** Compiled GPU kernels (Triton, Inductor, vLLM compile caches) are exported and
  restored by a later session that lists the earlier one in `caches_from`.
- **Off-GPU follow-ups.** Assembly options (null policy, table completion, document text) are
  measured on the stored answers on a laptop CPU with `tools/local_sweep.py`.

The development subset is the first N cases per split in a fixed hash order (`docextract-dev/1`),
chosen without looking at answers or results. It is used for error analysis, so a final score must
report it as seen.

## Execution and storage policy

All model weights, model caches, model loading, inference and training must stay on Kaggle or Modal.
This includes GLiNER and other small models. Local work may edit source, inspect public documentation
and small configuration/index metadata, and test the non-model extraction code. Do not download
checkpoint binaries locally or install a local model runtime as part of this plan.

Kaggle owns development and controlled trials. Modal L4 is reserved for a frozen final inference or
full-benchmark configuration. The runbook does not itself submit jobs or download models. Keep notebooks private and credentials/private prompts outside repository
content. Any future submission must state its total estimated cost immediately before launch.

The trial cash-cost estimate is $0 while free Kaggle quota is available. Private jobs were accepted
and elapsed time was measured; quota accounting was not audited. Cap each development session at
one hour and checkpoint results between models; this is an admission limit, not a throughput estimate.
Do not switch to paid compute or Modal development when free capacity is unavailable.

## What the source establishes

The inspected [Kaggle Docker source](https://github.com/Kaggle/docker-python/blob/e096cbc5b6ff50ab49130187f8677dedb9b5e9bd/Dockerfile.tmpl)
inherits a Colab image, freezes existing Torch-related packages, and references a Python 3.13 package
directory. Its [requirements](https://github.com/Kaggle/docker-python/blob/e096cbc5b6ff50ab49130187f8677dedb9b5e9bd/kaggle_requirements.txt)
allow `transformers>=5.0.0`. These files do not prove the image or environment assigned to an account.
The repository requires Python 3.12; use an isolated remote environment rather than modifying the
notebook's shared package installation. No assumption about current free hours or writable disk
capacity follows from the Docker source.

The provisional source-compatible starting point remains Python 3.12, Torch 2.8.0 with CUDA 12.6
wheels, and Transformers 5.3.0. PyTorch publishes that CUDA combination and a Linux CPython 3.12
wheel. This establishes wheel availability, not compatibility with a particular live Kaggle driver
or model. [PyTorch releases](https://pytorch.org/get-started/previous-versions/),
[wheel index](https://download.pytorch.org/whl/cu126/torch/).

Keep this environment separate from both the production core and the benchmark evaluator.
Use `runtimes/kaggle/uv.lock` with `uv sync --frozen`. Configure an explicit Torch
wheel index; do not let a resolver choose a different CUDA build implicitly.
[uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Preflight before checkpoint download

1. Verify that execution is inside Kaggle, select one visible T4, and stop if it is a local host or
   another platform. The runner guards this before importing model/download libraries. Such
   a guard prevents accidental execution; environment markers are not security attestation.
2. Record Python and the selected package versions, OS/libc, Torch build CUDA, NVIDIA driver,
   device name, compute capability, actual total/free device memory, host memory and writable disk.
   Verify an actual FP16 CUDA operation and SDPA on the selected device. Do not infer support from
   `nvidia-smi`'s displayed CUDA version alone.
3. Set remote cache paths before importing Hugging Face libraries. Use a dedicated remote directory
   outside the result-export directory for `HF_HOME`, `HF_HUB_CACHE`, `HF_XET_CACHE`, `HF_ASSETS_CACHE`,
   Torch and uv caches. Confirm resolved paths and available space in the remote worker. Do not dump
   the environment or credential files into telemetry.
4. Download exactly one pinned checkpoint at a time on the remote worker. Set explicit revision,
   tokenizer/template identity, safetensors loading and `trust_remote_code=False`. Public models do
   not require copying a local credential cache. Verify the resolved revision and weight hashes.
5. After preparation, run cache-only inference with `HF_HUB_OFFLINE=1` and `local_files_only=True`.
   Here, “local files” means files inside the remote Kaggle worker. Stop on a missing cache entry;
   it must not trigger a fallback to the user's computer or a different checkpoint.

Hugging Face reads its cache environment variables at import time. Start a fresh Python subprocess
if the notebook has already imported those libraries. `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and
`HF_HUB_DISABLE_TELEMETRY=1` limit implicit credential use and library telemetry; they do not replace
private storage and an explicit export policy.
[Hugging Face environment reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

All Python entry points use `uv run python` in the remote environment. No Torch/Transformers
dependency has been added to the local project by this research.

## Loading A and B

Use the existing pinned Qwen3.5-4B and NuExtract3 revisions in `manifests/model-research.json`.
NuExtract3 shares the Qwen3.5-4B architecture. Begin sequentially on a single T4 so load peaks and
host memory can be observed; use the second GPU for independent work only after resource gates pass.
Do not combine the two cards into one memory budget.

Transformers 5.3.0 has `Qwen3_5ForCausalLM` and a text configuration with
`base_config_key="text_config"`. Its weight conversion maps `model.language_model` into `model`.
The text class excludes vision and MTP parameters from its architecture. These source contracts
make text-only loading a candidate for the native-evidence trial.
[text configuration](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/configuration_qwen3_5.py),
[weight mapping](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/conversion_mapping.py),
[model classes](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

Preserve both 4B checkpoints' tied embeddings. Audit loading diagnostics and tensor keys before
generation; only intentionally excluded vision/MTP keys may be unused, with no unexplained missing
language parameters. The loader suppresses certain expected unused-key warnings, so absence of a
warning is insufficient. Require one remote, sequential comparison against the full multimodal
wrapper using identical text token IDs, template and generation settings. Record greedy-output
agreement and any numerical difference before accepting the reduced wrapper for both trial arms.
Do not keep both wrappers resident together to run that check.

Text-only loading can reduce resident parameters but will not necessarily reduce downloaded bytes:
text/vision tensors share files in both weight indices. NuExtract's separate MTP file must be
accounted for explicitly. No model-file contents were read to establish this index-level fact.

Use FP16, SDPA, batch one, explicit single-device placement, greedy non-thinking generation, neutral
repetition penalties and no CPU/disk offloading. Set `use_cache=True`, inference/evaluation mode,
and fresh per-task state. Begin with at most 2,048 input and 256 output tokens. Increase only after
memory and completion checks. Stop below the lower of 14 GiB or 90% of actual device memory. Include
load-time peaks, non-Torch device allocations and host memory rather than reporting only steady
Torch tensor allocation.

The generation code already sets `logits_to_keep=1` for ordinary supported generation. Preserve it
in custom prefill profiling; at vocabulary 248,320, a full 2,048-position FP16 logits tensor alone
requires 970 MiB. This is arithmetic, not measured runtime memory.
[generation source](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/generation/utils.py).

Do not compile optional Flash/FLA kernels during the first smoke comparison. Record the actual
DeltaNet and attention implementations. An SDPA setting controls full attention; it does not prove
that the DeltaNet path uses an accelerated kernel. The newer documentation's GB10/SM121 kernel
example and speedup are not T4 measurements.
[versioned Qwen documentation](https://huggingface.co/docs/transformers/v5.13.0/en/model_doc/qwen3_5).

## Compare extraction before changing the output protocol

The model must receive the user schema and bounded native evidence, without a hand-authored label,
column, row-count or expected-answer file. The learned runner must introduce new field and record
hypotheses before invoking the existing copying backend. The runner is implemented in `learned.py`; actual model quality remains to be measured.

The first paired comparison measures typed values and records using each model's native convention:
Qwen receives JSON Schema, NuExtract receives the strictly audited typed projection plus retained
descriptions/instructions. Unsupported projections remain in the report as unsupported cases.
Never use expected answers as an instance to help schema conversion. Preserve raw responses before
any parsing, validation, normalization or attempted repair. Record invalid JSON, duplicate object
keys, non-finite numbers, original-schema violations and output limits separately.

Then test the same models on local source bindings as a separate comparison. Resolve returned aliases
and half-open character spans against saved native text; assemble evidence from the actual source
atoms. Quoted values with several matching occurrences do not establish an occurrence. Measure
ambiguous/unbound fields and wrong-occurrence citations. A third paired experiment can compare
automatically inferred table mappings plus copying against full row regeneration. Do not attribute
that orchestration difference solely to the checkpoint.

Partition overlong input by schema subtree and source ranges; preserve a task-to-original-schema
coverage map. Do not truncate source pages, drop schema properties or silently accept a repaired
prefix at the output limit. The smoke budget is a capacity probe, not a promise that a wide record
fits. Save partial/raw output and schedule unfinished ranges under the same occurrence identities.

Implemented authored diagnostic set: six families, four documents each. Two per family are development cases;
two use different schema vocabulary, layout/template and document content for holdout. Design these
before viewing trial predictions; author truth with occurrence identities independently of inference.

| Family | Failure cases that must be represented |
| --- | --- |
| Sparse fields | Competing totals/dates/addresses, leading-zero IDs, distractors |
| Presence and types | Missing versus observed null, zero, defaults, enums, empty arrays |
| Regular tables | Identical distinct rows, repeated headers, changed units/columns |
| Narrative records | Multiple events per paragraph, negative regions, instruction-like text |
| Nested records | Identical parent names, child association, repeated child values |
| Continuation | Multipage records, overlapping windows, unfinished final record |

Twenty-four small documents can reveal broken assumptions; they cannot establish general extraction
quality. Do not promote a model on JSON validity alone. Report typed/critical-field accuracy,
complete-record precision/recall, exact occurrence counts, parent association and source-binding
precision separately. For value-only arms, duplicate counts are measurable but occurrence binding
is not; do not give those outputs invented citation credit.

Record input/output token counts, actual source/schema coverage, p50/p95 task latency, time to first
token, decode rate, load time, host/device peaks, truncation, retries and failure denominators. Keep
negative regions in discovery checks. Only then consider 9B NF4 or a GLiNER discovery arm, each with
its own load/quality gate. A full benchmark additionally requires the pinned remote schema inventory,
frozen runtime and inclusive cost forecast.

## Keep exports small and explicit

Keep weights, tokenizers, model caches, wheel caches, raw datasets, credentials and private prompts
on remote storage. Do not package or recursively download notebook working directories or caches.
Export an allowlisted set of code/revision hashes, sanitized environment/package versions, task
metrics, completion/failure summaries and approved small extraction/evidence results. These exports
must not contain prompts, account balances, tokens, credential paths or private input documents.
Record the export file list, sizes and hashes before transfer. This is a required runner design;
the private bootstrap now implements a small-result allowlist and the backend guards local execution.

## Implemented trial packet and current limits

The first environment-only Kaggle worker reported Python 3.13.15, Torch 2.11.0+cpu, Transformers
5.16.1, Accelerate 1.14.0, `uv`, and the `Batch` marker. It requested no GPU and imported no model
libraries. These are measurements of that worker, not proof of the GPU image or free quota.

The private T4 smoke uses the frozen source archive, a one-hour server timeout, a 3,300-second
internal deadline, and one active device per stage. It first checks input/output token hashes for
Qwen's full versus text-only wrapper, then runs six authored development cases for each 4B candidate.
A mismatch stops the comparison. Runtime setup and model downloads occur only in Kaggle. The
bootstrap exports approved small JSON results and the resolved dependency lock, with no caches,
weights, source snapshots, journals, credentials, private prompts or evaluation answers.

The 24-case packet is 12 development and 12 within-family holdout diagnostics. Holdout vocabulary,
root keys and typography change, but this small generator does not establish generalization to
unseen document families. No ExtractBench documents or answers are included.

The source-binding prototype preserves original characters and exposes a derived cell-text view.
Half-open offsets map back to saved characters; inserted separators have no invented character box.
It carries eight recent record contexts plus their ancestors, with explicit omissions in compact
context summaries. Every actual evidence window is still processed. Larger schemas/windows fail
explicitly at the measured token limit; automatic task subdivision and repair are not implemented.
Only explicit append proposals join continued fields. An empty proposal list never proves an empty
object array. Model semantic choices, anchor consistency and inventory recall require measurement.

The value-only scorer measures exact documents and complete top-level record subtrees using multiset
counts, preserving duplicates and nested relationships. Missing/corrupt/failed results remain in the
expected population. It is not the official ExtractBench F1 and does not score grounding. Time to
first token remains unmeasured by the current aggregate generation timer.

The evidence trial additionally exports private authored proposal diagnostics and saved extraction
results through an explicit filename/size allowlist. It never exports complete model prompts or
raw benchmark files. An optional quote is resolved only inside its selected source occurrence.
Regular fully populated tables can use checked header mappings to copy every physical row; sparse,
wrapped, pivoted or ambiguous repeated-header tables require further work.
