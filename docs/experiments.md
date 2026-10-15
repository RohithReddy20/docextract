# Automatic schema extraction: research and controlled trials

No checkpoint is selected and B0 has no model dependency. All model downloads, model caches and model
execution are restricted to Kaggle or Modal; none belong on the local machine, including small models.
Use Kaggle for development and controlled trials. Reserve Modal L4 for frozen final inference and
full-benchmark runs. Source inspection and actual remote trials are reported separately.
No training on ExtractBench answers. No paid work without a prior estimate of total cost and an
explicit spending guard. Account credentials, private prompts and account balance details belong
outside this repository.

The first learned comparison can use the existing native PDF evidence without waiting for a docparse
export. Run the pinned schema inventory remotely before a full benchmark. Freeze authored
development/holdout families: schema, renderer, vocabulary, pagination and document styles must
differ across the holdout. Include narrative records, negative regions, identical source occurrences,
competing mentions and changed headers. The real docparse export remains necessary for the intended
parser integration; it is not a prerequisite for native-text model trials.

The first two candidates have completed a small Kaggle value-generation smoke; see
[measured results](model-trials.md). The schema-only engine and remote inventory now exist.
The source-inspection notes below retain their original research context.

Compare these provisional checkpoints on identical bounded tasks and evidence windows:

| Arm | Checkpoint | Provisional revision | Representation to measure |
| --- | --- | --- | --- |
| A | Qwen/Qwen3.5-4B | bbaaae1f57d79c7bf696d619ffbbb0c2c8e5ef98 | FP16, non-thinking |
| B | numind/NuExtract3 | 9968a5bd4cf562d3700a57dad097be2b6dfee34b | Native typed template, FP16 |
| C | Qwen/Qwen3.5-9B | cc5442c03a5c0bff0bd4c6888d9a40029c637733 | NF4, FP16 compute |

A/B have a measured small-context runtime; C remains a research-plan candidate. This does not
establish quality, a full working-context envelope, or a winner.
Reverify each pinned model card and runtime license before use. Start with independent single-T4
smoke trials, not combined memory across GPUs: 2,048 input / 256 output tokens, batch one, no offload.
Record actual runtime kernels, GPU and host load peaks, latency and failure behavior. A checkpoint
fitting in memory does not establish usable throughput. Reject unstable or unsupported tasks rather
than dropping source pages, schema fields or unfinished records.

Compare common typed value/record outputs first. Measure source binding and regeneration versus
mapping-and-copying separately so an interface advantage is not attributed to model quality. NuExtract schema conversion
must report unsupported constructs; it must not silently choose a branch.

Required measurements before any frozen full run:

- Original-schema support/validity, exact typed values, critical-field accuracy, and complete-record
  precision/recall, including duplicates and parent-child associations.
- Record-discovery recall using an independently known inventory, plus audits of negative regions.
- Citation precision, page/box scores, occurrence correctness, multiline completeness and over-citation.
- Task input/output tokens, pages/cells/records/schema complexity, tail latency, selected kernels,
  host/GPU peaks, truncations, retries, cancellation outcomes and escalation rates.
- Full-corpus workload and inclusive load/idle/retry/storage/evaluation costs, with a conservative
  remaining-work forecast and enforceable reservations. Maintain an inclusive failed-attempt ledger.
- A frozen manifest of code, schema/data hashes, checkpoint/tokenizer/processor/template revisions,
  quantization, runtime wheels/image, generation limits, seeds, routing and failure denominators.

Use free development compute for trials. Reserve Modal L4 for frozen final inference/full-benchmark
work; no Modal development or training. State the total estimated spend before launching any GPU job.
The available private budget is a ceiling, not a throughput forecast. Do not add its balance here.
Keep every expected benchmark document in evaluation accounting, including failures and incomplete
metric support. An authored fixture score is only a contract check, never a general-quality score.

## Source-verified research update: 2026-10-14

At the time of this source-inspection snapshot, only public cards, configuration, licenses and
source code had been inspected. No model had been loaded or measured. Later remote results are
recorded in `model-trials.md`; no checkpoint is selected. Public model-file
hashes and implementation revisions are recorded in `manifests/model-research.json`.

### The learned step belongs before the current backend

`models.WorkUnit` already contains manually bound `FieldCandidate` values, source IDs and records.
`engine._validated_output()` requires the returned field set, record IDs, source IDs and lexemes to
match those inputs. This is a typed copying interface. Adding Transformers there would not discover
records or determine which source occurrence answers the user's schema.

The next implementation should add a separate interpretation interface:

```text
original schema -> supported schema tasks + all relevant source windows
  -> learned record discovery / field association / table mapping
  -> validated local source bindings
  -> existing copying, record assembly, journaling and original-schema validation
```

An interpretation task must receive schema definitions/descriptions and source evidence, without
human-authored label or column mappings. Its proposals need schema pointers, source spans, record
anchors, parent/continuation relationships and unresolved cases. They must be allowed to introduce
new record hypotheses. The final copier should continue to reject nonexistent source references.

Use compact window-local line/token aliases and maintain their source-character mapping outside
the model. Validate offsets/quotes against the supplied window. A model's claimed bounding box is
not native PDF evidence. Ambiguous duplicate-valued mentions remain unresolved unless the local
occurrence and context can be established. Arrays are assembled from occurrences, not value equality.

### Qwen versus NuExtract is a specialization comparison

The pinned [NuExtract3 card](https://huggingface.co/numind/NuExtract3/blob/9968a5bd4cf562d3700a57dad097be2b6dfee34b/README.md)
identifies Qwen3.5-4B as its base. Both pinned configs use `Qwen3_5ForConditionalGeneration` with
the same text-layer layout. A and B therefore share an architecture; NuExtract is not a fallback
for Qwen's kernel requirements. Both checked LICENSE files contain Apache-2.0.

NuExtract expects a typed extraction template. Its missing-information convention is `null` or `[]`,
so values alone cannot distinguish an absent field from an observed null or a proven empty array.
Preserve separate source-presence states. Its `verbatim-string` type normalizes whitespace; original
native characters and lexemes still need independent preservation. General `string` permits more
abstraction, so identifiers should not silently acquire that behavior through template conversion.

The provider's internal benchmark is useful author evidence, with a different evaluator and setup.
It does not establish a winner on this system, grounding, quantized inference or held-out schemas.

### The current NuExtract converter has a different API and a narrower contract

The verified PyPI `numind==0.4.0` wheel contains the same converter source inspected at SDK commit
`cc8d5895792c527ccaa20512f25820c07da45293`. Its
[converter](https://github.com/numindai/nuextract-platform-sdk/blob/cc8d5895792c527ccaa20512f25820c07da45293/src/numind/nuextract_utils/template_conversion/json_schema.py)
returns a dictionary with `template`, `schema_status`, `instance_status`, `adapted_instance`,
`incompatibilities` and `descriptions`. The pinned model-card tuple-unpacking example is stale for
this release. The SDK was inspected without installing it into docextract.

Set `omit_unsupported_branches=False`. Check conversion status and descriptions, then validate the
original schema independently: the template cannot represent required/nullability/bounds/length
rules even when shape conversion reports `fully_converted`. Some unions require an `instance` to
select a branch. Expected benchmark answers must never be supplied as that instance. Post-generation
candidate validation must preserve raw output and must not silently replace values with corrections.

These are source/API findings, not executed converter or model accuracy results.

### GLiNER2.5 is a bounded discovery candidate

The [pinned small checkpoint](https://huggingface.co/fastino/gliner2.5-small-v1/tree/3ec6d3dd7e1e93a7cf9b46096fa47aeda61c711c)
declares Apache-2.0 and a boundary architecture. Its card documents half-open character offsets;
load it with `AutoExtractor` and retain spans. It accepts its own schema rather than arbitrary JSON
Schema. It cannot supply text omitted by native parsing or replace perception on scans.

The pinned config has a candidate budget of 192 and 32 record instance queries. In the inspected
[record implementation](https://github.com/fastino-ai/GLiNER2/blob/b86d35c02f2c1a4d9840ea50197bd1b6d13b031e/gliner2/models/boundary/records.py),
natural mode seeds records from anchor occurrences; the fixed queries concern anchorless mode.
Neither setting proves unlimited recall. Test capacity pressure and keep bounded windows.

Inspecting [long-text merging](https://github.com/fastino-ai/GLiNER2/blob/b86d35c02f2c1a4d9840ea50197bd1b6d13b031e/gliner2/inference/chunking.py)
shows why spans must survive formatting and merging. Docextract should own occurrence identities,
cross-window reconciliation and parent associations. Independent trials must measure proposal recall,
false records and repeated identical occurrences; no end-to-end superiority is established.

### Memory and kernel estimates are still estimates

The pinned weight-index metadata reports approximately 8.68 GiB for each 4B checkpoint and 17.98 GiB
for 9B. These are index-reported tensor payloads, not measured load peaks. NuExtract publishes a
9,078,620,504-byte primary file plus a separate 241,200,704-byte multi-token-prediction file. Its index
includes both, while the Hub parameter summary reflects the primary file. Record whether MTP tensors
are loaded or unused, alongside missing/unexpected keys; repository size is not working model memory.

The [9B config](https://huggingface.co/Qwen/Qwen3.5-9B/blob/cc5442c03a5c0bff0bd4c6888d9a40029c637733/config.json)
has an untied output head, vocabulary 248,320 and hidden width 4,096. If both vocabulary matrices stay
FP16, they alone require about 3.79 GiB. Assuming every remaining parameter were four-bit gives an
optimistic parameter-storage floor around 7.34 GiB, before quantization metadata, unconverted modules,
caches, activations and workspaces. Calling this a roughly 4.5-GiB model would be misleading.

The inspected [Transformers 5.3.0 implementation](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
has optional fast kernels and native PyTorch delta-rule fallbacks with FP32 intermediates. Start by
measuring a pinned FP16/SDPA runtime without optional kernel compilation. No T4 throughput or peak
memory is verified by the existence of that fallback. The quantization library's documented hardware
support does not validate the entire model/runtime combination.

### Controlled experiment order

1. Freeze a proposed 24-document authored diagnostic set spanning scalar distractors, forms, tables,
   narrative events, nested parents and continuation. Include unknown/absent fields, genuine duplicate
   records, negative regions and instruction-like source text. The 24-case diagnostic generator is now implemented;
   local scripted tests remain contract evidence, not learned accuracy.
2. Run A and B on identical native evidence windows using their native input conventions, with
   thinking disabled, batch one, explicit single-device placement and no CPU/disk offload. Require
   schema-only task planning, with no human binding file. Start at 2,048 input/256 output tokens.
3. Compare typed field/record values first. Separately test source-binding output and automated
   table mapping/copying so interface changes are not attributed to checkpoint quality.
4. Admit C only after its quantized load/context gate passes. Its comparison changes both capacity
   and quantization; it cannot isolate the effect of parameter count. If informative, pair 4B FP16
   against 4B NF4 as an additional quantization control.
5. Add GLiNER proposals as one controlled discovery ablation. Add image crops only to investigate
   perception errors. Stop and reschedule unfinished source ranges rather than dropping pages or
   schema properties. Do not assume the proposed smoke output budget can fit a complete wide record.

Use original-schema validity, exact/typed and critical-field accuracy, complete-record precision and
recall, parent association, occurrence-level evidence precision and multiline support as separate
measures. Record token use, load/host/GPU peaks, kernels, p50/p95 latency, truncation, retries,
candidate saturation and residual unexamined/negative regions. Select on held-out schema/style
families; a small benchmark diagnostic subset only checks interfaces/failure modes.

Constrained decoding may enforce the small internal proposal protocol; it must not be used as an
accuracy claim. [JSONSchemaBench](https://arxiv.org/abs/2501.10868) separates schema coverage, efficiency
and output quality. A [recent small-model study](https://arxiv.org/abs/2609.23742) also separates schema
validity from content correctness, but its 14 authored tasks are not a document-extraction benchmark.
Run a paired decoding ablation before treating constraints as a quality improvement.

The plan itself does not launch GPU trials. A future free Kaggle trial has estimated cash cost $0 only
while free quota is available; its duration and quota consumption remain unmeasured. Re-estimate
total cost before launch. Modal remains reserved for a frozen final configuration and full-run plan.

### Remote runtime follow-up

The [Kaggle trial design](kaggle-trial.md) specifies the next comparison, preflight measurements,
remote-only cache placement and permitted result exports. The current Kaggle Docker source inherits
a moving Colab runtime and leaves Transformers open above version 5; it does not establish a tested
Python/CUDA/runtime combination for this project.

Transformers 5.3.0 already provides a text-only Qwen class, extraction of `text_config`, and the
multimodal-to-text checkpoint-key mapping. Measure this path for A/B before loading an unused vision
encoder. This is a proposed GPU-memory saving, not a measured result or a reduction in checkpoint
download size: text and vision tensors share checkpoint files. Preserve the checkpoint's embedding
tying, tokenizer, chat template and generation settings; reject missing language weights. The first
remote load needs an output-consistency check against the full wrapper on text-only input before
this path becomes the comparison baseline.

The same generation implementation already requests only the final-position logits for ordinary
generation. A custom smoke/profiling forward must also use `logits_to_keep=1`; otherwise a 2,048-token
FP16 logit tensor at this vocabulary alone is 970 MiB. That arithmetic is not a measured model peak,
and the ordinary `generate()` behavior is not a new optimization introduced by docextract.

The newer Transformers 5.13.0 documentation's illustrated extra kernel is gated to GB10/SM121.
Its published speedup does not establish a T4 benefit. Keep optional kernels and runtime upgrades as
separate, pinned, output-checked experiments. No runtime recipe has yet been validated on Kaggle.
