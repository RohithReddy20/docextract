# Measured model trials

Most sections below are small authored development diagnostics, not ExtractBench scores or evidence
of a winning checkpoint. The ExtractBench development runs (first section) use 20 benchmark documents
chosen by a fixed hash order without looking at answers; they are seen during development and must
be reported as such. All model downloads and execution occurred in private Kaggle workers. No model
was downloaded or loaded locally; Modal was unused.

## ExtractBench development runs

Population: the first 16 short and 4 medium documents in the `docextract-dev/1` hash order. Every
document stays in the scored population; an unfinished or failed document scores as an empty
prediction, which already earns value F1 0.312 on this subset (schema defaults and nulls). Scores
are from the official `ExtractEvaluator` in a separate process.

- **eb-dev v3 (no score).** All 193 windows went to vLLM as one batch; at about 41 s per window
  per T4, nothing had finished at the deadline. Fixed by streaming (each answer stored at once).
- **eb-dev v4 (first scores).** Qwen3.5-4B **0.573** and NuExtract3 **0.575** value F1 (short 0.632 /
  0.630, medium 0.339 / 0.355); grounded F1 0.29 / 0.26, page F1 0.36 / 0.42. In 2.1 hours of
  inference 13 and 14 of the 20 documents finished; the rest were assembled from the answers saved
  by the deadline. Prompt reading dominated: Qwen's 148 requests read 2.06M prompt tokens (median
  14.9K) and wrote 77K (median 24). One scanned 29-page tax return whose 155K-token schema splits into
  22 groups took 136 of the requests and 84% of the prompt tokens, because every group was asked
  about every window. The T4 read prompts at about 300 tokens/s; vLLM's prefix cache hit about 1-2%
  (its prefix caching is unreliable for this hybrid architecture). On scanned forms most lost points
  are values put in the wrong field and check-box states. Writing `null` or omitting an absent field
  scored the same: these schemas declare `default: null`, which the scorer applies.
  `manifests/trials/extractbench-dev-v4.json`.

  Routing, checked offline (no model; answers used only to measure): for each expected value printed in
  a development document, is a window containing it still asked about the field's schema group? With
  24K-character groups and a 72K-character prompt budget, the 29-page tax return needs 37 of 132
  (window, group) requests and keeps 319 of 320 locatable values; 12K/16K groups keep 317 with 51/44
  requests, 32K keeps 310 with 28, 48K keeps 257 with 21. Single-window forms keep every value.

  Consequences: Qwen3.5-4B is kept (tied accuracy, shorter answers, no template overhead);
  windows are sent only the schema groups their text mentions (`route_groups`), prompts are sized to
  the context budget (`prompt_budget_chars`), scanned pages can be shown to the model as images, and
  experiments now run as small multi-variant sessions.

- **eb-dev s1 (small session, 16 short documents, three prompt variants on one model load per GPU).**
  Value F1 on the short documents: cited **0.690**, plain values **0.722**, plain values with page
  images **0.756** (v4 Qwen on the same documents: 0.632). Every document finished. Plain values won
  on 11 of 16 documents with similar grounding; images helped scanned forms (0.34 → 0.83, 0.67 → 0.87)
  but duplicated records on an 8-page scan (0.69 → 0.47), fixed since by page-aligned image windows.
  MTP decoding accepted almost every draft (mean acceptance length 3.0 of 3). Re-assembly on a laptop
  reproduced the score exactly; document-text strings scored slightly lower (0.7525), table completion
  changed nothing. `manifests/trials/extractbench-dev-s1.json`.

- **eb-dev s2 (small session, 10 documents: the 4 medium and 6 short scanned forms).** Plain values
  with page images, schema routing and budgeted prompts: mean value F1 **0.682**, every document
  finished. The 155K-token-schema tax return went 0.689 → 0.874, the Japanese report and the 44-page
  fund report from unfinished to 0.465 and 0.555, the two scanned P4 forms with their own text layer
  (now shown as images) 0.60/0.70 → 0.907. Locally, merging one entity described in two sections lifted
  a valuation report 0.473 → 0.699 (mean 0.705) without changing any other document. The fund report
  exposed a parser bug: word gaps that PDFium marks with generated spaces were dropped
  ("TheBenefitsandRisksofLeveraging"); fixed. `manifests/trials/extractbench-dev-s2.json`.

- **eb-dev s3 (20 documents not seen before).** The full current configuration (plain values, page
  images, routing, notes, entity merging, check-box guidance, word-gap fix) on the next 16 short and 4
  medium documents: value F1 **0.763** (short 0.786, medium 0.671), grounded 0.37, page 0.55. The
  seen development documents reached about 0.78 with the same configuration, so the gains carry over
  to documents never looked at. `manifests/trials/extractbench-dev-s3.json`.
- **Modal L4 probe 1 (6 seen documents, research budget).** One L4 in bfloat16 with MTP: engine start
  3.9 minutes (6 s warm-up versus 6.5 minutes on a T4), 213K-token KV cache, Flash Attention and the
  fused GDN decode kernel; 1.59M prompt and 307K output tokens in 18 minutes with only about 11
  requests in flight; prefix-cache hits about 37%. 20 of 213 answers hit the 6,144-token limit and
  were split and regenerated (40% of output tokens wasted), mostly dense table windows. Value F1 0.773
  over the six; the first two long documents 0.801 and 0.691. Page images are re-sent with every
  routed schema group, so the 29-page tax return read 916K prompt tokens. The first attempt stopped at
  start-up (FlashInfer's sampler compiles CUDA code; disabled since greedy decoding does not need it).
  `manifests/trials/modal-l4-probe-1.json`.

- **Modal L4 probe 2 (7 seen documents, four variants on one model load, 40-minute cap reached).**
  Capping windows at 400 cells (`max_window_units`) finished more documents and scored higher on every
  table document finished in both variants (budget appendix 0.346 → 0.573, labour catalog 0.293 →
  0.341, price list 0.557 → 0.568); it also avoids answers that overflow the 6,144-token limit and are
  regenerated. Image-size variants on the tax return did not finish (partial: 0.776 at 1,280 px, 0.779
  at 768 px, 0.752 without images); 768 px images are now used only for schemas split into more than two
  groups, whose pages are shown once per group. Under load the L4 read 2,600-3,700 prompt tokens/s.
  `manifests/trials/modal-l4-probe-2.json`.

- **Modal dress rehearsal (30 documents, 29 never seen; final settings; two L4 shards).** Value F1
  **0.816** (unseen 0.819; short 0.856, medium 0.729, long 0.766), grounded 0.43, page 0.58. 192
  answers, only 5 at the output limit (20 of 213 before the 400-cell windows); 1.22M prompt and 231K
  output tokens in 30 GPU-minutes including start-up. At this rate the full benchmark needs about
  7.5-8 L4 hours. `manifests/trials/modal-rehearsal-1.json`.

## Native value generation, first smoke

Frozen source archive: `bf9ba39901f69ae2eea477fa82464874d6ab10b7d86976a5e696b0ebc6017dce`.
The complete sanitized measurements are in `manifests/trials/native-values-v1.json`.
Both pinned 4B models ran sequentially on one Tesla T4, with FP16, SDPA, greedy non-thinking
generation, native Torch DeltaNet fallbacks, batch one and no offloading. Their weight hashes and
language-parameter loading were checked. Python 3.12.3, Torch 2.8.0+cu126 and Transformers 5.3.0
worked on the assigned driver. The exact resolved environment is in `runtimes/kaggle/uv.lock`.

| Six development cases, including failures | Qwen3.5-4B | NuExtract3 |
| --- | ---: | ---: |
| Exact complete documents | 3/6 | 2/6 |
| Original-schema-valid documents | 3/6 | 4/6 |
| Completed generations | 5/6 | 5/6 |
| Correct complete top-level record subtrees | 5/9 | 4/9 |
| Peak reserved GPU memory during generation | 8.66 GiB | 8.72 GiB |

The six cases cover sparse fields, presence/null/default distinctions, duplicate table rows, narrative
events, nested parents and multipage continuation. Qwen returned a surrounding array for the sparse
object and malformed JSON with incorrect parent associations for the nested case. NuExtract inserted
unobserved nulls into optional string fields, omitted one legitimate identical table row, and retained
heading prefixes in parent names. Both rejected the multipage case at the deliberately small 2,048
input-token limit. No pages or schema fields were dropped and no malformed output was salvaged.

Actual generated-task input sizes were at most 1,676 tokens, with a 256-token output allowance.
This does not establish the proposed 6,144/1,536 working budget. Aggregate generation took about
20 seconds per model over five tasks; load, hashing and download overhead are recorded separately.
Time to first token was not measured. Neither throughput nor full-run cost can be forecast from this
small sample. The cash-cost estimate was $0 using free Kaggle quota; quota consumption was not audited.

One Qwen control produced identical input and greedy output token hashes through full multimodal
and text-only wrappers. Both returned the same incorrect root shape. This establishes parity for
one input, not extraction correctness, numerical parity generally, or NuExtract wrapper parity.

## Source-proposal protocol (v2–v4)

The model selects source occurrences, record anchors and table-header mappings; code copies text.
v2 (3 cases, 6,144/1,536 token limits): **0/6 outputs accepted**: path/index and alias/text confusion.
`manifests/trials/source-proposals-v2.json`.

v3 (same 3 cases, enumerated choices and format examples): **0/6 accepted under the whole-batch
validator**. Inspection showed mostly formatting failures: Qwen wrapped two answers in Markdown
fences; NuExtract wrote `null` for an optional separator in every answer. Re-checking those same
saved generations with the revised validator (one complete fence accepted, `null` means default,
invalid items rejected individually) gives 2 of 3 exact documents per model; both still omit the
page-2 continuation. No new inference was involved. `manifests/trials/source-proposals-v3.json`.
Peak reserved GPU memory reached 12.7 GiB at 4.1K input tokens with the reference PyTorch DeltaNet
kernels, at about 11 generated tokens per second.

v4 (all 24 diagnostics, both models in parallel on the two T4s, revised validator):

| 24 authored cases | Qwen3.5-4B | NuExtract3 |
| --- | ---: | ---: |
| Exact documents | 12 | 12 |
| Schema-valid documents | 16 | 16 |
| Exact: sparse / presence / tables | 4 / 4 / 4 | 4 / 4 / 4 |
| Exact: narrative / nested / continuation | 0 / 0 / 0 | 0 / 0 / 0 |
| Wall time for 24 cases | 1,243 s | 1,243 s |

`manifests/trials/source-proposals-v4.json`.

## Cited-value protocol (v1)

The model writes the user's JSON shape with `{"value", "source"}` at each scalar, over compact
line-oriented evidence; code types, locates and identifies records (see `architecture.md`). Same
24 cases, same models and token limits as v4:

| 24 authored cases | Qwen3.5-4B | NuExtract3 |
| --- | ---: | ---: |
| Exact documents, as run | 14 | — |
| Exact documents, saved outputs re-checked by the fixed parser | 16 | 20 |
| Schema-valid documents | 24 | 24 |
| Exact: sparse / presence / tables | 4 / 4 / 4 | 4 / 4 / 4 |
| Exact: narrative / nested / continuation | 4 / 0 / 0 | 4 / 4 / 0 |
| Max input tokens; peak reserved GPU memory | 563; 8.1 GiB | 387; 8.1 GiB |
| Wall time for 24 cases | 421 s | 421 s |

The as-run parser merged two records that cited the same sentence; the fix (records in one response
are never merged) and two parsing relaxations were applied to the saved outputs, not to new
generations. Qwen merged two identically named parent groups and, in the continuation family,
filled the schema field `id` with cell references; neither model joined a description continued on
page 2. The next run changes the prompt wording (`ref`, repeated headings, continued values).
`manifests/trials/cited-values-v1.json`.

These 24 documents are small authored diagnostics. They compare protocols and expose failure
classes; they do not estimate ExtractBench accuracy or select a model.

## Engine probe on one T4 (Qwen3.5-4B, FP16)

Hugging Face Transformers 5.3.0 with the reference PyTorch linear-attention kernels needed 12.97 GiB
peak for a 5,333-token prompt and ran out of memory at 10,274 tokens; with `fla-core` 0.5.2 Triton
kernels the peak was 12.73 GiB, also out of memory at 10,274 tokens, and generation was slower
(7.8 versus about 13 tokens/s on the 24 diagnostics). Long prompts on a T4 are not practical
through Transformers.

vLLM 0.30.0 installed and loaded the same checkpoint on a T4 (7.99 GiB weights), selected Triton
attention plus Triton linear-attention prefill/decode kernels in FP16, compiled and captured CUDA
graphs. It then stopped at a configuration check: 256 concurrent sequences exceed the 135 recurrent
state blocks that fit, so `max_num_seqs` is now set explicitly. Throughput is measured in the next
runs. `artifacts/kaggle/private/probe-v1-output` holds the private logs.

## Benchmark schema audit

A private Kaggle CPU worker downloaded the pinned three JSONL splits, verified their hashes and
inspected all 252 short, 98 medium and 20 long cases without fetching PDFs or loading a model. The
first pass found 70 distinct complete-schema hashes, 66 inside the declared profile; four (17 cases)
declare Draft 7. After explicit Draft 7 support, the rerun accepts **all 70 schemas and all 370
cases** (`manifests/dataset-schema-inventory.json`). Schema support is not extraction accuracy.
