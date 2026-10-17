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

- **Full benchmark, final-v1 (all 370 documents, frozen settings, three L4 shards; partial).** Value
  F1 **0.747** over all 370 (short 0.834, medium 0.600, long 0.370; unseen 0.743 on 299, seen 0.765 on
  71), grounded 0.39, page 0.53. The run stopped early: two shards were preempted 34 minutes in and
  resumed from stored answers, and after 2.4 hours the launching machine lost its connection while the
  launcher waited on the orchestrating call, so Modal cancelled the run about 15 minutes before its
  deadline (the launcher now spawns the run and returns). 301 documents finished and score **0.819**
  (unseen 0.825), matching the rehearsal; 69 did not (13 short, 42 medium, 14 long; 1,724 requests
  still queued) and score 0.435 from partial answers. The rehearsal under-predicted the work: 592 of
  2,154 answers (27%) reached the 6,144-token output limit on long tables (3% in the rehearsal), and
  with the KV cache full each L4 ran about 20 requests at once (about 600 prompt and 340 output tokens
  per second). 13.0M prompt and 6.3M output tokens; finishing needs roughly 5-6 more L4 hours.
  `manifests/trials/final-v1.json`.

- **Comparison with the leaderboard.** ExtractBench's own one-shot run of the same model
  (`qwen3_5_4b_vllm_extract_oneshot_structured_output_file`: every page as a 150-DPI image, the full schema,
  JSON-object output) scores 82.4 overall and 89.1 on short documents, above final-v1 (74.7, short 83.4).
  On finished short documents final-v1 lost on digital PDFs (84.2), which it read as text only, on
  check boxes (21% of check-box fields wrong), on scans where the model copied OCR misreadings, and on
  filling fields the answer key leaves empty.

- **Probes (one L4, a few documents, several methods per model load; `runtimes/modal/probe.py`,
  `tools/probe_jobs.py`, `tools/score_probe.py`).** Eight short documents (six final-v1 scored badly,
  two it scored well): final-v1 47.9; the leaderboard recipe 67.3 with the template as shipped (it opens
  a think block that the JSON constraint then cuts off) and 75.2 with thinking switched off; repeated
  runs agree on 7 of 8 documents. 200-DPI images 81.0 against 77.9 at 150 DPI (seven documents both
  finished); the PDF text layer helps dense tables and hurts documents whose text layer is broken;
  an extra instruction to follow field descriptions lowered scores; thinking did not finish within
  10 minutes. On 24 short documents never used before (12 digital, 12 scanned): final-v1 85.1,
  one-shot 150 DPI 87.9, 200 DPI 89.9, 200 DPI with the text layer **91.1** (digital 81.7 → 92.9,
  scanned 88.5 → 89.2). `manifests/probes/`.

- **Medium and long documents, whole document in one request.** Four medium documents (13-21 pages):
  final-v1 67.8, whole document at 150 DPI with the text layer **90.3**, at 200 DPI 89.2, 5-page
  batches merged 79.7 (the model needs the whole document). Six unseen medium documents: final-v1
  80.6, whole document 86.3 when a refused request (a 760K-character schema) or a cut-off answer takes
  the old pipeline's result. Five long documents (28-77 pages) with small answers: final-v1 62.6,
  hybrid 76.5 (20 → 77 on one). Page images shrink by a quarter per round until a request fits the
  context window.

- **Evidence located after extraction (`docextract.oneshot.locate_evidence`, CPU only).** Each printed
  value is found in the stored parse (native text or OCR) in document order and cited by page and box.
  On the 24 unseen short documents: grounded F1 56.2 and page F1 70.9, against 44.8 and 65.5 for
  final-v1's cited pipeline. Typing answers to the schema (`conform`: numbers for string fields printed
  without ".0", two-state check boxes left empty become false) adds 0.6 value F1.

- **Run-to-run noise.** Repeating the 200-DPI method on the same 24 documents gave 91.7 and 89.8:
  vLLM's batching changes greedy outputs on some documents by up to 8 points, so a 24-document test
  only separates methods that differ by about 3 points or more. 250 DPI, an instruction to copy values
  verbatim and an instruction on check boxes all landed within that noise (89.5-90.7).

- **Large answers in one request (32K output tokens).** Four medium documents whose answers run
  20-24K tokens (financial statements final-v1 never finished): 38.2 → 100.0, 16.9 → 99.8,
  1.2 → 96.7, and a 77-page document (images shrunk once to fit) 80.4 → 100.0; a fifth was still
  writing at the deadline. Long answers decode at about 25 tokens per second per request, so such
  documents cost about $0.1 each; 52 documents expect more than 10K output tokens.

- **Qwen3.5-9B with FP8 weights (Apache-2.0; 11.1 GiB on the L4, 176K tokens of KV cache).** Same
  method, same documents: 24 short check documents 93.2 against 91.7 and 89.8 for two Qwen3.5-4B runs
  (scanned 89.2 → 92.9); eight hard short documents 82.0 against 77.4; seven medium documents both
  models finished 95.9 against 90.8. ExtractBench's leaderboard shows the same order (one-shot 9B 86.1,
  4B 82.4). Generation took about as long as with the 4B; the first load (FP8 quantization, no compile
  cache) took 5 minutes. `manifests/probes/q9-1.json`, `q9-2.json`.

- **Locating evidence by labels (rejected).** Choosing, among repeated occurrences of a value, the one
  printed after its field's name lowered grounded F1 on three of five probe sets (long documents
  26.1 → 19.1); evidence stays in document order.

- **Fresh check through the product pipeline (fresh-1).** 17 documents never used before (12 short,
  5 medium), Qwen3.5-9B FP8, launcher → `tools/extract_remote.py --protocol oneshot` → official
  scoring. The 24-minute GPU limit stopped three documents before their answers (two medium, one short);
  on the 14 answered: 93.8 against 82.3 for final-v1 (short 93.9 vs 89.2, medium 93.4 vs 57.0). Citing
  one box per page instead of one per text line raised grounded F1 on the same answers from 37.8 to
  41.1. An unanswered document is now left empty and marked partial (before, two-state check boxes
  filled in from an empty answer scored 64.6 on one document). `manifests/trials/fresh-1.json`.

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
