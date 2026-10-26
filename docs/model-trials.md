# Measured model trials

Most sections below are small authored development diagnostics, not ExtractBench scores or evidence
of a winning checkpoint. The ExtractBench development runs (first section) use 20 benchmark documents
chosen by a fixed hash order without looking at answers; they are seen during development and must
be reported as such. All model downloads and execution occurred on remote workers (Kaggle, Modal and
Colab); no model was downloaded or loaded locally.

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

- **Long tables read page by page (g1).** The 14 long documents final-v1 left unfinished expect
  40K-500K tokens of rows (check registers, creditor lists, fund holdings, clinical listings); 11 have
  a text layer on every page. A 66-page clinical listing (951 rows; final-v1 50.9), the table read one
  page per request (page image at 150 DPI, the page's text layer) and the other fields from one
  whole-document request, Qwen3.5-9B FP8: value F1 **94.0** with 5 of 66 pages unanswered at the time
  limit; per-page row counts were exact on almost every page. Nearly all remaining errors were group
  labels of rows continued from the previous page (a subject number printed only on the group's first
  row): a model reading one page gives them the label printed below them. Adding the end of the
  previous page's text as context did not fix that (90.5% vs 89.5% of rows exactly right on the 48
  pages both variants answered) and added duplicated rows. A repair in code (`carry_forward`: a row
  whose group label is first printed below its own line takes the previous page's last label) brings
  the pages both answered to 99.3% of rows exactly right and the document to **95.7**; on the answer
  keys of 7 long documents (about 45K records split into pages) it changes nothing. Engine findings:
  32 sequences at once outgrew the KV cache, and vLLM's recompute-on-preemption left some page
  requests unfinished after 17 minutes; queuing all 291 requests at once crashed the worker (each
  holds its decoded page images); JSON-object answers were pretty-printed in every probe so far
  because vLLM's xgrammar backend reads the no-whitespace flag from the engine, not the request; the
  L4 read about 1,000-1,500 prompt tokens/s for page-image requests. The text layer now comes in the
  PDF's own order, as in every probe (regrouping lines across the page mixed the columns of
  multi-column listings; on 7 of 10 long documents fewer than 60% of lines were the same in both orders).
  `manifests/probes/g1.json`.

- **JSON without whitespace (g2, rejected).** Setting the no-whitespace flag for the whole engine (the
  only place xgrammar reads it) emptied the answers: 52 of 53 page answers on a fund-holdings document
  came back as `{}` or nearly so, including all 19 pages that list holdings, since the model's first
  token after `{` is a newline and, with whitespace forbidden, it closed the object instead; requests
  also queued for grammar compilation and generation fell to about 5 tokens/s. Stopped after 8 minutes;
  answers stay pretty-printed. `manifests/probes/g2.json`.

- **Fund holdings read page by page (g3).** A 59-page fund report whose schedule lists 588 holdings
  under asset-class, industry and issuer headings (final-v1 21.1): value F1 **71.5**, every page
  answered in 9 minutes of generation, 590 rows found for 588, document-level fields all right. What
  remains is reading, not completeness: the model gives a security's name without the rate, coupon
  and maturity printed on the same row (562 rows), and on pages that continue a section it cannot see
  the asset-class heading printed pages earlier, so it leaves it empty or puts an industry heading
  there (511 rows). The label repair cannot settle which heading level a value belongs to, so it
  changed one row. With 24 sequences the KV cache was the limit (96-100% used with about 17 answers
  running; 220-470 output tokens/s). `manifests/probes/g3.json`.

- **Check register read page by page (g4, stopped early).** A 123-page check register, 3,308 rows of
  12 fields (final-v1 0.0). Pretty-printed answers ran to about 4,500 tokens per page, so the whole
  document needs about 40 L4 minutes at the measured rate; stopped after 37 pages to save credit. On
  those pages every one of the 986 rows was found, and 96.3% of their fields match the answer key; most
  of the rest is one convention: on a check's continuation lines the key leaves the type code empty
  (it is not printed there) while the model carries it down like the fields whose descriptions say so.
  `manifests/probes/g4.json`.

- **The whole check register on a Colab L4 (c1).** The register g4 stopped after 37 pages, with the
  same requests and engine, run on a Colab L4 through Google's Colab CLI (`tools/colab_probe.py`).
  All 3,308 rows on all 123 pages were found and 96.1% of their fields match the answer key
  (ExtractBench score 96.1; final-v1 0.0). 1,062 of the 1,410 wrong fields are the type code carried
  down to continuation lines, where the key leaves it empty; unlike the check number, date, vendor and
  EFT flag, its schema description does not say to carry it. The request for the seven document-level
  fields (district, title, dates, total) failed: the 123 page images and their text do not fit the
  131K-token window even with images shrunk four times, so those fields stay empty. The product has
  the same gap in a larger form: it reads a table page by page only when a whole-document answer is
  cut off at the output limit, and a document too long for one request goes to the windowed pipeline
  instead (the probes split the request by configuration). On the 22 pages both runs had answered when
  compared, 21 answers were byte-identical to g4's on Modal. About 550K output tokens in about 40 L4
  minutes of generation, over two sessions. Colab lessons: a VM whose kernel sits idle is reclaimed
  after about half an hour, so the job now runs inside a kernel cell that waits for it; the CLI's
  kernel connection took up to 40 s to open, past the client's 10 s default (`REQUEST_TIMEOUT`); an
  A100 has no FP8 arithmetic and vLLM 0.30's FP8 kernel fails on it; no G4 was available. Rates seen:
  L4 1.54 and A100 5.3 compute units an hour. `manifests/probes/c1.json`.

- **The product on long documents that do not fit one request (l1, Colab L4).** After the change
  that reads such a document's lists of records page by page and its other fields from its first and
  last two pages (and refuses a request with more page images than the engine accepts), the product
  itself (launcher, `tools/extract_remote.py`, `extract_cited` with the oneshot protocol, official
  scoring) ran on two long documents the full run had failed: a 114-page creditor list went 6.9 →
  98.3 and a 72-page fund report with three lists 3.2 → 79.6. Every whole-document request was
  refused at 150, 100 and 72 DPI, then all 186 pages were read one at a time. The document-level
  fields came right from the edge pages (title, case number, debtor, report date). The rest of the
  loss: about 1-2% extra rows in the large lists (7,651 creditors for 7,554; 4,370 holdings for
  4,299), and a two-row per-fund summary list read on every page (57 rows for 2), because each page
  alone looks like a fund's schedule; small summary lists need another treatment. 1.55M output
  tokens (pretty-printed JSON, up to 16K per dense page) took about 2 L4 hours over three jobs on
  one VM: the KV cache held only 8-14 answers at once (220-420 tokens/s, less on the densest pages),
  so a time limit stopped the first two jobs and the next resumed from the stored answers on the same
  VM (`--reuse`). Fixed on the way: the product's import path on Colab, vLLM's engine process
  outliving a stopped product (it held the GPU's memory), and a pointless repeat of pages cut off by
  the engine's own output limit. `manifests/colab/l1.json`.

- **FP8 KV cache (k1, Colab L4).** The check register's first 40 pages read page by page as in c1,
  with the running answers' KV cache stored in FP8 (vLLM `kv_cache_dtype`, uncalibrated scales) and
  up to 48 answers at once. The cache held 343,567 tokens instead of 184,025, about 20 answers ran
  at once instead of 15, and generation went from 338 to 472 output tokens/s over the same stretch
  (+40%). Answers did not change: 39 of 40 byte-identical to c1's, all 1,055 rows found, 96.66% of
  fields right (c1 96.60%). The product's engine now takes `kv_cache_dtype` (`--kv-cache-dtype` in
  tools/extract_remote.py and the launchers). `manifests/probes/k1.json`.
- **Full benchmark through the product (full-1, Colab).** All 370 documents with the current method
  (`manifests/colab/full-1.json`): value F1 **0.913** (short 0.940, medium 0.855, long 0.850; final-v1
  0.747: 0.834, 0.600, 0.370), grounded F1 0.530 (0.392), page F1 0.637 (0.527); the 292 documents
  never looked at during development 0.914, the 78 seen 0.906. It started on two L4 VMs (halves
  `--shard 0/2`, `1/2`), paused when the launching laptop slept, and finished on one G4 (RTX PRO
  6000, 96 GB, FP8) from the stored answers: about 4,300 prompt and 2,000-3,600 output tokens/s at 32
  answers, roughly 5-15 L4s. Found on the way: (1) holding the page images of 64 queued
  whole-document requests nearly filled a 52 GB VM (now at most 192 images admitted); (2) a job
  crashed in native code right after building a 466,296-token prompt, so a request the engine would
  refuse at any resolution (more pages than it accepts, or text beyond 8 characters a token of its
  window) is now refused before its prompt is built; (3) on the G4 the OCR model failed on every
  scanned page (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` in that image), so 160 documents failed until
  OCR learned to run without cuDNN when a blank page fails with it; (4) the official scorer counts a
  field the answer leaves out as wrong where the key holds null, so every schema property an answer
  leaves out is now written as null and a blank string read as null (on 33 finished documents with
  local keys: 92.4 -> 94.5, none lower). Every document was then assembled with the same source and
  scored on the VM. Largest remaining losses: a 13F filing answered with 32 of its 1,223 holdings and
  a normal stop (no cut-off, so its table was not read page by page), GSA labor-category tables, and
  composed fields in page-by-page reading (issuer plus security leg). `manifests/trials/full-1.json`.

- **Lists missed or given in part (lists-1, Colab L4).** Two causes found in full-1's answers.
  (1) The product only recognized a list of records declared as a plain array, while 230 of the 370
  schemas declare optional lists (`anyOf: [array, null]`): a document with such a list whose answer
  was cut off, or that was too long for one request, went to the windowed fallback instead (24
  documents, mean 0.881; 18 are tax returns whose 760K-character schema cannot fit one request and
  stay there). Optional lists now count, and a document too long for one request asks for its fields
  outside tables first, from its first and last pages; when even that is refused, the whole document
  goes to the fallback before any page is read. (2) A complete answer can list part of a long table
  and stop: two 13F filings were answered with 25 of 1,168 and 32 of 1,223 holdings. The product now
  checks an answer's lists against the PDF's text (`partial_lists`): lines printed like a list's own
  rows (the same kinds of tokens in the same order) that come from none of its records, on pages none
  of its records come from, at least ten times its rows and 50 in all, have the list read again page
  by page, and the pages' rows replace the answer's when there are more. Of 88 documents checked (73
  answered in one request with lists spanning five or more pages, 15 long ones) it flags only the two
  filings. On the nine documents whose handling changed (full-1's stored answers reused): the filings
  0.052 -> 0.998 and 0.042 -> 0.998 (every row), a customs entry 0.825 -> 0.896, a 66-page contract
  0.597 -> 0.943, another 0.703 -> 0.721, the tax return unchanged (fallback, no page read); but two
  long contracts fell from 0.943 and 0.826 to 0.372 and 0.336 and a GSA price list from 0.551 to 0.467.
  Read one page at a time, their small line-item lists took numbered clause lists and item numbers
  repeated in other sections, and the price list took each labor category twice (its rate table and
  its qualifications table); a page alone cannot show which table the schema means. Net +0.32 on the
  benchmark mean. `manifests/colab/lists-1.json`.

- **Long documents shown by their text (long-1, long-2, Colab L4).** A document too long for one
  request even with 72-DPI images now gets one more whole-document request before any page is read
  alone: pages with a text layer are shown by their text, and only the first and last two pages and
  pages without text keep their images (at the first resolution). Four long contracts (66-95 pages,
  5-70 line items), full-1 (windowed fallback) -> page by page (lists-1) -> text only (long-1) -> text
  plus first and last pages (long-2): 0.597 -> 0.943 -> 0.855 -> **0.962**; 0.703 -> 0.721 -> 0.807 ->
  **0.956**; 0.826 -> 0.336 -> 0.870 -> **0.891**; 0.943 -> 0.372 -> 0.657 -> 0.615. With no page image at
  all, a purchase order's boxed fields were misread (its order number joined to the contract number,
  a wrong date); with the first pages' images they were all right, but that answer listed its two
  line items and left out three funding sub-items. Two long fund reports (59 and 72 pages) whose
  holdings are read page by page did not change. `manifests/colab/long-1.json`, `long-2.json`.

- **Room for 32K-token answers (out32-1, Colab L4).** Eight medium documents whose whole-document
  answers full-1 cut off at 16,384 output tokens and then read page by page (fund holdings,
  communications logs, an awards schedule; 148-340 rows), where reading pages alone put sector
  headings in the asset-class field, read headings and totals as rows, and lost the run of rows across
  pages. With room for 32,768 tokens seven answered in one request (22-28K tokens): 0.686 -> 0.997,
  0.793 -> 0.999, 0.728 -> 0.984, 0.778 -> 0.989, 0.756 -> 0.973, 0.762 -> 0.945, and the awards
  schedule 0.710 -> 0.533 (the right number of rows, values written differently); the eighth (340 rows)
  was cut off again and read page by page as before (0.679 -> 0.695). +0.33 on the benchmark mean. Long
  answers next to long prompts ran slowly on the L4 (a few tokens per second per answer with three at
  once), so such runs belong on a larger GPU. Raising the engine's output limit changes the identity
  of every request, so the answer store now also accepts an answer stored under the same engine with a
  lower limit (`--legacy-max-new-tokens`) unless that limit cut it off: a finished greedy answer would
  not change, a refusal still applies with less room left for the prompt, and only cut-off answers
  are asked again (121 of 21,610 stored). `manifests/colab/out32-1.json`.

- **More cut-off tables with 32K answers (out32-2, Colab L4).** Seven more such documents: a GSA price
  list 0.551 -> 0.897 (all 98 rows in one answer; page by page it had taken each category twice) and a
  209-row table 0.882 -> 0.963; a 196-row table fell 0.957 -> 0.873; four (207-335 rows) were cut off
  again and read page by page from the answers stored under the 16K limit, unchanged, which is how the
  store's reuse across output limits was confirmed on real answers (eight requests generated in all).
  `manifests/colab/out32-2.json`.

- **OCR text on every scanned document (ocr-2, Colab L4).** The 138 scanned documents that do not take
  the windowed fallback, run with `ocr_text` from the stored answers; the 100-minute limit stopped it
  with 98 finished. On those 98: 0.908 -> **0.916** (short 0.922 -> 0.927, medium 0.857 -> 0.878); 34
  better, 21 worse, 43 unchanged. The largest gains are scanned financial slide decks and claim
  forms (0.643 -> 0.857, 0.763 -> 0.910, 0.802 -> 0.964), the losses forms where the model copied an OCR
  misreading. +0.24 on the benchmark mean. `manifests/colab/ocr-2.json`.

- **A cut-off table read on from its last record (cont-1, Colab L4).** An answer cut off at the
  output limit used to be thrown away for its table, which was then read one page at a time. Now the
  records it completed are kept, the page its last record is printed on is found in the PDF text, and
  the table is read on from there a few pages per request (half as many pages as the answer covered,
  at most eight), each request given the last record read so far ("these pages continue a list ...
  list only the records printed after it; a record that continues its group keeps its group fields").
  A table that would take more than 12 such requests in a row is still read page by page, all at
  once. Six tables cut off even with room for 32K tokens (207-766 rows), against full-1: 0.688 ->
  0.945, 0.679 -> 0.899, 0.714 -> 0.885, 0.863 -> 0.894, 0.874 -> 0.855, 0.718 -> 0.703; +0.17 on the
  benchmark mean. Some rows are still lost where one request's answer is cut off and the next begins
  (282 of 340 rows in one). The engine's identity no longer hashes the engine and prompt source
  files (any edit to them discarded every stored answer): it names a prompt version, fixed at the
  values of the source that generated the stored answers, and `tests/test_prompts.py` pins the
  prompts already in use. `manifests/colab/cont-1.json`.

- **Reading on one page per request instead (cont-2, cont-3, Colab L4).** To stop rows being lost
  where one multi-page answer was cut off and the next began, the same six tables were read on one
  page per request, records printed above the last one read on its page dropped. Worse on the whole:
  0.945 -> 0.887, 0.899 -> 0.874, 0.885 -> 0.805, 0.894 -> 0.893, 0.855 -> 0.867, 0.703 -> 0.717
  against cont-1, the estimate 0.9260 -> 0.9256. Answers for a single page listed rows of the pages
  before it again (674 rows for a table of 561). Not kept: the multi-page read-on stays. A document
  may now take up to 48 rounds of requests (cont-2 stopped three after 12). cont-3 was lost three
  times to this machine's network dropping (the driver took the VM to have stopped answering and
  gave up); it now waits while it cannot reach Colab at all. `manifests/colab/cont-2.json`,
  `manifests/colab/cont-3.json`.

- **Qwen3.8-27B in place of Qwen3.5-9B (q27-1, Colab G4).** The leaderboard's one-shot runs score
  the 27B 89.75 against the 9B's 86.10. The same method (32K answers, scanned pages with their OCR
  text) with the 27B's own FP8 checkpoint (arm "d", 31 GB, so a G4 rather than an L4), on 30 short
  and 30 medium documents drawn at random: all 60 finished in 65 minutes, value F1 0.926 -> **0.963**
  against the 9B's best scores on the same documents (short 0.953 -> 0.971, medium 0.900 -> 0.954;
  33 better, 7 worse, none by more than 0.05). Largest gains where the 9B misread or skipped rows:
  GSA labor categories 0.437 -> 0.905, an earnings deck 0.558 -> 0.857, scanned remittance advices
  and RRC forms by 0.08-0.15. Scaled to the benchmark's 252 short and 98 medium documents, long ones
  unchanged: about 0.953 (+-0.007 from the sample). `manifests/colab/q27-1.json`.

- **Estimate from the targeted runs.** full-1's per-document scores with each run's documents
  replaced (lists-1's filings and customs form, long-2's contracts, out32-1, out32-2, ocr-2): value F1
  0.913 -> about **0.926** (short 0.940 -> 0.942, medium 0.855 -> 0.895, long 0.850 -> 0.868), from 120
  documents rerun. Not a run of the whole benchmark: documents whose answers would also change under
  the combined settings (cut off at 16K but not tested at 32K, the 40 scanned ones left unfinished)
  keep full-1's scores.

- **OCR text for scanned pages (ocr-1, Colab L4).** The model sees each scanned page as an image only
  (the text layer it is given is the PDF's own). With `ocr_text`, a page whose text came from OCR shows
  its OCR lines as its text. On 25 scanned documents with dev keys (21 short, 4 medium), generated
  afresh in one session with and without: 24 finished both, value F1 0.911 -> **0.926** (full-1's
  answers 0.917); short documents +0.8 points on average, the three medium ones +5.8, +9.2 and +3.6
  (a 4-page remittance whose claim number the model had read with one zero too few: 0.776 -> 0.879).
  One form fell 0.958 -> 0.823: the OCR misread its register number and split a casing row, and the
  model followed the text instead of the image. `manifests/colab/ocr-1.json`.

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
