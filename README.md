# docextract

Give it a PDF and a JSON Schema; get back JSON that follows the schema, every repeated record, and
for each value the exact characters on the page it came from. Apache-2.0, Python 3.12, uv. Separate
from docparse. Target benchmark: [ExtractBench](https://github.com/run-llama/ExtractBench).

## Where it stands

- **Works and tested locally:** native PDF text with stable character/line/cell ids and page
  coordinates; schema support for all 70 ExtractBench schemas (JSON Schema 2020-12 and Draft 7);
  record assembly that keeps identical printed rows separate; null/missing/default handling;
  crash-safe resumable processing; official ExtractBench output and scoring adapters.
- **Model extraction (no manual mappings):** in the recommended mode the whole document goes to the
  model in one request — every page as an image, the PDF's own text layer and the full schema — and
  the model answers with one JSON object (`extract_cited(..., protocol="oneshot")`, `docextract.oneshot`;
  `--protocol oneshot` in `tools/extract_remote.py`, as in `manifests/modal/fresh-1.json`). The
  answer is typed to the schema and every printed value is then located in the parsed page text
  (native text or OCR), so each value keeps its page and box as evidence. A request too long for the
  model's window is retried with smaller page images. An answer cut off at the output limit while
  writing a list of records (a table longer than one answer can hold) is completed page by page: one
  request per page image with its text, rows kept in page order, and rows continued across a page
  break given their group's label by code (`carry_forward`). A request that still does not fit, or an
  answer cut off elsewhere, falls back to the windowed pipeline (plain values per window, located by
  code). Recommended model: Qwen3.5-9B (Apache-2.0; `--arm c --quantization fp8`), weights quantized
  to FP8 at load so it fits a 24 GB L4. Models run only on Kaggle, Modal and Colab; nothing is
  downloaded locally.
- **Built for the real benchmark, being measured now:** OCR (docTR) for pages without a text layer,
  with sideways and upside-down scans read upright; filled PDF form fields read as text;
  splitting oversized schemas into field groups; streamed generation with vLLM (which runs
  Qwen3.5-4B on a T4; Hugging Face Transformers ran out of memory near 10K tokens), where every
  answer is saved as it finishes and partly answered documents still assemble; optional
  JSON-schema-constrained decoding, multi-token-prediction decoding, smaller windows for dense
  tables, and copying of table rows the model skipped. Stored answers can be re-assembled and
  scored on a laptop CPU without a model (`tools/replay_local.py`).
- **Measured about ExtractBench** ([workload](manifests/workload-inventory.json)): 24% of its 4,869
  pages have no text layer and 130 of 370 documents are entirely image-only; schemas reach ~158K
  tokens (56 documents exceed 12K); 20 long documents hold most of the answer volume.
- **ExtractBench, full benchmark (partial run, previous pipeline):** value F1 **0.747** over all 370
  documents (short 0.834, medium 0.600, long 0.370; the 299 documents never looked at during
  development 0.743). The run was cut short: 301 documents finished and score 0.819; 69 (mostly medium
  and long) did not finish ([record](manifests/trials/final-v1.json)). ExtractBench's own one-shot run of
  Qwen3.5-4B scores 82.4, which showed that the windowed text-first pipeline lost accuracy.
- **Since then, measured on samples only (no further full run):** the whole-document method beat the
  previous pipeline on every sample ([details](docs/model-trials.md)). On 17 documents never used
  before, through the product pipeline with Qwen3.5-9B: 93.8 against 82.3 on the 14 that were answered
  before the time limit (short 93.9 vs 89.2, medium 93.4 vs 57.0;
  [record](manifests/trials/fresh-1.json)). Earlier probes: 24 unseen short documents 85.1 → 93.2 (9B);
  medium documents 67.8 → 90.3 and 90.8 → 95.9 (4B → 9B); medium documents with 20K-token answers that
  the full run never finished 34 → 99; long documents with small answers 62.6 → 76.5. Evidence located
  after extraction: grounded F1 56.2 vs 44.8 on 24 short documents. Probes run with
  `runtimes/modal/probe.py` (a few documents, several methods per model load, a few cents each), or
  the same requests on a Colab GPU with `runtimes/colab/probe.ipynb` (answers saved to Google Drive as
  they finish; a rerun after a disconnect asks only what is left).
- **Long tables (14 long documents list 40K-500K tokens of rows), measured on three:** read page by
  page, a 66-page clinical listing went 50.9 → 95.7 and a 59-page fund schedule 21.1 → 71.5; on a
  123-page check register (0.0 in the full run) all rows of the 37 pages read before stopping were
  found, with 96.3% of fields right ([details](docs/model-trials.md)). Section headings printed pages
  before a row are still guessed wrong at times, and such documents cost 10-40 L4 minutes each.
- **Not done yet:** schemas too large for one request (the 155K-token tax return) still go through the
  windowed pipeline; long tables on scanned pages and in newspaper-style columns are untested; evidence
  for long documents is weaker than the windowed pipeline's (the same value printed in many places);
  handwriting is untested. The Modal launcher (`runtimes/modal/final_run.py`) prints its worst-case cost offline,
  refuses a final run until the configuration is frozen, and spawns runs so a disconnect cannot stop them.

A deterministic baseline that needs hand-written label/column mappings (`bindings`) is kept as a
contract test; it is not general extraction. The frozen full-run configuration (`manifests/final-run.json`)
used Qwen3.5-4B and the windowed pipeline; the recommended mode is described above.

## Run the authored example

Install [uv](https://docs.astral.sh/uv/), then run from this checkout:

```sh
uv sync --frozen
uv run python -m docextract demo --output artifacts/demo
```

The command authors and reads a real two-page PDF, resolves a user JSON Schema and binding plan,
extracts native evidence, copies typed values, assembles records and validates the original schema.
Generated artifacts stay ignored:

```text
artifacts/demo/
  authored-order.pdf
  schema.json
  bindings.json
  expected.json                 # authored fixture truth, never passed to extraction
  run/
    journal.sqlite              # durable task/raw-output commits
    result.json                 # data, citations/source atoms, records, validation, coverage, usage
    <run_id>/                   # versioned snapshot, work plan and saved result
```

Expected behavior: two parent departments, four item rows, six source records and seven committed
tasks. Two identical printed rows remain separate records. One description continues onto page two
and cites both pages. `document_id` retains `00042`; observed `NULL`, numeric zero, a missing contact,
and explicitly applied currency default have distinct provenance states.

The schema is valid and execution is complete. The result status is deliberately `partial`: the
contact is absent. Record/inventory completeness remains `not_independently_verified` independently
of the operational queue. Repeating the command resumes the same content-addressed tasks without
duplicating records or regenerating committed outputs.

Run the extractor separately on those same input files:

```sh
uv run python -m docextract schema-support artifacts/demo/schema.json
uv run python -m docextract extract artifacts/demo/authored-order.pdf \
  --schema artifacts/demo/schema.json --bindings artifacts/demo/bindings.json \
  --workspace artifacts/standalone --apply-defaults
uv run python tools/render_evidence.py artifacts/demo/authored-order.pdf \
  artifacts/demo/run/result.json --output artifacts/demo/evidence-overlays
```

For another document, supply its PDF, original schema and a `bindings/1` plan. For example, a plan's
`fields` maps JSON Pointers to printed labels (`{"/invoice_id": "Invoice ID"}`), while a table binds
an array path, key field and column labels. See the generated example for nested section bindings.
The CLI prints validation and completion status; `extract` returns an artifact even for a partial
or schema-invalid result. Unsupported schemas fail before parsing with exit code 2. An interrupted
or rejected generation raises and retains its journal; rerun with the same workspace to recover.

## Tests and official benchmark integration

```sh
uv run python -m pytest -q
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run --extra benchmark python -m pytest -q
uv run --extra benchmark python -m docextract project-extractbench \
  artifacts/demo/run/result.json --output artifacts/demo/extractbench.json
```

The first test command skips the optional official integration module when the extra is absent.
The `--extra benchmark` test run executes it against the exact ExtractBench Git commit and
`parse-bench==1.0.4` in `uv.lock`. It tests the actual provider registry and scorer, authored evidence
geometry, duplicate-row recall, saved raw-result round trips and normalization without source I/O.
No runner/provider SDK extras, paid APIs, datasets or model weights are needed.

In a benchmark runner process, call `docextract.integrations.extractbench.register()` once to register
`docextract_b0`. Supply `bindings` and `workspace` in pipeline config and the original user schema in
`InferenceRequest.schema_override`. The provider accepts a single PDF and has no sidecar/answer/rule
reader. `run_inference()` performs extraction; `normalize()` is an offline, deterministic projection
of its saved extraction result. It cannot repair or complete the extraction. B0's declared bindings
make this a contract baseline, not a general benchmark submission.

The output contains only user data and official field citations. Internal status, uncertainty,
source identities and telemetry stay in the saved raw result. Bracketed citation paths, one-based
pages and normalized COCO geometry are serialized at this boundary. Unrepresentable property paths
raise rather than silently changing names.

## Dataset preparation stays remote

The dataset revision is pinned to `51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc`; reference code commits are
in `manifests/references.json`. Local metadata inspection needs no dataset download:

```sh
uv run python tools/prepare_benchmark.py --output artifacts/benchmark-metadata
```

On a **Kaggle CPU worker**, prepare the schema audit without downloading PDF documents:

```sh
uv run python tools/prepare_benchmark.py \
  --download --schemas-only --output /kaggle/temp/extractbench
uv run python tools/schema_inventory.py \
  --snapshot /kaggle/temp/extractbench/snapshot --output /kaggle/working/schema-inventory.json
```

Preparation records SHA-256 hashes and verifies split hashes from the pinned metadata. Inventory
verifies those hashes, counts schemas and reports profile issues, persisting no answers or field
rules. The remote audit checked all 370 cases: 70 distinct full-schema hashes, all inside the
declared profile (JSON Schema 2020-12 or Draft 7). This counts complete schema definitions including
annotations, not semantic schema families; see `manifests/dataset-schema-inventory.json`. Keep ground truth in the evaluation process/storage, separate from
inference workers, when constructing the full remote run.

## Learned extraction and Kaggle trials

Two schema-only paths need no manual label/column binding plan. The primary **cited-value** path
(`docextract.cited.extract_cited`) asks the model for the user's JSON shape with a source reference at
every value, then types each value against the original schema, finds it in the cited native text,
identifies records by the characters they cite and validates the original schema. The older
**source-proposal** path (`docextract.learned.extract_learned`) asks for spans, record anchors and
table-header mappings and copies text itself. Both reject malformed items individually and split a
window that runs out of output, input or GPU memory. On the 24 authored diagnostics the cited path
is faster and more accurate; see [measured Kaggle trials](docs/model-trials.md). These are small
authored cases, not ExtractBench scores. No checkpoint is selected.

Prepare the small diagnostic packet without installing a model runtime locally:

```sh
uv run python tools/build_diagnostics.py --output artifacts/diagnostics
uv run python tools/package_kaggle.py \
  --cases artifacts/diagnostics/inference/cases.json \
  --output artifacts/kaggle/docextract-trial.tar.gz
```

There are 24 authored PDFs/28 pages with separate inference inputs and scoring truth. The archive
contains only allowlisted source, PDFs and schemas. It excludes answers, manual binding plans,
credentials, caches and weights. These are diagnostic cases, not a general extraction benchmark.

**Run only on Kaggle**, from the unpacked source directory under `/kaggle/temp`:

```sh
uv sync --project runtimes/kaggle --frozen
uv run --project runtimes/kaggle --no-sync python tools/kaggle_trial.py \
  --cases inference/cases.json --output /kaggle/temp/trial-a \
  --model-manifest manifests/model-research.json --arm a --mode values \
  --max-input-tokens 2048 --max-new-tokens 256 --timeout-seconds 3600
```

Use `--arm b` in a fresh process for NuExtract3. `--mode bindings` tests source binding separately.
For one PDF on Kaggle:

```sh
uv run --project runtimes/kaggle --no-sync python -m docextract extract-model input.pdf \
  --schema schema.json --workspace /kaggle/temp/extraction \
  --arm a --model-manifest manifests/model-research.json
```

The backend guards the platform before model-library imports and checks revisions, file hashes,
FP16/SDPA, device placement, tied embeddings, memory and actual kernels. All weights/caches stay in
remote temporary storage. `--cache-only` disallows missing-cache downloads. The source-only tests
verify that local loading fails before importing Torch or Hugging Face libraries.

`tools/build_kaggle_kernel.py` prepares a private smoke script containing the small archive; it does
not submit a job. Its bootstrap compares full/text wrapper token hashes before a six-document A/B
smoke. `--experiment bindings` selects the three-case evidence trial. Both use a locked isolated
remote runtime, enforce a worker deadline and export only small results.
The six cases are development diagnostics; they cannot choose a general-quality winner.

The official adapter also registers `docextract_model_trial`, requiring Kaggle and `arm`,
`model_manifest`, `workspace` configuration. It reuses the model within one worker and inherits pure
saved-result normalization. It is not a frozen full-benchmark pipeline.

## Boundaries and next work

- [Supported schema profile and extraction limits](docs/schema-profile.md): fixed objects, typed
  homogeneous arrays, nullable branches, local nonrecursive refs and original-schema validation.
- [Architecture and recovery](docs/architecture.md): native text/geometry before normalization,
  source occurrence identity, explicit continuation and SQLite durable boundaries.
- [Next controlled model experiment](docs/experiments.md): compare candidate models on common bounded
  tasks, then measure record/evidence quality, memory, throughput and full-workload cost before a full run.
- [Remote model trial design](docs/kaggle-trial.md): all model downloads, caches and execution stay
  on Kaggle or Modal. Kaggle is for development; Modal L4 is for frozen final inference/full runs.
- [Verified source contracts](docs/sources.md) and [dependency/license boundary](docs/licenses.md).
- [Local validation evidence](docs/validation.md), including recovery, geometry and actual official
  scorer checks.

Scans, handwriting, checkbox interpretation, complex table mappings, distributed execution and
paid-compute safeguards are not implemented. The learned prototype has no automatic repair or
subdivision after a token limit, no proven-empty object-array protocol, and no independently verified
inventory completeness. Model discovery and association still need broader held-out trials.
The versioned docparse adapter is injectable and fails
closed until a public snapshot exporter is supplied; docparse itself is not installed or modified.
The working CPU evidence producer is PDFium. No GPU job or large model/data download is needed for B0.
