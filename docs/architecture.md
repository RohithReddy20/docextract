# Extraction architecture

`extract(PDF, original_schema, workspace, bindings)` validates schema support before reading a PDF
or asking a backend to run. Production modules do not import ExtractBench types or evaluators.

1. `PdfiumParser` takes a byte hash, parser/PDFium versions, grouping parameters, and coordinate policy
   into a `parse-snapshot/1` identity. Every page, native character occurrence, token, derived cell,
   and line receives a snapshot-relative ID. Cells are gap-based hypotheses, not detected table cells.
2. `Schema` preserves the original JSON Schema and resolves supported local definitions for planning.
   Validation runs on the original schema. Constraints never disappear through model projection.
3. `Planner` scans all native lines. Declared labels bind root/nested scalar fields. A declared section
   creates an anchored parent record; a table header maps columns within that parent. Unbound object
   arrays, competing labels, and missing headers are reported. This is limited structural discovery.
4. `ModelBackend.generate(WorkUnit)` is a swappable copying interface after source binding. B0's
   `DeterministicBackend` performs strict typed conversion. It has no learned model, semantic field
   selection or OCR. Automatic extraction requires a separate learned discovery/association interface
   before these candidates are bound; replacing this backend alone cannot remove manual mappings.
   See the source-verified update in `docs/experiments.md`.
5. A SQLite journal saves raw output before validating it. It atomically commits the task's fields,
   evidence and unresolved items in one output payload. Assembly is a deterministic rebuild from
   committed outputs, never an append to the last user array.
6. `assemble` walks parent records in source order, creates arrays, then assigns output indices and
   evidence JSON Pointers. Identity uses source occurrences, never record values. Declared empty
   tables materialize `[]`; an undiscovered array is not assumed empty.
7. Original-schema validation, source coverage, unresolved items, usage and execution completion
   are saved alongside values and evidence. Normalization only projects that saved object.

## Learned interpretation prototype

`extract_learned(PDF, schema, workspace, backend)` adds schema-conditioned proposals before the
copying interface. It needs no binding plan. The backend receives every bounded native evidence
window and compact recent-record context. Proposals introduce anchored records, parent relations,
field spans, regular-table header mappings and explicit continuations. A mapping materializes
every physical row. Unique quotes within a selected cell can resolve offsets deterministically;
ambiguous local occurrences require explicit offsets. Quotes, offsets, aliases, schema scopes and
parent types are checked for each proposal item; an invalid item (and anything that depends on it)
is rejected and reported while the valid items commit. One complete Markdown fence and `null` for an
optional protocol field are accepted; truncated output is never repaired. Source occurrences, never
equal values, identify records. Final typed copying and original-schema validation remain
deterministic.

The interpretation journal supports the same durable recovery boundaries. Overlapping source spans
reuse occurrence identities; different source rows survive even when all values match. Implementation
and backend-source hashes participate in cache identity. Committed results rebuild the graph after
interruption. Three-level nested arrays and multipage append citations have scripted contract tests.

The Kaggle backend is optional and guarded before heavy imports. It uses pinned safetensors,
explicit single-device FP16/SDPA, no offload and no implicit custom model code. Production imports
require no model or benchmark package. `trial.py` separately preserves raw value-only responses;
these have no citation claim. See `kaggle-trial.md` for limits and remote validation status.

## Cited-value protocol

`extract_cited(PDF, schema, workspace, backend)` asks the model for the user's own JSON shape, with
`{"value": ..., "source": "<cell id>"}` at every scalar. The prompt shows compact native evidence: one
row per line, each cell prefixed by its id, cells separated by tabs. Code then:

1. types each value against the original schema (numbers, integers, booleans, enums; no locale or
   date guessing);
2. locates the value inside the cited cells (verbatim, case/space-normalized, numeric or ISO-date
   equivalence) and stores the exact characters; a value that cannot be found there keeps cell-level
   support and is marked `model_value_unverified`;
3. searches the window for uncited values, giving sibling records distinct occurrences, and marks
   values found nowhere as `model_value_unsupported`;
4. identifies records by the source characters they cite, so identical printed rows stay separate
   while the same row seen in two overlapping windows merges;
5. projects absent nullable fields as explicit `null` (status `absent_after_search`), omits absent
   non-nullable fields, and validates the original schema.

Malformed items are rejected individually and reported. Unfinished, oversized, out-of-memory or
unparseable windows are split in half until they finish; a single line that still fails is reported
as uninterpreted. A string value that is only a cell reference (e.g. `s4` for a field named `id`) is
rejected. The source-proposal protocol (`extract_learned`) remains for comparison and uses the same
per-item rejection and subdivision rules.

Additional mechanisms, each independently switchable and measured:

- **Field groups.** A schema larger than `max_schema_chars` is inlined and split by root (and, if
  needed, nested-object) properties into self-contained groups; arrays and leaves are never split,
  so a record's fields are always requested together. Every group runs over every evidence window;
  answers merge by JSON Pointer and the original schema is validated once.
- **Constrained decoding (vLLM).** `cited_output_schema` turns the user schema into the JSON Schema of
  a cited answer (value/source leaves, the user's types, enums and consts, optional properties); vLLM
  then cannot emit invalid JSON, invented keys or out-of-enum values.
- **Table-row completion.** If at least two records cite cells of single rows of a regular native
  table with a consistent field-to-column mapping, rows the model skipped are copied with the same
  mapping (numbers parsed, ISO dates only when the examples were normalized dates; booleans never
  guessed). Copies carry their cell evidence and transform `copied_table_row`.
- **Streaming, batching and replay.** Windows are independent, so `run_batch` queues every window
  of every document in a continuously batching engine (vLLM), documents in order and windows that
  share a schema field group (a shared prompt prefix) next to each other. Each answer is stored the
  moment it finishes, and a document is assembled as soon as its last window is answered. Answers are
  stored by a generation identity that covers only prompt inputs; snapshots (including OCR) are
  stored by document hash. A replay re-assembles stored answers under other deterministic options
  without generating again; a window with no stored answer (a run stopped at its deadline) is
  reported as unanswered instead of being split, so partly answered documents still assemble.
- **Plain values (optional protocol).** `values` asks for the user's JSON with plain values and
  shows the evidence without cell references, which roughly halves answer length. Code then locates
  each value in the window, giving sibling records distinct occurrences; a value printed only once in
  the window identifies its record like a citation does, so a row seen twice in overlapping windows is
  merged. Values found nowhere are kept and reported as unsupported.
- **Page images (optional).** With `page_images`, each recognized (scanned) page is also saved as an
  image (longest side 1,280 px) and the windows that cover it carry it to a model that reads images;
  windows then span at most `max_window_pages` pages. The image follows the schema and precedes the
  OCR text lines; the model reads check marks, handwriting and garbled text from it and still cites
  the nearest OCR cell. The answer's identity includes the image bytes.
- **Routing (optional).** With `route_groups`, a window is asked only about the schema field groups
  whose distinctive words (from property names, titles, descriptions, enum values) its text contains;
  every group keeps at least its best-matching window. On a 29-page scanned tax return with a
  155K-token schema this cut requests from 132 to 37 while every value printed in the document stayed
  reachable from a window that is asked about its field (319 of 320, checked offline).
- **Prompt budget and notes (optional).** `prompt_budget_chars` sizes windows to the room the largest
  schema group leaves. `factor_notes` writes a description sentence repeated by many fields once, as a
  numbered note in the root description, and cites it as `[N1]`: 22% shorter for the tax-return
  schema, 36% for one form family; applied only when it saves at least 10%.
- **Entity merging (optional).** With `merge_entities`, a record from a later window joins an earlier
  record of the same array when their field sets differ (a partial view, or complementary halves),
  they share a non-boolean value and their shared fields agree (case, spacing and punctuation ignored,
  near-identical text allowed, one disagreement tolerated when four or more agree). A summary table and
  a later detail section then give one record per entity; identical printed rows never merge.
- **Prompt schema.** The model reads the schema with redundant text removed: titles that only
  restate the property name, `default: null`, and the `anyOf` spelling of a nullable plain type
  (written as a type list). Titles that differ from the name, such as printed form labels, stay.
  Validation always uses the original schema.

## Coordinates and source text

Native PDF coordinates have origin at the bottom left. `page_transform` composes crop-origin
translation, page rotation and the top-left display frame. Each page saves this affine transform
and display dimensions; each atom saves native and display polygons. `crop_resize` and inverse
transforms support raster/model-crop coordinates without losing the native reference.

Tight PDFium character boxes provide evidence. Loose font boxes group punctuation with the right
text line. Token/line/cell text is a derived projection; original page text, native character indices,
character text, and polygons are saved first. Sparse text is retained without a character threshold.
There is no visibility/occlusion audit: native text layers can themselves be wrong or hidden.

The benchmark projection emits one normalized top-left COCO box per supporting line/page,
one-based page numbers, and bracketed zero-based array indices. It cites local characters from the
bound occurrence, rather than searching globally for the generated value. Multiple-page values
retain both pages. Defaults receive no source citations.

## Pages without a text layer

`OcrFallbackParser` keeps PDFium text for every page that has it and sends to an injected recognizer
the pages flagged `no_native_text_ocr_required`, plus scanned pages (at least half covered by images)
that carry under 300 real characters, such as a stamp or a header line over a scan: 11 short
ExtractBench documents have such pages. On those, OCR words over real text are dropped and native and
OCR lines share one reading order (`sparse_native_text`). OCR words become token atoms with display and
native polygons (`source_channel="ocr"`, word confidence in quality flags); they are grouped into
cells and lines like native text. No character boxes are invented: a citation on an OCR page is
word-level. The docTR recognizer (Apache-2.0) loads weights and therefore runs only on remote workers.
A page that reads poorly is also read turned by 90, 180 and 270 degrees, and the most legible
reading (confidence mass of plausible words) is kept; its word boxes are mapped back to the page as
rendered, reading order comes from the upright frame, and the page is flagged `ocr_rotated:<deg>`.
Handwriting is not yet handled.

Filled interactive form fields keep their values in widget annotations, which text extraction does
not read. Their values (text and choice fields; checked/unchecked boxes as ☒/☐) are added as
characters at the widget position (`source_channel="form_field"`) unless the same text is already
drawn on the page. In ExtractBench only one of 370 documents has widgets and none hold values
missing from the page text, so this matters for real filled forms rather than for the benchmark.

Words are split on spaces, on PDFium's generated word gaps (typeset text often has no space glyphs;
PDFium inserts spaces with empty boxes, which mark word starts), or on gaps between PDFium's advance
("loose") glyph boxes. Tight glyph
boxes of narrow glyphs such as `1` and `.` leave gaps over 3 pt inside one word, which previously
broke numbers like `11.50` into `1 1 .50`.

## Recovery and bounded processing

Task states are `pending -> running -> raw_saved -> validated -> committed`. Idempotency includes
document/parser snapshot, original schema, binding plan, defaults policy, backend identity and engine
version. A backend identity must include its model, prompt and generation configuration revisions.
Changing any of these creates a new run. Processing-window overlap does not change a task identity.

The worker holds a nonblocking OS file lock for its workspace, so another live worker cannot reclaim
running work. After process death the OS releases the lock. `raw_saved` and `validated` tasks are
revalidated without regeneration; committed tasks are reused. A crash before raw-save can require
another generation. SQLite uses WAL and full synchronization. One worker per workspace is supported
on macOS/Linux, not distributed leases or Windows.

An unfinished generation is saved but cannot commit, even if its text is valid JSON. B0 does not
implement automatic model repair, subdivision, cancellation of remote generation, or a billing ledger.
Those are gates before any paid backend is added. Raw output with invalid bindings also remains saved
for inspection; use a revised backend identity for a corrected attempt. Snapshot parsing/planning is
repeated on resume; model tasks are the durable boundary in B0.

Explicit continuation requires the configured marker after a source row, the next page, matching
header names and positions, the same anchored parent, and only permitted continuation columns.
Unfinished or incompatible fragments stay unresolved. No inference is made from identical row values
or repeated headers alone. Arbitrary wrapped rows, changed units/headers, pivot tables, and implicit
cross-page relationships require later discovery/interpretation work.

## Completion semantics

`execution_complete` means all planned tasks committed. Schema validity is independent. Coverage
counts native lines examined and exposes unclaimed line IDs. It does not count image-only regions,
measure record inventory recall, or assert that every relevant record was discovered. Both inventory
and record completeness remain `not_independently_verified`, even when status is `complete`.
The authored test generator supplies an external expected record set for that fixture only.

## Parser boundary

`DocparseAdapter` is a versioned, injected-exporter boundary pinned to the requested commit.
That commit has no public `parse-snapshot/1` export. Therefore the adapter deliberately refuses to run
until one is supplied by a separately managed docparse runtime. It checks export contract, commit,
adapter version and source hash. No docparse code, weights or dependency stack is vendored or installed.
Adding and verifying the upstream export remains work; CPU B0 uses the working PDFium implementation.
