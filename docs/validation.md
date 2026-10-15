# Validation evidence

Verified locally with CPython 3.12.11 and the locked dependency pins in `uv.lock`.
The repository changes are uncommitted. Private Kaggle diagnostics ran; no repository publication
or Modal work was performed. Model weights and large dataset files stayed remote.

- `uv run --extra benchmark python -m pytest -q`: **85 passed**. The official extra is pinned to
  ExtractBench `0e7647997211867060a645ac82deb9f7bc4a3b3f` and ParseBench 1.0.4.
- The earlier 47-test baseline also passed in an isolated production environment; its optional
  benchmark-free test run was 42 passed, 1 skipped. Those counts predate the learned-protocol tests.
- Ruff lint and format checks pass. Source distribution and wheel build with `uv build`.
- An isolated environment installed only production dependencies and extracted the authored PDF
  correctly with no ExtractBench or PyTorch installed.
- The CPU demo produces exactly two departments and four item rows: six anchored source records,
  including two identical-valued rows with distinct source identities. Seven tasks commit once.
- All expected typed data matches fixture truth. The original schema validates. There are 28 field
  evidence entries and 27 official citations; default currency and absent contact have no citations.
- The only unresolved item is the deliberately absent contact. Execution is complete, result status
  is partial and record/inventory completeness remains unverified. All 22 native lines are examined;
  unrelated headings/footers remain visible as unclaimed source regions.
- Original pages and both evidence overlays were rendered and visually inspected. No text/geometry
  clipping or evidence displacement was observed in this authored example.
- Crop and page rotations 0/90/180/270 are tested on real PDFs against PDFium `PageToDevice`, inverse
  transforms and dark raster pixels. Nonzero crop origins and crop/resize composition are covered.
- Recovery tests interrupt generation, raw-save, validation and commit, including an actual child
  process `os._exit` after raw-save. Saved generations are reused. Overlapping task windows and
  successful retries preserve occurrence counts.
- The real official scorer accepts authored scalar and duplicate-row citations at independently
  specified source coordinates. Removing one identical record lowers official recall. These are
  authored contract checks, not benchmark-quality measurements.
- Pure normalization is checked from a serialized official raw result with source file reads and
  extraction calls blocked. A contract-only injected docparse exporter fixture is tested with
  differently named source IDs; the real upstream exporter is still absent.
- The remote schema audit downloaded and hash-verified the three pinned JSONL splits only on Kaggle.
  It checked all 370 cases and found 70 distinct full-schema hashes; 66 meet the declared profile.
  Four schemas across 17 cases use an unsupported dialect. No benchmark PDFs were downloaded.
- Both pinned 4B models ran on a Kaggle T4. The six-case value-only development smoke returned
  3/6 exact documents for Qwen and 2/6 for NuExtract. See `model-trials.md` for failure accounting,
  runtime, memory and scope. The source/evidence trial is reported separately when completed.
- Scripted learned-protocol tests cover three-level nesting, multiple events in one source unit,
  local quote alignment, table mapping with identical rows, source-reference rejection, numeric
  precision rejection, and transactional recovery. They do not establish model accuracy.
- Training, model selection and full-benchmark inference/evaluation remain unperformed. No general
  quality or competitive score is established.

Reproduce the primary checks from the README. Generated PDFs, snapshots, journals, overlays and
evaluation artifacts stay under ignored `artifacts/`; model/data/cache/private paths are ignored.
