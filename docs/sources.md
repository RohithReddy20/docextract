# Source checks for the CPU milestone

The supplied research report is design input, not repository content. These important implementation
recommendations were checked directly against the requested snapshots or installed pinned packages.

| Recommendation | Verified source and consequence |
| --- | --- |
| Separate inference and normalization | [Official Provider](https://github.com/run-llama/ExtractBench/blob/0e7647997211867060a645ac82deb9f7bc4a3b3f/src/extract_bench/inference/providers/base.py): implement the real abstract interface; normalize saved results only. |
| Shared official output classes | [ExtractOutput re-export](https://github.com/run-llama/ExtractBench/blob/0e7647997211867060a645ac82deb9f7bc4a3b3f/src/extract_bench/schemas/extract_output.py), inspected installed parse-bench 1.0.4 FieldCitation: bracket paths, one-based pages, normalized COCO boxes. |
| Preserve duplicates, defaults and nested associations | [Unified metric](https://github.com/run-llama/ExtractBench/blob/0e7647997211867060a645ac82deb9f7bc4a3b3f/src/extract_bench/evaluation/metrics/extract/unified_evidence_metric.py): integration tests execute the actual pinned scorer. |
| Pin dataset outside default downloader | [Downloader](https://github.com/run-llama/ExtractBench/blob/0e7647997211867060a645ac82deb9f7bc4a3b3f/src/extract_bench/data/download.py): arbitrary snapshot revision is not exposed by its public download API. Preparation passes the exact dataset SHA and records hashes. |
| Native evidence before Markdown | [docparse textlayer](https://github.com/RohithReddy20/docparse/blob/9451b48e0db028053bdf150be79fdd2b69eed04f/docparse/textlayer.py): native character geometry exists, but the public snapshot export is absent; the 80-character text threshold should not discard sparse extraction evidence. |
| Avoid unreviewed parser dependency inheritance | [docparse metadata](https://github.com/RohithReddy20/docparse/blob/9451b48e0db028053bdf150be79fdd2b69eed04f/pyproject.toml) declares PyMuPDF and no license; the pinned Git tree has no LICENSE. Do not vendor or install it for the commercial CPU core. |
| Explicit coordinate conversion and serialized calls | [PDFium text API](https://pypdfium2.readthedocs.io/en/v4/python_api.html#pypdfium2.PdfTextPage.get_charbox): native boxes use left/bottom/right/top; PDFium calls cannot run concurrently in threads. The adapter holds a process-wide mutex. |
| Defaults are not validation-time insertion | [JSON Schema annotations](https://json-schema.org/understanding-json-schema/reference/annotations): preserve a separate default status and require an explicit application policy. |

The pinned Hugging Face [dataset tree](https://huggingface.co/datasets/llamaindex/ExtractBench/tree/51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc)
was checked via its metadata API. Only the three JSONL file sizes and LFS SHA-256 hashes were fetched.
The full corpus and complete unique-schema inventory have not been downloaded or executed locally;
the remote tools are included for that gate. No live leaderboard or model quality claims are made.
