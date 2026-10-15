# Commercial dependency boundary

New docextract code and the authored fixture generator are Apache-2.0 under `LICENSE`. No docparse
code, external model weights, proprietary prompts, benchmark answers or dataset documents are vendored.

The locked production runtime is JSON Schema validation (jsonschema, MIT), typed contracts (pydantic,
MIT) and PDFium evidence extraction (pypdfium2 4.30.0). Its installed wheel declares
`(Apache-2.0 OR BSD-3-Clause) AND LicenseRef-PdfiumThirdParty`. The complete bundled PDFium/component
notice from that wheel is preserved in `notices/PDFium-4.30.0-third-party.txt`. Preserve the appropriate
wheel/runtime notices when redistributing a binary build; a wrapper's license does not replace them.

The authored PDF uses built-in Helvetica, no external font files. ReportLab and pypdf are development
dependencies with BSD licenses; generated PDFs are test content owned by this project. No OCR model,
GPU framework or PyMuPDF is part of the core dependency graph.

The pinned docparse metadata includes PyMuPDF, and its pinned root has no explicit license.
`DocparseAdapter` specifies a boundary and validates an injected export, but does not grant rights to
that runtime. Before integrating or distributing docparse, establish the code license and audit its
included code, weights, PDF backend and export parity. A process boundary alone does not settle those
obligations. This is explicitly pending, not a commercially cleared parser integration.

ExtractBench and ParseBench project code is Apache-2.0. Their optional evaluation dependency graph
includes `Levenshtein` / `python-Levenshtein` 0.27.5, whose installed metadata declares GPL-2.0-or-later.
The benchmark extra is optional and never imported by production modules. Keep the evaluator as a
separate tool/environment and review its distribution obligations before bundling it into a product.
Commercial use and permissive redistribution are different questions.

No model has been selected or installed. Verify pinned checkpoint licenses, quantization/runtime
dependencies, source datasets and required notices before adding a model backend or training data.
