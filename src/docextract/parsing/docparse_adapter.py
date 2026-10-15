"""No implicit docparse install, private imports, model load or licensing workaround."""

import hashlib
from collections.abc import Callable
from pathlib import Path

from docextract.contracts import ParseSnapshot

DOCPARSE_COMMIT = "9451b48e0db028053bdf150be79fdd2b69eed04f"


class DocparseAdapter:
    """Inject a public snapshot exporter from a separately licensed parser runtime.

    The pinned docparse does not yet expose this export. Fail closed until a caller
    supplies it; do not fabricate fine-grained evidence from Markdown.
    """

    identity = {"backend": "docparse", "commit": DOCPARSE_COMMIT, "adapter_version": "docparse/1"}

    def __init__(self, exporter: Callable[[Path], ParseSnapshot] | None = None):
        self.exporter = exporter

    def parse(self, document: Path) -> ParseSnapshot:
        if self.exporter is None:
            raise RuntimeError("docparse requires a parse-snapshot/1 exporter; use PdfiumParser for CPU B0")
        snapshot = ParseSnapshot.model_validate(self.exporter(document))
        if snapshot.parser_identity.get("commit") != DOCPARSE_COMMIT:
            raise ValueError("docparse exporter commit does not match the pinned adapter")
        if snapshot.parser_identity.get("adapter_version") != "docparse/1":
            raise ValueError("Unsupported docparse exporter version")
        if snapshot.document_sha256 != hashlib.sha256(document.read_bytes()).hexdigest():
            raise ValueError("docparse export belongs to different document bytes")
        return snapshot
