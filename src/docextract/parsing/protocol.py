from pathlib import Path
from typing import Protocol

from docextract.contracts import ParseSnapshot


class ParserAdapter(Protocol):
    identity: dict

    def parse(self, document: Path) -> ParseSnapshot: ...
