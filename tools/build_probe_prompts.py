"""Build authored engine-probe prompts: the 24 cited diagnostic requests plus synthetic length sweeps."""

import argparse
import json
from pathlib import Path

from docextract.interpretation import evidence_units, render_evidence
from docextract.parsing.pdfium import PdfiumParser
from docextract.remote import CITED_INSTRUCTIONS

SWEEP_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "quantity": {"type": "integer"},
                    "amount": {"type": "number"},
                },
            },
        }
    },
}


def messages(schema: dict, document: str) -> list[dict]:
    content = f"JSON Schema:\n{json.dumps(schema, ensure_ascii=False)}\n\nDocument:\n{document}"
    return [{"role": "system", "content": CITED_INSTRUCTIONS}, {"role": "user", "content": content}]


def build(diagnostics: Path, sweep_chars=(6000, 12000, 24000, 48000, 72000)) -> list[dict]:
    rows = []
    packet = json.loads((diagnostics / "inference/cases.json").read_text())
    for case in packet["cases"]:
        schema = json.loads((diagnostics / "inference" / case["schema"]).read_text())
        units = evidence_units(PdfiumParser().parse(diagnostics / "inference" / case["pdf"]))
        rows.append(
            {
                "id": case["id"],
                "kind": "diagnostic",
                "max_tokens": 2048,
                "messages": messages(schema, render_evidence(units)),
            }
        )
    for target in sweep_chars:
        lines, alias, row = ["[page 1]"], 0, 0
        while sum(len(line) + 1 for line in lines) < target:
            amount = (row * 37 % 10000) / 100
            lines.append(f"s{alias} INV-{row:05d}\ts{alias + 1} {row % 7 + 1}\ts{alias + 2} ${amount:.2f}")
            alias, row = alias + 3, row + 1
        rows.append(
            {
                "id": f"sweep-{target}",
                "kind": "sweep",
                "max_tokens": 128,
                "messages": messages(SWEEP_SCHEMA, "\n".join(lines)),
            }
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.diagnostics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print({"prompts": len(rows), "output": str(args.output)})
