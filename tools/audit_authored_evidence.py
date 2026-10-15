"""Independent placement audit for the three frozen development evidence cases, not benchmark F1."""

import argparse
import hashlib
from pathlib import Path

from reportlab.pdfbase.pdfmetrics import getAscentDescent, stringWidth

from docextract.contracts import ExtractionResult, save_json
from docextract.geometry import bounds
from docextract.jsonutil import strict_json


def placements(family):
    """Authored Helvetica-11 positions, independent of the PDF extraction backend."""
    if family == "sparse":
        return {
            "/invoice_id": [(0, 48 + stringWidth("Invoice ID: ", "Helvetica", 11), 698, "0017")],
            "/amount_due": [(0, 48 + stringWidth("Amount due: ", "Helvetica", 11), 608, "25.50")],
        }
    if family == "tables":
        return {
            f"/transactions/{i}/{key}": [(0, x, 670 - 28 * i, value)]
            for i, row in enumerate(
                (("A0017", "2", "12.50"), ("A0017", "2", "12.50"), ("B0017", "1", "5.00"))
            )
            for key, x, value in zip(("code", "quantity", "amount"), (48, 226, 448), row, strict=True)
        }
    if family == "continuation":
        return {
            "/items/0/id": [(0, 48, 670, "0017")],
            "/items/0/description": [(0, 226, 670, "Extended support"), (1, 226, 668, "through December")],
            "/items/0/quantity": [(0, 448, 670, "1")],
            "/items/1/id": [(1, 48, 612, "B0017")],
            "/items/1/description": [(1, 226, 612, "Travel pack")],
            "/items/1/quantity": [(1, 448, 612, "3")],
        }
    raise ValueError("No independent placement oracle for this case")


def audit(result: ExtractionResult, case):
    expected_id = hashlib.sha256(f"authored-diagnostics/1/{case['family']}/0".encode()).hexdigest()[:16]
    if case["id"] != expected_id or result.snapshot.document_sha256 != case["pdf_sha256"]:
        raise ValueError("Placement oracle is limited to the frozen variant-zero authored PDF")
    expected = placements(case["family"])
    atoms = result.snapshot.atom_index()
    ascent, descent = getAscentDescent("Helvetica", 11)
    findings = []
    for pointer, fragments in expected.items():
        fields = [field for field in result.field_evidence if field.pointer == pointer]
        valid = len(fields) == 1 and fields[0].status == "observed"
        selected = [atoms[identity] for identity in fields[0].source_atom_ids] if fields else []
        used = set()
        for page, x, y, text in fragments:
            box = (
                x - 1,
                792 - y - ascent - 1,
                x + stringWidth(text, "Helvetica", 11) + 1,
                792 - y - descent + 1,
            )
            local = []
            for atom in selected:
                left, top, right, bottom = bounds(atom.polygon)
                if (
                    atom.page_index == page
                    and box[0] <= left <= right <= box[2]
                    and box[1] <= top <= bottom <= box[3]
                ):
                    local.append(atom)
                    used.add(atom.atom_id)
            local.sort(key=lambda atom: atom.native_char_index if atom.native_char_index is not None else -1)
            source_text = "".join(atom.text for atom in local)
            valid &= "".join(source_text.split()) == "".join(text.split())
            valid &= all(atom.kind == "character" for atom in local)
        valid &= len(used) == len(selected) and len(selected) > 0
        findings.append({"pointer": pointer, "correct_authored_source_occurrences": bool(valid)})
    return {
        "case": case["id"],
        "expected_fields": len(expected),
        "correctly_grounded_fields": sum(row["correct_authored_source_occurrences"] for row in findings),
        "unexpected_cited_fields": sum(
            field.pointer not in expected and bool(field.source_atom_ids) for field in result.field_evidence
        ),
        "fields": findings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for case in strict_json(args.answers.read_text())["cases"]:
        expected_id = hashlib.sha256(f"authored-diagnostics/1/{case['family']}/0".encode()).hexdigest()[:16]
        if case["family"] not in ("sparse", "tables", "continuation") or case["id"] != expected_id:
            continue
        path = args.results / case["id"] / "result.json"
        if path.exists():
            reports.append(audit(ExtractionResult.model_validate_json(path.read_text()), case))
        else:
            reports.append(
                {
                    "case": case["id"],
                    "expected_fields": len(placements(case["family"])),
                    "correctly_grounded_fields": 0,
                    "error": "missing_result",
                }
            )
    save_json(
        args.output,
        {
            "metric": "authored source placement and complete character support; not ExtractBench F1",
            "cases": reports,
        },
    )


if __name__ == "__main__":
    main()
