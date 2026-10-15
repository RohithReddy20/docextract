"""Count interactive form-field values per benchmark document on a remote worker; counts only.

A filled PDF form can keep its values in widget annotations instead of page text, where text
extraction does not see them. For every widget this records its field type, whether it holds a
value (or is checked), and whether that value also appears in the page's extracted text.
"""

import argparse
import json
import re
from collections import Counter
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from docextract.contracts import digest, save_json
from docextract.parsing.pdfium import PDFIUM_LOCK

FIELD_TYPES = {
    pdfium_c.FPDF_FORMFIELD_TEXTFIELD: "text",
    pdfium_c.FPDF_FORMFIELD_CHECKBOX: "checkbox",
    pdfium_c.FPDF_FORMFIELD_RADIOBUTTON: "radio",
    pdfium_c.FPDF_FORMFIELD_COMBOBOX: "combobox",
    pdfium_c.FPDF_FORMFIELD_LISTBOX: "listbox",
    pdfium_c.FPDF_FORMFIELD_PUSHBUTTON: "button",
    pdfium_c.FPDF_FORMFIELD_SIGNATURE: "signature",
}


def field_value(form, annot) -> str:
    length = pdfium_c.FPDFAnnot_GetFormFieldValue(form, annot, None, 0)
    if length <= 2:
        return ""
    buffer = (pdfium_c.FPDF_WCHAR * (length // 2))()
    pdfium_c.FPDFAnnot_GetFormFieldValue(form, annot, buffer, length)
    return bytes(buffer)[: length - 2].decode("utf-16-le", errors="replace")


def squeeze(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def document_forms(content: bytes) -> dict:
    counts = Counter()
    with PDFIUM_LOCK, closing(pdfium.PdfDocument(content)) as pdf:
        pdf.init_forms()
        form = pdf.formenv.raw if pdf.formenv else None
        for index in range(len(pdf)):
            with closing(pdf[index]) as page:
                with closing(page.get_textpage()) as text_page:
                    page_text = squeeze(text_page.get_text_bounded())
                for number in range(pdfium_c.FPDFPage_GetAnnotCount(page.raw)):
                    annot = pdfium_c.FPDFPage_GetAnnot(page.raw, number)
                    try:
                        if pdfium_c.FPDFAnnot_GetSubtype(annot) != pdfium_c.FPDF_ANNOT_WIDGET or form is None:
                            continue
                        kind = FIELD_TYPES.get(pdfium_c.FPDFAnnot_GetFormFieldType(form, annot), "other")
                        counts[f"widgets_{kind}"] += 1
                        if kind in ("checkbox", "radio"):
                            counts[f"checked_{kind}"] += bool(pdfium_c.FPDFAnnot_IsChecked(form, annot))
                            continue
                        value = field_value(form, annot).strip()
                        if value:
                            counts[f"valued_{kind}"] += 1
                            counts[f"valued_{kind}_missing_from_text"] += squeeze(value) not in page_text
                    finally:
                        pdfium_c.FPDFPage_CloseAnnot(annot)
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, totals = [], {}
    for split in ("short", "medium", "long"):
        summary = Counter()
        for line in (args.snapshot / f"{split}.jsonl").open("rb"):
            source = json.loads(line)
            case = {"case": digest(source.get("id") or source["pdf"])[:16], "split": split}
            try:
                pdf = (args.snapshot / source["pdf"]).resolve()
                if not pdf.is_relative_to(args.snapshot.resolve()):
                    raise ValueError("Document path escapes the snapshot")
                case |= document_forms(pdf.read_bytes())
            except Exception as error:  # every case stays in the population
                case["error"] = type(error).__name__
            rows.append(case)
            summary["documents"] += 1
            summary["documents_with_widgets"] += any(key.startswith("widgets_") for key in case)
            missing = sum(value for key, value in case.items() if key.endswith("_missing_from_text"))
            checked = sum(value for key, value in case.items() if key.startswith("checked_"))
            summary["documents_with_values_missing_from_text"] += missing > 0
            summary["documents_with_checked_boxes"] += checked > 0
            summary["values_missing_from_text"] += missing
            summary["checked_boxes"] += checked
            summary["errors"] += "error" in case
        totals[split] = dict(summary)
    save_json(args.output, {"content_exported": False, "splits": totals, "cases": rows})
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
