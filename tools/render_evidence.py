"""Draw saved display-coordinate evidence over actual PDFium rasters (dev dependencies)."""

import argparse
import hashlib
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
from PIL import ImageDraw

from docextract.contracts import ExtractionResult
from docextract.geometry import bounds


def render(pdf: Path, result: ExtractionResult, output: Path):
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != result.snapshot.document_sha256:
        raise ValueError("PDF bytes do not match the saved evidence snapshot")
    output.mkdir(parents=True, exist_ok=True)
    atoms = result.snapshot.atom_index()
    with closing(pdfium.PdfDocument(pdf)) as document:
        for page_index in range(len(document)):
            with closing(document[page_index]) as page, closing(page.render(scale=1.5)) as bitmap:
                image = bitmap.to_pil().convert("RGB")
                draw = ImageDraw.Draw(image)
                for field in result.field_evidence:
                    for atom_id in field.source_atom_ids:
                        atom = atoms[atom_id]
                        if atom.page_index == page_index:
                            draw.rectangle(
                                tuple(value * 1.5 for value in bounds(atom.polygon)),
                                outline=(220, 30, 50),
                                width=1,
                            )
                image.save(output / f"page-{page_index + 1}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.pdf, ExtractionResult.model_validate_json(args.result.read_text()), args.output)


if __name__ == "__main__":
    main()
