from __future__ import annotations

import hashlib
import re
import threading
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from docextract.contracts import PageEvidence, ParseSnapshot, SourceAtom, digest
from docextract.geometry import Affine, bounds, envelope, page_transform, rectangle

PDFIUM_LOCK = threading.Lock()


CHECKED, UNCHECKED = "\u2612", "\u2610"
CHOICE_FIELDS = (
    pdfium_c.FPDF_FORMFIELD_TEXTFIELD,
    pdfium_c.FPDF_FORMFIELD_COMBOBOX,
    pdfium_c.FPDF_FORMFIELD_LISTBOX,
)
BOX_FIELDS = (pdfium_c.FPDF_FORMFIELD_CHECKBOX, pdfium_c.FPDF_FORMFIELD_RADIOBUTTON)


def form_widgets(form, page) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Filled form-field values with their native rectangles: text, choice and box fields.

    Interactive values live in widget annotations, which text extraction does not read. A checked
    box reads as a ballot box with an X and an unchecked one as an empty ballot box.
    """
    widgets = []
    for number in range(pdfium_c.FPDFPage_GetAnnotCount(page.raw)):
        annot = pdfium_c.FPDFPage_GetAnnot(page.raw, number)
        try:
            if pdfium_c.FPDFAnnot_GetSubtype(annot) != pdfium_c.FPDF_ANNOT_WIDGET:
                continue
            kind = pdfium_c.FPDFAnnot_GetFormFieldType(form, annot)
            if kind in BOX_FIELDS:
                text = CHECKED if pdfium_c.FPDFAnnot_IsChecked(form, annot) else UNCHECKED
            elif kind in CHOICE_FIELDS:
                length = pdfium_c.FPDFAnnot_GetFormFieldValue(form, annot, None, 0)
                if length <= 2:
                    continue
                buffer = (pdfium_c.FPDF_WCHAR * (length // 2))()
                pdfium_c.FPDFAnnot_GetFormFieldValue(form, annot, buffer, length)
                text = re.sub(r"\s+", " ", bytes(buffer)[: length - 2].decode("utf-16-le", "replace")).strip()
                if not text:
                    continue
            else:
                continue
            rect = pdfium_c.FS_RECTF()
            if not pdfium_c.FPDFAnnot_GetRect(annot, rect):
                continue
            left, right = sorted((rect.left, rect.right))
            bottom, top = sorted((rect.bottom, rect.top))
            if right > left and top > bottom:
                widgets.append((text, (left, bottom, right, top)))
        finally:
            pdfium_c.FPDFPage_CloseAnnot(annot)
    return widgets


class PdfiumParser:
    def __init__(self, *, cell_gap: float = 14, line_tolerance: float = 3, forms: bool = True):
        self.forms = forms
        self.identity = {
            "backend": "pypdfium2",
            "version": str(pdfium.PYPDFIUM_INFO),
            "pdfium_version": str(pdfium.PDFIUM_INFO),
            "adapter_version": "native/4",
            "coordinate_policy": "crop-rotation-display/1",
            "cell_gap": cell_gap,
            "line_tolerance": line_tolerance,
        }
        if forms:
            self.identity["form_fields"] = "widget-values-as-text/1"

    def parse(self, document: Path) -> ParseSnapshot:
        content = document.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        snapshot_id = digest({"document": source_hash, "parser": self.identity})
        with PDFIUM_LOCK, closing(pdfium.PdfDocument(content)) as pdf:
            if self.forms:
                pdf.init_forms()
            pages = tuple(self._page(pdf, index, snapshot_id) for index in range(len(pdf)))
        return ParseSnapshot(
            snapshot_id=snapshot_id, document_sha256=source_hash, parser_identity=self.identity, pages=pages
        )

    def _page(self, pdf, index: int, snapshot_id: str) -> PageEvidence:
        with closing(pdf[index]) as page, closing(page.get_textpage()) as textpage:
            crop = tuple(page.get_bbox())
            rotation = page.get_rotation()
            transform, width, height = page_transform(crop, rotation)
            prefix = f"{snapshot_id}/p{index:04d}"
            atoms: list[SourceAtom] = []
            visible: list[SourceAtom] = []
            layout_boxes: dict[str, tuple] = {}
            # PDFium marks word gaps in text without space glyphs by generated spaces (and line ends)
            # whose boxes are empty; a character after one starts a new word.
            word_starts: set[int] = set()
            for char_index in range(textpage.count_chars()):
                text = textpage.get_text_range(char_index, 1)
                if text.isspace():
                    word_starts.add(char_index + 1)
                box = tuple(textpage.get_charbox(char_index))
                native = rectangle(box)
                atom = SourceAtom(
                    atom_id=f"{prefix}/char{char_index:06d}",
                    page_index=index,
                    kind="character",
                    text=text,
                    polygon=transform.polygon(native),
                    native_polygon=native,
                    native_char_index=char_index,
                )
                atoms.append(atom)
                layout_boxes[atom.atom_id] = tuple(textpage.get_charbox(char_index, loose=True))
                left, bottom, right, top = box
                if text and text not in "\r\n" and right > left and top > bottom:
                    if left >= crop[0] and bottom >= crop[1] and right <= crop[2] and top <= crop[3]:
                        visible.append(atom)
            # A page needs OCR when it has no native text, whatever form values sit on top of it.
            flags = () if visible else ("no_native_text_ocr_required",)
            form = pdf.formenv.raw if self.forms and getattr(pdf, "formenv", None) else None
            drawn = re.sub(r"\s+", "", textpage.get_text_range(0, textpage.count_chars())).lower()
            for number, (text, box) in enumerate(form_widgets(form, page) if form is not None else ()):
                # A value already drawn into the page content is not repeated.
                if text not in (CHECKED, UNCHECKED) and re.sub(r"\s+", "", text).lower() in drawn:
                    continue
                left, bottom, right, top = box
                # Nominal left-aligned advances, so word gaps stay below the cell gap.
                step = min((right - left) / len(text), 0.6 * min(top - bottom, 14.0))
                for offset, character in enumerate(text):
                    glyph = (left + offset * step, bottom, left + (offset + 1) * step, top)
                    native = rectangle(glyph)
                    atom = SourceAtom(
                        atom_id=f"{prefix}/form{number:04d}_{offset:04d}",
                        page_index=index,
                        kind="character",
                        text=character,
                        polygon=transform.polygon(native),
                        native_polygon=native,
                        source_channel="form_field",
                    )
                    atoms.append(atom)
                    layout_boxes[atom.atom_id] = glyph
                    visible.append(atom)
            lines = self._group_lines(visible, layout_boxes)
            for line_index, characters in enumerate(lines):
                atoms.extend(
                    self._line_atoms(characters, prefix, line_index, transform, layout_boxes, word_starts)
                )
            return PageEvidence(
                page_index=index,
                width=width,
                height=height,
                crop_box=crop,
                rotation=rotation,
                native_to_display=transform.values(),
                native_text=textpage.get_text_range(0, textpage.count_chars()),
                atoms=tuple(atoms),
                quality_flags=flags,
            )

    def _group_lines(self, characters: list[SourceAtom], layout_boxes: dict) -> list[list[SourceAtom]]:
        groups: list[list[SourceAtom]] = []
        for atom in sorted(
            characters, key=lambda a: (-layout_boxes[a.atom_id][3], layout_boxes[a.atom_id][0])
        ):
            box = layout_boxes[atom.atom_id]
            center = (box[1] + box[3]) / 2
            group = next(
                (
                    group
                    for group in groups
                    if abs(center - sum(layout_boxes[group[0].atom_id][1::2]) / 2)
                    <= self.identity["line_tolerance"]
                ),
                None,
            )
            if group is None:
                groups.append([atom])
            else:
                group.append(atom)
        return [sorted(group, key=lambda atom: bounds(atom.native_polygon)[0]) for group in groups]

    def _line_atoms(
        self,
        chars: list[SourceAtom],
        prefix: str,
        index: int,
        transform: Affine,
        layout: dict,
        word_starts: frozenset[int] | set[int] = frozenset(),
    ):
        """Split words on spaces, on PDFium's generated word gaps, or on advance-box gaps; tight glyph
        boxes of narrow glyphs such as "1" or "." leave >3 pt gaps inside one word, while advance boxes
        touch."""
        tokens: list[SourceAtom] = []
        buffer: list[SourceAtom] = []
        for atom in chars:
            left, bottom, _, top = layout[atom.atom_id]
            gap = left - layout[buffer[-1].atom_id][2] if buffer else 0.0
            starts = atom.native_char_index in word_starts
            if atom.text.isspace() or (buffer and (starts or gap > max(1.0, 0.25 * (top - bottom)))):
                if buffer:
                    tokens.append(
                        self._aggregate(
                            buffer, f"{prefix}/token{index:04d}_{len(tokens):04d}", "token", transform, ""
                        )
                    )
                    buffer = []
            if not atom.text.isspace():
                buffer.append(atom)
        if buffer:
            tokens.append(
                self._aggregate(
                    buffer, f"{prefix}/token{index:04d}_{len(tokens):04d}", "token", transform, ""
                )
            )
        if not tokens:
            return []
        cells: list[SourceAtom] = []
        group: list[SourceAtom] = []
        for token in tokens:
            if (
                group
                and layout[token.child_ids[0]][0] - layout[group[-1].child_ids[-1]][2]
                > self.identity["cell_gap"]
            ):
                cells.append(
                    self._aggregate(
                        group, f"{prefix}/cell{index:04d}_{len(cells):04d}", "cell", transform, " "
                    )
                )
                group = []
            group.append(token)
        cells.append(
            self._aggregate(group, f"{prefix}/cell{index:04d}_{len(cells):04d}", "cell", transform, " ")
        )
        line = self._aggregate(tokens, f"{prefix}/line{index:04d}", "line", transform, " ")
        return [*tokens, *cells, line]

    @staticmethod
    def _aggregate(children, atom_id, kind, transform, separator):
        native = envelope([child.native_polygon for child in children])
        text = separator.join(child.text for child in children)
        return SourceAtom(
            atom_id=atom_id,
            page_index=children[0].page_index,
            kind=kind,
            text=re.sub(r"\s+", " ", text) if kind != "token" else text,
            native_polygon=native,
            polygon=transform.polygon(native),
            child_ids=tuple(child.atom_id for child in children),
        )
