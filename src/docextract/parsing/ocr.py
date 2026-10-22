"""Native PDF text first; pages without a text layer get OCR word evidence.

OCR words become token atoms (no invented character boxes), grouped into cells and lines like native
text, with display and native polygons. The recognizer is injected; recognizers that load models run
only on remote workers.
"""

from __future__ import annotations

import hashlib
import re
from contextlib import closing
from pathlib import Path
from typing import Protocol

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from docextract.contracts import PageEvidence, ParseSnapshot, SourceAtom, digest
from docextract.geometry import Affine, bounds, envelope
from docextract.parsing.pdfium import PDFIUM_LOCK, PdfiumParser


class Recognizer(Protocol):
    identity: dict

    def __call__(self, images: list) -> list[list[dict]]:
        """Words per page image: text, relative top-left box (x0, y0, x1, y1), line key, confidence.

        A word may also carry `order_box`, its box in the upright frame it was read in, and
        `rotation`, the counterclockwise degrees the page image was turned to read it upright.
        """


def unrotate(box, angle: int):
    """Map a relative box read from an image turned counterclockwise by `angle` back to the page."""
    x0, y0, x1, y1 = box
    if angle == 90:
        return (1 - y1, x0, 1 - y0, x1)
    if angle == 180:
        return (1 - x1, 1 - y1, 1 - x0, 1 - y0)
    if angle == 270:
        return (y0, 1 - x1, y1, 1 - x0)
    return box


def legibility(words: list[dict]) -> float:
    """Confidence mass of plausible words; text read sideways scores low."""
    return sum(
        word["confidence"]
        for word in words
        if word["confidence"] >= 0.5 and sum(char.isalnum() for char in word["text"]) >= 2
    )


def read_upright(read, images: list, threshold=0.6) -> list[list[dict]]:
    """Read pages as given; a page that reads poorly is also read turned 90, 180 and 270 degrees
    and the most legible reading is kept, with boxes mapped back to the page as rendered."""
    pages = []
    for image, words in zip(images, read(images), strict=True):
        best, angle = words, 0
        if legibility(words) < threshold * max(1, len(words)):
            for turn in (90, 180, 270):
                candidate = read([image.rotate(turn, expand=True)])[0]
                if legibility(candidate) > legibility(best):
                    best, angle = candidate, turn
        pages.append(
            [
                {**word, "box": unrotate(word["box"], angle), "order_box": word["box"], "rotation": angle}
                for word in best
            ]
        )
    return pages


def page_image_path(store: Path, snapshot_id: str, page_index: int) -> Path:
    return store / snapshot_id / f"p{page_index:04d}.jpg"


def image_coverage(page) -> float:
    """Share of the page area covered by image objects (overlaps counted twice, capped at 1)."""
    width, height = page.get_size()
    area = 0.0
    for item in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
        left, bottom, right, top = item.get_pos()
        area += max(0.0, right - left) * max(0.0, top - bottom)
    return min(1.0, area / (width * height)) if width and height else 0.0


class OcrFallbackParser:
    def __init__(
        self,
        recognizer: Recognizer,
        *,
        native: PdfiumParser | None = None,
        dpi=200,
        cell_gap=14.0,
        sparse_chars=300,
        image_store: Path | None = None,
        image_side=1280,
    ):
        self.native = native or PdfiumParser()
        self.recognizer = recognizer
        self.dpi = dpi
        self.cell_gap = cell_gap
        self.sparse_chars = sparse_chars
        # Recognized pages are also kept as images (longest side `image_side`) for a model that reads them.
        self.image_store, self.image_side = image_store, image_side
        self.identity = {
            **self.native.identity,
            "ocr": recognizer.identity,
            "ocr_dpi": dpi,
            # Also scanned pages that carry only a little real text (a stamp, a header line).
            "ocr_policy": f"no-native-text-or-image-page-under-{sparse_chars}-chars/2+scan-flag/1",
            "ocr_cell_gap": cell_gap,
        }

    def _visible(self, page: PageEvidence) -> int:
        return sum(
            atom.kind == "character" and bool(atom.text.strip()) and atom.source_channel == "native_pdf"
            for atom in page.atoms
        )

    def parse(self, document: Path) -> ParseSnapshot:
        content = document.read_bytes()
        source_hash = hashlib.sha256(content).hexdigest()
        snapshot_id = digest({"document": source_hash, "parser": self.identity})
        pages, images, scans = [], {}, {}
        with PDFIUM_LOCK, closing(pdfium.PdfDocument(content)) as pdf:
            if self.native.forms:
                pdf.init_forms()
            for index in range(len(pdf)):
                page = self.native._page(pdf, index, snapshot_id)
                with closing(pdf[index]) as source:
                    textless = "no_native_text_ocr_required" in page.quality_flags
                    scanned = textless or image_coverage(source) >= 0.5
                    if textless or (scanned and self._visible(page) < self.sparse_chars):
                        images[index] = source.render(scale=self.dpi / 72).to_pil()
                    elif scanned:
                        # A scan with its own text layer keeps that text; its image is still kept for a
                        # model that reads images (the text layer can be poor).
                        page = page.model_copy(
                            update={"quality_flags": (*page.quality_flags, "scanned_text_layer")}
                        )
                        if self.image_store is not None:
                            scans[index] = source.render(scale=self.dpi / 72).to_pil()
                pages.append(page)
        if images:
            recognized = self.recognizer(list(images.values()))
            for index, words in zip(images, recognized, strict=True):
                pages[index] = self._ocr_page(pages[index], words, snapshot_id)
        if self.image_store is not None:
            for index, image in {**images, **scans}.items():
                path = page_image_path(self.image_store, snapshot_id, index)
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    copy = image.convert("RGB")
                    copy.thumbnail((self.image_side, self.image_side))
                    copy.save(path.with_suffix(".tmp"), format="JPEG", quality=90)
                    path.with_suffix(".tmp").replace(path)
        return ParseSnapshot(
            snapshot_id=snapshot_id,
            document_sha256=source_hash,
            parser_identity=self.identity,
            pages=tuple(pages),
        )

    def _ocr_page(self, page: PageEvidence, words: list[dict], snapshot_id: str) -> PageEvidence:
        to_native = Affine(*page.native_to_display).inverse()
        prefix = f"{snapshot_id}/p{page.page_index:04d}"
        lines: dict[tuple, list[SourceAtom]] = {}
        order = {}
        # On a page with some real text, OCR words over that text are dropped as duplicates.
        native = [bounds(atom.polygon) for atom in page.atoms if atom.kind == "token"]
        for number, word in enumerate(
            sorted(words, key=lambda w: (w["line"], w.get("order_box", w["box"])[0]))
        ):
            text = re.sub(r"\s+", " ", word["text"]).strip()
            if not text:
                continue
            x0, y0, x1, y1 = word["box"]
            center = ((x0 + x1) / 2 * page.width, (y0 + y1) / 2 * page.height)
            if any(b[0] - 1 <= center[0] <= b[2] + 1 and b[1] - 1 <= center[1] <= b[3] + 1 for b in native):
                continue
            polygon = (
                (x0 * page.width, y0 * page.height),
                (x1 * page.width, y0 * page.height),
                (x1 * page.width, y1 * page.height),
                (x0 * page.width, y1 * page.height),
            )
            atom_id = f"{prefix}/ocrword{number:06d}"
            # Reading order comes from the upright frame the word was read in.
            order[atom_id] = word.get("order_box", word["box"])
            lines.setdefault(tuple(word["line"]), []).append(
                SourceAtom(
                    atom_id=atom_id,
                    page_index=page.page_index,
                    kind="token",
                    text=text,
                    polygon=polygon,
                    native_polygon=to_native.polygon(polygon),
                    source_channel="ocr",
                    quality_flags=(f"ocr_confidence:{float(word.get('confidence', 0)):.3f}",),
                )
            )
        atoms: list[SourceAtom] = []
        ordered = sorted(lines.values(), key=lambda tokens: min(order[t.atom_id][1] for t in tokens))
        scale = page.height if words and words[0].get("rotation") in (90, 270) else page.width
        for line_index, tokens in enumerate(ordered):
            tokens.sort(key=lambda token: order[token.atom_id][0])
            cells, group = [], []
            for token in tokens:
                gap = (order[token.atom_id][0] - order[group[-1].atom_id][2]) * scale if group else 0
                if group and gap > self.cell_gap:
                    cells.append(group)
                    group = []
                group.append(token)
            cells.append(group)
            atoms.extend(tokens)
            for cell_index, members in enumerate(cells):
                atoms.append(
                    self._aggregate(members, f"{prefix}/ocrcell{line_index:04d}_{cell_index:04d}", "cell")
                )
            atoms.append(self._aggregate(tokens, f"{prefix}/ocrline{line_index:04d}", "line"))
        flags = ("ocr_text",) if atoms else ("no_native_text_ocr_required", "ocr_found_no_text")
        if words and words[0].get("rotation"):
            flags = (*flags, f"ocr_rotated:{words[0]['rotation']}")
        merged = (*page.atoms, *atoms)
        if native:
            flags = ("sparse_native_text", *(flag for flag in flags if flag != "no_native_text_ocr_required"))
            if not words or not words[0].get("rotation"):
                # Native and OCR lines share one top-to-bottom reading order.
                others = [atom for atom in merged if atom.kind != "line"]
                rows = sorted(
                    (atom for atom in merged if atom.kind == "line"), key=lambda a: bounds(a.polygon)[1]
                )
                merged = (*others, *rows)
        return page.model_copy(update={"atoms": merged, "quality_flags": flags})

    @staticmethod
    def _aggregate(children: list[SourceAtom], atom_id: str, kind: str) -> SourceAtom:
        polygon = envelope([child.polygon for child in children])
        return SourceAtom(
            atom_id=atom_id,
            page_index=children[0].page_index,
            kind=kind,
            text=" ".join(child.text for child in children),
            polygon=polygon,
            native_polygon=envelope([child.native_polygon for child in children]),
            source_channel="ocr",
            child_ids=tuple(child.atom_id for child in children),
        )


def _blank_page():
    from PIL import Image

    return Image.new("RGB", (256, 256), "white")


class DoctrRecognizer:
    """docTR (Apache-2.0) detection + recognition; loads weights, so remote workers only."""

    def __init__(self, device="cuda", det_arch="db_resnet50", reco_arch="crnn_vgg16_bn"):
        from docextract.vllm_engine import require_remote

        require_remote()
        import doctr
        from doctr.models import ocr_predictor

        self.model = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True).to(device)
        # A GPU image whose cuDNN libraries do not match (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH on
        # a Colab G4 failed every page) reads without cuDNN: the same model, slower convolutions.
        self.cudnn = True
        if str(device).startswith("cuda"):
            try:
                self.read([_blank_page()])
            except RuntimeError as error:
                if "CUDNN" not in str(error).upper():
                    raise
                self.cudnn = False
        self.identity = {
            "engine": "doctr",
            "version": doctr.__version__,
            "det": det_arch,
            "reco": reco_arch,
            "orientation": "retry-turned-when-illegible/1",
        }

    def __call__(self, images: list) -> list[list[dict]]:
        return read_upright(self.read, images)

    def read(self, images: list) -> list[list[dict]]:
        import numpy
        import torch

        enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = enabled and getattr(self, "cudnn", True)
        try:
            result = self.model([numpy.asarray(image.convert("RGB")) for image in images])
        finally:
            torch.backends.cudnn.enabled = enabled
        pages = []
        for page in result.pages:
            words = []
            for block_index, block in enumerate(page.blocks):
                for line_index, line in enumerate(block.lines):
                    for word in line.words:
                        (x0, y0), (x1, y1) = word.geometry
                        words.append(
                            {
                                "text": word.value,
                                "box": (x0, y0, x1, y1),
                                "line": (block_index, line_index),
                                "confidence": word.confidence,
                            }
                        )
            pages.append(words)
        return pages
