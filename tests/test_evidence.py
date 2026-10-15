import ctypes
import hashlib
from contextlib import closing

import pypdfium2 as pdfium
import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject
from reportlab.pdfgen.canvas import Canvas

from docextract.geometry import Affine, bounds, crop_resize, normalized_coco, page_transform
from docextract.parsing.docparse_adapter import DocparseAdapter
from docextract.parsing.pdfium import PdfiumParser


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_real_pdf_rotated_crop_geometry_agrees_with_pdfium_device_mapping(tmp_path, rotation):
    source = tmp_path / "source.pdf"
    canvas = Canvas(str(source), pagesize=(612, 792))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(100, 600, "ID: 00042")
    canvas.save()
    reader = PdfReader(source)
    page = reader.pages[0]
    page.cropbox = RectangleObject((20, 40, 580, 760))
    page.rotate(rotation)
    destination = tmp_path / f"crop-{rotation}.pdf"
    writer = PdfWriter()
    writer.add_page(page)
    writer.write(destination)
    snapshot = PdfiumParser().parse(destination)
    evidence = snapshot.pages[0]
    assert evidence.crop_box == (20, 40, 580, 760)
    assert evidence.rotation == rotation
    assert (evidence.width, evidence.height) == ((720, 560) if rotation % 180 else (560, 720))
    transform = Affine(*evidence.native_to_display)
    tokens = [atom for atom in evidence.atoms if atom.kind == "token"]
    assert [atom.text for atom in tokens] == ["ID:", "00042"]
    with closing(pdfium.PdfDocument(destination)) as pdf, closing(pdf[0]) as pdf_page:
        for token in tokens:
            for native, display in zip(token.native_polygon, token.polygon, strict=True):
                x, y = ctypes.c_int(), ctypes.c_int()
                assert pdfium.raw.FPDF_PageToDevice(
                    pdf_page,
                    0,
                    0,
                    int(evidence.width),
                    int(evidence.height),
                    0,
                    *native,
                    ctypes.byref(x),
                    ctypes.byref(y),
                )
                assert display == pytest.approx((x.value, y.value), abs=1.1)
                assert transform.inverse().point(display) == pytest.approx(native)
            x, y, width, height = normalized_coco(token.polygon, evidence.width, evidence.height)
            assert 0 <= x < x + width <= 1
            assert 0 <= y < y + height <= 1
        with closing(pdf_page.render(scale=1)) as bitmap:
            image = bitmap.to_pil()
            left, top, right, bottom = bounds(tokens[1].polygon)
            region = image.crop((int(left), int(top), int(right) + 1, int(bottom) + 1)).convert("L")
            assert region.getextrema()[0] < 64


def test_crop_resize_and_page_round_trip():
    transform, _, _ = page_transform((20, 40, 580, 760), 90)
    combined = transform.then(crop_resize(10, 15, 2, 3))
    point = (125.5, 604.25)
    assert combined.inverse().point(combined.point(point)) == pytest.approx(point)
    assert combined.point(point) == crop_resize(10, 15, 2, 3).point(transform.point(point))


def test_snapshot_ids_are_content_and_parser_bound(authored, tmp_path):
    pdf, _, _ = authored
    parser = PdfiumParser()
    first = parser.parse(pdf)
    copy = tmp_path / "random-filename.pdf"
    copy.write_bytes(pdf.read_bytes())
    second = parser.parse(copy)
    assert first == second
    assert first.document_sha256 == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert PdfiumParser(cell_gap=18).parse(pdf).snapshot_id != first.snapshot_id
    for kind in ("character", "token", "cell"):
        ids = [atom.atom_id for page in first.pages for atom in page.atoms if atom.kind == kind]
        assert ids and len(ids) == len(set(ids))
    assert "Document ID: 00042" in first.pages[0].native_text


def test_sparse_native_text_is_retained_and_blank_page_is_flagged(tmp_path):
    pdf = tmp_path / "sparse.pdf"
    canvas = Canvas(str(pdf))
    canvas.setFont("Helvetica", 12)
    canvas.drawString(50, 700, "00042")
    canvas.showPage()
    canvas.showPage()
    canvas.save()
    snapshot = PdfiumParser().parse(pdf)
    assert any(atom.text == "00042" for atom in snapshot.pages[0].atoms if atom.kind == "token")
    assert "no_native_text_ocr_required" in snapshot.pages[1].quality_flags


def test_docparse_adapter_fails_closed_until_public_export_available(authored):
    pdf, _, _ = authored
    with pytest.raises(RuntimeError, match="exporter"):
        DocparseAdapter().parse(pdf)
    snapshot = PdfiumParser().parse(pdf)
    with pytest.raises(ValueError, match="commit"):
        DocparseAdapter(lambda _: snapshot).parse(pdf)


def test_injected_docparse_contract_uses_ids_without_native_naming_assumptions(authored, tmp_path):
    from docextract.contracts import ParseSnapshot, digest
    from docextract.engine import extract
    from docextract.example import EXPECTED
    from docextract.parsing.docparse_adapter import DOCPARSE_COMMIT

    pdf, schema, plan = authored
    original = PdfiumParser().parse(pdf)
    identity = {
        "backend": "docparse",
        "commit": DOCPARSE_COMMIT,
        "adapter_version": "docparse/1",
        "fixture_export": True,
    }
    snapshot_id = digest({"document": original.document_sha256, "parser": identity})
    renamed = {
        atom.atom_id: f"{snapshot_id}/occurrence-{index}"
        for index, atom in enumerate(atom for page in original.pages for atom in page.atoms)
    }
    pages = tuple(
        page.model_copy(
            update={
                "atoms": tuple(
                    atom.model_copy(
                        update={
                            "atom_id": renamed[atom.atom_id],
                            "child_ids": tuple(renamed[child] for child in atom.child_ids),
                        }
                    )
                    for atom in page.atoms
                )
            }
        )
        for page in original.pages
    )
    # An injected contract fixture, not evidence that the real docparse exporter exists.
    fixture = ParseSnapshot(
        snapshot_id=snapshot_id,
        document_sha256=original.document_sha256,
        parser_identity=identity,
        pages=pages,
    )
    result = extract(
        pdf,
        schema,
        workspace=tmp_path,
        plan=plan,
        parser=DocparseAdapter(lambda _: fixture),
        apply_defaults=True,
    )
    assert result.data == EXPECTED
    assert result.snapshot.parser_identity["fixture_export"]


def test_filled_form_fields_become_text_beside_their_labels(tmp_path):
    from reportlab.pdfgen.canvas import Canvas

    from docextract.interpretation import evidence_units
    from docextract.parsing.pdfium import PdfiumParser

    path = tmp_path / "form.pdf"
    canvas = Canvas(str(path))
    canvas.drawString(48, 740, "Lease name:")
    canvas.acroForm.textfield(name="lease", value="Maple Unit", x=150, y=734, width=200, height=20)
    canvas.drawString(48, 700, "Oil")
    canvas.acroForm.checkbox(name="oil", checked=True, x=150, y=696, size=14)
    canvas.drawString(250, 700, "Gas")
    canvas.acroForm.checkbox(name="gas", checked=False, x=300, y=696, size=14)
    canvas.showPage()
    canvas.save()
    snapshot = PdfiumParser().parse(path)
    lines = [unit.text for unit in evidence_units(snapshot)]
    assert "Maple Unit" in lines and lines.index("Maple Unit") == lines.index("Lease name:") + 1
    assert lines[-4:] == ["Oil", "☒", "Gas", "☐"]
    form_atoms = [atom for atom in snapshot.pages[0].atoms if atom.source_channel == "form_field"]
    assert "".join(atom.text for atom in form_atoms) == "Maple Unit☒☐"
    assert not snapshot.pages[0].quality_flags
    plain = PdfiumParser(forms=False).parse(path)
    assert "Maple Unit" not in [unit.text for unit in evidence_units(plain)]


def test_windows_can_cap_cells_so_dense_tables_get_smaller_windows():
    from docextract.interpretation import EvidenceUnit, windows

    def unit(number, line, text):
        return EvidenceUnit(
            alias=f"s{number}",
            atom_id=f"a{number}",
            line_id=f"line{line}",
            page_index=0,
            text=text,
            bbox=(0, 0, 1, 1),
            offset_atom_ids=tuple(f"a{number}" for _ in text),
        )

    table = tuple(unit(line * 4 + cell, line, "9") for line in range(10) for cell in range(4))
    prose = tuple(unit(100 + line, 100 + line, "word " * 40) for line in range(3))
    plain = list(windows(table + prose, max_chars=10_000, overlap_lines=0, unit_overhead=12))
    assert len(plain) == 1
    capped = list(windows(table + prose, max_chars=10_000, overlap_lines=0, unit_overhead=12, max_units=12))
    assert [len(window) for window in capped] == [12, 12, 12, 7]
    assert all(window[0].line_id != window[-1].line_id or len(window) <= 12 for window in capped)


def test_page_aligned_windows_show_each_page_once():
    from docextract.interpretation import EvidenceUnit, windows

    def unit(number, page, line):
        return EvidenceUnit(
            alias=f"s{number}",
            atom_id=f"a{number}",
            line_id=f"p{page}l{line}",
            page_index=page,
            text="x" * 20,
            bbox=(0, 0, 1, 1),
            offset_atom_ids=tuple(f"a{number}" for _ in range(20)),
        )

    units = tuple(unit(page * 10 + line, page, line) for page in range(5) for line in range(4))
    # 4 lines of 20 chars per page; 200 chars hold 2.5 pages, so windows end at page boundaries.
    plain = list(windows(units, max_chars=200, overlap_lines=1, unit_overhead=0))
    aligned = list(windows(units, max_chars=200, overlap_lines=1, unit_overhead=0, align_pages=True))
    pages = [sorted({u.page_index for u in window}) for window in aligned]
    assert pages == [[0, 1], [2, 3], [4]]
    assert sum(len(window) for window in aligned) == len(units)
    assert any(
        len({u.page_index for u in window}) > 2 or window[0].line_id.endswith("l3") for window in plain
    )
    tiny = list(windows(units[:4], max_chars=50, overlap_lines=0, unit_overhead=0, align_pages=True))
    assert [len(window) for window in tiny] == [2, 2]


def test_words_set_apart_by_position_alone_stay_separate_words(tmp_path):
    from reportlab.pdfbase.pdfmetrics import stringWidth
    from reportlab.pdfgen.canvas import Canvas

    from docextract.interpretation import evidence_units
    from docextract.parsing.pdfium import PdfiumParser

    path = tmp_path / "tight.pdf"
    canvas = Canvas(str(path))
    x = 48.0
    for word in ["City", "of", "Springfield", "Water", "Revenue", "11.50"]:
        # Typeset text often has no space glyphs: words are placed with a small gap instead.
        canvas.drawString(x, 740, word)
        x += stringWidth(word, "Helvetica", 12) + 2.2
    canvas.showPage()
    canvas.save()
    assert [unit.text for unit in evidence_units(PdfiumParser().parse(path))] == [
        "City of Springfield Water Revenue 11.50"
    ]
