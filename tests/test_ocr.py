"""OCR fallback with an injected recognizer; no OCR model is loaded locally."""

import json

from PIL import Image, ImageDraw
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from docextract.cited import extract_cited
from docextract.interpretation import evidence_units, render_evidence
from docextract.models import GenerationResult
from docextract.parsing.ocr import OcrFallbackParser


def image_only_pdf(tmp_path):
    image = Image.new("RGB", (850, 1100), "white")
    ImageDraw.Draw(image).text((100, 100), "Invoice ID: 0017", fill="black")
    path = tmp_path / "scan.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792), invariant=1)
    canvas.drawString(48, 740, "Native page")
    canvas.showPage()
    canvas.drawImage(ImageReader(image), 0, 0, width=612, height=792)
    canvas.showPage()
    canvas.save()
    return path


class FakeRecognizer:
    identity = {"engine": "fake-ocr", "version": "1"}

    def __init__(self):
        self.calls = []

    def __call__(self, images):
        self.calls.append(len(images))
        return [
            [
                {"text": "Invoice", "box": (0.10, 0.10, 0.20, 0.12), "line": (0, 0), "confidence": 0.98},
                {"text": "ID:", "box": (0.21, 0.10, 0.25, 0.12), "line": (0, 0), "confidence": 0.97},
                {"text": "0017", "box": (0.26, 0.10, 0.32, 0.12), "line": (0, 0), "confidence": 0.95},
            ]
            for _ in images
        ]


def test_only_pages_without_text_are_recognized_and_cited_as_words(tmp_path):
    pdf = image_only_pdf(tmp_path)
    recognizer = FakeRecognizer()
    parser = OcrFallbackParser(recognizer)
    snapshot = parser.parse(pdf)
    assert recognizer.calls == [1]
    assert snapshot.pages[0].quality_flags == () and snapshot.pages[1].quality_flags == ("ocr_text",)
    words = [
        atom for atom in snapshot.pages[1].atoms if atom.source_channel == "ocr" and atom.kind == "token"
    ]
    assert [word.text for word in words] == ["Invoice", "ID:", "0017"]
    x0, y0 = words[0].polygon[0]
    assert abs(x0 - 61.2) < 1e-6 and abs(y0 - 79.2) < 1e-6
    assert "Invoice ID: 0017" in render_evidence(evidence_units(snapshot))

    class Backend:
        identity = {"backend": "scripted", "version": "1"}

        def generate_cited(self, task):
            unit = next(unit for unit in task.units if "0017" in unit.text)
            return GenerationResult(
                raw_text=json.dumps({"invoice_id": {"value": "0017", "source": unit.alias}})
            )

    schema = {"type": "object", "properties": {"invoice_id": {"type": "string"}}}
    result = extract_cited(pdf, schema, workspace=tmp_path / "run", backend=Backend(), parser=parser)
    evidence = result.field_evidence[0]
    assert result.data == {"invoice_id": "0017"} and evidence.support == "character"
    assert [result.snapshot.atom_index()[atom].text for atom in evidence.source_atom_ids] == ["0017"]


def test_a_sideways_scan_is_read_upright_and_mapped_back_to_the_page(tmp_path):
    from docextract.parsing.ocr import read_upright, unrotate

    page = Image.new("RGB", (100, 200), "white")
    page.putpixel((10, 10), (0, 0, 0))  # top-left marker of the page as rendered
    upright = [
        {"text": "Lease", "box": (0.1, 0.2, 0.3, 0.25), "line": (0, 0), "confidence": 0.97},
        {"text": "Carla", "box": (0.31, 0.2, 0.45, 0.25), "line": (0, 0), "confidence": 0.96},
        {"text": "Gonzales", "box": (0.1, 0.4, 0.4, 0.45), "line": (0, 1), "confidence": 0.95},
    ]
    garbage = [{"text": "ii", "box": (0.1, 0.1, 0.2, 0.9), "line": (0, 0), "confidence": 0.2}]
    seen = []

    def read(images):
        # Only the page turned 270 degrees counterclockwise puts the marker top-right: legible.
        seen.append([image.size for image in images])
        return [
            upright if image.size == (200, 100) and image.getpixel((189, 10)) == (0, 0, 0) else garbage
            for image in images
        ]

    words = read_upright(read, [page])[0]
    assert len(seen) == 4 and [word["rotation"] for word in words] == [270, 270, 270]
    assert words[0]["box"] == unrotate((0.1, 0.2, 0.3, 0.25), 270) == (0.2, 0.7, 0.25, 0.9)
    assert unrotate(unrotate((0.1, 0.2, 0.3, 0.4), 90), 270) != (0.1, 0.2, 0.3, 0.4)
    for angle in (90, 180, 270):  # the marker maps back to where it was printed
        turned = page.rotate(angle, expand=True)
        x, y = next(
            (x, y)
            for x in range(turned.width)
            for y in range(turned.height)
            if turned.getpixel((x, y)) == (0, 0, 0)
        )
        rx, ry = (x + 0.5) / turned.width, (y + 0.5) / turned.height
        back = unrotate((rx, ry, rx, ry), angle)
        assert abs(back[0] * 100 - 10.5) < 1 and abs(back[1] * 200 - 10.5) < 1

    legible = read_upright(lambda images: [upright for _ in images], [page])[0]
    assert [word["rotation"] for word in legible] == [0, 0, 0]

    class Rotated:
        identity = {"engine": "fake-rotated", "version": "1"}

        def __call__(self, images):
            return [words for _ in images]

    snapshot = OcrFallbackParser(Rotated()).parse(image_only_pdf(tmp_path))
    assert snapshot.pages[1].quality_flags == ("ocr_text", "ocr_rotated:270")
    lines = [unit.text for unit in evidence_units(snapshot) if unit.page_index == 1]
    assert lines == ["Lease Carla", "Gonzales"]


def test_a_scanned_page_with_a_little_real_text_is_also_recognized(tmp_path):
    image = Image.new("RGB", (850, 1100), "white")
    path = tmp_path / "stamped.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792), invariant=1)
    canvas.drawImage(ImageReader(image), 0, 0, width=612, height=792)
    canvas.drawString(48, 740, "Stamp 01")
    canvas.showPage()
    canvas.save()

    class Recognizer:
        identity = {"engine": "fake-sparse", "version": "1"}

        def __call__(self, images):
            return [
                [
                    # The printed stamp is read again by OCR; it must not appear twice.
                    {"text": "Stamp", "box": (0.078, 0.055, 0.13, 0.07), "line": (0, 0), "confidence": 0.9},
                    {"text": "Invoice", "box": (0.10, 0.30, 0.20, 0.32), "line": (0, 1), "confidence": 0.98},
                    {"text": "0017", "box": (0.21, 0.30, 0.27, 0.32), "line": (0, 1), "confidence": 0.95},
                ]
                for _ in images
            ]

    snapshot = OcrFallbackParser(Recognizer()).parse(path)
    assert snapshot.pages[0].quality_flags == ("sparse_native_text", "ocr_text")
    assert [unit.text for unit in evidence_units(snapshot)] == ["Stamp 01", "Invoice 0017"]
    plain = OcrFallbackParser(Recognizer(), sparse_chars=5).parse(path)
    assert [unit.text for unit in evidence_units(plain)] == ["Stamp 01"]


def test_scanned_pages_can_be_shown_to_the_model_as_images(tmp_path):
    from docextract.remote import cited_messages

    pdf = image_only_pdf(tmp_path)
    store = tmp_path / "page-images"
    parser = OcrFallbackParser(FakeRecognizer(), image_store=store)
    seen = []

    class Backend:
        identity = {"backend": "scripted-images", "version": "1"}

        def generate_cited(self, task):
            seen.append(task)
            unit = next(unit for unit in task.units if "0017" in unit.text)
            return GenerationResult(
                raw_text=json.dumps({"invoice_id": {"value": "0017", "source": unit.alias}})
            )

    schema = {"type": "object", "properties": {"invoice_id": {"type": "string"}}}
    result = extract_cited(
        pdf, schema, workspace=tmp_path / "run", backend=Backend(), parser=parser, page_images=store
    )
    assert result.data == {"invoice_id": "0017"}
    (task,) = seen
    assert len(task.page_images) == 1 and task.page_images[0].endswith("p0001.jpg")
    with Image.open(task.page_images[0]) as image:
        assert max(image.size) == 1280
    messages, _ = cited_messages(task, "a")
    kinds = [part["type"] for part in messages[1]["content"]]
    assert kinds == ["text", "image", "text"] and "scanned pages" in messages[0]["content"]
    nuextract, options = cited_messages(task, "b")
    assert [part["type"] for part in nuextract[0]["content"]] == ["image", "text"]
    assert "scanned pages" in options["instructions"]
    plain = extract_cited(pdf, schema, workspace=tmp_path / "plain", backend=Backend(), parser=parser)
    assert plain.data == result.data and seen[-1].page_images == ()
    assert seen[-1].task_id != task.task_id


def test_a_document_without_scanned_pages_shares_answers_with_the_text_only_variant(tmp_path):
    from docextract.batch import run_batch

    native = tmp_path / "native.pdf"
    canvas = Canvas(str(native), pagesize=(612, 792), invariant=1)
    canvas.drawString(48, 740, "Invoice ID: 0017")
    canvas.showPage()
    canvas.save()

    class Engine:
        identity = {"backend": "scripted-shared", "version": "1"}

        def __init__(self):
            self.generated = 0

        def generate_batch(self, tasks):
            self.generated += len(tasks)
            return [GenerationResult(raw_text=json.dumps({"invoice_id": "0017"})) for _ in tasks]

    schema = {"type": "object", "properties": {"invoice_id": {"type": "string"}}}
    engine = Engine()
    jobs = [
        ("text/doc", native, schema, {"protocol": "values"}),
        ("images/doc", native, schema, {"protocol": "values", "page_images": tmp_path / "page-images"}),
    ]
    outcomes = run_batch(jobs, root=tmp_path / "run", engine=engine)
    assert engine.generated == 1
    assert outcomes["text/doc"].data == outcomes["images/doc"].data == {"invoice_id": "0017"}


def test_a_scan_with_its_own_text_layer_keeps_the_text_and_its_image(tmp_path):
    from docextract.parsing.ocr import page_image_path

    image = Image.new("RGB", (850, 1100), "white")
    path = tmp_path / "layered.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792), invariant=1)
    canvas.drawImage(ImageReader(image), 0, 0, width=612, height=792)
    for row in range(12):  # a text layer well above the sparse threshold
        canvas.drawString(48, 740 - row * 20, f"Lease name Maple Unit row {row} operator Example Drilling Co")
    canvas.showPage()
    canvas.save()
    recognizer = FakeRecognizer()
    store = tmp_path / "page-images"
    snapshot = OcrFallbackParser(recognizer, image_store=store).parse(path)
    assert recognizer.calls == []  # the text layer is used, not OCR
    assert snapshot.pages[0].quality_flags == ("scanned_text_layer",)
    assert page_image_path(store, snapshot.snapshot_id, 0).exists()


def test_a_schema_split_into_many_groups_gets_smaller_page_images(tmp_path):
    pdf = image_only_pdf(tmp_path)
    store = tmp_path / "page-images"
    parser = OcrFallbackParser(FakeRecognizer(), image_store=store)
    seen = []

    class Backend:
        identity = {"backend": "scripted-sides", "version": "1"}

        def generate_cited(self, task):
            seen.append(task.page_image_side)
            return GenerationResult(raw_text="{}")

    wide = {
        "type": "object",
        "properties": {f"field_{n}": {"type": "string", "description": "x" * 200} for n in range(12)},
    }
    options = {"parser": parser, "page_images": store, "page_image_side": 1280, "page_image_side_split": 768}
    extract_cited(pdf, wide, workspace=tmp_path / "wide", backend=Backend(), max_schema_chars=700, **options)
    assert seen and set(seen) == {768}
    seen.clear()
    small = {"type": "object", "properties": {"invoice_id": {"type": "string"}}}
    extract_cited(pdf, small, workspace=tmp_path / "small", backend=Backend(), **options)
    assert seen == [1280]


def test_a_scanned_page_can_show_its_ocr_lines_as_its_text(tmp_path):
    from docextract.oneshot import extract_oneshot, recognized_texts

    pdf = image_only_pdf(tmp_path)
    parser = OcrFallbackParser(FakeRecognizer())
    snapshot = parser.parse(pdf)
    assert recognized_texts(snapshot, ("Native page", "")) == ("Native page", "Invoice ID: 0017")

    class Backend:
        identity = {"backend": "scripted", "version": "1"}

        def __init__(self):
            self.tasks = []

        def generate_cited(self, task):
            self.tasks.append(task)
            return GenerationResult(raw_text='{"invoice_id": "0017"}', finish_reason="stop")

    schema = {"type": "object", "properties": {"invoice_id": {"type": "string"}}}
    plain, read = Backend(), Backend()
    extract_oneshot(pdf, schema, workspace=tmp_path / "plain", backend=plain, parser=parser)
    extract_oneshot(pdf, schema, workspace=tmp_path / "ocr", backend=read, parser=parser, ocr_text=True)
    assert plain.tasks[0].page_texts[1].strip() == "" and read.tasks[0].page_texts[1] == "Invoice ID: 0017"
    assert plain.tasks[0].task_id != read.tasks[0].task_id
