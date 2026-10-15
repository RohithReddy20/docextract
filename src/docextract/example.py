"""Authored CPU fixture. The engine does not import these records or expected values."""

from pathlib import Path

from docextract.contracts import save_json
from docextract.planning import BindingPlan

SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["document_id", "supplier", "approved", "zero", "departments"],
    "properties": {
        "document_id": {"type": "string"},
        "supplier": {
            "type": "object",
            "required": ["name"],
            "additionalProperties": False,
            "properties": {"name": {"type": "string"}},
        },
        "approved": {"type": ["boolean", "null"]},
        "zero": {"type": "number"},
        "currency": {"type": "string", "default": "USD"},
        "missing_contact": {"type": "string"},
        "departments": {"type": "array", "items": {"$ref": "#/$defs/Department"}},
    },
    "$defs": {
        "Department": {
            "type": "object",
            "required": ["name", "items"],
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["sku", "description", "quantity", "amount", "tags"],
                        "additionalProperties": False,
                        "properties": {
                            "sku": {"type": "string"},
                            "description": {"type": "string"},
                            "quantity": {"type": "integer", "minimum": 0},
                            "amount": {"type": "number", "minimum": 0},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        }
    },
}

PLAN = {
    "fields": {
        "/document_id": "Document ID",
        "/supplier/name": "Supplier",
        "/approved": "Approved",
        "/zero": "Zero",
        "/currency": "Currency",
        "/missing_contact": "Contact",
    },
    "section": {
        "path": "/departments",
        "label": "Department",
        "field": "name",
        "continued_label": "Department continued",
    },
    "tables": [
        {
            "path": "/departments/*/items",
            "key": "sku",
            "columns": {
                "sku": "SKU",
                "description": "Description",
                "quantity": "Qty",
                "amount": "Amount",
                "tags": "Tags",
            },
            "continuation_columns": ["description"],
        }
    ],
}

EXPECTED = {
    "document_id": "00042",
    "supplier": {"name": "Northwind Supplies"},
    "approved": None,
    "zero": 0,
    "currency": "USD",
    "departments": [
        {
            "name": "Operations",
            "items": [
                {"sku": "A001", "description": "Notebook", "quantity": 2, "amount": 12.5, "tags": ["office"]},
                {"sku": "A001", "description": "Notebook", "quantity": 2, "amount": 12.5, "tags": ["office"]},
                {
                    "sku": "A002",
                    "description": "Extended support through December",
                    "quantity": 1,
                    "amount": 20,
                    "tags": ["service"],
                },
            ],
        },
        {
            "name": "Sales",
            "items": [
                {"sku": "A003", "description": "Travel pack", "quantity": 3, "amount": 9, "tags": []},
            ],
        },
    ],
}


def create_example(directory: Path) -> tuple[Path, dict, BindingPlan]:
    from reportlab.lib import colors
    from reportlab.pdfgen.canvas import Canvas

    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / "authored-order.pdf"
    canvas = Canvas(str(pdf), pagesize=(612, 792), invariant=1)
    canvas.setTitle("Authored docextract CPU example")
    positions = (48, 126, 350, 408, 486)

    def text(y, content, x=48, *, bold=False, size=11):
        canvas.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        canvas.setFillColor(colors.HexColor("#17324d") if bold else colors.black)
        canvas.drawString(x, y, content)

    def header(y):
        for x, label in zip(positions, ("SKU", "Description", "Qty", "Amount", "Tags"), strict=True):
            text(y, label, x, bold=True, size=10)
        canvas.setStrokeColor(colors.HexColor("#b7c8d8"))
        canvas.line(48, y - 7, 566, y - 7)

    def row(y, values):
        for x, value in zip(positions, values, strict=True):
            if value:
                text(y, value, x, size=10)

    text(746, "Purchase order", bold=True, size=22)
    text(708, "Document ID: 00042")
    text(682, "Supplier: Northwind Supplies")
    text(656, "Approved: NULL")
    text(630, "Zero: 0")
    text(574, "Department: Operations", bold=True)
    header(540)
    row(510, ("A001", "Notebook", "2", "12.50", '["office"]'))
    row(482, ("A001", "Notebook", "2", "12.50", '["office"]'))
    row(130, ("A002", "Extended support", "1", "20.00", '["service"]'))
    text(92, "Items continue on next page", size=10)
    text(40, "Authored extraction fixture | Page 1 of 2", size=9)
    canvas.showPage()
    text(746, "Purchase order - continued", bold=True, size=20)
    text(710, "Department continued: Operations", bold=True)
    header(676)
    row(646, ("", "through December", "", "", ""))
    text(610, "End items", size=10)
    text(548, "Department: Sales", bold=True)
    header(514)
    row(484, ("A003", "Travel pack", "3", "9.00", "[]"))
    text(448, "End items", size=10)
    text(40, "Authored extraction fixture | Page 2 of 2", size=9)
    canvas.save()
    save_json(directory / "schema.json", SCHEMA)
    save_json(directory / "bindings.json", PLAN)
    save_json(directory / "expected.json", EXPECTED)
    return pdf, SCHEMA, BindingPlan.model_validate(PLAN)
