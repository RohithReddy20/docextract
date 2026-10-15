from copy import deepcopy
from pathlib import Path

import pytest
from reportlab.pdfbase.pdfmetrics import stringWidth

pytest.importorskip("extract_bench")

from extract_bench.evaluation.metrics.extract.unified_evidence_metric import (  # noqa: E402
    compute_unified_evidence_metrics,
)
from extract_bench.inference.pipelines import get_pipeline  # noqa: E402
from extract_bench.inference.providers.registry import create_provider  # noqa: E402
from extract_bench.schemas.pipeline_io import InferenceRequest, RawInferenceResult  # noqa: E402
from extract_bench.test_cases.schema import ExtractFieldTestRule  # noqa: E402

from docextract.example import EXPECTED  # noqa: E402
from docextract.integrations.extractbench import benchmark_path, project_result, register  # noqa: E402

pytestmark = pytest.mark.benchmark


def test_official_projection_paths_coordinates_and_multiline_support(result):
    output = project_result(result)
    assert output.extracted_data == EXPECTED
    assert output.task_type == "extract"
    assert {citation.field_path for citation in output.field_citations} >= {
        "departments[0].items[0].amount",
        "supplier.name",
        "document_id",
    }
    path = "departments[0].items[2].description"
    assert {citation.page for citation in output.field_citations if citation.field_path == path} == {1, 2}
    assert not any(
        citation.field_path in ("currency", "missing_contact") for citation in output.field_citations
    )
    for citation in output.field_citations:
        x, y, width, height = citation.bbox
        assert citation.page >= 1
        assert 0 <= x < x + width <= 1
        assert 0 <= y < y + height <= 1


def test_official_scorer_accepts_authored_value_and_independent_label_geometry(result):
    output = project_result(result)
    x = 48 + stringWidth("Document ID: ", "Helvetica", 11)
    rule = ExtractFieldTestRule.model_validate(
        {
            "field_path": "document_id",
            "evidence": [
                {
                    "page": 1,
                    "bbox": [
                        x / 612,
                        (792 - 717) / 792,
                        stringWidth("00042", "Helvetica", 11) / 612,
                        11 / 792,
                    ],
                    "value": "00042",
                    "quote": "00042",
                }
            ],
        }
    )
    from docextract.example import SCHEMA

    for index, baseline in [(0, 510), (1, 482)]:
        rule_box = [
            408 / 612,
            (792 - baseline - 8) / 792,
            stringWidth("12.50", "Helvetica", 10) / 612,
            10 / 792,
        ]
        # Boxes come from the authoring coordinates, not projected predictions.
        nested_rule = ExtractFieldTestRule.model_validate(
            {
                "field_path": f"departments[0].items[{index}].amount",
                "evidence": [{"page": 1, "bbox": rule_box, "value": 12.5}],
            }
        )
        nested_metrics = compute_unified_evidence_metrics(
            EXPECTED,
            output.extracted_data,
            [nested_rule],
            output.field_citations,
            data_schema=SCHEMA,
            fuzzy_field_thresholds={},
        )
        assert (
            next(
                metric.value
                for metric in nested_metrics
                if metric.metric_name == "extract_unified_grounded_f1"
            )
            == 1
        )

    metrics = compute_unified_evidence_metrics(
        EXPECTED,
        output.extracted_data,
        [rule],
        output.field_citations,
        data_schema=SCHEMA,
        fuzzy_field_thresholds={},
    )
    values = {metric.metric_name: metric.value for metric in metrics}
    assert values["extract_unified_value_f1"] == 1
    assert values["extract_unified_page_f1"] == 1
    assert values["extract_unified_grounded_f1"] == 1
    assert all(not metric.metadata["value_incomplete"] for metric in metrics)
    assert all(not metric.metadata["grounded_incomplete"] for metric in metrics)


def test_official_assignment_preserves_duplicate_count(result):
    output = project_result(result)
    dropped = deepcopy(output.extracted_data)
    dropped["departments"][0]["items"].pop(0)
    from docextract.example import SCHEMA

    metrics = compute_unified_evidence_metrics(EXPECTED, dropped, [], data_schema=SCHEMA)
    recall = next(metric.value for metric in metrics if metric.metric_name == "extract_unified_value_recall")
    assert recall < 1


def test_provider_registration_and_pure_saved_normalization(authored, tmp_path, monkeypatch):
    pdf, schema, plan = authored
    register()
    learned_pipeline = get_pipeline("docextract_model_trial")
    assert learned_pipeline.provider_name == "docextract_model_trial"
    pipeline = get_pipeline("docextract_b0").model_copy(
        update={"config": {"bindings": plan.model_dump(), "workspace": str(tmp_path), "apply_defaults": True}}
    )
    provider = create_provider(pipeline)
    raw = provider.run_inference(
        pipeline,
        InferenceRequest(
            example_id="authored", source_file_path=str(pdf), product_type="extract", schema_override=schema
        ),
    )
    saved = RawInferenceResult.model_validate_json(raw.model_dump_json())
    before = saved.model_dump_json()

    def forbidden(*args, **kwargs):
        raise AssertionError("normalize must not read source files or call extraction")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr("docextract.integrations.extractbench.extract", forbidden)
    first = provider.normalize(saved)
    second = provider.normalize(saved)
    assert first == second
    assert before == saved.model_dump_json()
    assert first.output.extracted_data == EXPECTED
    assert first.raw_output["extraction_result"]["completion"]["status"] == "partial"
    learned_provider = create_provider(learned_pipeline)
    monkeypatch.setattr("docextract.remote.require_kaggle", forbidden)
    monkeypatch.setattr("docextract.learned.extract_learned", forbidden)
    assert learned_provider.normalize(saved).output == first.output


def test_ambiguous_benchmark_property_paths_fail_explicitly():
    assert benchmark_path("/rows/0/value", {"rows": [{"value": 1}]}) == "rows[0].value"
    with pytest.raises(ValueError, match="grammar"):
        benchmark_path("/a.b", {"a.b": 1})


def test_official_scoring_tool_keeps_failures_in_the_population(tmp_path):
    import json
    import shutil

    from reportlab.pdfgen.canvas import Canvas

    from docextract.cited import extract_cited
    from docextract.models import GenerationResult
    from tools.score_extractbench import score

    pdf = tmp_path / "doc.pdf"
    canvas = Canvas(str(pdf), invariant=1)
    canvas.drawString(48, 740, "Invoice ID: 0017")
    canvas.save()
    schema = {"type": "object", "properties": {"invoice_id": {"type": "string"}}}

    class Backend:
        identity = {"backend": "scripted", "version": "1"}

        def generate_cited(self, task):
            alias = task.units[0].alias
            return GenerationResult(raw_text=json.dumps({"invoice_id": {"value": "0017", "source": alias}}))

    evaluation = tmp_path / "evaluation"
    for name in ("good", "lost"):
        (evaluation / "short").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, evaluation / "short" / f"{name}.pdf")
        sidecar = {"tags": [], "data_schema": schema, "expected_output": {"invoice_id": "0017"}}
        (evaluation / "short" / f"{name}.test.json").write_text(json.dumps(sidecar))
    index = [
        {"id": name, "split": "short", "test_json": f"short/{name}.test.json"} for name in ("good", "lost")
    ]
    (evaluation / "index.json").write_text(json.dumps({"cases": index}))
    extract_cited(pdf, schema, workspace=tmp_path / "results" / "good", backend=Backend())
    report = score(evaluation, tmp_path / "results", "docextract_trial")
    rows = {row["id"]: row for row in report["cases"]}
    assert rows["good"]["extract_unified_value_f1"] == 1.0 and rows["good"]["result"] == "scored"
    assert rows["lost"]["result"] == "missing"
    assert report["groups"]["all"]["value_f1"] == 0.5
