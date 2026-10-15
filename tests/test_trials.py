import json
import tarfile
from pathlib import Path

import pytest

from docextract.models import GenerationResult
from docextract.schema import Schema
from docextract.trial import run_value_trial
from tools.build_diagnostics import build
from tools.package_kaggle import package
from tools.score_trial import canonical, score, top_level_records


class ScriptedValues:
    identity = {"backend": "scripted-values", "version": "1"}

    def __init__(self, text, finish_reason="stop"):
        self.text, self.finish_reason = text, finish_reason
        self.calls = 0

    def generate_values(self, schema, units):
        self.calls += 1
        assert units and schema
        return GenerationResult(raw_text=self.text, finish_reason=self.finish_reason)


def test_value_trial_saves_raw_without_grounding_claim_and_reuses_commits(authored, tmp_path):
    pdf, _, _ = authored
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}
    backend = ScriptedValues('{"id":"00042"}')
    first = run_value_trial(pdf, schema, workspace=tmp_path, backend=backend)
    second = run_value_trial(pdf, schema, workspace=tmp_path, backend=backend)
    assert first == second and backend.calls == 1
    assert first.validation.valid and first.data == {"id": "00042"}
    assert not first.citations_measured


@pytest.mark.parametrize(
    "text,reason", [('{"id":"00042"}', "length"), ('{"id":NaN}', "stop"), ('{"id":"1","id":"2"}', "stop")]
)
def test_value_trial_keeps_invalid_output_in_failure_accounting(authored, tmp_path, text, reason):
    pdf, _, _ = authored
    result = run_value_trial(
        pdf, {"type": "object"}, workspace=tmp_path, backend=ScriptedValues(text, reason)
    )
    assert result.data is None and result.error and result.generation.raw_text == text


def test_diagnostics_and_archive_separate_inference_from_answers(tmp_path):
    from pypdf import PdfReader

    info = build(tmp_path / "diagnostics")
    cases = Path(info["inference"])
    packet = json.loads(cases.read_text())
    answers = json.loads(Path(info["answers"]).read_text())
    assert len(packet["cases"]) == len(answers["cases"]) == 24
    assert sum(len(PdfReader(cases.parent / case["pdf"]).pages) for case in packet["cases"]) == 28
    assert len({case["id"] for case in packet["cases"]}) == 24
    for case, answer in zip(packet["cases"], answers["cases"], strict=True):
        assert set(case) == {"id", "pdf", "schema"}
        schema = json.loads((cases.parent / case["schema"]).read_text())
        assert Schema(schema).validate(answer["expected"]).valid
    archive = tmp_path / "trial.tar.gz"
    package(Path(__file__).resolve().parents[1], cases, archive)
    with tarfile.open(archive) as bundle:
        names = bundle.getnames()
        assert sum(name.endswith(".pdf") for name in names) == 24
        assert not any("/evaluation/" in name or "/private/" in name or "/tests/" in name for name in names)
        assert not any(
            name.endswith((".safetensors", ".gguf", ".pt", ".pth", "/example.py")) for name in names
        )
        assert not any("answers" in name or "bindings" in name or "/.git/" in name for name in names)
        assert all(not member.uname and not member.gname for member in bundle.getmembers())
    from tools.build_kaggle_kernel import build as build_kernel

    kernel = tmp_path / "private-kernel"
    prepared = build_kernel(archive, "test-owner/docextract-test", kernel)
    metadata = json.loads((kernel / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and metadata["enable_gpu"] and metadata["machine_shape"] == "NvidiaTeslaT4"
    assert not prepared["submitted"] and prepared["gpu_jobs_launched"] == 0
    compile((kernel / "smoke.py").read_text(), str(kernel / "smoke.py"), "exec")


def test_archive_rejects_answers_referenced_outside_inference_packet(tmp_path):
    info = build(tmp_path / "diagnostics")
    cases = Path(info["inference"])
    packet = json.loads(cases.read_text())
    packet["cases"][0]["schema"] = "../evaluation/answers.json"
    cases.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match="escapes"):
        package(Path(__file__).resolve().parents[1], cases, tmp_path / "bad.tar.gz")
    assert not (tmp_path / "bad.tar.gz").exists()


def test_authored_placement_audit_rejects_wrong_identical_row_occurrence(tmp_path):
    from docextract.learned import extract_learned
    from tools.audit_authored_evidence import audit

    info = build(tmp_path / "diagnostics")
    packet = json.loads(Path(info["inference"]).read_text())
    answer = json.loads(Path(info["answers"]).read_text())["cases"][8]
    case = packet["cases"][8]
    inputs = Path(info["inference"]).parent

    class MappingFixture:
        identity = {"backend": "scripted-table-mapping"}

        def generate(self, task):
            table = task.tables[0]
            return GenerationResult(
                raw_text=json.dumps(
                    {
                        "tables": [
                            {
                                "table_id": table.table_id,
                                "array_pointer": "/transactions",
                                "columns": [
                                    {"pointer": pointer, "header_alias": alias}
                                    for pointer, alias in zip(
                                        ("/code", "/quantity", "/amount"), table.header_aliases, strict=True
                                    )
                                ],
                            }
                        ]
                    }
                )
            )

    result = extract_learned(
        inputs / case["pdf"],
        json.loads((inputs / case["schema"]).read_text()),
        workspace=tmp_path / "run",
        backend=MappingFixture(),
    )
    assert audit(result, answer)["correctly_grounded_fields"] == 9
    first = next(field for field in result.field_evidence if field.pointer == "/transactions/0/code")
    fields = tuple(
        field.model_copy(update={"source_atom_ids": first.source_atom_ids})
        if field.pointer == "/transactions/1/code"
        else field
        for field in result.field_evidence
    )
    damaged = result.model_copy(update={"field_evidence": fields})
    assert damaged.data == result.data
    assert audit(damaged, answer)["correctly_grounded_fields"] == 8


def test_authored_scoring_counts_missing_documents_and_preserves_duplicate_records(tmp_path):
    info = build(tmp_path / "diagnostics")
    report = score(Path(info["answers"]), tmp_path / "missing")
    assert report["groups"]["all"]["expected_documents"] == 24
    assert report["groups"]["all"]["successful_documents"] == 0
    assert report["groups"]["all"]["exact_documents"] == 0
    case_id = report["cases"][0]["id"]
    damaged = tmp_path / "damaged" / case_id
    damaged.mkdir(parents=True)
    (damaged / "value-result.json").write_text('{"broken":')
    damaged_report = score(Path(info["answers"]), tmp_path / "damaged")
    assert damaged_report["groups"]["all"]["expected_documents"] == 24
    assert damaged_report["cases"][0]["invalid_saved_result"]
    one = top_level_records({"rows": [{"x": 1}]})
    two = top_level_records({"rows": [{"x": 1}, {"x": 1}]})
    assert sum((one & two).values()) == 1 and sum(two.values()) == 2
    assert canonical(True) != canonical(1) and canonical(False) != canonical(0)
    assert canonical({"parent": [{"name": "A", "child": [{"name": "X"}]}]}) != canonical(
        {"parent": [{"name": "A", "child": [{"name": "Y"}]}]}
    )
