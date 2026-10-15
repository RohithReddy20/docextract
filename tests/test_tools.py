import hashlib
import json
from pathlib import Path

import pytest

from tools.schema_inventory import REVISION, inventory


def test_inventory_verifies_pinned_bytes_and_omits_answers(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    files = []
    for split in ("short", "medium", "long"):
        path = snapshot / f"{split}.jsonl"
        path.write_text(
            json.dumps(
                {
                    "data_schema": json.dumps({"type": "object", "properties": {"code": {"type": "string"}}}),
                    "expected_output": "PRIVATE_ANSWER",
                    "field_rules": "PRIVATE_RULE",
                }
            )
            + "\n"
        )
        files.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (tmp_path / "dataset-manifest.json").write_text(
        json.dumps({"revision": REVISION, "mode": "downloaded_remote_snapshot", "files": files})
    )
    output = inventory(snapshot)
    assert output["case_counts"] == {"short": 1, "medium": 1, "long": 1}
    assert output["unique_schemas"] == 1
    assert "PRIVATE" not in json.dumps(output)
    assert "expected_output" not in json.dumps(output)
    (snapshot / "long.jsonl").write_text("")
    with pytest.raises(ValueError, match="hash mismatch"):
        inventory(snapshot)


def test_schema_audit_kernel_is_private_cpu_only_and_contains_no_dataset(tmp_path):
    from tools.build_schema_kernel import build

    prepared = build(Path(__file__).resolve().parents[1], "test-owner/schema-audit", tmp_path)
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and not metadata["enable_gpu"]
    assert not metadata["dataset_sources"] and not metadata["model_sources"]
    assert not prepared["submitted"]
    code = (tmp_path / "audit.py").read_text()
    compile(code, "audit.py", "exec")
    assert '"--schemas-only"' in code and '"dataset_jsonl_exported":False' in code


def test_remote_extraction_packet_rejects_answers_and_escaping_paths(tmp_path):
    from tools.extract_remote import load_cases

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4")
    (tmp_path / "a.schema.json").write_text(json.dumps({"type": "object"}))
    manifest = tmp_path / "cases.json"
    case = {"id": "abc", "pdf": "a.pdf", "schema": "a.schema.json"}
    manifest.write_text(json.dumps({"version": "inference-cases/1", "cases": [case]}))
    assert load_cases(manifest)[0][0] == "abc"
    for bad in ({**case, "expected": {}}, {**case, "pdf": "../a.pdf"}, {**case, "id": "a/b"}):
        manifest.write_text(json.dumps({"version": "inference-cases/1", "cases": [bad]}))
        with pytest.raises(ValueError):
            load_cases(manifest)


def test_remote_kernels_embed_no_answers_and_fill_every_placeholder(tmp_path):
    from tools.build_extractbench_kernel import build

    config = {
        "seconds": 600,
        "scoring_reserve_seconds": 60,
        "counts": {"short": 1},
        "runs": [{"name": "a-hf", "engine": "hf", "arm": "a", "device": 0}],
        "max_window_chars": 8000,
        "max_input_tokens": 6144,
        "max_new_tokens": 1024,
        "max_model_len": 8192,
        "vllm_packages": [],
        "fla_packages": [],
    }
    build(Path(__file__).resolve().parents[1], "test-owner/extractbench-dev", tmp_path, config)
    code = (tmp_path / "extractbench.py").read_text()
    compile(code, "extractbench.py", "exec")
    assert "__DOCEXTRACT_" not in code
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] and not metadata["dataset_sources"] and not metadata["kernel_sources"]
    variants = [{"name": "cited"}, {"name": "values", "protocol": "values"}]
    sharded = config | {
        "runs": [
            {
                "name": f"a{i}",
                "group": "a",
                "engine": "vllm",
                "arm": "a",
                "device": i,
                "shard": f"{i}/2",
                "variants": variants,
            }
            for i in range(2)
        ],
        "caches_from": ["test-owner/earlier-run"],
    }
    build(Path(__file__).resolve().parents[1], "test-owner/extractbench-dev", tmp_path / "s", sharded)
    compile((tmp_path / "s" / "extractbench.py").read_text(), "extractbench.py", "exec")
    metadata = json.loads((tmp_path / "s" / "kernel-metadata.json").read_text())
    assert metadata["kernel_sources"] == ["test-owner/earlier-run"] and metadata["enable_gpu"]


def test_final_run_plan_states_worst_case_cost_and_balances_shards():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    launcher = root / "runtimes/modal/final_run.py"
    source = launcher.read_text().split("import modal  #", 1)[0]
    namespace = {"__name__": "final_run_plan"}
    exec(compile(source, str(launcher), "exec"), namespace)
    config = json.loads((root / "manifests/final-run.json").read_text())
    hourly = namespace["gpu_hourly"]()
    assert 1.1 < hourly < 1.3
    expected = config["shards"] * config["gpu_hours"] * hourly
    assert namespace["worst_case_usd"](config) >= round(expected, 2)
    groups = namespace["assign_shards"](
        [{"id": str(i), "work": work} for i, work in enumerate([90, 50, 40, 10, 5, 5])], 2
    )
    assert sorted(sum(groups, [])) == [str(i) for i in range(6)]
    assert all(group == sorted(group, key=lambda i: [90, 50, 40, 10, 5, 5][int(i)]) for group in groups)
    printed = subprocess.run(
        [sys.executable, str(launcher), str(root / "manifests/final-run.json")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(printed.stdout)["worst_case_usd"] == namespace["worst_case_usd"](config)


def test_packet_selects_a_hash_order_slice_or_exact_case_ids():
    from docextract.contracts import digest
    from tools.extractbench_packet import select

    rows = [{"id": f"doc-{n}"} for n in range(10)]
    first = select(rows, 3)
    assert len(first) == 3 and select(rows, 3, offset=3)[0] not in first
    wanted = {digest("doc-4")[:16], digest("doc-7")[:16]}
    assert sorted(row["id"] for row in select(rows, 0, ids=wanted)) == ["doc-4", "doc-7"]


def test_documents_are_shared_between_gpus_by_estimated_work(tmp_path):
    from reportlab.pdfgen.canvas import Canvas

    from tools.extract_remote import balanced_share

    cases = []
    for name, pages in (("big", 9), ("a", 1), ("b", 2), ("c", 3), ("d", 3)):
        canvas = Canvas(str(tmp_path / f"{name}.pdf"))
        for _ in range(pages):
            canvas.drawString(48, 740, name)
            canvas.showPage()
        canvas.save()
        cases.append((name, tmp_path / f"{name}.pdf", {"type": "object"}))
    first, second = balanced_share(cases, 0, 2), balanced_share(cases, 1, 2)
    assert sorted(c[0] for c in first + second) == ["a", "b", "big", "c", "d"]
    assert [c[0] for c in first] == ["big"] and [c[0] for c in second] == ["a", "b", "c", "d"]
    assert balanced_share(cases, 0, 1) == cases
