import sqlite3
import subprocess
import sys

import pytest

from docextract.engine import IncompleteGeneration, extract
from docextract.example import EXPECTED
from docextract.models import DeterministicBackend
from docextract.runtime import exclusive_workspace


class CountingBackend(DeterministicBackend):
    def __init__(self):
        self.calls = 0

    def generate(self, task):
        self.calls += 1
        return super().generate(task)


@pytest.mark.parametrize("stage", ["generated", "raw_saved", "validated", "committed"])
def test_interruption_at_every_durable_boundary(authored, tmp_path, stage):
    pdf, schema, plan = authored
    backend = CountingBackend()
    raised = False

    def interrupt(actual, task):
        nonlocal raised
        if not raised and actual == stage and task.record and task.record.schema_pointer.endswith("/items"):
            raised = True
            raise InterruptedError("injected process interruption")

    with pytest.raises(InterruptedError):
        extract(
            pdf,
            schema,
            workspace=tmp_path,
            plan=plan,
            backend=backend,
            apply_defaults=True,
            window_size=2,
            overlap=1,
            fault=interrupt,
        )
    result = extract(
        pdf,
        schema,
        workspace=tmp_path,
        plan=plan,
        backend=backend,
        apply_defaults=True,
        window_size=2,
        overlap=1,
    )
    assert result.data == EXPECTED
    assert len(result.source_records) == 6
    assert backend.calls == (8 if stage == "generated" else 7)
    assert result.usage.generated_attempts == backend.calls


def test_overlap_and_successful_retry_reuse_committed_occurrences(authored, tmp_path):
    pdf, schema, plan = authored
    backend = CountingBackend()
    first = extract(
        pdf,
        schema,
        workspace=tmp_path,
        plan=plan,
        backend=backend,
        apply_defaults=True,
        window_size=3,
        overlap=2,
    )
    second = extract(
        pdf,
        schema,
        workspace=tmp_path,
        plan=plan,
        backend=backend,
        apply_defaults=True,
        window_size=2,
        overlap=1,
    )
    assert backend.calls == 7
    assert first == second
    assert second.usage.committed_tasks == 7


def test_os_exit_after_raw_save_recovers_without_generation(authored, tmp_path):
    pdf, schema, plan = authored
    from docextract.contracts import save_json

    save_json(tmp_path / "schema.json", schema)
    save_json(tmp_path / "plan.json", plan)
    script = """
import json, os, sys
from pathlib import Path
from docextract.engine import extract
from docextract.planning import BindingPlan
pdf, root = map(Path, sys.argv[1:])
def interrupt(stage, task):
    if stage == 'raw_saved': os._exit(23)
extract(pdf, json.loads((root/'schema.json').read_text()), workspace=root/'run',
        plan=BindingPlan.model_validate_json((root/'plan.json').read_text()),
        apply_defaults=True, fault=interrupt)
"""
    process = subprocess.run([sys.executable, "-c", script, str(pdf), str(tmp_path)], check=False)
    assert process.returncode == 23
    backend = CountingBackend()
    result = extract(pdf, schema, workspace=tmp_path / "run", plan=plan, backend=backend, apply_defaults=True)
    assert result.data == EXPECTED
    assert backend.calls == 6
    assert result.usage.generated_attempts == 7


def test_truncated_generation_is_saved_but_never_committed(authored, tmp_path):
    class Truncated(DeterministicBackend):
        def generate(self, task):
            return super().generate(task).model_copy(update={"finish_reason": "length"})

    pdf, schema, plan = authored
    with pytest.raises(IncompleteGeneration):
        extract(pdf, schema, workspace=tmp_path, plan=plan, backend=Truncated())
    with sqlite3.connect(tmp_path / "journal.sqlite") as db:
        state, raw, output = db.execute("SELECT state,raw,output FROM tasks").fetchone()
    assert state == "raw_saved" and raw is not None and output is None
    assert not (tmp_path / "result.json").exists()


def test_workspace_lock_does_not_reclaim_live_workers(authored, tmp_path):
    pdf, schema, plan = authored
    with exclusive_workspace(tmp_path), pytest.raises(RuntimeError, match="Another worker"):
        extract(pdf, schema, workspace=tmp_path, plan=plan)


def test_unknown_source_reference_is_not_committed(authored, tmp_path):
    class Unbound(DeterministicBackend):
        def generate(self, task):
            import json

            generation = super().generate(task)
            payload = json.loads(generation.raw_text)
            payload["fields"][0]["source_atom_ids"] = ["invented"]
            return generation.model_copy(update={"raw_text": json.dumps(payload)})

    pdf, schema, plan = authored
    with pytest.raises(ValueError, match="unbound"):
        extract(pdf, schema, workspace=tmp_path, plan=plan, backend=Unbound())
