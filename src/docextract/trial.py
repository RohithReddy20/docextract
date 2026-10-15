"""Value-only controlled trial; it never claims to provide grounded extraction."""

import hashlib
from contextlib import closing
from pathlib import Path
from typing import Any

from docextract.contracts import Contract, ValidationReport, digest, save_json
from docextract.interpretation import evidence_units, strict_json
from docextract.models import GenerationResult
from docextract.parsing.pdfium import PdfiumParser
from docextract.runtime import Journal, exclusive_workspace
from docextract.schema import Schema


class ValueTrialResult(Contract):
    contract_version: str = "value-trial/1"
    run_id: str
    snapshot_id: str
    data: dict[str, Any] | None
    validation: ValidationReport | None
    generation: GenerationResult
    error: str | None = None
    citations_measured: bool = False


def run_value_trial(pdf: Path, schema: dict, *, workspace: Path, backend) -> ValueTrialResult:
    compiled = Schema(schema)
    compiled.require_supported()
    with exclusive_workspace(workspace):
        snapshot = PdfiumParser().parse(pdf)
        run_id = digest(
            {
                "snapshot": snapshot.snapshot_id,
                "schema": schema,
                "backend": backend.identity,
                "trial": "values/1",
                "implementation": {
                    name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ("trial.py", "jsonutil.py", "schema.py", "interpretation.py")
                },
            }
        )
        save_json(workspace / run_id / "snapshot.json", snapshot)
        with closing(Journal(workspace / "values.sqlite")) as journal:
            journal.ensure(run_id, "values")
            state, raw, output = journal.get(run_id, "values")
            if state == "committed":
                result = ValueTrialResult.model_validate_json(output)
            else:
                if raw is None:
                    journal.start(run_id, "values")
                    generation = backend.generate_values(schema, evidence_units(snapshot))
                    journal.save_raw(run_id, "values", generation)
                else:
                    generation = GenerationResult.model_validate_json(raw)
                data, validation, error = None, None, None
                if generation.finish_reason != "stop":
                    error = "output_limit: no prefix salvage"
                else:
                    try:
                        data = strict_json(generation.raw_text)
                        if not isinstance(data, dict):
                            raise ValueError("Expected an object root")
                        validation = compiled.validate(data)
                    except ValueError as failure:
                        data, error = None, str(failure)
                result = ValueTrialResult(
                    run_id=run_id,
                    snapshot_id=snapshot.snapshot_id,
                    data=data,
                    validation=validation,
                    generation=generation,
                    error=error,
                )
                journal.validated(run_id, "values")
                journal.commit(run_id, "values", result)
        save_json(workspace / "value-result.json", result)
        return result
