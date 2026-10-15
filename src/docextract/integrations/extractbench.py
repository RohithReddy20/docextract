from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from extract_bench.inference.pipelines import register_pipeline
from extract_bench.inference.providers.base import Provider, ProviderConfigError
from extract_bench.inference.providers.registry import register_provider
from extract_bench.schemas.extract_output import ExtractOutput, FieldCitation
from extract_bench.schemas.pipeline import PipelineSpec
from extract_bench.schemas.pipeline_io import InferenceRequest, InferenceResult, RawInferenceResult
from extract_bench.schemas.product import ProductType

from docextract.contracts import ExtractionResult, digest
from docextract.engine import extract
from docextract.geometry import envelope, normalized_coco
from docextract.planning import BindingPlan
from docextract.schema import parts

EXTRACTBENCH_COMMIT = "0e7647997211867060a645ac82deb9f7bc4a3b3f"
DATASET_REVISION = "51bf3a03eb7feeaef8952be1fc2fc7f924ee72dc"


def benchmark_path(pointer: str, data: dict) -> str:
    path = ""
    current = data
    for part in parts(pointer):
        if isinstance(current, list):
            index = int(part)
            path += f"[{index}]"
            current = current[index]
        else:
            if not part or any(char in part for char in ".[]"):
                raise ValueError(f"Property cannot be represented in ExtractBench's path grammar: {part!r}")
            path += ("." if path else "") + part
            current = current[part]
    return path


def project_result(result: ExtractionResult, *, example_id: str = "offline", pipeline_name="docextract_b0"):
    """Pure projection of an immutable saved result. No PDF, schema, model or I/O access."""
    atoms = result.snapshot.atom_index()
    char_lines = {}
    for page in result.snapshot.pages:
        for line in (atom for atom in page.atoms if atom.kind == "line"):
            for token_id in line.child_ids:
                for char_id in atoms[token_id].child_ids:
                    char_lines[char_id] = line.atom_id
    citations = []
    for field in result.field_evidence:
        if field.status not in ("observed", "computed"):
            continue
        path = benchmark_path(field.pointer, result.data)
        groups = defaultdict(list)
        for atom_id in field.source_atom_ids:
            atom = atoms[atom_id]
            groups[(atom.page_index, char_lines.get(atom_id, atom_id))].append(atom)
        for (page_index, line_id), source_atoms in groups.items():
            page = result.snapshot.pages[page_index]
            token_text = []
            selected = {atom.atom_id for atom in source_atoms}
            line = atoms[line_id]
            for token_id in line.child_ids if line.kind == "line" else ():
                token = atoms[token_id]
                text = "".join(atoms[char_id].text for char_id in token.child_ids if char_id in selected)
                if text:
                    token_text.append(text)
            citations.append(
                FieldCitation(
                    field_path=path,
                    page=page_index + 1,
                    bbox=normalized_coco(
                        envelope([atom.polygon for atom in source_atoms]), page.width, page.height
                    ),
                    reference_text=" ".join(token_text) or line.text,
                    source=source_atoms[0].source_channel,
                    metadata={
                        "support": field.support,
                        "source_atom_ids": [atom.atom_id for atom in source_atoms],
                    },
                )
            )
    return ExtractOutput(
        example_id=example_id,
        pipeline_name=pipeline_name,
        extracted_data=deepcopy(result.data),
        field_citations=citations,
    )


class DocextractProvider(Provider):
    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        if request.schema_override is None:
            raise ProviderConfigError("An original schema_override is required; no evaluator sidecar reads")
        if request.input_files:
            raise ProviderConfigError("B0 accepts one PDF, not multi-file amendments")
        config = self.base_config | pipeline.config | (request.config_override or {})
        if "bindings" not in config:
            raise ProviderConfigError("B0 requires declared bindings; it is not a general model baseline")
        started = datetime.now(UTC)
        workspace = Path(config.get("workspace", "artifacts/extractbench")) / digest(request.example_id)
        result = extract(
            Path(request.source_file_path),
            request.schema_override,
            workspace=workspace,
            plan=BindingPlan.model_validate(config["bindings"]),
            apply_defaults=bool(config.get("apply_defaults", False)),
        )
        completed = datetime.now(UTC)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type="extract",
            raw_output={"extraction_result": result.model_dump(mode="json"), "cost_usd": 0.0},
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )

    def normalize(self, raw_result: RawInferenceResult) -> InferenceResult:
        result = ExtractionResult.model_validate(raw_result.raw_output["extraction_result"])
        output = project_result(
            result, example_id=raw_result.request.example_id, pipeline_name=raw_result.pipeline_name
        )
        return InferenceResult(
            request=raw_result.request,
            pipeline_name=raw_result.pipeline_name,
            product_type=raw_result.product_type,
            raw_output=deepcopy(raw_result.raw_output),
            output=output,
            started_at=raw_result.started_at,
            completed_at=raw_result.completed_at,
            latency_in_ms=raw_result.latency_in_ms,
        )


class DocextractModelProvider(DocextractProvider):
    """Optional learned trial provider; normalization remains inherited pure projection."""

    def run_inference(self, pipeline: PipelineSpec, request: InferenceRequest) -> RawInferenceResult:
        from docextract.learned import extract_learned
        from docextract.remote import GenerationOptions, KaggleTransformersBackend, require_kaggle

        require_kaggle()
        if request.schema_override is None or request.input_files:
            raise ProviderConfigError("Learned trial requires one PDF and the original schema_override")
        config = self.base_config | pipeline.config | (request.config_override or {})
        if "bindings" in config or not config.get("model_manifest") or config.get("arm") not in ("a", "b"):
            raise ProviderConfigError(
                "Use arm A/B and a pinned model_manifest; manual bindings are forbidden"
            )
        identity = {
            key: config.get(key)
            for key in ("arm", "model_manifest", "device", "max_input_tokens", "max_new_tokens", "cache_only")
        }
        started = datetime.now(UTC)
        if hasattr(self, "_learned_identity") and self._learned_identity != identity:
            raise ProviderConfigError("Use a fresh worker process when changing model/runtime configuration")
        if not hasattr(self, "_learned_backend"):
            self._learned_backend = KaggleTransformersBackend(
                config["arm"],
                manifest=Path(config["model_manifest"]),
                device=config.get("device", 0),
                options=GenerationOptions(
                    max_input_tokens=config.get("max_input_tokens", 6144),
                    max_new_tokens=config.get("max_new_tokens", 1536),
                ),
                offline=bool(config.get("cache_only", False)),
            )
            self._learned_identity = identity
        result = extract_learned(
            Path(request.source_file_path),
            request.schema_override,
            workspace=Path(config.get("workspace", "artifacts/model-extractbench"))
            / digest(request.example_id),
            backend=self._learned_backend,
            max_window_chars=config.get("max_window_chars", 6000),
            overlap_lines=config.get("overlap_lines", 2),
        )
        completed = datetime.now(UTC)
        return RawInferenceResult(
            request=request,
            pipeline=pipeline,
            pipeline_name=pipeline.pipeline_name,
            product_type="extract",
            raw_output={
                "extraction_result": result.model_dump(mode="json"),
                "cost_usd": None,
                "backend_identity": self._learned_backend.identity,
            },
            started_at=started,
            completed_at=completed,
            latency_in_ms=int((completed - started).total_seconds() * 1000),
        )


def register() -> None:
    """Call once in an ExtractBench runner process before selecting docextract_b0."""
    register_provider("docextract")(DocextractProvider)
    register_provider("docextract_model_trial")(DocextractModelProvider)
    register_pipeline(
        PipelineSpec(
            pipeline_name="docextract_b0",
            provider_name="docextract",
            product_type=ProductType.EXTRACT,
            config={},
        )
    )
    register_pipeline(
        PipelineSpec(
            pipeline_name="docextract_model_trial",
            provider_name="docextract_model_trial",
            product_type=ProductType.EXTRACT,
            config={},
        )
    )
