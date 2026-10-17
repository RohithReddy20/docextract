"""Build the requests of one extraction probe (see runtimes/modal/probe.py).

A probe configuration names a packet, a few document ids and the methods to compare. Every method
starts from ExtractBench's one-shot recipe for self-hosted vLLM models (its own prompt text and schema
preparation, imported from the pinned package) and overrides only what it lists. Schemas are read
from `artifacts/probes/data/<id>.schema.json` (inference inputs; fetch them first with
tools/probe_data.sh). Output: `artifacts/probes/<name>/jobs.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_bench.inference.providers.extract.direct_model_utils import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_INSTRUCTION,
    add_additional_properties_false,
    promote_repeated_structure,
)

ROOT = Path(__file__).resolve().parents[1]


def user_text(schema: dict, instruction: str) -> str:
    """The text part of ExtractBench's vLLM one-shot request (vllm_extract._build_user_content)."""
    return (
        "Extract every field from the attached document page image(s) according to the JSON schema below.\n\n"
        "JSON schema:\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "Return only one valid JSON object matching the schema. Do not wrap it in markdown fences.\n\n"
        f"{instruction}"
    )


def build(config: dict, data: Path) -> list[dict]:
    jobs = []
    for case_id in config["ids"]:
        schema = json.loads((data / f"{case_id}.schema.json").read_text())
        prepared = add_additional_properties_false(promote_repeated_structure(schema))
        for method in config["methods"]:
            if method.get("ids") and case_id not in method["ids"]:
                continue  # a method may be limited to some of the documents
            jobs.append(
                {
                    "id": case_id,
                    "variant": method["name"],
                    "pdf": f"packets/{config['packet']}/inference/{case_id}.pdf",
                    "system": method.get("system", DEFAULT_SYSTEM_PROMPT),
                    "user_text": user_text(
                        prepared, method.get("user_instruction", DEFAULT_USER_INSTRUCTION)
                    ),
                    "dpi": method.get("dpi", 150),
                    "json_object": method.get("json_object", True),
                    "max_tokens": method.get("max_tokens", 12288),
                    # JSON without free whitespace: the same content, but no runaway blank output.
                    "compact_json": method.get("compact_json", True),
                    "text_layer": method.get("text_layer", False),
                    "pages_per_request": method.get("pages_per_request"),
                    "template_options": method.get("template_options", {}),
                }
            )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    jobs = build(config, ROOT / "artifacts" / "probes" / "data")
    out = ROOT / "artifacts" / "probes" / config["name"]
    out.mkdir(parents=True, exist_ok=True)
    batch = {"engine": config["engine"], "limit_minutes": config.get("limit_minutes", 15), "jobs": jobs}
    (out / "jobs.json").write_text(json.dumps(batch))
    print(f"{len(jobs)} requests -> {out / 'jobs.json'}")


if __name__ == "__main__":
    main()
