from __future__ import annotations

from copy import deepcopy
from typing import Any

from jsonschema import Draft7Validator, Draft202012Validator, FormatChecker

from docextract.contracts import Contract, ValidationReport

ANNOTATIONS = {
    "$schema",
    "$id",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
}
CONSTRAINTS = {
    "enum",
    "const",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "required",
    "additionalProperties",
}
STRUCTURE = {"type", "properties", "items", "$defs", "$ref", "anyOf", "oneOf"}
DIALECTS = {
    None: Draft202012Validator,
    "https://json-schema.org/draft/2020-12/schema": Draft202012Validator,
    "http://json-schema.org/draft-07/schema#": Draft7Validator,
}


def escape(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def parts(pointer: str) -> list[str]:
    if not pointer:
        return []
    if not pointer.startswith("/"):
        raise ValueError(f"Expected JSON Pointer: {pointer}")
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def join(pointer: str, key: str | int) -> str:
    return f"{pointer}/{escape(str(key))}"


class SchemaSupportReport(Contract):
    supported: bool
    issues: tuple[str, ...]
    profile: str = "docextract-schema/2"


class UnsupportedSchema(ValueError):
    def __init__(self, report: SchemaSupportReport):
        self.report = report
        super().__init__("Unsupported schema: " + "; ".join(report.issues))


class Schema:
    def __init__(self, original: dict):
        if not isinstance(original, dict):
            raise UnsupportedSchema(
                SchemaSupportReport(supported=False, issues=("/: object schema required",))
            )
        self.validator_class = DIALECTS.get(original.get("$schema"), Draft202012Validator)
        self.validator_class.check_schema(original)
        self.original = deepcopy(original)
        self.support = self._support()

    def require_supported(self) -> None:
        if not self.support.supported:
            raise UnsupportedSchema(self.support)

    def resolve(self, node: dict) -> dict:
        node = deepcopy(node)
        if "$ref" in node:
            target: Any = self.original
            reference = node.pop("$ref")
            if not reference.startswith("#/$defs/"):
                raise ValueError(f"Only local $defs references are supported: {reference}")
            for part in parts(reference[1:]):
                target = target[part]
            node = self.resolve(target) | node
        for keyword in ("anyOf", "oneOf"):
            if keyword in node:
                branches = node.pop(keyword)
                non_null = [branch for branch in branches if branch != {"type": "null"}]
                if len(branches) != 2 or len(non_null) != 1:
                    raise ValueError("Only a single non-null branch plus null is supported")
                branch = self.resolve(non_null[0])
                types = branch.get("type")
                branch["type"] = [types, "null"] if isinstance(types, str) else list(types)
                node = branch | node
        return node

    def node(self, pointer: str) -> dict:
        """Resolved node at a schema pointer; cached, so callers must treat it as read-only."""
        cache = self.__dict__.setdefault("_nodes", {})
        if pointer not in cache:
            segments = parts(pointer)
            if not segments:
                cache[pointer] = self.resolve(self.original)
            else:
                parent = self.node("".join("/" + escape(part) for part in segments[:-1]))
                key = segments[-1]
                child = parent["items"] if self.kind(parent) == "array" else parent["properties"][key]
                cache[pointer] = self.resolve(child)
        return cache[pointer]

    def kind(self, node: dict) -> str:
        node = self.resolve(node)
        types = node.get("type")
        if isinstance(types, list):
            return next((item for item in types if item != "null"), "null")
        return types

    def validate(self, data: dict) -> ValidationReport:
        self.require_supported()
        validator = self.validator_class(self.original, format_checker=FormatChecker())
        errors = tuple(
            f"/{'/'.join(escape(str(part)) for part in error.absolute_path)}: {error.message}"
            for error in sorted(validator.iter_errors(data), key=lambda error: str(list(error.absolute_path)))
        )
        return ValidationReport(valid=not errors, errors=errors)

    def _support(self) -> SchemaSupportReport:
        issues: list[str] = []
        if self.original.get("$schema") not in DIALECTS:
            issues.append("/$schema: supported dialects are JSON Schema 2020-12 and Draft 7")

        def visit(node: Any, path: str, ancestors: tuple[str, ...] = ()) -> None:
            if not isinstance(node, dict):
                issues.append(f"{path}: boolean/dynamic schemas are unsupported")
                return
            if path and "$schema" in node and node["$schema"] != self.original.get("$schema"):
                issues.append(f"{path}: nested dialect changes are unsupported")
            for keyword in node.keys() - ANNOTATIONS - CONSTRAINTS - STRUCTURE:
                if not keyword.startswith("x-"):
                    issues.append(f"{join(path, keyword)}: unsupported keyword")
            reference = node.get("$ref")
            if reference:
                if reference in ancestors:
                    issues.append(f"{path}: recursive reference {reference}")
                    return
                if not reference.startswith("#/$defs/"):
                    issues.append(f"{path}: external/non-$defs reference {reference}")
                    return
                if node.keys() - ANNOTATIONS - {"$ref"}:
                    issues.append(f"{path}: constraint siblings of $ref are unsupported")
                try:
                    target = self.original
                    for part in parts(reference[1:]):
                        target = target[part]
                    visit(target, path, (*ancestors, reference))
                except (KeyError, TypeError):
                    issues.append(f"{path}: unresolved reference {reference}")
                return
            for keyword in ("anyOf", "oneOf"):
                if keyword in node:
                    if node.keys() - ANNOTATIONS - {keyword, "$defs"}:
                        issues.append(f"{path}: constraint siblings of nullable union are unsupported")
                    branches = node[keyword]
                    non_null = [branch for branch in branches if branch != {"type": "null"}]
                    if len(branches) != 2 or len(non_null) != 1:
                        issues.append(f"{join(path, keyword)}: only nullable unions are supported")
                        return
                    visit(non_null[0], join(path, keyword), ancestors)
                    return
            types = node.get("type")
            if isinstance(types, list) and (len(types) != 2 or "null" not in types):
                issues.append(f"{path}: only nullable type unions are supported")
            if not types:
                issues.append(f"{path}: explicit type is required")
            if isinstance(node.get("additionalProperties"), dict):
                issues.append(f"{path}: dynamic additionalProperties are unsupported")
            if node.get("format") not in (None, *FormatChecker.checkers):
                issues.append(f"{path}: unknown format {node['format']}")
            for key, child in node.get("properties", {}).items():
                visit(child, join(join(path, "properties"), key), ancestors)
            if "array" in ([types] if isinstance(types, str) else types or []):
                if "items" not in node:
                    issues.append(f"{path}: homogeneous items schema is required")
                else:
                    visit(node["items"], join(path, "items"), ancestors)

        visit(self.original, "")
        if not issues and self.kind(self.original) != "object":
            issues.append("/: root must be an object")
        return SchemaSupportReport(supported=not issues, issues=tuple(issues))
