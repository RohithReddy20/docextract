"""Cited extraction over many documents with one continuously batched generation engine.

Each document is first run against the store of saved answers: windows without a stored answer are
queued instead of generated. Queued windows stream through the engine; every answer is stored the
moment it finishes, and a document is assembled again as soon as its last outstanding window is
answered (subdivided windows are queued then). A run stopped at a deadline keeps every finished
answer, and a later replay can assemble partial documents from them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

from docextract.cited import CitedTask, Pending, extract_cited
from docextract.contracts import ExtractionResult, ParseSnapshot, save_json
from docextract.models import GenerationResult
from docextract.parsing.pdfium import PdfiumParser


class BatchEngine(Protocol):
    identity: dict[str, Any]

    def generate_batch(self, tasks: list[CitedTask]) -> list[GenerationResult]: ...


class StreamingEngine(Protocol):
    identity: dict[str, Any]

    def submit(self, tasks: list[CitedTask], first: bool = False) -> None:
        """Queue `tasks`, ahead of those already waiting when `first`."""
        ...

    def poll(self) -> list[tuple[str, GenerationResult]]:
        """Advance the engine and return the answers that finished, by task id."""
        ...

    def busy(self) -> bool: ...


class BatchedStream:
    """Streaming facade over an engine that answers whole lists of tasks at a time."""

    def __init__(self, engine: BatchEngine, batch_size: int):
        self.engine, self.batch_size = engine, batch_size
        self.identity = engine.identity
        self.queue: list[CitedTask] = []

    def submit(self, tasks: list[CitedTask], first: bool = False) -> None:
        self.queue = [*tasks, *self.queue] if first else [*self.queue, *tasks]

    def busy(self) -> bool:
        return bool(self.queue)

    def poll(self) -> list[tuple[str, GenerationResult]]:
        group, self.queue = self.queue[: self.batch_size], self.queue[self.batch_size :]
        if not group:
            return []
        answers = self.engine.generate_batch(group)
        return [(task.task_id, answer) for task, answer in zip(group, answers, strict=True)]


class GenerationStore:
    """Backend facade over durable stored generations; a missing task is queued, never generated.

    `legacy` names engines that differed from this one only by a lower output limit or by how many
    requests they ran at once: an answer stored for a task's alias under one of them stands for the
    task unless a lower limit cut it off (the same greedy answer, which a higher limit lets finish). An
    answer cut off at the task's own lower limit stands too, and a refusal stands since a higher output
    limit leaves less room for input. Task aliases follow the order of `legacy`.
    """

    def __init__(self, identity: dict[str, Any], path: Path, legacy: tuple[dict[str, Any], ...] = ()):
        self.identity = identity
        self.legacy = tuple(legacy)
        self.limit = identity.get("generation", {}).get("max_new_tokens")
        self.legacy_limits = tuple(other["generation"]["max_new_tokens"] for other in self.legacy)
        self.path = path
        self.saved: dict[str, GenerationResult] = {}
        self.queued: dict[str, CitedTask] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.saved[row["task_id"]] = GenerationResult.model_validate(row["generation"])

    def generate_cited(self, task: CitedTask) -> GenerationResult:
        if task.task_id in self.saved:
            return self.saved[task.task_id]
        for alias, limit in zip(task.aliases, self.legacy_limits, strict=False):
            answer = self.saved.get(alias)
            if answer is not None and (
                answer.finish_reason != "length"
                or (self.limit is not None and limit >= self.limit)
                or (task.max_tokens is not None and task.max_tokens <= limit)
            ):
                return answer
        self.queued[task.task_id] = task
        raise Pending([task.task_id])

    def store(self, task_id: str, generation: GenerationResult) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as handle:
            handle.write(json.dumps({"task_id": task_id, "generation": generation.model_dump()}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.saved[task_id] = generation


class CachedParser:
    """Parse each document once; with a store, snapshots persist by document hash so a replay
    re-assembles exactly the evidence (including OCR output) the stored answers were given."""

    def __init__(self, parser=None, store: Path | None = None, parse_missing=True):
        self.parser = parser or PdfiumParser()
        self.store = store
        self.parse_missing = parse_missing
        self.cache: dict[Path, Any] = {}

    def parse(self, document: Path):
        if document in self.cache:
            return self.cache[document]
        path = None
        if self.store is not None:
            path = self.store / (hashlib.sha256(document.read_bytes()).hexdigest() + ".json")
            if path.exists():
                self.cache[document] = ParseSnapshot.model_validate_json(path.read_text())
                return self.cache[document]
        if not self.parse_missing:
            raise FileNotFoundError("No stored snapshot for this document")
        snapshot = self.parser.parse(document)
        if path is not None:
            save_json(path, snapshot)
        self.cache[document] = snapshot
        return snapshot


def adjacent_prompts(tasks: list[CitedTask]) -> list[CitedTask]:
    """Order windows so those sharing a schema field group (a shared prompt prefix) are adjacent,
    which lets an engine with prefix caching reuse it; window order within a group is kept."""
    first: dict[str, int] = {}
    keys = [json.dumps(task.schema_part, sort_keys=True) if task.schema_part else "" for task in tasks]
    for key in keys:
        first.setdefault(key, len(first))
    order = sorted(range(len(tasks)), key=lambda index: first[keys[index]])
    return [tasks[index] for index in order]


def run_batch(
    cases: list[tuple[str, Path, dict] | tuple[str, Path, dict, dict]],
    *,
    root: Path,
    engine: BatchEngine | StreamingEngine,
    passes: int = 48,
    batch_size: int = 64,
    on_result=None,
    parser=None,
    legacy_identities: tuple[dict[str, Any], ...] = (),
    **options,
) -> dict[str, ExtractionResult | str]:
    store = GenerationStore(engine.identity, root / "generations.jsonl", legacy_identities)
    parser = parser if isinstance(parser, CachedParser) else CachedParser(parser)
    stream = engine if hasattr(engine, "submit") else BatchedStream(engine, batch_size)
    # A job is a document under its own options (for example one prompt variant); its id is also its
    # workspace path below `root`, so variants of one document stay apart.
    documents = {job[0]: (job[1], job[2], job[3] if len(job) > 3 else {}) for job in cases}
    outcomes: dict[str, ExtractionResult | str] = {}
    waiting: dict[str, set[str]] = {}
    attempts: dict[str, int] = {}
    in_flight: set[str] = set()

    def attempt(case_id) -> list[CitedTask]:
        """Assemble one document from stored answers; return the windows it still needs."""
        attempts[case_id] = attempts.get(case_id, 0) + 1
        pdf, schema, own = documents[case_id]
        store.queued.clear()
        try:
            outcome = extract_cited(
                pdf, schema, workspace=root / case_id, backend=store, parser=parser, **(options | own)
            )
        except Pending:
            if attempts[case_id] <= passes:
                waiting[case_id] = set(store.queued)
                fresh = [task for task in store.queued.values() if task.task_id not in in_flight]
                in_flight.update(task.task_id for task in fresh)
                return adjacent_prompts(fresh)
            outcome = "generation passes exhausted"
        except Exception as error:  # one failed document stays in the population as a failure
            outcome = f"{type(error).__name__}: {str(error)[:400]}"
        outcomes[case_id] = outcome
        if on_result:
            on_result(case_id, outcome)
        return []

    # Documents are queued in order, so a first-come engine finishes them roughly in order.
    stream.submit([task for case_id in documents for task in attempt(case_id)])
    # How much of a resumed run its stored answers covered.
    print(
        f"first pass: {len(outcomes)} of {len(documents)} documents from stored answers, "
        f"{len(in_flight)} requests queued for the other {len(waiting)}",
        flush=True,
    )
    while waiting:
        answers = stream.poll()
        if not answers and not stream.busy():
            break  # nothing is generating, so no waiting document can progress
        ready = []
        for task_id, generation in answers:
            store.store(task_id, generation)
            in_flight.discard(task_id)
            for case_id, outstanding in waiting.items():
                outstanding.discard(task_id)
                if not outstanding and case_id not in ready:
                    ready.append(case_id)
        for case_id in ready:
            del waiting[case_id]
            # A document's next requests go ahead of documents not yet begun: the long ones read on
            # a few pages at a time, one request after another, and queued last they kept a run going
            # with a few requests at once after the rest had finished.
            stream.submit(attempt(case_id), first=True)
    for case_id in documents:
        outcomes.setdefault(case_id, "generation passes exhausted")
    return outcomes


class ReplayEngine:
    """Answers only from a stored generation log; a task without a stored answer stays unanswered."""

    def __init__(self, identity: dict[str, Any]):
        self.identity = identity
        self.telemetry: list[dict] = []
        self.missing = 0

    def generate_batch(self, tasks: list[CitedTask]) -> list[GenerationResult]:
        self.missing += len(tasks)
        return [GenerationResult(raw_text="", finish_reason="not_stored") for _ in tasks]
