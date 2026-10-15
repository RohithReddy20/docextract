from __future__ import annotations

import fcntl
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from docextract.contracts import Contract
from docextract.models import GenerationResult


@contextmanager
def exclusive_workspace(workspace: Path):
    """A live worker cannot have its running tasks reclaimed by a second process."""
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / ".worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this workspace") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Journal:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS tasks (
            run_id TEXT NOT NULL, task_id TEXT NOT NULL, state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, raw TEXT, output TEXT,
            PRIMARY KEY (run_id, task_id))""")
        self.db.commit()

    def close(self):
        self.db.close()

    def ensure(self, run_id: str, task_id: str):
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO tasks(run_id,task_id,state) VALUES(?,?,'pending')", (run_id, task_id)
            )

    def get(self, run_id: str, task_id: str) -> tuple[str, str | None, str | None]:
        row = self.db.execute(
            "SELECT state,raw,output FROM tasks WHERE run_id=? AND task_id=?", (run_id, task_id)
        ).fetchone()
        return row

    def start(self, run_id: str, task_id: str):
        with self.db:
            self.db.execute(
                "UPDATE tasks SET state='running',attempts=attempts+1 WHERE run_id=? AND task_id=?",
                (run_id, task_id),
            )

    def release(self, run_id: str, task_id: str):
        """Undo start() when the backend deferred the task without generating anything."""
        with self.db:
            self.db.execute(
                "UPDATE tasks SET state='pending',attempts=attempts-1 WHERE run_id=? AND task_id=? "
                "AND state='running'",
                (run_id, task_id),
            )

    def save_raw(self, run_id: str, task_id: str, generation: GenerationResult):
        with self.db:
            self.db.execute(
                "UPDATE tasks SET state='raw_saved',raw=? WHERE run_id=? AND task_id=?",
                (generation.model_dump_json(), run_id, task_id),
            )

    def validated(self, run_id: str, task_id: str):
        with self.db:
            self.db.execute(
                "UPDATE tasks SET state='validated' WHERE run_id=? AND task_id=?", (run_id, task_id)
            )

    def commit(self, run_id: str, task_id: str, output: Contract):
        with self.db:
            self.db.execute(
                "UPDATE tasks SET output=?,state='committed' WHERE run_id=? AND task_id=?",
                (output.model_dump_json(), run_id, task_id),
            )

    def usage(self, run_id: str) -> dict:
        rows = self.db.execute("SELECT attempts,state,raw FROM tasks WHERE run_id=?", (run_id,)).fetchall()
        generations = [json.loads(raw) for _, _, raw in rows if raw]
        return dict(
            generated_attempts=sum(attempts for attempts, _, _ in rows),
            committed_tasks=sum(state == "committed" for _, state, _ in rows),
            input_tokens=sum(raw["input_tokens"] for raw in generations),
            output_tokens=sum(raw["output_tokens"] for raw in generations),
        )
