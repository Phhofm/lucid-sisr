"""SQLite-based state management for resumable pipeline execution."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class PipelineState:
    """Provides a thread-safe SQLite state DB for tracking pipeline progress."""

    def __init__(self, workspace: Path, db_name: str = "lucid_state.db"):
        self.db_path = workspace / ".lucid" / db_name
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                timeout=60.0,
                isolation_level=None,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()

    @contextmanager
    def get_cursor(self) -> Iterator[sqlite3.Cursor]:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT UNIQUE NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error_msg TEXT,
                image_width INTEGER,
                image_height INTEGER,
                total_tiles INTEGER DEFAULT 0,
                kept_tiles INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
            CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(source_path);
            CREATE TABLE IF NOT EXISTS run_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    def record_run_config(self, config: dict[str, Any]) -> None:
        """Persist the filtering gates this run is using."""
        assert self._conn is not None
        with self.get_cursor() as cur:
            cur.executemany(
                "INSERT OR REPLACE INTO run_config(key, value) VALUES (?, ?)",
                sorted(
                    (k, json.dumps(v, sort_keys=True)) for k, v in config.items()
                ),
            )

    def check_run_config(self, config: dict[str, Any]) -> None:
        """Refuse a resume whose recorded gates differ from the current ones.

        Half a dataset filtered with old gates and half with new ones is
        silent dataset corruption, so mismatch is fatal.
        """
        assert self._conn is not None
        with self.get_cursor() as cur:
            rows = dict(cur.execute("SELECT key, value FROM run_config").fetchall())
        if not rows:
            return  # legacy DB without a recorded config
        current = {k: json.dumps(v, sort_keys=True) for k, v in config.items()}
        changed = sorted(k for k in current if rows.get(k) != current[k])
        if changed:
            prev = {k: rows.get(k) for k in changed}
            raise RuntimeError(
                f"Cannot --resume: pipeline gates changed since the recorded "
                f"run: {changed} (recorded: {prev}). Re-run without --resume, "
                f"or delete the workspace to start fresh."
            )

    def register_source(self, source: Path, width: int | None = None, height: int | None = None) -> int:
        """Register a source image and return its row id."""
        with self.get_cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO sources (source_path, image_width, image_height) VALUES (?,?,?)",
                (str(source), width, height),
            )
            cur.execute("SELECT id FROM sources WHERE source_path = ?", (str(source),))
            return cur.fetchone()[0]

    def mark_source_status(self, source: Path, status: str, error_msg: str | None = None) -> None:
        with self.get_cursor() as cur:
            cur.execute(
                "UPDATE sources SET status = ?, error_msg = ?, last_updated = CURRENT_TIMESTAMP WHERE source_path = ?",
                (status, error_msg, str(source)),
            )

    def is_source_done(self, source: Path) -> bool:
        with self.get_cursor() as cur:
            cur.execute("SELECT status FROM sources WHERE source_path = ?", (str(source),))
            row = cur.fetchone()
            return row is not None and row[0] in ("scored", "finalized", "skipped")

    def get_pending_sources(self, all_sources: list[Path]) -> list[Path]:
        """Filter a list of source paths to only those still needing processing.

        Sources that are already 'extracted', 'scored', or 'skipped' are excluded.
        """
        if not all_sources:
            return []
        placeholders = ",".join("?" * len(all_sources))
        with self.get_cursor() as cur:
            cur.execute(
                f"SELECT source_path FROM sources WHERE status IN ('extracted', 'scored', 'skipped') AND source_path IN ({placeholders})",
                [str(p) for p in all_sources],
            )
            done = {r[0] for r in cur.fetchall()}
        return [p for p in all_sources if str(p) not in done]

    def get_completed_sources(self) -> list[str]:
        with self.get_cursor() as cur:
            cur.execute("SELECT source_path FROM sources WHERE status = 'finalized'")
            return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PipelineState:
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
