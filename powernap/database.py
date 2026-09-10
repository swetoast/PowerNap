from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from .model import Decision, OperationResult, SystemState


class Repository:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self._migrate_legacy_schema()
        self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_samples_ts ON samples(ts_ms);
            CREATE TABLE IF NOT EXISTS decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                recommended TEXT NOT NULL,
                reason TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts_ms);
            CREATE TABLE IF NOT EXISTS controls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_ms INTEGER NOT NULL,
                adapter TEXT NOT NULL,
                target TEXT NOT NULL,
                result TEXT NOT NULL,
                payload TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_controls_ts ON controls(ts_ms);
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        self.conn.commit()
        self._import_legacy_rows()

    def _migrate_legacy_schema(self) -> None:
        """Stage old or interrupted 0.9.0 tables for idempotent import."""
        tables = {row[0] for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        expected = {
            "samples": {"id", "ts_ms", "timestamp", "payload"},
            "decisions": {"id", "ts_ms", "recommended", "reason", "payload"},
            "controls": {"id", "ts_ms", "adapter", "target", "result", "payload"},
        }
        renamed = []
        with self.conn:
            for table, required in expected.items():
                legacy = f"{table}_legacy_090"
                if legacy in tables:
                    columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({legacy})")}
                    renamed.append((table, legacy, columns))
                    continue
                if table not in tables:
                    continue
                columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
                if required.issubset(columns):
                    continue
                self.conn.execute(f"ALTER TABLE {table} RENAME TO {legacy}")
                renamed.append((table, legacy, columns))
        self._legacy_tables = renamed

    def _import_legacy_rows(self) -> None:
        with self.conn:
            for table, legacy, columns in getattr(self, "_legacy_tables", []):
                if table == "samples" and {"ts", "timestamp", "payload"}.issubset(columns):
                    self.conn.execute(f"""INSERT INTO samples(ts_ms,timestamp,payload)
                        SELECT l.ts,l.timestamp,l.payload FROM {legacy} l
                        WHERE NOT EXISTS (SELECT 1 FROM samples n WHERE n.ts_ms=l.ts AND n.payload=l.payload)""")
                elif table == "decisions" and {"ts", "recommended", "reason", "payload"}.issubset(columns):
                    self.conn.execute(f"""INSERT INTO decisions(ts_ms,recommended,reason,payload)
                        SELECT l.ts,l.recommended,l.reason,l.payload FROM {legacy} l
                        WHERE NOT EXISTS (SELECT 1 FROM decisions n WHERE n.ts_ms=l.ts AND n.payload=l.payload)""")
                elif table == "controls" and {"ts", "adapter", "target", "result", "payload"}.issubset(columns):
                    self.conn.execute(f"""INSERT INTO controls(ts_ms,adapter,target,result,payload)
                        SELECT l.ts,l.adapter,l.target,l.result,l.payload FROM {legacy} l
                        WHERE NOT EXISTS (SELECT 1 FROM controls n WHERE n.ts_ms=l.ts AND n.payload=l.payload)""")
                self.conn.execute(f"DROP TABLE {legacy}")
        self._legacy_tables = []

    def record_cycle(self, state: SystemState, decision: Decision, results: list[OperationResult]) -> None:
        ts_ms = time.time_ns() // 1_000_000
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO samples(ts_ms,timestamp,payload) VALUES(?,?,?)",
                    (ts_ms, state.timestamp, json.dumps(asdict(state), default=str, separators=(",", ":"))),
                )
                self.conn.execute(
                    "INSERT INTO decisions(ts_ms,recommended,reason,payload) VALUES(?,?,?,?)",
                    (ts_ms, decision.recommended.name.lower(), decision.reason, json.dumps(decision.to_dict(), separators=(",", ":"))),
                )
                self.conn.executemany(
                    "INSERT INTO controls(ts_ms,adapter,target,result,payload) VALUES(?,?,?,?,?)",
                    [
                        (ts_ms, item.operation.adapter, item.operation.target, item.state.value, json.dumps(asdict(item), default=str, separators=(",", ":")))
                        for item in results
                    ],
                )
        except sqlite3.Error as exc:
            raise RuntimeError(f"Unable to record PowerNap cycle in {self.path}: {exc}") from exc

    def set_meta(self, key: str, value) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)", (key, json.dumps(value)))

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def prune(self, sample_days: int, event_days: int, now_ms: int | None = None) -> None:
        now_ms = time.time_ns() // 1_000_000 if now_ms is None else now_ms
        with self.conn:
            self.conn.execute("DELETE FROM samples WHERE ts_ms < ?", (now_ms - sample_days * 86_400_000,))
            self.conn.execute("DELETE FROM decisions WHERE ts_ms < ?", (now_ms - event_days * 86_400_000,))
            self.conn.execute("DELETE FROM controls WHERE ts_ms < ?", (now_ms - event_days * 86_400_000,))

    def report(self, limit: int = 25) -> dict[str, list[dict]]:
        limit = max(1, min(1000, int(limit)))
        return {
            "decisions": [dict(row) for row in self.conn.execute(
                "SELECT ts_ms,recommended,reason FROM decisions ORDER BY id DESC LIMIT ?", (limit,)
            )],
            "controls": [dict(row) for row in self.conn.execute(
                "SELECT ts_ms,adapter,target,result FROM controls ORDER BY id DESC LIMIT ?", (limit,)
            )],
        }

    def close(self) -> None:
        self.conn.close()
