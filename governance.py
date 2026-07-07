"""
governance.py — Audit logging, rate limiting, content monitoring, prompt versioning
"""

import hashlib
import json
import re
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional

DB_PATH = "chat_history.db"


class AuditLog:
    """Persistent, append-only log of all chatbot interactions."""

    def __init__(self):
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    session_id TEXT,
                    query TEXT,
                    response TEXT,
                    model TEXT,
                    latency_s REAL,
                    tokens_in INTEGER,
                    tokens_out INTEGER,
                    refused INTEGER,
                    citations TEXT,
                    prompt_version TEXT,
                    flags TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id)
            """)
            conn.commit()

    def log(self, session_id: str, query: str, response: str, model: str,
            latency_s: float, tokens_in: int = 0, tokens_out: int = 0,
            refused: bool = False, citations: Optional[list[str]] = None,
            prompt_version: str = "", flags: Optional[list[str]] = None):
        now = datetime.utcnow().isoformat()
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT INTO audit_log
                   (timestamp, session_id, query, response, model, latency_s,
                    tokens_in, tokens_out, refused, citations, prompt_version, flags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (now, session_id, query, response, model, latency_s,
                 tokens_in, tokens_out, 1 if refused else 0,
                 json.dumps(citations or []), prompt_version,
                 json.dumps(flags or [])),
            )
            conn.commit()

    def get_recent(self, limit: int = 50) -> list[dict]:
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [{
            "id": r[0], "timestamp": r[1], "session_id": r[2],
            "query": r[3][:100], "response": r[4][:100],
            "model": r[5], "latency_s": r[6],
            "tokens_in": r[7], "tokens_out": r[8],
            "refused": bool(r[9]), "citations": r[10],
            "prompt_version": r[11], "flags": r[12],
        } for r in rows]

    def get_stats(self) -> dict:
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            today = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE timestamp >= ?",
                (datetime.utcnow().isoformat()[:10],),
            ).fetchone()[0]
            model_dist = conn.execute(
                "SELECT model, COUNT(*) as cnt FROM audit_log GROUP BY model ORDER BY cnt DESC"
            ).fetchall()
            avg_latency = conn.execute(
                "SELECT AVG(latency_s) FROM audit_log"
            ).fetchone()[0] or 0.0
            total_tokens = conn.execute(
                "SELECT COALESCE(SUM(tokens_in + tokens_out), 0) FROM audit_log"
            ).fetchone()[0]
        return {
            "total_queries": total,
            "today_queries": today,
            "model_distribution": dict(model_dist),
            "avg_latency_s": round(avg_latency, 2),
            "total_tokens": total_tokens,
        }


class RateLimiter:
    """Per-session and global rate limiting."""

    def __init__(self, max_per_session: int = 40, max_per_minute: int = 10):
        self.max_per_session = max_per_session
        self.max_per_minute = max_per_minute
        self._sessions: dict[str, int] = {}
        self._minute_windows: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def check_session(self, session_id: str) -> tuple[bool, str]:
        with self._lock:
            count = self._sessions.get(session_id, 0)
            if count >= self.max_per_session:
                return False, f"Session limit ({self.max_per_session}) reached"
            now = time.time()
            window = self._minute_windows.setdefault(session_id, [])
            window[:] = [t for t in window if now - t < 60]
            if len(window) >= self.max_per_minute:
                return False, f"Rate limit ({self.max_per_minute}/minute) exceeded"
            window.append(now)
            self._sessions[session_id] = count + 1
            return True, ""

    def get_usage(self, session_id: str) -> dict:
        with self._lock:
            return {
                "session_queries": self._sessions.get(session_id, 0),
                "max_per_session": self.max_per_session,
                "remaining": self.max_per_session - self._sessions.get(session_id, 0),
            }


_FLAG_PATTERNS = {
    "pii_email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "pii_phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b",
    "injection_override": r"(?i)(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|above|prior)\s+instructions",
    "injection_system": r"(?i)(?:system|admin|developer)\s*(?::|prompt|instruction|command)",
    "injection_exfil": r"(?i)(?:reveal|show|output|leak|dump|print)\s+(?:your\s+)?(?:system\s+)?prompt",
    "profanity": r"(?i)\b(fuck|shit|damn|asshole|bastard)\b",
}


class ContentMonitor:
    """Flag potentially harmful or sensitive content."""

    def __init__(self):
        self._patterns = {k: re.compile(v) for k, v in _FLAG_PATTERNS.items()}

    def check_query(self, text: str) -> list[str]:
        flags = []
        for name, pattern in self._patterns.items():
            if pattern.search(text):
                flags.append(name)
        return flags

    def check_response(self, text: str) -> list[str]:
        flags = []
        if re.search(r"(?i)(?:sk-or-v1|sk-proj-)[a-zA-Z0-9]+", text):
            flags.append("api_key_leak")
        if re.search(r"(?i)(?:system prompt|grounding rule|SYSTEM_PROMPT_TEMPLATE)", text):
            flags.append("prompt_leak")
        return flags


class PromptVersion:
    """Track system prompt versions via SHA256 hashing."""

    def __init__(self):
        self._versions: dict[str, str] = {}

    def register(self, prompt_text: str) -> str:
        vhash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:12]
        self._versions[vhash] = prompt_text
        return vhash

    def get_version(self, vhash: str) -> Optional[str]:
        return self._versions.get(vhash)
