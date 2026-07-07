"""
memory.py — Conversation Memory + Entity Store for BVRIT Chatbot
=================================================================
Provides session-level and cross-session memory via SQLite persistence.

Components:
  - ConversationMemory: structured history with summarization
  - EntityStore: extract and persist entities from conversation
"""

import json
import re
import sqlite3
import threading
from datetime import datetime
from typing import Optional

from config import DB_PATH

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

_NAME_PATTERN = re.compile(
    r"\b(?:my name is|I am|I'm|call me|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
    re.IGNORECASE,
)
_DEPT_KEYWORDS = [
    "cse", "computer science", "ece", "electronics",
    "eee", "electrical", "it", "information technology",
    "csm", "ai", "aiml", "mechanical", "civil",
]
_TOPIC_KEYWORDS = {
    "admission": ["admission", "apply", "eligibility", "eamcet", "rank", "intake"],
    "fee": ["fee", "cost", "tuition", "scholarship", "hostel fee"],
    "placement": ["placement", "package", "salary", "recruiter", "company"],
    "faculty": ["faculty", "professor", "teacher", "hod", "staff"],
    "campus": ["hostel", "library", "lab", "campus", "transport", "bus"],
    "exam": ["exam", "semester", "deadline", "counselling", "date"],
}


def _extract_name(text: str) -> Optional[str]:
    match = _NAME_PATTERN.search(text)
    return match.group(1) if match else None


def _extract_departments(text: str) -> list[str]:
    text_lower = text.lower()
    found = []
    for kw in _DEPT_KEYWORDS:
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            found.append(kw.upper())
    return found


def _extract_topics(text: str) -> list[str]:
    text_lower = text.lower()
    topics = []
    for topic, kws in _TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in kws):
            topics.append(topic)
    return topics


# ---------------------------------------------------------------------------
# Entity Store
# ---------------------------------------------------------------------------

class EntityStore:
    """Persist and retrieve conversation entities across sessions."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_entities (
                    session_id TEXT,
                    entity_type TEXT,
                    entity_key TEXT,
                    entity_value TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (session_id, entity_type, entity_key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    session_id TEXT,
                    summary TEXT,
                    turn_count INTEGER,
                    updated_at TEXT
                )
            """)
            conn.commit()

    def set(self, entity_type: str, key: str, value: str):
        now = datetime.utcnow().isoformat()
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memory_entities
                   (session_id, entity_type, entity_key, entity_value, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (self.session_id, entity_type, key, value, now),
            )
            conn.commit()

    def get(self, entity_type: str, key: str) -> Optional[str]:
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                "SELECT entity_value FROM memory_entities WHERE session_id=? AND entity_type=? AND entity_key=?",
                (self.session_id, entity_type, key),
            ).fetchone()
        return row[0] if row else None

    def get_all(self) -> dict:
        with self._lock, sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT entity_type, entity_key, entity_value FROM memory_entities WHERE session_id=?",
                (self.session_id,),
            ).fetchall()
        result = {}
        for etype, ekey, evalue in rows:
            result.setdefault(etype, {})[ekey] = evalue
        return result

    def update_from_conversation(self, messages: list[dict]):
        """Scan messages for new entities and persist them."""
        for msg in messages:
            text = msg.get("content", "")
            if not text:
                continue
            name = _extract_name(text)
            if name and not self.get("user", "name"):
                self.set("user", "name", name)
            depts = _extract_departments(text)
            existing = self.get("user", "departments")
            existing_list = json.loads(existing) if existing else []
            merged = list(set(existing_list + depts))
            if merged != existing_list:
                self.set("user", "departments", json.dumps(merged))
            topics = _extract_topics(text)
            existing_topics = self.get("user", "topics")
            existing_topics_list = json.loads(existing_topics) if existing_topics else []
            merged_topics = list(set(existing_topics_list + topics))
            if merged_topics != existing_topics_list:
                self.set("user", "topics", json.dumps(merged_topics))

    def get_context_blurb(self) -> str:
        """Return a human-readable summary of remembered entities."""
        parts = []
        name = self.get("user", "name")
        if name:
            parts.append(f"User's name: {name}")
        depts_str = self.get("user", "departments")
        if depts_str:
            depts = json.loads(depts_str)
            if depts:
                parts.append(f"Departments mentioned: {', '.join(depts)}")
        topics_str = self.get("user", "topics")
        if topics_str:
            topics = json.loads(topics_str)
            if topics:
                parts.append(f"Topics discussed: {', '.join(topics)}")
        return " | ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Manages conversation history with summarization and entity tracking."""

    def __init__(self, session_id: str, max_verbatim_turns: int = 3):
        self.session_id = session_id
        self.max_verbatim_turns = max_verbatim_turns
        self.entities = EntityStore(session_id)

    def prepare_messages(
        self,
        history: list[dict],
        current_question: str,
    ) -> list[dict]:
        """Build message list: system, context, history, current question.

        Strategy:
        - Keep last `max_verbatim_turns` exchanges verbatim.
        - Older turns are condensed into a summary message.
        - Entity context is injected as a system-level note.
        """
        if not history:
            return self._build_with_entities([])

        msgs = [m for m in history if m["role"] in ("user", "assistant")]
        window = self.max_verbatim_turns * 2

        if len(msgs) <= window:
            return self._build_with_entities(msgs[-window:])

        older = msgs[:-window]
        summary_parts = []
        for m in older:
            label = "Student" if m["role"] == "user" else "Assistant"
            content = m.get("content", "")[:150]
            summary_parts.append(f"{label}: {content}")
        summary_text = "Previous conversation summary:\n" + "\n".join(summary_parts)

        recent = msgs[-window:]
        return self._build_with_entities(
            [{"role": "user", "content": summary_text}] + recent,
        )

    def _build_with_entities(self, history_msgs: list[dict]) -> list[dict]:
        """Inject entity memory as a system hint."""
        blurb = self.entities.get_context_blurb()
        if blurb:
            entity_msg = {
                "role": "system",
                "content": (
                    "REMEMBERED CONTEXT (from earlier in this conversation):\n"
                    + blurb
                    + "\n\nUse this context when relevant to the current question."
                ),
            }
            return [entity_msg] + history_msgs
        return history_msgs

    def update(self, messages: list[dict]):
        """Update entity store from the latest messages."""
        self.entities.update_from_conversation(messages)
