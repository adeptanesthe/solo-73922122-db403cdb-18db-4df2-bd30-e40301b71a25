"""In-memory session store with revision-based suggestion invalidation."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field

from .engine import Corpus, RevisionError, ThresholdError


@dataclass
class Session:
    session_id: str
    corpus: Corpus
    locked_terms: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    revision: int = 0                       # bumped on every lock/exclude change
    last_suggestion_rev: int | None = None  # revision when analysis was run
    last_suggestion_id: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._guard = threading.Lock()

    def create(self, corpus: Corpus) -> Session:
        sid = uuid.uuid4().hex
        session = Session(session_id=sid, corpus=corpus)
        with self._guard:
            self._sessions[sid] = session
        return session

    def get(self, session_id: str) -> Session:
        with self._guard:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    # -- revision operations ---------------------------------------------- #
    def revise(
        self,
        session_id: str,
        lock: list[str] | None = None,
        exclude: list[str] | None = None,
        unlock: list[str] | None = None,
        unexclude: list[str] | None = None,
    ) -> Session:
        session = self.get(session_id)
        with session.lock:
            locked = list(session.locked_terms)
            excluded = list(session.excluded_terms)

            for term in unlock or []:
                t = term.lower()
                if t in locked:
                    locked.remove(t)
            for term in unexclude or []:
                t = term.lower()
                if t in excluded:
                    excluded.remove(t)
            for term in lock or []:
                t = term.lower().strip()
                if t and t not in locked:
                    locked.append(t)
            for term in exclude or []:
                t = term.lower().strip()
                if t and t not in excluded:
                    excluded.append(t)

            conflict = sorted(set(locked) & set(excluded))
            if conflict:
                raise ThresholdError(
                    f"词同时被锁定和排除：{', '.join(conflict)}，操作已拒绝。"
                )
            unknown = [
                t for t in locked + excluded
                if session.corpus.term_index(t) is None
            ]
            if unknown:
                raise ThresholdError(
                    f"词不在语料词表中：{', '.join(sorted(set(unknown)))}。"
                )

            changed = (
                locked != session.locked_terms
                or excluded != session.excluded_terms
            )
            session.locked_terms = locked
            session.excluded_terms = excluded
            if changed:
                session.revision += 1  # old suggestions are now stale
                # drop any bound suggestion so a replayed id is reported as
                # invalidated by the revision change, not merely mismatched
                session.last_suggestion_id = None
                session.last_suggestion_rev = None
            return session

    def bind_suggestion(self, session: Session) -> tuple[str, int]:
        """Record a fresh suggestion against the current revision."""
        with session.lock:
            suggestion_id = uuid.uuid4().hex
            session.last_suggestion_id = suggestion_id
            session.last_suggestion_rev = session.revision
            return suggestion_id, session.revision

    def ensure_fresh(self, session: Session, suggestion_id: str | None) -> None:
        """Reject stale suggestions after a term revision (409 at API layer)."""
        with session.lock:
            if suggestion_id is None:
                return
            if session.last_suggestion_id is None:
                raise RevisionError(
                    "该建议已失效：会话中的词锁定/排除设置已变更，请重新分析。"
                )
            if suggestion_id != session.last_suggestion_id:
                raise RevisionError(
                    "建议 ID 与当前会话不匹配，可能来自更早的一次分析，请重新分析。"
                )
            if session.last_suggestion_rev != session.revision:
                raise RevisionError(
                    "该建议已失效：自建议生成以来锁定/排除词发生过修改，"
                    "旧建议不再有效，请重新分析以获取新建议。"
                )


store = SessionStore()
