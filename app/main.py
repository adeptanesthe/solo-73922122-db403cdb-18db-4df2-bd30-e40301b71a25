"""FastAPI application: corpus upload, query expansion, term revisions."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .engine import (
    AnalysisRequest,
    Corpus,
    CorpusError,
    EngineError,
    RevisionError,
    ThresholdError,
    analyze,
)
from .session import SessionStore, store

app = FastAPI(
    title="科研检索查询扩展系统",
    description=(
        "上传带相关性标签的论文摘要，选择正例候选词扩展查询，"
        "在召回下限与误检上限约束下重排结果并解释每个扩展词的覆盖变化。"
    ),
    version="1.0.0",
)
session_store: SessionStore = store

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AnalyzeBody(BaseModel):
    session_id: str
    query: str = Field(..., min_length=1)
    min_recall: float = Field(..., ge=0.0, le=1.0)
    max_fdr: float = Field(..., ge=0.0, le=1.0)
    max_false_positives: int | None = Field(default=None, ge=0)
    beam_width: int = Field(default=8, ge=1, le=64)
    max_terms: int = Field(default=12, ge=1, le=30)
    top_k: int = Field(default=50, ge=1, le=500)


class ReviseBody(BaseModel):
    session_id: str
    lock: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    unlock: list[str] = Field(default_factory=list)
    unexclude: list[str] = Field(default_factory=list)
    # 若携带旧建议 ID，服务端校验其是否已被本次修订作废
    stale_suggestion_id: str | None = None


class AnalyzeWithSuggestionBody(AnalyzeBody):
    suggestion_id: str | None = None


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
@app.exception_handler(EngineError)
async def engine_error_handler(request, exc: EngineError):
    status = 409 if isinstance(exc, RevisionError) else 400
    return _json_error(status, exc)


@app.exception_handler(CorpusError)
async def corpus_error_handler(request, exc: CorpusError):
    return _json_error(400, exc)


def _json_error(status: int, exc: Exception):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/demo-corpus")
def demo_corpus():
    """Serve the built-in demo corpus so the browser demo needs no upload."""
    path = Path(__file__).resolve().parent.parent / "data" / "demo_corpus.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="演示语料尚未生成。")
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        headers={"Content-Disposition": 'attachment; filename="demo_corpus.csv"'},
    )


@app.post("/api/sessions")
async def create_session(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="上传文件为空。")
    try:
        corpus = Corpus.from_file(raw, file.filename or "upload.csv")
    except CorpusError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # malformed csv/json etc.
        raise HTTPException(
            status_code=400, detail=f"文件解析失败：{exc}"
        ) from exc
    session = session_store.create(corpus)
    return {
        "session_id": session.session_id,
        "revision": session.revision,
        "corpus_size": len(corpus.labels),
        "n_relevant": corpus.n_relevant,
        "n_nonrelevant": corpus.n_nonrelevant,
        "vocab_size": len(corpus.vocab),
    }


@app.post("/api/analyze")
def analyze_endpoint(body: AnalyzeWithSuggestionBody):
    try:
        session = session_store.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新上传语料。")

    # 旧建议校验：如果前端带着一个 suggestion_id，先确认它没有被修订作废
    if body.suggestion_id:
        try:
            session_store.ensure_fresh(session, body.suggestion_id)
        except RevisionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    req = AnalysisRequest(
        query=body.query,
        min_recall=body.min_recall,
        max_fdr=body.max_fdr,
        max_false_positives=body.max_false_positives,
        locked_terms=session.locked_terms,
        excluded_terms=session.excluded_terms,
        beam_width=body.beam_width,
        max_terms=body.max_terms,
        top_k=body.top_k,
    )
    try:
        result = analyze(session.corpus, req)
    except (ThresholdError, CorpusError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    suggestion_id, revision = session_store.bind_suggestion(session)
    result["suggestion_id"] = suggestion_id
    result["revision"] = revision
    result["current_locked_terms"] = list(session.locked_terms)
    result["current_excluded_terms"] = list(session.excluded_terms)
    return result


@app.post("/api/revise")
def revise_endpoint(body: ReviseBody):
    try:
        session = session_store.get(body.session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在或已过期，请重新上传语料。")

    # 先用旧 ID 演示“旧建议失效”的显式报错路径
    if body.stale_suggestion_id:
        try:
            session_store.ensure_fresh(session, body.stale_suggestion_id)
        except RevisionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        session = session_store.revise(
            body.session_id,
            lock=body.lock,
            exclude=body.exclude,
            unlock=body.unlock,
            unexclude=body.unexclude,
        )
    except ThresholdError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "session_id": session.session_id,
        "revision": session.revision,
        "locked_terms": list(session.locked_terms),
        "excluded_terms": list(session.excluded_terms),
        "previous_suggestion_invalidated": True,
        "message": (
            "锁定/排除词已更新，修订号 +1；此前生成的建议已失效，"
            "请重新发起分析。"
        ),
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    try:
        session = session_store.get(session_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="会话不存在或已过期。")
    return {
        "session_id": session.session_id,
        "revision": session.revision,
        "locked_terms": list(session.locked_terms),
        "excluded_terms": list(session.excluded_terms),
        "last_suggestion_id": session.last_suggestion_id,
        "last_suggestion_rev": session.last_suggestion_rev,
    }


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
