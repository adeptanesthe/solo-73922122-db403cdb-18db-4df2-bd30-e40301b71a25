"""End-to-end API tests via FastAPI TestClient (no network sockets)."""

import io
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _csv_upload(records, name="corpus.csv"):
    buf = io.StringIO()
    buf.write("id,title,abstract,relevant\n")
    for r in records:
        buf.write(
            f'{r["id"]},{r["title"]},"{r["abstract"]}",'
            f'{"true" if r["relevant"] else "false"}\n'
        )
    return {"file": (name, buf.getvalue(), "text/csv")}


@pytest.fixture()
def session(records):
    resp = client.post("/api/sessions", files=_csv_upload(records))
    assert resp.status_code == 200, resp.text
    return resp.json()


def _analyze(session_id, **overrides):
    body = {
        "session_id": session_id,
        "query": "quantum error correction",
        "min_recall": 0.9,
        "max_fdr": 0.2,
    }
    body.update(overrides)
    return client.post("/api/analyze", json=body)


# --------------------------------------------------------------------------- #
# upload & validation
# --------------------------------------------------------------------------- #
def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_upload_csv_and_jsonl(records):
    resp = client.post("/api/sessions", files=_csv_upload(records))
    assert resp.status_code == 200
    data = resp.json()
    assert data["corpus_size"] == 12 and data["n_relevant"] == 6

    lines = [json.dumps(r) for r in records]
    resp2 = client.post(
        "/api/sessions",
        files={"file": ("c.jsonl", "\n".join(lines), "application/x-ndjson")},
    )
    assert resp2.status_code == 200
    assert resp2.json()["corpus_size"] == 12


def test_upload_rejects_bad_extension():
    resp = client.post(
        "/api/sessions",
        files={"file": ("x.docx", b"binary", "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_upload_rejects_missing_label():
    bad = [{"id": "a", "title": "t", "abstract": "quantum quantum",
            "relevant": True}]
    resp = client.post("/api/sessions", files=_csv_upload(bad))
    # single doc has no negatives -> 400
    assert resp.status_code == 400


def test_unknown_session_404():
    resp = _analyze("does-not-exist")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# analyze happy path & failure path
# --------------------------------------------------------------------------- #
def test_analyze_satisfied(session):
    resp = _analyze(session["session_id"])
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "satisfied"
    assert data["metrics"]["recall"] >= 0.9 - 1e-12
    assert data["metrics"]["fdr"] <= 0.2 + 1e-12
    assert data["suggestion_id"] and data["revision"] == 0
    # candidate pool annotated for the UI
    assert all({"term", "reason", "locked", "excluded"} <= set(c)
               for c in data["candidate_pool"])


def test_analyze_threshold_conflict(session):
    sid = session["session_id"]
    # force a contradictory setup: a locked polluting term (neural occurs
    # only in negative docs) makes the fdr constraint infeasible.
    client.post("/api/revise", json={
        "session_id": sid, "lock": ["neural"],
    })
    resp = _analyze(sid, min_recall=0.3, max_fdr=0.1)
    data = resp.json()
    assert data["status"] == "failed"
    assert data["failure"]["reasons"]
    assert "unconstrained_max_recall" in data["failure"]
    # ranking is still returned as best-effort output
    assert isinstance(data["ranking"], list)


def test_invalid_threshold_400(session):
    resp = _analyze(session["session_id"], min_recall=1.2)
    assert resp.status_code == 422  # pydantic validation


# --------------------------------------------------------------------------- #
# revisions & stale-suggestion invalidation
# --------------------------------------------------------------------------- #
def test_revision_invalidates_old_suggestion(session):
    sid = session["session_id"]
    r1 = _analyze(sid).json()
    suggestion_id = r1["suggestion_id"]
    assert r1["revision"] == 0

    # revise: exclude a term -> revision bumps to 1
    target = r1["candidate_pool"][0]["term"]
    rev = client.post("/api/revise", json={
        "session_id": sid, "exclude": [target],
    }).json()
    assert rev["revision"] == 1
    assert rev["previous_suggestion_invalidated"] is True
    assert target in rev["excluded_terms"]

    # replaying the stale suggestion id must be rejected with 409
    stale = client.post("/api/analyze", json={
        "session_id": sid, "query": "quantum error correction",
        "min_recall": 0.9, "max_fdr": 0.2, "suggestion_id": suggestion_id,
    })
    assert stale.status_code == 409
    assert "失效" in stale.json()["detail"]

    # a fresh analyze (no suggestion id) succeeds and binds a new suggestion
    fresh = _analyze(sid).json()
    assert fresh["status"] in {"satisfied", "failed"}
    assert fresh["revision"] == 1
    assert fresh["suggestion_id"] != suggestion_id
    assert target not in {c["term"] for c in fresh["candidate_pool"]
                          if not c["excluded"]}


def test_revise_with_stale_id_returns_409(session):
    sid = session["session_id"]
    r1 = _analyze(sid).json()
    client.post("/api/revise", json={
        "session_id": sid, "exclude": [r1["candidate_pool"][0]["term"]],
    })
    # another revise carrying the old suggestion id -> 409 explicit
    resp = client.post("/api/revise", json={
        "session_id": sid,
        "lock": [r1["candidate_pool"][1]["term"]],
        "stale_suggestion_id": r1["suggestion_id"],
    })
    assert resp.status_code == 409


def test_lock_then_unlock_restore_revision(session):
    sid = session["session_id"]
    r1 = _analyze(sid).json()
    term = r1["candidate_pool"][0]["term"]
    r2 = client.post("/api/revise", json={
        "session_id": sid, "lock": [term],
    }).json()
    assert r2["revision"] == 1 and term in r2["locked_terms"]

    r3 = client.post("/api/revise", json={
        "session_id": sid, "unlock": [term],
    }).json()
    assert r3["locked_terms"] == []
    assert r3["revision"] == 2  # every effective change bumps revision


def test_lock_exclude_conflict_400(session):
    sid = session["session_id"]
    r1 = _analyze(sid).json()
    term = r1["candidate_pool"][0]["term"]
    client.post("/api/revise", json={
        "session_id": sid, "lock": [term],
    })
    resp = client.post("/api/revise", json={
        "session_id": sid, "exclude": [term],
    })
    assert resp.status_code == 400


def test_excluded_term_never_selected(session):
    sid = session["session_id"]
    r1 = _analyze(sid).json()
    for term in r1["expanded_terms"]:
        client.post("/api/revise", json={
            "session_id": sid, "exclude": [term],
        })
    r2 = _analyze(sid).json()
    assert not (set(r2["expanded_terms"]) & set(r1["expanded_terms"]))


def test_locked_term_forced_into_solution(session):
    sid = session["session_id"]
    term = "decoherence"
    client.post("/api/revise", json={
        "session_id": sid, "lock": [term],
    })
    data = _analyze(sid, min_recall=0.3, max_fdr=0.5).json()
    assert term in data["selected_terms"]
    assert data["current_locked_terms"] == [term]


def test_session_getter(session):
    resp = client.get(f"/api/sessions/{session['session_id']}")
    assert resp.status_code == 200
    assert resp.json()["revision"] == 0
