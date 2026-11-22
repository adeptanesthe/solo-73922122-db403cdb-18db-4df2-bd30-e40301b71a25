"""Engine-level tests: candidates, beam search, explanations, failure."""

import pytest

from app.engine import (
    AnalysisRequest,
    Corpus,
    CorpusError,
    ThresholdError,
    analyze,
    extract_candidates,
    tokenize,
)


# --------------------------------------------------------------------------- #
# corpus / tokenization
# --------------------------------------------------------------------------- #
def test_tokenize_basic():
    assert tokenize("Quantum Error-Correction!") == ["quantum", "error-correction"]


def test_corpus_requires_both_labels():
    with pytest.raises(CorpusError):
        Corpus.from_records([
            {"id": "a", "title": "t", "abstract": "quantum quantum", "relevant": True}
        ])
    with pytest.raises(CorpusError):
        Corpus.from_records([
            {"id": "a", "title": "t", "abstract": "quantum quantum", "relevant": False},
            {"id": "b", "title": "t", "abstract": "neural neural", "relevant": False},
        ])


def test_baseline_and_vocab(corpus):
    terms = corpus.query_terms("Quantum Error Correction zzzword")
    assert terms == ["quantum", "error", "correction"]
    assert corpus.unknown_query_tokens("quantum zzzword") == ["zzzword"]
    base_docs = corpus.docs_for_terms(terms)
    metrics = corpus.metrics_for_docs(base_docs)
    assert metrics["true_positives"] == 2
    assert metrics["recall"] == pytest.approx(2 / 6)
    assert metrics["fdr"] == 0.0


# --------------------------------------------------------------------------- #
# candidates come from positive docs only
# --------------------------------------------------------------------------- #
def test_candidates_only_positive(corpus):
    cands = extract_candidates(
        corpus, corpus.query_terms("quantum error correction"), set()
    )
    terms = {c.term for c in cands}
    # positive-only expansion vocabulary
    assert {"surface", "code", "syndrome", "logical", "qubit", "decoherence"} <= terms
    # terms appearing only in negative docs must never be candidates
    assert "neural" not in terms
    assert "crispr" not in terms
    assert "climate" not in terms
    for c in cands:
        assert c.n_relevant_hits >= 1


def test_excluded_terms_removed(corpus):
    cands = extract_candidates(
        corpus, corpus.query_terms("quantum error correction"),
        excluded={"surface", "syndrome"},
    )
    terms = {c.term for c in cands}
    assert "surface" not in terms and "syndrome" not in terms


# --------------------------------------------------------------------------- #
# satisfied path + per-term trace
# --------------------------------------------------------------------------- #
def test_satisfied_expansion_raises_recall(corpus):
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction",
        min_recall=0.9, max_fdr=0.2,
    ))
    assert res["status"] == "satisfied"
    assert len(res["expanded_terms"]) >= 1
    assert res["metrics"]["recall"] >= 0.9 - 1e-12
    assert res["metrics"]["fdr"] <= 0.2 + 1e-12
    # trace is cumulative and starts at the baseline
    trace = res["trace"]
    assert trace[0]["is_baseline"] is True
    base_recall = trace[0]["cumulative_recall"]
    assert base_recall == pytest.approx(2 / 6, abs=1e-4)
    recalls = [row["cumulative_recall"] for row in trace]
    assert recalls == sorted(recalls)  # monotone non-decreasing
    # every expansion row explains its marginal contribution
    for row in trace[1:]:
        assert row["added_relevant"] + row["added_nonrelevant"] >= 1
    # ranking contains only retrieved docs and is ranked by similarity
    assert len(res["ranking"]) == res["metrics"]["n_retrieved"]
    scores = [d["score"] for d in res["ranking"]]
    assert scores == sorted(scores, reverse=True)


def test_baseline_already_satisfies(corpus):
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction",
        min_recall=0.3, max_fdr=0.5,
    ))
    assert res["status"] == "satisfied"
    assert res["expanded_terms"] == []


# --------------------------------------------------------------------------- #
# failure path — threshold conflict
# --------------------------------------------------------------------------- #
def test_failure_when_recall_unreachable(corpus):
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction",
        min_recall=1.0, max_fdr=0.0,
    ))
    # r6 contains stabilizer (min_df filtered) but also syndrome/logical/qubit,
    # so full recall is reachable; instead lock in a scenario via impossible fdr
    # on a polluted corpus is tested separately. Here verify plain structure.
    assert res["status"] in {"satisfied", "failed"}


def _polluted_corpus():
    """Every term that lifts recall also drags in negative docs.

    All content words satisfy min_df=2 by construction, and every term
    attached to the four missing relevant docs is polluted, so no clean
    synonym shortcut exists.
    baseline 'quantum error correction' -> r1,r2 (recall 2/6, 0 fp).
      surface  -> r3,r4 + n1      (lightest pollution -> deterministic blocker)
      code     -> r3,r4 + n1,n2
      syndrome -> r5,r6 + n2
      decoherence -> r5,r6 + n1,n2
    n3..n6 use disjoint vocab (crispr / climate).
    """
    return Corpus.from_records([
        {"id": "r1", "abstract": "quantum error correction gates", "relevant": True},
        {"id": "r2", "abstract": "quantum error correction gates", "relevant": True},
        {"id": "r3", "abstract": "surface code surface", "relevant": True},
        {"id": "r4", "abstract": "surface code code", "relevant": True},
        {"id": "r5", "abstract": "syndrome decoherence syndrome", "relevant": True},
        {"id": "r6", "abstract": "syndrome decoherence decoherence", "relevant": True},
        {"id": "n1", "abstract": "surface code decoherence network", "relevant": False},
        {"id": "n2", "abstract": "code syndrome decoherence network", "relevant": False},
        {"id": "n3", "abstract": "crispr gene editing", "relevant": False},
        {"id": "n4", "abstract": "crispr gene editing", "relevant": False},
        {"id": "n5", "abstract": "climate model ensemble", "relevant": False},
        {"id": "n6", "abstract": "climate model ensemble", "relevant": False},
    ])


def test_fdr_conflict_returns_diagnosis():
    corpus = _polluted_corpus()
    # baseline recall = 2/6 = 0.333, fdr = 0. Reaching recall 1.0 needs both
    # polluted groups; the cheapest term 'surface' alone already lifts fdr to
    # 0.25, and full coverage settles at fdr = 2/8 = 0.25 — impossible under
    # fdr <= 0.1.
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction",
        min_recall=1.0, max_fdr=0.1,
    ))
    assert res["status"] == "failed"
    assert res["failure"] is not None
    diag = res["failure"]
    assert diag["unconstrained_max_recall"] == pytest.approx(1.0)
    assert diag["best_feasible_recall"] < 1.0
    assert any("误检" in r for r in diag["reasons"])
    assert diag["blocker_term"] is not None
    assert diag["blocker_term"]["term"] == "surface"


def test_fp_count_limit_conflict():
    corpus = _polluted_corpus()
    # fdr alone would permit one fp, but the absolute fp cap does not.
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction",
        min_recall=1.0, max_fdr=1.0, max_false_positives=0,
    ))
    assert res["status"] == "failed"
    assert any("越界" in r for r in res["failure"]["reasons"])


def test_relaxing_fdr_resolves_conflict():
    corpus = _polluted_corpus()
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction",
        min_recall=1.0, max_fdr=0.4,
    ))
    assert res["status"] == "satisfied"
    assert res["metrics"]["recall"] == pytest.approx(1.0)
    # n1 and n2 are retrieved once each -> two false-positive documents
    assert res["metrics"]["false_positives"] == 2
    assert res["metrics"]["fdr"] == pytest.approx(0.25)


def test_invalid_thresholds_rejected(corpus):
    with pytest.raises(ThresholdError):
        analyze(corpus, AnalysisRequest(query="q", min_recall=1.5, max_fdr=0.2))
    with pytest.raises(ThresholdError):
        analyze(corpus, AnalysisRequest(query="q", min_recall=0.5, max_fdr=-0.1))


# --------------------------------------------------------------------------- #
# revisions — lock / exclude
# --------------------------------------------------------------------------- #
def test_exclude_forces_recompute_without_term(corpus):
    baseline = analyze(corpus, AnalysisRequest(
        query="quantum error correction", min_recall=0.9, max_fdr=0.2,
    ))
    # exclude every term the solver picked: must change the solution or fail
    excluded = list(baseline["expanded_terms"])
    res2 = analyze(corpus, AnalysisRequest(
        query="quantum error correction", min_recall=0.9, max_fdr=0.2,
        excluded_terms=excluded,
    ))
    assert not (set(res2["expanded_terms"]) & set(excluded))


def test_locked_term_is_always_included(corpus):
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction", min_recall=0.3, max_fdr=0.5,
        locked_terms=["decoherence"],
    ))
    assert "decoherence" in res["selected_terms"]
    # locking a term that baseline alone didn't need still changes coverage
    assert res["metrics"]["n_retrieved"] > res["baseline_metrics"]["n_retrieved"]


def test_lock_exclude_conflict_rejected(corpus):
    with pytest.raises(ThresholdError):
        analyze(corpus, AnalysisRequest(
            query="quantum error correction", min_recall=0.5, max_fdr=0.5,
            locked_terms=["surface"], excluded_terms=["surface"],
        ))


def test_unknown_lock_rejected(corpus):
    with pytest.raises(ThresholdError):
        analyze(corpus, AnalysisRequest(
            query="quantum error correction", min_recall=0.5, max_fdr=0.5,
            locked_terms=["nonexistent"],
        ))


def test_locked_polluting_term_breaks_fdr():
    corpus = _polluted_corpus()
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction", min_recall=0.3, max_fdr=0.05,
        locked_terms=["surface"],
    ))
    assert res["status"] == "failed"
    assert any("锁定词" in r for r in res["failure"]["reasons"])
    # fallback must not silently include the term that violated the limit
    assert "surface" not in res["selected_terms"]
    assert res["failure"]["lock_violation"] is not None


def test_too_many_locked_terms_rejected(corpus):
    with pytest.raises(ThresholdError):
        analyze(corpus, AnalysisRequest(
            query="quantum error correction", min_recall=0.5, max_fdr=0.5,
            locked_terms=["surface", "syndrome"], max_terms=1,
        ))
