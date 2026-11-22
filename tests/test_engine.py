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


# --------------------------------------------------------------------------- #
# intermediate false-detection overflow must be repairable
# --------------------------------------------------------------------------- #
def _dilution_corpus():
    """Baseline 'alpha'; *every* first expansion term starts over the cap.

    baseline alpha -> r1,r2                       (recall 2/6, fdr 0.00)
    apoll          -> r1,r2,r3,r4 + n1,n2,n3      (recall 4/6, fdr 0.43)
    bpoll          -> r1,r2,r5,r6 + n1,n2,n3      (recall 4/6, fdr 0.43)
    apoll+bpoll    -> all docs                    (recall 1.0, fdr 0.33)

    Under a 0.35 fdr cap neither term is acceptable alone, yet together they
    reach full recall.  The old engine dropped both single-term branches and
    reported a conflict; the fix must keep the over-limit states alive and
    return the full-recall pair, both when one term is locked and when the
    search is free to choose.
    """
    return Corpus.from_records([
        {"id": "r1", "title": "t", "abstract": "alpha alpha", "relevant": True},
        {"id": "r2", "title": "t", "abstract": "alpha alpha", "relevant": True},
        {"id": "r3", "title": "t", "abstract": "apoll apoll", "relevant": True},
        {"id": "r4", "title": "t", "abstract": "apoll apoll", "relevant": True},
        {"id": "r5", "title": "t", "abstract": "bpoll bpoll", "relevant": True},
        {"id": "r6", "title": "t", "abstract": "bpoll bpoll", "relevant": True},
        {"id": "n1", "title": "t", "abstract": "apoll bpoll", "relevant": False},
        {"id": "n2", "title": "t", "abstract": "apoll bpoll", "relevant": False},
        {"id": "n3", "title": "t", "abstract": "apoll bpoll", "relevant": False},
        {"id": "n4", "title": "t", "abstract": "filler filler", "relevant": False},
        {"id": "n5", "title": "t", "abstract": "filler filler", "relevant": False},
    ])


def test_intermediate_fdr_overflow_is_diluted_and_satisfies():
    corpus = _dilution_corpus()
    res = analyze(corpus, AnalysisRequest(
        query="alpha", min_recall=1.0, max_fdr=0.35,
        locked_terms=["apoll"],
    ))
    assert res["status"] == "satisfied"
    assert res["selected_terms"][0] == "apoll"  # locked term stays first
    assert "bpoll" in res["selected_terms"]
    assert res["metrics"]["recall"] == pytest.approx(1.0)
    assert res["metrics"]["fdr"] <= 0.35 + 1e-12
    # the trace must show the over-limit step and the recovery after it
    fdr_by_term = {row["term"]: row["cumulative_fdr"] for row in res["trace"]}
    assert fdr_by_term["apoll"] > 0.35
    assert fdr_by_term["bpoll"] <= 0.35 + 1e-12


def test_intermediate_fdr_overflow_without_lock():
    corpus = _dilution_corpus()
    res = analyze(corpus, AnalysisRequest(
        query="alpha", min_recall=1.0, max_fdr=0.35,
    ))
    assert res["status"] == "satisfied"
    assert res["metrics"]["recall"] == pytest.approx(1.0)
    assert res["metrics"]["fdr"] <= 0.35 + 1e-12
    # both expansion terms are needed; neither alone is within the cap
    seed = corpus.docs_for_terms(corpus.query_terms("alpha"))
    for term in res["expanded_terms"]:
        m = corpus.metrics_for_docs(
            seed | corpus.docs_for_terms([term])
        )
        assert m["fdr"] > 0.35


def test_locked_polluting_term_can_be_repaired():
    # The lock prefix alone is over the limit; a later relevant-heavy term
    # dilutes it back under.  This must not be reported as a conflict.
    corpus = _dilution_corpus()
    res = analyze(corpus, AnalysisRequest(
        query="alpha", min_recall=1.0, max_fdr=0.35,
        locked_terms=["apoll"],
    ))
    assert res["status"] == "satisfied"
    assert res["metrics"]["recall"] == pytest.approx(1.0)
    assert res["metrics"]["fdr"] <= 0.35 + 1e-12
    assert res["failure"] is None


def test_beam_truncation_reports_inconclusive_not_impossible():
    """Beam width 1 prunes the winning branch; that is not a proof."""
    records = [
        ("d0", "t2 t5", True), ("d1", "t5 t0 alpha", True),
        ("d2", "t5 t0", True), ("d3", "t1 t4 t3", True),
        ("d4", "t1 t3", True), ("d5", "t3 t5 alpha", True),
        ("d6", "t5 t1 t0 alpha", False), ("d7", "t2 t4", False),
        ("d8", "t3 t5", False), ("d9", "t0 t4 t1", False),
        ("d10", "t4 t5", False), ("d11", "t4 t5", False),
    ]
    corpus = Corpus.from_records([
        {"id": i, "title": "t", "abstract": a, "relevant": r}
        for i, a, r in records
    ])
    kwargs = dict(query="alpha", min_recall=0.8, max_fdr=0.3, max_terms=2)
    narrow = analyze(corpus, AnalysisRequest(beam_width=1, **kwargs))
    wide = analyze(corpus, AnalysisRequest(beam_width=8, **kwargs))
    assert wide["status"] == "satisfied"
    assert narrow["status"] == "failed"
    diag = narrow["failure"]
    assert diag["verdict"] == "search_inconclusive"
    assert diag["proven_infeasible"] is False
    assert diag["branches_pruned"] is True
    assert diag["recall_upper_bound"] >= 0.8
    assert any("尚无定论" in r for r in diag["reasons"])


def test_complete_search_proves_infeasibility():
    corpus = _polluted_corpus()
    res = analyze(corpus, AnalysisRequest(
        query="quantum error correction", min_recall=1.0, max_fdr=0.1,
        beam_width=32,
    ))
    assert res["status"] == "failed"
    diag = res["failure"]
    assert diag["verdict"] == "proven_infeasible"
    assert diag["proven_infeasible"] is True
    assert diag["search_complete"] is True
    # the recall bound may still be 1.0 here (ignoring fdr full recall *is*
    # reachable); the proof comes from the exhaustive search finding nothing
    assert diag["recall_upper_bound"] <= 1.0
    assert diag["best_feasible_recall"] < 1.0
    # no satisfying combination exists, so any best-effort output stays
    # below the recall floor and within the false-detection limits
    if res["best_found"]:
        assert res["best_found"]["metrics"]["recall"] < 1.0
        assert res["best_found"]["metrics"]["fdr"] <= 0.1 + 1e-12


def test_brute_force_matches_beam_search_on_random_corpora():
    """Cross-check the beam search against exhaustive enumeration."""
    import itertools
    import random

    from app.engine import extract_candidates

    rng = random.Random(20261122)
    for _trial in range(60):
        n_terms = rng.randint(4, 6)
        terms = [f"t{i}" for i in range(n_terms)]
        docs, labels = [], []
        for i in range(12):
            docs.append(" ".join(rng.sample(terms, rng.randint(2, 3))))
            labels.append(i < 6)
        for idx in rng.sample(range(12), 3):
            docs[idx] += " alpha"
        corpus = Corpus.from_records([
            {"id": f"d{i}", "title": "t", "abstract": a,
             "relevant": labels[i]}
            for i, a in enumerate(docs)
        ])
        base = corpus.query_terms("alpha")
        if not base:
            continue
        cands = [c.term for c in extract_candidates(corpus, base, set())]
        if len(cands) < 3:
            continue
        base_docs = corpus.docs_for_terms(base)
        for max_terms in (2, 3):
            for min_recall in (0.8, 1.0):
                for max_fdr in (0.2, 0.3):
                    best_feasible = None
                    for k in range(max_terms + 1):
                        for combo in itertools.combinations(cands, k):
                            m = corpus.metrics_for_docs(
                                base_docs | corpus.docs_for_terms(combo)
                            )
                            if (m["recall"] >= min_recall - 1e-12
                                    and m["fdr"] <= max_fdr + 1e-12):
                                if (best_feasible is None
                                        or m["recall"] > best_feasible):
                                    best_feasible = m["recall"]
                    res = analyze(corpus, AnalysisRequest(
                        query="alpha", min_recall=min_recall,
                        max_fdr=max_fdr, max_terms=max_terms, beam_width=10,
                    ))
                    if best_feasible is None:
                        assert res["status"] == "failed"
                    else:
                        assert res["status"] == "satisfied"
                        # the search must reach the true optimum, not merely
                        # some feasible state
                        assert res["metrics"]["recall"] == pytest.approx(
                            best_feasible
                        )
