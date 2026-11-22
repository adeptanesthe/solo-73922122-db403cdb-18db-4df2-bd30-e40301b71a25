"""Query-expansion core engine.

Pure-Python algorithm layer (no FastAPI dependency).  Implements:

* Corpus loading / validation (CSV, TSV, JSON / JSONL, list of dicts)
* TF-IDF + boolean retrieval with a shared vocabulary
* Candidate-term extraction from *positive* (relevant) documents only
* Beam search over expansion-term sets under recall / false-detection limits
* Per-term coverage change explanation
* Failure diagnosis when no expansion can satisfy the thresholds
* Re-evaluation after a user locks / excludes terms (revisions invalidate
  previous suggestions by design; the API layer enforces that with a
  revision id).

Terminology
-----------
recall   = retrieved relevant docs / all relevant docs
fdr      = retrieved non-relevant docs / all retrieved docs  (false discovery
           rate; "误检率").  When nothing is retrieved fdr is defined as 0.
The caller may additionally cap the absolute number of false positives
(``max_false_positives``).
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

from sklearn.feature_extraction.text import TfidfVectorizer


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class EngineError(ValueError):
    """Base class for all user-facing engine errors."""


class CorpusError(EngineError):
    """Malformed corpus upload."""


class ThresholdError(EngineError):
    """Thresholds that are impossible or inconsistent."""


class RevisionError(EngineError):
    """Stale suggestion used after the term locks were revised."""


# --------------------------------------------------------------------------- #
# Result status
# --------------------------------------------------------------------------- #
class Status(str, Enum):
    SATISFIED = "satisfied"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Tokenization — must be identical between the TF-IDF vectorizer and the
# boolean coverage counters, otherwise the per-term explanations drift.
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+\-]{1,}")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "by", "we", "our", "their",
    "this", "that", "these", "those", "it", "its", "as", "at", "from",
    "via", "can", "may", "which", "while", "both", "between", "through",
    "into", "using", "use", "used", "based", "such", "also", "than",
    "not", "but", "have", "has", "had", "they", "them", "his", "her",
    "his", "ours", "you", "your", "do", "does", "did", "will", "would",
    "about", "over", "under", "after", "before", "most", "more", "some",
    "any", "all", "each", "per", "its", "so", "if", "then", "thus",
    "without", "within", "during", "against", "among", "across",
}


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric token extraction with a small stopword list."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #
@dataclass
class Corpus:
    abstracts: list[str]
    labels: list[bool]                       # True == relevant (positive)
    titles: list[str]
    doc_ids: list[str]
    vectorizer: TfidfVectorizer
    tfidf: object                            # scipy sparse matrix (n_docs x n_vocab)
    bool_mat: object                         # boolean sparse matrix
    vocab: list[str]
    index_by_term: dict[str, int]
    docs_by_term: list[set[int]]             # term -> set of doc indices
    n_relevant: int
    n_nonrelevant: int

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_records(cls, records: list[dict]) -> "Corpus":
        if not records:
            raise CorpusError("语料为空：至少需要 1 篇论文摘要。")
        abstracts: list[str] = []
        labels: list[bool] = []
        titles: list[str] = []
        doc_ids: list[str] = []
        for i, rec in enumerate(records):
            abstract = str(rec.get("abstract", "")).strip()
            if not abstract:
                raise CorpusError(f"第 {i + 1} 行缺少 abstract 摘要内容。")
            abstracts.append(abstract)
            titles.append(str(rec.get("title", "")).strip() or f"document-{i + 1}")
            doc_ids.append(str(rec.get("id", "")).strip() or f"doc-{i + 1}")
            if "relevant" not in rec:
                raise CorpusError(f"第 {i + 1} 行缺少 relevant 相关性标签。")
            labels.append(_parse_label(rec["relevant"], i + 1))

        if not any(labels):
            raise CorpusError("语料中没有正例（relevant=true）文档，无法学习扩展词。")
        if all(labels):
            raise CorpusError("语料中没有负例（relevant=false）文档，无法评估误检率。")

        vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,                 # tokenize already lowercases
            min_df=2,
            sublinear_tf=True,
        )
        tfidf = vectorizer.fit_transform(abstracts)
        bool_mat = (tfidf > 0).astype(int)
        vocab = list(vectorizer.get_feature_names_out())
        index_by_term = vectorizer.vocabulary_
        # docs_by_term is computed from the same matrix so boolean coverage
        # and TF-IDF ranking share exactly one vocabulary.
        csr = bool_mat.tocsr()
        docs_by_term: list[set[int]] = [set() for _ in vocab]
        for j in range(csr.shape[1]):
            docs_by_term[j] = set(csr.getcol(j).nonzero()[0].tolist())

        return cls(
            abstracts=abstracts,
            labels=labels,
            titles=titles,
            doc_ids=doc_ids,
            vectorizer=vectorizer,
            tfidf=tfidf,
            bool_mat=bool_mat,
            vocab=vocab,
            index_by_term=index_by_term,
            docs_by_term=docs_by_term,
            n_relevant=sum(labels),
            n_nonrelevant=len(labels) - sum(labels),
        )

    @classmethod
    def from_file(cls, raw: bytes, filename: str) -> "Corpus":
        text = raw.decode("utf-8-sig", errors="strict")
        name = filename.lower()
        if name.endswith(".jsonl") or name.endswith(".ndjson"):
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif name.endswith(".json"):
            data = json.loads(text)
            if not isinstance(data, list):
                raise CorpusError("JSON 语料必须是对象数组。")
            records = data
        elif name.endswith((".csv", ".tsv", ".txt")):
            delim = "\t" if name.endswith(".tsv") else ","
            reader = csv.DictReader(io.StringIO(text), delimiter=delim)
            if reader.fieldnames is None:
                raise CorpusError("文件缺少表头。")
            records = [dict(row) for row in reader]
        else:
            raise CorpusError(
                "不支持的文件格式，请上传 .csv / .tsv / .json / .jsonl 文件。"
            )
        return cls.from_records(records)

    # -- basic accessors --------------------------------------------------- #
    def term_index(self, term: str) -> Optional[int]:
        return self.index_by_term.get(term.lower())

    def query_terms(self, query: str) -> list[str]:
        """Vocabulary terms hit by the raw query string (order preserved)."""
        seen: set[str] = set()
        out: list[str] = []
        for tok in tokenize(query):
            if tok in self.index_by_term and tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out

    def unknown_query_tokens(self, query: str) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for tok in tokenize(query):
            if tok not in self.index_by_term and tok not in seen:
                seen.add(tok)
                out.append(tok)
        return out

    def docs_for_terms(self, terms: Iterable[str]) -> set[int]:
        hits: set[int] = set()
        for term in terms:
            j = self.term_index(term)
            if j is not None:
                hits |= self.docs_by_term[j]
        return hits

    def metrics_for_docs(self, docs: set[int]) -> dict:
        n_retrieved = len(docs)
        tp = sum(1 for d in docs if self.labels[d])
        fp = n_retrieved - tp
        return {
            "n_retrieved": n_retrieved,
            "true_positives": tp,
            "false_positives": fp,
            "recall": tp / self.n_relevant if self.n_relevant else 0.0,
            "fdr": fp / n_retrieved if n_retrieved else 0.0,
        }


def _parse_label(value, row: int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if int(value) in (0, 1):
            return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "relevant", "rel", "p", "positive"}:
        return True
    if text in {"0", "false", "f", "no", "n", "not_relevant", "irrelevant",
                "non_relevant", "neg", "negative"}:
        return False
    raise CorpusError(f"第 {row} 行的 relevant 标签无法解析：{value!r}")


# --------------------------------------------------------------------------- #
# Candidate extraction
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    term: str
    n_relevant_hits: int
    n_nonrelevant_hits: int
    score: float
    reason: str


def extract_candidates(
    corpus: Corpus,
    baseline_terms: list[str],
    excluded: set[str],
    max_candidates: int = 40,
) -> list[Candidate]:
    """Positive-example candidate terms.

    A term is eligible when it occurs in at least one relevant document,
    occurs in >= 2 documents overall (already guaranteed by min_df=2),
    is not part of the baseline query and is not user-excluded.
    Scoring favours terms that cover many relevant docs while polluting few
    non-relevant ones (a transparent relevance-frequency purity score).
    """
    baseline = set(baseline_terms)
    candidates: list[Candidate] = []
    rel_doc_count = corpus.n_relevant
    nonrel_count = corpus.n_nonrelevant
    for term, j in corpus.index_by_term.items():
        if term in baseline or term in excluded:
            continue
        docs = corpus.docs_by_term[j]
        rel_hits = sum(1 for d in docs if corpus.labels[d])
        if rel_hits == 0:
            continue  # candidates must come from positive examples
        nonrel_hits = len(docs) - rel_hits
        rel_ratio = rel_hits / rel_doc_count
        purity = rel_hits / (len(docs) + 1e-9)
        # IDF-like bonus for discriminative terms
        idf = math.log((len(corpus.labels) + 1) / (len(docs) + 1)) + 1
        score = rel_ratio * purity * idf
        candidates.append(
            Candidate(
                term=term,
                n_relevant_hits=rel_hits,
                n_nonrelevant_hits=nonrel_hits,
                score=score,
                reason=(
                    f"出现在 {rel_hits} 篇正例、{nonrel_hits} 篇负例中，"
                    f"正例覆盖率 {rel_ratio:.0%}，纯度 {purity:.0%}"
                ),
            )
        )
    candidates.sort(key=lambda c: (-c.score, -c.n_relevant_hits, c.term))
    return candidates[:max_candidates]


# --------------------------------------------------------------------------- #
# Expansion search
# --------------------------------------------------------------------------- #
@dataclass
class SearchState:
    terms: tuple[str, ...]
    docs: set[int]
    metrics: dict


def _within_limits(metrics: dict, max_fdr: float,
                   max_fp: Optional[int]) -> bool:
    if metrics["fdr"] > max_fdr + 1e-12:
        return False
    if max_fp is not None and metrics["false_positives"] > max_fp:
        return False
    return True


def _beam_search(
    corpus: Corpus,
    candidates: list[Candidate],
    locked: list[str],
    baseline_docs: set[int],
    min_recall: float,
    max_fdr: float,
    max_fp: Optional[int],
    beam_width: int,
    max_terms: int,
) -> tuple[Optional[SearchState], list[SearchState], bool, Optional[str]]:
    """Return (best satisfying state, all explored frontier states,
    budget_exhausted, lock_violation_message)."""
    initial = SearchState(terms=(), docs=set(baseline_docs),
                          metrics=corpus.metrics_for_docs(baseline_docs))
    # Force locked terms first — they are non-negotiable user decisions.
    current = initial
    lock_violation: Optional[str] = None
    for term in locked:
        new_docs = current.docs | corpus.docs_for_terms([term])
        metrics = corpus.metrics_for_docs(new_docs)
        if not _within_limits(metrics, max_fdr, max_fp):
            lock_violation = (
                f"锁定词 {term!r} 单独加入后即突破误检上限"
                f"（fdr={metrics['fdr']:.3f}, fp={metrics['false_positives']}）"
            )
            break
        current = SearchState(
            terms=current.terms + (term,), docs=new_docs, metrics=metrics
        )
    if lock_violation:
        return None, [current], False, lock_violation

    if current.metrics["recall"] >= min_recall - 1e-12 and _within_limits(
        current.metrics, max_fdr, max_fp
    ):
        return current, [current], False, None

    candidate_terms = [c.term for c in candidates if c.term not in locked]
    frontier = [current]
    explored: list[SearchState] = [current]
    best: Optional[SearchState] = None
    budget_exhausted = False

    for _depth in range(max_terms - len(locked)):
        children: list[SearchState] = []
        for state in frontier:
            existing = set(state.terms)
            for term in candidate_terms:
                if term in existing:
                    continue
                new_docs = state.docs | corpus.docs_for_terms([term])
                metrics = corpus.metrics_for_docs(new_docs)
                if not _within_limits(metrics, max_fdr, max_fp):
                    continue
                child = SearchState(
                    terms=state.terms + (term,), docs=new_docs, metrics=metrics
                )
                children.append(child)
                explored.append(child)
                if metrics["recall"] >= min_recall - 1e-12:
                    if best is None or (
                        child.metrics["fdr"],
                        child.metrics["false_positives"],
                        len(child.terms),
                    ) < (
                        best.metrics["fdr"],
                        best.metrics["false_positives"],
                        len(best.terms),
                    ):
                        best = child
        if best is not None:
            return best, explored, False, None
        if not children:
            break
        # beam pruning: prefer high recall, then low fdr, then fewest terms
        children.sort(
            key=lambda s: (
                -s.metrics["recall"],
                s.metrics["fdr"],
                s.metrics["false_positives"],
                len(s.terms),
            )
        )
        # dedupe by term set
        deduped: list[SearchState] = []
        seen_sets: set[frozenset] = set()
        for s in children:
            key = frozenset(s.terms)
            if key not in seen_sets:
                seen_sets.add(key)
                deduped.append(s)
        frontier = deduped[:beam_width]
        # Pruning is only relevant to the failure when some pruned branch
        # reached a strictly better recall than every kept branch; otherwise
        # the cap is set by the data, not the beam width.
        if len(deduped) > beam_width:
            kept_best_recall = deduped[0].metrics["recall"]
            pruned_best_recall = max(
                s.metrics["recall"] for s in deduped[beam_width:]
            )
            if pruned_best_recall > kept_best_recall + 1e-12:
                budget_exhausted = True
    return best, explored, budget_exhausted, None


def _diagnose_failure(
    corpus: Corpus,
    candidates: list[Candidate],
    locked: list[str],
    baseline_docs: set[int],
    min_recall: float,
    max_fdr: float,
    max_fp: Optional[int],
    explored: list[SearchState],
    budget_exhausted: bool,
    lock_violation: Optional[str] = None,
) -> dict:
    """Explain *why* the thresholds could not be met."""
    # 1. Unconstrained best: add every candidate + locked term and see the
    #    maximum recall reachable without any false-detection limit.
    all_terms = [c.term for c in candidates] + [
        t for t in locked if t not in {c.term for c in candidates}
    ]
    ceiling_docs = baseline_docs | corpus.docs_for_terms(all_terms)
    ceiling = corpus.metrics_for_docs(ceiling_docs)

    # 2. For the unconstrained ceiling, which relevant docs remain missing?
    relevant_idx = [i for i, lab in enumerate(corpus.labels) if lab]
    missing_relevant = [i for i in relevant_idx if i not in ceiling_docs]

    # 3. Best recall achievable *within* the fdr / fp limits.
    within = [
        s.metrics["recall"]
        for s in explored
        if _within_limits(s.metrics, max_fdr, max_fp)
    ]
    best_feasible_recall = max(within) if within else 0.0

    # 4. Find the cheapest terms (in fdr cost) that would lift recall.
    #    Report the first term whose addition pushes past the limit.
    blocker: Optional[dict] = None
    current_docs = set(baseline_docs) | corpus.docs_for_terms(locked)
    remaining = [c for c in candidates if c.term not in locked]
    # Deterministic ordering: fewest non-relevant hits first (absolute
    # pollution), then most relevant hits, then lexicographic term name.
    remaining_sorted = sorted(
        remaining,
        key=lambda c: (
            c.n_nonrelevant_hits,
            -c.n_relevant_hits,
            c.term,
        ),
    )
    for cand in remaining_sorted:
        trial_docs = current_docs | corpus.docs_for_terms([cand.term])
        trial = corpus.metrics_for_docs(trial_docs)
        if not _within_limits(trial, max_fdr, max_fp):
            blocker = {
                "term": cand.term,
                "would_add_relevant": sum(
                    1
                    for d in corpus.docs_for_terms([cand.term])
                    if corpus.labels[d] and d not in current_docs
                ),
                "would_add_nonrelevant": sum(
                    1
                    for d in corpus.docs_for_terms([cand.term])
                    if not corpus.labels[d] and d not in current_docs
                ),
                "resulting_fdr": round(trial["fdr"], 4),
                "resulting_false_positives": trial["false_positives"],
            }
            break
        current_docs = trial_docs

    reasons: list[str] = []
    if lock_violation:
        reasons.append(lock_violation + "。请解除该锁定词或放宽误检上限。")
    if ceiling["recall"] + 1e-12 < min_recall:
        reasons.append(
            f"词表覆盖上限不足：即使加入全部 {len(all_terms)} 个候选词，"
            f"最高召回也只有 {ceiling['recall']:.1%}，"
            f"低于召回下限 {min_recall:.1%}。"
        )
        if missing_relevant:
            reasons.append(
                "有 "
                + str(len(missing_relevant))
                + " 篇正例摘要不含任何候选词（min_df=2 的词表无法覆盖其独有词），"
                  "建议补充语料或放宽词频限制。"
            )
    else:
        reasons.append(
            f"召回目标理论上可达（无约束最高召回 {ceiling['recall']:.1%}），"
            f"但在误检上限 fdr<={max_fdr:.1%}"
            + (f" 且误检数<={max_fp}" if max_fp is not None else "")
            + f" 下最高只能达到召回 {best_feasible_recall:.1%}，"
              "提升召回所需的词都会引入过多负例。"
        )
    if blocker:
        reasons.append(
            f"首个越界词为 {blocker['term']!r}：加入后新增正例 "
            f"{blocker['would_add_relevant']} 篇、负例 "
            f"{blocker['would_add_nonrelevant']} 篇，"
            f"fdr 升至 {blocker['resulting_fdr']:.3f}"
            + (
                f"、误检数 {blocker['resulting_false_positives']}"
                if max_fp is not None
                else ""
            )
            + "。可适当上调误检上限后重试。"
        )
    if budget_exhausted and best_feasible_recall + 1e-12 < min_recall:
        reasons.append(
            "候选组合数超过束宽（beam_width）已被剪枝，可增大 beam_width 再试。"
        )

    return {
        "unconstrained_max_recall": round(ceiling["recall"], 4),
        "best_feasible_recall": round(best_feasible_recall, 4),
        "missing_relevant_docs": [
            {"id": corpus.doc_ids[i], "title": corpus.titles[i]}
            for i in missing_relevant[:10]
        ],
        "blocker_term": blocker,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Per-term trace + ranking
# --------------------------------------------------------------------------- #
def _build_trace(
    corpus: Corpus,
    baseline_terms: list[str],
    selected_terms: list[str],
) -> list[dict]:
    """Coverage change contributed by each term, applied in selection order."""
    trace: list[dict] = []
    covered = corpus.docs_for_terms(baseline_terms)
    running = corpus.metrics_for_docs(covered)
    trace.append({
        "term": "（基线查询）",
        "is_baseline": True,
        "added_relevant": 0,
        "added_nonrelevant": 0,
        "cumulative_retrieved": running["n_retrieved"],
        "cumulative_recall": round(running["recall"], 4),
        "cumulative_fdr": round(running["fdr"], 4),
        "cumulative_false_positives": running["false_positives"],
        "note": "初始查询命中的文档集合",
    })
    for term in selected_terms:
        term_docs = corpus.docs_for_terms([term])
        added = term_docs - covered
        added_rel = sum(1 for d in added if corpus.labels[d])
        added_nonrel = len(added) - added_rel
        covered |= term_docs
        m = corpus.metrics_for_docs(covered)
        trace.append({
            "term": term,
            "is_baseline": False,
            "added_relevant": added_rel,
            "added_nonrelevant": added_nonrel,
            "cumulative_retrieved": m["n_retrieved"],
            "cumulative_recall": round(m["recall"], 4),
            "cumulative_fdr": round(m["fdr"], 4),
            "cumulative_false_positives": m["false_positives"],
            "note": (
                f"新增正例 {added_rel} 篇、负例 {added_nonrel} 篇；"
                f"累计召回 {m['recall']:.1%}，误检率 {m['fdr']:.1%}"
            ),
        })
    return trace


def _rank_documents(
    corpus: Corpus,
    query_terms: list[str],
    retrieved: set[int],
    top_k: int,
) -> list[dict]:
    """TF-IDF cosine ranking of the retrieved set for the expanded query."""
    if not query_terms:
        ranked = sorted(retrieved)
    else:
        qvec = corpus.vectorizer.transform([" ".join(query_terms)])
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(qvec, corpus.tfidf).ravel()
        ranked = sorted(retrieved, key=lambda d: (-sims[d], d))
    rows = []
    for rank, d in enumerate(ranked[:top_k], start=1):
        rows.append({
            "rank": rank,
            "id": corpus.doc_ids[d],
            "title": corpus.titles[d],
            "relevant": corpus.labels[d],
            "abstract": corpus.abstracts[d],
            "matched_terms": [
                t for t in query_terms if d in corpus.docs_for_terms([t])
            ],
            "score": round(float(sims[d]), 6) if query_terms else 0.0,
        })
    return rows


# --------------------------------------------------------------------------- #
# Public analysis entry point
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisRequest:
    query: str
    min_recall: float
    max_fdr: float
    max_false_positives: Optional[int] = None
    locked_terms: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)
    beam_width: int = 8
    max_terms: int = 12
    top_k: int = 50


def analyze(corpus: Corpus, req: AnalysisRequest) -> dict:
    # ---- validate thresholds ------------------------------------------- #
    if not 0.0 <= req.min_recall <= 1.0:
        raise ThresholdError("召回下限必须在 [0, 1] 区间内。")
    if not 0.0 <= req.max_fdr <= 1.0:
        raise ThresholdError("误检率上限必须在 [0, 1] 区间内。")
    if req.max_false_positives is not None and req.max_false_positives < 0:
        raise ThresholdError("误检数量上限不能为负数。")
    if req.beam_width < 1 or req.max_terms < 1:
        raise ThresholdError("beam_width 与 max_terms 必须为正整数。")

    locked = [t.lower() for t in dict.fromkeys(req.locked_terms) if t.strip()]
    excluded = {t.lower() for t in req.excluded_terms if t.strip()}
    conflict = sorted(set(locked) & excluded)
    if conflict:
        raise ThresholdError(
            f"词同时被锁定和排除：{', '.join(conflict)}，请先解除其中一种设置。"
        )
    unknown_locked = [t for t in locked if corpus.term_index(t) is None]
    if unknown_locked:
        raise ThresholdError(
            f"锁定词不在语料词表中：{', '.join(unknown_locked)}。"
        )
    if len(locked) > req.max_terms:
        raise ThresholdError(
            f"锁定词数量（{len(locked)}）超过最多扩展词上限 max_terms="
            f"{req.max_terms}，请解除部分锁定或调大 max_terms。"
        )

    baseline_terms = corpus.query_terms(req.query)
    unknown_tokens = corpus.unknown_query_tokens(req.query)
    baseline_docs = corpus.docs_for_terms(baseline_terms)
    baseline_metrics = corpus.metrics_for_docs(baseline_docs)

    candidates = extract_candidates(corpus, baseline_terms, excluded)
    # Locked terms that are not ordinary candidates still enter the search;
    # surface them in the candidate list for UI visibility.
    candidate_terms = {c.term for c in candidates}
    extra_locked = []
    for term in locked:
        if term not in candidate_terms and term not in baseline_terms:
            docs = corpus.docs_for_terms([term])
            rel = sum(1 for d in docs if corpus.labels[d])
            extra_locked.append(Candidate(
                term=term,
                n_relevant_hits=rel,
                n_nonrelevant_hits=len(docs) - rel,
                score=0.0,
                reason="用户锁定词（非自动候选）",
            ))
    all_candidates = candidates + extra_locked

    best, explored, budget_exhausted, lock_violation = _beam_search(
        corpus, candidates + extra_locked, locked, baseline_docs,
        req.min_recall, req.max_fdr, req.max_false_positives,
        req.beam_width, req.max_terms,
    )

    common = {
        "corpus_size": len(corpus.labels),
        "n_relevant": corpus.n_relevant,
        "n_nonrelevant": corpus.n_nonrelevant,
        "query": req.query,
        "baseline_terms": baseline_terms,
        "unknown_query_tokens": unknown_tokens,
        "thresholds": {
            "min_recall": req.min_recall,
            "max_fdr": req.max_fdr,
            "max_false_positives": req.max_false_positives,
        },
        "locked_terms": locked,
        "excluded_terms": sorted(excluded),
        "baseline_metrics": baseline_metrics,
        "candidate_pool": [
            {
                "term": c.term,
                "score": round(c.score, 6),
                "n_relevant_hits": c.n_relevant_hits,
                "n_nonrelevant_hits": c.n_nonrelevant_hits,
                "reason": c.reason,
                "locked": c.term in locked,
                "excluded": c.term in excluded,
            }
            for c in all_candidates
        ],
    }

    if best is None:
        diagnosis = _diagnose_failure(
            corpus, candidates + extra_locked, locked, baseline_docs,
            req.min_recall, req.max_fdr, req.max_false_positives,
            explored, budget_exhausted, lock_violation,
        )
        diagnosis["lock_violation"] = lock_violation
        # Best-effort fallback: only locked terms that were actually applied
        # before a violation (or all locked terms when the failure happened
        # later in the search).
        fallback_terms = list(explored[0].terms) if explored else []
        fallback_docs = baseline_docs | corpus.docs_for_terms(fallback_terms)
        return {
            **common,
            "status": Status.FAILED.value,
            "failure": diagnosis,
            "selected_terms": fallback_terms,
            "expanded_terms": [t for t in fallback_terms if t not in baseline_terms],
            "trace": _build_trace(corpus, baseline_terms, fallback_terms),
            "metrics": corpus.metrics_for_docs(fallback_docs),
            "ranking": _rank_documents(
                corpus, baseline_terms + fallback_terms, fallback_docs, req.top_k
            ),
        }

    selected = list(best.terms)
    expanded_terms = [t for t in selected if t not in baseline_terms]
    final_docs = baseline_docs | corpus.docs_for_terms(selected)
    return {
        **common,
        "status": Status.SATISFIED.value,
        "failure": None,
        "selected_terms": selected,
        "expanded_terms": expanded_terms,
        "trace": _build_trace(corpus, baseline_terms, selected),
        "metrics": corpus.metrics_for_docs(final_docs),
        "ranking": _rank_documents(
            corpus, baseline_terms + selected, final_docs, req.top_k
        ),
    }
