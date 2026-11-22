"""Delivery self-test: threshold-conflict and revision scenarios.

Runs the full HTTP stack in-process (FastAPI TestClient) against the demo
corpus and writes a Markdown report with the *measured* numbers to
reports/selftest_report.md.  Exits non-zero if any expectation breaks.

Scenarios covered
-----------------
S1  Baseline analysis: satisfied expansion with per-term coverage trace.
S2  Threshold conflict: contradictory recall/fdr limits -> failure +
    structured diagnosis (ceiling recall, blocker term, reasons).
S3  FDR relaxation resolves the same conflict (proves diagnosis is accurate).
S4  Revision — exclude terms: solution is recomputed without them and the
    old suggestion id is rejected with HTTP 409.
S5  Revision — lock a term: it is forced into every new solution.
S6  Lock/exclude contradiction is rejected; unlock restores feasibility.
S7  Intermediate fdr overflow is repaired (synthetic corpus): the only
    full-recall combination passes through an over-limit partial state; it
    must still be found, both free and with the polluting term locked.
S8  Beam-width truncation: a narrow beam misses a solution a wide beam finds;
    the failure must be reported as "inconclusive", never "impossible".
S9  Proven infeasibility: an exhaustive search with no solution must be
    reported as proven, with a safe recall upper bound and the best found.
"""

from __future__ import annotations

import itertools
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.engine import Corpus, AnalysisRequest
from app.engine import analyze as engine_analyze
from app.main import app

CORPUS_PATH = ROOT / "data" / "demo_corpus.csv"
REPORT_DIR = ROOT / "reports"

client = TestClient(app)
results: list[dict] = []


def record(scenario: str, ok: bool, detail: str, payload=None) -> None:
    results.append({"scenario": scenario, "ok": ok, "detail": detail,
                    "payload": payload or {}})
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {scenario}: {detail}")


def upload_demo() -> str:
    raw = CORPUS_PATH.read_bytes()
    resp = client.post(
        "/api/sessions",
        files={"file": ("demo_corpus.csv", raw, "text/csv")},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def analyze(sid: str, **overrides) -> dict:
    body = {
        "session_id": sid,
        "query": "quantum error correction",
        # 15/17 relevant docs are reachable (0.882); use 0.85 so the
        # satisfied path is exercised, and 0.95+ for ceiling failures.
        "min_recall": 0.85,
        "max_fdr": 0.2,
    }
    body.update(overrides)
    resp = client.post("/api/analyze", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _dilution_corpus() -> Corpus:
    """Baseline 'alpha'; *every* first expansion term starts over the cap.

    apoll alone / bpoll alone -> fdr 0.43 (> 0.35 cap), apoll+bpoll -> 0.33
    with full recall.  The only viable combination therefore crosses an
    over-limit intermediate state.
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


def _beam_truncation_corpus() -> Corpus:
    """beam_width=1 misses the winning pair; beam_width=8 finds it."""
    records = [
        ("d0", "t2 t5", True), ("d1", "t5 t0 alpha", True),
        ("d2", "t5 t0", True), ("d3", "t1 t4 t3", True),
        ("d4", "t1 t3", True), ("d5", "t3 t5 alpha", True),
        ("d6", "t5 t1 t0 alpha", False), ("d7", "t2 t4", False),
        ("d8", "t3 t5", False), ("d9", "t0 t4 t1", False),
        ("d10", "t4 t5", False), ("d11", "t4 t5", False),
    ]
    return Corpus.from_records([
        {"id": i, "title": "t", "abstract": a, "relevant": r}
        for i, a, r in records
    ])


def main() -> int:
    failures = 0

    # --------------------------------------------------------------- S1
    sid = upload_demo()
    s1 = analyze(sid)
    ok = (
        s1["status"] == "satisfied"
        and s1["metrics"]["recall"] >= 0.85 - 1e-9
        and s1["metrics"]["fdr"] <= 0.2 + 1e-9
        and len(s1["trace"]) == 1 + len(s1["expanded_terms"])
    )
    failures += 0 if ok else 1
    record(
        "S1 基线扩展分析（满足阈值）", ok,
        f"扩展词={s1['expanded_terms']}；召回 "
        f"{s1['baseline_metrics']['recall']:.3f}→{s1['metrics']['recall']:.3f}；"
        f"fdr {s1['baseline_metrics']['fdr']:.3f}→{s1['metrics']['fdr']:.3f}；"
        f"trace 行数={len(s1['trace'])}",
        payload=s1,
    )

    # --------------------------------------------------------------- S2
    s2 = analyze(sid, min_recall=1.0, max_fdr=0.05)
    f = s2.get("failure") or {}
    # In this corpus the recall ceiling is itself below 100% (two relevant
    # abstracts use document-unique words), so S2 demonstrates the
    # vocabulary-coverage failure mode under strict thresholds.
    ok = (
        s2["status"] == "failed"
        and bool(f.get("reasons"))
        and "unconstrained_max_recall" in f
        and f["unconstrained_max_recall"] < 1.0
    )
    failures += 0 if ok else 1
    blocker = f.get("blocker_term") or {}
    record(
        "S2 阈值冲突（召回 100% & fdr≤5%，词表上限不足）", ok,
        f"状态={s2['status']}；无约束最高召回={f.get('unconstrained_max_recall')}；"
        f"约束内最高召回={f.get('best_feasible_recall')}；"
        f"无法覆盖正例={[d['id'] for d in f.get('missing_relevant_docs', [])]}；"
        f"诊断条数={len(f.get('reasons', []))}",
        payload=s2,
    )

    # --------------------------------------------------------------- S2b
    # A conflict where the recall target is reachable but the fdr budget is
    # not: clean terms reach 0.882 recall, and the remaining polluted terms
    # needed beyond it push fdr over a 1% budget.
    s2b = analyze(sid, min_recall=0.85, max_fdr=0.01)
    fb = s2b.get("failure") or {}
    ok = (
        s2b["status"] == "failed"
        and fb.get("unconstrained_max_recall", 0) >= 0.85 - 1e-12
        and fb.get("best_feasible_recall", 1) < 0.85
        and any("误检" in r for r in fb.get("reasons", []))
    )
    failures += 0 if ok else 1
    bb = fb.get("blocker_term") or {}
    record(
        "S2b 阈值冲突（召回可达但误检预算不足）", ok,
        f"状态={s2b['status']}；无约束最高召回={fb.get('unconstrained_max_recall')}；"
        f"约束内最高召回={fb.get('best_feasible_recall')}；"
        f"首个越界词={bb.get('term')}（新增正例 {bb.get('would_add_relevant')} / "
        f"负例 {bb.get('would_add_nonrelevant')}，fdr→{bb.get('resulting_fdr')}）",
        payload=s2b,
    )

    # --------------------------------------------------------------- S3
    # Pair with S2b: relaxing the fdr budget to 40% removes the conflict.
    s3 = analyze(sid, min_recall=0.85, max_fdr=0.4)
    ok = s3["status"] == "satisfied" and s3["metrics"]["recall"] >= 0.85 - 1e-12
    failures += 0 if ok else 1
    record(
        "S3 放宽误检上限至 40% 后 S2b 冲突消解", ok,
        f"状态={s3['status']}；召回={s3['metrics']['recall']:.3f}；"
        f"fdr={s3['metrics']['fdr']:.3f}；FP={s3['metrics']['false_positives']}",
        payload=s3,
    )

    # --------------------------------------------------------------- S4
    old_suggestion = s1["suggestion_id"]
    excluded_now = []
    rev_info = None
    for term in s1["expanded_terms"]:
        resp = client.post("/api/revise", json={
            "session_id": sid, "exclude": [term],
        })
        assert resp.status_code == 200, resp.text
        excluded_now.append(term)
        rev_info = resp.json()
    assert rev_info is not None, "S1 produced no expansion terms to exclude"
    stale = client.post("/api/analyze", json={
        "session_id": sid, "query": "quantum error correction",
        "min_recall": 0.85, "max_fdr": 0.2,
        "suggestion_id": old_suggestion,
    })
    s4 = analyze(sid)  # fresh call without stale id
    # With every effective term excluded, the solver must never reuse them;
    # dropping back to baseline recall (and possibly failing the threshold)
    # is the correct, expected consequence.
    ok = (
        stale.status_code == 409
        and rev_info["revision"] >= 1
        and not (set(s4["expanded_terms"]) & set(excluded_now))
        and s4["metrics"]["recall"] <= s1["metrics"]["recall"] + 1e-9
    )
    failures += 0 if ok else 1
    record(
        "S4 修订-排除词后重算 & 旧建议 409 失效", ok,
        f"排除={excluded_now}；旧建议 HTTP={stale.status_code}；"
        f"新修订号={rev_info['revision']}；新扩展词={s4['expanded_terms']}；"
        f"状态={s4['status']}；召回={s4['metrics']['recall']:.3f}"
        f"（排除全部有效词后回落到基线，符合预期）",
        payload={"stale": stale.json(), "fresh": s4},
    )

    # --------------------------------------------------------------- S5
    # Use a fresh session: in S4 we excluded decoherence, so locking it
    # there would (correctly) raise a lock/exclude contradiction.
    sid2 = upload_demo()
    lock_term = "decoherence"
    resp = client.post("/api/revise", json={
        "session_id": sid2, "lock": [lock_term],
    })
    assert resp.status_code == 200, resp.text
    s5 = analyze(sid2, min_recall=0.5, max_fdr=0.5)
    ok = lock_term in s5["selected_terms"]
    failures += 0 if ok else 1
    record(
        "S5 修订-锁定词强制入选（独立会话）", ok,
        f"锁定={lock_term}；selected_terms={s5['selected_terms']}；"
        f"召回={s5['metrics']['recall']:.3f}；fdr={s5['metrics']['fdr']:.3f}",
        payload=s5,
    )

    # --------------------------------------------------------------- S6
    conflict = client.post("/api/revise", json={
        "session_id": sid2, "exclude": [lock_term],
    })
    unlock = client.post("/api/revise", json={
        "session_id": sid2, "unlock": [lock_term],
    })
    s6 = analyze(sid2, min_recall=0.5, max_fdr=0.5)
    # The real assertion: contradiction rejected; after unlock term no longer
    # forced (it may still be picked naturally, so check locked list instead).
    ok = conflict.status_code == 400 and unlock.json()["locked_terms"] == []
    failures += 0 if ok else 1
    record(
        "S6 锁定/排除矛盾拒绝 & 解除锁定", ok,
        f"矛盾请求 HTTP={conflict.status_code}（{conflict.json().get('detail')}）；"
        f"解锁后 locked_terms={unlock.json()['locked_terms']}；"
        f"当前修订号={unlock.json()['revision']}",
        payload={"conflict": conflict.json(), "unlock": unlock.json()},
    )

    # --------------------------------------------------------------- S7
    # Intermediate fdr overflow that a later term repairs.  Ground truth is
    # checked by brute force over all candidate combinations.
    dil = _dilution_corpus()
    brute_ok = []
    for lock in ([], ["apoll"]):
        cap = 0.35
        base_docs = dil.docs_for_terms(dil.query_terms("alpha"))
        lock_docs = dil.docs_for_terms(lock)
        best = None
        # exhaustive check over the small candidate vocabulary
        all_cands = [t for t in dil.index_by_term
                     if t not in dil.query_terms("alpha")]
        for k in range(0, len(all_cands) + 1):
            for combo in itertools.combinations(all_cands, k):
                if any(t in lock for t in combo):
                    continue
                m = dil.metrics_for_docs(
                    base_docs | lock_docs | dil.docs_for_terms(combo)
                )
                if m["recall"] >= 1.0 - 1e-12 and m["fdr"] <= cap + 1e-12:
                    best = combo
        req = AnalysisRequest(
            query="alpha", min_recall=1.0, max_fdr=cap, locked_terms=lock,
        )
        res = engine_analyze(dil, req)
        trace_fdr = {row["term"]: row["cumulative_fdr"] for row in res["trace"]}
        # Verify the trace itself shows an over-limit step: the reported
        # solution must be reachable only *through* an over-limit prefix.
        over_limit_step = any(
            row["cumulative_fdr"] > cap + 1e-12 for row in res["trace"]
        )
        ok = (
            res["status"] == "satisfied"
            and res["metrics"]["recall"] == 1.0
            and res["metrics"]["fdr"] <= cap + 1e-12
            and best is not None
            and over_limit_step
        )
        brute_ok.append(ok)
        failures += 0 if ok else 1
        record(
            f"S7{'' if not lock else 'b'} 中途误检超标后补足正例（"
            f"{'锁定 apoll' if lock else '自由搜索'}）", ok,
            f"锁定={lock}；真值组合={best}；扩展词={res['expanded_terms']}；"
            f"召回={res['metrics']['recall']:.3f}；fdr={res['metrics']['fdr']:.3f}"
            f"（上限 {cap:.0%}）；trace 中出现中途超标="
            f"{any(row['cumulative_fdr'] > cap + 1e-12 for row in res['trace'])}"
            f"（apoll 中途 fdr={trace_fdr.get('apoll', 'n/a')}）",
            payload=res,
        )

    # --------------------------------------------------------------- S8
    trunc = _beam_truncation_corpus()
    narrow = engine_analyze(trunc, AnalysisRequest(
        query="alpha", min_recall=0.8, max_fdr=0.3, max_terms=2, beam_width=1,
    ))
    wide = engine_analyze(trunc, AnalysisRequest(
        query="alpha", min_recall=0.8, max_fdr=0.3, max_terms=2, beam_width=8,
    ))
    fd = narrow.get("failure") or {}
    s8_ok = (
        narrow["status"] == "failed"
        and wide["status"] == "satisfied"
        and fd.get("verdict") == "search_inconclusive"
        and fd.get("proven_infeasible") is False
        and fd.get("branches_pruned") is True
        and fd.get("recall_upper_bound", 0) >= 0.8
    )
    failures += 0 if s8_ok else 1
    record(
        "S8 束宽截断 -> 尚无定论（而非证实无法达标）", s8_ok,
        f"beam_width=1 状态={narrow['status']}，判定={fd.get('verdict')}，"
        f"剪枝={fd.get('branches_pruned')}，召回上界={fd.get('recall_upper_bound')}；"
        f"beam_width=8 状态={wide['status']}，扩展词={wide['expanded_terms']}，"
        f"召回={wide['metrics']['recall']:.3f}",
        payload={"narrow": narrow, "wide": wide},
    )

    # --------------------------------------------------------------- S9
    # Demo corpus, full recall with a tiny fdr budget: the vocabulary ceiling
    # makes this provably impossible; best-found must be reported separately.
    s9 = analyze(sid, min_recall=1.0, max_fdr=0.05)
    f9 = s9.get("failure") or {}
    bf9 = s9.get("best_found")
    s9_ok = (
        s9["status"] == "failed"
        and f9.get("verdict") == "proven_infeasible"
        and f9.get("search_complete") is True
        and f9.get("recall_upper_bound", 1) < 1.0
        and bf9 is not None
        and bf9["metrics"]["fdr"] <= 0.05 + 1e-12
        and bf9["metrics"]["recall"] < 1.0
    )
    failures += 0 if s9_ok else 1
    record(
        "S9 穷尽搜索 -> 已证实无法达标 + 满足限制的最高召回", s9_ok,
        f"判定={f9.get('verdict')}；搜索完成={f9.get('search_complete')}；"
        f"召回上界={f9.get('recall_upper_bound')}；"
        f"满足限制的最高召回={f9.get('best_feasible_recall')}（组合 "
        f"{bf9['expanded_terms'] if bf9 else None}）；"
        f"无约束词表召回={f9.get('unconstrained_max_recall')}",
        payload=s9,
    )

    write_report(s1, s2, s2b, s3, s4, s5, s6, stale, failures,
                 s7=(dil, brute_ok[0], brute_ok[1]), s8=(narrow, wide), s9=s9)
    print(f"\n{len(results) - failures}/{len(results)} scenarios passed.")
    return 1 if failures else 0


def _metrics_row(title: str, data: dict) -> str:
    m = data["metrics"]
    b = data["baseline_metrics"]
    return (
        f"| {title} | {data['status']} | {b['recall']:.3f} | {m['recall']:.3f} "
        f"| {b['fdr']:.3f} | {m['fdr']:.3f} | {m['n_retrieved']} "
        f"| {m['true_positives']} | {m['false_positives']} |"
    )


def write_report(s1, s2, s2b, s3, s4, s5, s6, stale, failures,
                 s7=None, s8=None, s9=None) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / "selftest_report.md"
    lines: list[str] = []
    lines.append("# 交付前自测报告（阈值冲突、束宽与诊断场景）")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append("- 语料：`data/demo_corpus.csv`（45 篇，正例 17 / 负例 28；"
                 "其中 2 篇正例只含文档级独有词，用于触发词表覆盖上限）")
    lines.append("- 初始查询：`quantum error correction`")
    lines.append("- 方式：FastAPI TestClient 全栈在进程内调用，无外部网络")
    lines.append("")
    lines.append("## 场景结果汇总")
    lines.append("")
    lines.append("| 场景 | 结果 | 实测细节 |")
    lines.append("|---|---|---|")
    for r in results:
        lines.append(f"| {r['scenario']} | {'✅ PASS' if r['ok'] else '❌ FAIL'} | {r['detail']} |")
    lines.append("")

    lines.append("## 关键指标实测")
    lines.append("")
    lines.append("| 场景 | 状态 | 基线召回 | 最终召回 | 基线fdr | 最终fdr | 命中数 | TP | FP |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    lines.append(_metrics_row("S1 常规扩展 (R≥0.85,F≤0.2)", s1))
    lines.append(_metrics_row("S2 词表上限冲突 (R=1.0,F≤0.05)", s2))
    lines.append(_metrics_row("S2b 误检预算冲突 (R≥0.85,F≤0.01)", s2b))
    lines.append(_metrics_row("S3 放宽 fdr (R≥0.85,F≤0.4)", s3))
    lines.append(_metrics_row("S4 排除全部原扩展词后重算", s4))
    lines.append(_metrics_row("S5 锁定 decoherence", s5))
    lines.append("")

    lines.append("## S2 失败诊断原文（词表覆盖上限）")
    lines.append("")
    for reason in s2["failure"]["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## S2b 失败诊断原文（误检预算不足）")
    lines.append("")
    for reason in s2b["failure"]["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    if s7 is not None:
        dil, ok_free, ok_locked = s7
        lines.append("## S7 中途误检超标后补足正例（逐词 trace，锁定 apoll）")
        lines.append("")
        lines.append("| 步骤 | 新增正例 | 新增负例 | 累计召回 | 累计fdr | 累计FP |")
        lines.append("|---|---|---|---|---|---|")
        # Re-run to capture the trace for the report (deterministic)
        res_locked = engine_analyze(dil, AnalysisRequest(
            query="alpha", min_recall=1.0, max_fdr=0.35,
            locked_terms=["apoll"],
        ))
        for t in res_locked["trace"]:
            lines.append(
                f"| {t['term']} | {t['added_relevant']} | "
                f"{t['added_nonrelevant']} | {t['cumulative_recall']:.3f} "
                f"| {t['cumulative_fdr']:.3f} "
                f"| {t['cumulative_false_positives']} |"
            )
        lines.append("")
        lines.append(
            "- 自由搜索（无锁定）真值校验：" + ("通过" if ok_free else "失败")
        )
        lines.append(
            "- 锁定 apoll（锁定前缀即超标）真值校验："
            + ("通过" if ok_locked else "失败")
        )
        lines.append("")
    if s8 is not None:
        narrow, wide = s8
        lines.append("## S8 束宽截断的诊断原文")
        lines.append("")
        for reason in narrow["failure"]["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append(
            f"- 同参数下 beam_width=8 的结果：扩展词 {wide['expanded_terms']}，"
            f"召回 {wide['metrics']['recall']:.3f}，fdr {wide['metrics']['fdr']:.3f}"
        )
        lines.append("")
    if s9 is not None:
        lines.append("## S9 已证实无法达标时的诊断原文")
        lines.append("")
        for reason in s9["failure"]["reasons"]:
            lines.append(f"- {reason}")
        lines.append("")
        lines.append(
            f"- `verdict={s9['failure']['verdict']}`，"
            f"`recall_upper_bound={s9['failure']['recall_upper_bound']}`，"
            f"`best_feasible_recall={s9['failure']['best_feasible_recall']}`"
        )
        lines.append("")
    lines.append("## S1 逐词覆盖变化（trace 实测）")
    lines.append("")
    lines.append("| 步骤 | 新增正例 | 新增负例 | 累计召回 | 累计fdr | 累计FP |")
    lines.append("|---|---|---|---|---|---|")
    for t in s1["trace"]:
        lines.append(
            f"| {t['term']} | {t['added_relevant']} | {t['added_nonrelevant']} "
            f"| {t['cumulative_recall']:.3f} | {t['cumulative_fdr']:.3f} "
            f"| {t['cumulative_false_positives']} |"
        )
    lines.append("")
    lines.append("## 旧建议失效验证（S4）")
    lines.append("")
    lines.append(f"- 排除修订后携带旧 `suggestion_id` 调用 `/api/analyze` → "
                 f"HTTP **{stale.status_code}**")
    lines.append(f"- 响应内容：`{stale.json().get('detail')}`")
    lines.append("")
    lines.append(f"## 结论：**{len(results) - failures}/{len(results)} 通过**")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport written -> {path}")


if __name__ == "__main__":
    sys.exit(main())
