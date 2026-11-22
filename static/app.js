/* 科研检索查询扩展系统 — 原生 JS 前端（无框架、无构建步骤） */
"use strict";

const state = {
  sessionId: null,
  revision: 0,
  suggestionId: null,
  suggestionRev: null,
  locked: [],
  excluded: [],
  lastResult: null,
  candidateSet: new Set(),
};

const $ = (id) => document.getElementById(id);

function setStatus(el, text, kind) {
  el.textContent = text;
  el.className = "status " + (kind || "");
}

function showError(el, text) {
  el.textContent = text;
  el.classList.remove("hidden");
}
function hideError(el) { el.classList.add("hidden"); }

function fmtPct(x) { return (x * 100).toFixed(1) + "%"; }

async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    const err = new Error(data.detail || data.error || `HTTP ${resp.status}`);
    err.status = resp.status;
    err.payload = data;
    throw err;
  }
  return data;
}

/* ------------------------------------------------------------------ */
/* ① 语料上传                                                          */
/* ------------------------------------------------------------------ */
$("upload-btn").addEventListener("click", async () => {
  const file = $("file-input").files[0];
  const statusEl = $("upload-status");
  if (!file) { setStatus(statusEl, "请先选择文件。", "err"); return; }
  const fd = new FormData();
  fd.append("file", file);
  setStatus(statusEl, "上传并构建 TF-IDF 索引中…", "");
  try {
    const data = await api("/api/sessions", { method: "POST", body: fd });
    bootstrapSession(data);
    setStatus(
      statusEl,
      `上传成功：${data.corpus_size} 篇（正例 ${data.n_relevant} / 负例 ${data.n_nonrelevant}），词表 ${data.vocab_size} 项。`,
      "ok"
    );
  } catch (err) {
    setStatus(statusEl, "上传失败：" + err.message, "err");
  }
});

$("demo-btn").addEventListener("click", async () => {
  const statusEl = $("upload-status");
  setStatus(statusEl, "加载内置演示语料…", "");
  try {
    const fd = new FormData();
    const text = await fetch("/api/demo-corpus").then((r) => {
      if (!r.ok) throw new Error("演示语料不可用");
      return r.text();
    });
    fd.append("file", new File([text], "demo_corpus.csv", { type: "text/csv" }));
    const data = await api("/api/sessions", { method: "POST", body: fd });
    bootstrapSession(data);
    setStatus(
      statusEl,
      `演示语料已加载：${data.corpus_size} 篇（正例 ${data.n_relevant} / 负例 ${data.n_nonrelevant}），词表 ${data.vocab_size} 项。`,
      "ok"
    );
  } catch (err) {
    setStatus(statusEl, "演示语料加载失败：" + err.message, "err");
  }
});

function bootstrapSession(data) {
  state.sessionId = data.session_id;
  state.revision = data.revision;
  state.suggestionId = null;
  state.locked = [];
  state.excluded = [];
  state.lastResult = null;
  $("query-card").classList.remove("hidden");
  $("terms-card").classList.add("hidden");
  $("result-card").classList.add("hidden");
  updateRevisionBadge();
}

/* ------------------------------------------------------------------ */
/* ② 分析                                                              */
/* ------------------------------------------------------------------ */
$("analyze-btn").addEventListener("click", runAnalysis);

async function runAnalysis() {
  hideError($("query-error"));
  if (!state.sessionId) return;
  const maxFpRaw = $("max-fp").value.trim();
  const body = {
    session_id: state.sessionId,
    query: $("query-input").value,
    min_recall: parseFloat($("min-recall").value),
    max_fdr: parseFloat($("max-fdr").value),
    max_false_positives: maxFpRaw === "" ? null : parseInt(maxFpRaw, 10),
    beam_width: parseInt($("beam-width").value, 10),
    max_terms: parseInt($("max-terms").value, 10),
    top_k: parseInt($("top-k").value, 10),
    suggestion_id: state.suggestionId,
  };
  try {
    const data = await api("/api/analyze", {
      method: "POST",
      body: JSON.stringify(body),
    });
    state.suggestionId = data.suggestion_id;
    state.suggestionRev = data.revision;
    state.revision = data.revision;
    state.locked = data.current_locked_terms;
    state.excluded = data.current_excluded_terms;
    state.lastResult = data;
    state.candidateSet = new Set(data.candidate_pool.map((c) => c.term));
    renderTerms(data);
    renderResult(data);
    updateRevisionBadge();
  } catch (err) {
    if (err.status === 409) {
      // 旧建议已失效：清除本地建议 ID，提示用户重新分析
      state.suggestionId = null;
      state.suggestionRev = null;
      showError(
        $("query-error"),
        "⚠ 旧建议已失效（修订后未重新分析）。\n" + err.message +
        "\n请再次点击「运行扩展分析」以获取新建议。"
      );
    } else {
      showError($("query-error"), err.message);
    }
  }
}

/* ------------------------------------------------------------------ */
/* ③ 锁定 / 排除修订                                                    */
/* ------------------------------------------------------------------ */
function renderTerms(data) {
  $("terms-card").classList.remove("hidden");
  renderChips();
  const tbody = $("candidate-table").querySelector("tbody");
  tbody.innerHTML = "";
  for (const c of data.candidate_pool) {
    const tr = document.createElement("tr");
    const isLocked = c.locked || state.locked.includes(c.term);
    const isExcluded = c.excluded || state.excluded.includes(c.term);
    tr.innerHTML = `
      <td><strong>${escapeHtml(c.term)}</strong></td>
      <td>${c.n_relevant_hits}</td>
      <td>${c.n_nonrelevant_hits}</td>
      <td>${c.score.toFixed(4)}</td>
      <td class="hint">${escapeHtml(c.reason)}</td>
      <td class="row" style="margin:0;gap:6px;"></td>`;
    const actionTd = tr.lastElementChild;
    if (isLocked) {
      const btn = document.createElement("button");
      btn.className = "small undo";
      btn.textContent = "解除锁定";
      btn.onclick = () => revise({ unlock: [c.term] });
      actionTd.appendChild(btn);
      tr.style.background = "var(--green-bg)";
    } else if (isExcluded) {
      const btn = document.createElement("button");
      btn.className = "small undo";
      btn.textContent = "解除排除";
      btn.onclick = () => revise({ unexclude: [c.term] });
      actionTd.appendChild(btn);
      tr.style.background = "var(--red-bg)";
    } else {
      const lockBtn = document.createElement("button");
      lockBtn.className = "small lock";
      lockBtn.textContent = "锁定";
      lockBtn.onclick = () => revise({ lock: [c.term] });
      const exBtn = document.createElement("button");
      exBtn.className = "small exclude";
      exBtn.textContent = "排除";
      exBtn.onclick = () => revise({ exclude: [c.term] });
      actionTd.append(lockBtn, exBtn);
    }
    tbody.appendChild(tr);
  }
}

function renderChips() {
  const lockedBox = $("locked-chips");
  const excludedBox = $("excluded-chips");
  lockedBox.innerHTML = '<span class="hint">已锁定：</span>';
  excludedBox.innerHTML = '<span class="hint">已排除：</span>';
  if (state.locked.length === 0) lockedBox.innerHTML += '<span class="hint">（无）</span>';
  if (state.excluded.length === 0) excludedBox.innerHTML += '<span class="hint">（无）</span>';
  for (const term of state.locked) {
    lockedBox.appendChild(makeChip(term, "locked", () => revise({ unlock: [term] })));
  }
  for (const term of state.excluded) {
    excludedBox.appendChild(makeChip(term, "excluded", () => revise({ unexclude: [term] })));
  }
}

function makeChip(term, kind, onRemove) {
  const chip = document.createElement("span");
  chip.className = "chip " + kind;
  chip.textContent = term;
  const x = document.createElement("button");
  x.textContent = "×";
  x.title = "撤销";
  x.onclick = onRemove;
  chip.appendChild(x);
  return chip;
}

async function revise(patch) {
  hideError($("query-error"));
  try {
    const data = await api("/api/revise", {
      method: "POST",
      body: JSON.stringify({
        session_id: state.sessionId,
        stale_suggestion_id: state.suggestionId,
        ...patch,
      }),
    });
    state.revision = data.revision;
    state.locked = data.locked_terms;
    state.excluded = data.excluded_terms;
    // 修订即旧建议作废
    state.suggestionId = null;
    state.suggestionRev = null;
    updateRevisionBadge();
    renderChips();
    if (state.lastResult) renderTerms(state.lastResult);
    // 自动重算
    await runAnalysis();
  } catch (err) {
    if (err.status === 409) {
      state.suggestionId = null;
      updateRevisionBadge();
      showError($("query-error"), "⚠ " + err.message);
    } else {
      showError($("query-error"), err.message);
    }
  }
}

function updateRevisionBadge() {
  const badge = $("revision-badge");
  badge.classList.remove("hidden");
  const stale = state.lastResult && state.suggestionRev !== state.revision;
  if (state.suggestionId && !stale) {
    badge.textContent = `修订号 rev=${state.revision} · 当前建议有效`;
    badge.className = "badge fresh";
  } else if (state.lastResult || state.locked.length || state.excluded.length) {
    badge.textContent = `修订号 rev=${state.revision} · 旧建议已失效，请重新分析`;
    badge.className = "badge stale";
  } else {
    badge.textContent = `修订号 rev=${state.revision}`;
    badge.className = "badge";
  }
}

/* ------------------------------------------------------------------ */
/* ④ 结果渲染                                                          */
/* ------------------------------------------------------------------ */
function renderResult(data) {
  $("result-card").classList.remove("hidden");
  const summary = $("result-summary");
  const base = data.baseline_metrics;
  const fin = data.metrics;
  const satisfied = data.status === "satisfied";
  summary.innerHTML = `
    <div class="${satisfied ? "satisfied-banner" : "failed-banner"}">
      ${satisfied
        ? `✅ 找到满足阈值的扩展方案：扩展词 ${data.expanded_terms.length} 个 → ${data.expanded_terms.map(escapeHtml).join(", ") || "（无需扩展）"}`
        : "❌ 找不到同时满足召回下限与误检上限的扩展方案，原因见下方诊断。"}
    </div>
    <div class="metric-grid">
      ${metric("基线召回", fmtPct(base.recall), base.recall >= data.thresholds.min_recall ? "good" : "bad")}
      ${metric("扩展后召回", fmtPct(fin.recall), fin.recall >= data.thresholds.min_recall ? "good" : "bad")}
      ${metric("召回下限", fmtPct(data.thresholds.min_recall))}
      ${metric("基线误检率", fmtPct(base.fdr), base.fdr <= data.thresholds.max_fdr ? "good" : "bad")}
      ${metric("扩展后误检率", fmtPct(fin.fdr), fin.fdr <= data.thresholds.max_fdr ? "good" : "bad")}
      ${metric("误检率上限", fmtPct(data.thresholds.max_fdr))}
      ${metric("命中文档", `${fin.n_retrieved} / ${data.corpus_size}`)}
      ${metric("正例命中 (TP)", `${fin.true_positives} / ${data.n_relevant}`)}
      ${metric("误检 (FP)", `${fin.false_positives}`)}
    </div>
    <p class="hint">
      基线查询词：${data.baseline_terms.map(escapeHtml).join(", ") || "（词表中无匹配）"}
      ${data.unknown_query_tokens.length
        ? "｜未登录词（已忽略）：" + data.unknown_query_tokens.map(escapeHtml).join(", ")
        : ""}
    </p>`;

  // trace table
  const traceBody = $("trace-table").querySelector("tbody");
  traceBody.innerHTML = "";
  for (const row of data.trace) {
    const tr = document.createElement("tr");
    if (row.is_baseline) tr.style.background = "#f1f5f9";
    tr.innerHTML = `
      <td><strong>${escapeHtml(row.term)}</strong></td>
      <td>${row.added_relevant}</td>
      <td>${row.added_nonrelevant}</td>
      <td>${row.cumulative_retrieved}</td>
      <td>${fmtPct(row.cumulative_recall)}</td>
      <td>${fmtPct(row.cumulative_fdr)}</td>
      <td>${row.cumulative_false_positives}</td>
      <td class="hint">${escapeHtml(row.note)}</td>`;
    traceBody.appendChild(tr);
  }

  // failure diagnosis
  const box = $("failure-box");
  if (data.failure) {
    box.classList.remove("hidden");
    const f = data.failure;
    box.innerHTML = `
      <h4>失败诊断</h4>
      <ul>${f.reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
      <p class="hint" style="margin-top:8px;">
        无约束最高召回：<strong>${fmtPct(f.unconstrained_max_recall)}</strong> ｜
        误检约束内最高召回：<strong>${fmtPct(f.best_feasible_recall)}</strong>
        ${f.blocker_term
          ? `｜首个越界词：<code>${escapeHtml(f.blocker_term.term)}</code>`
          : ""}
      </p>
      ${f.missing_relevant_docs.length
        ? `<p class="hint">无法覆盖的正例：${f.missing_relevant_docs.map((d) => escapeHtml(d.id)).join(", ")}</p>`
        : ""}`;
  } else {
    box.classList.add("hidden");
    box.innerHTML = "";
  }

  // ranking
  $("ranking-meta").textContent =
    `（共 ${fin.n_retrieved} 篇命中，展示前 ${data.ranking.length} 篇；绿色=正例，红色=负例）`;
  const rankBody = $("ranking-table").querySelector("tbody");
  rankBody.innerHTML = "";
  const expansionSet = new Set(data.expanded_terms);
  for (const doc of data.ranking) {
    const tr = document.createElement("tr");
    const terms = doc.matched_terms
      .map((t) => `<span class="matched-term ${expansionSet.has(t) ? "expansion" : ""}">${escapeHtml(t)}</span>`)
      .join("");
    tr.innerHTML = `
      <td>${doc.rank}</td>
      <td>${escapeHtml(doc.id)}</td>
      <td><strong>${escapeHtml(doc.title)}</strong><br/><span class="hint">${escapeHtml(truncate(doc.abstract, 180))}</span></td>
      <td><span class="tag ${doc.relevant ? "rel" : "nonrel"}">${doc.relevant ? "正例" : "负例"}</span></td>
      <td>${terms}</td>
      <td>${doc.score.toFixed(4)}</td>`;
    rankBody.appendChild(tr);
  }
}

function metric(k, v, cls) {
  return `<div class="metric"><div class="k">${k}</div><div class="v ${cls || ""}">${v}</div></div>`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function truncate(s, n) { return s.length > n ? s.slice(0, n) + "…" : s; }
