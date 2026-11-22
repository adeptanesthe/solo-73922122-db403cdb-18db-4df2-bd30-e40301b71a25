# 科研检索查询扩展 Web 应用

上传带相关性标签的论文摘要集合，输入初始查询、**召回下限**与**误检率上限**；
系统仅从**正例文档**中提取候选扩展词，在约束下用束搜索（beam search）选择扩展项，
对检索结果做 TF-IDF 余弦重排，并逐词解释每个扩展词带来的覆盖变化。
找不到满足阈值的扩展时返回结构化失败原因；用户可**锁定 / 排除**任意词，
系统重算结果并通过修订号使旧建议立即失效。

技术栈：FastAPI + scikit-learn（TF-IDF / 稀疏矩阵 / 余弦相似度）+ 原生 HTML/CSS/JS，
无前端构建步骤。

## 目录结构

```
app/
  engine.py     纯算法层：语料、候选词、束搜索、逐词 trace、失败诊断
  session.py    内存会话与修订号（revision）/ 旧建议失效
  main.py       FastAPI 路由与静态资源
static/         原生前端（index.html / styles.css / app.js）
tests/          pytest 单元与 API 测试（40 项）
scripts/
  generate_demo_data.py  生成内置演示语料 data/demo_corpus.csv
  selftest.py            交付前自测（11 个阈值冲突/束宽/诊断/修订场景，产出报告）
  smoke.sh               运行中服务的 HTTP 端到端冒烟脚本
data/demo_corpus.csv     45 篇演示摘要（17 正例 / 28 负例）
reports/                 自测输出
```

## 快速开始（Docker Compose）

```bash
docker compose up --build
# 浏览器打开 http://localhost:8000
```

页面点击「使用内置演示语料」即可一键载入，再点「运行扩展分析」。

## 本地运行（无 Docker）

```bash
python3 -m pip install -r requirements.txt
python3 scripts/generate_demo_data.py          # 生成演示语料
python3 -m uvicorn app.main:app --port 8000
```

## 输入格式

CSV / TSV（表头 `id,title,abstract,relevant`）或 JSON / JSONL 对象数组。
`relevant` 支持 `true/false`、`1/0`、`yes/no`、`relevant/irrelevant`。

## 检索与约束模型

- 文档命中：查询词与扩展词的布尔 **OR** 并集；TF-IDF 与布尔矩阵共用同一词表。
  召回、误检率与误检数都按**最终命中文档的去重并集**计算。
- 召回 recall = 命中正例 / 全部正例；误检率 fdr = 命中负例 / 全部命中文档；
  可选 `max_false_positives` 限制误检绝对数量。
- 候选词：仅取正例中出现、词频 `min_df=2`、非基线词、未被排除的词，
  按「正例覆盖率 × 纯度 × IDF」打分。
- 束搜索逐层加入候选词。**中途误检率超标的中间组合不会被直接丢弃**：
  继续加词可能补足正例、把误检率稀释回阈值内（锁定词造成的超标同理）。
  只有被证明「无论如何延伸都无法同时满足召回/误检限制」的分支才会被剪掉，
  判定使用一个安全下界：误检数随加词单调不减，剩余词位最多能覆盖仍可触达
  的正例。
- 剪枝/束宽只影响搜索预算，不影响「已找到的最高召回」的选择：满足召回下限
  与误检限制的组合中取召回最高者（并列时取词数更少、误检更低者）。
- 锁定词强制前置纳入，但**不再**在中途越界时中止搜索；若锁定词最终无法被
  稀释回阈值，则进入失败诊断并说明。

## 失败诊断（找不到满足阈值的扩展时）

诊断区分两种情形：

- **已证实无法达标**（`verdict = proven_infeasible`）：搜索已穷尽（未因束宽
  剪枝、候选词池覆盖全部可用词），或安全召回上界本身低于下限。此时确认
  不存在同时满足词项与误检限制且达到召回下限的组合。
- **搜索尚无定论**（`verdict = search_inconclusive`）：束宽或候选词池截断了
  搜索，而安全上界仍不低于下限；仍可能存在达标组合，应增大 `beam_width` /
  候选词上限后重试。

两种情形都返回：

- `best_found`：**已找到且满足词项与误检限制**的组合中的**最高召回**方案
  （未达到召回下限时也如实给出；没有这样的组合时为 `null`）。
- `failure.recall_upper_bound`：最高可达召回的**可靠上界**，计入锁定词之后
  剩余的词位、仍可覆盖的正例，以及误检数单调不减给出的 fdr 下界；若搜索被
  截断则退回全词表上界 `global_recall_upper_bound`。
- `failure.best_feasible_recall`、`missing_relevant_docs`、`blocker_term`、
  `branches_pruned`、`candidate_pool_complete`、人类可读 `reasons`。
- 仍返回尽力而为（best-effort）的逐词 trace 与重排结果。

找到达标组合时照常展示逐词覆盖 trace 与 TF-IDF 重排结果。

## 修订与旧建议失效

- 每次有效锁定/排除/解锁/取消排除都会使会话 `revision +1`。
- 分析响应绑定 `suggestion_id` 与当时的修订号；修订后再携带旧 ID 调用
  `/api/analyze` 或 `/api/revise` 会得到 **HTTP 409**，提示重新分析。
- 同一词同时锁定与排除返回 **HTTP 400**。

## API 摘要

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/sessions` | 上传语料，创建会话（multipart file） |
| GET  | `/api/demo-corpus` | 下载内置演示语料 |
| POST | `/api/analyze` | 运行扩展分析（可带 suggestion_id 校验时效） |
| POST | `/api/revise` | 锁定/排除/解锁/取消排除，修订号 +1 |
| GET  | `/api/sessions/{id}` | 查看会话修订状态 |
| GET  | `/api/health` | 健康检查 |

## 测试与自测

```bash
python3 -m pytest tests/ -q          # 40 项单元 + API 测试
python3 scripts/selftest.py          # 11 个场景并写 reports/selftest_report.md
# 服务启动后：
bash scripts/smoke.sh http://127.0.0.1:8000
```

`selftest.py` 覆盖：常规满足、词表覆盖上限冲突、误检预算冲突、放宽上限消解、
排除词重算 + 409、锁定词强制入选、锁定/排除矛盾与解锁，以及本次修复的核心：

- **S7**：唯一达标组合必须先经过中途误检超标、再加词稀释——自由搜索与
  「锁定词即超标」两种情形都必须找到该组合（并用穷举真值校验）。
- **S8**：窄束宽漏解时报「搜索尚无定论」，宽束宽能找回同一组合。
- **S9**：穷尽搜索无解时报「已证实无法达标」，并单列满足限制的最高召回、
  可靠召回上界与无约束词表召回。
