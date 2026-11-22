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
tests/          pytest 单元与 API 测试（32 项）
scripts/
  generate_demo_data.py  生成内置演示语料 data/demo_corpus.csv
  selftest.py            交付前自测（7 个阈值冲突/修订场景，产出报告）
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
- 召回 recall = 命中正例 / 全部正例；误检率 fdr = 命中负例 / 全部命中文档；
  可选 `max_false_positives` 限制误检绝对数量。
- 候选词：仅取正例中出现、词频 `min_df=2`、非基线词、未被排除的词，
  按「正例覆盖率 × 纯度 × IDF」打分。
- 束搜索逐层加入候选词，剪去突破 fdr / fp 上限的分支，按召回→fdr→词数排序保留 beam。
- 锁定词强制前置纳入；若锁定词本身越界，直接进入失败诊断。

## 失败诊断（找不到满足阈值的扩展时）

返回：无约束词表覆盖上限、误检约束内可达最高召回、无法覆盖的正例文档、
首个越界词及其新增正/负例数与导致的 fdr、人类可读原因列表；
同时仍返回尽力而为（best-effort）的重排结果。

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
python3 -m pytest tests/ -q          # 32 项单元 + API 测试
python3 scripts/selftest.py          # 7 个场景并写 reports/selftest_report.md
# 服务启动后：
bash scripts/smoke.sh http://127.0.0.1:8000
```

`selftest.py` 覆盖：常规满足、词表覆盖上限冲突、误检预算冲突、放宽上限消解、
排除词重算 + 409、锁定词强制入选、锁定/排除矛盾与解锁。
