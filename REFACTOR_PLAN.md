# RAG4Prefa 重构方案

> 目标：在 `F:\project\课题\RAG4Prefa` 重新实现 `qianluxi/milvus`（v2.8.6）的全部可用功能，但用更简洁、更先进、可测试的技术路线重写。
> 约束：**功能对等**（原系统能做的都要能做）、**单一进程可跑**（不需要 Docker 也能演示）、**跨平台**（不再绑 Windows + Word）、**可评估**（课题验收要有指标，不是只有界面）。
> 代码量目标：Python ≲ 1 500 行、前端 ≲ 500 行（原系统 2 312 + 2 318 行，含大量死代码）。

---

## 一、原系统能力清单（重构后的验收基线）

| # | 能力 | 原实现方式 | 必须保留 |
|---|---|---|---|
| 1 | `.docx` 上传入库，按项目归类 | Flask `/upload` + Word COM 逐段读取 | ✅ |
| 2 | 文档层级还原（章 → 节 → 子节） | 正则猜层级 + Word 样式名 | ✅ 且更准 |
| 3 | 短节合并、超长截断 | 500 字符阈值 + md5 后缀塞进 varchar | ✅ 改为 token 阈值 |
| 4 | 向量化入库 | ModelScope gte 远程 + 逐条 insert | ✅ 改批量 + 幂等 upsert |
| 5 | 自然语言问答（检索 + 大模型整合） | 单轮 prompt 拼接 | ✅ 加引用编号 |
| 6 | 流式输出 | 先跑完再按空格切词 `sleep` | ✅ **真流式** |
| 7 | 按项目 / 年份 / 特征筛选 | 全局 context + 全表扫描 | ✅ 改为请求级 filter |
| 8 | 项目管理（列表 / 详情 / 删除） | 3 个接口，其中详情必 500 | ✅ 修好 |
| 9 | 项目统计（文档数 / 片段数 / 时间） | 字段恒为 0 | ✅ 且准确 |
| 10 | 跨项目对比、按特征找项目 | 死代码 | ✅ 作为独立接口实现 |
| 11 | 解析过程留痕 | 生成 `*_parsed_*.txt` | ⚠️ 改为结构化 JSON，可回放 |

**明确放弃**：Windows Word COM 解析（换跨平台解析器后层级识别质量只会更高）、`ChatModule` 双份前端代码、假流式、全局可变上下文。

---

## 二、技术选型对照

| 层 | 原方案 | 新方案 | 为什么 | 代价 |
|---|---|---|---|---|
| 包管理 | `requirements.txt`（不锁版本，含不存在的 `pythoncom`） | **uv** + `pyproject.toml`（锁 `uv.lock`） | 一条命令完成装/锁/跑，杜绝"装不上" | 需装 uv（`pip install uv` 或独立安装包） |
| Web | Flask 3 + 开发服务器 | **FastAPI** + Uvicorn | pydantic 入参校验、async、原生 SSE、自动 OpenAPI（验收演示友好） | 学习成本极低 |
| 文档解析 | `win32com` 驱动本机 Word（Windows 独占） | **python-docx** 读段落样式/编号/表格 → Section 树 | 跨平台、无 Office 依赖、可进 Docker、可单元测试 | 拿不到 Word 的"渲染编号"，需样式+正则双路兜底（已设计） |
| 分块 | 500 字符硬阈值，标题用 `" · "` 拼 | **标题树分块** + token 阈值 + 父级上下文（parent window） | 检索命中"片段"时能把整节标题链一起给 LLM，答案更准 | 略多存字段 |
| Embedding | ModelScope gte（硬编码 token） | **`Embedder` 协议**：远程 OpenAI 兼容 `/embeddings`（默认）/ 本地 `fastembed`（ONNX，无 torch） | 密钥进 `.env`；断网可切本地；维度写配置并校验 | 双实现约 120 行 |
| 向量库 | Milvus standalone `localhost:19530`（要 Docker） | **Milvus Lite**（`pip install milvus-lite`，单文件，同一个 `pymilvus` API） | 保留 Milvus 技术路线、零基础设施；生产切 standalone 只改 URI | Lite 不支持 HNSW/partition/BM25 函数，索引用 `AUTOINDEX` |
| 结构化事实来源 | `project_metadata.json` + Milvus 双写 | **SQLite 为唯一真相**（projects/documents/chunks/jobs），向量库是**派生索引** | 根治"字段对不上""从 Milvus 反推元数据"这类烂账 | 多一张表，但换来一致性 |
| 关键词召回 | 向量 Top-K 之后做子串过滤 | **SQLite FTS5（jieba 预分词）+ `bm25()`**，与向量结果 **RRF 融合** | 修复"关键词过滤截断召回"的硬伤，中文专有名词（缓粘结预应力/陶土砖）靠 BM25 兜底 | 需维护自定义词典 |
| 重排 | torch + sentence-transformers（默认关闭） | **fastembed ONNX CrossEncoder** 或远程 rerank API，可关 | 无 torch 依赖，装包体积从 ~2 GB 降到 ~100 MB | ONNX 模型精度略低于大模型重排 |
| 大模型 | 自写 `ModelScopeLLM`（同步、假流式） | **OpenAI 兼容客户端 + async stream** | 一套代码通吃 ModelScope / Qwen / DeepSeek / Ollama | — |
| 筛选条件解析 | 正则（且调用不存在的方法）+ 全局 context | **请求级 `Filter` 对象**：正则/日期库快路径 → LLM 结构化输出兜底 | 无状态、可单测、不串号 | — |
| 前端 | 2 318 行模板 + 内联 1 400 行 JS + 3 套 CSS 框架 | **单页 + Alpine.js（CDN）+ Tailwind（CDN）**，SSE 用 `fetch` + `ReadableStream` | 零构建、代码量 1/4、不再有"改了不生效"的双份文件 | 复杂交互不如框架 |
| 日志 | DEBUG 全开 + 14 处 `print()`，7 MB 日志入库 | **structlog** 结构化 + 级别可配 + request id | 可诊断、不进仓库 | — |
| 测试 | 0 | **pytest**（parser golden / chunker / filter / 幂等入库）+ 假向量库 | 这是"能从脚本变成系统"的分水岭 | — |
| 评估 | 无 | `eval/` 标注集 + Recall@k / MRR / nDCG / 引用命中率 / 延迟报表 | 课题验收需要可量化结论 | 需 30~50 条标注 |

---

## 三、目标架构

```
┌─────────────── 前端单页（Alpine + SSE） ───────────────┐
│  筛选器(项目/年份/关键词) │ 对话流(流式+引用卡片) │ 项目管理抽屉 │
└───────────────────────────┬───────────────────────────┘
                            │ HTTP / SSE
┌───────────────────────────▼───────────────────────────┐
│  api/    FastAPI 路由层（只做协议转换，不含业务）        │
│    routes_chat / routes_admin / routes_jobs            │
├───────────────────────────────────────────────────────┤
│  app/    应用编排（用例层）                             │
│    ingest.py  → parse → chunk → embed → upsert         │
│    ask.py     → filter → recall → fuse → rerank → LLM  │
│    admin.py   → 项目/文档/任务管理                      │
├───────────────────────────────────────────────────────┤
│  domain/ 纯领域：Section / Chunk / Project / Filter / Hit │
│           （pydantic，无 IO，无框架）                   │
├───────────────────────────────────────────────────────┤
│  adapters/ 可替换实现（全部面向 Protocol 编程）         │
│    parsers/  docx.py   (可选 pdf.py)                    │
│    embed/    openai_compat.py | fastembed_local.py      │
│    vector/   milvus_lite.py  (可切 standalone)          │
│    keyword/  sqlite_fts.py                              │
│    rerank/   onnx_crossencoder.py | remote.py | noop.py │
│    llm/      openai_compat.py                           │
│    store/    sqlite.py  (projects/documents/chunks/jobs)│
└───────────────────────────────────────────────────────┘
```

### 关键取舍：SQLite 是真相，向量库是索引

原系统最深的坑是「本地 JSON + Milvus + 日志文件」三处各存一份事实，导致统计恒 0、元数据靠扫全表反推。新设计：

- **SQLite**（`data/app.db`）持有 `projects / documents / chunks / jobs` —— 唯一真相，删除、统计、幂等判断都在这里。
- **Milvus Lite**（`data/vectors.db`）只持有 `chunk_id → embedding + 过滤用标量`，是**可重建的派生索引**。
- **FTS5** 表也在 SQLite 里，`chunks` 写入时同步维护，BM25 用内置 `bm25()`。
- 索引损坏/换模型 → `rag4prefa reindex` 从 SQLite 重建，不丢业务数据。

---

## 四、数据模型

### Milvus（派生索引，`chunks_vec`）

```
chunk_id      VARCHAR(64)  主键（非 auto_id，才能 upsert）
embedding     FLOAT_VECTOR dim 来自配置，建库时校验
project_name  VARCHAR(256)    ← 标量过滤
project_code  VARCHAR(64)
doc_year      INT32           ← 年份过滤不再比字符串
doc_id        VARCHAR(64)
```
索引 `AUTOINDEX` / `COSINE`（Lite 不支持 HNSW 参数，切 standalone 时由配置改回 HNSW）。

### SQLite

```sql
projects(name PK, code, started_on, created_at)
documents(id PK, project_name FK, filename, content_sha, section_count,
          chunk_count, status, uploaded_at)          -- 唯一键 (project_name, filename)
chunks(id PK, doc_id FK, ordinal, chapter_path, chapter_title, subsection_title,
       content, token_count, content_sha)
chunks_fts(chapter_title, subsection_title, content_tokens)   -- FTS5, jieba 预分词
jobs(id PK, kind, payload_json, state, progress, message, created_at, finished_at)
```

### 领域模型（pydantic）

```python
Section(title, level, kind: Literal["chapter","section","subsection","body","table"], text, number)
Chunk(doc_id, ordinal, chapter_path: list[str], title, text, token_count, sha)
Filter(projects: list[str] | None, year_from: int | None,
       year_to: int | None, keyword: str | None, feature: str | None)
Hit(chunk_id, score, project_name, filename, chapter_path, text, citations)
```

---

## 五、API 设计（`/api` 前缀，全量 OpenAPI 自动生成）

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/api/health` | 依赖探活（向量库/embedding/LLM 分开报告，不再"静默降级"） |
| POST | `/api/projects/{name}/documents` | 上传 docx（multipart，多文件）→ 返回 `job_id` |
| GET | `/api/jobs/{job_id}` | 入库进度（逐文件状态 + 失败原因） |
| GET | `/api/projects` | 项目列表（含真实 doc/chunk 数） |
| GET | `/api/projects/{name}` | 项目详情 + 章节分布 + LLM 摘要（带缓存） |
| DELETE | `/api/projects/{name}` | 级联删除（SQLite + 向量，事务化） |
| POST | `/api/chat` | **SSE 流式**：`{question, filter, top_k, rerank}` → `token*` → `citations` → `done` |
| POST | `/api/search` | 非流式检索（调试 / 评估 / 对比实验用） |

**筛选条件随请求传**，没有 `/context` 三个接口，也就没有跨用户串号。

### SSE 事件协议

```
event: meta      data: {"query_id":"...","filter":{...},"candidates":n}
event: token     data: {"text":"..."}
event: citations data: [{"n":1,"project":"广州项目","file":"广州扩初.docx","chapter":"3.2 悬挑"}]
event: done      data: {"elapsed_ms":1830,"tokens":312}
event: error     data: {"message":"..."}
```

---

## 六、三条核心流程

### 1. 入库（幂等）

```
doc_id = sha256(project | filename)[:16]
if documents 存在同 (project, filename) 且 content_sha 相同 → 直接跳过（原系统这里的坑）
Section 树 = parse_docx(path)          # 样式优先，编号/正则兜底，标 kind 与置信度
chunks    = chunk_by_tree(树, max_tokens=480, min_tokens=80, parent_window=True)
事务：写 documents + chunks + chunks_fts
批量 embed（batch=32，指数退避 + 结果缓存）→ Milvus upsert(chunk_id)
任一步失败 → job 记 failed 并保留已写片段（可 retry，不重复插入）
```

### 2. 检索（混合召回 + 融合）

```
filter → Milvus 标量 expr（project_name in [...] and doc_year between ...）
dense  = search(ef=64, limit=50)
sparse = FTS5 bm25(query 分词, limit=50)   # 中文专有名词兜底
fused  = RRF(dense, sparse, k=60)          # 不再"先取 Top-K 再过滤关键词"
hits   = rerank(query, fused[:30])[:top_k]  # 可关
```

### 3. 回答（真流式 + 引用）

```
prompt = 模板文件（system + few-shot + context 带 [n] 编号）
async for delta in llm.stream(prompt):  yield token
结束后补发 citations（[n] 与 hit 一一对应，前端可点开原文）
```

---

## 七、这次不会再犯的 7 个错（直接对应旧代码缺陷）

| 旧缺陷 | 新机制 |
|---|---|
| 传了不存在的参数 → 接口必 500 | 全部走 pydantic 模型 + 类型检查（mypy），签名不一致当场报错 |
| 调用未定义方法 | 同上；CI 跑 `mypy` + `pytest` |
| 同一函数定义两次 / 前端两份拷贝 | 无内联大段 JS；CSS/JS 只有一份，`ruff` 查重复定义 |
| 全局单例 context 串号 | 无模块级可变状态；**进程内唯一可变对象是 SQLite 连接池** |
| 重复上传重复入库 | `(project, filename, content_sha)` 唯一键 + 确定性 `chunk_id` + upsert |
| 统计字段恒 0 | 所有计数由 SQLite `COUNT(*)` 现算，不缓存、不猜键名 |
| 密钥/业务数据进 git | `.env` + `.gitignore` 首行就排除；仓库内只放 `*.example` |

---

## 八、实施阶段（每阶段可独立验收）

| 阶段 | 内容 | 交付物 | 验收标准 |
|---|---|---|---|
| **P0 骨架**（~0.5 天） | uv/pyproject、配置、领域模型、SQLite schema、FastAPI 工厂、`/health`、CLI（`init/ingest/ask/reindex`） | 能 `uv run uvicorn` 起来，`/health` 分别报告三项依赖 | `uv run pytest` 全绿（此时只有配置与 schema 测试） |
| **P1 入库**（~1 天） | docx 解析器、树分块、远程 embedder、Milvus Lite 写入、job 进度、幂等与重复上传跳过 | `rag4prefa ingest --dir F:/project/课题/Ref` 可跑 | 同一文件传两次 → 第二次 0 新增；章节树能对上 `Ref` 里真实 docx |
| **P2 检索问答**（~1 天） | filter 解析（规则+LLM）、稠密+BM25 RRF、ONNX 重排、prompt 模板、SSE | `POST /api/chat` 流式返回答案+引用 | 同一问题重排开/关各跑一次，结果可复现；断网时 local embedder 可切 |
| **P3 前端**（~0.5 天） | 单页：筛选器 / 流式对话 / 引用卡片 / 项目管理抽屉 / 上传进度 | `GET /` 可用 | 上传 → 提问 → 点引用看原文，全链路无刷新 |
| **P4 评估上线**（~0.5 天） | `eval/` 标注集 + `scripts/eval.py` 报表、Dockerfile（Linux 可构建）、README/部署说明 | `eval/report.md` | 输出 Recall@5/@10、MRR、nDCG@10、P95 延迟；Docker 镜像可启动 |

**总计约 3.5 个工作日**，中途每阶段结束都能停、能演示。

---

## 九、评估方案（课题验收的加分项）

- `eval/dataset.jsonl`：30~50 条 `{question, filter, gold_chunk_ids[], gold_answer_points[]}`（先用 LLM 生成候选 + 人工校对，从 `Ref` 里的真实文档出题）。
- 指标：`Recall@5 / Recall@10`、`MRR`、`nDCG@10`、`引用命中率`（回答里的 [n] 是否指向 gold）、`首 token 延迟 / 总时长`。
- 对照组：`dense only` vs `dense+BM25` vs `+rerank`，直接产出"混合检索带来多少提升"的结论 —— 这正是原系统缺失、而科研项目最需要的那部分。

---

## 十、风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| Milvus Lite 不支持 HNSW / partition / 内置 BM25 | 大规模下性能受限 | 索引与选型全部走配置；`VectorStore` 协议 + `milvus_standalone.py` 作为第二实现，切换只改 `VECTOR_URI` |
| `python-docx` 拿不到自动编号（Word 渲染态编号） | 层级识别退化 | 样式优先 → 编号字段 → 正则三层兜底，并把"置信度低"的段落单独记 `kind=body` + 告警，不硬猜 |
| 中文 BM25 分词质量 | 专有名词召回差 | jieba + 自定义词典（`resources/dict.txt` 收"缓粘结预应力/陶土砖/叠涩错砌"等），入库与查询用同一分词器 |
| 远程 embedding 限流 | 入库慢/失败 | 批量 + `tenacity` 指数退避 + embedding 结果按 `content_sha` 缓存，重跑不重复计费 |
| docx 里的表格/图片 | 内容丢失 | 表格转 Markdown 表格入库（P1）；图片暂时跳过并在 README 注明（P4 讨论是否接 OCR） |
| 标注集成本 | 评估做不完 | 先用 10 条跑通指标管线，再扩到 30~50 条 |

---

## 十一、待你确认的三个决策

1. **向量库**：默认 Milvus Lite（延续 Milvus 路线、零 Docker），是否可接受它在 Lite 模式下不支持 HNSW 参数？
2. **模型来源**：默认"远程优先、缺 key 自动回落本地 fastembed"（断网也能演示），是否合适？本地模型首跑要下 ~100 MB。
3. **评估模块**：是否纳入 P4（多约 0.5 天，但验收/论文里是硬通货）？

确认后我按 P0 → P4 顺序动手，每阶段结束给你可运行的状态。

---

## 十二、实施结果（2026-10-23 落地复盘）

### 已交付

| 模块 | 文件 | 行数 |
|---|---|---|
| 配置 / 领域模型 / 存储 schema | `config.py` `domain.py` `db.py` | 350 |
| 数据访问（全部 SQL） | `repo.py` | 392 |
| 入库 / 分块 / token | `ingest.py` `chunking.py` `tokens.py` | 400 |
| 检索问答 / 筛选解析 / 提示词 | `ask.py` `filters.py` `prompts.py` | 350 |
| 适配器（docx / embed / llm / vectors / rerank） | `adapters/` | 720 |
| HTTP / CLI / 装配 / 管理 | `api.py` `cli.py` `services.py` `admin.py` | 740 |
| 前端单页 | `web/index.html` | 438 |
| 测试 | `tests/` | 483（30 项，全离线） |
| 评估 | `scripts/eval.py` + `eval/` | 200 |

Python 合计 **2 838 行**，比原计划（≲1 500）高——差额基本是类型标注、docstring、CLI 与评估脚本。
原系统是 2 312 行 Python + 2 318 行模板，**代码量并没有减少，减少的是重复与死代码**：
新代码里没有一处复制粘贴的函数、没有未接线的接口、没有硬编码凭据，30 项测试覆盖了解析/分块/筛选/幂等/删除/重建/端到端检索。

### 计划与现实的偏差（都已修正进代码）

1. **Milvus Lite 在 Windows 可用**（`pymilvus 3.0.2` + `milvus-lite 3.2.2`）：计划里担心的平台限制没有出现。
   新增了"维度未知时先退化 numpy、首次入库后自动迁回 Milvus"的兜底，避免两个后端各写一半的静默不一致。
2. **"表格"不是边角情况，而是主要矛盾**：`Ref/` 里的申报表是一张 51×13 合并单元格大表，正文全在单元格里，
   且合并单元格会让同一段文字重复 13 次。原实现直接丢弃表格 → 我们改成"进入单元格逐段解析 + 按行去重 + 短单元格行保留成数据行"。
   这一项不修，真实文档解析出来只有一段 14 万字的"前言"。
3. **`paragraph.paragraph_format.outline_level` 在当前 python-docx 里不存在**，改为直接读 `w:outlineLvl` XML，
   顺带支持了自定义标题样式的大纲级别。
4. **`（一）`/`（1）` 的绝对层级因文档而异**，改成"相对级别"：挂在最近一个绝对级别标题的下一层，
   否则同一份文档里 `1.3.1 → （1）→（二）` 会越嵌越深。
5. **向量归档表**是计划外新增：把向量同时落 SQLite，换后端只需 `rebuild_store`，不必重新调 embedding（省时间也省钱）。

### 仍未做（明确留给下一轮）

- 评测集只有 3 条样例，需人工标到 30~50 条才能给出可信的 Recall/MRR 结论。
- 多轮追问的 query 改写、图片/文本框内容提取、`python-docx` 之外的 PDF 通道。
- Dockerfile 与 CI 尚未添加（依赖装完后加很快，但当前环境网络受限未验证构建）。
