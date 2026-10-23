# 交接说明 · RAG4Prefa

> 写给接手的 agent：你**没有**上一轮对话的上下文，这份文件是唯一入口。
> 读完这一份就能开始干活；需要更深的背景再看第 11 节列的文档。

**交接时间**：2026-10-23 16:40
**交接时状态**：可运行、可测试、已推送 GitHub，但**真实链路（真实 embedding / LLM）从未跑过**——这是当前最大的风险，见第 7 节 P0-1。

---

## 1. 一句话说清这是什么

把 `.docx` 工程文档（设计说明、申报书、规范）按**章节树**切成片段，建成本地知识库；
提问时用**向量 + BM25 混合召回**找依据，再让大模型**流式**输出带引用的答案。

它是对另一个仓库 `qianluxi/milvus`（分支 v2.8.6）的**功能等价重写**。
那个原版是一个新手工程师写的约 2 300 行单文件 Flask 应用，有 11 项功能但只有 5 项真能用
（详见 `F:\project\PyProject\milvus\PROJECT_REVIEW.md`）。本次重写的目标就是：功能不减，把架构做对。

---

## 2. 位置与环境

| 项目 | 值 |
|---|---|
| 代码目录 | `F:\project\课题\RAG4Prefa` |
| 远端仓库 | https://github.com/qianluxi/RAG4Prefa （**public**，`main` 分支） |
| 当前提交 | `0d1526b`，工作区干净，本地与 `origin/main` 一致 |
| Python | 3.13，**虚拟环境已建好**：`F:\project\课题\RAG4Prefa\.venv` |
| 解释器路径 | `./.venv/Scripts/python.exe`（Windows；所有命令都在项目根目录执行） |
| 依赖状态 | 已装齐（fastapi / uvicorn / python-docx / pymilvus 3.0.2 / milvus-lite 3.2.2 / openai / jieba / numpy / pytest / ruff） |
| 未安装 | `torch`、本地 embedding 模型（**按用户明确要求不装本地模型**） |

原版项目（只读参考，**不要改它**）：`F:\project\PyProject\milvus`，已检出 v2.8.6。
真实测试语料：`F:\project\课题\Ref\立体聚落异型大悬挑百年建造高质量建造技术研究-申请书26.10.23.docx`

**先跑这三条确认环境可用**：

```bash
cd "F:/project/课题/RAG4Prefa"
./.venv/Scripts/python.exe -m pytest -q          # 期望：30 passed
./.venv/Scripts/python.exe -m ruff check rag4prefa tests scripts   # 期望：All checks passed
./.venv/Scripts/python.exe -m rag4prefa doctor   # 看后端、维度、缺哪些凭据
```

---

## 3. 代码地图（2 838 行 Python + 438 行单页 HTML）

```
rag4prefa/
├── config.py     81    pydantic-settings，全部配置以 R4P_ 前缀从 .env 读
├── domain.py    119    Chunk / Section / Hit / Filter / JobState，纯数据，无依赖
├── db.py        119    SQLite schema（projects / documents / chunks / chunks_fts / vectors / jobs / meta）
├── repo.py      392    **全部 SQL 都在这里**，没有别处写 SQL
├── ingest.py    258    解析 → 分块 → 向量化 → 写库；含 rebuild_store / reindex
├── chunking.py   82    按标题树切块（父标题路径拼进文本，保住上下文）
├── tokens.py     51    中英混排的 token 估算（非真实 tokenizer）
├── ask.py       182    混合召回 → RRF 融合 → 可选重排 → 流式生成
├── filters.py   111    自然语言问题 → Filter（规则法，可单测，无 LLM 调用）
├── prompts.py    38    系统提示词与引用格式
├── admin.py      82    项目/文档的查询与级联删除
├── services.py  165    组件装配；Milvus 不可用时回退 numpy，拿到维度后自动迁回
├── api.py       249    FastAPI 路由（只做协议转换，不含业务逻辑）
├── cli.py       205    init / serve / doctor / ingest / reindex / ask / projects
└── adapters/    720    embed / llm / vectors / rerank / parsers/docx —— 所有外部依赖都关在这里
```

前端只有一个文件 `rag4prefa/web/index.html`（438 行，原生 JS，无构建步骤）。
`tests/` 30 项，**全部离线**（FakeEmbedder 确定性假向量 + 临时 Milvus/numpy），不花一分钱 API 费。

### 数据分层（关键，改动前必须理解）

**SQLite 是唯一事实来源，向量库只是可重建的索引。**

- `vectors` 表存了一份向量归档（float32 bytes）→ 换后端、索引损坏时 `rebuild_store()` 直接重建，**无需重新调 embedding**（省钱省时）。
- 所有计数（文档数、片段数）都是 `COUNT(*)` 现算，不存在"统计恒为 0"的可能。
- 查向量：Milvus 拿到 `(chunk_id, score)` 后回 SQLite 取正文 → 所以 Milvus 只是 `chunk_id → 向量` 的索引。

### 请求链路（`/api/chat`）

```
问题 + Filter → ask.build_filter()（规则补全筛选条件）
  → 并行：向量 Top-50  ‖  FTS5 bm25 Top-50
  → RRF 融合（k=60）→ 取 Top-8 → 可选远程重排
  → 组装带章节路径的上下文 → LLM stream=True → SSE
SSE 事件：meta → token* → citations → done；异常走 error
```

---

## 4. 已经做到什么（相对原系统的差异）

| 维度 | 原系统 `qianluxi/milvus` | 现在 |
|---|---|---|
| 文档解析 | Windows COM 调本机 Word，**Linux/容器内跑不起来** | `python-docx` 纯解析，跨平台 |
| 表格内容 | **直接丢弃**（而申报书正文全在表格里） | 进单元格逐段解析 + 合并单元格按行去重 |
| 状态管理 | 模块级单例，筛选条件**跨请求累积**、多用户串号 | 筛选条件随请求走，无全局可变状态 |
| 过滤位置 | 向量 Top-K **之后**做子串匹配（砍召回） | BM25 与向量**并行召回**后 RRF 融合 |
| 流式 | 先跑完再按空格 `sleep(0.05)` 假装打字 | LLM token 流直通 SSE |
| 幂等 | 无去重，重复上传成倍写入 | `(项目, 文件名, 内容哈希)` 唯一 + 确定性 chunk_id + upsert |
| 凭据 | ModelScope token **硬编码在源码并提交进 git** | 全部走 `.env`，`.env` 已 gitignore |
| 死代码 | 11 项功能 5 项可用，项目详情接口 100% 报 500 | 每个接口都有实现与测试 |

API 一览（实现见 `api.py`，说明见 README）：
`GET /api/health`、`POST /api/projects/{name}/documents`（上传）、`GET /api/jobs/{id}`、
`GET /api/projects`、`GET /api/projects/{name}`、`DELETE /api/projects/{name}`、
`POST /api/search`（非流式，调试用）、`POST /api/chat`（SSE）、`GET /`（前端页）。

---

## 5. 用户明确表达过的偏好（不要违背）

1. **不要本地模型**。模型走远程 OpenAI 兼容接口即可，不装 torch、不下权重。
2. **Milvus Lite 可以接受**，即使它不支持 HNSW 参数（统一用 AUTOINDEX）。
3. **评估模块（P4）要做**，Recall@k / MRR / nDCG 是课题验收材料的一部分。
4. 用户的诉求是「课题交付物」，不是玩具 demo：**代码质量、可复现性、可解释的取舍**比功能堆量重要。
5. 报告/文档用中文写；代码注释用中文；面向工程读者，别写营销腔。

---

## 6. 踩过的坑（都是实测出来的，别再踩一遍）

1. **docx 正文可能在表格里**。`Ref/` 那份申报书是一张 51×13 的合并单元格大表，正文全在单元格内，
   且合并单元格会让同一段文字重复 13 次。必须"进单元格逐段解析 + 按行去重"，
   否则整份文档只会解析出一节「前言」，其中 13/14 是重复内容。
2. **`paragraph.paragraph_format.outline_level` 在 python-docx 1.2 里不存在**，会抛 AttributeError。
   改为直接读 `w:outlineLvl` XML 属性（`docx.py` 里已封装）。
3. **`（一）` / `（1）` / `①` 的绝对层级因文档而异**：有的直接挂一级标题下，有的挂在 `1.3.1` 之下。
   所以 `heading_level()` 对它们返回 `0` 表示"相对级别"，由调用方挂到最近一个**绝对级别**标题的下一层。
   早期版本按 `len(stack)+1` 算，会出现 `1.3.1 →（1）→（二）` 越嵌越深。
4. **Milvus Lite 的 `uri` 父目录必须先存在**，否则报 "dir not exists"（不是自动创建）。
5. **Milvus Lite 在 Windows 上可用**（pymilvus 3.0.2 + milvus-lite 3.2.2 实测通过）。
   `.env.example` 里"Windows 不可用"的旧注释是错的，尚未修正。
6. **不要用内存里的向量初始化 `NumpyStore` 的维度**：早期 `_load()` 里用 `rows[0]` 覆盖 `self.dim`，
   空表时矩阵是坏的。现在维度要么来自配置、要么来自 `meta` 表。
7. **外键顺序**：`insert_chunks` 前必须先 `upsert_project`，否则 SQLite 外键约束失败（测试里踩过）。
8. 项目已有 `.gitattributes`（`* text=auto eol=lf`），别删——否则 Windows 上 CRLF 会污染 diff。

---

## 7. 待办清单（按优先级，每条都有验收标准）

### P0 · 必须先做，否则系统没被真实验证过

**P0-1　用真实 API 跑通端到端（最高优先）**
30 项测试全用 FakeEmbedder，**真实 embedding / LLM 的调用路径一次都没跑过**。
- 做法：`cp .env.example .env`，请用户提供 `R4P_EMBEDDING_API_KEY` / `R4P_LLM_API_KEY`（**不要**复用原仓库里泄漏的那个 ModelScope token）。
- 建议先把 `R4P_EMBEDDING_DIM` 显式填成模型真实维度（gte-base-chinese 是 768），否则首启会先回退 numpy 再迁移。
- 验收：`ingest` 那份申报书 → `ask "缓粘结预应力的张拉控制要点" --answer` 能返回带章节引用的答案；
  `/api/health` 的 `backend.name` 是 `milvus`、`vectors` > 0、`credentials_missing` 为空。
- 大概率会暴露的问题：ModelScope 端点是否支持 `stream=True`、embedding 批量上限、响应字段名差异。

### P1 · 功能缺口

**P1-1　评估集扩充到 30~50 条**
现在 `eval/dataset.example.jsonl` 只有 3 条数据。没有足量标注，Recall/MRR 数字没有说服力，
而这是课题验收要用的。标注方式用 `gold_snippets`（片段包含原文即算命中）比标 chunk_id 省力。
验收：`python scripts/eval.py --dataset eval/dataset.jsonl --top-k 5` 输出 `dense` vs `hybrid` 对照表。

**P1-2　Dockerfile + GitHub Actions CI**
用户上次已表示可以补。CI 只跑 `pytest` + `ruff`（全离线，无需密钥）。
Dockerfile 基础镜像 `python:3.13-slim`，注意：**不要装 torch**；Milvus Lite 在 Linux 下可用。
验收：镜像能 `docker run` 起来并响应 `/api/health`；Actions 在 PR 上绿。

**P1-3　PDF 通道**
现在是 `adapters/parsers/` 只有 docx。加 PDF 时在 `parsers/` 下新建 `pdf.py`，
输出同样的 `list[Section]` 即可复用 `chunking.py` 之后全流程——**不要把 PDF 逻辑塞进 docx.py**。
推荐 `pymupdf`（`fitz`，无 Java 依赖），按字号/加粗做标题识别，无法识别时退化为一节。

**P1-4　多轮对话的 query 改写**
现在追问（"它的张拉时机呢"）召回会偏。位置在 `ask.py`：`retrieve()` 前加一步改写，
把「上一轮问题 + 当前问题」交给 LLM 生成独立检索式（或用启发式拼接最近 2 轮关键词，省一次调用）。
注意：`/api/chat` 目前是无状态的，需要前端把 `history` 一起 POST 上来，接口要向后兼容（缺省为空）。

### P2 · 清洁工作

**P2-1　`.env.example` 第 1 行 `// 说明：...` 会触发 python-dotenv 警告**
`python-dotenv could not parse statement starting at line 1`（不致命，但脏）。改成 `#` 开头。

**P2-2　`.env.example` 里过时注释**
`R4P_VECTOR_BACKEND` 处写着"Milvus Lite（Windows 不可用）"，实测可用，应改为"Windows 可用"。
另外 `R4P_VECTOR_BACKEND=milvus` 与 `config.py` 默认 `auto` 不一致——建议样例保持 `auto`（对新手更稳）。

**P2-3　公开仓库的标识脱敏（需用户确认，见第 8 节）**

---

## 8. 需要用户拍板的两个问题（上一轮已提出，用户未回复）

1. **仓库是 public**。已自查无硬编码凭据（`.env` 未入库，样例只写 `sk-请替换`），
   但 README 用法示例、`tests/`、`eval/` 里出现了真实课题名「立体聚落」和本地路径 `F:/project/课题/Ref`。
   两个选项：① 用户把仓库改 private；② 把这些替换成 `demo-project` / `/path/to/docs` 这类中性占位。
   **动手前先问，不要擅自改用户可见的示例。**
2. **历史只有一个初始提交**（39 文件 +4 405 行）。若希望 commit 历史更像样
   （解析器 / 存储层 / 检索 / 服务分开），现在是重做代价最低的时刻。用户未表态，默认不动。

---

## 9. 工作约定（接手后请遵守）

- **改完必须跑**：`./.venv/Scripts/python.exe -m pytest -q` 和 `-m ruff check rag4prefa tests scripts`，两者都必须干净。
- **SQL 只写进 `repo.py`**；`api.py` 只做协议转换；外部依赖只进 `adapters/`。
- **新功能必须有离线测试**：`tests/conftest.py` 里有 `FakeEmbedder` 模式可循，不要写依赖付费 API 的测试。
- **不要引入本地模型 / torch**（用户明确要求）。
- **不要把密钥写进任何入库文件**，也不要复用原仓库里的 ModelScope token（那个已泄漏在公开仓库）。
- 提交信息用中文、说清"为什么"；不要用 `--no-verify`、不要 force push。
- 推送前 `git status` 确认没有 `data/`、`.venv/`、`.env` 混进去。

---

## 10. 交接时的验证记录（2026-10-23 16:40）

```
pytest -q                    → 30 passed in 8.81s
ruff check rag4prefa tests scripts → All checks passed
git status -sb               → ## main...origin/main（无未提交改动）
真实文档解析（申报书）        → 30 个章节、层级正确、26 个块、平均 330 tokens
Milvus Lite（Windows）        → 建集合 / upsert / 带标量过滤检索 / 统计 全部实测通过
HTTP                          → GET / 返回页面；/api/health 如实报告后端与缺失项；
                                无 key 时 /api/chat 返回 503 并指出缺哪个环境变量（不是 500）
```

---

## 11. 深入阅读顺序

1. `REFACTOR_PLAN.md`（本目录）—— 重构方案 + **第十二节有实施复盘**，记录计划与现实的 5 处偏差，值得先看。
2. `F:\project\PyProject\milvus\PROJECT_REVIEW.md` —— 原系统评估报告，说明"为什么要重写"的每一条证据都能定位到行号。
3. `README.md`（本目录）—— 面向使用者的安装、API、已知局限。
4. 原系统源码 `F:\project\PyProject\milvus\Vector_search_plus.py`（1 991 行）—— 只在需要对齐某个行为时查阅。
