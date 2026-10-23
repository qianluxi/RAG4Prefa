# RAG4Prefa

面向工程课题文档的检索问答系统：把 `.docx` 设计文档 / 申报材料 / 规范按章节建成知识库，
用**稠密向量 + BM25 混合召回**找依据，再让大模型**流式**给带引用的回答。

> 这是对 `qianluxi/milvus`（v2.8.6）的功能等价重写：同样能"上传文档 → 按项目/年份筛选 → 提问 → 得到答案"，
> 但去掉了 Word COM 绑定、全局可变状态和 4 300 行死代码。设计取舍见 `../RAG4Prefa/REFACTOR_PLAN.md`。

## 它解决了什么

| 常见坑 | 这里的做法 |
|---|---|
| 关键词过滤放在向量 Top-K 之后，召回被截断 | BM25 与向量**并行召回**，再用 RRF 融合 |
| 只按字数硬切，切出来的片段丢了章节上下文 | 按标题树切块，embedding 时带上完整层级路径 |
| 上传同一份文档重复入库 | `(项目, 文件名, 内容哈希)` 唯一 + 确定性 `chunk_id` + upsert |
| 统计数字恒为 0（读的键没人写） | **SQLite 是唯一事实**，计数一律 `COUNT(*)` |
| 多人使用时筛选条件互相串 | 筛选条件随请求传入，无全局状态 |
| "流式"其实是先跑完再逐词播放 | 大模型 token 流直通 SSE |
| 绑死 Windows + Word，进不了容器 | 纯 python-docx 解析，Linux/Docker 可跑 |

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[milvus,dev]"
cp .env.example .env              # 填入 embedding / LLM 的 base_url、api_key、model

python -m rag4prefa doctor        # 看后端、维度、缺哪些配置
python -m rag4prefa ingest "F:/project/课题/Ref" --project 立体聚落 --code kc2026 --year 2026
python -m rag4prefa serve --port 8000
```

打开 http://127.0.0.1:8000 ，左侧勾选项目/年份，直接提问。

命令行也可以直接问答：

```bash
python -m rag4prefa ask "缓粘结预应力的张拉控制要点" --project 立体聚落 --answer
python -m rag4prefa reindex          # 换 embedding 模型后重建索引
```

## 架构

```
api.py        FastAPI 路由（只做协议转换）
  ├── ask.py         检索 + 生成（混合召回 → RRF → 重排 → 流式）
  ├── ingest.py      解析 → 分块 → 向量化 → 写库
  ├── admin.py       项目/文档查询与删除
  ├── filters.py     问题 → Filter（规则优先，可单测）
  ├── chunking.py    标题树分块
  ├── repo.py        全部 SQL
  └── adapters/      embed / llm / vectors / rerank / parsers/docx
```

**数据分层**：SQLite 存事实（项目、文档、片段、任务、向量归档），Milvus 只是可重建的索引。
索引坏了或换后端，`rebuild_store` 直接从归档重建，**不需要重新调用 embedding**。

## 向量后端

`R4P_VECTOR_BACKEND` 三选一：

- `auto`（默认）：能用 Milvus 就用 Milvus，否则退化到 `numpy` 精确检索（结果可复现，靠 SQLite 里的向量归档）。
- `milvus`：Milvus Lite（单文件，无需 Docker）或 standalone，只改 `R4P_MILVUS_URI`。
  - 已验证：Windows + `pymilvus 3.0` + `milvus-lite 3.2` 可用。
  - Lite 不支持 HNSW 参数，统一 `AUTOINDEX`；切 standalone 后可自行换 `HNSW` 调参。
- `numpy`：零依赖暴力余弦，十万级片段内延迟仍在毫秒级。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 后端、索引维度、缺失的凭据（不再静默降级） |
| POST | `/api/projects/{name}/documents` | 上传 docx（multipart）→ `job_id` |
| GET | `/api/jobs/{id}` | 入库进度与逐文件结果 |
| GET | `/api/projects` | 项目列表（文档数/片段数为真实值） |
| GET | `/api/projects/{name}` | 详情 + 章节分布 + LLM 摘要（带缓存） |
| DELETE | `/api/projects/{name}` | 级联删除（SQLite + 向量） |
| POST | `/api/search` | 非流式检索（调试与评估用） |
| POST | `/api/chat` | SSE 流式问答 |

SSE 事件：`meta`（生效条件、候选数、各阶段耗时）→ `token`* → `citations` → `done`；异常走 `error`。

## 测试与评估

```bash
pytest -q                                  # 30 项，全部离线（假向量 + 临时 Milvus/numpy）
python scripts/eval.py --dataset eval/dataset.jsonl --top-k 5
```

`scripts/eval.py` 输出 Recall@k / MRR / nDCG@k，并对比 `dense` 与 `hybrid` 两种召回模式——
这是"混合检索到底有没有用"的量化依据。

## 已知局限

- 解析是基于排版结构的启发式：报表类文档正文在表格里，已支持；但图片、文本框内的文字不提取。
- token 用字/词估算而非真实 tokenizer，分块边界与模型实际计数有误差（阈值留了余量）。
- 未做多轮对话改写：追问需要带上上下文关键词，否则召回会偏。
- 单进程设计（SQLite + 进程内任务）。要多副本时把 SQLite 换成 Postgres、任务换队列，接口层无需改动。
