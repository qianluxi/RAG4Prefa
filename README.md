# RAG4Prefa

面向工程课题文档的检索问答系统：把 `.docx` 设计文档 / 申报材料 / 规范按章节建成知识库，
用**稠密向量 + BM25 混合召回**找依据，再让大模型**流式**给带引用的回答。

> 这是对 `qianluxi/milvus`（v2.8.6）的功能等价重写：同样能"上传文档 → 按项目/年份筛选 → 提问 → 得到答案"，
> 但去掉了 Word COM 绑定、全局可变状态和 4 300 行死代码。设计取舍见 `REFACTOR_PLAN.md`。

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
| 换模型后新旧向量混在一个索引里，检索静默变差 | 模型 / 维度 / 归档 / 后端条数四方校验，不一致就拒绝写入与检索 |
| 更新文档时 embedding 失败，旧版本被顺手删掉 | 先算完新向量、**先写索引再切事实**；失败时撤回新增向量，旧版本照常可检索 |
| 索引残缺时系统照常返回"看起来正常"的答案 | `/api/chat` 在开流之前就 503；`/api/search` 明确报错；`verify` 能逐 id 定位差异 |

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
python -m rag4prefa reindex                  # 换 embedding 模型后全量重建
python -m rag4prefa reindex --project 立体聚落  # 只重建一个项目（维度不变时可用）
python -m rag4prefa verify                   # 逐 id 核对「片段 / 归档 / 索引」，出问题返回值非 0
```

`reindex` 的语义：**先算完整个范围的新向量，再写 SQLite 归档，最后整体替换向量索引**；
中途失败不会留下半成品。只重建单个项目时不会清空其他项目的索引，但**必须沿用索引现有的模型**——
维度相同也不算数，两个模型的向量空间不能混。换模型走不带 `--project` 的全量重建，
维度变化会明确拒绝局部执行。详见下面「索引一致性」。

`verify` 与 `/api/health` 的分工：health 做廉价的条数比对（每次请求前都跑），
verify 做逐 id 的精确比对（要把索引主键全拉一遍，慢但能发现"条数相同、内容不同"）。

## 架构

```
api.py        FastAPI 路由（只做协议转换）
  ├── ask.py          检索 + 生成（混合召回 → RRF → 重排 → 流式）
  ├── ingest.py       解析 → 分块 → 向量化 → 写库；含 reindex / rebuild_store
  ├── consistency.py  索引一致性（模型/维度与归档是否自洽，能不能写、能不能查）
  ├── admin.py        项目/文档查询与删除
  ├── filters.py      问题 → Filter（规则优先，可单测）
  ├── chunking.py     标题树分块
  ├── repo.py         全部 SQL
  └── adapters/       embed / llm / vectors / rerank / parsers/docx
```

**数据分层**：SQLite 存事实（项目、文档、片段、任务、向量归档），Milvus 只是可重建的索引。
索引坏了或换后端，`rebuild_store` 直接从归档重建，**不需要重新调用 embedding**。

## 向量后端

`R4P_VECTOR_BACKEND` 三选一：

- `auto`（默认）：能用 Milvus 就用 Milvus，否则退化到 `numpy` 精确检索（结果可复现，靠 SQLite 里的向量归档）。
- `milvus`：Milvus Lite（单文件，无需 Docker）或 standalone，只改 `R4P_MILVUS_URI`。
  - Lite 不支持 HNSW 参数，统一 `AUTOINDEX`；切 standalone 后可自行换 `HNSW` 调参。
  - ⚠️ **Windows 上路径必须是纯 ASCII**：faiss 用窄字符 `fopen` 打开索引文件，中文目录名会让
    任何一次 `flush` / `release+load`（重建、换维度、重启后重开）直接失败。程序在打开集合前
    就会拒绝并说明原因，`auto` 会自动回退 numpy。要用 Milvus 就把 `R4P_DATA_DIR` 指到
    `C:/rag4prefa-data` 这类纯 ASCII 路径。Linux 没有这个限制。
- `numpy`：零依赖暴力余弦，十万级片段内延迟仍在毫秒级。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 后端、索引维度、模型一致性、缺失的凭据（不再静默降级） |
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
pytest -q                                  # 244 项，全部离线（假向量 + 本地假 OpenAI 服务 + 临时 Milvus/numpy）
python scripts/eval.py --dataset eval/dataset.jsonl --top-k 5
```

CI（`.github/workflows/ci.yml`）在每次 push / PR 上跑同一组命令：`ruff check rag4prefa tests scripts`
→ `pytest -q`。全部离线，不需要任何 Key。

`scripts/eval.py` 输出四组对照——`dense`（纯向量）、`sparse`（纯 BM25）、`hybrid`（向量 + BM25 + RRF）、
`rerank`（混合 + 重排）——这是"混合检索和重排各自带来多少收益"的量化依据。
没有配 `R4P_RERANK_URL` 时 `rerank` 一栏等于 `hybrid`（NoopReranker）。

指标定义（写死在 `tests/test_eval_metrics.py` 里，不接受含糊）：

- 三个指标共用**同一个 gold 集合，单位是 chunk**。
- `Recall@k = |TopK ∩ Gold| / |Gold|`，分母是**标注的相关项总数**，不是返回条数。
- `MRR` 取第一个相关结果的倒数排名。
- `nDCG@k` 按二值相关性计算，IDCG 用标注的相关项总数构造理想排序。
- 报告里每种召回方式**单独列一行有效样本数**：某个方法在部分题上失败时，均值建立在不同的
  题目集合上，不能拿"总题数"糊过去。

标注优先用 `gold_chunk_ids`（无歧义）；图省事可以写 `gold_snippets`，脚本会把每个片段解析成
**该题筛选范围内**包含它的全部 chunk 再计入 gold——不按范围解析的话，别的项目里出现的同一段
文字也会被算成相关项，而本题根本召回不到，Recall 会被系统性压低。解析不设上限（不静默截断），
写错的 `chunk_id` 会被剔除并写进报告的数据问题里。

## 索引一致性

一个索引里只能有一套向量空间。判断依据是**声明 + 事实 + 逐 id 对应**，任一不满足就**拒绝服务而不是降级**：

- `meta.embedding_model` / `embedding_dim`：上一次成功建索引时声明的模型与维度；
- `vectors.model` / `dim`：归档里每一行实际是谁算的；
- 片段与归档的逐 id 对应（缺向量、孤儿向量、模型未标注都算不一致）；
- 后端实际条数与归档条数的交叉校验（归档对、Milvus 少写几条，只查 SQLite 是看不出来的）。

出现任意一条，增量入库、局部 `reindex`、检索都会拒绝；`/api/health` 的 `index.consistent` /
`index.issues` 如实报告，`ok=false` 时先看这里。换模型只能全量 `reindex`（不带 `--project`），
局部重建必须沿用现有模型。

另外两条准入，被拒绝时同样是"什么都不改"：

- **库里有内容，就必须有一个被确认过的活动后端**（`meta.vector_backend`）。没有登记值意味着
  确认动作试过并失败了，不是"还没配"——此时查询（HTTP / CLI / 评估）、普通入库、删项目一律拒绝，
  `python -m rag4prefa rebuild` 从归档重建并登记即可恢复。`doctor` / `verify` / `rebuild` /
  全量 `reindex` 保留入口，否则没法修。
- **后端身份 = 类型 + 存储位置**：`meta.vector_backend_id` 记 `milvus|<uri>|<collection>` 或
  `numpy`。共用一份 `app.db` 的进程必须用同一份 `R4P_MILVUS_URI` / `R4P_MILVUS_COLLECTION`，
  否则会在第一次比对时被指出来（而不是两边各写一半、谁也查不到谁）。

`chunk_id` 是**内容寻址**的（`doc_id + 章节路径 + 正文哈希`），不是位置序号：同一 id 必然是同一段正文。
这是"先写索引、后切事实"能安全的前提——改一段不会让后面所有片段换号，也不会出现"新向量配旧正文"。

## 多用户与 API Key

前端右上角「API 设置」可以让每个人填自己的 Key，存在**本机浏览器的 localStorage**，
之后自动随请求带上；换人只需在列表里点一下，不用重新输入。服务端只在请求期间持有 Key，
不落库、不写日志。

请求头约定（curl 或脚本直接调也能用）：

| 头 | 含义 |
|---|---|
| `X-R4P-User` | 用户名，仅用于显示；中文需 `encodeURIComponent`（HTTP 头只能是 ASCII） |
| `X-R4P-Embedding-Key` | 入库与检索用的 Key |
| `X-R4P-LLM-Key` | 生成答案用的 Key |
| `X-R4P-LLM-Model` | 选哪个对话模型，取值须在 `GET /api/llm-models` 返回的清单里 |

两个 Key 可以只给一个：只带 embedding 就是"入库花我的额度，问答走公共配置"。
**两个 Key 都不给、只带 `X-R4P-LLM-Model` 也合法**：用服务端公共额度跑自选对话模型。
四个头都不带时完全沿用 `.env` 里的公共配置，旧脚本与 CLI 行为不变。
清单以外的模型名不会报错，而是静默回落到服务端默认——避免浏览器里存着已下线的名字
时整个请求失败。

**为什么 embedding 模型不能选、对话模型可以选**：embedding 决定向量空间。若允许每人自选，
同一批文档里就会并存两套向量——余弦相似度照样算得出来，只是语义错位，检索结果看起来正常
却不可信。所以 `embedding` 的 `base_url` / `model` / `dim` 由服务端 `.env` 统一决定。
对话模型只读"已检索到的片段"、不参与建索引，换它只影响措辞与推理深度，因此开放给用户。
`/api/credentials/verify` 会真调一次并回报维度，与服务端建库时的维度不一致会直接拒绝。

预设的对话模型都是逐个真调通过的（`Qwen3.5-35B-A3B` / `Qwen3.8-Flash-Next` /
`Qwen3.8-27B` / `Qwen3.5-122B-A10B`）。**不要直接拿 `/v1/models` 的结果去渲染下拉**：
那个目录里 DeepSeek V4 全系、MiniMax-M3 等一大半调用会返回 `has no provider supported`。

**这不是身份认证**：知道用户名的人可以填自己的 Key 顶替，服务端不校验密码，取回 Key 也不设防。
它解决的是"几个人分摊额度、各自记住自己的 Key"，不是"谁有权访问知识库"。

## 已知局限

- 解析是基于排版结构的启发式：报表类文档正文在表格里，已支持；但图片、文本框内的文字不提取。
- token 用字/词估算而非真实 tokenizer，分块边界与模型实际计数有误差（阈值留了余量）。
- 未做多轮对话改写：追问需要带上上下文关键词，否则召回会偏。
- 单进程设计（SQLite + 进程内任务）。要多副本时把 SQLite 换成 Postgres、任务换队列，接口层无需改动。
- **并发**：同一个 `sqlite3.Connection` 被请求与后台入库线程共用（`check_same_thread=False`），
  没有额外的写锁。单用户没问题，多人并发写之前应先改成每线程独立连接。
- **任务恢复**：进程重启时会把遗留的 `running/queued` 标成失败（不会永远显示"运行中"），
  但任务本身不能断点续跑，需要重新触发。
- **无身份认证**：上传与删除接口没有权限校验；「API 设置」里的用户名只是标签，不构成登录，
  别人知道用户名就能用自己的 Key 顶替。只能部署在可信内网。
- **上传无大小上限**：按后缀校验后一次性读入内存，没有流式落盘与容量限制。
- **Milvus 与中文路径**：Windows 下 Milvus Lite 无法在含非 ASCII 字符的路径下建索引（faiss 限制），
  程序会提前拒绝并自动回退 numpy；要跑 Milvus 请把数据目录指到纯 ASCII 路径。Linux 不受影响。
- **重建期间有极短不可用窗口**：全量重建（含同维度）走"写暂存集合 → rename 切换"，
  两次 rename 之间检索可能拿到集合不存在的错误。换来的是"绝不会返回新旧混杂的结果"。
- **日常一致性检查是必要条件，不是充分条件**：每个请求只比条数与模型/维度，发现不了
  "索引少了一个 id、同时多了另一个 id"这种等量替换。要确认到 id 级请跑
  `python -m rag4prefa verify`（代价是把索引主键全拉一遍，不能放进请求路径）。
  另外 SQLite 归档与向量库不在同一事务里，重建中断时会靠 `meta.index_state` 标记
  阻断检索——标记为 `ready` 之外的任何值都不提供服务，直到 `reindex` 或 `rebuild` 完成。
- **`rebuild` 的"完成"有明确判据**：归档可用（无缺行、每行带模型名、全库一套模型与维度）
  → 后端条数 = 归档条数 → 归档的模型与维度写回 `meta` 且标记回 `ready`（同一事务）。
  任一条不满足就**保留标记并返回非零**，不会出现"命令说恢复成功、实际还是查不了"。
  反过来说，归档本身残缺（有正文没向量）时 `rebuild` 帮不上忙，只能跑 `reindex` 重算。
