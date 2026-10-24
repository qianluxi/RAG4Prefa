# 交接说明 · RAG4Prefa

> 写给接手的 agent：你**没有**上一轮对话的上下文，这份文件是唯一入口。
> 读完这一份就能开始干活；需要更深的背景再看第 11 节列的文档。

**交接时间**：2026-10-24 22:00
**交接时状态**：可运行、**165 项离线测试全绿**；真实厂商 API 已端到端验证
（embedding 走硅基流动国际站 `Qwen/Qwen3-Embedding-0.6B`、LLM 走魔搭 `Qwen/Qwen3.5-35B-A3B`，
`POST /api/credentials/verify` 两项均通，`dim=1024` 与服务端配置一致）。
第九轮修复的是**失败路径**，不是功能：跨组件的索引切换留痕、入库部分写入的撤回清单、
凭据校验的回退来源、以及"只用公共额度换模型"在界面上做不到。
动手前请先读第 3 节的「索引不变量」，那十条是这批代码存在的理由。

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
| 当前提交 | 以 `git log -1 --oneline` 为准；本轮修复的基线是 `8fa40bf`（本地与 `origin/main` 一致，工作区干净） |
| Python | 3.13，**虚拟环境已建好**：`F:\project\课题\RAG4Prefa\.venv` |
| 解释器路径 | `./.venv/Scripts/python.exe`（Windows；所有命令都在项目根目录执行） |
| 依赖状态 | 已装齐（fastapi / uvicorn / python-docx / pymilvus 3.0.2 / milvus-lite 3.2.2 / openai / jieba / numpy / pytest / ruff） |
| 未安装 | `torch`、本地 embedding 模型（**按用户明确要求不装本地模型**） |

原版项目（只读参考，**不要改它**）：`F:\project\PyProject\milvus`，已检出 v2.8.6。
真实测试语料：`F:\project\课题\Ref\立体聚落异型大悬挑百年建造高质量建造技术研究-申请书26.10.23.docx`

**先跑这四条确认环境可用**：

```bash
cd "F:/project/课题/RAG4Prefa"
./.venv/Scripts/python.exe -m pytest -q          # 期望：165 passed
./.venv/Scripts/python.exe -m ruff check rag4prefa tests scripts   # 期望：All checks passed
./.venv/Scripts/python.exe -m rag4prefa doctor   # 看后端、维度、缺哪些凭据
./.venv/Scripts/python.exe -m rag4prefa verify   # 逐 id 核对片段/归档/索引（空库也返回 0）
```

---

## 3. 代码地图（3 797 行 Python + 731 行单页 HTML + 3 429 行测试）

```
rag4prefa/
├── config.py      81    pydantic-settings，全部配置以 R4P_ 前缀从 .env 读
├── domain.py     122    Chunk / Section / Hit / Filter / JobState，纯数据，无依赖
├── db.py         129    SQLite schema（projects / documents / chunks / chunks_fts / vectors / jobs / meta / embed_cache）
├── repo.py       767    **全部 SQL 都在这里**，没有别处写 SQL
├── consistency.py 305   **索引一致性**：状态诊断、阻断判定、逐 id 核对（verify）、切换标记
├── ingest.py     447    解析 → 分块 → 向量化 → 写索引 → 切事实；含 rebuild_store / reindex
├── chunking.py    90+   按标题树切块；chunk_id 内容寻址（同 id 必同正文）
├── tokens.py      51    中英混排的 token 估算（非真实 tokenizer）
├── ask.py        186    混合召回 → RRF 融合 → 可选重排 → 流式生成
├── filters.py    103    自然语言问题 → Filter（规则法，可单测，无 LLM 调用）
├── prompts.py     38    系统提示词与引用格式
├── admin.py       64    项目/文档的查询与级联删除
├── services.py   240    组件装配；Milvus 不可用时回退 numpy；`for_credentials()` 出按用户视图
├── credentials.py 363   **按用户解析 Key 与对话模型**：请求头 → 凭据池；报错根因分级
├── api.py        509    FastAPI 路由（只做协议转换，不含业务逻辑）
├── cli.py        270    init / serve / doctor / ingest / reindex / rebuild / ask / projects / verify
└── adapters/     990+   embed / llm / vectors / rerank / parsers/docx —— 所有外部依赖都关在这里
```

前端只有一个文件 `rag4prefa/web/index.html`（731 行，原生 JS，无构建步骤）。
`tests/` 165 项，**全部离线**（FakeEmbedder 确定性假向量 + 本地假 OpenAI 服务 + 临时 Milvus/numpy），
不花一分钱 API 费。其中 `test_real_sdk_chain.py` 走真实 openai SDK 与 SSE，专门防止「假向量绕过了缓存/网络」这类漏检；
`test_api_credentials.py` 让假服务记录 `Authorization`，用来证明"用户的 Key 真的被拿去调了厂商接口"，
而不是"接口返回了 200 就以为配对了"；
`test_consistency.py` / `test_ingest.py` 里有大量**故障注入**用例，反向断言"必须被拒绝""旧版本必须还在"，
防止把"程序执行了预期操作"当成"业务不变量成立"。

### 数据分层（关键，改动前必须理解）

**SQLite 是唯一事实来源，向量库只是可重建的索引。**

- `vectors` 表存了一份向量归档（float32 bytes）→ 换后端、索引损坏时 `rebuild_store()` 直接重建，**无需重新调 embedding**（省钱省时）。
- 所有计数（文档数、片段数）都是 `COUNT(*)` 现算，不存在"统计恒为 0"的可能。
- 查向量：Milvus 拿到 `(chunk_id, score)` 后回 SQLite 取正文 → 所以 Milvus 只是 `chunk_id → 向量` 的索引。

### 索引不变量（改这些代码前先读，`consistency.py` 是它的实现）

1. **一个索引只有一个向量空间。** `vectors.model` 必须全库一致，且等于 `meta.embedding_model`；
   **没有模型标注的历史行也算异常**（无法证明它是哪个空间算的）。维度相同不算数——
   两套 768 维向量混在一起，余弦照样算得出来，只是语义错位。
2. **换模型只能全量。** `reindex --project` 必须沿用现有模型；换模型走不带 `--project` 的全量重建。
3. **事实与归档同事务，索引先行。** 顺序是「算向量 → 写索引（派生）→ 一个事务切片段+归档 → 删淘汰向量 → 记 meta」。
   为什么索引在事实之前：`chunk_id` 内容寻址，同 id 必同正文，先写索引不会让旧版本配错正文；
   反过来（先切事实）一旦写索引失败，就会出现"新正文配旧向量"的双版本共存。失败时撤回本次新增的向量，
   旧版本完整可用；撤回也失败时，条数不一致会让检索被拒绝，而不是静默出错。
4. **归档与片段一一对应。** `vectors` 里不该有 `chunks` 里不存在的行，也不该缺行；
   缺行/孤儿都进 `blockers`，`reindex` 会补齐/清掉。
5. **重建不毁掉可用的旧索引。** 换不换维度都走同一条路：写 `__staging` → 校验条数 → rename 切换 → 删 `__retired`；
   进程被杀由 `_recover()` 在下次启动时还原。绝不原地覆盖正式集合（覆盖到一半就是新旧混杂）。
6. **更新文档不能丢旧版本。** embedding 在切换之前完成；写索引失败时旧片段、旧归档、旧索引原样保留。
7. **配置与索引冲突时拒绝服务，而不是降级。** 检索会返回 503 + 一句能读懂的话，
   而不是给出一份看起来正常、实际无关的引用。判断依据含**后端实际条数**——只查 SQLite 发现不了"归档对、索引少写几条"。
8. **chunk_id 由内容决定，不由位置决定**（`doc_id + 章节路径 + 正文哈希 + 同段内序号`）。
   同一 id 必是同一段正文；改一段只影响那一段，插入一段不会让后面全部换号。
9. **多用户可换 Key、可换对话模型，但 embedding 模型锁死。** 前端「API 设置」里的
   Key 与模型随 `X-R4P-*` 头到达服务端，`credentials.CredentialPool` 替换两个 Key 与
   `llm_model`，`embedding_base_url` / `embedding_model` / `embedding_dim` 一律沿用服务端配置。
   边界由用途决定：LLM 只读「已检索到的片段」、不参与建索引，换它只影响措辞与推理深度；
   embedding 换一个就是换一套语义坐标，同一索引里会并存两套向量。这不是权限设计
   ——它不是鉴权（见第 6 节）。

   可选模型清单是 `credentials.LLM_CHOICES`（白名单 + 中文说明），`GET /api/llm-models`
   供前端渲染下拉。**白名单不是从 `/v1/models` 抄的**：那个目录里 DeepSeek V4 全系、
   MiniMax-M3、Step-3.5-Flash 都在，调用却一律 `has no provider supported`。
   只有逐条真调通的才收——列在目录里 ≠ 真的能调。清单外的名字一律静默回落服务端默认，
   避免旧浏览器存着已下线模型名时整个请求失败。

10. **跨组件的切换必须留痕，而不是靠事后比对。** SQLite 归档与向量库不在同一个事务里，
    没法用 `BEGIN/COMMIT` 做两阶段提交，所以 `reindex` 在写归档**之前**把
    `meta.index_state` 置成 `rebuilding:<范围>`，全部切换成功后才连同模型、维度一起
    改回 `ready`（`repo.set_index_meta`，三者同事务）。只要这个值不是 `ready`，
    检索与增量写入一律拒绝——因为中断那一刻，归档与索引的**条数、模型名、维度可以完全
    一致**，逐项校验一条都挑不出来，而检索会拿旧向量去配新正文。
    恢复正常态有两条路：`reindex`（重新算向量、花钱）或 `rebuild`（直接用归档里的
    现成向量，不花钱）。局部重建只能消除**同项目**的标记；全量的坑必须全量填。

### 请求链路（`/api/chat`）

```
问题 + Filter → ask.build_filter()（规则补全筛选条件）
  → consistency.ensure_queryable()（混模型/缺向量/孤儿/后端条数不符 → 503，不进入生成）
  → 并行：向量 Top-50  ‖  FTS5 bm25 Top-50
  → RRF 融合（k=60）→ 取 Top-8 → 可选远程重排
  → 组装带章节路径的上下文 → LLM stream=True → SSE
SSE 事件：meta → token* → citations → done；异常走 error
```

`/api/chat` 的一致性校验刻意放在**开始流之前**：一旦开始 SSE，HTTP 状态码就改不了了，
只能发一个 error 事件，而客户端此时已经开始渲染。能给出 503 的时候要给 503。

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
5. **Milvus Lite 在 Windows 上可用，但路径必须纯 ASCII**（本轮实测的新结论，前几轮记错了）。
   faiss 用窄字符 `fopen` 写索引文件，`F:\project\课题\...` 这类含中文的路径在
   `flush` / `release+load`（重建、换维度、重启后重开）时会报
   `faiss::FileIOWriter ... No such file or directory`，之后集合直接不可用。
   同一份数据放 `C:/tmp/...` 完全正常，所以早期测试全在临时 ASCII 目录里跑，一直没暴露。
   现在 `MilvusStore.__init__` 会在开集合前拒绝并说明两条出路（换 ASCII 数据目录 / 用 numpy），
   `auto` 后端会自动回退 numpy。
6. **不要用内存里的向量初始化 `NumpyStore` 的维度**：早期 `_load()` 里用 `rows[0]` 覆盖 `self.dim`，
   空表时矩阵是坏的。现在维度要么来自配置、要么来自 `meta` 表。
7. **外键顺序**：`insert_chunks` 前必须先 `upsert_project`，否则 SQLite 外键约束失败（测试里踩过）。
8. 项目已有 `.gitattributes`（`* text=auto eol=lf`），别删——否则 Windows 上 CRLF 会污染 diff。
9. **内容寻址的 id 让"改一段"只影响那一段**：同一份申报书改末尾一段，26 个片段里 25 个 id 不变。
   测试断言这一性质时不要再用"id 数量变化"来判断，要看存量 id 的保留率。

---

## 7. 待办清单（按优先级，每条都有验收标准）

### 已完成（2026-10-23 第四轮，按第三次外部复审意见修复）

| 问题 | 根因 | 修复 |
|---|---|---|
| **一致性检查能报告却不能阻断** | `ensure_writable` 在归档条数为 0 时直接放行；`ensure_queryable` 只拦"混模型"，缺向量、孤儿、后端条数不符都能照常检索 | 判断集中到 `consistency._check()`，`blockers` 覆盖：混模型 / 未标注模型 / 混维度 / meta 与归档不符 / 缺向量 / 孤儿 / 后端条数与归档不符。**只有全空库才放行**。`ensure_writable` / `ensure_queryable` / `health` 用的是同一份判断 |
| "归档为 0 但片段还在"被判为一致 | 用"归档条数 == 0"代表"索引是空的" | 新增 `missing_vector_count` / `orphan_vector_count` / `unlabeled_vector_count`，按逐 id 对应关系判断；`/api/health` 分别报 `missing_vectors` / `orphan_vectors` / `unlabeled_vectors` |
| 只比 SQLite 内部的两个数 | 后端少写几条时，归档与片段完全自洽 | `inspect(conn, store_count=...)` 把后端实际条数纳入阻断；Milvus 的 `count()` 改用 `count(*)` 聚合（`get_collection_stats` 的 row_count 语义随版本而异，不能当阻断依据） |
| **更新文档的失败窗口** | 顺序是"先切片段、再写索引"，写索引失败时事实已是新版 | 改成"先写索引、再同事务切片段+归档、最后删淘汰向量"；失败时**精确撤回本次新增的 id**（不能撤旧版本沿用的 id），撤回失败则由条数不一致触发拒绝 |
| 同维度全量重建原地覆盖 | `if self.dim == dim: upsert + prune`，覆盖一半就是新旧混杂 | 删掉这条分支，一律 `__staging` → 校验 → rename；代价是重建期间有极短不可用窗口，换来"绝不返回混杂结果" |
| 进程被杀后文档永远停在 `indexing` | 只处理了 `jobs`，没处理 `documents` 状态 | 启动时 `mark_interrupted_documents()` 标成 failed，并提示重新导入 |
| **Gold 标注没按题型范围解析** | `resolve_gold` 用全库匹配，别处出现的同一段文字也算相关项 | `resolve_gold(conn, row, filt)` 用该题**生效的** filter 解析；先算 `effective` 再解析 gold，两者范围一致 |
| Gold 静默截断 | `chunk_ids_containing` 默认 `limit=50` | 默认不截断；调用方要给上限必须显式传并自行报告 |
| 失效/越界的 gold id 被当成有效标注 | 只做了字符串去重 | `repo.select_chunk_ids` 校验"存在且在范围内"，剔除并写进报告的数据问题 |
| 报告只有一个总样本数 | 某方法在部分题上失败时，均值建立在不同题目集合上 | `evaluated[mode]` 逐方法计数，报告新增「有效样本」列，样本数不一致时给出显式提示 |
| `verify` 能力缺失 | 条数相同但 id 不同（写坏、切换半成功）没有廉价手段发现 | 新增 `python -m rag4prefa verify`：拉取索引全部主键，与片段、归档三方比对 |

**对第三次复审那份 6 项验收清单的逐条回应**（都可当场验证）：

| 验收项 | 状态 | 怎么验 |
|---|---|---|
| 一致性检查真正阻断异常检索 | 已闭环 | `test_consistency.py` 四种故障注入 + `test_real_sdk_chain.py` 的 503 用例 |
| 文档更新具备失败保护 | 已闭环 | `test_ingest.py`：切事实失败 / 索引写一半失败 / 删旧向量失败 |
| 同维度全量重建经过故障测试 | 已闭环 | `test_milvus_store.py::test_rebuild_same_dim_also_goes_through_staging` 等 |
| Gold 标注与检索范围一致 | 已闭环 | `test_eval_pipeline.py` 的范围内解析、不截断、失效 id 三条 |
| 评测报告反映真实有效样本数 | 已闭环 | 报告新增「有效样本」列，样本数不等时显式提示 |
| 交接文档与实际仓库状态一致 | 已闭环 | 本节 + 第 2 节 + 第 10 节均按本轮实测重写 |

### 已完成（2026-10-23 第三轮，按第二次外部复审意见修复）

| 问题 | 根因 | 修复 |
|---|---|---|
| **同维度换模型会静默混用向量** | `reindex --project` 只比维度，不比模型；上一轮的测试甚至把这种混用写成"成功用例" | 新增 `consistency.py`：以 `meta.embedding_model`（声明）+ `vectors.model`（事实）双依据判断。增量入库、局部重建、检索三处都强制校验；局部重建只能沿用现有模型，换模型必须全量。`test_consistency.py` 反向断言"必须被拒绝" |
| **索引重建不是原子切换** | `MilvusStore.rebuild()` 一上来就 `drop_collection`，后续失败就没有索引了 | 同维度：原地 upsert + 剪孤儿，全程可检索；换维度：先写 `__staging` 并校验条数 → `rename_collection` 切换（元数据操作，不是删了重建）→ 成功后再删 `__retired`。失败自动回滚，启动时 `_recover()` 收拾残留 |
| **更新文档时 embedding 失败会丢掉旧版本** | `ingest_file()` 先删旧片段再调 embedding | 顺序改为「解析 → **先算完全部向量** → 一个事务换片段 → 覆盖向量 → 记 content_sha」。另修一个连带 bug：失败后重试会被 `content_sha` 判成 skipped，现在要求上一版是 `indexed` 才跳过 |
| 重建后归档留孤儿向量 | 全量重建只覆盖不删除；`iter_archived_vectors` 也不看 chunks 是否存在 | 全量重建调用 `delete_orphan_vectors()`；归档读取改为 JOIN chunks，孤儿永远不会被灌回索引 |
| 已有集合的维度不校验 | `_ensure_collection()` 发现集合存在就直接返回 | 从 schema 读出实际维度比对，不一致就明确失败并给出重建命令；配置未给维度时沿用集合实际维度 |
| health 不反映索引一致性 | 只检查凭据与后端是否存在 | `/api/health` 增加 `index.consistent` / `index.issues`，不一致时 `ok=false`，前端徽标区分"配置缺失"与"索引异常" |
| 评估指标的计数单位不统一 | Recall 按原文片段计数，MRR/nDCG 按 chunk 排序 | 片段标注在加载时解析成包含它的 chunk 集合，三个指标共用同一个 gold；补 4 组对照（dense / sparse / hybrid / rerank）与参数校验 |
| `Filter.feature` 解析了但不参与召回 | 早期设计的遗留字段 | 删除。能解析却无效的字段比没有更糟——它让人以为筛选生效了 |
| 任务永远停在 running | 入库任务在进程内跑，进程重启后没人收尾 | 启动时 `mark_interrupted_jobs()` 把 running/queued 标成 failed 并保留原进度说明 |

### 已完成（2026-10-23 第二轮，按第一轮外部评估意见修复）

| 问题 | 根因 | 修复 |
|---|---|---|
| **真实 embedding 链路必崩** | `Embedder._cache_get()` 查 `embed_cache` 表，但 `db.py` 的 SCHEMA 里没建 | 补建表（`(model, sha)` 主键）；测试见 `tests/test_embed_cache.py` |
| 样例配置首启即挂 | 样例同时给了 `BACKEND=milvus` 与 `DIM=0`，建集合时无维度可用 | 样例改 `auto`、密钥留空、首行改 `#`；显式 `milvus` 时报错带自救指引 |
| **局部 reindex 会删别的项目** | 无条件 `store.clear()`，而 numpy 后端的 `clear()` 删的是归档表 | 新增 `VectorStore.rebuild()`；局部重建只 upsert 本次范围；维度变化强制全量并明确报错 |
| reindex 不同步归档与元数据 | 新向量只写索引，`vectors` 归档仍是旧的，`meta` 不更新 | 顺序改为「先算完 → 写归档 → 写索引 → 写 meta」 |
| **Milvus 重启后检索不可用** | 新进程打开已有集合处于 `released` 状态，未 `load()` | 初始化与重建时显式 load；回归测试 `test_reopened_collection_is_searchable` |
| CLI `ask` 一跑就 TypeError | `ask.search()` 是 `async def`，却直接解包返回值 | 改 `asyncio.run()`；测试 `tests/test_cli.py` |
| **Recall@k 算成了 Precision** | 分母用返回条数而非标注相关总数 | 改 `|TopK ∩ Gold| / |Gold|`，nDCG 的 IDCG 同步修正；定义写进 `tests/test_eval_metrics.py` |
| SQL 散落在 repo 之外 | `services.py` / `ingest.py` 里有裸 SQL | meta 读写与重建查询收回 `repo.py` |

### P0 · 仍待完成

**P0-1　接真实厂商 API 跑一遍　→ 2026-10-24 已完成**
第九轮已在**真实 Milvus + 真实厂商**上跑通全链路（详见第 8 节第九轮记录）：
入库 3 块 → 中断重建被阻断 → `rebuild` 恢复 → `verify` 三方一致 → 检索命中 →
LLM 返回带 `[1]` 引用的答案。embedding 与 LLM 分属硅基流动国际站与魔搭，
`stream=False` 的短回答、批量上限、响应字段都没有意外。

尚未覆盖的真实链路只有两项，都不影响正确性：**SSE 流式**只在假服务上验证过
（`test_real_sdk_chain.py` 走真实 openai SDK，但目标是本地假服务），
以及**多用户并发**（`CredentialPool` 的驱逐与连接复用还没在真实并发下压过）。

具体做法与换厂商时的注意事项：
- 做法：`cp .env.example .env` 后填 `R4P_EMBEDDING_API_KEY` / `R4P_LLM_API_KEY`
  （**不要**复用原仓库泄漏的 ModelScope token）。
- 默认组合是「硅基流动**国际站** `Qwen/Qwen3-Embedding-0.6B` 做 embedding + 魔搭 Qwen 做 LLM」，
  两家 key 分别申请、互不通用。
- **硅基流动的国内站与国际站是两套独立账号**，这一点务必先确认再填 Key：

  | | 国内站 | 国际站 |
  |---|---|---|
  | 控制台 | cloud.siliconflow.cn | cloud.siliconflow.com |
  | 接口 | `api.siliconflow.cn/v1` | `api.siliconflow.com/v1` |
  | 向量模型 | `BAAI/bge-m3`（1024 维） | `Qwen/Qwen3-Embedding-0.6B/4B/8B`（1024/2560/4096） |
  | 重排 | `BAAI/bge-reranker-v2-m3` | `Qwen/Qwen3-Reranker-0.6B/8B` |

  两站都发 `sk-` 开头的 Key、都叫硅基流动，**唯一的区别是域名**。填串了第一次调用就收到
  `30014 Token is invalid`，与「Key 被删」同一个错误码，看报错分不出来。本机实测过：
  国际站申请的 Key 打国内站地址，100% 复现。
- **换 embedding 厂商时五项要一起改**：`BASE_URL` / `API_KEY` / `MODEL` / `DIM` / `BATCH`。
  批量上限是硬约束：硅基流动 32 可用（国际站实测 40 条一次也过），阿里百炼
  `text-embedding-v3/v4` **只能 10**，超了直接 400。
- 一个部署只能配一个 `embedding_base_url`，而用户只能换 Key、不能换地址。所以**同一部署的
  所有用户必须用同一个站的 Key**；混用时 Key 会一直 401，`auth_failure_hint()` 会提示去核对域名。
- 验收：`ask "缓粘结预应力的张拉控制要点" --answer` 返回带章节引用的答案；
  `/api/health` 的 `backend.name` 为 `milvus`（或已在 notes 里说明回退 numpy）、`vectors` > 0、`credentials_missing` 为空。
- 注：若数据目录在含中文的路径下，Milvus 会被拒绝并回退 numpy —— 这是已知平台限制，不是配置错误。

### P1 · 功能缺口

**P1-1　评估集扩充到 30~50 条**
现在 `eval/dataset.example.jsonl` 只有 3 条数据。没有足量标注，Recall/MRR 数字没有说服力，
而这是课题验收要用的。
- 标注方式：**优先 `gold_chunk_ids`**（评价单位就是 chunk，最不容易歧义）；
  图省事可以写 `gold_snippets`，脚本会解析成包含该片段的全部 chunk——但要注意，
  一个片段在语料里出现得越分散，gold 集合越大，Recall 越难满分，这是刻意的保守处理。
- 验收：`python scripts/eval.py --dataset eval/dataset.jsonl --top-k 5` 输出
  dense / sparse / hybrid / rerank 四行对照表；报告末尾应无"数据问题"段落。
- 注意 `rerank` 一栏在没有配 `R4P_RERANK_URL` 时等于 hybrid（NoopReranker）。

**P1-1b　正式评测集与模型版本要一起锁定**
报告里必须记下：embedding 模型名、维度、chunk 参数（`chunk_max_tokens` / `chunk_min_tokens`）、
`recall_k`、`rrf_k`。换了其中任何一个，前后两次数值不可直接比较。

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

### P2 · 可靠性（多人使用前必须处理，单机演示可暂缓）

1. **SQLite 并发**：同一个连接被请求线程与后台入库线程共用，`check_same_thread=False` 但没有写锁。
   短期加锁 + 并发测试；更好的做法是每线程独立连接。
2. **任务不可恢复（部分解决）**：启动时已把遗留的 `running/queued` 任务与停在 `indexing` 的文档都标成 failed，
   但任务本身仍是进程内 `asyncio.create_task`，进程一死就得重跑；要真正可恢复得落持久化任务表 + 重试。
3. **无身份认证（与「多用户 Key」不是一回事）**：上传/删除没有权限校验；前端「API 设置」里的
   用户名只是标签，谁都能填自己的 Key 顶替，也都能取回本地保存的 Key。它解决的是"分摊额度 +
   各自记住 Key"，不是"谁有权访问知识库"。公网部署前必须换成真正的登录 + 服务端密钥托管（加密、可吊销、可审计）。
4. **上传无大小上限**：一次性读入内存，应改流式落盘 + 上限 + 清理策略。
5. ~~模型一致性未强制校验~~ → 已在第三轮解决，见上表。
6. ~~一致性检查只报告不阻断~~ → 已在第四轮解决：`blockers` 覆盖缺向量/孤儿/未标注/后端条数不符，检索与写入统一拒绝。
7. **重建期间极短不可用窗口**：两次 rename 之间集合名短暂不存在。要消掉它得引入"活跃集合指针"（切换即改
   SQLite 里的一行），但会让多进程各自缓存的名字不一致，代价更大；当前单进程假设下不值得。

### 已关闭

**P2-3　~~公开仓库的标识脱敏~~　已关闭（2026-10-23 用户确认无需处理）**
用户明确：「立体聚落」不是真实课题名，是另一个项目的参考资料；`课题/Ref` 只是参考材料目录。
README / tests / eval 里的示例文字与路径按现状保留，**不要**再自行替换成占位符。

---

## 8. 两个待拍板问题的结果（2026-10-23 已答复，无需再问）

1. **仓库保持 public，不做标识脱敏。**
   用户确认：「立体聚落」并非真实课题名，是其他项目的参考资料；`课题/Ref` 仅为参考资料目录。
   已自查确无硬编码凭据（`.env` 未入库，样例只写 `sk-请替换`），可以保持公开。
2. **提交历史已按层拆分**：1 个初始提交 → 17 个提交（骨架 / 领域模型 / 配置 / 存储 / 解析 /
   分块 / 外部适配 / 入库 / 筛选与提示词 / 检索生成 / 服务装配 / API / 前端 / CLI / 评估 / 文档 / 交接）。
   每个带测试的提交都能独立通过 `pytest`，通过数单调递增 2 → 9 → 12 → 19 → 27 → 30。
   拆分前的旧历史保留在本地标签 `pre-split-backup`（指向 `f2a22cd`），需要回滚时
   `git reset --hard pre-split-backup` 再 force push 即可。

---

## 9. 工作约定（接手后请遵守）

- **改完必须跑**：`./.venv/Scripts/python.exe -m pytest -q` 和 `-m ruff check rag4prefa tests scripts`，两者都必须干净；
  动过索引相关代码再跑一次 `python -m rag4prefa verify`。
- **SQL 只写进 `repo.py`**；`api.py` 只做协议转换；外部依赖只进 `adapters/`。
- **新功能必须有离线测试**：`tests/conftest.py` 里有 `FakeEmbedder` 模式可循，不要写依赖付费 API 的测试。
  涉及失败路径的，按 `test_ingest.py` 里的 `HalfWritingStore` / `NoDeleteStore` 那样**注入故障**，
  断言"旧版本还在""必须被拒绝"，而不是只断言"函数返回了"。
- **测 Milvus 只能在纯 ASCII 路径下**（`tmp_path` 就是），中文路径会被 `_reject_non_ascii_uri` 直接拒绝；
  这不是测试写得不对，是平台限制。
- **不要引入本地模型 / torch**（用户明确要求）。
- **不要把密钥写进任何入库文件**，也不要复用原仓库里的 ModelScope token（那个已泄漏在公开仓库）。
- 提交信息用中文、说清"为什么"；不要用 `--no-verify`。**force push 只在用户明确要求重写历史时使用**，
  且必须先打备份标签（本次重写前的备份是 `pre-split-backup`）。
- 推送前 `git status` 确认没有 `data/`、`.venv/`、`.env` 混进去。

---

## 10. 交接时的验证记录（2026-10-24 20:20，第五轮：多用户 Key）

```
pytest -q                          → 134 passed（新增 19 项凭据用例）
ruff check rag4prefa tests scripts → All checks passed
前端脚本                          → 提取 <script> 过 `node --check`，无语法错误
真实服务（uvicorn + curl）：
  GET /api/health（无头）         → user={source: server, label: 服务端公共 Key}
                                    missing=[R4P_EMBEDDING_API_KEY, R4P_LLM_API_KEY]，backend=numpy
  GET /api/health（带三头）       → user={source: user, label: 张三}（%E5%BC%A0%E4%B8%89 解码正确）
                                    missing=[]，说明请求头确实覆盖了公共配置
  GET /                           → 200，页面内含 X-R4P-* 注入逻辑
```

第六轮（2026-10-24 20:45）修的是「报错读不懂」——不是功能缺失，是排查方向被误导：

```
pytest -q                          → 149 passed（新增 5 项根因识别用例）
ruff check rag4prefa tests scripts → All checks passed
用用户提供的真实 Key 走 POST /api/credentials/verify：
  embedding = sk-…（用户填的）
    → AuthenticationError 401 {"code":30014,"message":"Token is invalid."}
    → 同一把 Key 另测 DeepSeek / 百炼 / 月之暗面 / 智谱，四家全部拒绝
    → 硅基流动对任意不被接受的 Token 一律回 30014（连 "abc123" 也是），
      所以这个错误码只能证明「硅基流动不认它」，推不出具体原因
  llm = ms-…（用户填的）
    → 403 Forbidden: For account security, please complete your profile via email verification
    → 追加提示后：明确指出「这不是 Key 的问题：账号还没完成邮箱验证」并给处理入口
实测结论（写进配置建议）：
  魔搭 37 个可调用模型全是对话/图像，**没有 embedding 模型**，
  /v1/embeddings 对 Qwen3-Embedding-*、BAAI/bge-m3 一律 "Invalid model id"。
  → embedding 与 LLM 分属两家时，embedding 只能选硅基流动 / 百炼 / 智谱这类提供向量接口的。
```

**判断一把 Key 是「有效但被账号状态挡住」还是「根本认不出来」，用假 Key 对照**：

```
POST /v1/chat/completions（魔搭）
  真 Key  → 403 Forbidden: complete your profile via email verification
  假 Key  → 401 Authentication failed, please make sure that a valid ModelScope token is supplied.
  无 Key  → 401（同上）
```

两者错误文本不同，就证明这把 Token 被平台成功识别了，卡点在账号状态；错误文本一致则说明
Key 根本没被认出来。**不要用 `GET /v1/models` 判断 Key 有效性——该接口匿名可访问**，
带假 Key、不带 Key 都返回同样的 37 个模型，用它当证据会把无效 Key 判成有效。

值得注意的是 `sk-` 前缀并不专属某一家：硅基流动 / 百炼 / DeepSeek / 月之暗面 / 智谱都发
`sk-`，因此 `platform_mismatch()` 对 `sk-` 只能放行。判断归属只能靠"这把 Key 是在哪个
控制台生成的"，外部无法从形态推断。

三类根因的判定顺序（`api._explain_error`）：账号阻断 → 模型不在目录 → 鉴权失败。
它们互斥且结论相反：账号阻断意味着「Key 已通过鉴权」，此时再补一句「Key 未被接受」
会把人带去重建 Key，所以只补最先命中的那条。

第七轮（2026-10-24 21:15）：两把真实 Key 都打通了，根因是**硅基流动双站**：

```
pytest -q                          → 150 passed（新增双站识别用例）
ruff check rag4prefa tests scripts → All checks passed
POST /api/credentials/verify（国际站 Key + 魔搭 Key）：
  embedding → OK  model=Qwen/Qwen3-Embedding-0.6B  dim=1024
  llm       → OK  model=Qwen/Qwen3.5-35B-A3B       reply_chars=2
```

用户那把 `sk-` 是**国际站**（cloud.siliconflow.com）申请的，项目默认配的却是国内站
`api.siliconflow.cn`。同一把 Key 打国际站返回 `20012 Model does not exist`（模型名不对），
打国内站返回 `30014 Token invalid`。**两个错误码的差异本身就指明了问题在站点而不在 Key。**

向量模型选型（同平台，5 句中文探针，单条调用）：

| 模型 | 维度 | 近义 | 同域 | 弱相关 | 无关 | 区分度 |
|---|---|---|---|---|---|---|
| Qwen3-Embedding-0.6B | 1024 | 0.939 | 0.615 | 0.329 | 0.163 | **0.775** |
| Qwen3-Embedding-4B | 2560 | 0.917 | 0.608 | 0.377 | 0.326 | 0.591 |
| Qwen3-Embedding-8B | 4096 | 0.917 | 0.614 | 0.446 | 0.273 | 0.644 |

本项目的 chunk 上限 480 token，短文本检索上 0.6B 的分离度并不输大模型，而存储与检索
成本分别是 1 / 2.5 / 4 倍，故默认选 0.6B。样本只有 5 句，不足以断言「小模型更强」，
但足以说明「更大 ≠ 对本任务更好」；真要换 4B/8B，必须全量 reindex。

第八轮（2026-10-24 21:55）：对话模型改为用户可选，embedding 仍锁死。

```
pytest -q                    → 158 passed（新增 8 项模型选用例）
ruff / node --check          → 干净
GET /api/llm-models          → 4 项，服务端当前模型带 selected 标记
POST /api/credentials/verify：
  Qwen/Qwen3.8-Flash-Next    → OK  实际调用即所选模型，1.3s
  Qwen/Qwen3.5-122B-A10B     → OK  embedding 仍是 1024 维（未被波及）
  nonexistent/model          → 回落 Qwen/Qwen3.5-35B-A3B，不报错
```

候选淘汰记录（省得下次重试）：

| 模型 | 现象 | 处理 |
|---|---|---|
| ZhipuAI/GLM-5.2 | 头几次能调通，随后 `429 exceeded your current quota` | 排除：吃付费配额 |
| ZhipuAI/GLM-4.7-Flash | 持续 `429 访问量过大` | 排除：免费池排队 |
| DeepSeek-V4-Flash/Pro/4.1 | `has no provider supported` | 排除：目录挂了但无 provider |
| MiniMax-M3 / Step-3.5-Flash / LongCat | 同上 | 排除 |
| Qwen3.5-35B / 3.8-Flash-Next / 3.8-27B / 3.5-122B | 全部可用，1.3~3.1s | 收入清单 |

`fingerprint` 必须把 `llm_model` 算进去：只按 Key 取缓存的话，同名用户换模型会拿到
池子里的旧 LLM 实例——界面显示已切换、实际还在跑旧模型，而且不会报任何错。

第九轮（2026-10-24 22:00）：补失败路径。基线 `8fa40bf`，四项都只改行为、不加功能。

```
pytest -q                    → 165 passed（新增 7 项失败路径 / 回退来源用例）
ruff check                   → All checks passed
node --check（前端脚本）      → 通过
```

1. **reindex 的跨组件窗口。** 归档是 SQLite、索引是向量库，两者不在一个事务里；中断时
   条数/模型名/维度可以完全一致，靠比对根本发现不了。新增 `meta.index_state`
   （`rebuilding:*` / `rebuilding:<项目>` / `ready`），写归档前置位、成功后与模型维度
   同事务改回 `ready`；非 `ready` 时检索与增量写入一律拒绝。`verify` 命令也会因此报
   `ok=false`，并在 id 全对的情况下去掉"哪里少了几条"这种误导性指引。
   恢复有两条路：`reindex`（重算向量、花钱）和新增的 `rebuild` 子命令（用归档里现成的
   向量，不花钱）。局部重建只能消除同项目的标记，遇到全量标记会明确拒绝。
2. **入库部分写入的撤回清单。** `planned` 改成在 `store.upsert()` **之前**算出来。
   原来是等 upsert 正常返回后才统计"本次新增了哪些 id"，而它抛异常时不会返回——
   撤回清单永远是空的，逻辑等于不存在。新增用例让故障前确实写入新 id，并断言撤回后
   归档 id 集合逐项不变。
3. **凭据校验的回退来源。** `/api/credentials/verify` 原来用 `svc(request)`（当前用户
   视图）当空字段的回退值：A 已登录、给 B 只填 embedding 时，验的其实是 A 的 Key，
   却标成 `source=server`。改为取根服务的公共配置。回归测试在请求头放一把可用用户
   Key、请求体留空，断言真正发出的是 `sk-server`。
4. **"只用公共额度换模型"在前端做不了。** `saveCreds()` 曾强制至少填一个 Key，与后端
   已支持"只带模型头"的能力不一致。放开后补了一条端到端用例：无 Key + 只带
   `X-R4P-LLM-Model` 走 `/api/chat`，断言实际调用的是选中的模型、用的是服务端 Key。

教训：**"失败后旧版本完整可用"这类承诺，必须在故障注入下逐条验证，不能靠成功路径推断。**
本轮四个问题全都是"代码执行了预期操作、但业务不变量没成立"，正常路径的测试一个都抓不到。

真实链路冒烟（真实 Milvus Lite + 真实厂商，数据目录 `C:/tmp_r4p_smoke9` 为纯 ASCII 路径）：

```
入库 3 块                        → chunks=3, 索引=归档=3, index_state=ready
把 store.rebuild 打成故障后 reindex → 索引=归档=3（**条数完全相同**）
                                    index_state=rebuilding:* → 检索被拒绝
rebuild（用归档里的现成向量）      → 3 条, index_state=ready, 检索恢复
verify                           → ok=true，三类差异全为 0
真实检索                          → 3 条命中，首条「第1章 结构方案」
真实 LLM（魔搭 Qwen3.5-35B-A3B）  → 89 字，答案带 [1] 引用
```

这次冒烟的价值在于**证明了"条数相同"确实发现不了问题**：中断之后索引与归档的条数、
模型名、维度一模一样，`verify` 的三类 id 差异也全是 0——唯一的证据就是那条标记。
numpy 后端跑不出这个场景（索引与归档是同一张表），所以这条验证必须在 Milvus 上做。


第四轮（2026-10-23 22:10）的冒烟记录仍然有效：

```
真实链路冒烟（真实申报书 + 真实 Milvus Lite + 真实 openai SDK，假服务兜住网络，零费用；
            Milvus 段跑在 C:/tmp_r4p_smoke4，因为中文路径下 faiss 无法建索引）：
  0 中文数据目录                   → MilvusStore 直接拒绝并说明两条出路
  1 入库                          → 30 章节 / 26 块，chunks=archived=indexed=26，id 无重复
  2 verify 三方核对                → ok=true
  3 检索                          → embed 5ms / dense 4ms / sparse 1ms
  4 同维度全量重建换模型          → 走 __staging → rename，集合列表只剩 chunks_vec
  5 换维度重建(32→8)              → 同样走暂存切换，无残留集合
  6 重启后重开 Store              → count 不变，notes 为空
  7 注入残缺（删一条归档向量）    → health consistent=false、missing=1；检索被拒绝
  8 verify 定位 → reindex 修复    → 差异定位到具体方向（片段缺向量 1、索引多 1），修复后 ok=true
  9 同一文件改末尾一段            → 26 块保留 25 个 id，只新增 1 个（内容寻址生效）
全新目录 + 照抄 .env.example 首启：
  doctor                          → 不崩，回退 numpy 并写明"非 ASCII 路径"，如实报 missing 两个 key
```

环境实况：Python 3.13.14；fastapi 0.161.1 / uvicorn 0.54.0 / python-docx 1.2.0 /
pymilvus 3.0.2 / milvus-lite 3.2.2 / openai 3.37.0 / jieba 0.42.1 / numpy 2.5.4 /
pytest 9.1.1 / ruff 0.17.2 / pydantic-settings 2.16.1；**torch 未安装（符合约定）**。

> 注意 `sparse_ms`（BM25）首次查询约 700~800ms，是 jieba 词典首次加载的开销，
> 之后降到毫秒级。若在意首问延迟，可在启动时预热一次 `jieba.lcut_for_search("预热")`。

> 注：`create_app()` 的 services 在 lifespan 里挂到 `app.state`，
> 所以 `TestClient` 必须用 `with TestClient(app) as c:` 才会跑 startup，
> 否则 `/api/health` 会因 `State` 缺属性报 500。uvicorn 正常启动不受影响，写测试时注意这点。

---

## 11. 深入阅读顺序

1. `REFACTOR_PLAN.md`（本目录）—— 重构方案 + **第十二节有实施复盘**，记录计划与现实的 5 处偏差，值得先看。
2. `F:\project\PyProject\milvus\PROJECT_REVIEW.md` —— 原系统评估报告，说明"为什么要重写"的每一条证据都能定位到行号。
3. `README.md`（本目录）—— 面向使用者的安装、API、已知局限。
4. 原系统源码 `F:\project\PyProject\milvus\Vector_search_plus.py`（1 991 行）—— 只在需要对齐某个行为时查阅。
