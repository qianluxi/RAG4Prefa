# 交接说明 · RAG4Prefa

> 写给接手的 agent：你**没有**上一轮对话的上下文，这份文件是唯一入口。
> 读完这一份就能开始干活；需要更深的背景再看第 11 节列的文档。

**交接时间**：2026-10-26 15:20
**交接时状态**：可运行、**244 项离线测试全绿**；真实厂商 API 已端到端验证
（embedding 走硅基流动国际站 `Qwen/Qwen3-Embedding-0.6B`、LLM 走魔搭 `Qwen/Qwen3.5-35B-A3B`，
`POST /api/credentials/verify` 两项均通，`dim=1024` 与服务端配置一致）。
第十三轮补齐了三条**契约**：读写版本边界（在途查询不跨索引切换）、批次原子写入
（一批向量全进或全不进）、跨进程状态识别（代次 + 工作租约，不再靠猜数据库状态）。
第十四轮补的是四个**边界**：恢复动作不能绕过锁、快照不能把旧行认成新版本、
活动后端要落库并在锁内复核、失败回滚要还原被覆盖的行。
第十五轮把两条**状态转换**做成可证明的协议：打开 ≠ 确认（先证明内容再登记活动后端）、
无锁打开只加载不创建；同时给"归档改了、索引没跟上"加了独立标记。
第十六轮把"证明"从**身份**推进到**内容**：换后端一律从归档重写（id 相同不算证据）、
读路径也要核对"手里这个后端是不是登记的那个"、补偿失败必须落盘阻断检索、
读不到后端条数不再当成"确认为空"。
第十七轮把"**未确认**"从一句进程内结论升级成贯穿所有入口的准入：库里已有内容却没有
登记过活动后端时，查询（API + CLI + 评估）、普通入库、删项目一律在改任何数据之前拒绝；
同时补上"同名不等于同一个索引"（类型 + 地址 + 集合名一起登记为身份）与"名称相同不等于
对象已就绪"（快速启动要真的 `probe()` 一次），后端未就绪统一 503 而不是 500。
索引一致性这一阶段至此收尾（第十七轮复审通过）。第十八轮转入**下一阶段：能读的格式与
可证明的效果**——先补 PDF 通道（`parsers/pdf.py`，pdf 没有样式表，只能靠字号/加粗/行距
反推章节树），下一步是语料扩容与评估集扩充。
动手前请先读第 3 节的「索引不变量」，那三十一条是这批代码存在的理由。

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
| 当前提交 | **`3890610`**（"评估集在新语料上重建：删掉软件手册后，sparse 不再压过 hybrid"，第二十三轮），已推送 `origin/main`，本地与远端一致；往后以 `git log -1 --oneline` 为准 |
| 上一轮基线 | `959a106`（真机）→ `4d79d4c`（目录导入/删项目）→ `7051fb5`（Key 回落开关）→ `3890610`（评估集重建） |
| CI | `.github/workflows/ci.yml`：`3890610`、`7051fb5`、`4d79d4c` 上 Actions 均全绿（ubuntu-latest / Python 3.13，安装依赖 → ruff → pytest 287 项） |
| Python | 3.13，**虚拟环境已建好**：`F:\project\课题\RAG4Prefa\.venv` |
| 解释器路径 | `./.venv/Scripts/python.exe`（Windows；所有命令都在项目根目录执行） |
| 依赖状态 | 已装齐（fastapi / uvicorn / python-docx / pymilvus 3.0.2 / milvus-lite 3.2.2 / openai / jieba / numpy / pytest / ruff） |
| 未安装 | `torch`、本地 embedding 模型（**按用户明确要求不装本地模型**） |

原版项目（只读参考，**不要改它**）：`F:\project\PyProject\milvus`，已检出 v2.8.6。
真实测试语料：`F:\project\课题\Ref\立体聚落异型大悬挑百年建造高质量建造技术研究-申请书26.10.23.docx`

**先跑这四条确认环境可用**：

```bash
cd "F:/project/课题/RAG4Prefa"
./.venv/Scripts/python.exe -m pytest -q          # 期望：230 passed
./.venv/Scripts/python.exe -m ruff check rag4prefa tests scripts   # 期望：All checks passed
./.venv/Scripts/python.exe -m rag4prefa doctor   # 看后端、维度、缺哪些凭据
./.venv/Scripts/python.exe -m rag4prefa verify   # 逐 id 核对片段/归档/索引（空库也返回 0）
```

---

## 3. 代码地图（7 578 行 Python + 952 行单页 HTML + 6 505 行测试，均按非空行计）

行数为**非空行**（`grep -cv '^[[:space:]]*$'`），这样口径唯一、可以随时重算核对。

```
rag4prefa/
├── config.py      79    pydantic-settings，全部配置以 R4P_ 前缀从 .env 读；含 `allow_server_key_fallback` 公网开关
├── domain.py      98    Chunk / Section / Hit / Filter / JobState，纯数据，无依赖
├── db.py         117    SQLite schema（projects / documents / chunks / chunks_fts / vectors / jobs / meta / embed_cache）
├── repo.py       851    **全部 SQL 都在这里**，没有别处写 SQL；含索引代次（index_generation）
├── consistency.py 427   **索引一致性**：状态诊断、阻断判定、逐 id 核对（verify）、切换标记
├── concurrency.py 291   **写入协调**：维护锁、写事务串行、工作租约（判断别的进程还活着吗）
├── ingest.py     845    解析 → 分块 → 向量化 → 写索引 → 切事实；含 rebuild_store / reindex；入库批次整体占锁
├── chunking.py    93    按标题树切块；chunk_id 内容寻址（同 id 必同正文）
├── tokens.py      40    中英混排的 token 估算（非真实 tokenizer）
├── ask.py        216    混合召回 → RRF 融合 → 可选重排 → 流式生成；含索引代次确认
├── filters.py     88    自然语言问题 → Filter（规则法，可单测，无 LLM 调用）
├── prompts.py     28    系统提示词与引用格式
├── admin.py       84    项目/文档的查询与级联删除
├── services.py   564    组件装配；Milvus 不可用时回退 numpy；`for_credentials()` 出按用户视图
├── credentials.py 304   **按用户解析 Key 与对话模型**：请求头 → 凭据池；报错根因分级
├── api.py        557    FastAPI 路由（只做协议转换，不含业务逻辑）；`svc_billable()` 拦匿名额度
├── cli.py        369    init / serve / doctor / ingest / reindex / rebuild / ask / projects / delete / verify；目录导入逐文件报被跳过项
└── adapters/    1721    embed / llm / vectors / rerank / parsers（docx + pdf + 共用的标题规则）—— 外部依赖都关在这里
```

`scripts/` 另有 514 行（`eval.py` 419 + `relabel_eval.py` 195），合计非空 7 309 行——`rag4prefa/` 本身是 6 795 行。

解析层三个文件的分工，改动前先看清：

- `parsers/headings.py`：**与格式无关**的判据——正文清洗、中文编号正则、`build_sections`
  （把 `(级别, 文本)` 流折成章节树）。两个解析器共用，否则同一份文档换个格式就换一棵章节树，
  内容寻址的 `chunk_id` 全部对不上。
- `parsers/docx.py`：Word 有结构，判级顺序是「大纲级别 → 标题样式 → 编号正则」。
- `parsers/pdf.py`：pdf 没有结构，只有字号/加粗/坐标，走「统计正文字号 → 按字号与加粗判标题
  → 编号定级」，另含页眉页脚剔除、跨块段落拼接、表格按 `|` 保留。
- `parsers/__init__.py`：`parse_document(path)` 按扩展名分派，未知格式明确报错。

前端只有一个文件 `rag4prefa/web/index.html`（741 行，原生 JS，无构建步骤）。
`tests/` 277 项，**全部离线**（FakeEmbedder 确定性假向量 + 本地假 OpenAI 服务 + 临时 Milvus/numpy），
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

11. **"恢复完成"必须可验证，不能只是"函数返回了"。** `rebuild_store` 的完成条件有四条，
    缺一条都不算恢复：① 归档能当恢复源（有正文就有归档向量、每行都带模型名、全库一套
    模型与维度、**且每行向量的实际字节长度 = 声明维度 × 4**）；② 后端实际条数 = 归档条数；
    ③ 归档的模型与维度写回 `meta` **并且**标记回 `ready`（同一个事务）；④ 孤儿归档行在
    ①②之后才清理。
    为什么第 ③ 条不能省：换模型重建中断时，归档已经是新模型、元数据还停在旧模型。
    只恢复索引和标记，一致性检查会继续以"元数据模型 ≠ 归档模型"拒绝检索——命令报成功、
    系统查不了，是最容易把人带偏的一种"假恢复"。
    为什么第 ① 条要查字节长度：`dim` 是声明，`vec` 的 float32 字节数才是事实。只比声明时，
    "每行都写 1024 维、实际只有 768 个浮点"的归档能通过全部检查，恢复出来的索引与 meta
    对不上；numpy 后端更会直接崩在 `vstack` 的一句 "dimensions must match" 上。
    推论一：**空范围的重建不得清除与自己无关的标记**。`reindex --project 某项目` 在没有
    片段时只返回 0 条，不动标记；只有"真的全空（片段、归档、索引都是空的）"才允许置
    `ready`，否则等于替一笔没走完的全量重建宣布成功。
    推论二：**"空库"不能抵消未完成标记**。`IndexState.blockers` 先判 `pending` 再判
    `is_empty`：空只说明没有可比对的数据，证明不了那笔重建走完了。反过来先放行写入，
    新内容一落地又会撞上标记，等于把数据收进来再拒绝检索。

12. **自动迁移到 Milvus 是可选升级，失败必须原地降级。** 启动时因维度未知回退到 numpy，
    首次入库拿到维度后 `maybe_promote()` 试图迁回 Milvus。这条路径挂在入库成功之后
    （`api.py` / `cli.py` 都紧跟 `ingest_batch`），所以它**任何异常都不许外溢**——
    api 那个 `try` 块会把任务状态改写成 `failed`，而数据其实已经写好了。
    两条实现约束：
    - **目标后端重建成功之前，归档一个字节都不许动**。numpy 的内存索引就是 `vectors`
      表的缓存，先删孤儿行再失败会让它的条数与归档脱节，一次可选迁移把本来能用的后端
      弄成不可检索（第十一轮修正，旧实现正是先删后建）；
    - 捕获范围是 `Exception`，不是 `ReindexError`/`RebuildError` 白名单。pymilvus 在
      造集合、写向量、加载集合、文件被占用时抛的是它自己的异常，一样只该降级。

13. **改事实或改索引的动作必须互斥。** `ingest` / `reindex` / `rebuild` / 后端迁移 /
    删项目共用一把**维护锁**（`concurrency.maintenance_lock`）。粒度是"一次动作"，
    不是一条 SQL：
    - `reindex` **从读片段快照到写完元数据全程独占**，中间那段 embedding 跑几分钟也算。
      只锁写入的话，算完才发现事实表已经变了——那时条数可能仍然一样，只有 chunk_id
      不一样，条数校验一条都发现不了；
    - 跨进程：`uvicorn` 与 `python -m rag4prefa reindex` 是两个进程，所以用**操作系统
      文件锁**（Windows `msvcrt.locking` / POSIX `flock`）。进程被杀时内核自动释放，
      不会留下需要人工清理的锁文件；占锁者写成旁边的 `maintenance.lock.holder`，
      拒绝时能说出"在等哪个动作、pid 多少、从几点开始"；
    - 抢不到锁有两种语义，调用方按场景选：`timeout=0` 立刻抛 `MaintenanceBusy`
      （API → 503/409，用户看到"稍后重试"而不是一个迟到的 failed），`timeout=None`
      排队（CLI 维护命令）；
    - 同一线程可重入（`ingest_batch`→`ingest_file`、`maybe_promote`→`rebuild_store`
      都是嵌套的）。重入深度必须记在**线程**上，记在锁对象实例上等于没有可重入——
      这条真的把线程锁死过一次。
    - 另外，**写事务本身也要串行**（`db.transaction` 与 `NumpyStore` 直写共用一把
      `RLock`）：整进程共用一条 SQLite 连接，`check_same_thread=False` 只让它"能用"。
      读不加锁——共享连接上的读可能看到一次尚未提交的中间态，但写入是串行的，
      最坏是读到一份马上要回滚的数据，不会写坏；要做到读隔离得改成每工作单元一条
      连接，属于后续阶段。

14. **恢复验收看 id 集合，条数只是必要条件。** `rebuild_store` 与全量 `reindex` 在
    宣布 `ready` 之前都要 `store.ids()` == 目标 id 集合（`_assert_backend_matches`）。
    两个版本的条数可以一模一样、id 完全不同（重建期间有并发写入时就是这样），那种库
    能检索、`health` 也看不出异常。`verify_ids` 早就在做这件事，但它不在主路径上——
    恢复动作的判据不能比日常校验还弱。

15. **后端指针归根服务，坏归档不许阻断启动。**
    - `for_credentials()` 生成的是**请求级视图**，它只是快照；`store` / `fell_back`
      只能由 `services.owner`（根服务）改。在视图上迁移会出现"Milvus 建好了、meta 也
      更新了、根服务还在查 numpy"的成功假象，而且根服务 `fell_back` 仍是 True，
      下个请求再迁一遍。
    - `NumpyStore` 的构造阶段**不因归档损坏而抛错**：它在 `build_services()` 里就被
      创建，构造即失败等于归档一坏服务起不来、`reindex` 也进不去——而那条命令正是
      唯一修复入口（按正文重算、整表重写，不需要读旧索引）。坏行只跳过并记 notes，
      阻断交给 `consistency`（`IndexState.malformed`）。

16. **在途查询不得跨过一次索引切换。** 检索路径用**代次**（`meta.index_generation`）
    做一次 seqlock：取号 → 一致性检查 → 算查询向量 → **确认代次没变** → 稠密检索 →
    **再确认一次**。切换会在两处推代次（置 `rebuilding:*` 时、写完 `meta` 时），
    所以任何一次跨越都能被发现。
    - 不能靠维护锁：一次重建要几分钟，让所有提问排队等它结束是不可接受的；
    - 也不能只在入口确认：`ensure_queryable` 与 `search` 之间隔着一次网络往返
      （算查询向量），切换正好落在那段时间里时，旧模型的向量会去查新索引。
      维度相同、后端是 Milvus 时，这种检索**不报错**，只是语义错位。
    - 配套：`data_revision` 是另一个计数器，只随"内容"变（入库、删项目）。它**不**
      否决在途查询——否则"有人上传文档时别人就问不了问题"，比原问题更烦人。

17. **一批向量要么全进、要么全不进。** `NumpyStore.upsert` 先整批校验，再一次性
    `with sqlite_write(), conn:` 提交。逐条 `INSERT OR REPLACE`、校验放在循环里的写法
    会留下"事务开着、第一条已写"的状态：随后**任何**一次 `commit()`（比如另一个线程
    更新任务进度）都会把它提交；而入库的回滚逻辑只删"本次新增的 id"，
    删不掉被覆盖的旧行——"失败时旧版本仍完整可用"这句话就不成立了。

18. **"谁还活着"要有客观依据，不能看数据库状态。** `build_services()` 只在**没有**
    活着的服务进程时才把 `running`/`queued` 任务标成中断，判据是
    `concurrency.WORKER_LEASE`：服务进程整个生命周期占着的文件锁，进程退出由内核释放。
    库里只有 `state='running'` 一个事实，分不出"上一个进程死了"和"另一个进程正在写"；
    以前每次装配都无条件清理，于是跑一次 `doctor` 就能把在线入库标成 failed。
    同一条规则的两个推论：
    - 项目与任务状态必须在**维护锁内**创建（`ingest_batch` 里做）。锁外预创建，
      一旦被别的维护动作挡住就是一条假记录（空项目 + failed 任务）；
    - 批次用的后端要用 `StoreRef` 在**用的时候**解析，不能把 `root.store` 在批次
      开始前定死：迁移换了后端之后，旧对象还在写，而根服务已经在查新后端了。

19. **崩溃恢复会删集合，所以它必须在维护锁里。** `_recover()` 看到"正式集合 +
    `__staging`"并存时，判定暂存是上次留下的半成品直接删掉——而**正在重建**的进程
    恰恰把半成品放在那里。构造函数是每个进程启动都走的路、又发生在任何锁之外，
    它删一次，那笔重建就白跑；事后条数、模型、维度全都对得上，查不出是谁动的手。
    现在 `__init__` 只连客户端，`_open()` 分三种情况：
    - 干净（只有正式集合）→ 无锁打开，只做"已存在则跳过"的建表与 load，不删任何东西。
      这条路径必须无锁，否则启动会被别人的入库挡住几分钟；
    - 全新库（三个集合都没有）→ 排队建集合（秒级，且没有别人的半成品可破坏）；
    - 有残留 → `timeout=0` 抢锁，**抢不到就 `_defer()`**：宁可这次查不了，也不能替
      另一个进程决定它的暂存集合是不是垃圾。`_defer` 后所有对外方法抛
      `BackendUnavailable`，`health` 会报 `store_error` 而不是假装索引是空的。
    - **无锁路径不许创建任何集合**（第十五轮补）。`_open()` 看到"正式集合在、没有残留"
      与真正 load 之间隔着一段时间，另一个进程可能正好在这段时间里做完两次 rename。
      旧实现这时调 `_ensure_collection`，发现正式集合不存在就**建了一个空的**：对方第二次
      rename 撞名失败，而它的回滚条件（`__retired` 在、正式集合不在）已经不成立，
      旧索引被永久留在 `__retired` 里。现在 `_open_clean()` 只加载"此刻仍然存在"的正式
      集合，发现它消失了就返回 False、重看布局（`_OPEN_ATTEMPTS` 次）后改走持锁路径。
      **没有锁，就没有资格创建、删除、重命名任何集合**——这条要当规则记，不是当优化记。

20. **内存快照必须与它记下的代次同代。** `_load` 是 seqlock 式双读：
    取代次 → 读行 → **组快照** → 再取代次，两次相同才发布。原来的"读行 → 组矩阵 →
    读代次"允许"旧行 + 新代次"，此后 `_refresh_if_stale` 认为缓存就是最新，**永久**
    少那几条。组快照也必须放在比较之前，否则等于承认"比较完之后再变就不算变"。
    连续 `_SNAPSHOT_ATTEMPTS` 次不一致就抛 `SnapshotUnstable`（继承 `IndexMismatch`），
    绝不发布一份没证明过的快照；构造阶段则留空快照 + **不记代次**（停在 `(-1,-1)`，
    必然与真实值不同），下一次读会再试。

21. **活动后端是库的属性，不是进程的偏好。** `meta.vector_backend` 记录被确认可用的
    后端（`VECTOR_BACKEND_KEY`）。`auto` 在库已启用 Milvus 后**不再静默回退 numpy**：
    回退本身不报错，但本进程的写入只更新"归档 + numpy 索引"，在线服务查的 Milvus
    停在旧版，而两边条数一致时连交叉校验都发现不了。
    装配（`open_store`）发生在拿锁**之前**，所以维护动作在**锁内**还要
    `consistency.ensure_backend_current`：库里记的与手里拿的不是同一个就拒绝，
    且必须排在项目/任务写入之前（被拦住时一个字节都没改）。CLI 传 `StoreRef` 只解决
    "本进程指针过期"，跨进程那一层靠这条。**所有改数据的动作都要过这一关**——
    入库、重建、局部重建、删项目一个不能漏（删项目漏过一次，第十五轮补上）。

22. **"能打开"和"是当前版本"是两件事，中间必须有一道证明。**（第十五轮）
    三个动作分开，混在一起就出过事：
    - **打开**（`open_store`）：只证明这个后端能用。**不写 `meta.vector_backend`**；
    - **确认**（`Services.settle_backend`，启动时跑）：逐 id 核对后端内容与归档。
      一致才登记；不一致就从归档重建（不重算 embedding），重建并校验通过后再登记；
      证明不了就**不登记**，并说清怎么修（`rebuild`）；
    - **换后端**（`rebuild_store`，唯一入口）：从归档整体重建到目标后端，`_assert_backend_matches`
      过了才把后端名与 `ready` 一起写进 `meta`（`set_index_meta(backend=...)` 一个事务）。
    为什么不能省掉中间这道：遗留的 Milvus 集合可以与归档**条数完全相同、chunk_id 全不同**
    （重建中断、换过一批文档），"能打开"于是被当成"就是它"，后续写入落进内容不对的索引，
    而日常校验只比计数。
    两个推论：
    - `auto` 必须**尊重库里已经定下的选择**——库记着 numpy 就打开 numpy。否则每次重启
      都会"看到 Milvus 能用就切回去"，把上一个人显式做的决定悄悄推翻；
    - "显式改成 numpy 再 rebuild"这条路必须真的走得通。以前的顺序是 `ensure_backend_current`
      拦住一切维护动作，包括唯一能改后端的 `rebuild`——文档写的切换步骤在那种状态下
      根本执行不了。

23. **"归档改了、索引没跟上"要有自己的标记。** 改年份不动向量、只覆盖过滤列，于是
    `documents` / `vectors` 归档 / 后端标量列三处必须一起对齐。路径是：置
    `unsynced:<doc_id>` → 改文档与归档（同事务）→ 覆盖后端 → 撤成**上一次的状态**
    （不是无条件 `ready`，否则会顺手擦掉别人没走完的重建）。后端那一步失败时标记留着，
    检索继续拒绝——只返回一句 `failed` 是不够的：任务失败了，检索还在拿旧过滤列回答，
    而这比报错难发现得多。`rebuild` 一条命令可以清除 `rebuilding:*` 与 `unsynced:*`
    两类标记（归档里已经是新值，覆盖一遍即可）。

24. **失败回滚要还原被覆盖的行，不只是删新增的。** `upsert` 是整行覆盖：新旧版本
    共用的 `chunk_id` 上，`doc_year` / `project_code` 已经被写成这一版的值。
    事实事务失败后只删"新增"的 id，那些行就停在"新版过滤元数据 + 旧版正文"——
    条数一条不差、模型维度也对，按年份过滤却错一年。
    所以写之前先留一份旧行（`repo.archived_vectors_by_ids`），失败时
    **先写回旧行、再删新增**；两步都不能省（只写回会留下无正文的向量占召回名额）。

25. **内容相同不等于状态相同。** 入库跳过条件是"`indexed` + 同 `content_sha`"，
    漏了 `doc_year`（当时 `DocumentInfo` 里根本没有这个字段）。结果是用户改年份被
    静默跳过，界面还显示"已存在"。现在年份变化走一条**只改元数据**的路径：
    `documents` + `vectors` 归档同事务改（`repo.set_document_year`，顺带推
    `data_revision`），再按归档整行覆盖后端过滤列；`FileOutcome.note` 说明
    "内容未变，只把年份更新为 N"，不会被误读成重新入库。改之前先验证归档是否完整，
    不完整就拒绝（改到一半比不改更乱）。

26. **换后端要证明的是内容，不是身份。**（第十六轮）
    `settle_backend` 在"登记值不是手里这个后端"时**一律从归档重写一遍**
    （`rebuild_store`），逐 id 校验通过后由它把后端名与 `ready` 写进同一个事务。
    只比 id 集合是上一轮的做法，它证明的是"同一批片段"：Milvus 里还有向量值、
    `doc_year`、`project_code`，从 Milvus 切到 numpy、在 numpy 上重建过或改过年份之后，
    旧集合的 id 可以一个不差、内容整整差一代。**同一后端且已登记**才走快速启动，
    否则启动时间会随库变大。
    归档为空时同样不许 fail-open：`count()` 抛异常是"读不到"，**不是零**
    （以前 `except: leftover = 0` 会把 SDK 超时读成"确认为空"，接着登记）。

27. **查询也要证明手里这个后端是登记的那个。**（第十六轮）
    `ask.retrieve` 在准入阶段调 `consistency.ensure_active_backend`；服务层另有
    `Services.backend_confirmed`（三态：`None`=没跑过确认、`True`=确认过、
    `False`=试过但没证明），`False` 时 `/api/chat` 与 `/api/search` 直接 503。
    为什么代次不够：它只能发现"这次查询跨越了切换"。一个长跑服务装配时持有 Milvus，
    之后别的进程把活动后端换成 numpy——本次查询的起点已经在切换之后，代次从头到尾
    没变，两次确认都通过，而它手里的索引早就是另一份了。**这类失败不报错，只返回旧内容。**

28. **补偿失败必须落盘，不能只写进错误信息。**（第十六轮）
    入库的 `upsert` 抛异常不等于批次没生效；"写回旧行"这一步再失败，索引里就是
    一半旧行、一半新行——id 集合、条数、模型、维度**全都可以一致**。所以
    `_restore_vectors` 失败时置 `unsynced:<doc_id>`；删项目时后端删除失败同理
    （`unsynced:project:<名字>`）。**回滚成功则不置标记**：网络抖一下导致的失败本来
    是干净的，没必要为此让整个库停服到人工介入。

29. **"未确认"是库的状态，不是某个进程的私事。**（第十七轮）
    准入必须由**同一条判据**覆盖所有入口，不能只在 API 路由上加 `if`：
    - 库的层面（`consistency.ensure_registered`）：**有内容**却**从没登记过活动后端**
      → 拒绝。这条判据的含义要说准：登记是"有人逐 id 校验过"的唯一记录，
      `settle_backend` 只在证明不了时什么都不写，所以"有内容 + 没登记"= 确认动作跑过并失败了，
      不是"首次部署还没配"（首次部署时库是空的）。空库照旧放行。
    - 进程的层面（`Services.backend_confirmed is False`）：本进程试过、没证明成功。
      两条缺一不可——只有库那条挡不住"登记着 milvus、本进程 defer 了"；
      只有进程那条挡不住"CLI/上传入口压根没跑确认流程"。
    落到入口：查询走 `assert_query_ready`（`/api/chat`、`/api/search`、`cli ask`、`scripts/eval`），
    普通写入走 `assert_write_ready`（`/api/projects/{}/documents`、`DELETE /api/projects/{}`、
    `cli ingest`），用例层再各有一道（`_ingest_batch_locked`、`admin.delete_project`）。
    **诊断与修复动作保留入口**：`doctor` / `verify` / `rebuild` / 全量 `reindex` 不调用它们，
    否则修不了。全量 `reindex` 成功之后必须**自己登记**（类型 + 身份）——只写模型/维度/状态
    会得到"命令报成功、检索仍被拒"的假修复；局部 `reindex` 没有这个资格，在没有登记的库上
    直接拒绝并指向全量。

30. **同名不等于同一个索引：类型 + 存储位置才是身份。**（第十七轮）
    `meta.vector_backend` 只记类型（`numpy` / `milvus`）。两个进程共用一份 SQLite、却各自
    配了不同的 `R4P_MILVUS_URI` / `R4P_MILVUS_COLLECTION` 时，`store.name` 都是 "milvus"，
    所有按类型名的检查都会放行，而它们写的根本不是同一个集合。
    现在 `meta.vector_backend_id` 与类型**成对登记**：Milvus 是
    `milvus|<规范化 uri>|<collection>`（URI 归一：URL 去尾斜杠、本地路径取 `abspath`），
    numpy 固定为 `numpy`（它的"位置"就是这份 SQLite，没有第二种可能）。
    `ensure_backend_current` / `ensure_active_backend` / `settle_backend` 都要比对；
    历史库没有这一栏时**就地补记**（`repo.remember_backend_identity`，比较后再写，
    不覆盖别人刚换的后端），不为此重建一遍。换后端（`rebuild`）会连身份一起改写。

31. **"名字对得上"不等于"对象已就绪"。**（第十七轮）
    启动的快速路径（登记的后端名 == 手里的后端名）以前直接 `backend_confirmed = True`。
    但 `MilvusStore._open()` 抢不到维护锁时会 `_defer()`，留下一个 `_ready=False` 的对象，
    它的 `name` 照样是 "milvus"：`/api/health` 报正常，查询要等到 `search()` 里才发现不可用，
    而那里只处理 `IndexMismatch`，于是变成 500。
    现在快速路径也必须有证据（`Services._prove_current` → `store.probe()`：Milvus 真的读一次
    条数，numpy 没有"打开"这一步），证明不了就是 `False` + note，`health` 报不健康。
    配套：`BackendUnavailable` 在 API 层有独立处理器 → **503**（"现在不行"，不是服务器故障），
    SSE 流里则以 `error` 事件收尾。

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
10. **"重排生效了"这件事必须用候选池深度来证明，不能看调用记录。** 两种写法都会留下
    完整的调用日志、分数也确实变了，但召回一点不变好：①先截断到 top_k 再重排
    （重排只在一小撮里挑）；②给重排的融合池与对照组的融合池深度不同（两行数字的差异
    里混进了池深的影响）。判据只有一个——**重排拿到的候选数 == 对照组的候选数，且等于
    截断前的全池**。第二十二轮两处都踩到了，见下节。
11. **`Settings` 会读本机 `.env`，测试里没显式钉住的字段会跟着开发机变。** 本轮把重排打开后，
    一条"未配置重排时应提示未启用"的断言在本机反转、在 CI 上仍然绿（CI 没有 `.env`）。
    凡是断言"某能力缺失"的用例，必须显式传 `rerank_url=""` 之类的空值，不能靠默认。
12. **启动脚本里"打印地址"必须排在"确认端口能绑"之后**（2026-10-26 现场反馈）。第一版
    `start.bat` 先 `echo 地址` 再 `serve`：端口被占时 uvicorn 报 `[Errno 10048]`，脚本走到
    `:end` 没有 `pause`，黑窗口立刻关闭——用户只看到闪过一行地址，浏览器打开是
    "拒绝连接"，反馈上来是"双击没启动成功"，还去试了 Streamlit 的 8501。
    地址只代表"打算用"，不代表"已就绪"：现在先用 `netstat -ano -p TCP | findstr ":PORT "`
    + `LISTENING` 探端口，被占就**不启动第二个实例**并直接给出可打开的地址，打印地址前置
    一句"看到 `Application startup complete` 再打开"；`serve` 的返回码要接住（`0` 正常、
    `-1073741510` 是 Ctrl+C、其余打印码 + 排查命令后 `pause`），**每条失败路径都 `pause`**。
    这类缺陷语法正确、能跑完、退出码还是 0，只有断言"顺序"和"失败必停窗"才抓得住，
    见 `tests/test_start_bat.py`（四条都做过变异反证）。
13. **答案里的 Markdown 必须"先转义、再解析"**（2026-10-26）。前端早期只把 `[n]`
    换成角标、其余原样输出，`**粗体**`、`* 列表`、`1. 编号` 全成了源码，用户反馈
    "没有层次感"。补渲染器时顺序不能反：模型输出是不可信文本，`& < > " '` 先变实体，
    之后插入的标签才只可能来自渲染器自己；反过来"先替换再转义"会把 `<script>` 一起
    放行。两个连带细节：`[2](url)` 的**链接替换必须排在 `[n]` 角标之前**（否则拆成
    "角标 + 裸地址"）、行内代码要先摘成占位符（否则 `` `a**b**c` `` 里的星号被当强调）。
    流式渲染每来一个 token 拿全文重渲一次即可（答案几 KB），未闭合的 `**` 会短暂显示
    成源码，闭合后自动恢复——不追求"半个标签"的实时美化。
    验收见 `tests/test_web_markdown.py`（抽出源码交给 node 跑真实实现，四条反证）。
14. **页码是展示字段，绝不能进 `chunk_id`**（2026-10-26）。给引用来源加"文件页码"时，
    PDF 有物理页码、docx 没有（Word 分页是渲染期动态算的，文件里不存在稳定页码）。
    页码只能作为 `Section.page` / `Chunk.page` 的展示字段透传到 citations，**不能参与
    内容寻址**——否则"同一段正文因排版微调页码变了"就会换 id，把内容寻址退化成位置寻址
    （`tests/test_chunking.py::test_page_is_not_part_of_chunk_identity` 钉死这一点）。
    db 迁移用 `_migrate`（`PRAGMA table_info` 判列，缺了 `ALTER TABLE ADD COLUMN`）——
    `CREATE TABLE IF NOT EXISTS` 对已有库**不会**加列，旧库补列后 page 全是 NULL，
    要拿到页码得 `ingest --force` 重新解析对应文档（`reindex` 只读归档、不重新解析，
    拿不到 page）。

15. **页码要落到"片段自己的起始页"，不能只继承章节标题页**（2026-10-27 复审 P1-2）。
    第一节实现只把标题行的页码写进 `Section.page`，分块时整节继承——于是一节从第 4 页
    开始、正文排到第 29 页时，这 8 个片段全标"第 4 页"。用户按页码翻过去找不到原文，
    引用就不可核。现在 `Section.line_pages` 逐行记页码，`build_chunks` 按片段在节内的
    偏移定位到行、取那一行的页码。三个必须记住的点：
    - **定位要在"去掉换行"的扁平文本上做**。`split_by_tokens` 按 `(?<=[。！？；\n])`
      切句，行尾没有句号的行会连换行一起成为一句，片段内部因此**可以带换行**；
      拿带换行的片段去原文 `find()` 一律落空，页码会静默退回节起始页（真实语料上
      表现为页码交错跳回，如 `[22,22,25,22,26,...]`）。游标也必须按扁平后的长度推进。
    - **行数与页码条数必须严格相等**（`headings.join_lines` 自己剥首尾空白，不能对拼好的
      字符串 `.strip()`，否则错位）。对不上就退回节起始页——**错位的页码比没有页码更糟**，
      它看起来是可信的。
    - 页码仍然**不参与 `chunk_id`**：`build_chunks` 只改 `page` 字段，实测六份文档
      新旧 id 完全一致，所以补页码不需要重建索引，`verify` 也不会报不一致。

16. **`--force` 重入库必须沿用已有年份**（2026-10-27 复审 P1-3）。`--force` 最典型的用途
    就是"内容没变、只想补个新提取的字段"（例如页码），这时用户不会重申 `--year`。
    把 `None` 当"清空"会让 `documents` / `vectors` 归档 / 后端标量列三处同时变空，
    按年份过滤从此漏掉整份文档，而入库结果显示成功。`ingest._resolve_year()` 定死规则：
    显式给了用新的，没给就沿用旧值；**新文档没有旧值可沿用**，仍是 None（不能填 0 或今年）。

17. **"服务端不代付"要按接口真正需要的 Key 逐个判**（2026-10-27 复审 P1-1）。
    `from_headers()` 只要带任意一个头就返回非空凭据（只传 `X-R4P-LLM-Model` 也算），
    而 `CredentialPool.settings_for()` 会把没提供的 Key 静默补成 `.env` 里那两把。
    所以 `svc_billable(request, need=(...))` 必须按角色校验：入库/检索要
    `embedding`，问答要 `embedding` + `llm`。另外两条不走主入口的侧门同样要堵：
    - `/api/credentials/verify`：空字段不许回落服务端 Key 真探测，否则它就是一条免费
      探测通道（`source` 回 `missing`，不发请求）；
    - 项目详情的摘要生成：判据不能用 `user_label`（**用户名对任何请求都非空**，只传
      模型头也是 `anonymous`），要用 `Services.llm_billable`。
    测试断言必须看"厂商实际收到的 Authorization / 有没有发生调用"，不能只看状态码。

---

## 7. 待办清单（按优先级，每条都有验收标准）

### 本轮已修（2026-10-27 复审三项 P1）

| 问题 | 根因 | 修复 |
|---|---|---|
| **关掉服务端 Key 回落仍能消耗服务端额度** | `svc_billable()` 只判 `creds is None`，而 `from_headers()` 带任意一个头就返回非空凭据（模型头也算）、`settings_for()` 再把缺的 Key 补成服务端那两把 | `svc_billable(request, need=(...))` 按角色校验（入库/检索→embedding，问答→embedding+llm）；`/api/credentials/verify` 空字段不再回落探测（`source=missing`）；项目摘要改用 `Services.llm_billable`（不能用 `user_label`，它对任何请求都非空） |
| **PDF 引用页码是章节标题页** | `build_sections()` 在正文分支只存文本、丢页码，整节继承 `Section.page` | `Section.line_pages` 逐行记页码；`build_chunks` 在"去换行的扁平文本"上定位片段起始行取页码；行数与页码条数不等则退回节起始页 |
| **`ingest --force` 会把 `doc_year` 清成 NULL** | CLI `--year` 默认 None，`force` 绕过跳过分支后按 None 重写三处 | `ingest._resolve_year()`：显式给了用新的，没给就沿用库里已有值；新文档仍为 None |

反证：7 条变异（退回"凭据非空就放行" / verify 回落探测 / 摘要按用户名判 / 年份按参数清空 /
片段继承节起始页 / 游标按带换行的长度推进 / build_sections 丢正文页码）各自**精确变红**。

### 已知问题（P2，本轮刻意不做，避免扩大改动面）

| 问题 | 影响 | 下一步 |
|---|---|---|
| 单文档删除没有公开入口 | `repo.delete_document` / `admin.delete_document` 已实现，但 CLI 与 API 都没有对应命令/路由 | 补 `delete-doc` 与 `DELETE /api/projects/{p}/documents/{id}`，并覆盖"后端删除失败 → 置 `unsynced` → 恢复" |
| DOCX 嵌套表格可能重复产出 | `_walk()` 已递归遍历单元格子表格，随后又显式遍历 `cell.tables` | 造一份含嵌套表格的真实 docx，断言每段只入库一次 |
| 多轮对话里旧回答的引用角标可能定位到新来源 | 引用抽屉是全局单例，新回答会覆盖内容，旧回答的 `[n]` 处理器仍指向旧卡片 | 每条回答各自保存引用数据，点旧角标时加载该回答自己的来源 |
| "正在检索回答中"在 `meta` 事件就消失 | `meta` 在 LLM 首 token 之前发出，提示会提前闪掉 | 改到第一个 `token` 事件再移除 |
| 公开入口缺"重新解析但不重算 embedding"的能力 | 补展示字段（如页码）也必须 `--force` 重算全部向量，白花钱 | 加 `refresh-metadata`：只重解析并更新 `page` / `section_count`，`chunk_id` 不变时跳过 embedding |

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

**P1-1　评估集扩充到 30~50 条　→ 2026-10-26 已完成（提交 `f859f90`）**
交付物：`eval/dataset.jsonl`（30 题，贴近原文术语）、`eval/dataset.paraphrase.jsonl`
（10 题，只描述意图、不用原文术语）、`eval/corpus.json`（语料指纹）、
`scripts/relabel_eval.py`（按锚点重算标注）、`tests/test_eval_dataset.py`（3 项守护）。
- 标注方式：**优先 `gold_chunk_ids`**（评价单位就是 chunk，最不容易歧义）；
  图省事可以写 `gold_snippets`，脚本会解析成包含该片段的全部 chunk——但要注意，
  一个片段在语料里出现得越分散，gold 集合越大，Recall 越难满分，这是刻意的保守处理。
- 实测（纯 BM25 / `--top-k 5`）：基础 30 题 **Recall@5 = 0.95**，改写 10 题 **0.50**。
  基础集在稀疏检索上就接近满分，说明它测的是关键词覆盖，不是语义检索——所以
  **两份必须分开报**，合成一个平均数会把"语义检索没提升"这个真实结论抹掉。
- 语料指纹 + 正文锚点解决的是「解析器一改 `chunk_id` 全变、旧标注静默失效」：
  `chunk_id` 是内容寻址的，章节路径改一个字 id 就全换，评估照样跑完、只是 Recall 莫名变低。
  现在 `relabel_eval.py` 先校验 `eval/corpus.json` 的 `ids_sha`，不一致就要求重标。
- **四行对照已补齐**（2026-10-26 傍晚，真机）：见下面 P1-1c 的表格，原始输出在
  `eval/reports/baseline-30.md` 与 `eval/reports/paraphrase-10.md`。

**P1-1c　消融对照的实测值（锁死坐标系，换任何一项都不可比）**
坐标系：`Qwen/Qwen3-Embedding-0.6B`（1024 维）· chunk 80~480 token · `recall_k=50` ·
`rrf_k=60` · 重排 `Qwen/Qwen3-Reranker-0.6B` · 库内 323 块。

| 召回方式 | 基础 30 Recall@5 | MRR | nDCG@5 | 改写 10 Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|---|
| dense | 0.817 | 0.704 | 0.733 | 0.500 | 0.333 | 0.376 |
| sparse | 0.950 | 0.834 | 0.857 | 0.500 | 0.275 | 0.332 |
| hybrid | 0.900 | 0.721 | 0.770 | 0.700 | 0.520 | 0.565 |
| rerank | 0.950 | 0.732 | 0.788 | 0.800 | 0.542 | 0.606 |

两点必须写进报告，否则这张表会被读错：
- **基础集上 sparse > hybrid（−0.050）**：语料 96% 是一份软件手册，专有名词密度极高，
  融合在稀释 BM25 的精确命中。这是语料构成问题，不是融合算法问题。
- **重排的 MRR/nDCG 有 ±0.005 抖动**（同一数据跑两次 0.737 / 0.732），Recall 稳定。
  别把小数点后三位当精确值。
- 没配 `R4P_RERANK_URL` 时 `rerank` 一栏必须等于 hybrid（NoopReranker），
  且报告会自己写一行「未启用重排」。**两行不相等就说明管道走岔了**，
  不要把它当成重排的收益——这一轮就是这么被骗了一次（见第 10 节第二十二轮）。

**P1-1b　正式评测集与模型版本要一起锁定**
报告里必须记下：embedding 模型名、维度、chunk 参数（`chunk_max_tokens` / `chunk_min_tokens`）、
`recall_k`、`rrf_k`。换了其中任何一个，前后两次数值不可直接比较。
`eval/corpus.json` 已经落盘了 chunk 参数与每份文档的 `chunks` / `ids_sha`，`rrf_k` 与
`recall_k` 仍要从配置里抄进报告——它们不在指纹覆盖范围内，改了不会让标注失效，只会让数字悄悄变。

**P1-2　Dockerfile + GitHub Actions CI**（CI 部分 2026-10-26 已完成，Docker 仍未做）
用户上次已表示可以补。CI 只跑 `pytest` + `ruff`（全离线，无需密钥）。

- **已完成**：`.github/workflows/ci.yml`（push/PR 触发，`pip install -e ".[dev]"` → ruff → pytest，
  Python 3.13，不装 milvus extra 时相关用例走 `importorskip`）。动机是第十七轮复审里那句
  "GitHub 没有 check run，测试结果仍属于提交报告"——把结论变成服务端可查的证据。
- 仍待做：Dockerfile 基础镜像 `python:3.13-slim`，注意：**不要装 torch**；Milvus Lite 在 Linux 下可用
  （但要避开非 ASCII 路径）。验收：镜像能 `docker run` 起来并响应 `/api/health`。

**P1-3　PDF 通道　→ 2026-10-26 已完成**
`parsers/pdf.py`（pymupdf，`>=1.24,<2`）+ `parsers/headings.py`（与 docx 共用的标题规则）。
识别策略与三条踩过的坑写在 `pdf.py` 的模块注释里，要点：

- **编号只定级，不认定"是不是标题"**。看到 `（一）` 就提级会把每段首行变成标题，
  章节树退化成"每段一节"（真实公文上复现过）。必须有视觉证据（更大或加粗）。
- **正文字号按全文原始 span 统计，表格也算在内**。只统计"文本条目"会让
  「一页表格 + 一个标题」这种页面把标题字号当正文，接着标题被当页边短行删掉。
- **跨 block 的段落要拼回去**（判据是行距 ≤ 一个字号且上句没说完），否则
  "住房城乡 / 建设科技" 中间留换行，分词与向量都会被干扰。
- **不做 OCR**：没有文字层直接报错说明原因。扫描件要么依赖本地模型要么依赖外部服务，
  两条都超出当前部署约束；静默入库空文档会让用户去怀疑检索参数。
- `pymupdf` 必须锁 `<2`：2.x 会把 `onnxruntime` 与版面分析模型一起装进来。

**P1-3 补充（2026-10-26 下午，拿真实工程文档跑出来的四条）**
前四条坑是"造样例"撞出来的，下面四条只有**真实文档**能撞出来——样例总是太干净：

1. **取得到字 ≠ 取到的是汉字**。方正书版等排版软件导出的 Type1 子集字体常缺
   ToUnicode 映射，`get_text()` 返回的是字形编号（`!"#$%&'()*`）。只查字符数的闸门
   对它完全失效：1 万多个"字符"会一路入库，把 BM25 文档频率和向量空间一起拖坏，
   全程无报错。判据是「汉字占比 ≥ 0.15 **或** 拉丁词密度 ≥ 40/千字」，阈值按实测标定
   （正常中文 0.63~0.86、正常英文 127/千字、乱码 0.000 且 19/千字）。
2. **目录页要整页跳过**。目录条目就是「编号一行、标题一行」，点线引导符行占比 ≥ 30%
   且 ≥ 3 行即判为目录页；多数页都命中说明判据失效，一律不跳（否则等于整篇清空）。
3. **编号与标题名分两行要拼回**。`1.1` 一行、`装配式设计要点` 下一行，不合并就得到
   一个只有编号的空节点 + 一个层级由字号排名决定的子节点。YJK 手册 461 节里
   195 个是这种空节点（42%），合并后降到 41 个。
4. **`第1 章` 中间的空格**会让 `^第[\d]+[章篇]` 失配，一级标题退回字号排名 →
   整棵树错位。另外 `1）`（只有右括号）是公式编号而不是标题，`_NUMBERED` 的 `\S`
   会把它当"标题文字"，上海计算细则因此多出 22 个公式碎片章节。

另有一个顺序问题：**两道准入闸门必须看全文、并且在跳目录之前判断**。
先跳目录再数剩余字符，会把"目录 5 页 + 正文 1 页"误判成扫描件。

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
8. **多进程必须共用同一份 Milvus 配置**（2026-10-26 已强制，不再是"写在清单上的约定"）：
   `meta.vector_backend_id` 记录 `milvus|<规范化 uri>|<collection>`，与类型成对登记；
   两个进程若各配一个地址/集合，第一次比对（`ensure_backend_current` / `ensure_active_backend` /
   启动确认）就会被拒，并同时打印两边的身份。历史库第一次启动时补记一次，不需要重建。
   运维侧仍然只有一条规矩：**同一份 `app.db` 只能配同一份 `.env`**。

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

第十轮（2026-10-24 22:20）：补**恢复契约**。基线 `14eb273`，两处都是"恢复动作自己不可靠"。

```
pytest -q                    → 173 passed（新增 8 项恢复边界用例）
ruff check                   → All checks passed
```

1. **空范围的 reindex 会清掉全量标记。** `reindex --project 某个不存在的项目` 走到
   `rows` 为空的分支，旧代码直接 `set_index_state(ready)`。于是那笔没走完的全量重建
   被"替它"宣布成功：归档已是新版、索引还是旧版、条数/模型/维度全都一致，从此再没有
   任何痕迹能发现。现在**只有局部范围就返回 0 条、不动任何标记**，并在返回值里说明
   还差哪一步；全量且无片段时，先确认"片段表真的空"（`chunk_count == 0`）再清；
   片段表非空却一条重建输入都取不到（JOIN 不上项目/文档）属于数据缺失，直接拒绝。
2. **`rebuild` 只恢复索引、不同步元数据。** 换模型重建中断后归档是新模型、`meta`
   还是旧模型；`rebuild` 把索引和标记弄好了，一致性检查却继续以"元数据模型 ≠ 归档模型"
   拒绝检索——**命令报成功、系统查不了**。现在完成条件是三条：归档能当恢复源（无缺行、
   每行带模型名、全库一套模型与维度）→ 后端条数 = 归档条数 → 归档的模型与维度连同
   `ready` 在一个事务里写回。任一不满足就抛错并**保留**标记，`rebuild` 返回非零退出码。

真实 Milvus 闭环（`C:/tmp_r4p_smoke10`，真 embedding + 真 Milvus Lite + 假"换模型"）：

```
入库 2 块（Qwen3-Embedding-0.6B）      → 索引=归档=2
换模型 + store.rebuild 直接失败         → 归档条数 2→2（完全没变）、模型已换成新名、
                                        元数据仍是旧模型、标记 rebuilding:* → 检索被拒
rebuild                                → 元数据模型=新名、维度=1024、标记=ready、
                                        索引 2 条、verify ok=true、检索 2 条命中
reindex --project 不存在的项目          → 0 条，标记仍为 rebuilding:*（不再被误清）
```

教训（第十轮）：**"恢复完成"不能用"恢复函数没有抛异常"来定义。** 恢复动作本身是一条
业务路径，它的完成条件要能被复核——否则最坏的情况是它把保护标记清了、把系统留在
"看起来正常、其实查不了"的状态里，比不恢复更难排查。

第十一轮（2026-10-24 23:05）：基线 `6217664`。这轮盯的是**"可选动作失败后留下什么"**。

```
pytest -q                    → 183 passed（新增 10 项：4 项迁移降级、4 项归档校验、2 项空库标记）
ruff check                   → All checks passed
```

1. **可选迁移失败会把能用的后端弄成不可用。** `rebuild_store` 先 `delete_orphan_vectors()`
   再 `store.rebuild()`。目标后端是 Milvus、源后端是 numpy 时两者共用 `vectors` 表：
   删完孤儿、Milvus 建集合失败、`maybe_promote` 捕获后继续用原来的 numpy 对象，而它的
   内存索引里还留着那些行。"迁移失败"于是变成了"条数对不上、检索被拒"。
   现在**归档在目标后端校验通过之前一个字节都不动**，孤儿行放到最后清。
   `reindex` 的全量分支同理（原来也是先删后建）。
2. **可选迁移的异常没被接住。** `maybe_promote` 只捕获 `ReindexError`/`RebuildError`，
   而 `MilvusStore.rebuild()` 在建集合/写向量/load 失败时抛的是 pymilvus 自己的异常。
   它挂在 `ingest_batch` 之后、同一个 `try` 里——一次没接住就把已经 done 的任务改成
   failed。现在构造与重建两段都按 `Exception` 兜底，只记 notes、保留原后端。
3. **空库会把未完成的重建标记"抵消"掉。** `IndexState.blockers` 先判 `is_empty` 再返回，
   空库上带着 `rebuilding:*` 也报一致；`ensure_writable` 复用它，于是新内容被收进来，
   再被同一条标记拒绝检索。现在标记先判、空判后置，同项目局部重建的例外照旧保留。
4. **`dim` 字段没和实际字节长度交叉验证。** 每行都声明 1024 维、实际只有 768 个浮点时，
   只比声明的校验全部通过；numpy 会崩在 `vstack` 的 "dimensions must match"，
   Milvus 侧则会先把这份坏归档宣布为恢复成功。现在逐行核对 `len(vec) == dim * 4`，
   不符就抛 `ReindexError` 并保留标记，指向"从原文重新生成向量"。

教训（第十一轮）：**可选项的失败路径要和主路径一样明确"谁没被改动"。**
"迁移是可选的"只是说它失败不该报错，不表示它可以先动共享数据再失败——
对 numpy 而言，`vectors` 表就是它的索引本体，删一行就是删它的索引。

第十二轮（2026-10-24 23:50）：基线 `06b5e19`。这轮是**架构级**的，前几轮都在补单次
调用的异常边界，这一轮补的是"两件事同时发生"。

```
pytest -q                    → 192 passed（新增 9 项：6 项互斥、1 项 id 集合、1 项空库顺序、1 项根服务归属）
ruff check                   → All checks passed
跨进程烟测（真起一个 Python 进程持锁）：
  子进程持锁                    → 主进程 is_busy 报出「跨进程重建，pid 12496，自 … 起」
  主进程 timeout=0              → MaintenanceBusy（错误里写清在等谁）
  主进程 timeout=30             → 排队 4.0s，子进程退出后立刻拿到
  子进程正常退出                → holder 说明文件已清理，锁自动释放
```

1. **重建与入库并发（P1）。** `reindex` 的执行顺序是"读快照 → 算 embedding（几分钟）
   → 置标记 → 写归档 → 换索引"。前两步期间新入库照跑，而入库只在入口检查一次一致性；
   重建算完拿着旧快照写回去，条数可能仍然对得上、id 已经不一样。现在 `reindex`
   全程持有维护锁，入库抢不到锁就明确失败。
2. **两个上传任务同时跑、其中一个触发后端切换（P1）。** 根服务换到 Milvus 之后，
   另一个任务还握着旧的 numpy 对象继续写。现在上传任务在启动时取**根服务当前**的
   后端；迁移本身也在维护锁内，且只由根服务执行。
3. **共享连接没有写串行（P1 附带）。** `check_same_thread=False` 只解决"能不能用"，
   两个线程的事务可以互相 `COMMIT`/回滚。现在所有写事务经同一把 `RLock`。
4. **请求级视图执行迁移（P2）。** 视图是 `replace()` 出来的临时对象，`store` 与
   `fell_back` 改了也留不下。现在视图指回 `root`，`maybe_promote()` 转交给根服务。
5. **空库重建先删归档再清后端（P2）。** 后端清空失败就停在"归档已改、索引还在"。
   顺序改成先确认后端空了、再删归档。
6. **恢复只核对条数（P2）。** 补 `store.ids()` 与目标 id 集合的比对。
7. **坏归档导致 `reindex` 起不来（P2）。** `NumpyStore` 构造阶段不再抛错，坏行跳过并
   记账；`consistency` 新增 `malformed` 阻断项，指向"从原文重新生成向量"。

教训（第十二轮）：**"每一步都对"不等于"同时发生也对"。**
前十一轮把单个动作的失败路径修得差不多了，但两个动作各自成功、结果仍然是错的库。
互斥不是性能问题，是正确性问题——所以抢不到锁要说"稍后重试"，而不是排队等到一个
真正的失败；锁的粒度也必须覆盖"算"的那段时间，不能只覆盖"写"的那一下。


第十三轮（2026-10-26 10:30）：基线 `f85ce2c`。这轮不再补零散的异常分支，而是收敛到
三条契约：**读写版本边界**、**批次原子写入**、**跨进程状态识别**。

```
pytest -q                    → 200 passed（新增 8 项）
ruff check                   → All checks passed
真机（8899，numpy 后端）：verify → 三方 id 全对；health → consistent=true / 8 条
跨进程租约（真起服务 + CLI）：服务在跑 → doctor 报"另一个服务进程…已跳过中断清理"，任务不被动
                            服务被杀 → doctor 不再报该提示（租约随进程由内核释放）
```

1. **在途查询跨过索引切换（P1）。** `ensure_queryable` 与 `search` 之间隔着一次网络
   往返；那几秒里另一个进程可以换完模型，查询向量仍是旧模型的，于是"能查、但语义
   错位"。现在用代次做两级确认（算完向量一次、检索完一次），跨代就抛 `IndexMismatch`
   让调用方重试。查询仍然不持有维护锁——让提问排队几分钟等重建结束是不可接受的。
2. **批次对着旧后端对象写（P1）。** 上传任务在开始时就拿到了 `services.store`；
   那个指针期间可能被迁移换掉，而维护锁管不住"手里这个对象是哪一代"。
   现在 API 传 `StoreRef(lambda: root.store)`，每次使用才解析；`ingest_batch` 也在
   整批入口抢锁，抢不到时一个字节都没改。
3. **`NumpyStore.upsert` 半批提交（P1）。** 逐条写、循环内校验，第 2 条不合格时
   第 1 条已经躺在隐式事务里，任何一次 `commit()` 都会把它提交。改成整批校验 +
   单事务提交。
4. **跨进程看不到缓存变化（P2）。** 维护锁只保证不同时写；CLI 重建完把锁一放，
   API 里的 numpy 矩阵就是旧的，条数还对得上。新增 `meta.index_generation` /
   `data_revision`：`NumpyStore` 每次读之前核对，变了就重新加载（`_Snapshot` 整体
   替换，避免"新 ids 配旧矩阵"）。
5. **`is_busy()` 预检查当准入（P2）。** 它只是"问一句"，问完到动手之间还有窗口。
   现在真正的准入是批次级维护锁，项目与任务都在锁内创建；被拒绝时不留空项目。
6. **CLI 装配顺手把在线任务标成中断（P2）。** 见不变量 18。

教训（第十三轮）：**"加锁了"不等于"边界画对了"。**
这一轮四个问题里，三个都不是"忘记加锁"，而是锁的范围与外层对象的时间错位：
查询在锁外等网络、批次握着加锁之前取到的对象、缓存跟着进程走而锁跟着文件走。
写这类代码时要问的不是"有没有锁"，而是"**我手里的这个状态，是在哪一刻取的**"。


第十四轮（2026-10-26 11:05）：基线 `04d122c`（文档订正 `8b22227` 不算代码）。
这轮盯的是**动作与它依赖的状态之间那段时间**。

```
pytest -q                    → 207 passed（新增 7 项）
ruff check                   → All checks passed
旧代码反证（git stash push -- rag4prefa）：
  tests/test_review14.py 五个用例在旧代码上全红 → 新用例确实咬住了旧行为
Milvus 并发（真 Milvus Lite，ASCII 临时目录）：
  另一线程持维护锁 + 已建 __staging → 本进程打开被推迟，暂存集合仍在
  （旧代码的构造函数会把它删掉）
真机（8899 服务在跑，新代码做 CLI）：verify → 三方 id 全对 / 8 条 / ok=true
  （改之前这条命令直接崩在 PermissionError，见下面第 7 条）
```

1. **启动恢复绕过维护锁（P1）。** `MilvusStore.__init__` 里调 `_recover()`，它会删
   `__staging`；而 `build_services()` 先 `open_store()` 再查租约，这一段没有锁。
   进程 A 持锁重建、进程 B 启动，B 就把 A 的半成品删了。现在构造只连客户端，
   `_open()` 按"干净 / 全新 / 有残留"分三路，有残留时 `timeout=0` 抢锁，
   抢不到就 `_defer()`——宁可这次查不了，也不能替别人做决定。
2. **快照被"追认"为新版本（P1）。** `_load` 读行与读代次不是同一瞬间：中途提交的
   写入不在矩阵里、代次却是新的，此后 `_refresh_if_stale` 认为缓存最新，**永久**
   少那几条。改成双读、组快照夹在两次读代次之间，不一致整份重来；连续 5 次不一致
   抛 `SnapshotUnstable`，不发布没证明过的快照。
3. **CLI 在锁外绑定后端（P2）。** 传 `StoreRef` 只解决"本进程指针过期"，
   跨进程那一层不够。补上 `meta.vector_backend`（活动后端落库）+ `auto` 不再静默
   回退 + 维护动作在**锁内** `ensure_backend_current` 复核，三件一起才闭合。
4. **回滚不还原被覆盖的旧行（P2）。** 共用 `chunk_id` 的 `doc_year` 已被 upsert
   改成新版；事实事务失败后只删新增，那些行就停在"新过滤元数据 + 旧正文"。
   写之前留旧行，失败时先写回、再删新增。
5. **改年份被当成"没变化"（P2）。** 跳过条件漏了 `doc_year`，且 `DocumentInfo`
   压根没有这个字段（新测试第一次红正是因为 AttributeError，印证了这条）。
   补字段 + 年份变化走只改元数据的路径。
6. **上传文件策略（说明）。** `upload_dir` 是**原始文档归档区，落盘即持久**，
   不是临时暂存区——被维护动作挡下时文件留在原地是设计意图（重试/审计要用），
   同名覆盖不会越积越多。任务消息里补一句"无需重新选择文件"。
7. **锁文件打不开时 CLI 直接崩（P1，真机撞到的）。** 服务进程持着租约时，Windows 上
   别的进程 `open(worker.lock, "a+b")` 报 `Permission denied`——不是"锁定失败"，
   是**打开**就失败。`_FileLock.try_acquire` 只把 `msvcrt.locking` 放在 `try` 里，
   于是 `doctor` / `verify` / `ingest` 在服务运行时全部崩在 `worker_alive()`，
   而它本来只是想说一句"有人在跑，别动别人的任务"。
   现在 `open` 失败也算"抢不到锁"：`worker_alive` 报"有人活着"（保守，不会误清理
   在线任务），维护锁报"稍后重试"。用**同名目录**占住锁文件名可以在任何平台稳定复现。

教训（第十四轮）：**"条数没变"是最不可靠的验收标准。**
这五个问题里有四个的最终表现一模一样：条数对、模型对、维度对、
`health` 报 `consistent=true`，只有内容或过滤结果错了。计数是必要条件、不是充分条件——
凡是"只改其中一部分内容"的动作，验收断言都必须落在**内容**上（哪个 id 在不在、
哪一年的过滤能不能筛到、这份快照是哪一版），而不是"总数没变"。


第十五轮（2026-10-26 11:50）：基线 `c828feb`。这轮盯的是
**"这个动作凭什么认为自己可以动"**。

```
pytest -q                    → 220 passed（新增 13 项：test_review15.py 12 + Milvus 竞态 1）
ruff check                   → All checks passed
旧代码反证（git stash push -- rag4prefa），6 处断言真的红了：
  auto 在库里记着 numpy 时仍去构造 MilvusStore  → 断言"不该再试探"失败
  open_store 直接把打开的后端写进 meta           → 断言"记录没被改写"失败
  改年份失败后 index_state 仍是 ready            → 断言 unsynced:* 失败
  删项目不核对后端                              → 断言"拒绝"失败
  Milvus 快速打开补建空集合                      → count 返回 0（那一条旧向量丢了）
  其余为 AttributeError（settle_backend / recoverable_chunk_ids 是新入口，旧代码没有）
```

1. **后端切换把未验证的 Milvus 设为活动索引（P1）。** `open_store()` 只要集合能打开就
   写 `meta.vector_backend`，而"能打开"证明不了"里面装的是归档这一套"：遗留集合可以与
   归档**条数相同、chunk_id 全不同**。现在拆成三件事：**打开**（不登记）、**确认**
   （`Services.settle_backend()` 逐 id 核对，不一致就从归档重建，通过才登记；证明不了就
   不登记并说明怎么修）、**换后端**（`rebuild_store` 唯一入口，`ready` 与后端名同一个事务）。
   两个推论也补上了：`auto` 尊重库里已定的 numpy（否则每次重启都推翻上一次的决定）；
   `rebuild` 不再被 `ensure_backend_current` 挡住——文档里写的"改成 numpy 再 rebuild"
   以前在那种状态下根本走不通。
2. **Milvus 无锁快速打开仍有 TOCTOU（P1）。** `_open()` 看到"干净布局"与真正 load 之间
   隔着一段时间，另一个进程可能正好在这段时间里做完两次 rename。旧实现这时会
   `_ensure_collection` 补建一个**空集合**，对方第二次 rename 撞名失败、回滚条件又不成立，
   旧索引永久留在 `__retired` 里。现在 `_open_clean()` 只加载"此刻仍然存在"的正式集合，
   消失了就返回 False、重看布局后改走持锁路径。**没有锁就没有资格创建/删除/重命名集合。**
3. **改年份失败后检索照常（P2）。** 归档已是 2025、后端还是 2024，条数一条不差，
   任务返回 failed 而已。现在置 `unsynced:<doc_id>` → 改文档与归档 → 覆盖后端 → 撤成
   **上一次的状态**（不是无条件 ready）；后端那一步失败时标记留着，检索继续拒绝，
   `rebuild` 一条命令清除。
4. **删项目不核对后端身份（P2）。** 维护锁只保证"没有别人同时在改"，锁不住"手里这个
   后端是不是库里记着的那个"。现在 `ensure_backend_current` 排在 `repo.delete_project`
   之前——否则先删事实与归档、再往错的索引发一批 id，删除既不完整也说不清以哪边为准。
   顺带给 API 补了 `IndexMismatch` → 409（以前这类失败会变成 500）。

教训（第十五轮）：**"动作成功"和"我知道自己在动什么"是两件事。**
这四个问题里没有一行代码算错，全错在"它凭什么认为自己可以动"：能打开集合就当它是当前
版本、看到干净布局就当它一直是干净的、返回一句 failed 就当系统已经停了、拿着维护锁就当
手里的后端是对的。写这类代码时，每个动作前面都要能补一句"我掌握了什么证据"；
拿不出证据的，只能拒绝或延后——**"看起来没问题"不是证据**。


第十六轮（2026-10-26 14:40）：基线 `daeb605`。这轮盯的是
**"证明到什么程度才算证明"**。

```
pytest -q                    → 230 passed（新增 10 项，test_review16.py）
ruff check                   → All checks passed
旧代码反证（git stash push -- rag4prefa），8 处断言真的红了：
  settle 只比 id：内容过期也直接登记        → 断言"必须重写内容"失败（rebuilds == 0）
  换后端不推 index_generation              → 断言"代次必须变大"失败
  长跑服务拿着旧后端继续查                  → 断言"拒绝检索"失败（旧代码照常返回结果）
  settle 抢不到锁仍返回"确认过"             → 断言 is False 失败（旧代码根本没有这个结论）
  空归档 count() 抛异常被当成 0             → 断言"不许登记"失败（旧代码登记了）
  回滚失败只写进错误文本                    → 断言"检索已停用"失败、标记仍是 ready
  删项目时后端删除失败                      → 抛的是 RuntimeError，没有标记
  其余 2 条是正向守卫（回滚成功不留标记、删除成功清标记），旧代码同样通过
```

1. **换后端只核对 id，证明不了内容（P1）。** 上一轮登记判据是 id 集合，而它只证明
   "同一批片段"：Milvus 里还有向量值、`doc_year`、`project_code`。从 Milvus 切到 numpy、
   在 numpy 上重建过向量或改过年份之后，旧集合的 id 可以一个不差、内容整整差一代。
   现在只要登记值不是手里这个后端，就**一律从归档重写**（`rebuild_store` 逐 id 校验 +
   `set_index_meta(backend=)` 一个事务登记），同一后端已登记才走快速启动。
   顺带补上了那条分支原本漏掉的 `index_generation` 推进：换索引必须作废在途查询。
2. **运行中的服务可能还在查已不是活动后端的索引（P1）。** 代次只能发现"这次查询跨越了
   切换"，发现不了"这个进程从一开始拿的就是另一份"。现在 `ask.retrieve` 准入时校验
   `store.name == repo.get_vector_backend(conn)`；服务层再用 `backend_confirmed` 三态
   （`None`/`True`/`False`）区分"没跑过确认"和"跑过但没证明"，后者让 `/api/chat` 与
   `/api/search` 直接 503。`health` 也把 `recorded` 与 `confirmed` 分开报。
3. **补偿失败不落盘（P2）。** `upsert` 抛异常不等于批次没生效；"写回旧行"再失败时，
   索引里是一半旧行、一半新行，而 id 集合、条数、模型、维度全都可以一致。
   `_restore_vectors` 现在失败即置 `unsynced:<doc_id>`；删项目时后端删除失败置
   `unsynced:project:<名字>` 并抛 `IndexMismatch`（API 是 409）。**回滚成功则不置标记**，
   避免一次网络抖动就让整库停服。
4. **读不到后端条数被当成"确认为空"（P2）。** 空归档分支的 `except: leftover = 0`
   把 SDK 超时、连接被占都读成"后端是空的"，接着登记；之后 `open_store` 又把它当
   "已有活动后端"，这个从没被证实过的后端就一路被信任下去。现在读取失败即
   `backend_confirmed=False` 并说明"是读不到、不是空"。

教训（第十六轮）：**上一轮把"证明"写成了协议，这一轮才发现协议证明的是最弱的那一层。**
id 集合一致、正在被打开、上次登记过——三条都是"看起来已经验证过"，而它们关于**内容**
什么也没说。这类缺口的共同形态是：**断言的对象比它要保护的东西低一维**。
问"这个证据能证伪什么"比问"它证明了什么"更有用：能证伪"换了一批片段"，证伪不了
"换了一批向量"；能证伪"这次查询跨了版本"，证伪不了"这个进程从头就拿错了索引"。

第十七轮（2026-10-26 15:20）：基线 `75e49ce`。这轮盯的是
**"这个结论有没有贯穿所有入口"**。

```
pytest -q                    → 244 passed（新增 14 项，test_review17.py）
ruff check                   → All checks passed
旧代码反证（git stash push -- rag4prefa scripts），14 项全红，其中核心几条：
  未登记的库上 ask.retrieve                  → 断言"拒绝检索"失败（旧代码照常返回）
  cli ask 不走准入                            → 断言退出码为 1 失败（旧代码返回结果）
  未登记的库上 ingest_batch                   → 断言抛 IndexMismatch 失败（旧代码照写）
  未登记的库上删项目                          → 断言 503 失败（旧代码删掉了）
  全量 reindex 成功但不登记                   → 断言"登记了"失败（旧代码 get_vector_backend 仍是 None）
  deferred 的 Milvus 被报已确认               → 断言 confirmed is False 失败
  同名不同地址两个后端                        → 断言"拒绝"失败（旧代码只比 name）
  count() 抛 BackendUnavailable              → 断言 503 失败（旧代码 500）
```

真机验证（`C:/tmp/r4p-unreg` 上跑真实库的副本，主库只读）：

```
doctor（未登记 + 另一个进程占着维护锁）
  → backend={recorded:null, confirmed:false, identity:"numpy", recorded_identity:null}
  → index.issues 出现"本进程没能确认活动后端……检索与普通写入都会拒绝"，ok=false
cli ask "悬挑结构"    → 退出码 1，"检索已拒绝：本进程启动时没能确认活动后端……rebuild"
cli ingest … --project 准入验证
                     → 退出码 1，"入库未执行：写入已拒绝……"；jobs/projects/chunks 一个没变
POST /api/search（同状态的服务，8898） → 503 + 同一句可读的话
DELETE /api/projects/美睿金座（同状态）  → 503；项目、片段、归档都还在
rebuild              → rebuilt=8，vector_backend=numpy，vector_backend_id=numpy
恢复后 doctor        → confirmed:true、consistent:true；登记表两项俱全
主服务（8899，新代码） /api/health → backend 里多了 identity / recorded_identity，confirmed:true
GitHub Actions（run 40284754083）  → success：安装依赖 / 静态检查 / 离线测试 三步全绿，41 秒
```

1. **未确认只保护了 API 的两个路由（P1）。** 判据挂在调用路径上，于是同一个库在
   `curl /api/search` 和 `python -m rag4prefa ask` 下给出两种结论；上传入口还在往
   没被证明过的索引里写。根因更靠前：`ensure_backend_current` 在"没登记"时**直接放行**，
   而"有内容 + 没登记"的真实含义是"确认动作跑过并失败了"（首次部署时库是空的）。
   现在这条判据下沉进 `consistency.ensure_registered`，查询/普通入库/删项目共用，
   诊断与修复入口（`doctor`/`verify`/`rebuild`/全量 `reindex`）保留并各自说明职责。
2. **快速启动把"名字相同"当成"已经确认"（P2）。** `_defer()` 的 MilvusStore 名叫 milvus、
   却没打开；`/api/health` 报正常，失败被推迟到 `search()` 里，还变成 500。
   现在快速路径也要 `store.probe()`（Milvus 真读一次条数），证明不了即 `False`；
   `BackendUnavailable` 在 API 层有独立处理器 → 503。
3. **`milvus` 只代表类型，不代表位置（架构边界）。** 两个进程共用一份 SQLite、各配一个
   URI/集合时，`store.name` 相同、索引却是两份。现在 `meta.vector_backend_id` 与类型
   **成对登记**（`milvus|<规范化 uri>|<collection>` / `numpy`），检查与登记都带上它；
   历史库就地补记（比较后再写，不覆盖别人刚换的后端）。

教训（第十七轮）：**"我这边挡住了"不等于"这件事被挡住了"。**
准入如果只在某一条调用路径上生效，它保护的其实是那条路径，而不是那个不变量；
判断"够不够"的方法是**把入口列一遍**（HTTP、CLI、脚本、后台任务、库函数），
逐个问"它凭什么可以动"。另一个反复出现的形态是**同义词被当成同一个东西**：
"名字相同"被当成"同一个索引"，"注册过"被当成"打开着"——每一条都值得再问一句
"它们还能在哪一层不一样"。


PDF 通道（2026-10-26 16:40）：`pymupdf 1.28.2`，`255 passed`，ruff 干净。

新增 `tests/test_pdf_parser.py` 11 项。**旧代码反证**：把两处判据分别打掉后
（`_is_furniture` 恒返回 False、去掉"视觉证据"要求），对应两条用例立刻变红，
还原即恢复全绿——证明确实是这两条判据在起作用，不是恰好通过。

```
真实文件（课题/Ref 里的省建设厅通知，3 页，Chrome 打印生成）：
  解析                              → 7 节：文档标题 / 一、申报内容 / 二、申报要求 /
                                      三、申报方式 / 四、申报时间 > 附件：
  分块                              → 6 块，chapter_path 全部正确
  端到端（真实 SQLite + NumpyStore + 假向量）
                                    → status=indexed，索引=归档=后端=6，blockers=[]
```

这一轮踩到的四个坑，都不是"写错了代码"，而是**判据选错了层**：

1. **"看到编号就是标题"会把每段首行提成标题。** 真实 PDF 里"（二）科研项目负责人应为…
   具有本领域或相关领域高级"正好 55 字、不以标点收尾，被提成标题后正文挂在它下面，
   章节树变成"每段一节"。改成"先有视觉证据（更大或加粗），编号只负责定级"。整篇一个
   视觉标题都没有时，才退回"只信编号"。
2. **正文字号必须按原始 span 统计，表格要算进去。** 先按"过滤后的文本条目"统计，
   「一页表格 + 一个标题」这种页面把标题的 16pt 当成了正文，于是标题不再"比正文大"，
   接着被页边短行规则当页眉删掉，整篇退化成"前言"一节。两条修：统计口径换成原始 span，
   以及**页边的大字一律不当页眉**（页眉最多比正文大半磅左右）。
3. **误判保护在短文档上会反过来帮倒忙。** "丢弃比例 > 30% 就全放回"本意是防整页表格，
   但两页的小文件里页眉页脚本来就占一半段数，结果一条都没丢掉。改成"丢完不剩才放回"，
   比例保护只对 20 段以上的长文档生效。
4. **同段被拆成多个 block 时换行会切词。** "住房城乡 / 建设科技" 中间留一个换行，
   分词和向量都受影响。判据用行距——段内 gap≈7.5、段间≈13.6（字号 9.1），
   阈值取一个字号宽，实测能干净分开。

另一个顺手修的真 bug：**表格与正文的收集顺序**。原来一页里先把所有表格收进来、再收
文本块，整页的表格都跑到正文前面；现在按 y 排回原顺序，用例锁住了这一点。


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

真实工程文档实测（2026-10-26 17:30，第十九轮）：`264 passed`，ruff 干净。

用户提供的 `课题/工程文档` 共 5 份，拿它们过准入 + 解析：

```
YJK 手册（368 页）         → indexed  307 节 301 块  5.0s
沪建建材 250 号（24 页）    → 拒收：0 个字符，纯扫描件
装配式评价标准 DBJ33T       → 拒收：10899 字符全是字形编号（汉字占比 0.0%）
上海计算细则（14 页）       → indexed   26 节  13 块  0.5s
审查意见单 docx            → indexed    1 节   1 块（填表类，全文约 700 字）
合计 315 块 / 86 963 字，平均 276 字/块，索引=归档=后端=315，blockers=[]
```

**这一轮的结论不是"解析器又多支持了什么"，而是"真实文档的质量分布长什么样"。**
一个客观事实：5 份里 2 份进不来，而它们恰好是本课题最核心的两份**规范**。
这不是解析器的缺陷——扫描件要 OCR、缺 ToUnicode 的字体任何工具都还原不了；
但如果不做可读性闸门，那 10 899 个乱码字符会全部入库，且**没有任何报错**。

新增 4 类判据，每条都用打掉判据的方式反证过（共 4 项，全部精确变红）：

1. **可读性闸门**（`_ensure_readable`）：汉字占比 ≥ 0.15 **或** 拉丁词密度 ≥ 40/千字。
   阈值来自实测标定，不是拍的——正常中文 0.63~0.86、正常英文 127/千字、
   真乱码 0.000 且 19/千字。两道边界都留了 2 倍以上余量。
2. **目录页整页跳过**（`_looks_like_toc`）：点线引导符行 ≥ 3 且占有效行 ≥ 30%。
   YJK 手册 368 页里只有 p2~p6 命中（35~37 行 / 71~75 行 ≈ 0.49），正文页 < 0.1。
3. **编号与标题名合行**（`_merge_number_only_heads`）：YJK 手册空节点 195 → 41，
   空节占比 40% → 10%，章节树从 461 节收到 307 节。
4. **章节空格与公式编号**：`第1 章` 容忍空格；`1）` 不再算编号（`_NUMBERED` 的 `\S`
   改为 `[^\s)）]`），上海计算细则的 22 个公式碎片章节归位。

顺带修了一个**顺序**问题：两道准入闸门必须看全文、且必须在跳目录之前判断。
先跳目录再数剩余字符，会把"目录 5 页 + 正文 1 页"误判成扫描件——测试一造出来就红了。

反证过程中有两次"打掉判据仍全绿"，都是**测试自己没鉴别力**，不是代码没问题：

- 目录用例断言的是"目录条目不成为标题"，但目录行与正文同字号时它本来也成不了标题。
  改成断言"目录文本不出现在任何 section 的正文里"才有鉴别力。
- `第1 章` 用例里章的字号本来就最大，按字号排名恰好也是 1 级，两种判据结果相同。
  把字号故意反过来（节 18pt > 章 14pt），只有认编号才能得出 1/2。

教训（第十九轮）：**样例总是比真实数据干净。** 前十八轮的 PDF 判据全部由自造样例验证，
一上真实文档就连撞四处——目录页、空节点、`第1 章` 的空格、公式编号。
这四条的共性不是"错了"，而是**只在对的样本上成立**：每一条判据都假设了某种排版，
而真实排版总有不满足假设的那一份。所以"支持 PDF"这件事，验收标准不该是
"能解析我造的那份"，而是**"能在真实文档上说出它为什么解析不了"**——
两份进不来的文件，现在的输出是可操作的（换可复制版本 / 做 OCR），这比静默入库有用。

### 第二十二轮（2026-10-26 傍晚）：真机入库 + 评估四行对照

**Key 到位后的真机链路**（`.env` 填 `R4P_EMBEDDING_API_KEY` / `R4P_LLM_API_KEY`）：

- 先各打一次冒烟：embedding 返回 1024 维（与库里登记的维度对得上）、LLM 正常回话。
  **先冒烟再入库**：315 块的入库要跑十几批 embedding，Key 填错站点时第一批就报
  `30014 Token is invalid`，跟"Key 被删"同码，直接拿大批量去试会浪费时间也看不出原因。
- 入库三份可用文档：`软件手册` 301 块（307 节）、`装配式规范` 13+1 块。
- `verify`：片段 323 / 归档 323 / 索引 323，四种缺失全 0；`doctor` 的
  `credentials_missing` 清空、`consistent=true`、`issues=[]`。
- 库内 323 = 本轮 315 + 早先入库的美睿金座计算书 8 块。**同一库只能一套向量空间**，
  所以新语料必须沿用 `Qwen/Qwen3-Embedding-0.6B` / 1024——换 4B 会被 `consistency` 直接拒绝。

**四行对照（真机，`eval/reports/` 里是原样输出）**

| 召回方式 | 基础 30 Recall@5 | MRR | 改写 10 Recall@5 | MRR |
|---|---|---|---|---|
| dense | 0.817 | 0.704 | 0.500 | 0.333 |
| sparse | 0.950 | 0.834 | 0.500 | 0.275 |
| hybrid | 0.900 | 0.721 | 0.700 | 0.520 |
| rerank | 0.950 | 0.732 | 0.800 | 0.542 |

结论有三条，第三条最重要：

1. **基础集上纯 BM25 压过混合召回 0.05。** 语料 96% 是一份软件手册，按钮名参数名这类
   专有词密度极高，融合是在稀释而不是增强。这不是融合的实现问题，是**语料太单一**——
   A2（扩语料）没做完，这个反常就会一直挂着。
2. **改写集才是语义检索的战场**：dense 与 sparse 都只有 0.50，hybrid 0.70，重排 0.80。
   同一句"两面都叠合的那种墙要打几折"，不含原文术语时两路单独召回都会漏。
3. **只报一份会得出互相矛盾的结论**（"检索已经很好" vs "BM25 没用"）。两份必须并列报。

**这一轮抓到的三个真问题，都是"看起来生效"的假象：**

1. **评估里 rerank 行与 hybrid 行用了不同深度的候选池。** 旧代码 rerank 把 k 换成
   `max(k*4, 20)` 再走同一段融合：融合池深 100、只把前 20 条交给重排；hybrid 是 50 深的池子取前 5。
   于是在**没有配重排服务**（NoopReranker）的情况下两行数字仍然不同（0.933 vs 0.900），
   报告读起来就像"重排带来 +0.033"，实际一个候选都没被重排过。
   现在两行共用 `_hybrid_pool_size(k)`，Noop 下两行必须完全相等，
   报告也会显式写「未启用重排」。
2. **CLI 里先截断、后重排。** 第一版 `cmd_ask` 先 `ask.search(top_k=3)` 拿到 3 条，
   再把这 3 条交给重排——重排只在一小撮里挑。真实查询上验证过：正确答案在融合结果里
   排第 9，把 25 条全交给重排它能升到第 1；只给 3 条时它压根不在里面，
   而日志、分数、`rerank=True` 全都正常。现在改成 `ask.search` 内部"先重排、后截断"。
3. **`Settings` 会读本机 `.env`。** 重排一开，那条"未配置时应提示未启用"的断言在本机反转，
   在 CI 上仍是绿的（CI 没 `.env`）——正是"本地绿 ≠ CI 绿"的典型。用例现在显式传空值。

**顺带的两个决定**

- **重排的开关语义统一成"看装的什么，不看配置写的什么"**：`ask.rerank_available()`
  判 `isinstance(reranker, NoopReranker)`，三处出口（`stream_answer` 默认值、CLI `--rerank`、
  `/api/health` 的 `rerank_configured`）共用它。配了 URL 就默认开，`--no-rerank` / `rerank:false`
  仍可显式关掉。理由：报告里的 rerank 一行必须对应用户实际跑到的那条链路，
  否则那行数字描述的是一条只能靠脚本复现的旁路。
- **前端那个"启用重排"开关以前是一个安静的谎**：没配重排时勾上也不会有任何效果。
  现在没配就置灰并写明原因。

**两个新的计量事实**

- **重排分数有抖动**：同一份数据两次跑，Recall@5 完全一致（0.950 / 0.800），
  MRR 分别是 0.737 和 0.732。所以引用重排结果时，小数点后三位不要当"精确值"讲，
  差异在 ±0.005 量级内应视为同一次测量。dense / sparse / hybrid 是确定性的，没有这个问题。
- 融合后候选可能少于 `recall_k`（实测 25 条），因为同 id 去重与筛选在融合前就生效了。

**验证**：`277 passed`（270 → 277）、`ruff check rag4prefa tests scripts` 干净、
前端 script 块 `node --check` 通过。真机端到端问"双面叠合墙的修正系数是多少"，
答案给出 0.75 并带 `[1][2]` 引用。

---

### 第二十三轮（2026-10-26 晚）：换语料 + 服务端 Key 回落开关

**语料从"一份软件手册占 96%"换成 6 份分散的工程文档**（66 块、两个项目：
`装配式规范` 21 块 + `装配式计算书` 45 块）。上一轮那条结论——"基础集 sparse > hybrid
是因为语料太单一、专有名词密度过高导致融合稀释了 BM25 的精确命中"——在本轮得到验证：
换语料后四行对照的反常消失了。

| 数据集 | dense | sparse | hybrid | rerank |
|---|---|---|---|---|
| 基础 25 题 Recall@5 | 0.960 | 0.920 | 0.960 | 0.920 |
| 改写 10 题 Recall@5 | 0.700 | 0.800 | 0.800 | 0.800 |

hybrid 不再输给纯 sparse，说明之前吃亏确实是语料问题、不是融合实现问题。
评估集 30 题里 17 题原本问软件手册的按钮/参数名，随语料作废，全部按新语料
重写锚点、重算 `gold_chunk_id`，`corpus.json` 指纹同步重算。

**服务端 Key 回落开关（回答"上线后两个 Key 不能放 .env 里"）**：

架构上早就支持"Key 随请求走"（浏览器经 `X-R4P-*` 头带自己的 Key，服务端
`for_credentials()` 换视图、不落库）。缺的是"没带 Key 的请求怎么办"——原来一律
回落到 `.env` 公共 Key，公网部署下这是无底洞：匿名请求无限消耗额度、账单一个人付。

新增 `R4P_ALLOW_SERVER_KEY_FALLBACK`（默认 true，单机演示/CLI/curl 行为不变）。
关掉后，入库 / 检索 / 问答三个花钱入口对"没带 Key"的请求直接 401，并告诉用户去哪
填自己的 Key——而不是等厂商报 "Token is invalid"（那时额度已烧掉一部分，用户还会
去反复核对一把他根本没填过的 Key）。项目摘要页"顺手调一次 LLM"的侧门一并堵上。

**顺手补的两个缺口**（都是"看起来成功、实际没发生"）：

1. 目录导入以前只按 .docx/.pdf glob，其余文件既不入库也不出现在任何输出里——
   用户以为都进去了，其实少了几份，语料缺文件是个安静的错误。现在逐文件分类，
   跳过项逐个报"为什么 + 怎么办"。
2. 删项目以前只有 HTTP 接口，命令行用户删掉磁盘文件后没法清库。现在
   `rag4prefa delete --project` 默认只预览影响面，`--yes` 才真删。

**验证**：`287 passed`（277 → 287）、ruff 干净、前端 `node --check` 通过。
凭据回落 6 条新用例都做了反证：把 `svc_billable` 的拦截条件换成 `False`，两条
"匿名应 401"的用例精确变红（200 ≠ 401），而"带用户 Key 仍放行""默认回落仍放行"
不受影响。

---

## 11. 深入阅读顺序


1. `REFACTOR_PLAN.md`（本目录）—— 重构方案 + **第十二节有实施复盘**，记录计划与现实的 5 处偏差，值得先看。
2. `F:\project\PyProject\milvus\PROJECT_REVIEW.md` —— 原系统评估报告，说明"为什么要重写"的每一条证据都能定位到行号。
3. `README.md`（本目录）—— 面向使用者的安装、API、已知局限。
4. 原系统源码 `F:\project\PyProject\milvus\Vector_search_plus.py`（1 991 行）—— 只在需要对齐某个行为时查阅。
