# 交接说明 · RAG4Prefa

> 写给接手的 agent：你**没有**上一轮对话的上下文，这份文件是唯一入口。
> 读完这一份就能开始干活；需要更深的背景再看第 11 节列的文档。

**交接时间**：2026-10-26 15:20
**交接时状态**：可运行、**610 项离线测试全绿**；真实厂商 API 已端到端验证
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

## 3. 代码地图（9 649 行 Python + 1 319 行单页 HTML + 11 081 行测试，均按非空行计）

行数为**非空行**（`grep -cv '^[[:space:]]*$'`），这样口径唯一、可以随时重算核对。

```
rag4prefa/
├── config.py        152    pydantic-settings，全部配置以 R4P_ 前缀从 .env 读；含 `allow_server_key_fallback` 公网开关、`fusion`、受控词表四个开关
├── domain.py        119    Chunk / Section / Hit / Filter / JobState，纯数据，无依赖
├── db.py            151    SQLite schema（projects / documents / chunks / chunks_fts / vectors / jobs / meta / embed_cache / ocr_cache）
├── repo.py          974    **全部 SQL 都在这里**，没有别处写 SQL；含索引代次（index_generation）
├── consistency.py   427   **索引一致性**：状态诊断、阻断判定、逐 id 核对（verify）、切换标记
├── concurrency.py   291   **写入协调**：维护锁、写事务串行、工作租约（判断别的进程还活着吗）
├── ingest.py        986    解析 → 分块 → 向量化 → 写索引 → 切事实；含 rebuild_store / reindex；OCR 装配与缓存也在这层
├── chunking.py      144    按标题树切块；chunk_id 内容寻址（同 id 必同正文）
├── tokens.py         40    中英混排的 token 估算（非真实 tokenizer）
├── ask.py           465    混合召回 → 本路标准化融合 → 可选重排 → 流式生成；术语通道与重排查询的拼法也在这层
├── filters.py        88    自然语言问题 → Filter（规则法，可单测，无 LLM 调用）
├── prompts.py        30    系统提示词与引用格式
├── admin.py         159    项目/文档的查询与级联删除；含清空整库（先清索引后清事实）
├── services.py      658    组件装配；Milvus 不可用时回退 numpy；`for_credentials()` 出按用户视图；词表在装配时读一次
├── credentials.py   360   **按用户解析 Key 与对话模型**：请求头 → 凭据池；报错根因分级
├── api.py           781    FastAPI 路由（只做协议转换，不含业务逻辑）；`svc_billable()` 拦匿名额度；含逐文件删除与清空
├── ocr_setup.py      51  **扫描件识别的实际参数**（端点/模型/Key/并发）与"现在能不能用"；免费档已实测否决
├── snapshot.py      171   **语料快照**：从库里导出文档数/片段数/全量 id 与指纹，给评测报告一个可追溯的"当时有什么"
├── processcontrol.py 168 **停服**：netstat 认 PID + `/api/health` 认身份，认不出就不杀
├── cli.py           506    init / serve / stop / doctor / ingest / reindex / rebuild / ask / projects / delete / delete-doc / clear / verify
└── adapters/       2825    embed / llm / ocr / vectors / rerank / term_map / query_expand / parsers（docx + pdf + 共用的标题规则）—— 外部依赖都关在这里
```

`scripts/` 另有 3 397 行（`eval.py` 1 316 + `relabel_eval.py` 183 + `snapshot_corpus.py` 72 +
审计/诊断/抽取脚本），合计非空 13 046 行——`rag4prefa/` 本身是 9 649 行。

解析层三个文件的分工，改动前先看清：

- `parsers/headings.py`：**与格式无关**的判据——正文清洗、中文编号正则、`build_sections`
  （把 `(级别, 文本)` 流折成章节树）。两个解析器共用，否则同一份文档换个格式就换一棵章节树，
  内容寻址的 `chunk_id` 全部对不上。
- `parsers/docx.py`：Word 有结构，判级顺序是「大纲级别 → 标题样式 → 编号正则」。
- `parsers/pdf.py`：pdf 没有结构，只有字号/加粗/坐标，走「统计正文字号 → 按字号与加粗判标题
  → 编号定级」，另含页眉页脚剔除、跨块段落拼接、表格按 `|` 保留。
- `parsers/__init__.py`：`parse_document(path)` 按扩展名分派，未知格式明确报错；
  `parse_document_report()` 额外返回 `ParseReport`（哪些页走了 OCR、哪些页没有文字层）。
- `adapters/ocr.py`：视觉模型识别单页（远端 OpenAI 兼容接口，不装本地模型）+ 两套提示词。
  `vlm` 是给通用视觉模型的 7 条中文规则，`document` 是给文档解析专用模型的原生指令（不带
  system）。它只负责「这一页的图 → Markdown」；「哪几页要识别」「Markdown 怎么折成事件流」
  都在 `pdf.py`；「用哪把 Key、能不能用」在 `ocr_setup.py`，两者不混。

前端只有一个文件 `rag4prefa/web/index.html`（1 319 行，原生 JS，无构建步骤）。
`tests/` 553 项，**全部离线**（FakeEmbedder 确定性假向量 + 本地假 OpenAI 服务 + 临时 Milvus/numpy），
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

18. **关窗口 ≠ 关服务，停服要先认身份再杀**（2026-10-27 现场反馈）。用户点黑窗口右上角
    的 X 后，再双击 `start.bat` 仍然报"端口已在监听"，而"去那个窗口按 Ctrl+C"已经做不
    到了——窗口没了。实测原因：`.venv\Scripts\python.exe` 是**启动器**（1 个线程、6 MB），
    真正监听端口的是它拉起的子进程（`C:\Users\...\python.exe`，16 线程、65 MB）；杀子进程
    后启动器会自行退出，**杀启动器不一定带走子进程**。
    - 判据一"谁在听"只能认 `LISTENING` 且端口**精确相等**（`endswith(":8899")`：`in` 会把
      `:88990` 判进来）。`TIME_WAIT`/`ESTABLISHED` 不占监听端口，停服后立刻看会有一堆
      TIME_WAIT，那是客户端残留。
    - 判据二"是不是本程序"靠 `/api/health` 新增的 `app` 字段。**问不到（None）与答"不是"
      （False）必须分开**：前者可能是卡死的本程序（给 `--force` 出路），后者明确是别家的
      服务（**不能递 `--force`**，那等于劝人误杀）。反过来，JSON 里**没有** `app` 字段
      （旧版本）算 None 而不是 False，否则升级到一半时停不掉旧进程。
    - "答了"和"没答"的区别不是学院派：`taskkill` 不可回滚，误杀别人家的服务比停不掉更糟。

19. **"跳过"必须是一个能被看见的动作**（2026-10-27）。上传路由当年写的是
    `if suffix not in SUPPORTED_SUFFIXES: continue`——用户选了 9 个文件、页面只多出 6 份，
    剩下 3 份**不出现在任何地方**：不是失败、不是跳过，是消失。他只能猜"是不是按类型分了类"。
    现在逐个返回 `{filename, reason}`，任务汇总逐份列出结果。判断标准很简单：
    凡是不入库的文件，都要有一条能被用户看到的原因，否则等于系统悄悄改了用户的选择。

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
| 单文档删除没有公开入口 | 已解决（2026-10-27） | `delete-doc` 命令 + `DELETE /api/projects/{p}/documents/{id}` + 项目管理里逐份删除，见第 10 节第二十五轮 |
| DOCX 嵌套表格可能重复产出 | `_walk()` 已递归遍历单元格子表格，随后又显式遍历 `cell.tables` | 造一份含嵌套表格的真实 docx，断言每段只入库一次 |
| 多轮对话里旧回答的引用角标可能定位到新来源 | 引用抽屉是全局单例，新回答会覆盖内容，旧回答的 `[n]` 处理器仍指向旧卡片 | 每条回答各自保存引用数据，点旧角标时加载该回答自己的来源 |
| "正在检索回答中"在 `meta` 事件就消失 | `meta` 在 LLM 首 token 之前发出，提示会提前闪掉 | 改到第一个 `token` 事件再移除 |
| 公开入口缺"重新解析但不重算 embedding"的能力 | 补展示字段（如页码）也必须 `--force` 重算全部向量，白花钱 | 加 `refresh-metadata`：只重解析并更新 `page` / `section_count`，`chunk_id` 不变时跳过 embedding |
| OCR 没有页级进度 | 24 页要跑 3.5 分钟，期间任务只显示"解析与分块"，看着像卡死 | 把 `_ocr_events` 的完成页数经 `repo.update_job` 写进 `message`（每页一次，别每批一次） |
| OCR 页里"红头文件标题被拆成多行"会长出几个空章节 | 视觉模型把每个物理行都标成标题（实测 250 号文封面变成 3 个 0 字章节）。片段层面无害（空节不出块），但下一块的 `chapter_path` 会挂到"标题的第一行"下 | 连续两个标题之间没有任何正文时合并成一个（要区分"父标题 + 子标题"这种合法情形，别硬合） |
| OCR 页里表格单元格可能被当成标题 | 评分表这类宽表格，模型会把加粗的单元格标成 `#`，于是"建筑形体规则性 a（15分）"成了章节 | 无法只靠文本判据区分；先记为已知局限，等遇到具体影响检索的案例再处理 |
| OCR 结果无法逐页人工复核 | 识别错了只能在库里看到结果，没有"第 N 页原文 vs 识别文本"的对照出口 | 缓存里本来就有页图哈希，可加 `rag4prefa ocr-dump --doc-id X --page N` 导出对照 |

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
- **"不做 OCR"已经过时（2026-10-27）**：没有文字层的页现在可以走 OCR（`R4P_OCR_MODE`，
  默认 `off`）。但两条判据还成立——**默认关闭**时直接报错说明原因，绝不静默入库空文档；
  而且**判"有没有文字层"必须逐页做**，整份判断会让混合型文档的扫描页静默丢失。
  详见第二十六轮与 `adapters/ocr.py` 模块头。
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

### 第二十四轮（2026-10-27 上午）：把"怎么关掉它"变成一条命令

现场反馈：关掉黑窗口后再双击 `start.bat`，仍然提示"端口 8899 已经在监听"，而提示里
说的"去那个窗口按 Ctrl+C"已经不可能执行——窗口没了。查下来不是端口预检误报：

```
PID 21048  F:\...\.venv\Scripts\python.exe -m rag4prefa serve ...   （16 线程，监听 8899）
PID 21352  F:\...\.venv\Scripts\python.exe -m rag4prefa serve ...   （1 线程，父进程）
```

`.venv\Scripts\python.exe` 是**启动器**，真正跑服务的是它拉起的子进程。实验确认：
`taskkill /PID <子> /T /F` 之后启动器自行退出、端口释放；反过来只关窗口/只杀启动器
不保证带走子进程。所以"点 X"这条路本来就不可靠，缺的是**能对已经失联的进程动手的入口**。

本轮新增：

1. `rag4prefa stop --port 8899 [--force]`（`processcontrol.py`）：`netstat -ano` 认
   LISTENING 的 PID → `/api/health` 认身份 → `taskkill /T /F` → **复查端口**。
2. `/api/health` 新增 `app: "rag4prefa"`，作为"端口上是不是本程序"的唯一依据。
3. `stop.bat`（GBK + CRLF，参数与退出码都透传）、`start.bat` 的占位提示改成打印
   PID 与进程名并指向 `stop.bat`。

**验证**（真机，非 mock）：占用时双击 `start.bat` 正确打出 `PID 29772 / python.exe`；
`stop.bat` 停掉后端口释放、`taskkill` 的返回码由 bat 原样传出；在 8899 上起一个自报
`{"app": "someone-else"}` 的假服务，`stop.bat` 拒绝执行且该进程**没有被杀**。
`344 passed`（323 → 344）、ruff 干净。六条变异全部精确变红：去掉 LISTENING 判据、
端口用 `in` 代替 `endswith`、缺 `app` 字段判成 False、`--force` 不生效、refused 报 0、
杀完不复查端口。

---

### 第二十五轮（2026-10-27 上午）：清空整库 + 逐文件删除 + 让"少掉的文件"看得见

用户提出两件事：要一个清空数据库的入口、要在项目管理里删掉某个具体文件（连向量一起）。
顺带问了第三个问题，答案比功能本身更值得记：他往 `F:\project\课题\工程文档` 放了 9 个文件，
项目管理里只有 6 份（两个项目各 3 份），其余 3 份去哪了。

**先查事实，再写代码**。逐个文件试解析（只解析、不入库、不花钱）得到三种不同的原因：

| 文件 | 原因 |
|---|---|
| `南湖…装配式表格.doc` | 老式 OLE 二进制，不是 zip；同目录已有同名 `.docx` 且已入库，这份本来就是重复的 |
| `《关于进一步明确…》沪建建材【2025】250号.pdf` | 可提取字符 **0 个** → 扫描件，不做 OCR |
| `《装配式建筑评价标准》DBJ33T 1165-2024.pdf` | 10899 个字符里汉字占 **0.0%** → 字体缺 ToUnicode，取出来是字形编号 |

**项目不是按文件类型分类的**：`装配式规范` / `装配式计算书` 是导入时 `--project` 指定的
项目名，检索范围里的两个标签就是这两个项目名。所以"9 个文件两个项目"之间没有对应关系，
缺的 3 份与分类无关，纯粹是没进去。

这一轮发现的两个真问题：

1. **上传路由对不支持的后缀直接 `continue`**，连任务消息都不留痕。用户选了 9 个文件、
   页面只多出 6 份，剩下的从结果里消失——不是失败、不是跳过，是**不存在**。现在逐个
   返回 `{filename, reason}`（`explain_unsupported` 给"为什么 + 怎么办"），202 响应与任务
   payload 都带上；任务结束的汇总也改成逐份列出成功/失败/被挡下，不再贴一段 JSON。
2. **清空没有入口，而它是唯一的修复出口**：索引卡在 `unsynced` 或半途重建时，别的破坏性
   动作都会被 `assert_write_ready` 拦住，这时需要的正是"从头来过"。所以 `clear` 刻意
   **不要求"索引自证一致"**，但要求后端身份对得上（认不出手里这个索引是不是库里的那个，
   就可能把别人家的数据擦掉）。

**顺序与判据**（新的不变量，值得记住）：

- 清空走「先清索引、后清事实」，与入库相反。反过来先删归档时，若后端删不掉，归档已经没了、
  `rebuild` 失去恢复源——结果同样是空，但那不叫"清干净了"，叫"数据被悄悄扔掉"。
- 三条完成判据缺一不可：后端条数确认为 0（`clear()` 返回不等于清干净，**读不到也不算 0**）
  + 五张业务表归 0 + `meta` 的模型/维度声明清掉、状态回 ready。删完还核一遍 `data_counts`。
- `meta` 里留下的 `vector_backend` / `vector_backend_id` **不清**：它描述"这个库怎么跑"，
  不是数据；`embed_cache` 也保留（省下次入库的钱）。
- 删单个文件沿用 `delete_document` 已有的锁 + `unsynced:project:doc:<id>` 留痕；
  新加的只是入口与**项目归属校验**：路径里带项目名不是冗余，`doc_id` 属于别的项目时直接
  404，而不是照着 id 把另一边的数据删掉。
- 清空接口要求 body 里 `confirm` 与 `"清空"` 完全一致（400 否则）。前端弹窗保护不了 curl，
  真正的闸在服务端。

**验证**：356 passed（344 → 356）、ruff 干净、前端 `node --check` 通过；四条变异精确变红
（不校验后端条数 / 顺序反了 / 删文档不校验项目归属 / 清空不校验确认词）。真机在 `data/app.db`
的副本上跑通：`delete-doc` 删 13 块 → 66/66 变 53/53、`verify` 三方一致；`clear` 报
`before 2/6/66/66`、`after 全 0`，清完 `health.ok=True`、`issues=[]`。真库 6 份 / 66 块未被触碰。

### 第二十六轮（2026-10-27 下午）：OCR 模块（扫描件入库）

**决定用远端视觉模型，不是本地 OCR**。Tesseract 要装系统二进制 + `chi_sim`，PaddleOCR /
RapidOCR 要落 ONNX 模型，都算"引入本地模型"；更要紧的是它们**只给文字**，不给「这是一级
标题」「这是一张表」——后者正是切块依赖的东西。视觉模型一次调用同时给出文字、层级与表格。

实测（2026-10-27，硅基流动国际站）：

| 模型 | 一页 150dpi | 层级判定 |
| --- | --- | --- |
| `Qwen/Qwen3-VL-8B-Instruct` | 18s | 内容准确，但**漏标 `#`**（见下） |
| `Qwen/Qwen3-VL-32B-Instruct` | 42s | 更差：把正文续行标成了一级标题 |
| 魔搭 `ERNIE-4.5-VL-28B-A3B-PT` | — | `invalid_model` |
| 魔搭 `InternVL3_5-241B-A28B` | — | `user token expired` |

「更大 ≠ 更适合」，与 embedding 那边（0.6B 优于 4B/8B）是同一个结论。默认锁 8B。

**结构**：`adapters/ocr.py` 只做「页图 → Markdown」；「哪几页要识别」和「Markdown 怎么折成
事件流」在 `parsers/pdf.py`；`ingest.py` 负责装配与缓存。`parse_document_report()` 返回
`(sections, ParseReport)`，`parse_document()` 保留原签名（只要章节）。

**四条不能动的判据**：

1. **逐页判"有没有文字层"，不是整份。** 以前整份闸门看全文，混合型文档（封面扫描、
   正文可复制）能过闸，扫描那几页在 `_collect` 里取不到任何东西 → **静默丢页**，任务里
   一句提示都没有。与"不支持的格式静默 continue"是同一类问题，只是发生在页这一级。
   现在 `ParseReport.missing_pages` 会把页码报进任务汇总，开 OCR 则自动补上。
2. **编号定级，`#` 只证明"是不是标题"。** 视觉模型看得见字号与位置，但数 `#` 不可靠；
   编号是文档自带的。判据仍是 `headings.text_heading_level()`，与 docx/pdf 文字层共用一套，
   两种来源的层级才不会各走一套。
3. **标题必须再过 `looks_like_title()`。** 32B 把「独立建筑单体可不实施装配式建筑；」
   （上一段续行，以分号结尾）标成了一级标题；不拦就会长出一棵和正文无关的树。
4. **没标 `#` 的行也要判一次，但只信编号、且长度 ≤ 40 字。** v1 提示词下 8B 在一份 24 页
   公文的第 2、8、9 页**一条 `#` 都没给**，17 636 个字全落进同一节、章节路径全空。
   反过来，一条跨页断在半句的正文（「4. 施工图审查机构应按照…」55 字）会因为"开头是编号
   又没有句末标点"被当成标题——40 字的长度上限就是挡它的（中文标题极少超过 40 字）。

**缓存是正确性前提，不是省钱优化**。视觉模型输出不确定；`chunk_id` 内容寻址，正文差一个字
就是另一批 id。没有 `ocr_cache`（键 = 页图 sha256 + 模型名 + 提示词版本），`--force` 重入库
会把同一份文件变成"另一批 id"的新文档，旧向量成孤儿，而 `verify` 只比 id 集合是否自洽，
查不出来。顺带的好处：100 页里第 87 页失败，重跑只补那一页。

**必须注意的两个实现细节**：

- **同一批里逐字节相同的页图只识别一次**（`_ocr_events` 里按 sha 去重）。空白页、模板页
  渲染出来完全一样，不去重时几个线程会同时"查缓存未命中"，同一张图付几次钱——
  缓存救不了并发下的这一刻。
- **缓存读写要拿 `sqlite_write()`**。识别是多线程的，整进程共用一条 sqlite 连接；
  线程 A 在 INSERT 时线程 B 的 SELECT 会在**同一条连接内**看见未提交、正在写的页，
  实测报 `sqlite3.OperationalError: Could not decode to UTF-8 column 'text'`。
  锁只包两次访问（微秒级），网络调用留在锁外，并发照旧。

**顺带修的一个测试隐患**：`tests/` 里的 `Settings(...)` 会读本机 `.env`。这次把 OCR 打开后，
`test_eval_dataset` 的语料夹具直接调了付费接口，chunk 指纹也因此对不上。现在两份夹具都带
`_env_file=None`（且显式 `ocr_mode="off"`）——"离线全绿"是这套测试的底线，
开发者改 `.env` 不该改变用例行为。

**验证**：24 页真扫描件（上海 250 号文）端到端跑通 —— 23 页 OCR、第 23 页（印章页）识别后
仍无文字并被点名报出，48 节 / 57 块，章节路径到「一、实施范围 >（三）工业建筑类项目」，
内容与原文逐条对上；`verify` 三方一致；检索能召回「（二）公共建筑类项目」那一段。
381 passed、ruff 干净、前端 `node --check` 通过；**10 条变异全部精确变红**
（逐页判据回到整份 / 不查乱码 / 不认 `<EMPTY>` / 标题不看是不是句子 / 层级用 `#` 个数 /
页码恒为 1 / 同图不去重 / 去掉页数上限 / 缓存直通 / 把空结果当没识别过）。

**代价要说清**：一页 16~18 秒、一次调用一份钱，`R4P_OCR_MAX_PAGES`（默认 200）是账单闸门，
超了直接拒绝而不是先跑一半。默认 `off`，开启后缺 Key 会在装配阶段就报错（不是等到第 1 页）。

### 第二十七轮（2026-10-27 傍晚）：OCR 档位（免费 / 付费可切换）

**动机来自一个产品问题**：系统要面向"不愿意付费的普通用户"。OCR 是全流程里唯一按次计费的
环节（embedding 一次入库、LLM 每问一次但很便宜），所以它是"零成本能不能成立"的关键。

**先查证再动手**（这类政策会变，不能凭记忆）：硅基流动已上线两个 **¥0** 的文档解析模型
——`PaddlePaddle/PaddleOCR-VL-1.5`（0.9B，OmniDocBench v1.5 94.5%）与
`deepseek-ai/DeepSeek-OCR`（3B，8K 上下文）；智谱 `glm-4v-flash` 也永久免费（并发 1~2、
高峰 429）。但**免费档放在国内站**，实测现有国际站 Key 打国内站返回
`30014 Token is invalid`——是"站点不认这把 Key"，不是"模型不存在"。

**为什么 OCR 能换模型而 embedding 不能**：embedding 换模型 = 换一套语义坐标，同一集合混两套
向量会让检索"看着正常但不可信"，必须全库唯一。OCR **不产生向量**，它只把像素变成文本，
文本再走同一条切块与向量化流程。真正要防的是「同一份文档两次入库得到不同 `chunk_id`」，
那由 `ocr_cache`（键含模型名 + 提示词版本）挡住，不靠锁死模型。这条写进 `ocr_profiles.py`
模块头，与 `credentials.py` 模块头形成对照。

**两档的分工**（`ocr_profiles.py`）：

| 档位 | 端点 / 模型 | 来源 | 提示词 |
|---|---|---|---|
| `free` | 国内站 `PaddleOCR-VL-1.5`，并发 2 | **代码常量**，不读 `.env` | `document`（原生指令，不带 system） |
| `paid` | `.env` 的 `R4P_OCR_*` | 全部来自 `.env` | `vlm`（那套 7 条中文规则） |

免费档的端点写死在代码里，不是为了省事：那是"这一步零成本"的全部依据（`free-*` 计价项 +
专用模型）。放进 `.env` 迟早被人改成付费模型，界面上却还挂着"免费"两个字。

**四条踩过的坑**：

1. **档位不能写回 `Settings`**。最初想用 `model_post_init` 把免费档的端点覆盖到配置里，
   问题是 `Settings` 只有一份——覆盖之后同进程再切回付费档，就取不回 `.env` 里那个国际站
   地址了。改成「档位解析集中在 `effective(settings)`，`Settings` 始终保留原值」，
   两档配置同时活着。
2. **免费档的 Key 必须单独放**（`R4P_OCR_FREE_API_KEY`），且**不回落**到 `R4P_OCR_API_KEY`。
   回落的结果是国际站 Key 打到国内站 → `30014`，与「Key 被删」同一个错误码，用户会去
   反复重建一把本来就没错的 Key。
3. **用户填的 Key 要落到这一档读的那个字段**。统一写 `ocr_api_key` 的话，免费档的
   `effective()` 读的是 `ocr_free_api_key`，用户填的 Key 静默失效——界面显示"已保存"、
   请求仍在用服务端那把。抽成 `settings_for_ocr()`，`CredentialPool` 与 `verify` 共用。
4. **`user_key()` 的角色查表不能是二分**。原来是
   `creds.embedding_key if role=="embedding" else creds.llm_key`，加第三个角色后
   `else` 分支把 OCR 读成了 LLM——strict 模式查 OCR Key 会读到 LLM 那把，
   "用户没填"被误判成"填了"。改成 `_ROLE_FIELDS` 映射。

**缓存键刻意不一致**：`paid` 保持 `模型名#v2`（旧 `ocr_cache` 全部有效，不重复付费），
`free` 用 `模型名#document-d1`。两档输出格式完全不同，绝不能共用一份缓存——共用的话切档后
会读到"格式对不上的旧文本"，而 `verify` 查不出来（它只比 id 集合自洽，不比内容是不是这一版）。

**接口与界面**：新增 `/api/ocr-profiles`；`/api/credentials/verify` 多返回 `ocr` 一项
（缺 Key 报 `ok:false, source:missing`，不假装通过）；`health.ocr` 报
`profile / label / free / preset / key_source`；页头加一个 `OCR 免费档 / 付费档 / 关闭` 的
标签（点它进设置），因为它决定"这次要花多少钱"，不该藏在抽屉里。新增请求头
`X-R4P-OCR-Key`、`X-R4P-OCR-Profile`，与既有凭据同一条规矩：浏览器保管、服务端不落库。

**验证**：411 passed、ruff 干净、前端 `node --check` 通过；新增 `test_ocr_profiles.py` 30 项，
**6 条变异全部精确变红**（免费档回落到付费 Key / 用户 Key 一律写付费字段 / 统一缓存键格式 /
给专用模型也套 7 条规则 / 并发不按档位取 / 角色查表退化成二分）。

**当时的诚实标注（当天晚上就被推翻）**：免费档尚未在本机跑通过一次真实识别。拿到国内站 Key
之后的第一件事就是拿真实扫描件试——结论见下一轮。

### 第二十八轮（2026-10-27 夜）：免费档实测否决，改为"不配置就不处理扫描件"

**实测对象**：同一份 24 页扫描件（上海 250 号文），同一条代码路径，只换模型。

| | 付费 `Qwen3-VL-8B` | 免费 `PaddleOCR-VL-1.5` |
|---|---|---|
| 24 页 | 3 分 39 秒（4 并发） | 17 分钟只完成 **8/24**（2 并发） |
| 单页 | 15~18 秒 | 普通页 55 秒；公式页 **250 秒** |
| 结果 | 23 页识别 + 1 页空白，48 节 / 57 块 | 8 页里 **2 页伪造内容** |

两页坏得可复现：第 3 页是**纯中文正文**（无表格无公式），整页被编成英文的
"The area of a circle is given by A = πr²"，**150 / 120 / 100 dpi 全部复现**；第 8 页退化成
`1. 1. 2. 3. 4. …` 的数字重复并顶到 `max_tokens`。换 `deepseek-ai/DeepSeek-OCR` 更糟：
一页满屏 `}}}]`，一页 0.9 秒只回 3 个字符。

**判据**：不是"准不准"，而是**伪造正文会被切块、写向量、被引用成答案，而且不报错**。
RAG 里这是最坏的一类失败——不崩溃，只是胡说。降 dpi、改提示词、压 `max_tokens` 都无效
（三档分辨率输出同一段伪造内容），没有"调一调还能用"的空间。

**产品决策（用户拍板）**：扫描件只走付费视觉模型；没配置就**引导去配**，用户不配就明确
"无法处理扫描件"，绝不静默跳过。

**代码变更**：
- `ocr_profiles.py` → `ocr_setup.py`，只剩一条路径（`effective()` 合并服务端 Key 与用户 Key，
  给出 `ready`）。删掉 `OCR_PROFILES` / `resolve_profile` / `key_for` / `PromptPreset` /
  `DOCUMENT_PRESET` / `ocr_preset` / `ocr_profile` / `ocr_free_api_key`。
- 删 `GET /api/ocr-profiles`、请求头 `X-R4P-OCR-Profile`、CLI `--ocr-profile`；前端去掉档位
  下拉，页头标签改为 `OCR 就绪 · 模型` / `OCR 未配置` / `OCR 关闭` 三态。
- **`OCR_SETUP_HINT = "【需要配置视觉模型】"`**：缺 Key 与扫描件被拒的消息都以它开头；
  `/api/health` 多一个 `ocr.ready`；任务输出命中该前缀时，页面弹出「配置视觉模型」按钮直通设置。
- 缓存键**刻意保持** `模型名#v2`——已有 `ocr_cache` 全部有效，不重复付费（有用例守着）。

**验证**：392 passed、ruff 干净、`node --check` 通过；`test_ocr_setup.py` 11 项，
**6 条变异全部精确变红**（`ready` 不看 Key / 两处文案去掉前缀 / `health` 忽略 `mode=off` /
缓存键加后缀 / 前端提示语改字）。最后一条是"两个进程共享常量"的守卫——改一头忘了另一头，
表现是"扫描件被拒了但引导不弹"。

**教训**：①"能调通"≠"能用"，免费/便宜档必须拿**真实文档**逐页看输出，不能只看 200 OK；
②一次跑不完先别认定卡死，逐页取证才知道是 250 秒一页；③`}}}]`、数字递增、跳语种的英文公式
是三类不同的退化，共用同一个判据拦不住，只能靠"不配就不用"。

---

### 第二十九轮（2026-10-27 夜）：各付各的——不代付时扫描件必须进不来

**起因**：用户把这套东西交给别人用，问"我的 Key 能不能直接给另一个用户"。查下来发现
**`R4P_ALLOW_SERVER_KEY_FALLBACK=false` 对 OCR 根本没生效**：`/api/chat` 按
`("embedding","llm")` 逐个判、`/api/search` 判 embedding，而**上传接口的 `need` 只有
`("embedding",)`**——`KEY_ROLES` 里明明有 `ocr` 这个角色，却没有一个调用点用它。

于是打开"不代付"之后，别人不填视觉模型 Key 也能上传扫描件，服务端照用 `.env` 那把识别，
按页计费记在主人账上；页头还显示「OCR 就绪」。整条链路里唯一真正花钱的地方，恰好是没闸门的。

**不能简单地把 `"ocr"` 加进上传接口的 `need`**：上传时还没解析，系统不知道这批文件里有没有
扫描件；硬要求会让只传 docx 的人也被要求填视觉 Key。闸门必须落在"这一页真的需要识别"。

**四处改动，缺一条就漏**：

1. `credentials.settings_for_ocr()` 从两规则变三规则：填了→用他的；没填且**接受代付**→
   沿用服务端；没填且**不代付**→**清空**（下游 `effective().ready` 自然为假）。
2. `ingest._build_ocr()` **不再在构造期校验 Key**。它对每份文件都会调用（是
   `parse_document_report` 的实参），包括 docx 和带文字层的 pdf——它们从不调用识别函数，
   却会跟着整份失败。判据落到 `_cached_ocr` 的第一次"未命中缓存"。
3. `_cached_ocr()` 改成**先算缓存键查表、命中就直接返回**，未命中才构造识别器。
   两个结果都是要的：缓存命中**不需要任何 Key**（结果是库里现成的，凭什么还要求付费能力）；
   顺便让 `cache_key(settings)` 从 `VlmOcr` 上独立出来——算缓存键本来就不需要 Key。
4. `services._ocr_status()` 的 `ready` 加 `payable = owned or allow_server_key_fallback`。
   为什么不能只靠第 1 条：没带任何 `X-R4P-*` 头的**根视图就是服务端主人自己**，
   `for_credentials(None)` 返回的 settings 原样带着 `.env` 那把，页头会谎报"就绪"。

**顺带修的两处**：

- `VlmOcr.close()` 的 docstring 写着"用完要还"，但**从来没人调用**——每份文件泄漏一个
  httpx 连接池。延迟构造后更隐蔽，于是给 `OcrOptions` 加 `close` 字段（默认无操作，
  假后端不必实现），`_ingest_locked` 用 `try/finally` 包住解析。
- `OcrError` 的正文是写给普通用户的，`_ingest_locked` 不再给它加 `OcrError: ` 前缀——
  那既让前端 `startsWith(OCR_SETUP_HINT)` 失效，也对用户没有信息量。其他异常照旧留类型名。

**文案分两种人写**（`VlmOcr._missing_message`）：不代付时说"填入你自己的视觉模型 Key，
**扫描件无法入库**"；单机演示说"服务端还没配好（缺少 X / Y）"。对访客说"服务端没配好"，
他会去找管理员，而正确的动作是打开设置。

**配置定位改为各付各的**：`.env` 与 `.env.example` 都写 `R4P_ALLOW_SERVER_KEY_FALLBACK=false`，
注释说明"多一个人访问就该是 false，true 只适合单机自用"。前端跟着分两种状态：
页头 `OCR 待配置`（该你填）vs `OCR 未配置`（服务端没配），`who` 在严格模式下显示"未填 Key"
而不是"公共密钥"——后者在严格模式里是句假话。

**测试隔离的老问题被这次撞出来了**：把 `.env` 改成 false 之后，7 条与开关毫无关系的用例
集体变红。根因是 `conftest` 的 autouse fixture **只在收尾恢复 `os.environ`**，每个用例
执行时仍带着本机 `.env`；`test_real_sdk_chain` / `test_api_credentials` 里的 `Settings()`
更是直接读 `.env` 文件。两头都修：fixture 改成**前后各恢复一次**，那两处 `Settings(...)`
补 `_env_file=None`。

**验证**：399 passed（新增 7 条）、`ruff check` 干净、`node --check` 通过；
4 条变异全部精确变红——`settings_for_ocr` 不清空 / `_ocr_status` 忽略开关 /
查缓存前先构造识别器 / 构造期校验 Key。每条都对应一个会花钱或会误伤的场景。

**教训**：①"配置项存在"≠"每个入口都执行了它"——`KEY_ROLES` 定义了角色，不等于有调用点
用它，接线要 grep 到底；②闸门要放在**信息最全的那一层**（知道这一页要不要识别），
放在入口只能靠猜；③测试读本机配置是慢性病，只在收尾恢复环境变量治不好。

---

### 第三十轮（2026-10-27 深夜）：外部复审 P2——凭据是三个角色，判据只写了两个

**起因**：外部复审（对着 `3a18f14`）认可上一轮把计费闸门放到"真正识别页"这一层，但指出
**前端 `saveCreds()` 的保存判据只判了 embedding 与 LLM**：

```js
if (!result.embedding.ok && !result.llm.ok) { /* 不保存 */ }
```

于是"只填视觉模型 Key、另两栏留空"的人会看到三项结果、其中一项通过，却被判"未保存"——
什么都没存下来。凭据是**分角色**的（扫描件要 embedding + ocr，问答要 embedding + llm），
任何一项通过都说明"这个用户名下至少能用一件事"，挡着不存只会让人重复填同一把 Key。

**改动**：

1. `saveCreds` 的判据改为 `[embedding, llm, ocr].some(ok)`：至少一项通过就存；三项全不通
   仍不存（存下来只会让之后每次请求都 401）。
2. 保存成功的提示语列出**还没通过的那几项**（`lacks` → `lackNote`），并说明补齐后重新验。
   只说"已保存"会让人以为配全了，然后在提问撞上 401 的那一刻才发现另一栏是空的。
3. 设置面板文案纠错：原写"留空 = 用服务端配好的那把"，在默认的"各付各的"下是假话，
   改为"留空就是扫描件无法入库；扫描件还要上面那把 Embedding Key，两把缺一不可"。
4. README 的角色表把"扫描件入库"从"视觉模型 Key"改成 **Embedding + 视觉模型**，
   并写清"只填 `X-R4P-OCR-Key` 不够"：上传入口先要 embedding，这不是 bug。
5. 新增「部署边界」小节：三条销毁接口（`/api/admin/clear`、删项目、删文件）**没有身份校验**，
   `confirm: "清空"` 只防手滑；开放公网前必须挡在鉴权后面。额度那层已收口，删库那层没有。
6. `.env.example` 删掉"免费档按内置 2 并发跑，不读这一行"的过期说明（免费档已整体移除）。

**测试**：新增 `tests/test_web_credentials.py`（3 条），按函数名从 `index.html` 里抽出
`saveCreds` 正文再断言——这类错误 `pytest` 看不见、`node --check` 也完全正确（语法没错）。

**变异验证踩到的坑值得记**：第一版断言写的是 `"lacks" in body`，把 `lackNote` 换成常量空串
照样通过——**断言选错了对象**（测"变量存在"而不是"它被用上"）。改成三条：算了吗
（`lacks.length`）、名字拼进去了吗（`lacks.join`）、最终那句提示语里有没有 `${lackNote}`。
四条变异随后全部变红：判据回退成两个 / `some` 改 `every` / 汇总退化成常量 / 算了却不拼进提示语。

**验证**：402 passed（+3）、`ruff check` 干净、`node --check` 通过。

**没做的一件事**：管理接口授权仍缺（复审列为"有条件阻断"）。当前定位是单机/可信内网，
README 已把边界写死；要真正多人使用，得连登录与权限一起做，不做半套。

---

### 第三十一轮（2026-10-28 上午）：复审 P2——凭据"存了"不等于"用了"

**起因**：外部复审对 `d437e7c` 提了三条 P2。核心那条不是文案：**验证失败的 Key 仍被
整体写进 localStorage，又原样随请求发出去**。于是"设置里刚告诉你这把不通"与"真跑时
照样把它送出去、在异步入库任务里失败"同时成立。另两条：`clearCreds()` 无论什么部署都说
"后续请求使用服务端公共 Key"（各付各的部署里这是假话，与本文件 `syncWho()` 自相矛盾）；
`saveCreds()` 开头还留着"不填 Key 也合法（走公共额度）"的旧默认。

**改动**：

1. 把三条判据抽成**可执行块**（`index.html` 里 `CRED-LOGIC-START/END` 之间）：
   `roleEnabled` / `pendingRoles` / `credentialHeaders` / `shouldSave` /
   `clearedMessage` / `noUserMessage`。抽出来不是为了好看，是为了能**在 Node 里真跑**。
2. 存储多一个 `verified: {embedding, llm, ocr}`：填错的 Key 仍然存着（免得用户重打），
   但 `credentialHeaders()` 不发它；用户列表标"（Embedding、视觉模型未启用）"。
   **旧数据没有 `verified` 按"启用"处理**——升级一次就把人家一直在用的 Key 停掉，
   比原 bug 更严重。
3. `clearCreds()` / 无当前用户时的提示按 `serverKeyFallback` 分支。
4. `saveCreds()` 的注释改成"留空在单机演示下走公共额度、在各付各的部署下等于用不了"，
   不再把一种部署的默认写成普遍规则。

**测试换了一层**：`tests/test_web_credentials.py` 从"正则匹配源码"改成"抽出标记块、
在 Node 里执行、断言行为"。没装 Node 整段 skip（`shutil.which("node")`），其余用例仍离线。

**变异**：12 条全红，其中两条是自我纠错换来的——
① `some -> every` 一开始没抓住，因为"全不通不保存"这一半还留在调用点里，行为测试够不着；
把它也搬进块里（`shouldSave`）才覆盖到。
② `saveCreds` 不写 `verified` 时，所有行为测试照绿（缺字段等于"永远启用"），所以另加一条
**接线断言**：`verified:` 必须写在保存里、`shouldSave(result)` 必须被调用。

**验证**：410 passed（+8）、`ruff check` 干净、`node --check` 通过、12/12 变异变红。

**仍未做**：管理接口授权（P1）。它现在是唯一一条"代码没防、只有文档提示"的缺口，
而且没有不付代价的做法——开了校验，本机从界面删文档也要先配口令。

---

### 第三十二轮（2026-10-28）：管理口令——"谁能删库"这一层补上了

**起因**：复审连续两轮把 P1 放在"有条件阻断"：`POST /api/admin/clear`、
`DELETE /api/projects/{项目}`、`DELETE /api/projects/{项目}/documents/{id}` 三条
不可撤销的入口**没有任何身份校验**。`confirm: "清空"` 只防手滑；额度那层早就收口了，
"谁能删"这层一直空着。上一轮我只写了 README 边界，用户拍板要补。

**规则（`api.require_admin`，写在 `config.admin_token` 的注释里）**：

| 部署 | 口令 | 结果 |
|---|---|---|
| 单机演示 `fallback=true` | 没配 | 放行——打开页面的人就是主人 |
| 各付各的 `fallback=false` | 没配 | **503**，说清去哪儿配。宁可删不了，也不让访客清库 |
| 任意 | 配了 | 缺头/错头 403（`hmac.compare_digest` 定长比较） |

**为什么单机演示不强制**：本地开发每删一个文件都要先去改 `.env` 再重启，安全收益为零、
摩擦是实的；而"各付各的"这个开关本身就是"有别人在访问"的声明，用它可以当判据，
不必新增一个"我是不是多人部署"的开关。

**顺序**：口令校验在 `confirm` 之前。反过来的话，没权限的人会从 `400 确认词不对` 里读出
"只要打对字就能删"——这条是变异测试逼出来的：我第一版用例用正确的确认词去测，
两种情况都回 403，根本分不出先后。

**前端**：`X-R4P-Admin-Token` 由 `adminHeader()` 生成，**在取"当前用户"之前**就放进
`authHeaders()`——晚一步就会变成"清掉自己的 Key 就再也删不掉文件"。口令存在
`credStore().admin`（部署级，不跟用户走），设置面板单列一栏与一个"保存管理口令"按钮，
刻意**不塞进「保存并验证」**：那个按钮会返回"验证失败、未保存"，而口令其实存了，
两句话同时成立会让人怀疑自己。

**测试**：`tests/test_admin_auth.py`（10 条，四种部署组合 × 三条路由）、
`test_web_credentials.py`（+2 条：口令头与当前用户无关、接线顺序）。10 条变异全红。

**落地注意**：本项目 `.env` 是 `fallback=false`，所以**本机也必须配 `R4P_ADMIN_TOKEN`**，
否则从界面删东西会被 503 挡住（CLI 不受影响）。已在 `.env` 里生成了一个随机串。

**第一次真用就撞上了（同日补）**：用户点「清空库」被拒。实测返回 403——服务端要求口令，
而浏览器里那份口令是空的。根因有两层，第二层更容易漏：

1. 口令栏在设置面板**最下面**，跟三把 Key 不是一个按钮，用户没注意到；
2. **页面被浏览器缓存了**。`/` 是 `FileResponse` 且没设缓存头，改版前打开的标签页刷新
   也未必拿到新 HTML——那种页面上**根本没有「管理口令」这一栏**，于是"被拒绝"看起来像
   后端故障，实际上前端连填的地方都没有。

对应两处改法：`require_admin` 把"没带"与"带错"分开说（前者说去哪儿填、怎么保存，
后者指向 `.env` 里那份值；原来合成一句"管理口令不对"，会让**什么都没填**的人去核对自己
填了什么）；`index()` 加 `Cache-Control: no-cache`（带 ETag，没改动仍 304，不增加传输）。
`tests/test_admin_auth.py` +2 条：两句文案必须不同、首页响应头必须禁缓存。变异（把两句
合成一句）变红。435 passed / `ruff check` 干净。

---

### 第三十三轮（2026-10-28）：评测口径——报告必须自带"这批数字是在什么语料上算的"

**起因**：课题开题要引用"公众问法下检索性能下降约 20%"。复审提出评测报告的版本口径可疑，
核查之后发现比"题量不一致"更严重——**报告里没有记录语料身份**。

**核查结果（这些都是当时仓库里的实际状态）**：

| 项 | 实际 | 问题 |
|---|---|---|
| `eval/corpus.json` | 6 份 / 66 块 | 停在 250 号文、DBJ33/T 1165、杭建工业 1 号入库**之前** |
| 运行时库 | 9 份 / 152 块 | 多出的 86 块会参与检索并改变排名 |
| `dataset.jsonl` | **25 题** | 但 `baseline-30.md` 自称 30 题，两份报告都写同一个数据集路径 |
| `baseline-30` vs `-new` | — | **结论方向相反**：前者"稀疏 0.950 > 混合 0.900"，后者"混合 0.960 > 稀疏 0.920" |
| `paraphrase-10` vs `-new` | 各 10 题 | dense 先 0.500 后 0.700，中间无模型/参数改动记录 |

**修法（四件事，缺一件就还是"看起来正常"）**：

1. **`rag4prefa/snapshot.py`**：从数据库导出语料快照（文档数、片段数、全量 `chunk_id`、整体
   `ids_sha`）。**必须从库里导，不能重新解析**——补语料时旧源文件被全删（2026-10-28 实测，
   `corpus.json` 里 6 份一份不剩），重新解析既拿不回旧文件，也可能因解析器改进算出不同的 id。
2. **`scripts/eval.py`** 报告头固定写六项：坐标系、语料快照与 `ids_sha`、数据集哈希与题数、
   生成时间、代码提交与工作区是否干净。快照与当前库不一致时打「**语料已变化…不可直接比较**」。
3. **`--save` 命名** `<数据集>-<日期>-<ids_sha 前 8 位>.md`。刻意不用 `-new`：`baseline-new`
   用了两个月就不再新，而它指向的语料已经换过两轮。
4. **`archive/`** 归档 4 份旧报告，头部写清为什么不可复现。保留是为了留下"当时的结论"。

**顺手补上的一个静默失效**：源文件删光之后，`test_eval_dataset.py` 的两条用例由"通过"变成
**跳过**——它守的正是"解析器一改 gold 就静默失效"。以 skip 姿态消失的守护最难发现，
看一眼摘要只会以为"CI 环境没有语料"。补 `test_dataset_gold_survives_without_source_files`：
拿快照里的全量 id 离线判定 gold 是否还在，不需要源文件。

**语料从 66 块扩到 152 块之后，同一批题的数字变了（同一快照、同一天、同参数）**：

| 召回方式 | 基础 25 题 | 口语 10 题 |
|---|---|---|
| dense | 0.840（原 0.960） | **0.400**（原 0.700） |
| sparse | 0.880（原 0.920） | 0.700（原 0.800） |
| hybrid | 0.880（原 0.960） | 0.600（原 0.800） |
| rerank | 0.840（原 0.920） | 0.800（原 0.800） |

**结论**：旧数字不止"口径不清"，而是**系统性偏乐观**——66 块的候选池太小，干扰项不足。
开题引用必须用新表，并且要写明口语集 `n=10`、单题权重 0.1，只能看趋势。

**教训**：①"报告写了参数"不等于"报告能复现"，少了**语料身份**这一项，换过语料的两份报告
长得一模一样；②**跳过**是最隐蔽的失败形态，比红更危险——它不动声色地把守护撤走了；
③`chunk_id` 内容寻址意味着**新增文档不改旧文档的 id**，所以补语料不会让旧 gold 失效，
但会让按文字匹配的 `gold_snippets` 解析出更大的集合，Recall 分母跟着变——所以顺序必须是
「先定语料 → 再算 gold → 最后出基线」，不能反过来。

**验证**：433 passed（+12 快照与渲染、+2 离线 gold 校验，3 项需本地语料的仍跳过）、
`ruff check` 干净、8 条变异全红（指纹跟着行序、变化检测漏语料、gold 检测返空、
同名文档互相覆盖、数据集哈希漏注释、坏快照抛异常、警告不渲染、题数不渲染）。

---

### 第三十四轮（2026-10-28）：语料换血——md 能进了，水印型扫描件被挡住了

**背景**：用户清空旧库、按 `01_省级技术文件 / 02_国家行业标准 / 03_浙江省政策文件 /
04_技术应用与案例 / 05_国家政策文件 / 06_十五五与2026年最新政策` 六类重备了语料
（83 份，且仍在补），要求继续开发。做之前先立了一条规矩：**入库之前先体检**。

**`scripts/corpus_audit.py`（新）**：只读、不调 embedding、不写库，逐份报告
页数 / 章节 / 片段 / token / 缺页，并按课题六类主题词算"对题片段占比"。它是
"花钱之前唯一能看清这批文件会变成什么"的地方——解析失败与格式不支持在入库时
才会暴露，而那时已经付过一轮解析的代价。

**md 解析器（新，`adapters/parsers/markdown.py`）**：语料里 37 份 md 此前**一律被拒**
（`_UNSUPPORTED_HINTS` 写着"Markdown 暂无解析器"），而拒绝理由并不成立——md 的标题
是显式语法。两周内的取舍：

| 事项 | 决定 | 理由 |
|---|---|---|
| 标题来源 | `#` 优先，没有则退回 `text_heading_level`（与 pdf 共用） | 政策原文常无 `#`，靠「一、」「（一）」分节 |
| `#` 处理 | **剥掉再当标题** | 标题进章节路径、路径进 `chunk_id`，留着 `#` 会让整份 id 变 |
| 清洗范围 | 只做"整行"级：网页模板行 + 日期/流水号；行内去 `**`、反引号、图片语法 | 判错的代价是正文消失，且不可见 |
| 不做 | 智能去导航、表格结构化、脚注 | 都是结构性重建，收益不在检索上 |
| 编码 | UTF-8(BOM) → GBK → replace | 抓取工具留下的 GBK 很常见，整份拒收会连可用正文一起扔 |

**水印型扫描件（`_page_is_scanned`，本轮最重要的发现）**：住建部「信息公开浏览专用」
的 pdf 是整页扫描图 + 一行水印「住房城乡建设部信息公开　浏览专用」。一页 **37 个字符**、
全是汉字、字符数不为 0——字符数闸门与可读性闸门**一道都拦不住**。实测 11 份真实文件：

| 类别 | 页数 | 每页字符中位 | 图片合计覆盖中位 |
|---|---|---|---|
| 水印型（JGJ/T 491、整体厨房、整体卫生间、模数协调、尺寸协调、GB 50210、GB 55038…） | 25~98 | **37（恒定）** | **1.00** |
| 正常（设计导则 28 页 / X1 61 页 / 验收规定 42 页 / A2 图集 72 页 / A1 示例 120 页） | 28~120 | 126~820 | **0.00** |

判据因此只能看版面：**图片合计覆盖 ≥ 0.6 且文字 ≤ 120 字**。三个实测细节都进了用例：
①水印页由**多张图拼成**（一页 8 张、单张仅 15%），判"单张最大覆盖"会全漏；
②图集里的图是矢量绘制（drawings），不进这一项，所以 72 页图集不会被误判；
③整页图 + 大量文字（背景图上盖了文字层）按正文处理——先看字数再量几何，否则白花识别费。
"每页都是水印"与"每页都是乱码"要报**两句不同的话**：前者是"这版本来就是浏览专用"，
后者是"换一份可复制版本"。拿 `visible` 反推类型会把 2642 字的乱码报告成水印。

**入库与清理（真实数字）**：64 份进批（8 份水印型当时按旧闸门"成功入库"了假内容）、
1270 片段；随后删掉 **8 份假内容**（水印）与 **4 份工作笔记**（`README.md`、
`待下载清单.md`、`问题记录.md`、`政府信息公开申请书…`——它们混在语料目录里，
讲的是"资料库自己"，被问政策问题时会被召回）。最终 **52 份 / 1210 片段 / 1210 向量**，
`verify` 四项全 0、`index_state=ready`。

**教训**：
1. **"有文字"不等于"有正文"**。三条闸门（字符数、汉字占比、乱码）都是"文字层质量"的
   判据，水印型三条全过——它检验的是"这一层字像不像人写的"，而问题在于"这层字是水印"。
   新维度只能从版面上量。
2. **入库前体检**。13 份进不去（8 水印 + 3 扫描 + 2 缺 ToUnicode）是在体检报告里一次看全的；
   没有它，这些会以"入库成功"的形态留在库里。
3. 语料目录里放工作笔记是常态。目录递归不会替你分辨"这是资料"还是"这是关于资料的笔记"。

**验证**：447 passed（+8 markdown、+4 水印，3 项需本地语料的仍跳过）、`ruff check` 干净、
11 条变异全红（md：日期判据、`#` 未剥、模板标签、front matter、GBK、代码块保护；
水印：单张最大覆盖、整份只看字符数、字数短路、阈值失效）。

---

### 第三十五轮（2026-10-28）：语料清单过期之后——"守护以跳过的姿态消失"的三种形态

**起因**：重建评测集时 `test_snapshot.py::test_snapshot_agrees_with_manifest_on_shared_documents`
变红，报「清单与快照没有任何共同文档，这份用例事实上没有校验任何东西」。

**根因是三层，一层比一层深**：

| 层 | 现象 | 真正的问题 |
|---|---|---|
| 1 | `eval/corpus.json`（旧 6 份 / 66 块）与新语料零交集 | 清单没有随语料换血重建，"标注时的语料"这句话变成一句没人核的声明 |
| 2 | `test_eval_dataset.py` 三条用例**整体跳过** | 语料重排成 `文献资料/0X_分类/` 之后，`CORPUS_DIR / filename` 一条都拼不出来，而"找不到"被读成"本机没有这套语料" |
| 3 | 把定位修好后，`test_corpus_matches_manifest` 变红：D5 的片段数 32 → 31 | 库里那一块是**入库时开着 OCR** 认出来的（第 1 页是整页扫描图，只有印章）；纯离线解析本就不可能复现 |

第 3 层是关键发现：**"重新解析再比指纹"这件事，在用过 OCR 的文档上证明不了任何东西**。
全量 52 份重解析实测 106 秒，其中 5 份有 OCR 页、id 必然不同（A1 设计示例 105 → 68 块，
一份就 63 秒）。继续按"全量重解析"守，代价是每次 `pytest` 多两分钟，换来的却是一个
永远红或永远跳过的判据。

**改法（三份记录各答一个问题）**：

| 文件 | 来源 | 回答什么 | 谁在读 |
|---|---|---|---|
| `eval/corpus.json` | 数据库，`build_manifest.py` | 标注时那批语料是什么 | `relabel_eval.py`（临时库）、`test_snapshot.py`（与快照交叉校验） |
| `eval/corpus-snapshot.json` | 数据库，`snapshot_corpus.py` | 上次出报告时库里有什么（全量 id） | `eval.py`（判漂移）、`test_eval_dataset.py`（离线判 gold 过期） |
| `eval/parser-canary.json` | **源文件**，`build_manifest.py --canary` | 当前解析器对 7 份样本输出什么 | `test_eval_dataset.py`（解析器没变） |

- **`scripts/build_manifest.py`（新）**：从库导出清单（52 份 / 1210 块）。按**原始文件名**
  自己遍历 `documents`，不套快照那套 `名字 [项目]` 的键——清单是拿去拼 `CORPUS_DIR / 名字` 的，
  套错了会得到一条指向不存在文件的死路径。同名文件直接拒绝（这种格式表达不了，与其悄悄丢
  一份不如要求改名），0 块文档也拒绝（那说明解析失败，记进清单等于把失败记成"语料"）。
- **`snapshot.locate_corpus_file()`（新）**：按文件名在整棵树里找，找不到返回 `None`（CI 上
  没语料是常态），同名多份抛 `LookupError`。实现刻意不用 `rglob(filename)`：文件名里的
  `[` `]` 会被当通配符，`标准 [2024版].pdf` 能命中 `标准 2.pdf`，于是"找到"和"找错"分不出来。
- **解析器金丝雀（7 份 / 466 块 / 约 3 秒）**：按"格式 × 结构"取样本，不按"哪份最常被问到"
  （X1 长多级编号 pdf、01/29 长短公文、A6 docx、Z3/23 政策 md、17 标准信息卡）。它报的是
  "解析器变了没有"——定位靠快照与 `gold_missing`，两件事分开。
- **覆盖面用例**：金丝雀必须覆盖每种**解析器**（`.md` 与 `.markdown` 算一族，同一个函数不必
  验两遍），且不少于 5 份。这条挡的是"它红了就把红的那条删掉"——删到只剩 md 就永远绿着。

**顺带修的**：`relabel_eval.py` 临时库模式同样按文件名定位（否则分类后必然报"语料缺文件"）；
`pdf.py` 里上一轮留下的重复 `_ensure_readable(raw)`。

**教训**：
1. **三种"守护消失"的姿态，跳过是最难发现的**。变红会有人看；跳过只在摘要里多一个 `s`。
   所以凡是"缺文件就 skip"的地方，skip 的理由要写明"此刻丢的是哪道校验"。
2. **判据的成本要算进去**。106 秒的全量重解析不是"更严格"，而是"迟早被 `-x` 跳过的严格"；
   7 份 3 秒的金丝雀能天天跑。广度换不来检出率，样本的代表性才换得来。
3. **"重新解析比指纹"这条判据有适用边界**：它只能证明"纯文字层解析没变"。库里用过 OCR 的
   文档，离线复现不出同样的 id——这时不是发现了解析器回归，而是判据选错了对象。
4. **清单过期不会自己喊**。`corpus.json` 停在旧语料时，所有依赖它的用例都"照常通过"或
   "整体跳过"，只有那句 `assert shared` 把它点出来了。文件与语料同寿这件事，得有人守。

**验证**：454 passed（72 秒，其中金丝雀约 6 秒）、`ruff check` 干净、5 条变异全红
（清单指纹与快照不符、金丝雀片段数、金丝雀 id 指纹、定位退回扁平拼路径、文件名当通配符）。

---

### 第三十六轮（2026-10-28）：第一版基线——年份全空、融合白做、重排失灵

**背景**：语料换血（52 份 / 1210 片段）与评测集重建（31 / 15 / 5）都到位之后，跑第一版基线。
跑的过程里撞到两个"数据没错、但结果不可用"的问题，两个都值得单独记。

#### 1. `doc_year` 全为 NULL：年份是过滤条件，不是展示字段

第一版基线只有 **29/31** 题有效，两条题报「1 个 gold_chunk_id 不存在或不在本题筛选范围内」。
查下去不是标注错，是 **52 份文档的 `doc_year` 全是 NULL**：

- 问句里出现年份时 `ask.build_filter` 会解析出 `year_from=year_to=2025`；
- SQL 侧是 `d.doc_year >= 2025 AND d.doc_year <= 2025`，**NULL 一条都过不去**；
- 于是"2025 年补贴包含什么"这类问题，对着一个装着 2025 年文件的库，可以召回为空，
  而界面上完全正常。

`scripts/backfill_doc_year.py`（新）补了 50 份，判据顺序是**文件名年份 → 完整发文日期
（`20XX 年 X 月 X 日`）→ 来源 URL 里的日期**，都没有就留 NULL：

| 判据 | 份数 | 说明 |
|---|---|---|
| 文件名 | 48 | 取**最后**一个四位年份（`..._2021vs2025.md` 最后那个才是讲的版本） |
| 完整发文日期 | 2 | 03（2024 年4 月30 日印发）、26（`来源：…/art/2024/4/30/`） |
| 留 NULL | 2 | 05/06 两份既无日期也无文号，正文只在引用《…行动方案（2023—2027 年）》 |

**"正文里第一个年份"这条判据是错的**，实测把 03 判成 2023（真值 2024）、把 05/06 判成 2027
（那是引用目标年）。政策文本开头常写目标年，这个坑与 PDF 那次的"字符数不等于正文"同类：
**看着有依据，依据指向的不是同一个东西。**

留 NULL 是刻意的取舍：带年份的问法召回不到那两份，但写一个猜的年份会让过滤悄悄收窄，
而且没有任何提示。补年份走的是入库那条路（`ingest_file(doc_year=…)`），不是 UPDATE——
年份同时存在于 `documents`、`vectors` 归档与后端标量列三处。

#### 2. 四行对照分不开"融合"与"重排"

第一版只有 `dense / sparse / hybrid / rerank` 四行，而 `rerank` 恒等于 `hybrid` 再重排。
口语集上看到 `dense 0.467 > hybrid 0.333 = rerank 0.333`，只能得出"重排加融合等于白做"——
**分不清是融合把候选池拉低了，还是重排本身没用**。这两件事的处置完全不同。

加了第五行 `dense_rerank`（纯向量 + 重排，候选池深度与 `rerank` 取同一个
`_hybrid_pool_size`），并把分派改成显式表 `_HANDLERS`（if/elif 的兜底分支会让"加了方式
忘了实现"变成"多一行与 rerank 一模一样的数字"）。第一版基线因此要重跑。

#### 第一版基线（52 份 / 1210 片段，`ids_sha fe8e22a230713556`，`4e0b340` 之后）

| 召回方式 | 基础 31 Recall@5 | MRR | 口语 15 Recall@5 | MRR |
|---|---|---|---|---|
| dense | 0.774 | 0.646 | **0.467** | **0.383** |
| dense_rerank | **0.935** | 0.725 | 0.400 | 0.313 |
| sparse | 0.806 | 0.661 | 0.267 | 0.222 |
| hybrid | 0.839 | 0.635 | 0.333 | 0.300 |
| rerank | **0.935** | **0.727** | 0.333 | 0.280 |

三条结论，都是**与旧语料相反的**（旧表：基础集混合 0.880 最高、开重排 −0.040；
口语集要开重排才从 0.400 拉回 0.800）：

1. **基础集收益几乎全来自重排，不是融合。** 混合相对纯向量 +0.065，单开重排 +0.161，
   `rerank` 与 `dense_rerank` 打平（0.935 / 0.935）。融合这一步在这批语料上是白做的。
   旧语料（9 份规范 + 计算书）上"融合稀释"的解释**不成立**了——新语料 52 份里有大量
   政策 md，BM25 不再天然命中原词。
2. **口语集上一切手段都在往下走。** 纯向量 0.467 已是最高，对它重排 −0.067、融合重排 ±0.000。
   最好的那一路从 0.935 掉到 0.467，**连重排都失灵**——瓶颈在查询表示与意图理解，
   不在排序。这是方向①最硬的一条证据，比旧版"重排能救口语题"的说法更接近问题本质。
3. **库外 5 题召回非空 5/5、top1 分数 0.025~0.032（RRF）**：系统没有任何拒答阈值。
   报告里新加了"top1 分数量纲不同，不能跨行比"的说明——同一张表里 dense 是余弦、sparse 是
   BM25 原始分（16.271）、hybrid 是 RRF（0.032），不写这句，读者会把 16 分读成"最有把握"。

**教训**：
1. **过滤条件的"空值"不是"中性值"。** NULL 年份在 SQL 比较里等于"永远不匹配"，
   52 份文档因此对所有带年份的问法隐身。凡是被当过滤维度的字段，入库时就该有值或明确报缺。
2. **`--year` 不带不等于"自动识别"**。入库的语义是"沿用已有年份"，这是刻意的；缺口要靠
   一次性回填补，回填的判据要能说清"这个数字从哪来"，猜的宁可留空。
3. **对照组的设计比对照组里的数字重要。** 四行里有两行把两个变量绑在一起，跑出来的差值
   就没法解释。补一行 `dense_rerank` 的成本是 6 行代码，换来的是"融合有没有用"和
   "重排有没有用"两个能分别回答的问题。
4. **结论会随语料反转。** 旧结论"混合最强、重排有害"在 9 份规范上成立，在 52 份（含 37 份
   政策 md）上不成立。报告里那句语料指纹不是形式主义——它就是"这条结论对哪批语料成立"的边界。

**验证**：459 passed（72 秒）、`ruff check` 干净、3 条变异全红（候选池只用 k、`dense` 也走
重排、漏掉 `dense_rerank` 分派）+ 量纲说明 1 条。基线报告：`eval/reports/dataset-20261028-fe8e22a2.md`
与 `dataset.paraphrase-20261028-fe8e22a2.md`。

---

## 10.4 第三十六轮：清库重导，撞出"目录导入不递归"（2026-10-28）

用户要求"把旧的资料都删除、重新导入"。库里 52 份与磁盘逐字节一致（`content_sha` 全对得上），
所以这次重导**不是修复，是按用户要求重来**：`clear --yes` 清掉 52 文档 / 1210 片段 / 1210 向量，
再从 `文献资料/` 的分类目录整批导回。

#### 1. `ocr_mode=auto` 让整批导入变成"逐页视觉识别"

第一次重导没加 `--ocr`，10 分 50 秒只过 3/45：第 4 份是 21 页无文字层的评价标准，
`.env` 里 `R4P_OCR_MODE=auto` 且 OCR Key 有值 → 走 OCR 路径，`ocr_concurrency=4` 四路并发
逐页调视觉模型。按这个速率，8 份水印型 + 3 份无文字层（合计 320 多页）要跑一个多小时，
而且**按页计费**。

- **`--ocr off` 重跑：45 份 1 分 51 秒**，13 份不可读的按各自理由明确失败。
- 批量入库该显式关 OCR。`auto` 的语义是"给扫描件补内容"，不是"每批都默认做一遍"。
- **`doctor` 的 `ocr.ready=false` 不等于没配 Key。** 那个字段判的是"访客能不能用服务端这把
  Key"（`owned or allow_server_key_fallback`），命令行是服务端主人自己，用的是 `.env` 那把，
  所以 `ready=false` 而 OCR 照跑。判断"这台机器有没有 OCR"要看 `key_source` 与 `.env`。

#### 2. 目录导入只拿第一层，30 份语料无声消失

`_collect_files` 用的是 `iterdir()`，非递归，且**子目录被显式跳过且不进 `skipped`**。

- 语料按 `文献资料/0X_分类/子类/` 组织后：`03_浙江省政策文件/市级/`、`04_技术应用与案例/`
  四个子目录共 30 份全部消失，任务里只有 45 份，**报告一切正常**。
- 这正是它自己注释里那句"语料不全是个安静的错误"——只不过这次是目录结构变了触发它。
  此前 52 份是靠显式文件清单（`data/_batch1.json`）导进去的，所以从没撞到。
- 改成 `_walk_corpus()`：`rglob` 递归、以 `.` 开头的目录整个跳过（`工程文档/.workbuddy/`
  里的工作记忆会被 rglob 当语料收走）、按 `seen` 去重（同时给目录和目录里的文件只算一次，
  否则白花一遍 embedding）。2 条用例 + 3 处变异全红（退回 `iterdir`、不跳隐藏目录、不去重）。

#### 3. 终态与覆盖率

`clear` 之后两次导入（先 45 份、再补 `04_技术应用与案例` 与 `03_浙江省政策文件` 递归）：
**62 份 / 1769 片段 / 1769 向量**，`verify` 四项全 0，`index_state=ready`。

| | 数量 |
|---|---|
| `文献资料/` 内支持格式 | 80 |
| 其中工作笔记（不入库） | 5 |
| 应有语料 | 75 |
| 已入库 | **62** |
| 未入库 | 13（8 水印型 + 3 无文字层 + 2 缺 ToUnicode，全部有独立报错文案） |

年份：`backfill_doc_year.py --apply` 补 50 份，**12 份保持 NULL**（10 篇论文 + 05/06 两份规定）。
论文没有完整发文日期，按现有判据推不出；留空比猜一个年强。

#### 第二版基线（62 份 / 1769 片段，`ids_sha e07720f90185232d`）

| 召回方式 | 基础 31 Recall@5 | MRR | 口语 15 Recall@5 | MRR |
|---|---|---|---|---|
| dense | 0.774 | 0.646 | **0.467** | **0.383** |
| dense_rerank | **0.935** | 0.724 | 0.400 | 0.313 |
| sparse | 0.839 | 0.672 | 0.267 | 0.233 |
| hybrid | 0.839 | 0.652 | 0.333 | 0.289 |
| rerank | **0.935** | **0.731** | 0.333 | 0.280 |

**唯一的变化在基础集的 BM25：0.806 → 0.839，正好等于 hybrid**——10 篇论文标题与术语密集，
BM25 命中更准，融合这一步的增量被吃干（相对纯稀疏 +0.000）。dense 与口语集完全不动，
说明新增语料对向量召回既没帮忙也没添乱（也没挤进 top-5）。三条结论方向不变。

**教训**：
1. **"重导一遍"要先问"重导能改变什么"。** 52 份与磁盘逐字节一致时，清库重导唯一确定的结果是
   再付一遍 embedding，以及三份语料记录全部作废。这次按用户要求做了，但要先把影响面说清。
2. **目录结构是被代码假设的，不是被文档描述的。** 语料从扁平改成两层分类，导入器没有任何
   一行代码变，行为却从"全都进来"变成"少 40%"。**没有报错的环节最需要用例。**
3. **默认值在批量场景里会放大成成本。** `ocr_mode=auto` 对单份扫描件是贴心的，对一批 45 份
   就是"逐页计费 + 一小时"。批量入口应该显式指定，不要依赖 `.env`。

**验证**：461 passed（83 秒）、`ruff check` 干净、3 处变异全红。报告：
`eval/reports/dataset-20261028-e07720f9.md`、`dataset.paraphrase-20261028-e07720f9.md`
（旧指纹的报告已从工作区删除，git 历史里仍可查）。

---

## 10.5 第三十七轮：13 份不可读标准到底怎么进库（2026-10-28）

用户要求"这 13 份最好能用 OCR 或其他手段处理入库"。先做实测再决定，结论是**能做，但必须
先把两件事补上再批量跑**，否则识别结果和原文在检索里长得一模一样，事后无从分辨。

#### 实测（SiliconFlow `Qwen3-VL-8B-Instruct`，150 dpi，现有 `PROMPT_VERSION=v2`）

| 页 | 耗时 | 结论 |
|---|---|---|
| JGJ/T 491 p12 | 27.6s | 3.1.1~3.1.11 条文号连续、内容逐字对得上；**3.1.10 漏了一个"用"字** |
| GB 50210 p15 | 29.9s | 4.1.6~4.2.1 的 10%/3 间/100m²/M20/2m/50mm 全部准确 |
| GB 50210 p18（表格页） | 36.6s | 表 4.3.10 的 4 行数值与检验方法**逐格对得上** |
| GB 55037 p30 | 51.2s | 6.3.1~6.4 全文准确，耐火极限 2.00h/1.00h 无错 |
| DB33 英文目次 | 184.6s | 内容对（双语标准），但耗时异常；10% 的页会慢到 3 分钟 |

单页账单：**输入 1350 + 输出 437 token**，$0.18/M 与 $0.68/M → 约 $0.0006/页。
4 并发实测 **2~4 页/分钟**，135 页用了 54 分钟——真正的成本是时间，不是钱。

#### 批量前补的三件事

1. **`documents.ocr_pages`**（`db._migrate` 补列 + `repo.upsert_document` 写入 + API/前端）。
   `NULL` = 加列之前入库的、不知道；`[]` = 记过了、确实没识别；页码数组 = 识别稿。
   两者不能合并：前者不能说"这是原文"。前端在文档列表里显示「识别 N 页」，悬停给出页码。
2. **识别稿的目录页判据**（`_looks_like_ocr_toc`）。`_toc_pages` 跑在文字层上，这批文件
   恰好没有可用文字层，目录页会进 OCR 变成 `# Contents` 加一串编号节点。而**点线引导符
   活不过识别**：`1 General provisions ......... (1)` 被还原成 `1 General provisions (1)`，
   原来那条"5 个以上的点"的判据一次都不会命中。改用"行尾是页码 + 行都很短 + 不含表格竖线"。
3. **`scripts/ocr_audit.py`**：条文断档、表被引用却缺正文、`▢` 占位符、片段字数塌陷。
   第一版只扫 `chunks.text`，把 title 里的条文号全报成断档（4 份报了 23 处，几乎全是误报）；
   **必须把 title 与 chapter_path 一起算进编号池**，修正后降到 3 处。
   第二版又发现那 3 处里有 2 处是判据本身不成立：**表号不能按连续性查**。GB 55038 的表
   7.4.1 之后直接是表 7.4.5（中间三个编号是条文，正文都在），JGJ/T 467 报"缺图 5.2.6"
   而 5.2.6 本来就是条文、没有配图。改成"被引用却没表头"后剩 1 处，已回原页确认为真。
   图不核——图注没表头可对照，图号天然不连续。

#### 批一结果（4 份 / 135 页 / 54 分钟）

| 文档 | 识别页 | 片段 | 条文断档 | 表缺正文 | ▢ |
|---|---|---|---|---|---|
| JGJ/T 491-2021 装配式内装修技术标准 | 45 | 74 | 0 | 0 | 0 |
| JGJ/T 467-2018 装配式整体卫生间 | 39 | 50 | 0 | 0 | 0 |
| JGJ/T 477-2018 装配式整体厨房 | 25 | 33 | 1 | 0 | 0 |
| GB 55038-2025 住宅项目规范 | 26 | 49 | 0 | 0 | 0 |

**确认 1 处真实丢失**：JGJ/T 477 第 14 页末「5.3.2 厨房家具的安装应符合下列规定：」整行
消失，其下属 1~4 条被挂到 5.3.3 名下——渲染原页对照看得很清楚，条文就在页面上。
135 页里 1 处，约 0.7%；这类错误**只能靠人眼在原页上确认**，`▢` 判据对它无效（模型没标
"认不出"，它是"看不见"）。

#### 批二（9 份 / 387 页 / 约 2 小时）与第三版基线

按页数从小到大执行，先跑小份取早期信号。其中 04 评价标准 21 秒就完——是**命中了页级
缓存**（之前那次被中断的 auto 导入已识别过它的页），一分钱没多花，这正是 `ocr_cache`
按页图哈希做键的理由。GB 55037 是最慢的一份（66 页，单页实测 51 秒）。

**跑完发现 GB 55037 的结果是碎片**：66 页里 44 页只回 1~93 字，p61 只有 11 个字。渲染原页
看，文字完整清晰、带斜向水印。逐项对照后定位到两件事：

1. **图像太大，模型就静默截断。** 这份 PDF 的页面是 1653×2397 pt（普通 A4 的 2.8 倍），
   150 dpi 渲出来 3444×4994 像素。同一渲染做对照：200 dpi → 1 字、150 dpi → 1 字、
   100 dpi → 547 字、长边限 1600 → 533 字。**没有任何报错**，只是内容变少。
   处理：`_render_for_ocr` 只在长边超过 `_OCR_MAX_EDGE` 时才改走 `Matrix`——
   A4 普通页仍走 `dpi=`，**已付费的缓存一个都不失效**。
2. **间歇性停早，靠重试补。** p61 单独调用三次得到 11 / 665 / 668 字，`p15` 则三次都是
   1 字（稳定失败，是上面那个尺寸问题）。处理：识别结果 + 页图墨迹联合判"疑似被截断"，
   就再问一次，**只取更长的那次**。

改完重跑 GB 55037：99 片段 / 27,392 字（原 42 片段 / 9,228 字），目录页这次也正确跳过。

**教训**：
1. **"能识别"和"能逐字引"是两件事。** 表格、条文号、数值都可靠，但会整行丢失，且丢的
   正是"5.3.2"这种最短的一行标题。凡是要引原文的地方，必须能说清"这句是识别稿"。
2. **判据要跑在它真正作用的那份数据上。** 目录判据跑文字层、审计判据只读正文，两处都是
   "判据本身没错，取错了对象"。还有更隐蔽的一种：**判据的前提不成立**。表号被当成条文号
   查连续性，可表号本来就不连续——这类错误看不出破绽，因为代码写得对、跑得通，只是报出来
   的东西全要人工推翻。凡是"报了一堆、逐条核对几乎全是误报"，先查前提，再查实现。
3. **识别稿的体检必须包含 title。** 规范类文档的条文号天然有一半落在标题里（`4.3.9 龙骨
   隔墙应符合下列规定：`），只看正文等于天生漏掉一半编号。
4. **缓存会把坏结果固化成"事实"。** 截断的识别结果不报错、不为空，原样进 `ocr_cache`，
   之后每次重跑都命中它——看起来"识别很稳定"，其实是同一个错误被反复读到。凡是进缓存的
   生成结果，写入前必须有"这次生成是完整的"判据。
5. **"图大到模型放弃"和"图片坏了"在结果上长得一样。** 两者都表现为返回内容变少，
   区分办法只有一个：换一种渲染尺度再问一次。判不了就当成尺寸问题试小图——
   小图的代价只是清晰度下降，大图的代价是整页丢内容。
6. **指标动了先查"答案还在不在"，再查"是不是变难了"。** 查法便宜：内容寻址的 id 在
   不在库里。查完才能说这是回退还是规模效应——两种情况的处理完全相反，前者要修代码，
   后者要改预期。

语料定稿：**75 份 / 2560 片段**（`ids_sha 208ec1576dfc11cc`），verify 四项全 0，
13 份识别稿全部到位，`doc_year` 回填后仍留 12 份 NULL（10 篇论文 + 2 份省级文件，
文件名与来源都推不出年份，保持 NULL）。

#### 第三版基线（池深 25；报告存档在 `eval/reports/archive/dataset*-pool25-208ec157.md`）

> 这一版的 `hybrid`/`rerank` 两行声明 50 深、实际只有 25 条候选（见第四版的 P1-1），
> 口语集 `rerank 0.267` 与基础集 `dense_rerank 0.806` 就是那个 bug 留下的数字。
> 同名报告已被第四版覆盖，旧内容按上表存档，不再当基线。

| 召回方式 | 基础 31 题 Recall@5 | MRR | 口语 15 题 Recall@5 | MRR |
|---|---|---|---|---|
| dense | 0.710 | 0.583 | **0.400** | **0.367** |
| dense_rerank | 0.806 | 0.623 | 0.333 | 0.283 |
| sparse | 0.774 | 0.610 | 0.267 | 0.233 |
| hybrid | 0.774 | 0.576 | 0.333 | 0.289 |
| rerank | **0.839** | **0.631** | 0.267 | 0.222 |

绝对数比第二版全面下降，但**不是回退**：31+15 个黄金片段逐一核对，全部仍在库内且
内容未变（`chunk_id` 是内容寻址的，id 不变就等于正文没变）。下降全部来自候选池——
评估的 filter 是整个「装配式内装」项目，语料从 1769 涨到 2560 片段（+45%），
同题材标准文本彼此相像，top-5 竞争变难。**结论方向不变**：融合对 BM25 零增量
（0.774 vs 0.774）、基础集靠重排（+0.096）、口语集上重排反而 −0.067。

**验证**：484 passed、`ruff check` 干净。批一那 6 条变异 + 短页判据 2 条 + 表图判据 4 条
+ 大页缩放/碎片重试 5 条，共 17 条变异全红。

#### 口语集逐题归因（`scripts/diagnose_recall.py`）

用户问"下一步是不是该提高召回"，先做诊断再答。15 道口语题拆成三堆：

| 归因 | 题数 | 修法 |
|---|---|---|
| 至少一种召回进前 5 | 7 | 本来能答对 |
| 进池但最好排名 6~17 | 3 | 排序问题：调 recall_k / 重排可救 |
| 三种召回都捞不进池 | 5 | **词表鸿沟，换 embedding 救不了** |

那 5 道「没进池」的题，gold 都在库里、内容也对得上，只是用户的大白话和原文术语之间
隔着一条**知识**鸿沟，不是相似度鸿沟：`「会不会掉下来」` ↔ `「悬挂较重物体应采用专用
连接件」`、`「电线坏了能换吗」` ↔ `「设备管线宜与主体结构相分离，方便维修更换」`、
`「老房子能用吗」` ↔ `「既有建筑可参照执行」`、`「吊顶含不含灯具」` ↔ `「含龙骨，不含
电器设备」`。向量和 BM25 都靠"词面/语义相似"匹配，跨不过"掉下来 = 悬挂较重物体"这层
推理——**换 embedding 模型、调 rrf_k 都无济于事**，只能做查询改写/扩展（召回前把口语
翻成术语变体），这正是课题方向①的核心手段。

结论：**下一阶段不是"提高召回准确率"，是"查询理解"**。三路按性价比排序：查询改写
（补那 5 道，最大头）＞ 调 recall_k + 重排（救那 3 道）＞ 换 embedding（收益最小）。

#### 查询术语扩展（`adapters/query_expand.py`，方向①第一刀）

按上面诊断动手。设计三条铁律：

1. **扩展不是替换**。`QueryExpander.expand()` 返回「原话 + 术语」——LLM 改偏了，
   原来能命中的也不丢；最坏回到现状，不会更差。
2. **失败降级回原话**。LLM 不可用、超时、返回空，一律 `return question`，
   检索照常走；扩展是纯增强，绝不能成为检索的新故障点。
3. **输出是关键词串，不是新问句**。`_SYSTEM` 提示词固定要求「纯关键词、空格分隔」，
   短、稳、便宜；BM25 吃词面、向量吃语义锚点，都不需要自然语言问句。

挂载点：`ask.retrieve()` 加可选参数 `expander: QueryExpander | None = None`（默认 None，
行为与加之前完全一致）。扩展串同时喂给 dense（`embed_one(expanded)`）和 sparse
（`fts_search(expanded)`）两个通道。`scripts/eval.py` 加 `--expand` 开关做对照实验，
报告会标注「使用了术语扩展」，且 `--save` 会把带扩展的报告写成 `-expand.md` 后缀，
不会覆盖同名基线。**扩展默认不注入生产链路**（`stream_answer`/`search`/CLI/`/api/search`
都不带 expander）——首版收益为负，默认开启只是白付一次 LLM 调用的延迟与成本。

真实 LLM（魔搭 `Qwen3.5-35B-A3B`）对 5 道「没进池」题的扩展词，术语**字面**命中 gold
原文关键词（`「会不会掉下来」`→`墙体 悬挂 吊柜 固定 承载力 锚固`），但**对照实验证伪了
这个思路的首版形态**：

| 通道 | 未扩展 | 扩展后 |
|---|---|---|
| dense | 0.400 | 0.467 |
| sparse | 0.267 | 0.267 |
| hybrid | 0.333 | 0.333 |
| rerank | 0.267 | 0.333（nDCG 降） |

两个负结论：

1. **收益来源错了**。dense 的 +0.067 不是来自救回 5 道「没进池」题——那 5 道**一道都
   没救回**，全是把「进池排第 6」的题拉进前 5。逐题核对（`diagnose_recall.py` 对照跑）
   确认：目标题没解决。
2. **扩展词「帮倒忙」**。「挂电视吊柜会不会掉」扩出「吊柜 电视」，这两个从原话继承的
   干扰词在语料里大量出现在「集成厨房」语境，把正确片段（4.1.9 墙面悬挂较重物体）
   挤出了 top 50。**「原话 + 术语」这个形态的陷阱：保留原话 = 保留原话里的干扰词**，
   而干扰词在语料里的频率往往比术语更高。

教训（第三十八轮）：**「术语命中」≠「检索命中」**。扩展词字面看着对，不等于它能让
gold 进候选池——判断一个查询改写有没有效，唯一标准是 gold 排名，不是扩展词好不好看。
以及：**负结果要连同实验设施一起留**。`--expand` 开关、`diagnose_recall.py`、5 条变异
都是可复现的对照基础设施，后续任何人想换提示词或换 Query2Doc 路线，都能直接重跑对照。

验证：单元 6 用例 + 集成 3 用例，5 条变异全红（只喂 dense / 只喂 sparse / 不降级 /
替换丢原话 / 空结果不降级）。493 passed、`ruff` 干净。

#### 第四版：先修实验设施，再谈结论（2026-10-28，`0254df7`）

专家复审提了三条 P1，前两条不是"结论对不对"，而是**设施在骗人**，先修设施。

**P1-1 候选池"声明 50、实得 25"。** `eval._hybrid_pool_size(5)=50` 只传给了
`recall_k`，而 `ask.retrieve` 返回前又按 `max(recall//2, top_k*3)` 截成 25。
更麻烦的是 `dense_rerank` 走稠密直出的 50 条：于是"融合 vs 重排"两行的池深本来就
不同，相减不再等于单个变量。改法是把「每通道取多少」与「融合后留多少」分成
`recall_k` / `candidate_k` 两个参数，不给时沿用旧默认（生产行为不变）。
修复后两组数字各只动了一行：

- **口语集 `rerank` 0.267 → 0.333**：原来只有 25 条候选，gold 落在 26~50 名时
  **根本送不到重排器面前**。池深修复不是"纸面整洁"，它改的是重排能看见什么；
- 基础集另四行（dense / sparse / hybrid / rerank）一字未变，`dense_rerank` 0.806 → 0.839。
  这一行的候选池本轮没动，一题之差；重跑两次都是 0.839，而旧那份报告是在"有未提交改动"
  的工作区里生成的，差异无法复现——先按服务端/批次抖动记，别把它读成收益。

数字没变不等于没问题：设施从"说的和做的不一样"变成"说 50 就是 50"。

**P1-2 扩展实验没隔离变量。** `--expand` 把扩展串一路喂到重排器，rerank 那行的
变化里混着"候选集变了"与"重排判断变了"。现在默认只改召回、重排读原话（实验 B），
要研究重排是否也受益必须显式 `--expand-rerank`（实验 C）。
同时修掉 `ask.retrieve` 里"保留原话所以不会更差"的错误注释——拼接会改变整串文本，
排序跟着变，原来命中的片段会被挤出去。

**P1-3 工件不可复现。** `--save` 现在同时落逐题 JSONL（扩展串、各召回方式的 gold
排名与 top-k）；`_code_state` 把 `eval/reports/` 排除出"未提交改动"判定（报告自己
就是产物，否则第一份写完就把后面每份都标脏）；报告坐标系记真实池深。

修完重跑，口语 15 题（`dataset.paraphrase.jsonl`）：

| 召回方式 | A 基线 | B 只扩召回 | C 召回+重排都用扩展串 |
|---|---|---|---|
| dense | 0.400 / MRR 0.367 | **0.467** / 0.356 | **0.467** / 0.356 |
| dense_rerank | 0.333 / 0.283 | 0.333 / 0.283 | 0.333 / 0.283 |
| sparse | 0.267 / 0.233 | 0.267 / **0.267** | 0.267 / 0.267 |
| hybrid | 0.333 / 0.289 | 0.333 / 0.283 | 0.333 / 0.283 |
| rerank | 0.333 / 0.283 | 0.333 / 0.283 | 0.333 / 0.283 |

基础 31 题同步重跑（同一提交、工作区干净）：dense 0.710 / sparse 0.774 / hybrid 0.774 /
rerank 0.839，与第三版逐项相同；`dense_rerank` 0.839（0.633），见上一段说明。
四份报告与逐题 CSV 都在 `eval/reports/`，`A/B/C` 分别对应无后缀 / `-expand` /
`-expand-rerank`。

四个结论，都比"扩展有没有用"更值得记：

1. **扩展是"换题"不是"救命"。** 逐题看（`eval/reports/*-expand.jsonl`）：
   dense 只把「吊顶装灯风机」从池外拉进第 2（唯一真救回），同时把「装修完多久能住」
   从 hybrid 第 1 挤到池外（该题 gold 本身是新闻稿，标错，见复核报告）；
   其余 4 题只是名次挪动 1~2 位。**5 道词表鸿沟题仍然一道没救回**——
   第三十八轮的负结论在修好的设施上重跑，依然成立。
2. **LLM 扩展的随机性大于它想测的效应。** 同一份数据跑两次，15 题扩展串**没有一道
   相同**；hybrid 因此从 0.333 跳到 0.400 又回到 0.333——±0.067 就是一题的权重，
   等于噪声。所以新增 `--expand-from`：把上一轮落盘的扩展串冻结成
   `eval/expansions/paraphrase-20261028.jsonl`，B/C 共用同一组词才可比。
3. **扩展对重排判断没有影响。** B 与 C 用同一组扩展串，`rerank` 一行的逐题 gold
   排名**完全一致**（1/1/4/1…），说明 Qwen3-Reranker 读不读扩展词，排出来的顺序没变。
   之前 C 跑出 0.400 是扩展串换了一批带来的，不是重排受益——这正是 P1-2 那个混杂
   变量造成的假信号。
4. **15 题撑不起结论。** 单题权重 0.067，扩展随机性刚好也是这个量级。要继续做，
   得先把口语集扩到 50 题以上（专家建议按技术解释/施工维护/安全/补贴/适用范围分层，
   并保留库外题）。

**标注复核**（`eval/annotation-audit-20261028.md`）：15 题逐条看正文，11 ✓ / 2 △ /
2 ✗。两条明确错标：**#4「要准备哪些材料」**标到了空白表头（应改
`3fcd5d9f…`「（二）申领…提交以下材料」），**#10「装修完多久能住」**标到了一篇
样板房完工新闻（语料里只有"污染物检测合格"，没有"等多少天"）。两条存疑（#3 的 gold
掺着网页导航文字且范围是"厨房换新"不是"老房子翻新"；#12 标的是格式条款备案而非
"坑"）。**改标会动基准，留待一次性定完再出第五版基线**，这一版先把"哪些题的数字
不能当真"写清楚。

**语料体检对账**（`eval/corpus-audit-20261028.md`）：审计 67 份 / 1835 片段 vs 库内
75 份 / 2560 片段，不是漏数——1835 − 66（5 份工作笔记）+ 791（13 份识别稿）= 2560。

**验证**：498 → 500 passed（新增候选池 5 条 + 扩展串复用 2 条）、`ruff` 干净。

#### 第五版：并行改写通道——保留原话解决了稀释，但没跨过鸿沟（2026-10-29）

第四版把扩展的形态证伪了（拼串），但那是"扩展**形态**没用"还是"扩展**没用**"？
专家给的实验 B 是**并行双通道**：原话独立保留，改写串各查一路，再融合。这一轮把它做出来。

实现：`ask.retrieve(..., rewrites=(...), rewrite_weight=0.5)`，`rrf_multi` 按权重融合；
eval 增 `hybrid_parallel` / `rerank_parallel` 两行（默认不跑）与 `--expand-parallel`、
`--rewrite-weight`。**基线行不传 rewrites**——它们是"不做改写"的对照组。

设计上的三条承诺，都用实测钉住了（`tests/test_parallel_retrieval.py`，6 条变异全红）：

1. 原话通道的候选**一条不丢**：改写权重给到 1.0，原话 top-k 仍在融合结果里；
2. 权重真生效：1.0 与 0.05 给出不同顺序；
3. 基线行（dense/sparse/hybrid/rerank）**与不改写时逐字一致**——本次三份报告的
   dense 0.400 / sparse 0.267 / hybrid 0.333 / rerank 0.333 与第四版逐项相同。

**结果（口语 15 题，冻结扩展串）**：三个权重各跑一遍，基线行不动，只有并行行变。

| 权重 | hybrid | hybrid_parallel | rerank | rerank_parallel |
|---|---|---|---|---|
| 0.3 | 0.333 / MRR 0.289 | **0.400** / 0.269 | 0.333 / 0.283 | 0.333 / 0.283 |
| 0.5 | 0.333 / 0.289 | 0.333 / 0.256 | 0.333 / 0.283 | 0.333 / 0.283 |
| 1.0 | 0.333 / 0.289 | 0.333 / 0.267 | 0.333 / 0.283 | 0.333 / 0.283 |

- 0.3 那 +0.067 **就是一题**（`#9 吊顶装灯风机`，池外 → 第 5）。同一道题在拼串版也能救回。
- 0.5 / 1.0 净零：救回 `#9`、挤掉 `#10 装修完多久能住`（那题标注本身有问题，见复核）。
  挤掉的原因是改写通道的候选**数量多、分数低**，累加后压过"只在单一通道出现"的 gold——
  权重越小越不容易发生，所以 0.3 反而"更好看"。**这是权重的真实权衡，不是调参运气。**
- `rerank_parallel` 在三个权重下都与 `rerank` 一模一样：重排一进来，差别就被它抹平了。
- **5 道词表鸿沟题，三个权重下全部 0 救回。**

**为什么还是没救回——逐通道归因**（`data/_channel_probe.py`，池深 50，两路各自独立看）：

| 归因 | 题数 | 说明 |
|---|---|---|
| 原话已进 Top5 | 7 | 本来就是对的 |
| 改写池内有 gold（15~35 名） | 4 | `#3` `#4` `#6` `#9`：融合/排序没把它拉到前面 |
| **两路池内都没有** | 4 | `#8` `#11` `#14` `#15`：改写词**根本没打到** |

关键在这 4 道"改写没打到"的题上。LLM 给的词**看着很对**，但都不是规范原文用的那个搭配：

| 题 | LLM 扩展词里最好的 | 排名 | 规范原词 | 排名 |
|---|---|---|---|---|
| `#6` 挂电视会不会掉 | 悬挂 | 21 | **专用连接件** | **2** |
| `#8` 电线坏了能换吗 | 更换 | 12 | **维修更换** | **6** |
| `#11` 老房子能用吗 | 既有建筑 | 26 | **参照执行** | **5** |
| `#14` 吊顶含不含灯具 | 集成吊顶 | 池外 | **不含电器设备** | **1** |
| `#15` 怎么知道做得好 | 检验批 | 36 | **隐蔽工程验收** | **6** |

另外两件事也测出来了：

- **稀释是真的，但是次要因素。** 逐词单独查**总是**比拼成一串好：
  `#8` 拼串池外 → 逐词 12、`#6` 35 → 21、`#15` 池外 → 36、`#11` 池外 → 26。
  但即使逐词，也没有一道进 Top5——所以"改成逐术语通道"值得试，**但单靠它救不回来**。
- **主因是术语从哪来。** 人工挑的规范原词都进 Top5（1/2/5/6/6），LLM 自由联想出来的
  近义/上位概念最好也只有 12~36 名。而且要的不是"更多同义词"，是**原文的固定搭配**。

这条负结论与专家的判断对上了：下一步不是继续调融合权重、池深或重排，而是
"公众表达 → 规范用语"的**受控对照表**（专家审过的词表），LLM 只能负责"把口语归到
哪一类意图"，不能负责"说出条文里的那个词"。

**工程上撞到一个真问题：** 这条链路上所有读路径共用**一条** sqlite 连接
（`Embedder.embed_one` 读写 `embed_cache`、`NumpyStore.search` 读代次与归档、
`repo.fts_search` 本身就是 SQL）。第一版用 `asyncio.gather` 并发多路查询，实测在
15 题里随机炸 1 题 `InterfaceError: bad parameter or other API misuse`，报告上只表现
为"有效样本 14/15"。**所以"并行"只能指候选来源并行，不是执行并行**：多一路改写
就多一次远程 embedding 往返，延迟线性增长。报告里的耗时（hybrid 19 ms →
hybrid_parallel 37 ms）**低估**了这个成本——`embed_cache` 命中了，真实部署要再加一次
往返。`tests/test_parallel_retrieval.py::test_channels_are_queried_serially` 用探针
钉住"任一路不得重叠执行"，改回 gather 必红。

报告：`eval/reports/dataset.paraphrase-20261029-208ec157-expand-parallel-w{03,05,1}.md`
与同名 `.jsonl`（逐题 gold 名次、P@5、每题耗时）。**验证**：500 → 508 passed、`ruff` 干净。

#### 第六版：受控词表——词对了，而且要**重排也读同一批词**（2026-10-29）

专家给的第一条建议就是做「公众表达 → 规范用语」对照表。先手写 8 条（5 条对鸿沟题、
3 条对本来就会做的题做对照），每条带 `source` 指到条文原话，匹配是对原问句做子串
判断（分组 `all`，不分词）——**能一眼看出为什么注入这几个词**，这是它相对 LLM 改写
唯一的优势，也是它必须先被证明的原因。

**第一次跑，几乎没有效果。** w=0.3/0.5 时 `hybrid_parallel` 与基线一字不差，w=1.0
只救回 1 题（`#8`）。词明明对了（单术语 BM25 gold 排 2/6/5/1/6），为什么进不了前 5？

两处实测把原因钉住了：

1. **RRF 按"在几条通道里出现"累加。** 原话那一路的 dense 与 sparse 高度重合，同一个
   候选通常拿两份分（`1/(60+r)` × 2）；而规范术语只有一个通道能召回 gold（术语的
   向量表示很差，`dense` 通道 gold 全部池外），只拿一份。于是 gold 在融合里天然垫底：
   `#14` 的术语 BM25 第 1 名，融合后排 **43**；`#11` 第 5 名 → 融合后 23。
   把权重从 1 调到 3 只把 `#14` 抬到 19——**放大权重只会放大"两条通道都占"的候选的优势**，
   不是解法。
2. **真正的断点在重排。** 把候选池做成并集（原话 50 + 各术语 50），gold 五题**全在池内**，
   但用**口语问句**重排仍然排 19/20/5/7/12；把重排查询换成**规范术语串**，立刻变成
   3/1/1/1/1。也就是说：上一轮"扩展改善不了重排判断"的结论，是因为扩展词本身就是错的
   （近义词）。给对了词，重排读得懂——它读不懂的是口语。

接进评测（`--term-map` + `--term-rerank`，术语仍作独立通道、原话那一路不动）：

| 配置 | `hybrid` | `rerank` | 鸿沟 5 题进前 5 |
|---|---|---|---|
| 基线（无词表） | 0.333 | 0.333 | 0 |
| 词表，重排读原话，w=0.5 | 0.333 | 0.400 | 1 |
| 词表，重排读原话，w=1.0 | 0.400 | 0.400 | 1 |
| 词表，重排读术语串，w=0.5 | 0.333 | 0.533 | 4 |
| **词表，重排读术语串，w=1.0** | 0.400 | **0.667** | **5** |

逐题：`#6 → 3`、`#8 → 1`、`#11 → 1`、`#14 → 1`、`#15 → 1`。7 道词表未覆盖的题
**逐项与基线一致**；3 道本来就对的题（`#5`/`#7`/`#13`）仍是第 1 名。**一题都没丢。**

顺手测出两件工程事实：

- **`terms`（每词一路）与 `joined`（拼成一串）在开了 `--term-rerank` 之后结果完全相同**
  （逐题 gold 名次一字不差）。形态只在"重排读原话"时才有差别——那时拼串会稀释 BM25。
  生产链路应该用 `joined`：少 N−1 次 embedding 往返，结果一样。
- **w=0.5 会丢掉 `#11`。** `#11` 的两个术语一个第 5、一个第 26，权重不够时被原话两份分
  压住。这不是"调参调出来的"，是上面第 1 条的结构性后果——要低于 1.0 的权重，就得先
  解决"术语通道只拿一份分"这件事（比如并集 + 重排，而不是 RRF 加权）。

**这条结论的边界（必须一起读）**：8 条规则里 5 条是**看着这 15 题的失败写的**，所以它
证明的是「给对词 → 这条链能通」（可行性），**不是**「词表能泛化」。真正的检验要等
50 题分层问题集与一份不参照测试结果的词表；词表也**尚未接入生产链路**，现在是评估侧
开关。词表现状：`rag4prefa/terms/public_to_standard.json`，8 条，命中 8/15 题。

报告：`eval/reports/dataset.paraphrase-20261029-208ec157-termmap-{terms,joined}-w{05,1}[-rerank].md`
与同名 `.jsonl`（逐题术语、gold 名次、P@5）。**验证**：508 → 525 passed（新增词表
17 条），7 条变异全部捕获（未命中兜底返回原话 / `all` 退化成 `any` / 去掉空组校验 /
`rewrite` 忽略形态 / 术语拼进召回串 / 未覆盖不记账 / 基线行也读术语串），`ruff` 干净。

#### 第七版：口语集扩到 50 题——标注先定，再换样本（2026-10-29）

第六版把机制搞清楚了（杠杆在重排查询、不在融合方式），但所有结论都建立在 15 题上，
单题权重 0.067，和要测的效应同量级。这一轮先把**标注定稿**，再**换样本**，最后才出基线。
顺序不能反：先扩题再改标，等于白跑一次。

**一、4 处标注一次性改完**（`annotation-audit-20261028.md` 的 △/✗）+ 4 类处置不同：

| # | 处置 | 依据 |
|---|---|---|
| 3 | 保问法、换 gold（导航页 → 28 号文补贴标准） | "老房子翻新"与"旧房装修厨卫改造补贴"是同一件事，错在挑错段，不在问法 |
| 4 | 换 gold（空白清单 → 「（二）申领…提交以下材料」） | 空白表回答"哪些品类能补"，不是"要交什么" |
| 10 | 换问法（"多久"→"怎么判断能不能住"）+ 原问法移入库外集 | 语料没有"等多少天"，用 Recall 惩罚它是标注错误 |
| 12 | 换问法（"坑"→"格式条款"） | "坑"在别的条款里，该段只讲格式条款备案 |

**二、口语集 15 → 50 题**，按技术解释 8 / 施工维护 7 / 安全 7 / 补贴 7 / 适用范围 6 分层。
出题约束写在 `eval/paraphrase-50-20261029.md`：先写问句后找答案、**不参考受控词表**、
gold 必须读过正文且能回答。库外集补到 6 条（新增"装修完多久能住进去"）。

**三、这一版基线**（语料 `208ec157`，50 题全部计入）：

| 召回方式 | 老 15 题 | 新 35 题 | 全部 50 题 |
|---|---|---|---|
| dense | 0.533 | 0.629 | 0.600 |
| dense_rerank | 0.533 | 0.771 | **0.700** |
| sparse | 0.333 | 0.343 | 0.340 |
| hybrid | 0.333 | 0.514 | 0.460 |
| rerank | 0.533 | 0.771 | **0.700** |

三条读法，缺一条就会读错：

1. **老 15 题 0.400 → 0.533 是改标带来的**，检索代码一行没动。#3/#4 换成正确 gold 后进了
   前 5。把它当成"这一版检索变好了"是错的。
2. **新 35 题比老题好搜是出题方式造成的**：35 题里 31 题的 gold 在出题当时就在检索候选里
   （先看候选、再读正文确认），只有 4 题靠关键词全文搜到。所以新题的绝对数偏乐观，
   **50 题的 0.600 不能与第四版的 0.400 比大小**，只能比同一批题在不同配置间的差值。
3. **重排净 +0.100，但它是两种效应相抵**：补贴组 0.286→1.000（重排分得清 28 号文与
   29 号文的年份/金额），施工维护组 0.857→0.714、安全组 1.000→0.857（重排读不出"动作"
   类问题的意图）。分组表在这份文档里，别只引用净值。

**四、换样本暴露的真问题：等权 RRF 让 hybrid 系统性低于 pure dense**

50 题上 `hybrid 0.460` 比 `dense 0.600` 低 0.140，施工维护组更是 **0/7**（同组 dense 6/7）。
单题实测「卫生间地面要不要往地漏那边找坡？」：gold 在 dense 第 1、BM25 不进前 50、
融合后第 **22**。

原因不是实现写错，是等权 RRF 的算术：单路排第 1 得 `1/61≈0.0164`，"两路各排 30"
得 `1/91+1/90≈0.0220`。**单路强命中被双路弱命中系统性压过**——与第四十一轮在
"原话 vs 改写"通道上遇到的是同一个机制，只是这次在 dense/BM25 之间，而两者没有主次，
不能像改写那样靠"原话恒为 1.0"自然解决。

`tests/test_parallel_retrieval.py` 用构造数据把这条钉住了两条：等权下弱胜强；
权重抬到临界值（`w > 2.06`，2.0 仍输）才翻回来。**这一轮不动融合算法**——它是产品行为
变更，需要单独一轮做权重扫描与分组验证；基线先固定，否则"改了什么"说不清。

**五、受控词表在 50 题上的复现，以及它没证明什么**

| 组 | 题数 | rerank（原话） | rerank_parallel（词表 + 术语重排） |
|---|---|---|---|
| 词表命中 | 8 | 0.375 | **0.750** |
| 未命中 | 42 | 0.762 | 0.762 |
| 全部 | 50 | 0.700 | 0.760 |

救回 3 题（#6/#8/#15）、0 题变差，42 道未覆盖题的 gold 名次完全相同——归因比 15 题那次
干净得多（那次只有 10 道对照题）。但**新出的 35 题，词表命中率 0%**：8 条映射覆盖的是
老 15 题的 5 个失败点，换一批不看词表出的题一条都不命中。所以第六版的结论不变、
边界更硬了：**机制可行，尚未证明可推广**。

**顺手修了一个会吞掉历史的问题**：`_report_path` 原来只带语料指纹，承诺的是"同语料
同名"，没承诺"同数据集同名"。本轮实测踩到：同一天、同一份语料、同一套配置，把口语集
从 15 题换到 50 题重跑，文件名一字不差，15 题那份 `…-termmap-terms-w05-rerank.md` 被
静默覆盖——而上一节引用的正是那个名字。现在名字里加了数据集 sha 前 8 位
（`dataset.paraphrase-3da85a91-20261029-208ec157.md`），`test_eval_metrics.py` 加了一条
用例钉住"同语料不同数据集必须不同名"。被覆盖的 15 题版本已归档为
`eval/reports/archive/…-termmap-terms-w05-rerank.paraphrase15.md`。

**复现**：

```bash
python scripts/eval.py --dataset eval/dataset.paraphrase.jsonl \
  --modes dense,dense_rerank,sparse,hybrid,rerank --save --diagnostics data/_diag_p50.jsonl
python scripts/eval.py --dataset eval/dataset.paraphrase.jsonl --modes rerank,rerank_parallel \
  --term-map rag4prefa/terms/public_to_standard.json --term-rerank --save
```

**验证**：525 → 527 passed（新增 2 条 RRF 机制用例），`ruff` 干净。
报告：`eval/reports/dataset.paraphrase-3da85a91-20261029-208ec157.md` 与 `…-termmap-terms-w05-rerank.md`。

**下一步（按优先级）**：①写一份不看测试集的词表，验泛化；②给 dense/BM25 加权做权重扫描，
或用"名次并集 + 重排"替代等权 RRF；③词表接生产链路（现在是评估侧开关）。

#### 第八版：修融合——把"展示位对半分"改成"按突出程度分配"（2026-10-29）

第七版留了个明确的账：`hybrid` 比纯 `dense` 低 0.140，施工维护组 0/7，并写明"这一轮不动
融合算法"。这一轮就是那一轮，做法是先离线扫、再动生产代码。

**一、先在缓存上扫，不急着改代码**

81 题（口语 50 + 基础 31）的 dense/sparse top-50 先落盘，之后任意策略都是纯 CPU。
仓库里留下的是 `scripts/diagnose_fusion.py`（只保留 rrf / zscore 两条常用对照 + 缓存复用；
当时那 20 多种策略是一次性探针，没入库）。扫过的策略：

| 策略 | 口语 R@5 | 基础 R@5 | 说明 |
|---|---|---|---|
| 等权 RRF（现状） | 0.440 | 0.774 | 基准 |
| RRF 加权 w=2.5 | 0.560 | 0.710 | 单一权重，救口语伤基础 |
| RRF k=60→5 | 0.640 | 0.839 | k 越小越偏头部 |
| RRF 窗口 10 | 0.640 | 0.839 | **单点峰**：win5/15 掉回 0.580 |
| 名次取平均 | 0.520 | 0.742 | 丢掉"双路共振加分"就退步 |
| 名次并集/交错 | 0.580~0.600 | 0.710~0.839 | 有输有赢 |
| **本路 z-score 等权** | **0.640** | **0.871** | α 在 0.4~0.6 是平台，不是尖峰 |

选 z-score 的三个理由，逐条都有数字：①两套数据集同时改善；②参数有平台（不像窗口 10 是
单点峰，那种"扫出来的最优"多半是过拟合 50 题）；③MRR 也涨（0.313→0.389、0.576→0.688），
说明不只是多挤进几题，第一名更准了。

**二、实现上做了两个保守选择**

- **低于本路均值的候选给 0 分，不给负分。** 负分等于说"本路排第 50"比"本路没命中"更差；
  在稠密 + BM25 上无所谓，但并行改写通道会借它把原话候选压到改写候选之后，破坏"改写
  只能加、不能减"（第四十轮的核心结论）。两种写法在 81 题上指标完全一样（0.640/0.871，
  MRR 0.389/0.688），选不破坏不变量的那个。
- **一路分数全同（std≈0）时不投票，但候选仍以 0 分入表。** 说不出谁更靠前就别假装知道；
  不能整路丢掉——那是这路召回的全部产出。若所有路都退化，退回名次法，保证顺序确定、结果不空。

旧行为保留为 `fusion=rrf`（配置项 + `--fusion rrf`），随时可回退。

**三、两版同一天前后脚跑（语料 `208ec157`，池深 50）**

| 召回方式 | 基础 31（rrf → zscore） | 口语 50（rrf → zscore） |
|---|---|---|
| dense | 0.710 → 0.710 | 0.600 → 0.600 |
| dense_rerank | 0.806 → 0.806 | 0.680 → 0.680 |
| sparse | 0.774 → 0.774 | 0.340 → 0.340 |
| hybrid | 0.774 → **0.871** | 0.460 → **0.640** |
| rerank | 0.839 → **0.871** | 0.700 → 0.680 |

前三行是对照组（不经过融合），**逐题一字不差**，所以后两行的差值只可能来自融合。
逐题：基础 赢 3 输 0，口语 赢 10 输 1（输的那题是"省级奖补发给谁"，sparse 独有精确词
"奖补"被压了一点）。

**四、这一轮最重要的发现：重排会把融合的收益吃掉，但只在池深 50 时**

口语 `rerank` 那一行 0.700 → 0.680（还退了 1 题），看着像"改了没用"。查下来是三件事：

1. **重排对池内每条独立打分，池内顺序不影响结果**，只有"gold 有没有进池"才影响。
   池深 50 时两版的池内覆盖率一样（口语都是 41/50，基础都是 31/31）。
2. 生产默认池深是 **25**（`candidate_k = max(recall_k//2, top_k*3)`），不是 50。
   池深 25 时覆盖率就拉开了：口语 rrf 38/50 → zscore **41/50**，基础 30/31 → 31/31。
3. 用生产池深实测（两种融合逐题交替跑，摊平重排服务的时间漂移）：
   **口语 0.640 → 0.720（赢 4 输 0）、基础 0.839 → 0.871（赢 2 输 1）。**

所以"融合对重排没用"是个**被 50 深评测池制造出来的假象**。以后评"融合类"改动，必须同时报
生产池深（25）下的重排结果，否则会把收益读丢。

**五、顺手确认的两件旧账**

- **重排服务的绝对值会漂。** 上一版报告 `dense_rerank` 0.839，这一版两版都是 0.806，而这一行
  不经过融合、代码没动。同一天重跑两版均复现 0.806；单题同进程连跑 4 次一致、跨时段不同。
  结论：任何"比上一版涨了多少"都要同一天把两个配置各跑一遍再作差。
- **`top_n` 只拿回前 5 条。** 查那 1 题时发现 gold 的 `rerank_score` 是 None——不是打了 0 分，
  是请求只取 `top_n=5`，排在 6 名以后的候选根本没参与打分。所以"重排没救回"有两种含义
  （判断错了 / 压根没看），目前只能看前者。

**六、防呆**

`_report_path` 现在**始终**把融合方式写进文件名（`-fusionrrf` / `-fusionzscore`），默认值
也不例外——只给非默认值加后缀，等于"哪天默认值变了，旧报告就被顶掉"，那正是上一轮
dataset_sha 踩过的同一个坑。上一轮那两份报告已按新规则改名（数字未动）。

**复现**：

```bash
python scripts/eval.py --dataset eval/dataset.jsonl \
  --modes dense,dense_rerank,sparse,hybrid,rerank --save
python scripts/eval.py --dataset eval/dataset.jsonl \
  --modes dense,dense_rerank,sparse,hybrid,rerank --fusion rrf --save
```

报告：`eval/reports/dataset-5d9cf3ff-20261029-208ec157-fusion{zscore,rrf}.md`、
`eval/reports/dataset.paraphrase-3da85a91-20261029-208ec157-fusion{zscore,rrf}.md`。
默认池深（25）的重排对照走 `scripts/diagnose_fusion.py --pool 25 --rerank`
（`eval.py` 固定用 50 深池，看不到这一段）。

**写这个诊断脚本时踩了两个坑，都是"对照必须与被对照的链路用同一套算术"：**

1. 不能借 `ev.dense_hits` 取候选——它会丢掉 `chunks` 表里查不到的 id，而 `ask.retrieve`
   用的是原始 pairs。差一条候选就换人。
2. 融合排序**不能加 `kv[0]` 做 tie-break**。`ask._to_hits` 用的是稳定排序，同分保持插入序
   （稠密那一路先出现）；而 `1/61` 恰好是「dense 第 1」与「sparse 第 1」最容易撞上的分数。
   加了按 id 排，口语整行就从 0.460 变 0.440——差别全在这一个同分上。

修完之后脚本与 `eval.py` 的 `hybrid` 行逐题一致（基础 24/31 = 0.774、口语 23/50 = 0.460）。

**验证**：528 → 537 passed（新增 `tests/test_fusion.py` 9 条），6 条变异全部捕获
（去掉本路标准化 / 恢复负分 / 全同分那一路照样投票 / 不退回名次法 / 权重失效 /
`retrieve` 不读配置），`ruff` 干净。

**下一步（按优先级）**：①写一份不看测试集的词表，验泛化（这条从第六版欠到现在，第九版做了）；
②词表接生产链路；③`top_n` 与池深的关系——重排到底该看多少条候选，现在是被 top_k 绑死的。

#### 第九版：按条文建词表——覆盖翻倍，增益没涨；真正有用的是「怎么拼重排查询」（2026-10-29）

**这一版把第六版欠下的账还了，结论是负的，但顺手挖出一个可直接落地的改动。**

**方法先定，再选词。** `eval/termmap-v2-protocol.md` 写在选词之前，三条准入判据（术语逐字
可查 / 公众说法不能在条文里出现 / 一条只表达一个概念）、条数按主题域定而不按题目数定、
指标先写死。`scripts/check_term_map.py` 把前两条变成机械校验——它当场抓出我自己三处
`source` 写法错误（`（二）申领` 粘在文档名上，指向就不在库了）。

**28 条的来源。** 先机械抽：只从「规范句」（含 应/宜/不应/不得）里抽 2~6 字 n-gram，
跨 ≥3 份文档复现、去掉含情态动词的句式碎片（`应符合设计要`）与同频子串，2560 条片段出
5889 个候选，提纯比高的正是 `翘曲`/`检修口`/`存水弯`/`水封高度`/`燃烧性能等级`/`防火封堵`。
再按五个主题域各取 4~8 条。**覆盖率 8/50 → 15/50，但每条词条最多覆盖 2 题**——它仍是
按问法索引的查找表，约 0.5 题/条，没有规模效应。

**净效果没有增加。** 基线都是 `rerank 0.680`（三次独立运行都一样，可横比）：

| 配置 | 覆盖 | Δ | 覆盖组 | 命中组救援率 |
|---|---|---|---|---|
| v1（8 条）+ append | 8 | +0.100 | 0.375 → 1.000（5 赢 0 输） | 5/8 = 62% |
| v2（28 条）+ 纯术语串 | 15 | +0.040 | 0.667 → 0.800（5 赢 3 输） | 4/15 = 27% |
| v2（28 条）+ append | 15 | +0.080 | 0.667 → 0.933（4 赢 0 输） | 4/15 = 27% |

条数翻 3.5 倍，净增益没有增加（差 0.020 正好是一题，在噪声地板内，不能说更低、也不能说更高）。

**两套独立切法互证，指向同一个结论。** `public_side` 自查（5 条 `recalled`）与「v1 是否也
覆盖」这件客观事实，给出**完全相同**的 5 题 / 10 题分组：v2 相对 v1 新覆盖的 10 题只多救
1 题，另外 3 次救援全落在 v1 早已覆盖的那 5 题上——而那 5 题正是写词表时记得住的题。
所以「按条文建词表」买到的是覆盖面，不是准确度。

**真正的收获：纯术语串会打坏本来已对的题，`append` 不会。** 纯术语串那版 3 道输题基线
**全是第 1 名**。固定候选池、只换 `rerank_query` 的三方对照（`scripts/diagnose_term_rerank.py`）：

| 重排查询 | 覆盖的 15 题进前 5 |
|---|---|
| 原话 | 11/15 |
| 术语串 | 12/15 |
| **原话 + 术语串** | **14/15** |

机制很直白：那几道题重排本来就能读懂，换查询等于拿掉信号。`--term-rerank` 因此从布尔开关
变成三态（`none` / `terms` / `append`，裸写仍等于 `terms`），**`append` 才是该用的形态**。
改这个枚举时踩了同一个坑两次：`"none"` 是真值，`if args.term_rerank` 恒成立——`stage`
标签和参数校验各中一次，`--modes` 非法时报的却是「`--term-rerank` 需要同时给 `--term-map`」。

**顺手量出噪声地板。** 三次独立运行的基线行逐题对比：口语 48/50 名次全同、2 题偏 1 位、
0 题跨 top-5。基础 31 题上一题就是 0.032，而 `append` 那版的 +0.032 全部来自**未覆盖组**
的一题——那条路径的代码与基线一模一样，所以它只能是抖动。**单次运行的 Recall@5 有 ±1 题
地板，小于 2 题的差别不要单独解释**——所以这一版所有结论都用赢/输题数表述。

**报告命名又补了一处**：只写「用了词表」不写「哪一份」，v1 与 v2 会互相覆盖——而这两份的
结论正好相反，是最该并存的两份。现在文件名里带词表 stem（`-termmap-public_to_standard.v2-`）。

**验证**：537 → 539 passed（新增 append 语义用例、`public_side` 取值校验），`ruff` 干净。
`eval/termmap-v2-20261029.md` 是这一版的完整记录。

**下一步（按优先级）**：①把 `append` 接到生产链路（它在评估里已经证明是净正，且不伤
专业问法集）；②`public_side` 那 5 条 `recalled` 说明「选题来源」才是瓶颈——真正的盲性
设计是按文档分半构建再交叉评测；③匹配方式从子串换成可解释的语义匹配（这是"查找表→
知识层"唯一的路，但会引入新的不可解释性，要单独一轮）。

**下一步（已部分执行）**：①已做（见第十版）；②③仍待。

---

#### 第十版：把 `append` 接进生产链路，并顺手证伪了「`joined` 与 `terms` 一字不差」（2026-10-29）

第九版留的账是"`append` 在评估里净正（4 赢 0 输）、不伤专业问法集，该接进生产链路"。
这一轮只做这一件事，另加一条对**默认值**的当场复核。

**接法只有一条规矩：禁止在 `retrieve` 里隐式注入。**

评估脚本的基线行（`hybrid` / `rerank`）要求"完全不做改写"，它们也调 `retrieve`。若把它
塞进 `retrieve` 按配置自动生效，那么 `.env` 里一开词表，**所有基线行都会悄悄多一路通道**，
报告里两行的差值就不再是单个变量——第三十八轮读错 0.400→0.467 就是这种形态。
所以注入点放在两个**生产入口**（`stream_answer` / `search`），由它们把词表当参数往
下传：`term_channels()` 产出改写串与术语列表，改写串进 `rewrites`（原话那一路不动），
术语列表给 `build_rerank_query()` 拼「原话 + 术语」。

**配了却没传，必须报错。** `R4P_TERM_MAP` 只在装配时读一次（`build_services` →
`Services.term_map`），调用方漏传时若静默当"没有词表"，界面、日志、指标全都正常，只有
口语问句悄悄拿不到术语通道——这类失效没有任何一处会报出来。所以 `_require_term_map_wired`
在请求上直接失败。**坏词表则相反**：不让服务起不来（它是可选增强项），但降级必须可见——
`term_map_error` 进 `/api/health` 与 notes，`enabled=false`、`rerank=none`。

**默认值当场复核：`joined` 真的和 `terms` 一样吗？**

第九版在 15 题（v1 词表）上量到"开了 `--term-rerank` 之后两种形态逐题一字不差"，据此定了
生产默认 `joined`（少 N−1 次 embedding 往返）。但那是 15 题 + 8 条词，而 v2 的 50 题报告
用的是 `terms`。默认值不能靠一条老结论，于是同一天把 50 题用 `joined` 重跑一遍
（`…-termmap-public_to_standard.v2-joined-w05-rerankappend-fusionzscore.md`，16:05；`terms`
那份是 15:28 跑的）：

| 行 | `terms` | `joined` | 逐题名次变化 |
|---|---|---|---|
| `hybrid`（基线，不含术语） | 0.640 | 0.640 | **0 题** |
| `hybrid_parallel` | 0.620 | 0.620 | 2 题（3→2、3→2，都变好） |
| `rerank`（基线，不含术语） | 0.680 | 0.680 | **0 题** |
| `rerank_parallel` | 0.760 | 0.760 | 1 题（1→2，仍在 top-5） |

**结论要按"覆盖"和"名次"分开说**：两种形态的 Recall 逐行相同，名次上 2 题前移 1 题后移
（MRR 0.596→0.586、nDCG 0.637→0.630，一题的量级）。所以"一字不差"在 50 题上不成立，
但差别落在 ±1 题的噪声地板内、且**没有一题跨 top-5**——不足以推翻"取 `joined` 省往返"
这个选择，足以把第九版那句"完全相同"改写成"Recall 相同、名次 1~2 题抖动"。
两条基线行 0 题变化这一点更重要：它证明**原话通道确实一条都没被动过**，这是整层设计的
唯一承诺。

**代价**：每题多一次 embedding（术语串在问句间高度重复，第二次起命中 `embed_cache`；
报告里 `hybrid_parallel` 平均 286 ms 是首次无缓存的那一次）。收益仍严格等于被覆盖的题：
15/50 覆盖、覆盖组 0.667 → 0.933、未覆盖 35 题名次全同。**默认仍然关着**（`R4P_TERM_MAP`
留空），因为 35/50 命中率意味着它对多数问句什么也不做，而每次问答都要为覆盖的题付一次往返。

**验证**：539 → 553 passed（新增 `tests/test_term_map_production.py` 14 条：通道数、重排查询、
未覆盖不多开通道、漏传必炸、坏词表降级可见、两个 HTTP 入口都真的把词表传下去），`ruff` 干净。

---

#### 第十一版：词表误触发——先量，再修，代价逐条可查（2026-10-29）

专家复审（第十版之后那份）指出 `term_map.py` 的子串匹配有"词对了、意错了"的风险，举例
`ceiling-scope`。我的第一反应是"纸上假设"，因为 v2 只在口语集上跑过。量完之后发现不是：
**专业问法集 31 题触发 6 次，其中 2 次明确错、1 次靠运气**（认定标准写进
`eval/termmap-misfire-20261029.md` 的逐题表）。

| 问句 | 触发 | 问题 |
|---|---|---|
| 吊顶安装灯具需要满足什么构造要求？ | `ceiling-scope` | 注入补贴品类表的「不含电器设备」，与构造无关 |
| 架空楼地面内敷设管线时，检修口宜设置在什么位置？ | `pipeline-hidden` | 「检修口」的「修」被当成「维修」，注入「管线分离」 |
| 装配式内装修的设备和管线与主体结构的关系… | `pipeline-hidden` | 命中的是「装**修**」的「修」；注入的术语恰好是答案，纯属运气 |

两种错法各有机制：**单字当证据**（`修`/`板`/`级` 在中文里是语素不是词）与**意图词与对象词
混在同一个 OR 组**（`包含/算不算` 与 `灯具/浴霸` 原来是一组，等于"提到灯具就算品类边界题"）。

**一条规则砸了自己一脚。** 先写的是全局硬规则"组内还有多字备选时，单字不算证据"。落地后
立刻发现它把 v1 的主力映射废了：`wall-hanging` 第二组全是单字（掉/塌/坠/固定/牢/承重/安全），
「会不会掉下来」靠的是「掉」——那正是 15 题实验里把 5 道鸿沟题全推进前 5 的那条。

一刀切不行，放水也不行。所以规则默认严格，另开一个**按词条**的豁免
`single_char_evidence`（默认 `false`），把"我知道这条在拿语素当词用"写进数据：

| 词表 | 豁免条数 | 为什么 |
|---|---|---|
| v1 | 2（`wall-hanging`、`pipeline-repair`） | 15 题实验的真实口径，改了复现不出那份报告 |
| v2 | 2（`pipeline-hidden`、`panel-crack`） | 50 题的报告在这个口径下跑的 |
| v3 | **0** | 测试钉死（`test_v3_has_no_single_char_exemption`） |

这不是给历史开后门，是**技术债登记**：哪条映射依赖弱证据，写在文件里、看得见、能逐条清算，
而不是变成"新规则对老词表不生效"这种没人解释得清的现象。同理 v3 的 `tests` 里有一条断言
v3 与 v2 的 `(id, terms, source, public_side)` **完全一致**——收紧误触发很容易顺手变成换术语，
那就是另一组实验了，还会让 v2 上的所有数字失去可比性。

**代价与收益都逐条可查**（`strict=False` 保留老口径专门用来做反事实）：

| 数据集 | 题数 | v2 触发 | v3 触发 | 说明 |
|---|---|---|---|---|
| 口语集 | 50 | 15 | 14 | 丢的那题靠单字「裂 + 板」，见下 |
| 专业问法集 | 31 | 6 | 3 | 丢的 3 题全是噪音 |
| 库外集 | 6 | 1 | 1 | 北京补贴题仍触发——那是地域问题，词表修不了 |

**口语集丢的那 1 题不补**：「两块板拼的缝，以后会不会裂开？」要救它就得把「板拼」这种
非自然搭配写进词表（问句原文是"两块**板拼**的缝"）。那是照着测试题写规则，不是补全日常说法。
凡是真的日常搭配（`裂开`/`开裂`/`板缝`）加了也过不了第二组（"板"仍是单字），所以不加。
这题本身带「裂开」二字，BM25 有机会自己命中。

**闸门是数据不是注释。** `eval/term-negatives.jsonl` 7 条负例（同词异意图 2 / 对象词当意图词 1 /
单字语素 3 / 适用对象错 1），审计脚本 `scripts/audit_term_misfire.py` 与
`tests/test_term_map_constraints.py` 共用同一份。**鉴别力当场验过：拿冻结的 v2 跑会抓出 3 条
违规（退出码 1），v3 全过**——`test_negative_set_would_have_caught_v2` 把这件事钉住，
因为"只在 v3 上全绿"的闸门可能是根本没接上。

**检索侧零代价，而且丢覆盖的那题变好了**（从干净提交 `7311c41` 重跑 50 题，报告头写"工作区
干净"）：`hybrid` 0.640 / `hybrid_parallel` 0.620 / `rerank` 0.680 / `rerank_parallel` 0.760，
四行与 v2 配置逐题相同；唯二的变化都落在丢覆盖的那道题上——「两块板拼的缝，以后会不会裂开？」
不再注入「裂缝 翘曲」，名次 2→1、4→2。注入语义本来是对的，坏在"稀释"：原话已经把"缝/裂开"
说清楚了，再叠两个近义术语只是把信号摊薄（与第九版"纯术语串打坏第 1 名"同一个机制）。
另一处 `rerank` 的 4→3 是重排近分抖动（那题词表没碰，`hybrid` 一动不动，正好排除词表）。

**没做完的**：检索侧只量了口语集、单次运行（差 0 题只能说"没观察到代价"）；28 条里只有 2 条被这次改动
碰到，其余 26 条不等于没错，只等于没测到；负例集 7 条是我自己写的，同样带着"我想到哪几种
错法"的偏差。`intent` 是标签不是分类器——它不会因为"这题像政策题"就去触发。

**验证**：553 → 569 passed（新增 `tests/test_term_map_constraints.py` 16 条），`ruff` 干净。

---

#### 第十二版：报告身份补源码指纹，并从干净提交重跑（2026-10-29）

专家复审 P0-2。二十份报告写着「代码 `3c77c36`（有未提交改动）」——`rev-parse` 在脏工作区里
描述的不是跑出这组数字的那份代码，等于这批报告的代码身份是空的。

补的是**内容哈希**（`src_sha`），不是 `git diff`：diff 只说"跟某个提交比变过"、不说"变成
什么"，两份都标着"有改动"时救不了你。范围刻意收在 `rag4prefa/**` + `scripts/eval.py` 的
`.py`/`.json`——改 README 或加个一次性诊断脚本不该让上一份报告显得不可比，指纹一旦对无关
改动敏感，人就会开始无视它。词表的 `.json` 必须在内：报告名里只有词表的文件名，原地改几条
映射时文件名一个字都不变。`--print-fingerprint` 供人当场复算，否则印出来的只是另一个要信
的字符串。

**两个当场踩到的坑**：①`.gitattributes` 是 `eol=lf` 但 Windows 上 `core.autocrlf=true`，
工作区全 CRLF，不归一则同一份代码在 Linux 上算出另一个 sha——"可复算"当场失效；②
`git status --porcelain` 首行是 ` M x.py`，整段 `.strip()` 吞掉状态位前的空格，按固定列切
路径得到 `cripts/eval.py`（老 bug，以前没人用过这些文件名）。

**重跑结果：9 份报告、40 行指标，38 行一字不差。** 参数逐份照旧报告头部抄，不按今天的默认
值重跑。变的 2 行在同一份报告里**对调**（`rerank` 0.839→0.871、`rerank_parallel` 0.871→0.839）
——不是"词表变好了"，是 1 题在两行之间换了位置。旧报告的数字本身可信，缺的只是身份。

**顺着这条线量出重排抖动：1 题 / 3 分钟。** 同批次两次运行，`rerank` 行配置完全相同（不含
术语通道、重排读原话），17:50 是 0.871、17:53 是 0.839；**同批的 `hybrid` 行 0 题变化**。
责任锁死在远程重排侧，不是代码/语料/模型。那道题是「施工图审查评分表…Q1 项的满分」：融合池里
稳定第 1，9 分钟内四次运行 `rerank` 给出 无/4/无/5。一题 = 0.032。

**三句话因此要改**：

1. 我上一轮对专家说"0.871 与 0.839 是 rrf 与 zscore 的正常差值，不是漂移"——**对了一半**。
   rrf/zscore 确实差 1 题（17:47 vs 17:50），zscore 自己在 17:50 与 17:53 之间也在同两个值
   之间跳（也 1 题）。**两个效应撞在同一对数字上**，31 题分母下用总数分不开，只能靠"同批次
   对照行"判。
2. "词表在专业问法集上 +0.032"**符号不稳**：旧（脏）append +0.032、干净 terms −0.032、
   干净 append −0.032。降级为"看不出影响"。**"append 上生产"的依据只剩口语集 4 赢 0 输**。
3. **误触发的检索代价第一次量到了，而且 `append` 确实挡住了它**：`pipeline-hidden` 在
   「架空楼地面…检修口宜设在什么位置」上误触发，`terms` 形态把 `rerank_parallel` 从第 1 名
   打到前 5 之外（重排查询换成了术语串，不再读"检修口"）；`append` 保留原话，名次仍第 1。
   同一处误触发，代价完全取决于形态——比 15 题的 14/15 更贴机制。

**验证**：569 → 580 passed（新增 `tests/test_report_identity.py` 11 条，三处判据各打掉验过
会红），`ruff` 干净。9 份旧报告原样归档到 `eval/reports/archive/*.as-of-<提交>-dirty.*`，
不删。完整记录见 `eval/report-identity-20261029.md`。

---

## 10.6 第四十八轮：把词表的收益拆开，再用两批新题量它的泛化（2026-10-29）

专家复审留了两条 P1：①词表那 +0.080 是"多召回一路候选"与"换掉重排查询"的合成，分不开；
②词表泛化未证（只覆盖 15/50，10 道新覆盖只救 1 道）。这一轮各做一个。

### P1-1：2×2 固定池消融（`scripts/term_ablation.py`）

`eval.py` 只能给"两个都开"和"两个都关"两行——基线的 `baseline_rerank_query` 有"带改写串
就退回原话"的保护，"池不动、只换重排查询"那一格在报告里根本不存在。新脚本把四格补全：
池（原话 / +术语通道）× 查询（原话 / 原话+术语串）。

同一题的两个池各算一次，池内每条候选被四种组合各打一次分。**池深 25 与 50 从一次 50 深的
全序里派生**（融合排序与 `candidate_k` 无关，25 池 = 50 池前 25 条；重排对池内每条独立打分）。
两条前提各有用例，且测试**直接调脚本函数**——第一版在用例里把派生逻辑重写了一遍，"把
`pool[:depth]` 改成 `pool`"这个变异从断言下面溜过去，9 条全绿。

14 题（28% 命中词表），池深 25（生产值）：

| 格 | 池 | 查询 | Recall@5 | 池内含 gold |
|---|---|---|---|---|
| P0Q0 | 原话 | 原话 | 0.643 | 11/14 |
| P0Q1 | 原话 | +术语串 | 0.714 | 11/14 |
| P1Q0 | +术语 | 原话 | 0.786 | **14/14** |
| P1Q1 | +术语 | +术语串 | **1.000** | 14/14 |

逐题归因：只靠术语通道 1 题、只靠重排查询 0 题、两条路都行 1 题、必须两者 3 题。
**"池内含 gold" 11/14 → 14/14 是最直接的机制证据**：三题的 gold 原本根本不在池里，
重排再准也救不回来。上一轮那句"重排只是池深假象"这次有了独立测量。

池深 50 的 P1Q1 反而低（1.000 → 0.929）：承重墙那题在更大池子里被近分候选挤出去。
全量基线 0.720（25 深）/ 0.680（50 深）与历史报告逐位相同，说明口径一致。
记在 `eval/termmap-ablation-20261029.md`。

### P1-2：两批独立题集（41 题，触发 1 题）

口语 50 题里 35 题是"拿检索器找答案"时出的，31 题的 gold 出题时已在候选里——用它验泛化
等于用筛过的样本证总体。所以另做两批：

**A 批（27 题）**：固定种子抽条文 → 读正文写问句。**这批有问题**：我读过条文，写出来的
"口语"一定带条文的词，题也太容易（基线 0.815）。

**B 批（14 题）**：先写 40 条口语问句（没查库）→ 用规范术语做子串匹配找条文
（`scripts/find_gold_by_text.py`，不用向量也不用 BM25，避免"检索器给什么就标什么"）。
40 条留下 14 条，其余 26 条照实丢弃。40 条原始问句一并提交，让"只留 14 条"这件事能被看见。

| 题集 | 题数 | 基线 `rerank` | 触发 | 生产形态 | 差 |
|---|---|---|---|---|---|
| A 批 | 27 | 0.815 | 1 | 0.852 | +0.037 |
| B 批 | 14 | **0.500** | **0** | 0.500 | 0.000 |
| 口语 50 题 | 50 | 0.680 | 14 | — | — |

**题越难、越需要它，触发越少**：B 批 0.500 一次没出手，A 批 0.815 反倒动手 1 次。
逐题判"该不该命中"分四类（D 该命中 / C 有落差但无此条 / B 题材外 / A 无落差）：
41 题里 D 类 9 题，实际认出 1 题；误触发 0（唯一那次触发是对的，7 条负例闸门全过）。

差在**词形**不在语义：「有味道上来」不是「返味」、「看到水印」不是「渗漏」、
「补多少」不是「补贴」、「板材」不算「材料」。跟 A17 与 A25/A26 是同一种病——同一件事，
说出"补贴"两字就中，说"补多少"就不中。

**这一轮不改词表**：现在改就是拿刚出的题调词表，下一批又会重演（v1→v2 已经演过一次：
条数翻 3.5 倍，净增益没涨）。`R4P_TERM_MAP` 继续默认留空。记在
`eval/termmap-independent-20261029.md`。

### 顺带修掉的

- 独立题集有一题的 gold 被过滤器吃掉：「到2027年…」被 `filters._SINGLE` 当成文档年份，
  而正文在 2024 年的文件里。**这是产品侧的语义歧义**（政策文件里的"到2027年"是目标年），
  本轮只改问句，行为记在 P1-2 报告的遗留项里。
- 消融脚本补了进度输出与重试（三次仍失败就抛，不吞掉失败）。

**验证**：580 → 591 passed（新增 `tests/test_p1_ablation.py` 11 条，三处判据各做变异验过
会红），`ruff` 干净。两份独立题集的评估报告都从干净提交重跑（`76ef6ca`，`src_sha 9cb5bf5ad67c`）。

---

---

## 10.7 第四十九轮：按专家五步走——年份语义、证据闸门、题集冻结、识别稿体检（2026-10-29）

专家复审给了五步顺序，这一轮按顺序做，**没有跳步**（第五步"开放词表默认值"的前提没满足，
所以仍然关闭）。

### 第一步：年份过滤器把目标年当成资料年（`filters.py`）

`_SINGLE = (20\d{2})\s*年` 见到年份就筛，于是「到2027年，老旧住宅电梯要怎么处理」被理解成
「只看 2027 年的文件」，而正文在 2024 年的文件里——**检索前的静默否决**，后面的混合召回、
重排、词表全都救不回来。

判据改成「年份后面紧跟资料标记才筛」：强标记（年版 / 年文件，本身就是版本的意思）、
弱标记（标准/通知/方案…，必须写成「A年的X」，否则「2027年标准体系」的目标年读法会误判）、
发文动词（年发布 / 年印发）。再加两条：`到/至/截至/目标 + 年份` 一律当目标年跳过；
一句话两个资料年份取跨度。

**行为变化要记住**：「2026年省里对建筑业监管有什么安排」这类「年份在句首、后面不接资料名词」
的问句不再缩小范围。候选池变大该由排序承担，不该由过滤器静默清空。
四个题集 122 题的 gold 全部照常解析（复核过）。

变异验过两条判据会红：去掉目标年豁免、去掉年份联合传递。

### 第二步：证据闸门（`rag4prefa/sufficiency.py`）

先说清楚这一层的边界：它**不判断「这段文字是否支持这个结论」**，只做「能命名一个确定性理由
时拒答」。三类信号，只有前两类阻断：

| 信号 | 判据 | 结果 |
|---|---|---|
| `no_hits` | 一条候选都没有 | 收拢到一处（原来只在 `stream_answer` 里判） |
| `geo_absent` | 问句点名的地域，库里没有**一份关于它的文件** | 地域类 7/7 |
| `weak_top` | 最相关候选分数低于门槛 | 实测不可用，默认关 |

**判据用文件名不用正文**，这是本轮最容易做错的一处，第一版就写错了：拿「全库正文有没有提过
这个地名」判断，而本库正文提到「北京」的片段有 17 条（参编单位名单、企业简介"承建了北京
APEC 峰会"、区域规划"积极接轨上海"），却没有一份北京的资料——闸门对这一整类永远不触发，
而且不报错。改成文件名口径后，覆盖集正好是 浙江 22 / 绍兴 7 / 宁波 5 / 台州 4 / 杭州 3 /
湖州 2 / 嘉兴 1。

**分数闸门默认关，是实测结论**：122 道库内题 top1 分最低 0.2692，15 道库外题最高 0.9999，
分布完全重叠，库外题常常比库内题还高（「北京补贴标准」0.9996）。这个重排服务判的是
「相关不相关」，不是「能不能支撑答案」。写进 `config.py`，换模型要重量。

结果：库外 15 题拒 7 题（**全是地域类，geo 7/7、time 0/2、missing_topic 0/3、near_topic 0/3**），
库内 122 题 × 6 种召回方式**误拒 0**。做不到的 8 题要的是语义蕴含，规则层没有。

被判死的不是"难度"而是"判据不存在"：`time`/`missing_topic`/`near_topic` 三类都没有确定性特征，
下一轮该做 LLM 判官（可独立评估，与"在提示词里求模型别猜"的区别就在这里）。

`refuse_mode` 三态 off/warn/on，默认 on；warn 只写 meta，用来量误拒率。拒答时把缺的东西
说清楚（"库里没有一份资料是关于北京的"），不把候选送进模型；前端状态栏同步显示理由。

### 第三步：词表与题集冻结；没找到答案的问句不丢

- v3 词表与 41 道独立题**冻结**，身份（文件名 + sha）现在写进消融报告头。
- B 批 40 条问句里没找到条文的 26 条，之前只交代了一句"留 14 条"，问句本身没入库。
  现在收进 `eval/dataset.independent-b-unlabeled.jsonl`：不参与 Recall，只做两件事——
  看闸门行为、留给第二个人复核。**这件事不能由写题的人做**（自己批自己的卷子）。

### 第四步：识别稿体检重跑 + 从干净提交重跑四份题集

`eval/ocr-audit-20261029.md`：13 份识别稿，4 份有具体线索（GB55037 五处条文断档、GB50210
一处断档 + 表 15.0.6 缺表头、DBJ33T1165 与 JGJT477 各一处）。对照 103 个 gold：
**只有 1 个落在断档章节内**（且它本身在库里），而**20 个 gold 来自识别稿文档**。
后一个数字要记住：评测里 19% 的 gold 依赖视觉模型认字的质量。

四份题集从 `db99c1e`（源码 sha `6cd2081a5024`，工作区干净）重跑：
基础 31 题 hybrid 0.871 / rerank 0.839；口语 50 题 hybrid 0.640 / rerank 0.680；
独立 A 0.852 / 0.815；独立 B 0.143 / 0.500——与历史逐位相同。

### 第五步：词表默认值——不动

前提（独立题集上可重复的净收益 + 误触发与拒答达标）没满足，`R4P_TERM_MAP` 继续留空。
`.env.example` 里那句「开了 +0.080」已经改掉：那是覆盖题的局部数值，不是任意问句的期望收益。

### 顺带

消融报告头补了**实验身份**：代码提交 + 源码 sha + **实验脚本自身 sha** + 词表 sha + 数据 sha。
通用指纹只覆盖 `rag4prefa/**` 与 `scripts/eval.py`，而这个实验的测量逻辑在
`scripts/term_ablation.py` 里——只报前者会出现"生产代码一致、实验脚本换了"的两份报告。

**验证**：591 → 610 passed（新增 `tests/test_sufficiency.py` 15 条、`tests/test_filters.py` 5 条；
五处判据各做变异验过会红——地域判据失效、两个门槛共用一个数、闸门算了不挡、只看正文不看
文件名、地域判据退回正文口径；最后一条第一版没抓住，因为测试自己注入了判据，改成让闸门走
自己的默认口径才抓住）。`ruff` 干净。

---

## 10.8 第五十轮：库外题扩到 33 题，判官实测（2026-10-30）

上一轮末尾定的下一步是「先把 `near_topic` 从 3 题扩到 20 题以上，再做 LLM 判官」。
两步都做了，但**判官没有通过**，没有接入生产。

### 一、`near_topic` 扩到 21 题，写法比数字重要

`eval/refuse-near-topic-protocol.md`：只读文件名（75 份），按"标准/政策这类文件通常不写的
属性"出题——工日、税率、保险费率、品牌型号、投诉渠道、产值、社保、排名、贴息、就业人数。
**定稿前不跑检索、不看候选**，共写 34 条；再用 `find_gold_by_text.py` 做子串核验
（不走向量、不走 BM25），剔掉 13 条其实答得出的（保修 5 年、撞击声、VOC 限量、试点企业 12 家）。

第 13 条**是被判官反过来抓出来的**：「能少产生多少建筑垃圾」我按「吨/减量」找 0 命中，
语料原文却是「现场固废物减少了 70%」。所以：**子串 0 命中只能证明"我没想到的那个词不在
库里"，证明不了"库里没有"**。同义词覆盖不到，就会把答得出的题记成该拒。

### 二、判官 v1：库外 33/33，库内误拒 41/122

`sufficiency.judge_by_llm`（`prompts.SYSTEM_JUDGE`）判"可/不可 + 一句理由"；
`scripts/judge_sufficiency.py` 先冻结候选再判，规则闸门与判官看同一批片段。
结果（`eval/refusal-judge-20261030.md`）：

| | 规则闸门 | 判官 v1 |
|---|---:|---:|
| 库外 33 题（该拒） | 7 | **33** |
| 库内 122 题（不该拒） | 0 | **41** |

规则一条也拦不住的 `time` 2 / `missing_topic` 3 / `near_topic` 21，判官全拦住了。
代价是 34% 的库内题被误拒——按产品看不可接受。

**41 题必须拆开看**（`scripts/diagnose_judge.py`）：21 题的 gold 就在它看到的 5 条片段里
（判官读走了眼），20 题是"这批片段确实答不了"（责任在检索）。后者说明它在正确识别
"证据不足"，只是把"检索没送上来"一并报成了"库里没有"。

主体失败形态一句话：**问句要结论，语料给要求**。「吊顶装灯会不会塌」↔「应满足荷载计算要求」；
「台面多高才不累腰」↔「宜为 800/850/900mm」；「怎么判断能不能住」↔ 检测项目与限值。
人能一步读过去，判据不认。另有两条更具体的：把"后排式 150 / 下排式 250"读成"数值冲突"；
把 2 年缺陷责任期与 5 年保修期读成"规定不一致"。

### 三、判官 v2：按问法分型，**没测完**

`prompts.SYSTEM_JUDGE_V2` 先分型再判：取值型（多少/几天/多少钱/哪个部门）必须出现同类型取值；
做法型（会不会/能不能/怎么弄）给出对应措施就算可；两类共同保留"别的地区/年份/项目 → 不可"。

跑到 118/156 报 429，退避重试后仍失败，最小请求探测返回 **`insufficient balance`**——
是模型账户余额耗尽，不是限流。已判部分看得到形态变化（「装修也要把图送审吗」v1 不可 → v2 可），
但**没有全量数字，不给 v2 结论**。报告里明写了预期的代价：分型会放过"问了做法、库里确实没有"
的约 6 题，目标是把误拒压到 ≤5% 且库外拦截不塌，达不到就是这条路走不通。

### 四、判官不进生产

`refuse_mode=on` 仍然只有规则闸门。判官代码只被评估脚本调用，`.env.example` 里没有它的开关
（写清了原因）。可接入的三种形态按风险排序记在报告第六节：只做提示 → 只对取值型问句阻断 →
当作"扩大召回再判一次"的信号。

### 五、顺带修掉的

**判官结论也要按题缓存**。第一次跑 v2 时候选全在缓存里、判官结论没落盘，被余额打断后
118 次调用白花。现在 `judge-hits.json`（候选）与 `judge-verdicts.json`（结论，key 带版本与 k）
分开存，续跑只补缺口——这也是以后换提示词做 A/B 的前提。
`--from-dump` 让重排版式不再花模型调用；`--rejudge` 才是真的重判。

**验证**：610 → 624 passed（新增 `tests/test_judge.py` 14 条；两处变异验过会红——
「不可」出现在理由行时被判成否定、格式不可解析退化成 `False`）。`ruff` 干净。

---
---

## 11. 深入阅读顺序


1. `REFACTOR_PLAN.md`（本目录）—— 重构方案 + **第十二节有实施复盘**，记录计划与现实的 5 处偏差，值得先看。
2. `F:\project\PyProject\milvus\PROJECT_REVIEW.md` —— 原系统评估报告，说明"为什么要重写"的每一条证据都能定位到行号。
3. `README.md`（本目录）—— 面向使用者的安装、API、已知局限。
4. 原系统源码 `F:\project\PyProject\milvus\Vector_search_plus.py`（1 991 行）—— 只在需要对齐某个行为时查阅。
