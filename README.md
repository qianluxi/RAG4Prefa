# RAG4Prefa

面向工程课题文档的检索问答系统：把 `.docx` / `.pdf` 设计文档 / 申报材料 / 规范按章节建成知识库，
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
| 绑死 Windows + Word，进不了容器 | 纯 python-docx / pymupdf 解析，Linux/Docker 可跑 |
| pdf 没有样式表，章节结构全丢 | 按字号 / 加粗 / 行距反推层级；编号只定级，页眉页脚与页码剔除 |
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

### Windows 上一键启动：`start.bat`

装完依赖后双击根目录的 `start.bat` 即可。**默认端口是 8899**（不是 8000，也不是
Streamlit 的 8501），`start.bat 8000` 可改。窗口里看到

```
INFO:     Application startup complete.
```

才算起来了，这时再打开 `http://127.0.0.1:8899`。它不解压、不安装，只做四件事，
而这四件恰好是最容易做错的：

1. **`cd` 到脚本所在目录。** `R4P_DATA_DIR=data` 是相对路径，从别处启动会安静地
   新建一个空库——页面上显示"没有语料"，数据其实还在 `data\app.db` 里。
2. **用 `.venv\Scripts\python.exe`。** 系统 python 没装依赖。
3. **先探端口，再谈地址。** 端口已被监听时**不启动第二个实例**，而是把占着端口的
   PID 与进程名打出来，并告诉你可以打开哪个地址；失败一律 `pause`，窗口不会一闪
   而过。早期版本是先 `echo 地址` 再 `serve`，端口被占时 uvicorn 报 `[Errno 10048]`
   后窗口立刻关闭，用户只看到闪过一行地址、浏览器打开拒绝连接，反馈上来是"双击没
   启动成功"——地址只能代表"打算用"，不能代表"已就绪"。
4. **启动前跑一次 `doctor`**，缺 Key 或索引不一致就在这里报出来，而不是等点了
   提问才看到 503。

脚本不该被改的细节：它是 **GBK 编码 + CRLF**。`cmd.exe` 按系统代码页解析 `.bat`，
UTF-8 的中文不只是乱码——多字节字符会吞掉后面的命令字符（实测 `:fail` 被吃成 `ail`）。
仓库里靠 `.gitattributes` 的 `working-tree-encoding=GBK` 维持"存的 UTF-8、落的 GBK"。
上述约束由 `tests/test_start_bat.py` 钉住（顺序、失败必停窗、只认 LISTENING、编码）。

### 怎么停：`stop.bat`

优先在服务窗口里按 `Ctrl+C`。**点窗口右上角的 X 不算停止**：那只关掉了 `cmd`，真正
监听端口的是 `python.exe`（`.venv` 里那个是启动器，服务是它拉起的子进程），所以再
双击 `start.bat` 还是同一句"已经在监听"。这时双击 `stop.bat`：

```bat
stop.bat                :: 停默认端口 8899
stop.bat 8900           :: 停指定端口
stop.bat 8899 /force    :: 健康检查没回应（服务卡死）时强制停
```

它会先连 `/api/health` 看 `app` 字段，**只停自报 `rag4prefa` 的进程**；认不出来就只把
PID 和进程名打给你，不动手——按端口号杀进程，误杀别人家的服务只需要一次巧合。
这些判据由 `tests/test_processcontrol.py`（谁在听 / 是不是本程序 / 敢不敢动手）与
`tests/test_stop_bat.py`（参数透传、退出码、编码）钉住。

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
  └── adapters/       embed / llm / vectors / rerank / parsers（docx / pdf / 共用标题规则）
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
| POST | `/api/projects/{name}/documents` | 上传 docx / pdf（multipart）→ `job_id` |
| GET | `/api/jobs/{id}` | 入库进度与逐文件结果 |
| GET | `/api/projects` | 项目列表（文档数/片段数为真实值） |
| GET | `/api/projects/{name}` | 详情 + 章节分布 + LLM 摘要（带缓存） |
| DELETE | `/api/projects/{name}` | 级联删除（SQLite + 向量） |
| POST | `/api/search` | 非流式检索（调试与评估用） |
| POST | `/api/chat` | SSE 流式问答 |

SSE 事件：`meta`（生效条件、候选数、各阶段耗时）→ `token`* → `citations` → `done`；异常走 `error`。
`citations` 每项带 `page`——**该片段自己**的起始物理页码（1-based，PDF 才有；docx 为 `null`，
用章节路径定位）。它不是"本章节从第几页开始"：一节跨 4~29 页时，各片段按自己落在的页返回。

前端交互：答案里的 `[n]` 角标可点击，会滑出右侧「引用来源」抽屉并高亮对应片段；
抽屉默认收起，也可点气泡下的「查看来源」按钮打开。检索到生成期间气泡下方显示
「正在检索回答中…」的动态提示。

## 测试与评估

```bash
pytest -q                                  # 322 项，全部离线（假向量 + 本地假 OpenAI 服务 + 临时 Milvus/numpy）
python scripts/eval.py --dataset eval/dataset.jsonl --top-k 5
python scripts/eval.py --dataset eval/dataset.paraphrase.jsonl --top-k 5
```

评估集分两份，**不要混成一个平均数**。下面是真机实测（库内 323 块 = 315 块标注语料 + 8 块存量计算书，
`Qwen/Qwen3-Embedding-0.6B` 1024 维，chunk 80~480 token，`recall_k=50`，`rrf_k=60`，
重排 `Qwen/Qwen3-Reranker-0.6B`）：

| 召回方式 | 基础 30 题 Recall@5 | MRR | 改写 10 题 Recall@5 | MRR |
|---|---|---|---|---|
| dense（纯向量） | 0.817 | 0.704 | 0.500 | 0.333 |
| sparse（纯 BM25） | **0.950** | **0.834** | 0.500 | 0.275 |
| hybrid（向量 + BM25 + RRF） | 0.900 | 0.721 | 0.700 | 0.520 |
| rerank（混合 + 重排） | **0.950** | 0.732 | **0.800** | **0.542** |

三句话读完这张表：

1. **基础集上纯 BM25 就打了 0.95，混合召回反而低 0.05。** 语料里 96% 是一份软件手册，
   按钮名、参数名这类专有词密度极高，分词后几乎等于原样命中——这类语料上融合是稀释而不是增强。
2. **改写集才是语义检索的战场。** 同一个"两面都叠合的那种墙要打几折"，不用原文术语之后
   BM25 掉到 0.50，混合召回 0.70，开重排 0.80。
3. **只报一份会给出误导性的结论。** 单看基础集会得出"检索已经很好"，单看改写集会得出
   "BM25 没用"，两个都不对。

标注改动后要跑 `python scripts/relabel_eval.py`：它按每题的正文锚点重算 `gold_chunk_id`，
并校验 `eval/corpus.json` 里的语料指纹。**改了解析器就必须重跑**——`chunk_id` 是内容寻址的，
章节路径变一个字 id 就全换，旧标注会静默失效（评估照样跑完，只是 Recall 莫名变低）。

报告本身带「坐标系」一行（模型、维度、chunk 参数、`recall_k`、`rrf_k`、重排模型），
因为换掉其中任何一项，前后两次数值就不可比。实测样本存在 `eval/reports/`。

CI（`.github/workflows/ci.yml`）在每次 push / PR 上跑同一组命令：`ruff check rag4prefa tests scripts`
→ `pytest -q`。全部离线，不需要任何 Key。

`scripts/eval.py` 输出四组对照——`dense`（纯向量）、`sparse`（纯 BM25）、`hybrid`（向量 + BM25 + RRF）、
`rerank`（混合 + 重排）——这是"混合检索和重排各自带来多少收益"的量化依据。
**没配 `R4P_RERANK_URL` 时报告会显式写「未启用重排」**，而不是让 `rerank` 一栏悄悄等于 `hybrid`
（那样读者会把两行差异当成重排的收益；曾经就因为两行用了不同深度的候选池，差出过一个 +0.033）。

重排是**可选增强**：配了 URL 默认开启，也可以 `--no-rerank` 显式关掉。它多一次网络往返，
换来的是上面改写集那 0.10 的 Recall。`/api/health` 的 `rerank_configured` 会告诉前端
"这台机器到底有没有重排"，免得界面上留一个勾了也不生效的开关。

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
- **PDF 有两种进不来的情况，都会明确报错、不静默入库**：
  一是扫描件（整页是图片、没有文字层），需要 OCR 服务，与"不引入本地模型"的约束冲突，暂不做；
  二是字体缺 ToUnicode 映射——方正书版等排版软件常见，文字层里存的是字形编号，
  取出来是 `!"#$%&'()*` 这样的串，任何工具都无法还原，只能换一份能选中文字的版本。
  两者报的是不同的消息，因为用户能做的事不一样。
- **PDF 的章节层级是反推出来的**：依据字号、加粗、行距与编号，不是文档自带的样式表。
  标题与正文同字号、又不加粗、还不用编号的排版，会退化成一节（内容不丢，只是没有层级）。
  表格识别依赖框线，无边框的"空格对齐表"会被当成正文。
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
