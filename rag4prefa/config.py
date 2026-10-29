"""集中配置：唯一的配置入口，全部来自环境变量 / .env，代码里不再出现密钥与魔法值。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="R4P_", extra="ignore"
    )

    # 数据
    data_dir: Path = Path("data")

    # 向量化。默认硅基流动国际站 + Qwen3-Embedding-0.6B：1024 维，与下面的 1024 对得上。
    #
    # 硅基流动的国内站（siliconflow.cn）与国际站（siliconflow.com）是**两套独立账号**，
    # Key 不通用，模型目录也不同：国内站有 BAAI/bge-m3，国际站只有 Qwen3-Embedding 系列。
    # 在 .com 申请的 Key 填给 .cn 的地址只会在第一次调用时收到 "Token is invalid"，
    # 看起来像 Key 被删——这是本项目最容易搞错的一处，两站对应关系写在 .env.example。
    embedding_base_url: str = "https://api.siliconflow.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 0  # 0 = 探测后写库
    embedding_batch: int = 32  # 百炼 text-embedding-v3/v4 只能 10
    embedding_timeout: float = 60.0

    # 生成
    llm_base_url: str = "https://api-inference.modelscope.cn/v1"
    llm_api_key: str = ""
    # 魔搭的模型目录会下线旧 ID，写死一个名字迟早会收到
    # "Model id : ... has no provider supported"。拿不准时先列一遍当前可调用的：
    #   curl -s https://api-inference.modelscope.cn/v1/models
    # 这个列表不需要鉴权，直接能看到该平台当前供哪些模型。
    llm_model: str = "Qwen/Qwen3.5-35B-A3B"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # ── OCR（可选，默认关闭）──────────────────────────────
    #
    # 扫描件没有文字层，只能从像素认字。项目约束是不引入本地模型，所以走**视觉模型**
    # 的 OpenAI 兼容接口：一次调用同时拿到文字、标题层级和表格结构，正好是切块需要的
    # 三样东西；而传统 OCR（Tesseract / PaddleOCR）只给文字，还要往磁盘落一个模型。
    #
    # 实测（2026-10-27，同一页 150dpi）：
    #   · 硅基流动国际站 Qwen/Qwen3-VL-8B-Instruct  18s，内容准确；
    #   · 同站 Qwen/Qwen3-VL-32B-Instruct           42s，且**层级判得更差**
    #     （把正文续行标成了 #）。所以默认用 8B，不是越大越好。
    #   · 魔搭当前列出的两个视觉模型都调不通：ERNIE-4.5-VL 报 invalid_model、
    #     InternVL3_5 报 user token expired——和 LLM 那边一样，`/v1/models` 里
    #     列出来不等于这个账号能调，只有真发一次请求才算数。
    #
    # 与 embedding / LLM 是三次独立的三方调用，Key 不通用，所以单独一组配置。
    # `off` 时整条 OCR 路径不加载，解析行为与之前完全一致。
    ocr_mode: Literal["off", "auto", "all"] = "off"
    ocr_base_url: str = ""
    ocr_api_key: str = ""
    ocr_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    ocr_dpi: int = 150
    # 单份文档最多识别这么多页。不是性能限制，是**账单闸门**：一份 300 页的扫描件
    # 按 18s/页 要跑一个半小时、付三百次调用，而用户很可能只是拖错了文件。
    # 超了直接拒绝并说清怎么放开，不做"先跑一半"。
    ocr_max_pages: int = 200
    # 页与页之间互相独立，并发只影响等待时间：24 页串行要七分钟。
    ocr_concurrency: int = 4

    # 检索
    recall_k: int = 50
    top_k: int = 8
    rrf_k: int = 60
    # 稠密与稀疏两路候选怎么合成一套顺序（见 `ask.fuse_scores`）。
    #
    #   zscore（默认）——各自在本路内部标准化后加权求和。两路候选几乎不相交时
    #                    （口语问句在规范文本上做 BM25 就是这种情形），等权 RRF 会
    #                    把展示位对半分，让单路强命中输给双路弱命中。
    #   rrf          ——经典 Reciprocal Rank Fusion，只用名次。保留为可回退项。
    #
    # 换它等于换排序行为，已标注在评估报告的上下文里（`fusion` 字段），
    # 两份报告的数字不能跨这个字段直接比大小。
    fusion: Literal["zscore", "rrf"] = "zscore"
    chunk_max_tokens: int = 480
    chunk_min_tokens: int = 80

    # 重排（可选）
    rerank_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # ── 受控词表（可选，默认关闭）──────────────────────────
    #
    # 口语问句卡住的往往不是语义相似度，而是**词不是条文里的那个搭配**：「会不会掉下来」
    # 查不到，条文原话「专用连接件」一查就进第 2 名。这一层把公众说法映射成规范术语，
    # 命中的术语作为**独立通道**参与融合（原话那一路一条都不动），并让重排也读到它。
    #
    # 留空即关闭，行为与没有这一层时完全一致。开关打开后每次问答多一次 embedding
    # 往返（一条术语通道），这是收益之外必须一起看的成本。
    term_map: str = ""
    # 术语注入形态：`joined` = 所有命中术语拼成一串（默认），`terms` = 每个术语一条通道。
    # 取 joined 是因为两者**Recall 相同、名次只差 1~2 题**（2026-10-29 在 50 题口语集上
    # 同一天重跑：四条行 Recall 一模一样，`hybrid_parallel` 2 题前移、`rerank_parallel`
    # 1 题后移且仍在 top-5），而 joined 少 N−1 次远程 embedding。
    # 别把这句写成"逐题一字不差"——那是 15 题（v1 词表、重排读术语串）上的结论，
    # 在 50 题上不成立（见 HANDOFF 第十版）。形态只在"重排读原话"时才有实质差别：
    # 那时拼串会把 BM25 的单个精确搭配稀释掉。
    term_map_form: Literal["terms", "joined"] = "joined"
    # 重排读哪个串：
    #   append（默认）—— 原话 + 术语串，术语只做加法；
    #   terms          —— 只读术语串；
    #   none           —— 只读原话。
    # **这里的默认值与评估脚本的默认值（`none`）不同，是有意的**：评估的默认值定义
    # "完全不做改写"的对照组，生产的默认值定义"开了词表就用验过的那个形态"。
    # 依据见 `ask.build_rerank_query`：纯术语串会把三道原本已排第 1 的题打出去。
    term_rerank: Literal["none", "terms", "append"] = "append"
    # 术语通道的融合权重（原话恒为 1.0）。取评估时用的那个值，别在部署里另调一套。
    term_rewrite_weight: float = Field(default=0.5, gt=0, le=1)

    # ── 证据闸门（可选拒答，默认开启）──────────────────────
    #
    # 候选非空时，结果一定会被送进大模型；`prompts.SYSTEM_ANSWER` 里那句"片段不足就说
    # 资料中未涉及"只是提示词约束，不是保证。库外集 6 道题在每种检索方式下都召回了
    # 非空候选（北京问出宁波的方案、日本问出绍兴的图集），这是**检索层没有地域概念**
    # 暴露出来的，不是排序误差。
    #
    #   off  —— 不判，行为与加这一层之前完全一致；
    #   warn —— 判、写进 meta 与 `/api/health`，但不影响回答（用来观察误拒率）；
    #   on   —— 判到就拒答，不把候选送给大模型。
    #
    # 判据只有两类（见 `rag4prefa/sufficiency.py`）：候选为空、问到的地域全库没提过、
    # 最相关候选的分数低于门槛。**它不判断语义蕴含**，所以没有"看起来很自信就放行"。
    refuse_mode: Literal["off", "warn", "on"] = "on"
    # 两个门槛量纲不同（重排分 vs 余弦相似度），必须分开配，不能共用一个数。
    # 0 = 不设分闸（默认）。
    #
    # **默认 0 是实测结论，不是偷懒**：2026-10-29 用 `scripts/calibrate_refusal.py`
    # 在当天服务上量了 122 道库内题与 15 道库外题的 top1 分——库内最低 0.2692、
    # 库外最高 0.9999，两个分布**完全重叠**，库外题的 top1 常常比库内题还高
    # （"北京补贴标准"拿了 0.9996）。也就是说这个重排服务对"主题沾边但答不了"
    # 的题照样给高分，不存在一个能把这批分开的门槛。
    # 换重排/embedding 模型后必须重量，别把这里的 0 当成"这个信号没用"的永久结论。
    refuse_min_rerank: float = 0.0
    refuse_min_dense: float = 0.0

    # 向量后端：auto 先试 Milvus，不可用则退到 numpy（会在 /health 与启动日志中说明）
    vector_backend: Literal["auto", "milvus", "numpy"] = "auto"
    milvus_uri: str = "data/vectors.db"
    milvus_collection: str = "chunks_vec"

    log_level: str = Field(default="INFO")

    # 未携带用户 Key 的请求，是否允许回落到 `.env` 里那份**服务端公共 Key**。
    #
    # 单机演示必须开着：CLI、curl、旧客户端都不带 `X-R4P-*` 头，关掉就什么都调不通。
    # 但公网部署必须关掉——那份 Key 的额度是"任何能访问端口的人"共用的，匿名请求
    # 可以无限消耗它，而账单只有一个人付。关掉之后，未带 Key 的请求会收到 401，
    # 并被告知去哪里填自己的 Key，而不是先烧掉几百次调用再报额度不足。
    allow_server_key_fallback: bool = True

    # 管理口令：保护"清空库 / 删项目 / 删文档"这三条不可撤销的入口。
    #
    # 它不是账号体系，只是一把共享口令（请求头 `X-R4P-Admin-Token`），因为在这之前
    # 这三条路由**没有任何身份校验**：`confirm: "清空"` 只防手滑，谁能访问端口谁就能删。
    #
    # 规则（见 `api.require_admin`）：
    #   - 配了口令 → 校验，对了才放行（单机也可以配，等于给自己加一道门）；
    #   - 没配、且 `allow_server_key_fallback=false`（即"给别人用"的部署）→ 直接 503，
    #     说清去哪儿配。宁可删不了，也不能让任何访客删掉全库；
    #   - 没配、且是单机演示（fallback=true）→ 不校验：那时打开页面的人就是主人，
    #     强行要求口令只会让本地开发多一步。
    admin_token: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def missing_credentials(self) -> list[str]:
        """返回缺失的关键配置，供 /health 直接暴露，而不是启动时静默降级。"""
        missing: list[str] = []
        if not self.embedding_api_key:
            missing.append("R4P_EMBEDDING_API_KEY")
        if not self.llm_api_key:
            missing.append("R4P_LLM_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
