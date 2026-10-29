"""问答用例：混合召回 → RRF 融合 → 可选重排 → 真流式生成。

关键修正（对照原系统）：
- 关键词参与**召回**（BM25），不再是在 Top-K 之后做子串过滤；
- 稠密与稀疏结果用 RRF 融合，而不是各取一半拼起来；
- 生成走 token 流，客户端首字延迟≈模型首 token 延迟。
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from . import consistency, repo
from .adapters.embed import Embedder
from .adapters.llm import LLM
from .adapters.query_expand import QueryExpander
from .adapters.rerank import NoopReranker, Reranker
from .adapters.vectors import VectorStore
from .config import Settings
from .consistency import IndexMismatch
from .domain import Filter, Hit
from .filters import describe, parse_filter
from .prompts import SYSTEM_ANSWER, build_answer_prompt


def rrf_multi(
    runs: Sequence[tuple[Sequence[tuple[str, float]], float]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """带权重的多通道 RRF：每个 (排序列表, 权重) 按名次累加 `weight/(k+rank+1)`。

    权重不是"顺手加的调参旋钮"，它是并行检索能不能成立的关键。同权（改写通道与
    原话通道都是 1.0）意味着改写词与原话平起平坐，干扰词照样能把正确片段顶下去——
    那就是首版「原话 + 术语拼成一个串」的老问题换了个写法。原话通道必须更重，
    改写只负责**补充**候选，不负责重排既有候选。
    """
    scores: dict[str, float] = {}
    for ranked, weight in runs:
        for rank, (chunk_id, _) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank + 1)
    return scores


def rrf(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion：只用名次，不用各自分数量纲，天然免疫"分数不可比"。"""
    return rrf_multi(((dense, 1.0), (sparse, 1.0)), k=k)


def _to_hits(conn: sqlite3.Connection, scores: dict[str, float], limit: int) -> list[Hit]:
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    details = repo.get_chunks(conn, [cid for cid, _ in ordered])
    hits: list[Hit] = []
    for chunk_id, score in ordered:
        info = details.get(chunk_id)
        if not info:
            continue  # 索引里有、事实表里没有 → 说明索引脏了，跳过而不是编造内容
        hits.append(Hit(score=score, **info))
    return hits


def _ensure_same_index(conn: sqlite3.Connection, version: int, stage: str) -> None:
    """确认"我看到的那一代索引"还没被换掉。

    为什么需要它：`ensure_queryable` 与真正检索之间隔着一次网络往返（算查询向量）。
    那几秒里另一个进程完全可以做完整量重建——换了模型、维度还一样，于是查询向量
    出自旧模型、索引已是新的一版。余弦照算、接口照返回，只有语义悄悄错位：**这类
    结果不报错，只是不可信**，比报错难发现得多。

    维护锁保护的是"写"，不能把查询也塞进去排队——一次重建要几分钟，让所有提问等
    它结束是不可接受的。所以用代次做一次轻量的版本确认（读一条 meta，成本可忽略）：
    算完向量后确认一次，检索完成后再确认一次。第二次是必须的——切换可能正好发生在
    `search` 执行期间，那时第一次确认已经通过了。
    """
    now = repo.get_index_generation(conn)
    if now != version:
        raise IndexMismatch(
            f"{stage}向量索引被换掉了（代次 {version} → {now}），本次检索作废。"
            "查询向量可能出自旧模型，继续用它查新索引会得到'能跑但语义错位'的结果。"
            "直接重试即可。"
        )


async def retrieve(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    question: str,
    filt: Filter,
    recall_k: int | None = None,
    candidate_k: int | None = None,
    expander: QueryExpander | None = None,
    rewrites: Sequence[str] = (),
    rewrite_weight: float = 0.5,
) -> tuple[list[Hit], Filter, dict[str, float]]:
    """返回 (融合后的候选, 实际生效的筛选条件, 各阶段耗时)。

    两个深度不是同一个数，必须分开：

    - `recall_k`：**每个通道**各取多少条（稠密、BM25 各自的上限）；
    - `candidate_k`：RRF 融合后**保留多少条**交给下游（重排、评测、展示）。

    旧实现把后者写死为 `max(recall // 2, top_k * 3)`，于是 `recall_k=50` 实际只拿到
    25 条。生产问答只取前 top_k 条，少一点无所谓；但评估里 `hybrid` 与 `rerank` 都
    声明"50 深的池子"，实际是 25，两行数字的差值就不再只等于"有没有重排"这一个变量
    （2026-10-28 专家复审 P1）。`candidate_k=None` 保留旧行为，生产链路不受影响。

    查询有**两种**用法，互斥：

    - `expander`：把口语问句拼成「原话 + 术语」一个串再查（首版做法）；
    - `rewrites`：原话仍然独立查一路，改写串各自查一路，结果按权重融合。

    后者才是"扩展不改变原有排序"的正解。拼串的问题不是"丢了原话"——原话还在串里，
    但整串文本变了，向量与 BM25 的排序都跟着变，原来命中的片段可能被新加的干扰词
    挤出去（实测 0.400→0.467，5 道目标题一道没救回）。把它当独立通道，原话那一路的
    名次与候选就一条都不会动，改写顶多往池子里加东西。
    """
    # 形参校验放在最前面：两个都用上不会报错、只会算出一组没人能解释的数，
    # 这种"能跑但无意义"的组合必须在花钱调接口之前就拒绝。
    if rewrites and expander is not None and expander.available:
        raise ValueError(
            "rewrites 与 expander 不能同时给：前者是并行通道（原话独立保留），"
            "后者是把术语拼进原话。两种形态混在一起，结果无法归因。"
        )
    if not 0.0 < rewrite_weight <= 1.0:
        raise ValueError(f"rewrite_weight 必须在 (0, 1] 内，收到 {rewrite_weight}")

    recall = recall_k or settings.recall_k
    pool = candidate_k if candidate_k is not None else max(recall // 2, settings.top_k * 3)
    timings: dict[str, float] = {}
    # 先取号再检查：顺序反过来的话，"检查通过 → 这次切换开始 → 取号拿到新号"会被
    # 当成一路没变，白检查一次。
    version = repo.get_index_generation(conn)
    # 手里这个后端是不是库里记着的活动后端。代次只能发现"这次查询跨越了切换"，
    # 发现不了"这个进程从一开始拿的就是另一份索引"：长跑服务不会因为别人换了后端
    # 就自动换过来，它只会继续查旧索引，一声不响（2026-10-24 第十六轮复审）。
    consistency.ensure_active_backend(conn, store)
    # 查询向量与文档向量必须出自同一个模型，否则检索"能跑但结果无意义"。
    # 同时核对后端实际条数与归档：归档对、索引少写几条时，稠密召回会静默漏内容。
    consistency.ensure_queryable(
        conn, embedder.model, store.dim, store_count=store.count()
    )

    primary = question
    if expander is not None and expander.available:
        t0 = time.perf_counter()
        primary = await asyncio.to_thread(expander.expand, question)
        timings["expand_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    # 通道表：主查询恒重 1.0，改写串各按 rewrite_weight。
    # 空串与重复串不能算一个通道——多算一次就是给它加权，而且是悄悄加权。
    channels: list[tuple[str, float]] = [(primary, 1.0)]
    seen = {primary}
    for item in rewrites:
        text = (item or "").strip()
        if text and text not in seen:
            seen.add(text)
            channels.append((text, rewrite_weight))

    t0 = time.perf_counter()
    # **三处检索都必须串行，一个 gather 也不能用。** 这不是保守，是这条链路上
    # 所有读路径共用**一条** sqlite 连接：
    #
    # - `Embedder.embed_one` 会读写 `embed_cache`（SELECT + INSERT OR REPLACE + commit）；
    # - `NumpyStore.search` 先 `_refresh_if_stale` 读代次与归档行；
    # - `repo.fts_search` 本身就是 SQL。
    #
    # 同一时刻多线程碰同一个 connection 会抛
    # `InterfaceError: bad parameter or other API misuse`；写路径并发还可能悄悄丢一次
    # commit。实测代价很直观：口语集 15 题里随机中 1 题，报告上只表现为
    # "`hybrid_parallel` 有效样本 14/15"——均值照算，读者根本不会注意到。
    #
    # 于是"并行改写"这个名字说的是**候选来源并行**，不是执行并行：多一路改写就多一次
    # 远程 embedding 往返，延迟线性增长。这个成本必须记在结论里（延迟是这条路
    # 能不能上生产的门槛），而不是假装并发掉了。
    vectors = [await asyncio.to_thread(embedder.embed_one, text) for text, _ in channels]
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _ensure_same_index(conn, version, "计算查询向量期间")

    t0 = time.perf_counter()
    dense_runs: list[Sequence[tuple[str, float]]] = []
    for vec in vectors:
        dense_runs.append(
            await asyncio.to_thread(lambda v=vec: store.search(v, limit=recall, filt=filt))
        )
    timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    # 切换也可能正好落在 search 里面：那条路径已经通过了上一次确认，
    # 结果里却可能混着两代索引的分数，只能丢掉重来。
    _ensure_same_index(conn, version, "检索期间")

    t0 = time.perf_counter()
    sparse_runs: list[Sequence[tuple[str, float]]] = []
    for text, _ in channels:
        sparse_query = f"{text} {filt.keyword}" if filt.keyword else text
        # BM25 走同一条 sqlite 连接，因此串行：并发读写同一个 connection 在 WAL 下
        # 多半能跑，但"多半"不是可以把正确性押上去的理由。
        sparse_runs.append(
            await asyncio.to_thread(
                lambda sq=sparse_query: repo.fts_search(conn, sq, limit=recall, filt=filt)
            )
        )
    timings["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    runs: list[tuple[Sequence[tuple[str, float]], float]] = []
    # strict=True：这三个列表都由 `channels` 派生，长度不等就是代码写错了。
    # 默认的静默截断会把它变成一个"少算了一路通道"的分数表，算得出来但不对。
    for (_, weight), dense_run, sparse_run in zip(
        channels, dense_runs, sparse_runs, strict=True
    ):
        runs.append((dense_run, weight))
        runs.append((sparse_run, weight))
    fused = rrf_multi(runs, k=settings.rrf_k)
    hits = _to_hits(conn, fused, limit=pool)

    # 原始分按**主通道**登记，方便前端/评估观察召回来源。改写通道的分数只参与融合，
    # 不覆盖这里的语义——"这条是不是原话召回的"才是这一栏要回答的问题。
    dense_map = dict(dense_runs[0])
    sparse_map = dict(sparse_runs[0])
    for hit in hits:
        hit.dense_score = dense_map.get(hit.chunk_id)
        hit.sparse_score = sparse_map.get(hit.chunk_id)
    return hits, filt, timings


def rerank_available(reranker: Reranker | None) -> bool:
    """重排是不是真的会做事。

    判据用「装的是不是 NoopReranker」而不是「配置里有没有 URL」：前者是实际行为，
    后者只是来源之一（测试、自建实现都可以不经 URL 注入）。三处调用点——`stream_answer`
    的默认值、CLI、`/api/health`——必须用同一个判断，否则会出现"界面上开关是灰的、
    后台其实在重排"这种对不上的状态。
    """
    return reranker is not None and not isinstance(reranker, NoopReranker)


async def stream_answer(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    llm: LLM,
    reranker: Reranker,
    settings: Settings,
    *,
    question: str,
    filt: Filter,
    top_k: int | None = None,
    use_rerank: bool | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    top_k = top_k or settings.top_k
    # None 表示"跟随服务端配置"：装了真实重排就重排，Noop 就当没这回事。
    # 这样评估报告里的 rerank 一行对应的就是用户实际跑到的那条链路，
    # 而不是一条只能靠脚本复现的旁路。
    do_rerank = rerank_available(reranker) if use_rerank is None else use_rerank
    started = time.perf_counter()
    # 术语扩展是实验开关，默认不启用：首版对照实验（口语集）在 hybrid/rerank 上
    # 无正收益（详见 HANDOFF 第三十八轮），默认开启只会让每次问答多一次 LLM 调用
    # 却拿不到好处。要开请在调用点显式传 expander。
    hits, effective, timings = await retrieve(
        conn, store, embedder, settings, question=question, filt=filt
    )

    rerank_error: str | None = None
    if do_rerank and hits:
        try:
            hits = await reranker.rerank(question, hits, top_k)
        except Exception as exc:  # 重排是增强项：失败要告诉调用方，但别中断回答
            rerank_error = f"{type(exc).__name__}: {exc}"
    final = hits[:top_k]

    yield (
        "meta",
        {
            "filter": effective.model_dump(exclude_none=True),
            "filter_desc": describe(effective),
            "candidates": len(hits),
            "timings": timings,
            "rerank_error": rerank_error,
        },
    )

    if not final:
        yield "citations", []
        yield "token", {"text": "在当前检索范围内没有找到相关资料。可以放宽项目或年份条件后重试。"}
        yield "done", {"elapsed_ms": round((time.perf_counter() - started) * 1000)}
        return

    prompt = build_answer_prompt(question, final, describe(effective))
    async for delta in llm.stream(SYSTEM_ANSWER, prompt):
        yield "token", {"text": delta}

    yield (
        "citations",
        {
            "items": [
                {
                    "n": n,
                    "chunk_id": h.chunk_id,
                    "project": h.project_name,
                    "project_code": h.project_code,
                    "filename": h.filename,
                    "chapter": " > ".join(h.chapter_path),
                    "page": h.page,
                    "score": round(h.rerank_score or h.score, 4),
                    "snippet": h.text[:240],
                }
                for n, h in enumerate(final, start=1)
            ]
        },
    )
    yield "done", {"elapsed_ms": round((time.perf_counter() - started) * 1000)}


async def search(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    question: str,
    filt: Filter,
    top_k: int | None = None,
    reranker: Reranker | None = None,
    use_rerank: bool | None = None,
    expander: QueryExpander | None = None,
    rewrites: Sequence[str] = (),
    rewrite_weight: float = 0.5,
) -> tuple[list[Hit], Filter, dict[str, float]]:
    """检索（可选重排），返回前 `top_k` 条。

    **顺序铁律：先重排、后截断。** 先截断再重排等于让重排在一小撮候选里挑，
    调用记录、分数、日志一样不少，召回却一点不变好——这种"看起来生效"的写法
    比不重排更糟，因为它会让人以为知道了重排的效果。

    `rewrites` 是并行改写通道（见 `retrieve`），默认空。**生产链路不注入**：
    首版扩展实验是负结果，要开必须由调用方显式传，并自行承担多一路 embedding 的延迟。
    """
    limit = top_k or settings.top_k
    hits, effective, timings = await retrieve(
        conn,
        store,
        embedder,
        settings,
        question=question,
        filt=filt,
        expander=expander,
        rewrites=rewrites,
        rewrite_weight=rewrite_weight,
    )
    do_rerank = (
        rerank_available(reranker) if use_rerank is None else use_rerank
    ) and reranker is not None
    if do_rerank and hits:
        hits = await reranker.rerank(question, hits, limit)
    return hits[:limit], effective, timings


def build_filter(
    question: str, request_filter: Filter | None, known_projects: Sequence[str]
) -> Filter:
    parsed = parse_filter(question, known_projects)
    return request_filter.merge(parsed) if request_filter else parsed
