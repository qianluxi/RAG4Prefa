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


def rrf(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float]],
    *,
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion：只用名次，不用各自分数量纲，天然免疫"分数不可比"。"""
    scores: dict[str, float] = {}
    for ranked in (dense, sparse):
        for rank, (chunk_id, _) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


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
) -> tuple[list[Hit], Filter, dict[str, float]]:
    """返回 (融合后的候选, 实际生效的筛选条件, 各阶段耗时)。

    两个深度不是同一个数，必须分开：

    - `recall_k`：**每个通道**各取多少条（稠密、BM25 各自的上限）；
    - `candidate_k`：RRF 融合后**保留多少条**交给下游（重排、评测、展示）。

    旧实现把后者写死为 `max(recall // 2, top_k * 3)`，于是 `recall_k=50` 实际只拿到
    25 条。生产问答只取前 top_k 条，少一点无所谓；但评估里 `hybrid` 与 `rerank` 都
    声明"50 深的池子"，实际是 25，两行数字的差值就不再只等于"有没有重排"这一个变量
    （2026-10-28 专家复审 P1）。`candidate_k=None` 保留旧行为，生产链路不受影响。
    """
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

    # 查询术语扩展：把口语问句扩成「原话 + 术语」，喂给 BM25 与向量两个通道。
    # 这是可选项——没传 expander（或 expander 不可用）时，query 就是原话，行为与
    # 加扩展之前完全一致。扩展失败在 expander 内部已降级回原话。
    #
    # 但"保留原话"只保证"扩不出来时退回原话"，**不保证"扩出来不会更差"**：
    # 拼接改变的是整串文本，向量与 BM25 的排序都会跟着变，原来命中的片段也可能被
    # 新加的干扰词挤出去。首版对照实验正是这样拿到 0.400→0.467、却一道目标题都没
    # 救回（HANDOFF 第三十八轮）。要"原来的排序一定不变"，只能把原话当作**独立通道**
    # 并行检索再融合，而不是拼成一个串。
    expanded = question
    if expander is not None and expander.available:
        t0 = time.perf_counter()
        expanded = await asyncio.to_thread(expander.expand, question)
        timings["expand_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    vector = await asyncio.to_thread(embedder.embed_one, expanded)
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    _ensure_same_index(conn, version, "计算查询向量期间")

    t0 = time.perf_counter()
    dense = await asyncio.to_thread(lambda: store.search(vector, limit=recall, filt=filt))
    timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    # 切换也可能正好落在 search 里面：那条路径已经通过了上一次确认，
    # 结果里却可能混着两代索引的分数，只能丢掉重来。
    _ensure_same_index(conn, version, "检索期间")

    t0 = time.perf_counter()
    sparse_query = f"{expanded} {filt.keyword}" if filt.keyword else expanded
    sparse = await asyncio.to_thread(
        lambda: repo.fts_search(conn, sparse_query, limit=recall, filt=filt)
    )
    timings["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    fused = rrf(dense, sparse, k=settings.rrf_k)
    hits = _to_hits(conn, fused, limit=pool)

    # 把两路原始分补回结果，方便前端/评估观察召回来源
    dense_map = dict(dense)
    sparse_map = dict(sparse)
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
) -> tuple[list[Hit], Filter, dict[str, float]]:
    """检索（可选重排），返回前 `top_k` 条。

    **顺序铁律：先重排、后截断。** 先截断再重排等于让重排在一小撮候选里挑，
    调用记录、分数、日志一样不少，召回却一点不变好——这种"看起来生效"的写法
    比不重排更糟，因为它会让人以为知道了重排的效果。
    """
    limit = top_k or settings.top_k
    hits, effective, timings = await retrieve(
        conn, store, embedder, settings, question=question, filt=filt, expander=expander
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
