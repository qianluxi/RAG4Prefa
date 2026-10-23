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

from . import repo
from .adapters.embed import Embedder
from .adapters.llm import LLM
from .adapters.rerank import Reranker
from .adapters.vectors import VectorStore
from .config import Settings
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


async def retrieve(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    question: str,
    filt: Filter,
    recall_k: int | None = None,
) -> tuple[list[Hit], Filter, dict[str, float]]:
    """返回 (融合后的候选, 实际生效的筛选条件, 各阶段耗时)。"""
    recall = recall_k or settings.recall_k
    timings: dict[str, float] = {}
    query = f"{question} {filt.keyword}" if filt.keyword else question

    t0 = time.perf_counter()
    vector = await asyncio.to_thread(embedder.embed_one, question)
    timings["embed_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    dense = await asyncio.to_thread(lambda: store.search(vector, limit=recall, filt=filt))
    timings["dense_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    t0 = time.perf_counter()
    sparse = await asyncio.to_thread(lambda: repo.fts_search(conn, query, limit=recall, filt=filt))
    timings["sparse_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    fused = rrf(dense, sparse, k=settings.rrf_k)
    hits = _to_hits(conn, fused, limit=max(recall // 2, settings.top_k * 3))

    # 把两路原始分补回结果，方便前端/评估观察召回来源
    dense_map = dict(dense)
    sparse_map = dict(sparse)
    for hit in hits:
        hit.dense_score = dense_map.get(hit.chunk_id)
        hit.sparse_score = sparse_map.get(hit.chunk_id)
    return hits, filt, timings


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
    use_rerank: bool = False,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    top_k = top_k or settings.top_k
    started = time.perf_counter()
    hits, effective, timings = await retrieve(
        conn, store, embedder, settings, question=question, filt=filt
    )

    rerank_error: str | None = None
    if use_rerank and hits:
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
) -> tuple[list[Hit], Filter, dict[str, float]]:
    hits, effective, timings = await retrieve(
        conn, store, embedder, settings, question=question, filt=filt
    )
    return hits[: top_k or settings.top_k], effective, timings


def build_filter(
    question: str, request_filter: Filter | None, known_projects: Sequence[str]
) -> Filter:
    parsed = parse_filter(question, known_projects)
    return request_filter.merge(parsed) if request_filter else parsed
