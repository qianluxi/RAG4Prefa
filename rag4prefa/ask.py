"""问答用例：混合召回 → RRF 融合 → 可选重排 → 真流式生成。

关键修正（对照原系统）：
- 关键词参与**召回**（BM25），不再是在 Top-K 之后做子串过滤；
- 稠密与稀疏结果用 RRF 融合，而不是各取一半拼起来；
- 口语问句可以多挂一条**术语通道**（受控词表 → 规范用语），原话那一路不动；
- 生成走 token 流，客户端首字延迟≈模型首 token 延迟。
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

from . import consistency, repo
from .adapters.embed import Embedder
from .adapters.llm import LLM
from .adapters.query_expand import QueryExpander
from .adapters.rerank import NoopReranker, Reranker
from .adapters.term_map import TermMap
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


# 一路候选的分数方差低于这个数，就认为它"所有条目同分"，无法表达谁更靠前。
_FLAT_STD = 1e-12


def fuse_scores(
    runs: Sequence[tuple[Sequence[tuple[str, float]], float]],
    *,
    mode: str = "zscore",
    k: int = 60,
) -> dict[str, float]:
    """把多路 (排序列表, 权重) 合成一张分数表；`mode` 决定用什么刻度衡量"有多靠前"。

    - `rrf`：只用名次（见 `rrf_multi`）。免疫"两路分数不可比"，**代价是默认两路
      质量相当**——它给每一路的头部都留展示位。
    - `zscore`（默认）：各自在**本路内部**标准化后加权求和，等于按"这一路认为它
      有多突出"来分配话语权。

    为什么换掉等权 RRF（2026-10-29，口语集扩到 50 题才看清）：50 题里 17 题的 gold
    只有稠密这一路找得到，BM25 那边回的是另一批不相干的片段；两路几乎不相交时，
    名次求和退化成"把展示位对半分"——dense 的第 k 名被挤到第 2k 名。单题实测
    「卫生间地面要不要往地漏那边找坡」：gold 在 dense 第 1、BM25 不进前 50、融合后
    第 22。算术上 `1/61≈0.0164` 打不过 `1/91+1/90≈0.0220`：**单路强命中被双路弱命中
    压过**。结果就是 `hybrid 0.440` 反而低于纯稠密的 `0.600`。
    换 zscore 后两套题集同时改善：口语 0.440→0.640、基础 0.774→0.871，MRR
    0.313→0.389、0.576→0.688，且逐题没有"救一题丢一题"的交换（口语丢 1、基础丢 0）。

    **低于本路均值的候选只给 0，不给负分。** 负分意味着"本路排第 50"比"本路根本
    没命中"更差；在稠密 + BM25 两路上这点无所谓，但并行改写通道会因此把原话的候选
    压到改写候选之后，破坏"改写只能加、不能减"这条不变量（第四十轮的核心结论）。
    实测两种写法在 81 题上指标完全相同（0.640/0.871，MRR 0.389/0.688），
    那就选不会破坏不变量的那个。

    **一路分数全同（std≈0）时不参与融合**：这时它只能说明"这些片段被命中了"，
    说不出谁更靠前，硬给一份分等于拿列表顺序伪造名次。若所有路都这样（例如 BM25
    一条都没命中、或全部同分），退回名次法——顺序仍然确定，结果不会为空。

    `weight` 的语义与 `rrf_multi` 一致：原话 1.0，改写通道按 `rewrite_weight` 折。
    """
    if mode == "rrf":
        return rrf_multi(runs, k=k)
    if mode != "zscore":
        raise ValueError(f"未知的融合方式：{mode}（可选 zscore / rrf）")
    scores: dict[str, float] = {}
    contributing = 0
    for ranked, weight in runs:
        if not ranked:
            continue
        values = [score for _, score in ranked]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        if std <= _FLAT_STD:
            # 这一路分数全同：说不出谁更靠前，**但它的候选不能整路丢掉**——
            # 那是这路召回的全部产出。以 0 分入表，让它仍可被另一路或重排救回来。
            for chunk_id, _ in ranked:
                scores.setdefault(chunk_id, 0.0)
            continue
        for chunk_id, score in ranked:
            standout = max((score - mean) / std, 0.0)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * standout
        contributing += 1
    if not contributing:
        return rrf_multi(runs, k=k)
    return scores


def term_channels(
    term_map: TermMap | None, question: str, *, form: str = "joined"
) -> tuple[tuple[str, ...], list[str]]:
    """词表命中 → (检索用的改写串, 命中的术语)。

    两件事分开返回，因为去向不同：改写串交给 `retrieve` 当**独立通道**（原话那一路
    一条都不动），术语列表给重排拼查询、给前端解释"这题为什么注入了这几个词"。

    没命中时返回空元组与空列表，**不返回原话**——把原话当术语塞回去，"词表没覆盖"
    就会被记成"词表给了词"，两种结论正好相反。

    形态（`terms`/`joined`）的合法性交给 `TermMap.rewrite` 判定，这里不另写一份：
    两份判据迟早会分家，而分家的表现是"命令行收 `terms`、生产收 `joined` 之外的拼法"。
    """
    if term_map is None:
        return (), []
    terms = term_map.match(question)
    if not terms:
        return (), []
    return tuple(term_map.rewrite(question, form=form)), terms


def build_rerank_query(question: str, terms: Sequence[str], *, mode: str = "append") -> str:
    """重排读哪个串。

    - `none`：只读原话（等于没有词表这一层）；
    - `terms`：只读术语串。**不要用它当默认值**——2026-10-29 在 v2 词表覆盖的 15 题上
      实测：原话 11/15、纯术语串 12/15、原话 + 术语串 14/15。纯术语串会把**原本已经
      排第 1** 的题打出去，机制很直白：那几题重排本来就读得懂口语，换串等于拿掉信号；
    - `append`（默认）：原话 + 术语串，术语只做加法。

    没命中术语时三种形态都退回原话——重排看到的东西不能因为"词表没覆盖"而变。
    """
    if mode == "none" or not terms:
        return question
    joined = " ".join(terms)
    if mode == "terms":
        return joined
    if mode == "append":
        return f"{question} {joined}"
    raise ValueError(f"未知的重排串形态：{mode}（可选 none / terms / append）")


def _require_term_map_wired(term_map: TermMap | None, settings: Settings) -> None:
    """配了词表却没把词表对象传进来，是装配漏了，不是"这次不用"。

    `R4P_TERM_MAP` 只在装配时读一次（`build_services`），调用方负责把它传进来。
    漏传时若静默当"没有词表"，界面、日志、指标一切正常，只有口语问句悄悄拿不到
    术语通道——这类失效没有任何一处会报出来，所以宁可在请求上直接失败。
    """
    if term_map is None and settings.term_map.strip():
        raise ValueError(
            "配置了 R4P_TERM_MAP，但这次调用没有拿到词表对象：装配漏传了"
            "（调用方应传 services.term_map）。不在这里静默退回“不用词表”。"
        )


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
    fused = fuse_scores(runs, mode=settings.fusion, k=settings.rrf_k)
    hits = _to_hits(conn, fused, limit=pool)

    # 原始分按**主通道**登记，方便前端/评估观察召回来源。改写通道的分数只参与融合，
    # 不覆盖这里的语义——"这条是不是原话召回的"才是这一栏要回答的问题。
    #
    # `hit.score` 是**融合分**（zscore 模式下可能为负），只用来定顺序，不是相似度：
    # 它不跨查询可比，也不能解读成"相关程度"。要看相似度用 `dense_score`，
    # 要看重排判断用 `rerank_score`。
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
    term_map: TermMap | None = None,
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
    #
    # 受控词表是另一回事：它不调 LLM、命中可肉眼复核，2026-10-29 接入前已在两套
    # 题集上量过净效果。`term_map=None`（默认）表示不启用，行为与没有这一层时一致。
    _require_term_map_wired(term_map, settings)
    rewrites, terms = term_channels(term_map, question, form=settings.term_map_form)
    hits, effective, timings = await retrieve(
        conn,
        store,
        embedder,
        settings,
        question=question,
        filt=filt,
        rewrites=rewrites,
        rewrite_weight=settings.term_rewrite_weight,
    )

    rerank_error: str | None = None
    if do_rerank and hits:
        try:
            query = build_rerank_query(question, terms, mode=settings.term_rerank)
            hits = await reranker.rerank(query, hits, top_k)
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
            # 注入了哪几个术语、以及**为什么**注入。没有这两栏，"这题答案为什么变了"
            # 在界面上无从解释；而这一层唯一的承诺就是可审计。
            "terms": terms,
            "term_entries": term_map.explain(question) if term_map is not None else [],
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
    term_map: TermMap | None = None,
) -> tuple[list[Hit], Filter, dict[str, float]]:
    """检索（可选重排），返回前 `top_k` 条。

    **顺序铁律：先重排、后截断。** 先截断再重排等于让重排在一小撮候选里挑，
    调用记录、分数、日志一样不少，召回却一点不变好——这种"看起来生效"的写法
    比不重排更糟，因为它会让人以为知道了重排的效果。

    `rewrites` 是并行改写通道（见 `retrieve`），默认空。**生产链路不注入**：
    首版扩展实验是负结果，要开必须由调用方显式传，并自行承担多一路 embedding 的延迟。

    `term_map` 是受控词表（公众表达 → 规范用语）。它走的是同一条并行通道机制，
    但术语是人写的、命中可复核，所以默认由装配层注入（`services.term_map`）；
    是否启用完全取决于 `R4P_TERM_MAP` 有没有配。
    """
    limit = top_k or settings.top_k
    _require_term_map_wired(term_map, settings)
    term_rewrites, terms = term_channels(term_map, question, form=settings.term_map_form)
    if rewrites and term_rewrites:
        raise ValueError(
            "同时给了 rewrites 与命中的词表术语：两套改写串混在一起，收益无法归因到任何一方"
        )
    hits, effective, timings = await retrieve(
        conn,
        store,
        embedder,
        settings,
        question=question,
        filt=filt,
        expander=expander,
        rewrites=rewrites or term_rewrites,
        # 词表通道用配置里的权重；显式给了 rewrites 就按调用方给的权重走
        # （那是有明确意图的实验，不该被配置里的数悄悄改掉）。
        rewrite_weight=rewrite_weight if rewrites else settings.term_rewrite_weight,
    )
    do_rerank = (
        rerank_available(reranker) if use_rerank is None else use_rerank
    ) and reranker is not None
    if do_rerank and hits:
        hits = await reranker.rerank(
            build_rerank_query(question, terms, mode=settings.term_rerank), hits, limit
        )
    return hits[:limit], effective, timings


def build_filter(
    question: str, request_filter: Filter | None, known_projects: Sequence[str]
) -> Filter:
    parsed = parse_filter(question, known_projects)
    return request_filter.merge(parsed) if request_filter else parsed
