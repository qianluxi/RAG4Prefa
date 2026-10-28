"""检索质量评估：Recall@k / MRR / nDCG@k，四组召回方式对照。

数据格式（JSONL，一行一条）：
    {"question": "...", "filter": {"projects": ["甲项目"]},
     "gold_chunk_ids": ["..."], "note": "可选备注"}

也支持只写"必须命中的原文片段"（标注省力）：
    {"question": "...", "gold_snippets": ["缓粘结预应力", "张拉时机"]}

**评价单位统一为 chunk。** 片段标注会在加载时解析成语料里包含该片段的全部 chunk_id，
之后 Recall / MRR / nDCG 用的是同一个 gold 集合。这一点必须钉死：如果 Recall 按原文片段
计数、MRR 和 nDCG 按 chunk 排序，同一个系统会同时算出 1.0 和 0.61，报告没法解释。

**gold 的解析范围必须等于检索的筛选范围。** 题为"甲项目"时，乙项目里出现同一段文字并不
算本题的相关片段——把它算进 gold，只会让 Recall 永远差一截，且误差与模型无关。

用法：
    python scripts/eval.py --dataset eval/dataset.jsonl --top-k 5
    python scripts/eval.py --modes dense,sparse,hybrid,rerank
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa import ask, consistency, repo, snapshot  # noqa: E402
from rag4prefa.adapters.query_expand import build_expander  # noqa: E402
from rag4prefa.domain import Filter, Hit  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

MODES = ("dense", "dense_rerank", "sparse", "hybrid", "rerank")
ROOT = Path(__file__).resolve().parents[1]


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return rows


def resolve_gold(conn, row: dict, filt: Filter) -> tuple[set[str], list[str]]:
    """把一条标注换算成 chunk_id 集合，返回 (gold, 警告)。

    三条硬规则，都会影响指标能不能解释：

    1. **范围必须和检索一致**。检索会用该题生效的 `filt` 过滤候选；如果 gold 却在全库
       范围内解析，别的项目/年份里出现的同一段文字也会被算成相关项——那些片段本题根本
       召回不到，于是 Recall 被系统性压低。这个误差与模型无关，纯属标注口径问题。
    2. **标注不能静默截断**。旧实现给 `chunk_ids_containing` 留了 `limit=50`，一段话
       在语料里出现超过 50 次时 gold 会被悄悄砍掉，指标反而变好看。
    3. **id 必须真实存在且在范围内**。写错的 id 不能当成有效标注，否则分母虚高。
    """
    warnings: list[str] = []
    declared = [str(x) for x in row.get("gold_chunk_ids") or []]
    gold = repo.select_chunk_ids(conn, declared, filt=filt)
    if len(gold) != len(set(declared)):
        unknown = sorted(set(declared) - gold)
        warnings.append(
            f"{len(unknown)} 个 gold_chunk_id 不存在或不在本题筛选范围内，已剔除"
            f"（例：{unknown[:3]}）"
        )

    snippets = [s for s in dict.fromkeys(row.get("gold_snippets") or []) if s]
    for snippet in snippets:
        found = repo.chunk_ids_containing(conn, snippet, filt=filt)
        if not found:
            # 标错了或语料已变：留着会让 Recall 永远达不到 1，必须让人看见
            outside = repo.chunk_ids_containing(conn, snippet)
            if outside:
                warnings.append(
                    f"标注片段只出现在本题筛选范围之外：{snippet!r}"
                    f"（全库命中 {len(outside)} 条，范围内 0 条）"
                )
            else:
                warnings.append(f"标注片段未在语料中找到：{snippet!r}")
            continue
        gold.update(found)
    return gold, warnings


def recall_at_k(hits: list[Hit], gold: set[str], k: int) -> float:
    """Recall@k = |TopK ∩ Gold| / |Gold|。

    分母是标注的相关 chunk 总数，**不是**返回条数——按返回条数算得到的是 Precision，
    曾在"两标一中"的场景把 1.0 算成 0.4。
    """
    if not gold:
        return 0.0
    return len({h.chunk_id for h in hits[:k]} & gold) / len(gold)


def mrr(rels: list[float]) -> float:
    for i, rel in enumerate(rels, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[float], k: int, total: int) -> float:
    """二值相关性下的 nDCG@k；IDCG 用标注的相关 chunk 总数构造理想排序。"""
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(total, k)))
    return dcg / idcg if idcg else 0.0


# ── 召回方式 ───────────────────────────────────────────────


async def dense_hits(services, question: str, filt: Filter, k: int) -> list[Hit]:
    embedder = services.require_embedder()
    vector = await asyncio.to_thread(embedder.embed_one, question)
    pairs = await asyncio.to_thread(lambda: services.store.search(vector, limit=k, filt=filt))
    details = repo.get_chunks(services.conn, [cid for cid, _ in pairs])
    return [Hit(score=s, **details[cid]) for cid, s in pairs if cid in details]


async def sparse_hits(
    services, question: str, filt: Filter, k: int, keyword: str | None = None
) -> list[Hit]:
    query = f"{question} {keyword}" if keyword else question
    pairs = await asyncio.to_thread(
        lambda: repo.fts_search(services.conn, query, limit=k, filt=filt)
    )
    details = repo.get_chunks(services.conn, [cid for cid, _ in pairs])
    return [Hit(score=s, **details[cid]) for cid, s in pairs if cid in details]


def _hybrid_pool_size(k: int) -> int:
    """混合召回的候选池深度。至少 50，且随 k 放大。

    这个数必须**显式写死并被 hybrid 与 rerank 共用**：RRF 融合后的头部顺序与候选深度
    有关，池子不同，同一个问题拿到的 top-k 就不同。曾经 rerank 是把 k 换成
    `max(k*4, 20)` 再走同一段融合代码——融合池深变成 100、却只把其中前 20 条交给重排，
    而 hybrid 是 50 深的池子取前 5。**池深和截断位置两个变量一起动**，于是在没有配重排
    服务（NoopReranker）的情况下两行数字仍然不同（真实数据上差到 +0.033），报告读起来
    就像"重排带来了 +0.033"，实际上一个候选都没被重排过。

    2026-10-28 专家复审又发现一处更隐蔽的同类问题：这个值只传给了 `recall_k`，
    而 `ask.retrieve` 当时会按 `max(recall//2, top_k*3)` 再截一刀（50 → 25）。
    声明 50、实得 25，`hybrid` 与 `dense_rerank` 的池深还因此不一样（后者拿的是
    dense 直出的 50 条），两行相减不再是单个变量。现在两个深度都显式传下去。
    """
    return max(k * 5, 50)


async def hybrid_pool(services, question: str, filt: Filter, k: int) -> list[Hit]:
    """取深度 = `_hybrid_pool_size(k)` 的融合候选池。

    `recall_k` 与 `candidate_k` 必须**同时**给足：前者决定两个通道各取多少，
    后者决定融合后保留多少。只给前者时会被 `ask.retrieve` 的默认截断砍回一半，
    而报告上写的池深仍是 50。
    """
    pool = _hybrid_pool_size(k)
    hits, _, _ = await ask.retrieve(
        services.conn,
        services.store,
        services.require_embedder(),
        services.settings,
        question=question,
        filt=filt,
        recall_k=pool,
        candidate_k=pool,
    )
    return hits


async def hybrid_hits(services, question: str, filt: Filter, k: int) -> list[Hit]:
    return (await hybrid_pool(services, question, filt, k))[:k]


async def rerank_hits(
    services, question: str, filt: Filter, k: int, *, rerank_query: str | None = None
) -> list[Hit]:
    """候选池与 hybrid 完全一致：两行之间唯一的变量是"有没有重排"。

    `rerank_query` 单独给：术语扩展只改召回查询时，重排仍必须读原话，
    否则"改写让候选变了"与"改写让重排判断变了"两件事混在一个差值里。
    """
    candidates = await hybrid_pool(services, question, filt, k)
    return await services.reranker.rerank(rerank_query or question, candidates, k)


async def dense_rerank_hits(
    services, question: str, filt: Filter, k: int, *, rerank_query: str | None = None
) -> list[Hit]:
    """纯向量召回 + 重排。候选池深度与 `rerank` 取同一个值。

    为什么要这一行（2026-10-28 第一版基线逼出来的）：原来的四行把"融合"和"重排"
    绑在一起——`rerank` 恒等于 `hybrid` 再重排，于是口语集上看到
    `dense 0.467 > hybrid 0.333 = rerank 0.333`，只能得出"重排加融合等于白做"，
    分不清是**融合拉低了候选池**还是**重排本身没用**。这两件事的结论完全不同：
    前者要改融合权重，后者要换重排模型。多这一行就是把这个变量拆开。
    """
    candidates = await dense_hits(services, question, filt, _hybrid_pool_size(k))
    return await services.reranker.rerank(rerank_query or question, candidates, k)


async def _dense_mode(services, question, filt, k, keyword, *, rerank_query=None):  # noqa: ARG001
    return await dense_hits(services, question, filt, k)


async def _dense_rerank_mode(services, question, filt, k, keyword, *, rerank_query=None):
    return await dense_rerank_hits(services, question, filt, k, rerank_query=rerank_query)


async def _sparse_mode(services, question, filt, k, keyword, *, rerank_query=None):  # noqa: ARG001
    return await sparse_hits(services, question, filt, k, keyword)


async def _hybrid_mode(services, question, filt, k, keyword, *, rerank_query=None):  # noqa: ARG001
    return await hybrid_hits(services, question, filt, k)


async def _rerank_mode(services, question, filt, k, keyword, *, rerank_query=None):  # noqa: ARG001
    return await rerank_hits(services, question, filt, k, rerank_query=rerank_query)


# 显式分派表，而不是 if/elif 加一个兜底 return：兜底分支会让"加了一种方式但忘了写实现"
# 变成"多出一行与 rerank 一模一样的数字"，报告里看不出任何异常。
# `test_eval_modes.py` 断言这张表与 `MODES` 一一对应。
_HANDLERS = {
    "dense": _dense_mode,
    "dense_rerank": _dense_rerank_mode,
    "sparse": _sparse_mode,
    "hybrid": _hybrid_mode,
    "rerank": _rerank_mode,
}


async def retrieve_for(
    services, mode: str, question: str, filt: Filter, k: int, keyword: str | None,
    *, rerank_query: str | None = None,
) -> list[Hit]:
    """`question` 是**召回**用的串（术语扩展开启时是扩展串）；`rerank_query` 是重排
    读的串，不传就与召回相同。两者分开，是为了让"扩展只改召回"成为可测的事实。"""
    handler = _HANDLERS.get(mode)
    if handler is None:  # pragma: no cover - 参数校验在 main() 里先拦一道
        raise ValueError(f"未知的召回方式：{mode}")
    return await handler(
        services, question, filt, k, keyword, rerank_query=rerank_query
    )


# ── 主流程 ─────────────────────────────────────────────────


def evaluate(
    services, rows: list[dict], k: int, modes: list[str],
    snapshot_path: Path | None = None, refuse_rows: list[dict] | None = None,
    expander=None, expand_rerank: bool = False,
) -> dict:
    return asyncio.run(
        _evaluate(
            services, rows, k, modes, snapshot_path, refuse_rows=refuse_rows,
            expander=expander, expand_rerank=expand_rerank,
        )
    )


def _code_state() -> dict:
    """跑这组数字时代码处在哪个版本。

    报告只写"指标 + 参数"是不够的：同一个库、同一份数据集，改过检索融合或提示词之后
    数字会变，而报告本身看不出来。有这条，"这两份报告为什么不一样"才有一个可查的答案。

    取不到就返回空字典——没装 git、不在仓库里跑，都不该让评估失败。
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        # 只看被跟踪文件：未提交改动会让"这个提交号"变得名不副实。
        # 但 `eval/reports/` 要排除：报告本身就是产物，一次 `--save` 必然改动它。
        # 不排除的话，第一份报告写完后工作区就"脏"了，之后每份报告都会把自己标成
        # "有未提交改动"——那个标记也就失去了意义（2026-10-28 专家复审 P1）。
        dirty = subprocess.run(
            [
                "git", "status", "--porcelain", "--untracked-files=no", "--",
                ".", ":(exclude)eval/reports",
            ],
            cwd=ROOT, capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    if not commit:
        return {}
    return {"commit": commit, "dirty": bool(dirty)}


def _run_context(
    services, snapshot_path: Path | None = None, k: int | None = None
) -> dict:
    """跑出这组数字时用的坐标系。缺一项，数字就不能跟前一次直接比。"""
    settings = services.settings
    state = consistency.inspect(services.conn)
    rerank_on = ask.rerank_available(getattr(services, "reranker", None))
    current = snapshot.build_snapshot(
        services.conn,
        chunk_min_tokens=settings.chunk_min_tokens,
        chunk_max_tokens=settings.chunk_max_tokens,
    )
    saved = snapshot.load_snapshot(snapshot_path or (ROOT / snapshot.DEFAULT_PATH))
    return {
        # 以索引里登记的为准（state），它才是这批向量真正的来源；
        # 配置只是"打算用什么"，两者不一致时检索本来就会被拒绝。
        "embedding_model": state.model or settings.embedding_model,
        "embedding_dim": state.dim or settings.embedding_dim,
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_min_tokens": settings.chunk_min_tokens,
        "recall_k": settings.recall_k,
        # 混合/重排两行用的真实池深。只写 recall_k 是会骗人的：
        # 曾经 recall_k=50 而融合后只剩 25 条，报告上却看不出这个差。
        "candidate_k": _hybrid_pool_size(k) if k else None,
        "rrf_k": settings.rrf_k,
        "rerank_model": settings.rerank_model if rerank_on else "",
        # 语料身份跟指标一起走。换成"当时库里有多少片段"之后，
        # "两份报告能不能比"就从靠回忆变成了一次字符串比较。
        "corpus": {
            "documents": current["document_count"],
            "chunks": current["chunk_count"],
            "ids_sha": current["ids_sha"],
            "drift": snapshot.compare(saved, current),
        },
    }


async def _evaluate(
    services, rows: list[dict], k: int, modes: list[str],
    snapshot_path: Path | None = None, refuse_rows: list[dict] | None = None,
    expander=None, expand_rerank: bool = False,
) -> dict:
    metrics: dict[str, dict[str, list[float]]] = {
        m: {"recall": [], "mrr": [], "ndcg": []} for m in modes
    }
    # 库外问题单独记账：它们没有 gold，进不了 Recall；而"召回到了什么"本身是结论——
    # 系统当前没有拒答阈值，任何问题都会拿到 top-k，这组题量的就是这件事。
    refuse: dict[str, dict] = {m: {"n": 0, "nonempty": 0, "top1": []} for m in modes}
    refuse_detail: list[dict] = []
    # 每种召回方式实际评估了多少题，必须各自记账：某一方法在部分样本上失败时，
    # 用"总有效题数"代表所有方法，会让不同方法的均值建立在不同题目集合上，不可比。
    evaluated: dict[str, int] = {m: 0 for m in modes}
    warnings: list[str] = []
    skipped: list[str] = []
    diagnostics: list[dict] = []

    for row in rows:
        question = str(row.get("question") or "").strip()
        if not question:
            skipped.append("缺少 question 字段的记录")
            continue
        request_filter = Filter(**row["filter"]) if row.get("filter") else Filter()
        # 先算出生效筛选条件，再解析 gold：两者必须是同一个范围
        effective = ask.build_filter(question, request_filter, repo.project_names(services.conn))
        gold, row_warnings = resolve_gold(services.conn, row, effective)
        warnings.extend(f"{question}：{w}" for w in row_warnings)
        if not gold:
            skipped.append(question)
            continue

        # 术语扩展：把口语问句扩成「原话 + 术语」再检索。gold 仍然按原问句解析——
        # 扩展只改"怎么查"，不改"查什么是对的"。
        #
        # 扩展默认**只作用于召回**，重排仍读原话（对照实验 B）：否则 rerank 那一行的
        # 变化里同时混着"候选集变了"和"重排判断变了"两个原因，谁也说不清收益来自哪。
        # 想研究"扩展词是否也改善重排判断"（实验 C）要显式 `expand_rerank=True`。
        search_question = question
        if expander is not None and expander.available:
            search_question = expander.expand(question)
        rerank_query = search_question if expand_rerank else question

        per_row = {"question": question, "search_question": search_question, "modes": {}}
        for mode in modes:
            try:
                hits = await retrieve_for(
                    services, mode, search_question, effective, k, effective.keyword,
                    rerank_query=rerank_query,
                )
            except consistency.IndexMismatch:
                raise  # 配置问题，不是单条样本的问题
            except Exception as exc:  # 单条失败不该中断整轮评估
                warnings.append(f"{question}（{mode}）：{type(exc).__name__}: {exc}")
                continue
            rels = [1.0 if h.chunk_id in gold else 0.0 for h in hits]
            metrics[mode]["recall"].append(recall_at_k(hits, gold, k))
            metrics[mode]["mrr"].append(mrr(rels))
            metrics[mode]["ndcg"].append(ndcg_at_k(rels, k, len(gold)))
            evaluated[mode] += 1
            # 逐题留证：只报均值时，"涨了 +0.067 到底是救回了哪道题"无从回答。
            # 记第一条 gold 的排名与 top-k 的 id 列表，够回答"是不是这道题"。
            first = next((i for i, r in enumerate(rels, start=1) if r > 0), None)
            per_row["modes"][mode] = {
                "first_gold_rank": first,
                "hits": [h.chunk_id for h in hits[:k]],
            }
        diagnostics.append(per_row)

    for row in refuse_rows or []:
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        request_filter = Filter(**row["filter"]) if row.get("filter") else Filter()
        effective = ask.build_filter(question, request_filter, repo.project_names(services.conn))
        tops: dict[str, dict] = {}
        for mode in modes:
            try:
                hits = await retrieve_for(
                    services, mode, question, effective, k, effective.keyword
                )
            except consistency.IndexMismatch:
                raise
            except Exception as exc:
                warnings.append(f"{question}（{mode}）：{type(exc).__name__}: {exc}")
                continue
            refuse[mode]["n"] += 1
            if hits:
                refuse[mode]["nonempty"] += 1
                refuse[mode]["top1"].append(hits[0].score)
                tops[mode] = {
                    "filename": hits[0].filename,
                    "title": hits[0].title,
                    "score": hits[0].score,
                }
        refuse_detail.append({"question": question, "tops": tops})

    return {
        "modes": metrics,
        "evaluated": evaluated,
        "refuse": refuse,
        "refuse_detail": refuse_detail,
        "samples": len(rows) - len(skipped),
        "skipped": skipped,
        "warnings": warnings,
        # 逐题证据：均值的来源。`--diagnostics` 把它落成 JSONL，
        # 让"哪道题被救回、哪道题被挤掉"可以事后核对，而不是只有一句结论。
        "diagnostics": diagnostics,
        # 没配 R4P_RERANK_URL 时 reranker 是 NoopReranker：那一行必须被标成"未启用"，
        # 否则读者会把它当成"重排的收益"。判据与 CLI / /api/health 共用同一个函数。
        "rerank_enabled": ask.rerank_available(getattr(services, "reranker", None)),
        # 数值的"坐标系"必须跟数值一起走：换模型/换维度/换 chunk 参数/换 rrf_k 之后
        # 前后两次数值不可比，而报告本身不会提醒下一届读者这件事。
        "context": _run_context(services, snapshot_path, k),
    }


def run(
    dataset_path: Path, k: int, modes: list[str],
    snapshot_path: Path | None = None, refuse_path: Path | None = None,
    expand: bool = False, expand_rerank: bool = False,
    diagnostics_path: Path | None = None,
) -> dict:
    services = build_services()
    try:
        # 评估数字只有在"索引已被证明是归档这一版"时才有意义：拿一份内容过期的索引
        # 算出来的 Recall 会非常像真的，只是它衡量的是另一批向量。
        services.assert_query_ready()
        rows = load_dataset(dataset_path)
        # 库外题是另一个文件：它们没有 gold，混进主集会让"标注有效题目数"虚高，
        # 也会让 resolve_gold 把它们当"空标注"跳过。
        refuse_rows = (
            load_dataset(refuse_path)
            if refuse_path is not None and refuse_path.is_file()
            else []
        )
        if not rows and not refuse_rows:
            raise ValueError(f"数据集为空：{dataset_path}")
        expander = None
        if expand:
            expander = build_expander(services.settings, getattr(services, "llm", None))
        result = evaluate(
            services, rows, k, modes, snapshot_path, refuse_rows=refuse_rows,
            expander=expander, expand_rerank=expand_rerank,
        )
        result["refuse_dataset"] = str(refuse_path) if refuse_rows else ""
        result["dataset"] = str(dataset_path)
        result["expanded"] = bool(expander is not None and expander.available)
        # 扩展作用在哪一段，必须跟着数字走：只写在命令行参数里，
        # 报告读者会以为 rerank 那行的变化也是"召回变好"带来的。
        result["expand_stage"] = (
            ("rerank" if expand_rerank else "recall") if result["expanded"] else ""
        )
        # 数据集的身份与语料身份同等重要：`eval/dataset.jsonl` 这一个路径
        # 曾经先后对应过 30 题和 25 题两个版本，报告看不出来。
        result["dataset_sha"], result["dataset_rows"] = snapshot.dataset_digest(dataset_path)
        result["generated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        result["code"] = _code_state()
        if diagnostics_path is not None:
            # 一行一题：扩展串 + 每种召回的 gold 排名与 top-k。均值的来源在这里，
            # 不在报告里——报告只负责给结论，这里负责让人能反驳结论。
            diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
            with diagnostics_path.open("w", encoding="utf-8") as fh:
                for item in result["diagnostics"]:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            result["diagnostics_path"] = str(diagnostics_path)
        return result
    finally:
        services.close()


def report(result: dict, k: int) -> str:
    modes: dict[str, dict[str, list[float]]] = result["modes"]
    evaluated: dict[str, int] = result.get("evaluated") or {
        m: len(v["recall"]) for m, v in modes.items()
    }
    lines = [
        "# 检索评估报告\n",
        f"- 标注有效的题目数：{result['samples']}"
        f"（跳过 {len(result['skipped'])} 条无标注/无问题的记录）",
        f"- 指标：Recall@{k}、MRR、nDCG@{k}，相关性按 chunk 二值判定",
        f"- Recall@{k} = |TopK ∩ Gold| / |Gold|，Gold 为**本题筛选范围内**的相关 chunk 集合",
        "- 片段标注已解析成范围内包含它的全部 chunk，三个指标共用同一个 Gold",
    ]
    ctx = result.get("context") or {}
    corpus: dict = {}
    if ctx:
        rerank = f"`{ctx['rerank_model']}`" if ctx.get("rerank_model") else "未启用"
        pool = f" · 混合池深={ctx['candidate_k']}" if ctx.get("candidate_k") else ""
        lines.append(
            f"- 坐标系：embedding `{ctx['embedding_model']}`（{ctx['embedding_dim']} 维）· "
            f"chunk {ctx['chunk_min_tokens']}~{ctx['chunk_max_tokens']} token · "
            f"recall_k={ctx['recall_k']} · rrf_k={ctx['rrf_k']} · 重排 {rerank}{pool}"
        )
        corpus = ctx.get("corpus") or {}
        if corpus:
            lines.append(
                f"- 语料快照：{corpus['documents']} 份文档 / {corpus['chunks']} 个片段 · "
                f"ids_sha `{corpus['ids_sha']}`"
            )
    if result.get("dataset"):
        bits = [f"`{result['dataset']}`"]
        if result.get("dataset_rows"):
            bits.append(f"{result['dataset_rows']} 题")
        if result.get("dataset_sha"):
            bits.append(f"sha `{result['dataset_sha']}`")
        lines.append("- 数据集：" + " · ".join(bits))
    if result.get("generated_at"):
        code = result.get("code") or {}
        tail = ""
        if code.get("commit"):
            state = "有未提交改动" if code.get("dirty") else "工作区干净"
            tail = f" · 代码 `{code['commit']}`（{state}）"
        lines.append(f"- 生成于 {result['generated_at']}{tail}")
    if result.get("expanded"):
        stage = result.get("expand_stage") or "recall"
        if stage == "rerank":
            lines.append(
                "- **检索使用了 LLM 术语扩展（召回 + 重排都读扩展串）**："
                "口语问句先扩成「原话 + 术语」，重排也用扩展串。"
                "此配置下 rerank 那行的变化同时包含“候选集变化”与“重排判断变化”。"
            )
        else:
            lines.append(
                "- **召回阶段使用了 LLM 术语扩展**：口语问句先扩成「原话 + 术语」再检索；"
                "**重排仍读原话**，因此 rerank 那行的差值只反映“候选集变化”。"
            )
    if result.get("diagnostics_path"):
        lines.append(f"- 逐题证据：`{result['diagnostics_path']}`")

    # 语料变了还沿用旧快照，是最容易出错的一种"看起来正常"：数字算得出来、
    # 也像是真的，只有跟上一份报告并排时才说不清为什么不一样。
    drift = corpus.get("drift") or []
    if drift:
        lines.append("")
        lines.append(
            "> **语料已变化**：" + "；".join(drift) + "。"
            f"`{snapshot.DEFAULT_PATH.as_posix()}` 记录的是旧语料，"
            "本报告对应当前库，与按旧快照跑出的报告**不可直接比较**。"
            "更新记录：`python scripts/snapshot_corpus.py`。"
        )
    lines += [
        "",
        f"| 召回方式 | 有效样本 | Recall@{k} | MRR | nDCG@{k} |",
        "|---|---|---|---|---|",
    ]
    for mode, values in modes.items():
        if not values["recall"]:
            lines.append(f"| {mode} | 0 | - | - | - |")
            continue
        lines.append(
            f"| {mode} | {evaluated.get(mode, 0)} | "
            f"{statistics.mean(values['recall']):.3f} | "
            f"{statistics.mean(values['mrr']):.3f} | "
            f"{statistics.mean(values['ndcg']):.3f} |"
        )
    # 样本数不一致时，横向比较不严谨——必须写出来，而不是靠读者自己数
    if len(set(evaluated.values())) > 1:
        lines.append(
            "\n> 注意：各召回方式的有效样本数不同，均值建立在不同的题目集合上，"
            "横向比较只能作参考。失败原因见下方「需要处理的数据问题」。"
        )
    def _mean(mode: str) -> float | None:
        values = modes.get(mode, {}).get("recall") or []
        return statistics.mean(values) if values else None

    def _gain_over(base: str) -> float | None:
        """hybrid 相对另一路召回的 Recall 差；任一方没有有效样本就不给数。"""
        base_mean, hybrid_mean = _mean(base), _mean("hybrid")
        if base_mean is None or hybrid_mean is None:
            return None
        return hybrid_mean - base_mean

    # 只报"相对纯向量"的增益会诱导读者以为混合召回就是最好的。
    # 语料里专有名词密度高时 BM25 往往压过混合召回，这个负值必须一起报出来。
    comparisons = [
        (label, gain)
        for label, base in (("纯向量", "dense"), ("纯稀疏", "sparse"))
        if (gain := _gain_over(base)) is not None
    ]
    if comparisons:
        lines.append("")
        for label, gain in comparisons:
            lines.append(f"混合召回相对{label}：Recall@{k} {gain:+.3f}")

    # 重排的收益要**单独**算，不能只报"混合 → 重排"那一跳：那一跳里同时有"融合"和
    # "重排"两个变量。2026-10-28 第一版基线就撞在这里——口语集上 hybrid = rerank = 0.333、
    # dense = 0.467，只报这一跳会让人以为"重排没用"，而对纯向量重排是 −0.067、
    # 对融合重排是 ±0.000，两件事的处置完全不同（前者换模型，后者改融合）。
    plain, reranked = _mean("dense"), _mean("dense_rerank")
    if plain is not None and reranked is not None and result.get("rerank_enabled", False):
        lines.append(f"重排相对纯向量：Recall@{k} {reranked - plain:+.3f}")

    if "rerank" in modes and not result.get("rerank_enabled", False):
        lines.append(
            "\n> `rerank` 一栏**未启用重排**（没配 `R4P_RERANK_URL`，走的是 NoopReranker）："
            "候选池与 hybrid 相同，数值应当与 hybrid 完全一致。"
            "若两者不同，说明两条管道哪里走岔了，而不是重排有收益。"
        )
    else:
        hybrid_mean, rerank_mean = _mean("hybrid"), _mean("rerank")
        if hybrid_mean is not None and rerank_mean is not None:
            lines.append(f"开重排相对 hybrid：Recall@{k} {rerank_mean - hybrid_mean:+.3f}")
    refuse = result.get("refuse") or {}
    detail = result.get("refuse_detail") or []
    if detail:
        lines.append("")
        lines.append("## 库外问题（语料里没有答案）")
        lines.append("")
        lines.append("| 召回方式 | 题数 | 召回非空 | top1 分数 |")
        lines.append("|---|---|---|---|")
        for mode, stat in refuse.items():
            total = stat.get("n") or 0
            if not total:
                lines.append(f"| {mode} | 0 | - | - |")
                continue
            mean = statistics.mean(stat["top1"]) if stat["top1"] else None
            score = f"{mean:.3f}" if mean is not None else "-"
            lines.append(f"| {mode} | {total} | {stat['nonempty']}/{total} | {score} |")
        lines.append("")
        lines.append(
            "> 这组问题在语料里**没有**对应内容。当前系统没有拒答阈值：任何问题都会返回 top-k，"
            "所以「召回非空 100%」说明的不是检索好，而是它不会说「不知道」。"
            "这组数字是课题方向①的起点，不是这一版的缺陷。"
        )
        # 分数不是一个尺度：余弦 ∈ [0,1]、BM25 无上界、RRF 只有几十分之一。
        # 不写这句，读者会对着一张表把 16.271 与 0.032 当成"BM25 更有信心"。
        lines.append("")
        lines.append(
            "> top1 分数**量纲不同**：dense 是余弦相似度、sparse 是 BM25 原始分、"
            "hybrid/rerank 是 RRF 融合分。只能在同一行内比较题目之间，"
            "不能跨行比高低——BM25 的 16 分不代表它比余弦 0.03 更有把握。"
        )
        lines.append("")
        pick = "hybrid" if "hybrid" in refuse else next(iter(refuse), "")
        lines.append(f"逐题 top1（按 {pick}）：")
        for item in detail:
            top = (item.get("tops") or {}).get(pick) or {}
            got = (
                f"`{top['filename']} · {top['title']}`（{top['score']:.3f}）"
                if top
                else "无召回"
            )
            lines.append(f"- {item['question']} → {got}")

    if len(result["warnings"]) == 0 and not result["skipped"]:
        return "\n".join(lines)
    lines.append(f"\n## 需要处理的数据问题（{len(result['warnings'])}）\n")
    lines.extend(f"- {w}" for w in result["warnings"][:20])
    if result["skipped"]:
        lines.append(f"- 无有效标注而被跳过：{len(result['skipped'])} 条")
    return "\n".join(lines)


def _report_path(result: dict, dataset_path: Path) -> Path:
    """按「数据集 + 日期 + 语料指纹」命名。

    刻意不用 `-new` 这种名字：`baseline-new` 用了两个月就不再是"新"的，
    而它指向的语料已经变过两轮；同一天换语料重跑也会互相覆盖。
    带上 ids_sha 之后，同名即同语料，不同语料必然不同名。
    """
    corpus = (result.get("context") or {}).get("corpus") or {}
    stamp = (result.get("generated_at") or "")[:10].replace("-", "")
    tag = (corpus.get("ids_sha") or "nosnapshot")[:8]
    # 带术语扩展与不带是两种不同的检索配置，必须分开命名：
    # 否则 `--expand` 的对照实验会覆盖掉同一语料、同一数据集的基线报告。
    # "只扩召回"与"连重排一起扩"又是两个实验，报告也不能同名。
    stage = result.get("expand_stage") or ""
    if stage == "rerank":
        suffix = "-expand-rerank"
    elif stage:
        suffix = "-expand"
    else:
        suffix = ""
    return ROOT / "eval" / "reports" / f"{dataset_path.stem}-{stamp}-{tag}{suffix}.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", default="dense,dense_rerank,sparse,hybrid,rerank")
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--save", action="store_true",
        help="写入 eval/reports/，文件名带日期与语料指纹（推荐，取代手动 --out）",
    )
    parser.add_argument("--snapshot", default="", help="语料快照路径")
    parser.add_argument(
        "--refuse",
        default="eval/dataset.refuse.jsonl",
        help="库外问题集（默认 eval/dataset.refuse.jsonl，不存在则跳过）",
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="检索前用 LLM 把口语问句扩展成「原话 + 术语」，只作用于召回（重排仍读原话）",
    )
    parser.add_argument(
        "--expand-rerank",
        action="store_true",
        help="让重排也读扩展串（实验 C，需同时给 --expand；默认重排读原话）",
    )
    parser.add_argument(
        "--diagnostics",
        default="",
        help="把逐题结果（扩展串、各召回方式的 gold 排名与 top-k）写成 JSONL",
    )
    args = parser.parse_args()

    if args.expand_rerank and not args.expand:
        print("--expand-rerank 需要同时给 --expand", file=sys.stderr)
        return 2

    if args.top_k < 1:
        print("--top-k 必须 ≥ 1", file=sys.stderr)
        return 2
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        print(f"未知的召回方式：{', '.join(unknown)}；可选：{', '.join(MODES)}", file=sys.stderr)
        return 2
    if not modes:
        print("至少要指定一种召回方式", file=sys.stderr)
        return 2

    path = Path(args.dataset)
    if not path.exists():
        print(f"数据集不存在：{path}（可先复制 eval/dataset.example.jsonl）", file=sys.stderr)
        return 2

    try:
        result = run(
            path,
            args.top_k,
            modes,
            Path(args.snapshot) if args.snapshot else None,
            refuse_path=Path(args.refuse) if args.refuse else None,
            expand=args.expand,
            expand_rerank=args.expand_rerank,
            diagnostics_path=Path(args.diagnostics) if args.diagnostics else None,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except consistency.IndexMismatch as exc:
        print(f"索引与配置不一致，评估不成立：{exc}", file=sys.stderr)
        return 3

    if not result["samples"]:
        print("没有任何样本带有效标注，无法评估", file=sys.stderr)
        return 2

    text = report(result, args.top_k)
    print(text)
    target = args.out or (str(_report_path(result, path)) if args.save else "")
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(text, encoding="utf-8")
        # 报告默认与逐题证据成对落盘：只有均值时，"涨了 0.067"这种结论没法被反驳，
        # 也没法被复核。JSONL 与报告同名不同后缀，一眼能对上。
        if args.save and not args.diagnostics:
            diag = Path(target).with_suffix(".jsonl")
            with diag.open("w", encoding="utf-8") as fh:
                for item in result["diagnostics"]:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n已写入 {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
