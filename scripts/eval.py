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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa import ask, consistency, repo  # noqa: E402
from rag4prefa.domain import Filter, Hit  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

MODES = ("dense", "sparse", "hybrid", "rerank")


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


# ── 四组召回 ───────────────────────────────────────────────


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
    """
    return max(k * 5, 50)


async def hybrid_pool(services, question: str, filt: Filter, k: int) -> list[Hit]:
    """取深度 ≥ `_hybrid_pool_size(k)` 的融合候选池，不做截断。"""
    hits, _, _ = await ask.retrieve(
        services.conn,
        services.store,
        services.require_embedder(),
        services.settings,
        question=question,
        filt=filt,
        recall_k=_hybrid_pool_size(k),
    )
    return hits


async def hybrid_hits(services, question: str, filt: Filter, k: int) -> list[Hit]:
    return (await hybrid_pool(services, question, filt, k))[:k]


async def rerank_hits(services, question: str, filt: Filter, k: int) -> list[Hit]:
    # 候选池与 hybrid 完全一致：两行之间唯一的变量是"有没有重排"。
    candidates = await hybrid_pool(services, question, filt, k)
    return await services.reranker.rerank(question, candidates, k)


async def retrieve_for(
    services, mode: str, question: str, filt: Filter, k: int, keyword: str | None
) -> list[Hit]:
    if mode == "dense":
        return await dense_hits(services, question, filt, k)
    if mode == "sparse":
        return await sparse_hits(services, question, filt, k, keyword)
    if mode == "hybrid":
        return await hybrid_hits(services, question, filt, k)
    return await rerank_hits(services, question, filt, k)


# ── 主流程 ─────────────────────────────────────────────────


def evaluate(services, rows: list[dict], k: int, modes: list[str]) -> dict:
    return asyncio.run(_evaluate(services, rows, k, modes))


def _run_context(services) -> dict:
    """跑出这组数字时用的坐标系。缺一项，数字就不能跟前一次直接比。"""
    settings = services.settings
    state = consistency.inspect(services.conn)
    rerank_on = ask.rerank_available(getattr(services, "reranker", None))
    return {
        # 以索引里登记的为准（state），它才是这批向量真正的来源；
        # 配置只是"打算用什么"，两者不一致时检索本来就会被拒绝。
        "embedding_model": state.model or settings.embedding_model,
        "embedding_dim": state.dim or settings.embedding_dim,
        "chunk_max_tokens": settings.chunk_max_tokens,
        "chunk_min_tokens": settings.chunk_min_tokens,
        "recall_k": settings.recall_k,
        "rrf_k": settings.rrf_k,
        "rerank_model": settings.rerank_model if rerank_on else "",
    }


async def _evaluate(services, rows: list[dict], k: int, modes: list[str]) -> dict:
    metrics: dict[str, dict[str, list[float]]] = {
        m: {"recall": [], "mrr": [], "ndcg": []} for m in modes
    }
    # 每种召回方式实际评估了多少题，必须各自记账：某一方法在部分样本上失败时，
    # 用"总有效题数"代表所有方法，会让不同方法的均值建立在不同题目集合上，不可比。
    evaluated: dict[str, int] = {m: 0 for m in modes}
    warnings: list[str] = []
    skipped: list[str] = []

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

        for mode in modes:
            try:
                hits = await retrieve_for(
                    services, mode, question, effective, k, effective.keyword
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

    return {
        "modes": metrics,
        "evaluated": evaluated,
        "samples": len(rows) - len(skipped),
        "skipped": skipped,
        "warnings": warnings,
        # 没配 R4P_RERANK_URL 时 reranker 是 NoopReranker：那一行必须被标成"未启用"，
        # 否则读者会把它当成"重排的收益"。判据与 CLI / /api/health 共用同一个函数。
        "rerank_enabled": ask.rerank_available(getattr(services, "reranker", None)),
        # 数值的"坐标系"必须跟数值一起走：换模型/换维度/换 chunk 参数/换 rrf_k 之后
        # 前后两次数值不可比，而报告本身不会提醒下一届读者这件事。
        "context": _run_context(services),
    }


def run(dataset_path: Path, k: int, modes: list[str]) -> dict:
    services = build_services()
    try:
        # 评估数字只有在"索引已被证明是归档这一版"时才有意义：拿一份内容过期的索引
        # 算出来的 Recall 会非常像真的，只是它衡量的是另一批向量。
        services.assert_query_ready()
        rows = load_dataset(dataset_path)
        if not rows:
            raise ValueError(f"数据集为空：{dataset_path}")
        result = evaluate(services, rows, k, modes)
        result["dataset"] = str(dataset_path)
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
    if ctx:
        rerank = f"`{ctx['rerank_model']}`" if ctx.get("rerank_model") else "未启用"
        lines.append(
            f"- 坐标系：embedding `{ctx['embedding_model']}`（{ctx['embedding_dim']} 维）· "
            f"chunk {ctx['chunk_min_tokens']}~{ctx['chunk_max_tokens']} token · "
            f"recall_k={ctx['recall_k']} · rrf_k={ctx['rrf_k']} · 重排 {rerank}"
        )
    if result.get("dataset"):
        lines.append(f"- 数据集：`{result['dataset']}`")
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
    if len(result["warnings"]) == 0 and not result["skipped"]:
        return "\n".join(lines)
    lines.append(f"\n## 需要处理的数据问题（{len(result['warnings'])}）\n")
    lines.extend(f"- {w}" for w in result["warnings"][:20])
    if result["skipped"]:
        lines.append(f"- 无有效标注而被跳过：{len(result['skipped'])} 条")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", default="dense,sparse,hybrid,rerank")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

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
        result = run(path, args.top_k, modes)
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
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
