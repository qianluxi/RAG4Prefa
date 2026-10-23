"""检索质量评估：Recall@k / MRR / nDCG@k，四组召回方式对照。

数据格式（JSONL，一行一条）：
    {"question": "...", "filter": {"projects": ["甲项目"]},
     "gold_chunk_ids": ["..."], "note": "可选备注"}

也支持只写"必须命中的原文片段"（标注省力）：
    {"question": "...", "gold_snippets": ["缓粘结预应力", "张拉时机"]}

**评价单位统一为 chunk。** 片段标注会在加载时解析成语料里包含该片段的全部 chunk_id，
之后 Recall / MRR / nDCG 用的是同一个 gold 集合。这一点必须钉死：如果 Recall 按原文片段
计数、MRR 和 nDCG 按 chunk 排序，同一个系统会同时算出 1.0 和 0.61，报告没法解释。

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


def resolve_gold(conn, row: dict) -> tuple[set[str], list[str]]:
    """把一条标注换算成 chunk_id 集合，返回 (gold, 警告)。

    `gold_chunk_ids` 直接可用；`gold_snippets` 用正文匹配解析——一个片段若出现在多个
    chunk 里，它们都算相关项（命中任意一条即满足该片段），这样三个指标的分母一致。
    """
    gold = {str(x) for x in row.get("gold_chunk_ids") or []}
    warnings: list[str] = []
    snippets = [s for s in dict.fromkeys(row.get("gold_snippets") or []) if s]
    for snippet in snippets:
        found = repo.chunk_ids_containing(conn, snippet)
        if not found:
            # 标错了或语料已变：留着会让 Recall 永远达不到 1，必须让人看见
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


async def hybrid_hits(services, question: str, filt: Filter, k: int) -> list[Hit]:
    hits, _, _ = await ask.retrieve(
        services.conn,
        services.store,
        services.require_embedder(),
        services.settings,
        question=question,
        filt=filt,
        recall_k=max(k * 5, 50),
    )
    return hits[:k]


async def rerank_hits(services, question: str, filt: Filter, k: int) -> list[Hit]:
    candidates = await hybrid_hits(services, question, filt, max(k * 4, 20))
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


async def _evaluate(services, rows: list[dict], k: int, modes: list[str]) -> dict:
    metrics: dict[str, dict[str, list[float]]] = {
        m: {"recall": [], "mrr": [], "ndcg": []} for m in modes
    }
    warnings: list[str] = []
    skipped: list[str] = []

    for row in rows:
        question = str(row.get("question") or "").strip()
        if not question:
            skipped.append("缺少 question 字段的记录")
            continue
        gold, row_warnings = resolve_gold(services.conn, row)
        warnings.extend(f"{question}：{w}" for w in row_warnings)
        if not gold:
            skipped.append(question)
            continue
        request_filter = Filter(**row["filter"]) if row.get("filter") else Filter()
        effective = ask.build_filter(question, request_filter, repo.project_names(services.conn))

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

    return {
        "modes": metrics,
        "samples": len(rows) - len(skipped),
        "skipped": skipped,
        "warnings": warnings,
    }


def run(dataset_path: Path, k: int, modes: list[str]) -> dict:
    services = build_services()
    try:
        rows = load_dataset(dataset_path)
        if not rows:
            raise ValueError(f"数据集为空：{dataset_path}")
        return evaluate(services, rows, k, modes)
    finally:
        services.close()


def report(result: dict, k: int) -> str:
    modes: dict[str, dict[str, list[float]]] = result["modes"]
    lines = [
        "# 检索评估报告\n",
        f"- 样本数：{result['samples']}（跳过 {len(result['skipped'])} 条无标注/无问题的记录）",
        f"- 指标：Recall@{k}、MRR、nDCG@{k}，相关性按 chunk 二值判定",
        f"- Recall@{k} = |TopK ∩ Gold| / |Gold|，Gold 为标注的相关 chunk 集合",
        "- 片段标注已解析成语料中包含它的全部 chunk，三个指标共用同一个 Gold\n",
        f"| 召回方式 | Recall@{k} | MRR | nDCG@{k} |",
        "|---|---|---|---|",
    ]
    for mode, values in modes.items():
        if not values["recall"]:
            lines.append(f"| {mode} | - | - | - |")
            continue
        lines.append(
            f"| {mode} | {statistics.mean(values['recall']):.3f} | "
            f"{statistics.mean(values['mrr']):.3f} | {statistics.mean(values['ndcg']):.3f} |"
        )
    if {"dense", "hybrid"} <= set(modes) and modes["dense"]["recall"] and modes["hybrid"]["recall"]:
        gain = statistics.mean(modes["hybrid"]["recall"]) - statistics.mean(
            modes["dense"]["recall"]
        )
        lines.append(f"\n混合召回相对纯向量：Recall@{k} {gain:+.3f}")
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
