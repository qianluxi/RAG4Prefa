"""检索质量评估：Recall@k / MRR / nDCG@k，并对比稠密与混合召回。

数据格式（JSONL，一行一条）：
    {"question": "...", "filter": {"projects": ["甲项目"]}, "gold_chunk_ids": ["...", "..."],
     "note": "可选备注"}

也支持不写 chunk_id、只写"必须命中的原文片段"（更好标注）：
    {"question": "...", "gold_snippets": ["缓粘结预应力", "张拉时机"]}

用法：
    python scripts/eval.py --dataset eval/dataset.jsonl --top-k 5
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

from rag4prefa import ask, repo  # noqa: E402
from rag4prefa.domain import Filter, Hit  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            rows.append(json.loads(line))
    return rows


def relevance(hit: Hit, gold_ids: set[str], gold_snippets: list[str]) -> float:
    if gold_ids:
        return 1.0 if hit.chunk_id in gold_ids else 0.0
    if gold_snippets:
        return 1.0 if any(s in hit.text for s in gold_snippets) else 0.0
    return 0.0


def n_gold(gold_ids: set[str], gold_snippets: list[str]) -> int:
    """该问题在语料里一共标了多少个相关项（Recall 的分母）。"""
    return len(gold_ids) if gold_ids else len(gold_snippets)


def retrieved_gold(
    hits: list[Hit], gold_ids: set[str], gold_snippets: list[str], k: int
) -> set[str]:
    """Top-k 命中的标准答案标识；用集合去重，避免同一片段被多次召回时重复计数。"""
    top = hits[:k]
    if gold_ids:
        return {h.chunk_id for h in top if h.chunk_id in gold_ids}
    if gold_snippets:
        # 一个标注片段在任意一条命中里出现即算找回
        return {s for s in gold_snippets if any(s in h.text for h in top)}
    return set()


def recall_at_k(
    hits: list[Hit], gold_ids: set[str], gold_snippets: list[str], k: int
) -> float:
    """Recall@k = |TopK ∩ Gold| / |Gold|。

    分母是标注的相关项总数，**不是**返回条数——按返回条数算得到的是 Precision，
    曾在两标一中的场景把 1.0 算成 0.4。
    """
    total = n_gold(gold_ids, gold_snippets)
    if not total:
        return 0.0
    return len(retrieved_gold(hits, gold_ids, gold_snippets, k)) / total


def mrr(rels: list[float]) -> float:
    for i, rel in enumerate(rels, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[float], k: int, total: int) -> float:
    """二值相关性下的 nDCG@k；IDCG 用标注的相关项总数构造理想排序。"""
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(total, k)))
    return dcg / idcg if idcg else 0.0


async def dense_only(services, question: str, filt: Filter, k: int) -> list[Hit]:
    embedder = services.require_embedder()
    vector = await asyncio.to_thread(embedder.embed_one, question)
    pairs = await asyncio.to_thread(lambda: services.store.search(vector, limit=k, filt=filt))
    details = repo.get_chunks(services.conn, [cid for cid, _ in pairs])
    return [Hit(score=s, **details[cid]) for cid, s in pairs if cid in details]


async def run(dataset_path: Path, k: int, modes: list[str]) -> dict:
    services = build_services()
    rows = load_dataset(dataset_path)
    results: dict[str, dict[str, list[float]]] = {
        m: {"recall": [], "mrr": [], "ndcg": []} for m in modes
    }

    for row in rows:
        question = row["question"]
        gold_ids = set(row.get("gold_chunk_ids") or [])
        gold_snippets = row.get("gold_snippets") or []
        request_filter = Filter(**row["filter"]) if row.get("filter") else Filter()
        effective = ask.build_filter(question, request_filter, repo.project_names(services.conn))

        for mode in modes:
            if mode == "dense":
                hits = await dense_only(services, question, effective, k)
            else:
                hits, _, _ = await ask.retrieve(
                    services.conn,
                    services.store,
                    services.require_embedder(),
                    services.settings,
                    question=question,
                    filt=effective,
                    recall_k=max(k * 5, 50),
                )
                hits = hits[:k]
            rels = [relevance(h, gold_ids, gold_snippets) for h in hits]
            total = n_gold(gold_ids, gold_snippets)
            results[mode]["recall"].append(recall_at_k(hits, gold_ids, gold_snippets, k))
            results[mode]["mrr"].append(mrr(rels))
            results[mode]["ndcg"].append(ndcg_at_k(rels, k, total))
    services.close()
    return results


def report(results: dict, k: int, n: int) -> str:
    lines = [
        "# 检索评估报告\n",
        f"- 样本数：{n}",
        f"- 指标：Recall@{k}、MRR、nDCG@{k}（相关性按 gold 二值判定）",
        f"- Recall@{k} = |TopK ∩ Gold| / |Gold|，分母为标注的相关项总数\n",
        f"| 召回模式 | Recall@{k} | MRR | nDCG@{k} |",
        "|---|---|---|---|",
    ]
    for mode, metrics in results.items():
        lines.append(
            f"| {mode} | {statistics.mean(metrics['recall']):.3f} | "
            f"{statistics.mean(metrics['mrr']):.3f} | {statistics.mean(metrics['ndcg']):.3f} |"
        )
    if {"dense", "hybrid"} <= set(results):
        gain = statistics.mean(results["hybrid"]["recall"]) - statistics.mean(
            results["dense"]["recall"]
        )
        lines.append(f"\n混合召回相对纯向量：Recall@{k} {gain:+.3f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="检索质量评估")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", default="dense,hybrid")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"数据集不存在：{path}（可先复制 eval/dataset.example.jsonl）", file=sys.stderr)
        return 2
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = asyncio.run(run(path, args.top_k, modes))
    text = report(results, args.top_k, len(load_dataset(path)))
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
