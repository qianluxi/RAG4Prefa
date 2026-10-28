"""逐题归因：口语集为什么 Recall@5 只有 0.400。

不重新发明检索，直接复用 scripts/eval.py 里的 dense/sparse/hybrid 三种召回。
对每道口语题，回答三个问题：

1. gold 片段进没进**完整候选池**（dense/sparse 各取 50，hybrid 取池深 50）。
   —— 没进 = 召回问题（得动 embedding / 召回策略）；
   —— 进了但排在 6 名以后 = 排序问题（得动重排 / 调 recall_k）；
   —— 进了且排前 5 = 本来能答对（说明那 0.400 是被别处拖的）。

2. gold 片段在三种召回里的**最好排名**是多少。

3. 那道题输掉时，top-1 命中的是什么（看是不是"捞到了近似的但没捞到 gold 本身"）。

用法：
    python scripts/diagnose_recall.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa import ask, repo  # noqa: E402
from rag4prefa.domain import Filter  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402
from scripts.eval import (  # noqa: E402
    dense_hits,
    hybrid_pool,
    load_dataset,
    resolve_gold,
    sparse_hits,
)

ROOT = Path(__file__).resolve().parents[1]
POOL = 50


def _rank(hits, gold: set[str]) -> int | None:
    """gold 在命中列表里的最好排名（1-based）；没命中返回 None。"""
    for i, h in enumerate(hits, start=1):
        if h.chunk_id in gold:
            return i
    return None


async def _one(services, question: str, filt: Filter, gold: set[str]) -> dict:
    dense = await dense_hits(services, question, filt, POOL)
    sparse = await sparse_hits(services, question, filt, POOL)
    hybrid = await hybrid_pool(services, question, filt, 5)
    return {
        "dense_rank": _rank(dense, gold),
        "sparse_rank": _rank(sparse, gold),
        "hybrid_rank": _rank(hybrid, gold),
        "dense_top1": dense[0].title[:40] if dense else None,
        "sparse_top1": sparse[0].title[:40] if sparse else None,
    }


async def main() -> None:
    services = build_services()
    services.assert_query_ready()
    rows = load_dataset(ROOT / "eval" / "dataset.paraphrase.jsonl")

    results = []
    for row in rows:
        question = str(row.get("question") or "").strip()
        request_filter = Filter(**row["filter"]) if row.get("filter") else Filter()
        effective = ask.build_filter(question, request_filter, repo.project_names(services.conn))
        gold, _warn = resolve_gold(services.conn, row, effective)
        if not gold:
            continue
        r = await _one(services, question, effective, gold)
        results.append((question, gold, r))

    # ── 归因 ──
    miss_pool = 0      # 三种召回都捞不进池（召回问题）
    in_pool_low = 0    # 捞进池但最好排名 > 5（排序问题）
    in_top5 = 0        # 至少一种召回排进前 5（本该答对）
    rows_out = []
    for question, _gold, r in results:
        best = min(
            x for x in (r["dense_rank"], r["sparse_rank"], r["hybrid_rank"])
            if x is not None
        ) if any(
            x is not None for x in (r["dense_rank"], r["sparse_rank"], r["hybrid_rank"])
        ) else None
        if best is None:
            miss_pool += 1
            bucket = "没进池"
        elif best <= 5:
            in_top5 += 1
            bucket = "进了前5"
        else:
            in_pool_low += 1
            bucket = "进池但排后"
        rows_out.append((question, best, bucket, r))

    print(f"口语集 {len(results)} 题，归因：")
    print(f"  没进候选池（召回问题）   : {miss_pool}")
    print(f"  进池但最好排名 > 5（排序）: {in_pool_low}")
    print(f"  至少一种召回进前 5       : {in_top5}")
    print()
    print("=" * 96)
    for question, best, bucket, r in rows_out:
        print(f"\n[{bucket}] {question}")
        print(
            f"  最好排名 {best}   dense={r['dense_rank']} "
            f"sparse={r['sparse_rank']} hybrid={r['hybrid_rank']}"
        )
        print(f"  dense top1  : {r['dense_top1']}")
        print(f"  sparse top1 : {r['sparse_top1']}")


if __name__ == "__main__":
    asyncio.run(main())
