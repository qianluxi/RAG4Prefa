"""融合方式的离线对照：为什么用本路标准化，以及生产池深下重排差多少。

两件事合成一个脚本，因为它们共用同一份候选缓存：

1. **默认**（`--pool 50`，不调重排）：把两种融合的 Recall@5 / MRR / 池内覆盖率并排，
   用来回答"换融合方式到底改了什么"。候选缓存落在 `data/.fusion-cache-*.jsonl`，
   同一份数据集只算一次——dense 检索要走远程 embedding，扫第二遍就是白花钱。
2. **`--rerank`**：对两种融合各自取前 `--pool` 条送一次真实重排。
   这是生产场景（`candidate_k = max(recall_k // 2, top_k * 3)` = 25 深）。
   评测脚本 `eval.py` 固定用 50 深池，看不到这一段——重排对池内每条独立打分，
   池内顺序不影响它，只有"gold 有没有进池"才影响。

用法：
    python scripts/diagnose_fusion.py --dataset eval/dataset.paraphrase.jsonl
    python scripts/diagnose_fusion.py --dataset eval/dataset.paraphrase.jsonl \
        --pool 25 --rerank
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts/eval.py")
assert _spec and _spec.loader
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)

from rag4prefa import ask, repo  # noqa: E402
from rag4prefa.domain import Filter, Hit  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

K = 60


def fuse_rrf(runs) -> list[str]:
    scores: dict[str, float] = {}
    for ranked, weight in runs:
        for rank, (chunk_id, _) in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (K + rank + 1)
    # 同分时**不按 id 排**：`ask._to_hits` 用的是稳定排序，同分保持插入序，
    # 也就是"稠密那一路先出现"。加了 `, kv[0]` 会让「dense 第 1 与 sparse 第 1 同分」
    # 这一情形换人——`1/61` 正是最容易撞上的分数（2026-10-29 实测整行差 1 题）。
    return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def fuse_zscore(runs) -> list[str]:
    """与 `ask.fuse_scores(mode="zscore")` 同一套算术，单独写一份是为了让对照不依赖被测代码。"""
    scores: dict[str, float] = {}
    for ranked, weight in runs:
        if not ranked:
            continue
        values = [v for _, v in ranked]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if std == 0:
            for chunk_id, _ in ranked:
                scores.setdefault(chunk_id, 0.0)
            continue
        for chunk_id, value in ranked:
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight * max((value - mean) / std, 0.0)
    return [c for c, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def cache_path(dataset: Path) -> Path:
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()[:8]
    return ROOT / "data" / f".fusion-cache-{dataset.stem}-{digest}.jsonl"


async def build_cache(services, dataset: Path) -> list[dict]:
    cached = cache_path(dataset)
    if cached.exists():
        rows = [json.loads(ln) for ln in cached.read_text(encoding="utf-8").splitlines() if ln]
        print(f"复用缓存 {cached.name}（{len(rows)} 题）")
        return rows
    pool = max(services.settings.recall_k, 50)
    rows = []
    for row in ev.load_dataset(dataset):
        question = str(row["question"]).strip()
        request_filter = Filter(**row["filter"]) if row.get("filter") else Filter()
        effective = ask.build_filter(question, request_filter, repo.project_names(services.conn))
        gold, _ = ev.resolve_gold(services.conn, row, effective)
        if not gold:
            continue
        # **不能走 `ev.dense_hits`**：它会丢掉 chunks 表里查不到的 id（`if cid in details`），
        # 而 `ask.retrieve` 用的是原始 pairs。差一条候选就能让 RRF 的第 5 名换人——
        # 2026-10-29 实测：「验收的时候业主必须到场吗？」两版第 5 名不同，整行差 1 题。
        # 对照脚本必须与被对照的那条链路用同一套候选，所以这里直接调 store / repo。
        vector = await asyncio.to_thread(services.require_embedder().embed_one, question)
        dense = await asyncio.to_thread(
            lambda v=vector, f=effective: services.store.search(v, limit=pool, filt=f)
        )
        sparse_query = f"{question} {effective.keyword}" if effective.keyword else question
        sparse = await asyncio.to_thread(
            lambda q=sparse_query, f=effective: repo.fts_search(
                services.conn, q, limit=pool, filt=f
            )
        )
        rows.append({
            "question": question,
            "gold": sorted(gold),
            "dense": [[c, sc] for c, sc in dense],
            "sparse": [[c, sc] for c, sc in sparse],
        })
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    print(f"写入缓存 {cached.name}（{len(rows)} 题）")
    return rows


def _rank(ids: list[str], gold: set[str]) -> int | None:
    return next((i for i, c in enumerate(ids, 1) if c in gold), None)


async def report(services, dataset: Path, pool: int, rerank: bool, top_k: int) -> None:
    rows = await build_cache(services, dataset)
    print(f"\n{'=' * 74}\n{dataset.name}（{len(rows)} 题，池深 {pool}，top-{top_k}）")
    stats: dict[str, list[tuple[str, int | None]]] = {"rrf": [], "zscore": []}
    covered = {"rrf": 0, "zscore": 0}

    for row in rows:
        gold = set(row["gold"])
        runs = (
            ([(c, s) for c, s in row["dense"]], 1.0),
            ([(c, s) for c, s in row["sparse"]], 1.0),
        )
        for name, fn in (("rrf", fuse_rrf), ("zscore", fuse_zscore)):
            # 融合后**先过滤孤立 id 再截断**，与 `ask._to_hits` 同一条规则：
            # 索引里有、`chunks` 表里没有的 id 会被生产链路直接跳过（"不编造内容"）。
            # 漏了这条，一个孤立 id 就能顶掉第 5 名的 gold——2026-10-29 实测整行差 1 题。
            ordered = fn(runs)[: pool * 2]
            known = repo.get_chunks(services.conn, ordered)
            ids = [c for c in ordered if c in known][:pool]
            if any(c in gold for c in ids):
                covered[name] += 1
            if rerank:
                info = repo.get_chunks(services.conn, ids)
                hits = [Hit(score=0.0, **info[c]) for c in ids if c in info]
                ranked = await services.reranker.rerank(row["question"], hits, top_k)
                rank = _rank([h.chunk_id for h in ranked], gold)
            else:
                rank = _rank(ids[:top_k], gold)
            stats[name].append((row["question"], rank))

    if not rerank:
        print(f"{'融合':<8}{'池内含 gold':>12}{'Recall@top_k':>14}{'MRR':>8}")
        for name in ("rrf", "zscore"):
            got = stats[name]
            hit = sum(1 for _, r in got if r)
            mrr = sum(1 / r for _, r in got if r) / len(got)
            print(f"{name:<8}{covered[name]:>8}/{len(got)}{hit:>11}/{len(got)}{mrr:>8.3f}")
    else:
        print(f"{'融合':<8}{'池内含 gold':>12}{'重排后 Recall@top_k':>20}{'MRR':>8}")
        for name in ("rrf", "zscore"):
            got = stats[name]
            hit = sum(1 for _, r in got if r)
            mrr = sum(1 / r for _, r in got if r) / len(got)
            print(f"{name:<8}{covered[name]:>8}/{len(got)}{hit:>17}/{len(got)}{mrr:>8.3f}")

    win = lose = 0
    for (question, a), (_, b) in zip(stats["rrf"], stats["zscore"], strict=True):
        ok = lambda x: x is not None and x <= top_k  # noqa: E731
        if ok(b) and not ok(a):
            win += 1
            print(f"    + {a} -> {b}  {question[:36]}")
        elif ok(a) and not ok(b):
            lose += 1
            print(f"    - {a} -> {b}  {question[:36]}")
    print(f"  zscore 相对 rrf：赢 {win} 输 {lose}")


def main() -> int:
    parser = argparse.ArgumentParser(description="融合方式对照（含生产池深下的重排）")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--pool", type=int, default=50, help="融合后保留多少条（生产默认 25）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--rerank", action="store_true", help="对池内候选跑真实重排（慢，每题一次远程调用）"
    )
    args = parser.parse_args()
    datasets = args.dataset or ["eval/dataset.paraphrase.jsonl", "eval/dataset.jsonl"]
    services = build_services()
    for item in datasets:
        path = Path(item)
        if not path.is_absolute():
            path = ROOT / path
        asyncio.run(report(services, path, args.pool, args.rerank, args.top_k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
