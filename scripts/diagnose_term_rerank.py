"""三种重排查询形态的对照：原话 / 术语串 / 原话+术语串。

只问一件事：把重排查询换成术语串，为什么会让**原本就在第 1 名**的题掉出去？

2026-10-29 实测（口语 50 题，`rerank 0.680 → rerank_parallel 0.720`）：净 +0.040，
但拆开是 5 赢 3 输，而 3 道输的题基线全部是第 1 名。也就是说"术语串比原话好"这个
结论只对**读不懂口语的那批题**成立；对本来就能读懂的题，它是在拿掉信号。

候选池固定为带术语通道的那一份（`rerank_hits(..., rewrites=...)`），三种形态只换
`rerank_query`，所以差值只可能来自查询串本身。

用法：
    python scripts/diagnose_term_rerank.py --dataset eval/dataset.paraphrase.jsonl \
        --term-map rag4prefa/terms/public_to_standard.v2.json
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa import ask, repo  # noqa: E402
from rag4prefa.adapters.term_map import load_term_map  # noqa: E402
from rag4prefa.domain import Filter  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

TOP_K = 5


def _eval_module():
    spec = importlib.util.spec_from_file_location("ev", ROOT / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank(hits, gold: set[str]) -> int | None:
    for i, hit in enumerate(hits, 1):
        if hit.chunk_id in gold:
            return i
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--term-map", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    ev = _eval_module()
    tm = load_term_map(args.term_map)
    rows = ev.load_dataset(args.dataset)
    services = build_services()

    variants = ("原话", "术语串", "原话+术语串")
    tallies = {v: 0 for v in variants}
    covered = 0
    print(f"{'题':<34}{'原话':>6}{'术语':>6}{'两者':>6}   术语串")
    for row in rows:
        question = row["question"]
        rewrites = tuple(tm.rewrite(question, form="terms"))
        if not rewrites:
            continue
        covered += 1
        filt = Filter(**row["filter"]) if row.get("filter") else Filter()
        effective = ask.build_filter(question, filt, repo.project_names(services.conn))
        gold, _ = ev.resolve_gold(services.conn, row, effective)
        joined = " ".join(rewrites)

        ranks = {}
        for name, query in (
            ("原话", question),
            ("术语串", joined),
            ("原话+术语串", f"{question} {joined}"),
        ):
            hits = await ev.rerank_hits(
                services, question, effective, args.top_k,
                rerank_query=query, rewrites=rewrites, rewrite_weight=0.5,
            )
            ranks[name] = _rank(hits, gold)
            if ranks[name] is not None and ranks[name] <= args.top_k:
                tallies[name] += 1

        def cell(r: int | None) -> str:
            return "✓" if r is not None and r <= args.top_k else ("·" if r is None else str(r))

        print(
            f"{question[:32]:<34}{cell(ranks['原话']):>6}{cell(ranks['术语串']):>6}"
            f"{cell(ranks['原话+术语串']):>6}   {joined}"
        )

    print(f"\n覆盖 {covered} 题，命中前 {args.top_k} 的题数：")
    for v in variants:
        print(f"  {v:<12}{tallies[v]:>4}/{covered}  = {tallies[v] / covered:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
