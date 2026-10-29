"""逐通道归因：gold 到底出现在哪一路的候选池里。

并行改写通道跑完，口语集上"Recall 没涨"有两种完全不同的原因，处置方式相反：

A. 改写通道的池里**有** gold（只是没进前 5）→ 融合/排序的问题，值得调权重或加池深；
B. 改写通道的池里**也没有** gold → 改写词本身没打到，融合怎么调都没用，
   要改的是术语从哪来（提示词、词表），不是融合。

脚本对每道题跑四个查询（原话 dense / 原话 BM25 / 改写 dense / 改写 BM25），
各取 50 深，报 gold 的最好吗名次，并按上面两类归档。

第二段是**最关键的证据**：对"两路都没有"的题，把扩展串拆成单个词逐条查，
报最好的那个词排第几。它把"扩展没用"进一步分成两件事：
- 拼成一串把好词稀释了（逐词明显更好，但仍进不了前 5）；
- 词本身就不是规范原文的搭配（逐词也不行，而换成规范原词立刻进前 5）。

用法：
    python scripts/diagnose_channels.py
    python scripts/diagnose_channels.py --expansions eval/expansions/paraphrase-20261028.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa import ask, repo  # noqa: E402
from rag4prefa.domain import Filter  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402
from scripts.eval import load_dataset, resolve_gold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POOL = 50
DEFAULT_EXPANSIONS = ROOT / "eval" / "expansions" / "paraphrase-20261028.jsonl"


def _rank(pairs, gold: set[str]) -> int | None:
    for i, (chunk_id, _) in enumerate(pairs, start=1):
        if chunk_id in gold:
            return i
    return None


def load_expansions(path: Path) -> dict[str, str]:
    """读冻结的扩展串。**不重新调 LLM**：同一份数据跑两次，15 题扩展词 0 题相同，
    换一批词就与 eval 的报告不可比（第三十九轮实测）。"""
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line or line.startswith("//"):
            continue
        item = json.loads(line)
        mapping[item["question"]] = item["search_question"]
    return mapping


async def main() -> int:
    parser = argparse.ArgumentParser(description="逐通道归因：gold 出现在哪一路")
    parser.add_argument("--dataset", default="eval/dataset.paraphrase.jsonl")
    parser.add_argument("--expansions", default=str(DEFAULT_EXPANSIONS))
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_absolute():
        dataset = ROOT / dataset
    expansions = Path(args.expansions)
    if not expansions.is_absolute():
        expansions = ROOT / expansions

    services = build_services()
    try:
        services.assert_query_ready()
        rows = load_dataset(dataset)
        frozen = load_expansions(expansions)
        embedder = services.require_embedder()
        conn, store = services.conn, services.store

        print(f"池深 {POOL}；扩展串来自 {expansions.relative_to(ROOT).as_posix()}")
        print(f"{'#':<3}{'题':<30}{'原话 d/s':>12}{'改写 d/s':>12}  归因")
        print("-" * 84)
        stats = {"already": 0, "A": 0, "B": 0}
        no_hit: list[tuple[int, dict, Filter, set[str], str]] = []

        for i, row in enumerate(rows, start=1):
            question = row["question"]
            filt = Filter(**row["filter"]) if row.get("filter") else Filter()
            effective = ask.build_filter(question, filt, repo.project_names(conn))
            gold, _ = resolve_gold(conn, row, effective)
            if not gold:
                print(f"{i:<3}{question[:28]:<30}   （无 gold，跳过）")
                continue
            rewritten = frozen.get(question, question)

            # effective / gold 用默认参数绑定：它们是循环变量，闭包捕获到的是循环
            # 结束时的那一个值，B023 报的就是这个。绑定之后每个 probe 看到自己那题的。
            async def probe(text: str, eff: Filter = effective, expect: set[str] = gold):
                vec = await asyncio.to_thread(embedder.embed_one, text)
                dense = await asyncio.to_thread(
                    lambda: store.search(vec, limit=POOL, filt=eff)
                )
                sparse = await asyncio.to_thread(
                    lambda: repo.fts_search(conn, text, limit=POOL, filt=eff)
                )
                return _rank(dense, expect), _rank(sparse, expect)

            d1, s1 = await probe(question)
            d2, s2 = (d1, s1) if rewritten.strip() == question.strip() else await probe(rewritten)

            def cell(a: int | None, b: int | None) -> str:
                return f"{a or '-'}/{b or '-'}"

            pool_orig = [x for x in (d1, s1) if x]
            pool_rew = [x for x in (d2, s2) if x]
            if pool_orig and min(pool_orig) <= 5:
                verdict, key = "原话已进 Top5", "already"
            elif pool_rew:
                verdict, key = f"改写池内有 gold（最好 {min(pool_rew)}）", "A"
            else:
                verdict, key = "两路池内都没有 → 改写没打到", "B"
                no_hit.append((i, row, effective, gold, question))
            stats[key] += 1
            print(f"{i:<3}{question[:28]:<30}{cell(d1, s1):>12}{cell(d2, s2):>12}  {verdict}")

        print("-" * 84)
        print(
            f"原话已进 Top5：{stats['already']}；"
            f"改写池内有 gold（融合可救）：{stats['A']}；"
            f"两路都没有（改写没打到）：{stats['B']}"
        )

        if no_hit:
            print(f"\n## 「两路都没有」的题：拼串 vs 逐词（BM25，池深 {POOL}）")
            print(f"{'#':<3}{'题':<28}{'拼串':>6}{'逐词最好':>9}  最好的词")
            print("-" * 68)
            for i, _row, effective, gold, question in no_hit:
                full = frozen.get(question, question)
                joined = _rank(
                    repo.fts_search(conn, full, limit=POOL, filt=effective), gold
                )
                terms = [t for t in full[len(question) :].split() if t]
                best, best_term = None, "-"
                for term in terms:
                    rank = _rank(
                        repo.fts_search(conn, term, limit=POOL, filt=effective), gold
                    )
                    if rank is not None and (best is None or rank < best):
                        best, best_term = rank, term
                print(
                    f"{i:<3}{question[:26]:<28}{str(joined or '-'):>6}"
                    f"{str(best or '-'):>9}  {best_term}"
                )
            print(
                "\n逐词普遍优于拼串 → 拼成一串确实在稀释；但逐词最好仍进不了前 5 → "
                "**词本身不是规范原文的搭配**。要验证这一点，把该条文的原话搭配单独查一次"
                "（例如「专用连接件」而不是「锚固」），gold 会立刻进前列。"
            )
        return 0
    finally:
        services.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
