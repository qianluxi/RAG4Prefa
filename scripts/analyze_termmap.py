"""比较「词表注入」与同一次运行里的基线行，回答三个问题。

1. **救回了几题、丢了几题**：只看 `first_gold_rank`，不看均值——一题在 50 题上就是 0.02，
   均值很容易被两题的此消彼长抹平。
2. **未被覆盖的题是不是真的一个字都没变**：这是这套实验设计唯一的不变量。词表覆盖不到
   的题不该被任何东西影响，否则"覆盖组 vs 未覆盖组"的对照就不是同一批题之间的对照。
3. **增益是否集中在少数词条上**：`public_side=recalled` 的那些是动笔时已经想到某道题写的，
   它们带来的增益不算泛化证据，要单独算一遍。

用法：
    python scripts/analyze_termmap.py data/_diag_p50_v2.jsonl \
        --pair rerank=rerank_parallel --map rag4prefa/terms/public_to_standard.v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

TOP_K = 5


def _load(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _hit(rank: int | None) -> bool:
    return rank is not None and rank <= TOP_K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("diagnostics", type=Path)
    ap.add_argument("--pair", default="rerank=rerank_parallel", help="基线模式=处理模式")
    ap.add_argument("--map", type=Path, default=None, help="读 public_side 用")
    ap.add_argument(
        "--reference-map", type=Path, default=None,
        help="另一份词表（通常是 v1）。用它把覆盖组再切一刀：两表都覆盖 / 只有本表覆盖",
    )
    args = ap.parse_args()

    base_mode, treat_mode = args.pair.split("=")
    rows = _load(args.diagnostics)

    side = {}
    if args.map:
        spec = json.loads(args.map.read_text(encoding="utf-8"))
        side = {e["id"]: e.get("public_side", "vernacular") for e in spec["entries"]}

    # 用"另一份词表是否也覆盖"当作**客观**的"是否被针对性写过"的代理：
    # v1 是照着老 15 题的失败点写的，所以"v1 也覆盖"这件事本身就是一个事实记录，
    # 不依赖作者对 `public_side` 的自查。两套切法结论一致，才说明这个分类不是编的。
    ref = None
    if args.reference_map:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from rag4prefa.adapters.term_map import load_term_map

        ref = load_term_map(args.reference_map)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        entries = [e["id"] for e in (row.get("term_entries") or [])]
        row["_entries"] = entries
        if not entries:
            buckets["未覆盖"].append(row)
            continue
        buckets["覆盖"].append(row)
        kinds = {side.get(e, "vernacular") for e in entries}
        buckets["覆盖-含recalled" if "recalled" in kinds else "覆盖-纯vernacular"].append(row)
        if ref is not None:
            key = "覆盖-两表都覆盖" if ref.hits(row["question"]) else "覆盖-仅本表覆盖"
            buckets[key].append(row)

    print(f"{args.diagnostics.name}  {base_mode} → {treat_mode}  共 {len(rows)} 题\n")
    print(f"{'组':<20}{'题数':>4}{'基线P@5':>9}{'处理P@5':>9}{'Δ':>8}  赢/输")
    order = [
        "覆盖", "覆盖-纯vernacular", "覆盖-含recalled",
        "覆盖-两表都覆盖", "覆盖-仅本表覆盖", "未覆盖",
    ]
    for name in order:
        group = buckets.get(name) or []
        if not group:
            continue
        b = sum(_hit(r["modes"][base_mode]["first_gold_rank"]) for r in group)
        t = sum(_hit(r["modes"][treat_mode]["first_gold_rank"]) for r in group)
        win = sum(
            1
            for r in group
            if _hit(r["modes"][treat_mode]["first_gold_rank"])
            and not _hit(r["modes"][base_mode]["first_gold_rank"])
        )
        lose = sum(
            1
            for r in group
            if _hit(r["modes"][base_mode]["first_gold_rank"])
            and not _hit(r["modes"][treat_mode]["first_gold_rank"])
        )
        print(
            f"{name:<20}{len(group):>4}{b / len(group):>9.3f}{t / len(group):>9.3f}"
            f"{(t - b) / len(group):>+8.3f}  {win}/{lose}"
        )

    uncovered = buckets.get("未覆盖") or []
    # 不变量只看**指标依赖的东西**：gold 名次、以及 top-k 的集合。
    # 逐位比较 hits 会把重排服务的分数抖动算成泄漏——同一池子重排三次，末位两个近分
    # 候选（分差 0.001~0.02）的先后本来就会翻（2026-10-29 实测）。那是测量噪声。
    changed = [
        r
        for r in uncovered
        if r["modes"][base_mode]["first_gold_rank"] != r["modes"][treat_mode]["first_gold_rank"]
    ]
    boundary = [
        r
        for r in uncovered
        if r["modes"][base_mode]["first_gold_rank"] == r["modes"][treat_mode]["first_gold_rank"]
        and set(r["modes"][base_mode]["hits"]) != set(r["modes"][treat_mode]["hits"])
    ]
    reorder = [
        r
        for r in uncovered
        if set(r["modes"][base_mode]["hits"]) == set(r["modes"][treat_mode]["hits"])
        and r["modes"][base_mode]["hits"] != r["modes"][treat_mode]["hits"]
    ]
    print(f"\n不变量：未覆盖 {len(uncovered)} 题里 **gold 名次**变了 = {len(changed)}")
    for r in changed[:5]:
        print(f"  ! {r['question'][:40]}")
        print(f"    基线 {[h[:8] for h in r['modes'][base_mode]['hits']]}")
        print(f"    处理 {[h[:8] for h in r['modes'][treat_mode]['hits']]}")
    print(f"（另 {len(boundary)} 题 top-k 集合不同但 gold 名次相同——第 5 名边界的近分互换；")
    print(f"  另 {len(reorder)} 题只是末位顺序不同、集合相同。两者都是重排分数抖动，不影响指标）")
    for r in boundary:
        print(f"    ~ {r['question'][:34]}  gold 仍是第 "
              f"{r['modes'][base_mode]['first_gold_rank']} 名")

    print("\n覆盖组逐题（基线名次 → 处理名次）：")
    for r in buckets.get("覆盖") or []:
        b = r["modes"][base_mode]["first_gold_rank"]
        t = r["modes"][treat_mode]["first_gold_rank"]
        flag = "↑" if _hit(t) and not _hit(b) else ("↓" if _hit(b) and not _hit(t) else " ")
        kinds = ",".join(sorted({side.get(e, "?") for e in r["_entries"]}))
        print(f"  {flag} {str(b):>4} → {str(t):<4} [{kinds}] {r['question'][:34]}")
        print(f"       {r['_entries']} → {r['rewrites']}")

    # 每条词条最多覆盖几题：这个数决定"留一法"能证明什么
    per_entry: dict[str, int] = defaultdict(int)
    for r in rows:
        for e in r["_entries"]:
            per_entry[e] += 1
    multi = {e: n for e, n in per_entry.items() if n > 1}
    print(f"\n每条词条覆盖的题数：最多 {max(per_entry.values())}，"
          f"覆盖多题的词条 {len(multi)} 条 {multi if multi else ''}")
    print("（若每条只覆盖 1 题，则留一法等价于逐题看这一题有没有被救回，"
          "也就不存在『规模效应』）")


if __name__ == "__main__":
    main()
