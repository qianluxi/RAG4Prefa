"""量拒答判据的分数量级：库内有答案的题 vs 库里根本没有答案的题，分数分布差多少。

拒答门槛不能凭感觉写一个 0.5。`rerank_score` 是远程重排服务的原始分，绝对值会随
服务版本漂（`eval/report-identity-20261029.md` 里量过同一配置跨时段 0.839→0.806），
所以门槛必须**在同一批数据上量出来**，并且把"量的时候是什么分布"一起记下来——
否则半年后没人知道这个阈值还成不成立。

这里只做测量，不改行为：
- 库内题（有 gold）：记录 top1 分、gold 的分与名次 → 看"正确题的最低分"在哪；
- 库外题：记录 top1 分 → 看"错误题的最高分"在哪；
两个分布的间隔就是门槛能放的位置。间隔不存在（重叠）时必须说出来，而不是挑一个
能把当前 6 道题分开的数——那只是把这 6 道题背下来了。

用法：
    python scripts/calibrate_refusal.py
    python scripts/calibrate_refusal.py --save eval/reports/refusal-calibration.md
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa import ask, repo  # noqa: E402
from rag4prefa.domain import Filter  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

IN_CORPUS = (
    "eval/dataset.jsonl",
    "eval/dataset.paraphrase.jsonl",
    "eval/dataset.independent.jsonl",
    "eval/dataset.independent-b.jsonl",
)
OUT_OF_CORPUS = "eval/dataset.refuse.jsonl"


def _eval_module():
    spec = importlib.util.spec_from_file_location("ev_refusal", ROOT / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


async def _ask(services, ev, conn, row: dict, *, with_rerank: bool) -> dict:
    question = str(row.get("question") or "").strip()
    request = Filter(**(row.get("filter") or {}))
    eff = ask.build_filter(question, request, repo.project_names(conn))
    hits, _, _ = await ask.search(
        conn,
        services.store,
        services.embedder,
        services.settings,
        question=question,
        filt=eff,
        reranker=services.reranker if with_rerank else None,
        term_map=None,
    )
    gold, _warnings = ev.resolve_gold(conn, row, eff) if row.get("gold_chunk_ids") else (set(), [])
    top = hits[0] if hits else None
    gold_rank = None
    gold_score = None
    for i, hit in enumerate(hits, start=1):
        if hit.chunk_id in gold:
            gold_rank = i
            gold_score = hit.rerank_score
            break
    return {
        "question": question,
        "n_hits": len(hits),
        "top1_rerank": None if top is None else top.rerank_score,
        "top1_dense": None if top is None else top.dense_score,
        "top1_filename": "" if top is None else top.filename,
        "gold_rank": gold_rank,
        "gold_score": gold_score,
        "gold": sorted(gold),
    }


def _summary(name: str, rows: list[dict]) -> dict:
    scores = [r["top1_rerank"] for r in rows if r["top1_rerank"] is not None]
    dense = [r["top1_dense"] for r in rows if r["top1_dense"] is not None]
    return {
        "name": name,
        "n": len(rows),
        "scored": len(scores),
        "min": min(scores) if scores else None,
        "p10": _pct(scores, 0.10),
        "median": statistics.median(scores) if scores else None,
        "p90": _pct(scores, 0.90),
        "max": max(scores) if scores else None,
        "dense_min": min(dense) if dense else None,
        "dense_median": statistics.median(dense) if dense else None,
        "dense_max": max(dense) if dense else None,
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render(inside: list[dict], outside: list[dict], summaries: list[dict]) -> str:
    lines = [
        "### 分数分布",
        "",
        "| 组 | 题数 | min | p10 | 中位 | p90 | max |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['name']} | {s['n']}（有分 {s['scored']}） | {_fmt(s['min'])} | "
            f"{_fmt(s['p10'])} | {_fmt(s['median'])} | {_fmt(s['p90'])} | {_fmt(s['max'])} |"
        )
    lines.append("")

    inside_scores = [r["top1_rerank"] for r in inside if r["top1_rerank"] is not None]
    outside_scores = [r["top1_rerank"] for r in outside if r["top1_rerank"] is not None]
    if inside_scores and outside_scores:
        lo = min(inside_scores)
        hi = max(outside_scores)
        lines.append(
            f"库内最低 top1 分 {_fmt(lo)}，库外最高 top1 分 {_fmt(hi)}；"
            + (
                f"**两个分布不重叠**，任何落在 ({_fmt(hi)}, {_fmt(lo)}) 之间的门槛都能把这批分开。"
                if hi < lo
                else "**两个分布重叠**，不存在能把这批全分开的单一门槛。"
            )
        )
        lines.append("")

    lines.append("### 库外题逐题（该拒但当前会作答的那些）")
    lines.append("")
    lines.append("| 问题 | 候选数 | top1 分 | top1 出处 |")
    lines.append("|---|---|---|---|")
    for r in outside:
        lines.append(
            f"| {r['question'][:34]} | {r['n_hits']} | {_fmt(r['top1_rerank'])} | "
            f"{r['top1_filename'][:38]} |"
        )
    lines.append("")

    missed = [r for r in inside if r["gold_rank"] is None or r["gold_rank"] > 5]
    if missed:
        lines.append("### 库内但 gold 不在前 5 的题（门槛放在 gold 分之下会误拒它们）")
        lines.append("")
        lines.append("| 问题 | top1 分 | gold 名次 | gold 分 |")
        lines.append("|---|---|---|---|")
        for r in missed:
            rank = "池外" if r["gold_rank"] is None else r["gold_rank"]
            score = _fmt(r["gold_score"])
            lines.append(f"| {r['question'][:34]} | {_fmt(r['top1_rerank'])} | {rank} | {score} |")
        lines.append("")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description="量拒答门槛能放的位置")
    parser.add_argument("--save", type=Path, default=None, help="把结果写成 Markdown")
    parser.add_argument("--dump", type=Path, default=None, help="逐题明细存 JSON")
    parser.add_argument("--no-rerank", action="store_true", help="不用重排，只看融合分")
    args = parser.parse_args()

    services = build_services()
    ev = _eval_module()
    with_rerank = not args.no_rerank and ask.rerank_available(services.reranker)

    inside: list[dict] = []
    outside: list[dict] = []
    for rel in IN_CORPUS:
        path = ROOT / rel
        if not path.exists():
            continue
        for row in ev.load_dataset(path):
            rec = await _ask(services, ev, services.conn, row, with_rerank=with_rerank)
            rec["dataset"] = Path(rel).name
            inside.append(rec)
            print(f"[库内 {len(inside)}] {rec['question'][:26]}", file=sys.stderr, flush=True)

    for row in ev.load_dataset(ROOT / OUT_OF_CORPUS):
        rec = await _ask(services, ev, services.conn, row, with_rerank=with_rerank)
        rec["dataset"] = Path(OUT_OF_CORPUS).name
        outside.append(rec)
        print(f"[库外 {len(outside)}] {rec['question'][:26]}", file=sys.stderr, flush=True)

    if args.dump:
        # 先落盘再渲染：渲染段出过一次 TypeError，一次四分钟的测量就此全部丢失。
        args.dump.write_text(
            json.dumps({"inside": inside, "outside": outside}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    summaries = [_summary("库内（有 gold）", inside), _summary("库外（该拒）", outside)]
    body = render(inside, outside, summaries)
    print(body)
    if args.save:
        args.save.write_text(body + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
