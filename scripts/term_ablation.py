"""术语词表的两条作用路径拆开测：多召回一路候选 / 换掉重排查询（2×2 固定池消融）。

背景：第四十一轮到现在，"词表让口语题从 0.640 到 0.720"这个数字一直是**两件事的合成**——

  ① 命中的术语作为独立通道，多召回一批候选（gold 可能因此**进池**）；
  ② 重排查询从原话换成「原话 + 术语串」（gold 在池内时可能因此**往前排**）。

现有 `eval.py` 只能给出"两个都开"和"两个都关"两行，中间两格做不出来：基线的
`baseline_rerank_query` 有"带改写串就退回原话"的保护，那是为了让对照组干净，代价是
"池不动、只换重排查询"这一格**在报告里不存在**。于是一个 +0.080 的差值，无法回答
"到底是多召回救回来的，还是重排重读换回来的"——而这决定了下一步该往哪投：
补词表覆盖面（召回），还是换词表形态（重排查询）。

2×2 的四格：

|            | 重排读原话 | 重排读原话+术语 |
|------------|-----------|----------------|
| 池=原话     | P0Q0      | P0Q1           |
| 池=+术语通道 | P1Q0      | P1Q1           |

P0Q0 就是 `rerank` 基线；P1Q1 是当前生产形态；P0Q1 与 P1Q0 是缺的那两格。

**同一道题的两个池各只算一次**，池内每条候选被四种组合各打一次分。能这么省是因为
重排对池内每条**独立打分、与池内顺序无关**——`tests/test_p1_ablation.py` 钉住了这条
（把池打乱后重排，top-k 必须逐位相同）。池深 25 与 50 也不各跑一遍：融合排序与
`candidate_k` 无关，25 池就是 50 池的前 25 条，所以从一次 50 深的全序里按池成员过滤
就能派生。

重排重复 `--repeat` 次：同一池三次重排，末位两个近分候选（分差 0.001~0.02）先后会翻，
一题就是 0.067（15 题分母）——与要测量的效应同量级。主指标取多数票，同时把"翻动"
单独报出来，免得把一个边界题读成趋势。

用法：
    python scripts/term_ablation.py --repeat 3
    python scripts/term_ablation.py --save eval/reports/termmap-ablation-20261029.md
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa import ask, repo  # noqa: E402
from rag4prefa.adapters.term_map import load_term_map  # noqa: E402
from rag4prefa.domain import Filter  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

# 四格。名字里的 P/Q 是"池"与"查询"，顺序固定，报告与测试都按这个顺序读。
CELLS = ("P0Q0", "P0Q1", "P1Q0", "P1Q1")
CELL_LABELS = {
    "P0Q0": ("原话", "原话"),
    "P0Q1": ("原话", "原话+术语串"),
    "P1Q0": ("+术语通道", "原话"),
    "P1Q1": ("+术语通道", "原话+术语串"),
}


def _eval_module():
    """借用 `eval.py` 的数据加载与 gold 解析，避免两处口径漂移。"""
    spec = importlib.util.spec_from_file_location("ev_ablation", ROOT / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_gold_rank(seq, gold: set[str]) -> int | None:
    for i, hit in enumerate(seq, start=1):
        if hit.chunk_id in gold:
            return i
    return None


def _majority_rank(ranks: list[int | None]) -> int | None:
    """重复重排下取众数；平局取最小名次（不替读者挑一个更乐观的数）。"""
    known = [r for r in ranks if r is not None]
    if not known:
        return None
    counts = Counter(known)
    best = max(counts.values())
    return min(r for r, c in counts.items() if c == best)


def _ranks_at_depths(full: list, pool: list, gold: set[str], depths) -> dict[int, int | None]:
    """把一次「全序」按池成员过滤，得到各池深下的 gold 名次。

    这是"两档池深只跑一次重排"的全部实现。它成立的前提有两层，缺任何一层都会
    静默算错（而报告上只是两个正常的数）：
      - 融合排序与 `candidate_k` 无关，所以深池的前 N 条就是浅池；
      - 重排对池内每条独立打分，所以浅池的排序 = 全序里属于浅池的那些条目的相对顺序。
    两条都由 `tests/test_p1_ablation.py` 分别钉住，且测试直接调本函数——
    不调的话，"测试自己重写一遍派生"会让变异从断言下面溜过去（第一次就这么翻的）。
    """
    ranks: dict[int, int | None] = {}
    for depth in depths:
        allowed = {h.chunk_id for h in pool[:depth]}
        ordered = [h for h in full if h.chunk_id in allowed]
        ranks[depth] = _first_gold_rank(ordered, gold)
    return ranks


def _hit(ranks: list[int | None], top_k: int) -> bool:
    """多数票命中：半数以上（含）的重复里 gold 进了前 top_k。"""
    votes = sum(1 for r in ranks if r is not None and r <= top_k)
    return votes * 2 >= len(ranks)


async def _rerank_full(services, query: str, pool, *, attempts: int = 3):
    """重排整池，返回全序。

    一轮要调几百次远程服务，偶发一次 5xx 不该让整轮作废（那是二十分钟）。
    但**不吞掉失败**：重试到第 3 次仍失败就抛——那是环境问题，继续跑下去
    得到的是"某一格少了几题"的均值，报告上完全看不出来。
    """
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return await services.reranker.rerank(query, pool, len(pool))
        except Exception as exc:  # noqa: BLE001 - 远程服务的异常类型不统一
            last = exc
            await asyncio.sleep(1.0 * (attempt + 1))
    assert last is not None
    raise last


async def _retrieve_pool(services, question: str, filt, rewrites, deep: int, weight: float):
    hits, _, _ = await ask.retrieve(
        services.conn,
        services.store,
        services.require_embedder(),
        services.settings,
        question=question,
        filt=filt,
        recall_k=deep,
        candidate_k=deep,
        rewrites=rewrites,
        rewrite_weight=weight,
    )
    return hits


async def measure(
    services,
    ev,
    rows: list[dict],
    term_map,
    *,
    top_k: int = 5,
    depths: tuple[int, ...] = (25, 50),
    repeat: int = 3,
    progress: bool = False,
) -> dict:
    """逐题跑四格，返回原样的 rank 记录（聚合留给 `summarise`）。"""
    deep = max(depths)
    weight = services.settings.term_rewrite_weight
    records: list[dict] = []

    for index, row in enumerate(rows, start=1):
        question = str(row.get("question") or "").strip()
        if not question:
            continue
        request_filter = Filter(**row["filter"]) if row.get("filter") else Filter()
        effective = ask.build_filter(
            question, request_filter, repo.project_names(services.conn)
        )
        gold, _ = ev.resolve_gold(services.conn, row, effective)
        if not gold:
            continue

        rewrites = tuple(term_map.rewrite(question, form="joined")) if term_map else ()
        joined = " ".join(rewrites)
        queries = {"Q0": question}
        if joined:
            # 只加不减：术语串永远拼在原话后面。纯术语串那一版已实测会在"本来读得懂"
            # 的题上把第 1 名打出去，这里不重复验它。
            queries["Q1"] = f"{question} {joined}"

        pools = {"P0": await _retrieve_pool(services, question, effective, (), deep, weight)}
        if rewrites:
            pools["P1"] = await _retrieve_pool(
                services, question, effective, rewrites, deep, weight
            )
        else:
            # 没命中词表 = 没有术语通道可开。此时 P1 与 P0 是同一份池、Q1 也不存在，
            # 四格按定义退化成一格。**不能给它编一个"术语池"**：那样"未覆盖题四格
            # 全同"就成了脚本自己造的事实，而不是词表的事实。
            pools["P1"] = pools["P0"]

        cells: dict[str, dict] = {}
        for pid in ("P0", "P1"):
            for qid, query in queries.items():
                key = f"{pid}{qid}"
                if key in cells:
                    continue
                # 池相同且查询相同 → 同一格，不重复花钱调重排。
                if pools[pid] is pools["P0"] and qid == "Q0" and "P0Q0" in cells:
                    cells[key] = cells["P0Q0"]
                    continue
                runs: dict[int, list[int | None]] = {d: [] for d in depths}
                for _ in range(repeat):
                    # 传 len(pool) 让重排返回**全序**，否则只能看到前 k 名，
                    # 池深 25 与 50 就没法从同一次调用里派生。
                    full = await _rerank_full(services, query, pools[pid])
                    per_depth = _ranks_at_depths(full, pools[pid], gold, depths)
                    for depth, rank in per_depth.items():
                        runs[depth].append(rank)
                cells[key] = runs

        if progress:
            # 一轮要调几百次远程重排，没有进度就只能盯着空日志猜它是不是卡住了。
            # 打到 stderr，不污染 stdout 的 Markdown。
            mark = "命中" if rewrites else "未命中"
            print(f"[{index}/{len(rows)}] {mark} {question[:28]}", file=sys.stderr, flush=True)
        records.append(
            {
                "question": question,
                "covered": bool(rewrites),
                # 只记**真正生效**的命中；`explain` 里还有被单字规则挡下的条目
                # （`matched_by` 为空），混进来会让"覆盖了几条"虚高。
                "terms": [e["id"] for e in term_map.explain(question) if e.get("matched_by")]
                if term_map
                else [],
                "rewrite": joined,
                "cells": cells,
                "depths": depths,
                "top_k": top_k,
            }
        )
    return {"records": records, "depths": depths, "top_k": top_k, "repeat": repeat}


def _mean_rows(records: list[dict], cell: str, depth: int, top_k: int) -> dict:
    """一组题在某格某池深上的三项数：命中率、MRR、池内含 gold 的题数。"""
    hits = 0
    mrr_values: list[float] = []
    has_gold = 0
    unstable = 0
    n = 0
    for rec in records:
        runs = rec["cells"].get(cell, {}).get(depth)
        if runs is None:
            continue
        n += 1
        if _hit(runs, top_k):
            hits += 1
        mrr_values.append(statistics.fmean(1.0 / r if r else 0.0 for r in runs))
        if any(r is not None for r in runs):
            has_gold += 1
        if len({r for r in runs}) > 1:
            unstable += 1
    if not n:
        return {"n": 0, "recall": 0.0, "mrr": 0.0, "pool_has_gold": 0, "unstable": 0}
    return {
        "n": n,
        "recall": hits / n,
        "mrr": statistics.fmean(mrr_values),
        "pool_has_gold": has_gold,
        "unstable": unstable,
    }


def summarise(result: dict, *, group: str = "all") -> dict:
    records = result["records"]
    if group == "covered":
        records = [r for r in records if r["covered"]]
    elif group == "uncovered":
        records = [r for r in records if not r["covered"]]
    top_k = result["top_k"]
    return {
        group: {
            str(depth): {cell: _mean_rows(records, cell, depth, top_k) for cell in CELLS}
            for depth in result["depths"]
        },
        "n": len(records),
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _file_sha(path: Path) -> str:
    """文件内容指纹（CRLF 归一化）。

    换行必须归一化：Windows 工作区是 CRLF、`eol=lf` 只作用于提交进去的内容，
    不归一化的话同一份代码在两台机器上算出不同的 sha（这个坑 P0-2 踩过一次）。
    """
    blob = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(blob).hexdigest()[:12]


def experiment_identity(ev, dataset: Path, term_map_path: Path) -> dict:
    """这次实验的完整身份：生产代码 + **实验脚本自己** + 词表 + 数据。

    通用评测指纹（`eval._source_fingerprint`）只覆盖 `rag4prefa/**` 与 `scripts/eval.py`，
    而这个消融的测量逻辑在 `scripts/term_ablation.py` 里——只报前者会出现"生产代码一致，
    实验脚本换了"的报告，两份数字却看着可比。词表与数据同理：同一句 term_rewrite_weight
    配两个词表版本，结果不是同一件事。
    """
    code = ev._code_state()
    return {
        "commit": code.get("commit"),
        "src_sha": code.get("source_sha") or code.get("src_sha"),
        "dirty": code.get("dirty"),
        "script_sha": _file_sha(Path(__file__)),
        "term_map": term_map_path.name,
        "term_map_sha": _file_sha(term_map_path),
        "dataset": dataset.name,
        "dataset_sha": _file_sha(dataset),
    }


def render_identity(identity: dict) -> str:
    return (
        "### 实验身份\n\n"
        f"- 代码：`{identity['commit'] or '未知'}` · 源码 sha `{identity['src_sha'] or '未记录'}`"
        f"（{'工作区有未提交改动' if identity['dirty'] else '工作区干净'}）\n"
        f"- 实验脚本 `scripts/term_ablation.py` sha `{identity['script_sha']}`\n"
        f"- 词表：`{identity['term_map']}` sha `{identity['term_map_sha']}`\n"
        f"- 数据：`{identity['dataset']}` sha `{identity['dataset_sha']}`\n"
    )


def render(result: dict) -> str:
    lines: list[str] = []
    covered = summarise(result, group="covered")
    everything = summarise(result, group="all")

    if covered["n"]:
        lines.append(f"### A. 命中词表的题：四格完整（n={covered['n']}）\n")
        for depth, cells in covered["covered"].items():
            lines.append(f"**池深 {depth}**\n")
            lines.append("| 格 | 候选池 | 重排查询 | Recall@5 | MRR | 池内含 gold | 名次翻动 |")
            lines.append("|---|---|---|---|---|---|---|")
            for cell in CELLS:
                row = cells[cell]
                pool_label, query_label = CELL_LABELS[cell]
                lines.append(
                    f"| {cell} | {pool_label} | {query_label} | {_fmt(row['recall'])} | "
                    f"{_fmt(row['mrr'])} | {row['pool_has_gold']}/{row['n']} | "
                    f"{row['unstable']}/{row['n']} |"
                )
            base = cells["P0Q0"]["recall"]
            lines.append("")
            lines.append(
                "主效应（Recall@5）："
                f"术语池 @原话 {_signed(cells['P1Q0']['recall'] - base)} ／ "
                f"术语池 @append {_signed(cells['P1Q1']['recall'] - cells['P0Q1']['recall'])} ／ "
                f"重排查询 @原话池 {_signed(cells['P0Q1']['recall'] - base)} ／ "
                f"重排查询 @术语池 "
                f"{_signed(cells['P1Q1']['recall'] - cells['P1Q0']['recall'])}"
            )
            interaction = (cells["P1Q1"]["recall"] - cells["P1Q0"]["recall"]) - (
                cells["P0Q1"]["recall"] - base
            )
            lines.append(f"交互项：{_signed(interaction)}")
            lines.append("")

    # 未覆盖词表的题只有 P0Q0 有定义：它们没命中词表，就没有术语池、也没有「原话+术语串」
    # 这个查询，生产形态在它们身上**退化成基线**而不是"跑出 0"。所以这里不并排比较
    # 「全量基线 vs 全量生产」——那需要给一半的题补一个不存在的结果。改成把全量基线与
    # 覆盖组的两格并排放，每列都是真跑出来的数。
    if everything["n"]:
        lines.append(f"### B. 全部题：全量基线 + 覆盖组（n={everything['n']}）\n")
        lines.append(
            f"命中词表 {covered['n']} 题（{covered['n'] / everything['n']:.0%}）。"
            "未命中的题四格退化成同一格，不参与消融。\n"
        )
        lines.append("| 池深 | 全量基线 P0Q0 | 覆盖题基线 P0Q0 | 覆盖题生产 P1Q1 | 覆盖组差值 |")
        lines.append("|---|---|---|---|---|")
        for depth, cells in everything["all"].items():
            base = cells["P0Q0"]
            cov = covered["covered"].get(depth) if covered["n"] else None
            if not cov:
                lines.append(f"| {depth} | {_fmt(base['recall'])} | — | — | — |")
                continue
            cov_base = cov["P0Q0"]["recall"]
            cov_prod = cov["P1Q1"]["recall"]
            lines.append(
                f"| {depth} | {_fmt(base['recall'])} | {_fmt(cov_base)} | {_fmt(cov_prod)} | "
                f"{_signed(cov_prod - cov_base)} |"
            )
        lines.append("")
    return "\n".join(lines)


def _signed(value: float) -> str:
    return f"{value:+.3f}"


def render_per_question(result: dict, depth: int) -> str:
    """逐题四格。覆盖题才列出——未覆盖题四格按定义相同，列出来只是噪音。"""
    top_k = result["top_k"]
    records = [r for r in result["records"] if r["covered"]]
    if not records:
        return ""
    lines = [
        "| 题 | 术语串 | P0Q0 | P0Q1 | P1Q0 | P1Q1 | 重复一致 |",
        "|---|---|---|---|---|---|---|",
    ]

    def cell(rec: dict, name: str) -> str:
        runs = rec["cells"].get(name, {}).get(depth)
        if runs is None:
            return "—"
        return "✓" if _hit(runs, top_k) else ("·" if all(r is None for r in runs) else "✗")

    for rec in records:
        stable = all(
            len({r for r in rec["cells"].get(c, {}).get(depth, [])}) <= 1 for c in CELLS
        )
        lines.append(
            f"| {rec['question'][:26]} | {rec['rewrite'][:22]} | "
            + " | ".join(cell(rec, c) for c in CELLS)
            + f" | {'是' if stable else '否'} |"
        )
    return "\n".join(lines)


def _reranker_problem(services) -> str | None:
    """没有重排服务时，Q0 与 Q1 两格会逐位相同。

    "重排查询毫无作用"于是成了被造出来的结论而不是测出来的——宁可拒绝跑。
    """
    if services.reranker.__class__.__name__ == "NoopReranker":
        return "未配置重排服务（NoopReranker），四格里的 Q 因子不可测"
    return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=ROOT / "eval/dataset.paraphrase.jsonl")
    ap.add_argument(
        "--term-map",
        type=Path,
        default=ROOT / "rag4prefa/terms/public_to_standard.v3.json",
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--depths", default="25,50", help="池深，逗号分隔；25 是生产值")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（调试用）")
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    services = build_services()
    problem = _reranker_problem(services)
    if problem:
        print(problem, file=sys.stderr)
        return 2

    ev = _eval_module()
    term_map = load_term_map(args.term_map)
    rows = ev.load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
    depths = tuple(sorted({int(d) for d in args.depths.split(",") if d.strip()}))
    result = await measure(
        services, ev, rows, term_map,
        top_k=args.top_k, depths=depths, repeat=args.repeat, progress=True,
    )

    body = render(result)
    print(body)
    for depth in depths:
        table = render_per_question(result, depth)
        if table:
            print(f"\n### 逐题（池深 {depth}）\n")
            print(table)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "# 术语词表 2×2 固定池消融",
            "",
            f"数据：`{args.dataset.name}`　词表：`{args.term_map.name}`　"
            f"top-k={args.top_k}　池深={','.join(map(str, depths))}　重复={args.repeat}",
            "",
            render_identity(experiment_identity(ev, args.dataset, args.term_map)),
            "",
        ]
        args.save.write_text("\n".join(header) + body + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
