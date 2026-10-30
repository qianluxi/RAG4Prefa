"""把判官的"误拒"拆成两种性质完全不同的东西。

判官判的是「**这 k 条片段**能不能回答」，而题集标的是「**库里**有没有答案」。两者不是
一回事，所以"判官误拒 16 题"这句话本身没有信息量——必须再问一句：那 16 题里，答案在
不在它看到的那 5 条里？

  - **gold 在 top-k 内却判不可** → 判官读走了眼，是它自己的错。这类才叫误判。
  - **gold 不在 top-k 内却判不可** → 判官说的是一句真话：这批片段确实答不了。
    产品上仍表现为"库里明明有却说没有"，但责任在检索层，不在判官。

不拆开这两类，就会得出"判官不能用"或"判官可用"其中一个错误结论。拆开之后，
结论变成"判官在多严格的口径上工作"，而这是可以调的（提示词、给多少片段）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa.domain import Filter  # noqa: E402


def _eval_module():
    spec = importlib.util.spec_from_file_location("ev_diag", ROOT / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="eval/reports/judge-v1-k8-20261030.json")
    ap.add_argument("--cache", default="eval/reports/judge-hits-k8.json")
    ap.add_argument("--positive", nargs="*", default=[
        "eval/dataset.jsonl", "eval/dataset.paraphrase.jsonl",
        "eval/dataset.independent.jsonl", "eval/dataset.independent-b.jsonl",
    ])
    ap.add_argument("--db", default=str(ROOT / "data" / "app.db"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    ev = _eval_module()
    saved = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    # 缓存格式 v2 之后 key 是"问题+Filter"的哈希，条目里才带问句。这里统一拆成
    # 问句 → 候选列表，脚本其余部分不必知道 key 长什么样。
    cache = {}
    for entry in json.loads(Path(args.cache).read_text(encoding="utf-8"))["hits"].values():
        if isinstance(entry, dict) and "question" in entry:
            cache.setdefault(entry["question"], entry.get("hits", []))
    row_by_q = {}
    for path in args.positive:
        for row in load_rows(Path(path)):
            row_by_q.setdefault(str(row.get("question") or "").strip(), row)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    seen = set()  # 同一句话在四份题集里只出现一次（去掉了跨题集重复，与主脚本一致）
    buckets: dict[str, list[dict]] = {"gold_in_topk": [], "gold_out": [], "no_gold": []}
    for r in saved["results"]:
        if r["kind"] != "positive":
            continue
        q = r["question"]
        if q in seen:
            continue
        seen.add(q)
        row = row_by_q.get(q)
        if row is None:
            continue
        filt = Filter(**row["filter"]) if row.get("filter") else Filter()
        gold, _ = ev.resolve_gold(conn, row, filt)
        tops = cache.get(q, [])[: args.k]
        rank = next((i for i, h in enumerate(tops, start=1) if h["chunk_id"] in gold), None)
        item = {
            "question": q,
            "category": r["category"],
            "judge_can_answer": r["judge_can_answer"],
            "reason": (r["judge_reason"] or "").replace("\n", " "),
            "gold_rank": rank,
            "n_gold": len(gold),
            "top1": tops[0]["filename"] if tops else "",
        }
        if not gold:
            buckets["no_gold"].append(item)
        elif r["judge_can_answer"] is False:
            buckets["gold_in_topk" if rank else "gold_out"].append(item)

    out = [
        "# 判官误拒拆解（2026-10-30）",
        "",
        "| 分栏 | 题数 | 含义 |",
        "|---|---:|---|",
        f"| 判官判不可，且 gold 在 top-{args.k} 内 | {len(buckets['gold_in_topk'])} | "
        "判官读走了眼，责任在它 |",
        f"| 判官判不可，但 gold 不在 top-{args.k} 内 | {len(buckets['gold_out'])} | "
        "它说的是真话：这批片段确实答不了，责任在检索 |",
        f"| 题集本身没有可解析的 gold | {len(buckets['no_gold'])} | 不参与判定 |",
        "",
    ]
    for name, title in (
        ("gold_in_topk", "一、判官真误判（gold 就在它看到的片段里）"),
        ("gold_out", "二、片段里确实没有答案（检索没送上来）"),
    ):
        out += [f"## {title}", ""]
        if not buckets[name]:
            out += ["（无）", ""]
            continue
        out += ["| 题集 | 问题 | gold 名次 | 判官理由 |", "|---|---|---|---|"]
        for it in buckets[name]:
            out.append(
                f"| `{it['category']}` | {it['question']} | {it['gold_rank'] or '—'} | "
                f"{it['reason'][:44]} |"
            )
        out.append("")
    text = "\n".join(out)
    print(text)
    if args.save:
        Path(args.save).write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
