"""误触发审计：词表在同一批问题上"多注入了什么"、"漏掉了什么"。

为什么单独一个脚本而不是塞进 `check_term_map.py`：那个脚本管的是**准入**
（术语能不能在语料里逐字查到、覆盖率多少），是写词表时的闸门；这个脚本管的是
**运行后果**（在专业问法集上乱触发了几题、同题撞了几条不同意图、单字规则挡掉了
几题），是改匹配层或分组之后必须重新跑的账。两者的失败含义不同，别混在一起看。

四块输出：

1. **负例闸门**——`eval/term-negatives.jsonl` 里 `forbid` 的词条一律不许命中。
   违反即退出码 1，可以直接挂 CI。
2. **触发率**——每个数据集上有多少题被注入、按 `intent` 分组。
   专业问法集（`eval/dataset.jsonl`）的触发率是这里最要紧的数：那批题的问句里
   本来就有规范用语，被口径词表碰一下基本都是噪音。
3. **多意图冲突**——同一题撞上两条**不同** `intent` 的映射，术语会一起拼进重排查询，
   串越长越模糊。这类题逐条列出来。
4. **收紧代价**——`strict=False`（单字也当证据）与默认口径的差集：
   单字规则到底挡掉了哪些命中。挡掉的是噪音就继续收紧，挡掉的是真命中就得
   把那个语境写成多字备选（`裂开` / `做坡`），别让引擎猜。

用法：
    python scripts/audit_term_misfire.py rag4prefa/terms/public_to_standard.v3.json
    python scripts/audit_term_misfire.py <词表> --dataset eval/dataset.jsonl \
        --save eval/reports/x.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa.adapters.term_map import TermMap, load_term_map  # noqa: E402

DEFAULT_DATASETS = (
    "eval/dataset.paraphrase.jsonl",
    "eval/dataset.jsonl",
    "eval/dataset.refuse.jsonl",
)
DEFAULT_NEGATIVES = "eval/term-negatives.jsonl"

# 触发率单独看没有意义，要跟"这批题本来就该不该被碰"一起看。按文件名认，
# 免得绝对路径 / 相对路径的口径差异让整列变成空白。
DATASET_ROLE = {
    "dataset.paraphrase.jsonl": "口语集（该碰）",
    "dataset.jsonl": "专业问法集（基本不该碰）",
    "dataset.refuse.jsonl": "库外集（只管有没有加重误导）",
}


def _rows(path: Path) -> list[dict]:
    """读 jsonl，跳过空行与 `//` 注释（与 `scripts/eval.py` 的约定一致）。"""
    out: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return out


def _questions(path: Path) -> list[str]:
    return [str(row.get("question") or "").strip() for row in _rows(path)]


def check_negatives(tm: TermMap, path: Path) -> tuple[list[str], int]:
    """返回（违规说明, 用例数）。词条 id 打错也算违规——否则闸门会因为拼错而永远通过。"""
    problems: list[str] = []
    cases = _rows(path)
    known = {entry.id for entry in tm.entries}
    for index, case in enumerate(cases, start=1):
        question = str(case.get("question") or "").strip()
        forbid = [str(x) for x in (case.get("forbid") or [])]
        if not question or not forbid:
            problems.append(f"第 {index} 条缺 question 或 forbid")
            continue
        unknown = [x for x in forbid if x not in known]
        if unknown:
            problems.append(f"第 {index} 条的 forbid 里有词表不存在的 id：{unknown}")
            continue
        fired = {entry.id for entry in tm.hits(question)}
        bad = [x for x in forbid if x in fired]
        if bad:
            detail = "，".join(
                f"{entry.id}（{entry.explain(question)}）" for entry in tm.hits(question)
                if entry.id in bad
            )
            problems.append(
                f"[{case.get('kind', '?')}] {question} → 不该触发却触发了：{detail}"
            )
    return problems, len(cases)


def audit_dataset(
    tm: TermMap, path: Path
) -> dict[str, object]:
    questions = _questions(path)
    fired = [(q, tm.hits(q)) for q in questions if tm.match(q)]
    by_intent: Counter[str] = Counter()
    for _q, entries in fired:
        for intent in {entry.intent for entry in entries}:
            by_intent[intent] += 1
    conflicts = [
        (q, entries)
        for q, entries in fired
        if len({entry.intent for entry in entries}) > 1
    ]
    dropped = []
    for q in questions:
        strict_ids = {e.id for e in tm.hits(q)}
        relaxed_ids = {e.id for e in tm.hits(q, strict=False)}
        lost = sorted(relaxed_ids - strict_ids)
        if lost:
            dropped.append((q, lost))
    return {
        "path": str(path),
        "role": DATASET_ROLE.get(path.name, ""),
        "total": len(questions),
        "fired": fired,
        "by_intent": by_intent,
        "conflicts": conflicts,
        "dropped": dropped,
    }


def _render(tm: TermMap, stats: list[dict], problems: list[str], cases: int) -> str:
    lines: list[str] = []
    lines.append("# 词表误触发审计")
    lines.append("")
    lines.append(f"- 词表：`{len(tm)}` 条 · 意图分布 "
                 + "、".join(f"{k} {v}" for k, v in
                             sorted(Counter(e.intent for e in tm.entries).items())))
    lines.append(f"- 负例闸门：{cases} 条，违反 {len(problems)} 条")
    lines.append("")
    if problems:
        lines.append("## 负例违反（必须修）")
        lines.append("")
        for problem in problems:
            lines.append(f"- {problem}")
        lines.append("")
    lines.append("## 各数据集触发率")
    lines.append("")
    lines.append("| 数据集 | 角色 | 题数 | 触发 | 触发率 | 按意图（题） |")
    lines.append("|---|---|---|---|---|---|")
    for stat in stats:
        total = int(stat["total"])
        fired = len(stat["fired"])  # type: ignore[arg-type]
        rate = f"{fired / total:.0%}" if total else "-"
        intents = "、".join(f"{k} {v}" for k, v in stat["by_intent"].most_common()) or "-"
        name = Path(str(stat["path"])).name
        lines.append(
            f"| `{name}` | {stat['role']} | {total} | {fired} | {rate} | {intents} |"
        )
    lines.append("")

    for stat in stats:
        if not stat["fired"]:
            continue
        lines.append(f"## {Path(str(stat['path'])).name} 逐题")
        lines.append("")
        for question, entries in stat["fired"]:  # type: ignore[union-attr]
            tags = " + ".join(f"{e.id}({e.intent})←{e.explain(question)}" for e in entries)
            lines.append(f"- {question}")
            lines.append(f"  → {tags}")
        lines.append("")
        conflicts = stat["conflicts"]
        if conflicts:
            lines.append(f"### 多意图冲突：{len(conflicts)} 题")
            lines.append("")
            for question, entries in conflicts:  # type: ignore[union-attr]
                ids = " + ".join(f"{e.id}/{e.intent}" for e in entries)
                lines.append(f"- {question} → {ids}")
            lines.append("")

    dropped_all = [(stat, stat["dropped"]) for stat in stats if stat["dropped"]]
    lines.append("## 单字规则挡掉了什么（strict=False 的反事实）")
    lines.append("")
    if not dropped_all:
        lines.append("（无）")
    for stat, dropped in dropped_all:
        lines.append(f"**{Path(str(stat['path'])).name}**")
        for question, ids in dropped:  # type: ignore[union-attr]
            lines.append(f"- {question} → 挡掉 {ids}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="词表误触发审计")
    parser.add_argument("term_map", help="词表 json 路径")
    parser.add_argument(
        "--dataset", action="append", default=None,
        help=f"可重复；默认 {' / '.join(DEFAULT_DATASETS)}",
    )
    parser.add_argument("--negatives", default=DEFAULT_NEGATIVES, help="负例集 jsonl")
    parser.add_argument("--save", default="", help="把报告写到这个路径（.md）")
    args = parser.parse_args(argv)

    map_path = Path(args.term_map)
    tm = load_term_map(map_path if map_path.is_absolute() else ROOT / map_path)
    if tm is None:
        parser.error("词表路径为空")

    neg_path = Path(args.negatives)
    neg_path = neg_path if neg_path.is_absolute() else ROOT / neg_path
    problems, cases = (check_negatives(tm, neg_path) if neg_path.exists() else ([], 0))
    if not neg_path.exists():
        print(f"警告：负例集不存在（{neg_path}），闸门这一节会被跳过", file=sys.stderr)

    paths = [p if Path(p).is_absolute() else ROOT / p for p in (args.dataset or DEFAULT_DATASETS)]
    stats = [audit_dataset(tm, Path(p)) for p in paths if Path(p).exists()]

    report = _render(tm, stats, problems, cases)
    print(report)
    if args.save:
        out = Path(args.save)
        out = out if out.is_absolute() else ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"已写入 {out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
