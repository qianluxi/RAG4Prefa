"""从语料里分层抽样，产出「独立题集」的候选条文清单。

为什么要这个脚本：现有 50 题里 35 题是"用检索器找答案"时出的，31 题的 gold 在出题
当时就已经出现在召回候选里——**可见性偏差**让那批题天然偏易。检验词表有没有泛化，
需要一批"出题时没看过检索结果"的题。

所以流程反过来：先按固定种子从库里抽条文，抽到哪条就出哪条，再读正文写问句。
抽样只看**文档清单与正文**，不看任何召回结果。种子与分层规则都写死在这里，
别人可以用同一条命令复现出同一批候选（顺序、id 全部相同）。

分层按文档性质，不按"好不好搜"：

- `standard` 标准/规范/规程/导则/图集/审查要点
- `policy`   政策文件、实施细则、工作方案、公报、建议答复、规划
- `paper`    研究论文（`P00_`~`P18_` 两位数字编号的那批）
- `news`     项目报道、政府网消息、日报稿

每条候选要求正文 ≥ `--min-chars` 字：短片段（封面、目录、上一篇/下一篇）出不了题，
但**剔除只按长度这一条先验规则**，与检索结果无关，剔除数量也会打印出来。

用法：
    python scripts/sample_independent.py --seed 20261029
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 50 题现有的标注文件。抽样要避开它们已经用过的片段：新题集的价值在于"没被看过"，
# 与旧题撞同一段就退化成复测。
OLD_DATASETS = (
    "eval/dataset.jsonl",
    "eval/dataset.paraphrase.jsonl",
)

PAPER_RE = re.compile(r"^P\d{2}_")
NEWS_RE = re.compile(r"^(P\d+_|Z\d_|A\d+_|D\d)")

# 出不了题的片段，全部是**看文本形态就能判**的规则，与检索结果无关：
# 图注（"1—门垛尺寸；2—中心线"）、参考文献、期刊排版页、目录/前言、导航空壳。
# 放宽这些门槛就会让"随机抽样"变成"人挑好出的"，偏差就回来了，所以规则写死在这里，
# 剔除条数照实打印。
_BAD_PATH = ("references", "reference", "参考文献", "前言", "目录", "索引", "序言")
_CAPTION_RE = re.compile(r"^\s*\d+\s*[—－-]\s*\S")
_DECOR = ("article in press", "downloaded from", "上一篇", "下一篇", "扫一扫")


def _chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    han = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return han / len(text)


def _classify(filename: str) -> str:
    """先判报道，再判标准。反过来的话"绍兴市家装厨房换新补助标准_绍兴日报"会被算成
    标准文件——名字里有"标准"，实际是一篇新闻稿。
    """
    # `P10_嘉兴市第一批...清单` 这类文件名以 P + 两位数字开头，实际是报道清单，
    # 不是论文。名单类关键词优先于编号。
    if PAPER_RE.match(filename) and not any(
        k in filename for k in ("清单", "名单", "详情", "报道", "观摩", "试点")
    ):
        return "paper"
    if NEWS_RE.match(filename) and (
        "日报" in filename or "网" in filename or "报道" in filename
    ):
        return "news"
    if any(
        k in filename
        for k in (
            "GB", "JGJ", "DBJ", "DB33",
            "标准", "规范", "规程", "导则", "图集", "要点", "审查", "设计示例",
        )
    ):
        return "standard"
    if any(
        k in filename
        for k in (
            "浙建管发", "浙政发", "实施细则", "实施方案", "工作方案", "公报", "答复",
            "规划", "行动方案", "举措", "通知", "定额", "计价", "价格",
        )
    ):
        return "policy"
    if NEWS_RE.match(filename):
        return "news"
    return "other"


def _usable(filename: str, text: str, min_chars: int) -> str | None:
    """返回剔除理由（None = 可用）。

    这里剔除的都是**非叙述性文本**：没有能被"问一句话"回答的内容。
    判据只看文本自身，不看它能不能被检索到——后者一旦进来，抽样就变成了挑题。
    """
    if len(text) < min_chars:
        return "too_short"
    lowered = text.lower()
    if any(word in lowered for word in _DECOR):
        return "decor"
    if _CAPTION_RE.match(text):
        return "caption"
    if _chinese_ratio(text) < 0.30:
        return "not_chinese"
    # 前言/编制说明：只有单位与起草人名单。
    if "编制组" in text or "主要起草人" in text:
        return "preface"
    # 公式符号说明：整段是 "$f_i$ —— …" 之类的符号表。
    if text.count("$") >= 6:
        return "formula_symbols"
    # 图集的图注与图签栏：拆开后只剩尺寸数字与"审核/制图"。
    if "图集号" in text or "制 图" in text or "审 核" in text:
        return "drawing"
    return None


def _old_chunks() -> set[str]:
    used: set[str] = set()
    for rel in OLD_DATASETS:
        path = ROOT / rel
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            used.update(str(x) for x in row.get("gold_chunk_ids") or [])
    return used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data/app.db"))
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--min-chars", type=int, default=120)
    ap.add_argument(
        "--per-stratum", default="standard:12,policy:12,news:8",
        help="每层抽几份文档（每份文档抽 1 条条文）",
    )
    ap.add_argument("--out", default=str(ROOT / "eval/independent-candidates.json"))
    args = ap.parse_args()

    quotas = {}
    for item in args.per_stratum.split(","):
        name, count = item.split(":")
        quotas[name] = int(count)

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    used = _old_chunks()

    # 每份文档的可用条文：够长、不是旧题用过的、不是图注/参考文献/导航空壳。
    by_doc: dict[tuple[str, str], list[sqlite3.Row]] = {}
    dropped: dict[str, int] = {}
    for row in conn.execute(
        "select c.id, c.doc_id, c.text, c.chapter_path, c.title, d.filename, d.doc_year "
        "from chunks c join documents d on d.id = c.doc_id"
    ):
        if row["id"] in used:
            dropped["old_chunk"] = dropped.get("old_chunk", 0) + 1
            continue
        path = (row["chapter_path"] or "").lower()
        if any(word in path for word in _BAD_PATH):
            dropped["bad_path"] = dropped.get("bad_path", 0) + 1
            continue
        reason = _usable(row["filename"], (row["text"] or "").strip(), args.min_chars)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        by_doc.setdefault((_classify(row["filename"]), row["filename"]), []).append(row)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    report: list[str] = []
    for stratum, quota in quotas.items():
        docs = sorted(name for (s, name) in by_doc if s == stratum)
        if len(docs) < quota:
            report.append(f"{stratum}：可用文档只有 {len(docs)} 份，配额 {quota} 调低")
        chosen_docs = rng.sample(docs, min(quota, len(docs)))
        for name in sorted(chosen_docs):
            rows = sorted(by_doc[(stratum, name)], key=lambda r: r["id"])
            row = rng.choice(rows)
            picked.append(
                {
                    "stratum": stratum,
                    "doc_id": row["doc_id"],
                    "filename": row["filename"],
                    "doc_year": row["doc_year"],
                    "chapter_path": row["chapter_path"],
                    "title": row["title"],
                    "chunk_id": row["id"],
                    "text": (row["text"] or "").strip(),
                }
            )
    detail = "、".join(f"{k}={v}" for k, v in sorted(dropped.items()))
    report.append(f"避开旧题片段 {len(used)} 个；剔除：{detail}")

    out = Path(args.out)
    out.write_text(json.dumps(picked, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n".join(report))
    print(f"抽出 {len(picked)} 条候选 → {out}")
    per = {s: sum(1 for p in picked if p["stratum"] == s) for s in quotas}
    print("分层计数：" + "、".join(f"{k}={v}" for k, v in per.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
