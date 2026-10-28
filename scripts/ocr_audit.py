"""识别稿体检：把"哪些页是视觉模型认的、认得怎么样"量出来。

为什么需要它：`documents.ocr_pages` 只记录**范围**（哪几页），不记录**质量**。
而视觉模型是会漏字的——实测 JGJ/T 491 的 3.1.10 少了「应采**用**绿色施工模式」
里的"用"字。这类错误在检索结果里和原文长得一模一样，唯一能自动发现它的线索是
文档自身的一致性：

1. **条文号连续性**。同一节里 `3.1.1 / 3.1.2 / 3.1.5`——缺 3.1.3 和 3.1.4 说明
   那一段（很可能在某一页的中间）没被识别出来。规范类文档的编号是人工编排的，
   极少本来就跳号，所以这是**最灵敏**的自动判据。
2. **表号、图号连续性**。同上，但更容易漏：表 4.3.10 之后直接跳到 4.3.12，
   多半是中间那张表整页丢了。
3. **占位符 ▢**。提示词要求"认不出的字用 ▢ 占位，不要猜"。所以 ▢ 的数量就是
   模型自认看不清的字数——它标记的位置，人眼一定要过一遍。
4. **片段字数塌陷**。同一份文档里，某个片段只有几十个字而中位数是四五百，
   通常意味着那几页的识别中途断了。

只读：不调模型、不写库、不改文件。可以对着一个已经在跑的库反复跑。

用法：
    python scripts/ocr_audit.py                          # 只看有识别页的文档
    python scripts/ocr_audit.py --md eval/ocr-audit.md   # 顺便写一份报告
    python scripts/ocr_audit.py --doc 11_JGJT491-2021_装配式内装修技术标准.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa.db import connect  # noqa: E402

# `3.1.11` / `4.3.10`：规范条文号。前后不能再接数字，否则 `2018.10.1` 这种日期会混进来。
_CLAUSE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{1,3})(?![\d.])")
# `表 4.3.10` / `图 3.0.1` / `表4.3.10`
_TABLE = re.compile(r"[表图]\s*(\d{1,2})\.(\d{1,2})\.(\d{1,3})(?![\d.])")
# 提示词约定：认不出的字用 ▢ 占位（不是"猜一个"）。数量本身就是质量信号。
_PLACEHOLDER = "▢"


@dataclass
class DocAudit:
    filename: str
    ocr_pages: int
    chunks: int
    chars: int
    placeholder: int = 0
    clause_gaps: list[str] = field(default_factory=list)
    table_gaps: list[str] = field(default_factory=list)
    slim_chunks: int = 0

    @property
    def chars_per_chunk(self) -> int:
        return round(self.chars / self.chunks) if self.chunks else 0

    @property
    def problems(self) -> list[str]:
        out: list[str] = []
        if self.clause_gaps:
            out.append(f"条文号断档 {len(self.clause_gaps)} 处")
        if self.table_gaps:
            out.append(f"表/图号断档 {len(self.table_gaps)} 处")
        if self.placeholder:
            out.append(f"{self.placeholder} 个认不出的字")
        if self.slim_chunks:
            out.append(f"{self.slim_chunks} 个片段异常短")
        return out


def _gaps(triples: list[tuple[int, int, int]], label: str) -> list[str]:
    """同一节（前两段相同）里第三段不连续的地方。"""
    by_section: dict[tuple[int, int], set[int]] = {}
    for a, b, c in triples:
        by_section.setdefault((a, b), set()).add(c)
    gaps: list[str] = []
    for (a, b), used in sorted(by_section.items()):
        # 只查"本来就有序列"的节：单条编号无从判断连续性
        if len(used) < 2:
            continue
        missing = sorted(set(range(min(used), max(used) + 1)) - used)
        if missing:
            head = "、".join(f"{a}.{b}.{m}" for m in missing[:6])
            more = f" 等 {len(missing)} 个" if len(missing) > 6 else ""
            gaps.append(f"{label} {a}.{b} 缺 {head}{more}")
    return gaps


def audit_doc(conn: sqlite3.Connection, row: sqlite3.Row) -> DocAudit:
    # **必须把 title 与 chapter_path 一起算进"这份文档有多少编号"。**
    # 第一版只扫 `chunks.text`，于是所有被识别成标题的条文号（`4.3.9 龙骨隔墙应符合
    # 下列规定：` 就落在 title 里）全都报成"断档"——实测 4 份文档报了 23 处，逐条核对
    # 后绝大部分是误报。判据取错对象，比没有判据更浪费时间。
    rows = conn.execute(
        "SELECT text, title, chapter_path FROM chunks WHERE doc_id = ? ORDER BY ordinal",
        (row["id"],),
    ).fetchall()
    texts = [r["text"] for r in rows]
    numbers = "\n".join(
        "\n".join([r["title"], *json.loads(r["chapter_path"] or "[]"), r["text"]]) for r in rows
    )
    chars = [len(t) for t in texts]
    median = statistics.median(chars) if chars else 0
    return DocAudit(
        filename=row["filename"],
        ocr_pages=len(json.loads(row["ocr_pages"])),
        chunks=len(texts),
        chars=sum(chars),
        placeholder=numbers.count(_PLACEHOLDER),
        clause_gaps=_gaps(
            [(int(a), int(b), int(c)) for a, b, c in _CLAUSE.findall(numbers)], "条文"
        ),
        table_gaps=_gaps(
            [(int(a), int(b), int(c)) for a, b, c in _TABLE.findall(numbers)], "表/图"
        ),
        # 中位数的一半：低于它的片段要么是短标题，要么是"识别到一半断了"。
        # 阈值故意放宽——短章节本来就存在，这里只抓明显塌陷的那种。
        slim_chunks=sum(1 for n in chars if n and n < max(80, median / 2)),
    )


def report(rows: list[DocAudit], *, db_path: Path) -> str:
    lines = [
        "# 识别稿体检",
        "",
        f"- 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 数据库：`{db_path}`",
        f"- 有识别页的文档：{len(rows)} 份",
        "",
        "| 文档 | 识别页 | 片段 | 字/片段 | 认不出 | 条文断档 | 表图断档 | 异常短片段 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.filename[:40]} | {r.ocr_pages} | {r.chunks} | {r.chars_per_chunk} "
            f"| {r.placeholder} | {len(r.clause_gaps)} | {len(r.table_gaps)} | {r.slim_chunks} |"
        )

    details = [r for r in rows if r.clause_gaps or r.table_gaps or r.placeholder]
    lines += ["", "## 需要人眼过的位置", ""]
    if not details:
        lines.append("没有发现断档或占位符。")
    for r in details:
        lines.append(f"### {r.filename}")
        for gap in r.clause_gaps + r.table_gaps:
            lines.append(f"- {gap}")
        if r.placeholder:
            lines.append(f"- 模型自认看不清的字 {r.placeholder} 个（正文里的 ▢）")
        lines.append("")

    lines += [
        "> 断档不等于一定漏了：规范本来就允许「本章不设某节」，也会出现整节作废。",
        "> 判据的价值在于**缩小人眼要看的范围**——把 500 页变成十几处，",
        "> 而不是自动给出「这份识别对了」的结论。占位符 ▢ 的位置则是确定的漏字。",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="识别稿体检")
    parser.add_argument("--db", default="data/app.db")
    parser.add_argument("--md", help="把报告写到这个路径")
    parser.add_argument("--doc", help="只看这一份（文件名）")
    parser.add_argument("--quiet", action="store_true", help="不打印报告")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    if not db_path.is_file():
        print(f"找不到数据库：{db_path}", file=sys.stderr)
        return 2
    conn = connect(db_path)
    try:
        sql = "SELECT id, filename, ocr_pages FROM documents WHERE ocr_pages IS NOT NULL"
        params: tuple[str, ...] = ()
        if args.doc:
            sql += " AND filename = ?"
            params = (args.doc,)
        rows = [
            r
            for r in conn.execute(sql + " ORDER BY filename", params)
            # 空数组 = 这份文档明确没有识别页，不必进报告
            if json.loads(r["ocr_pages"])
        ]
        if not rows:
            print("库里没有任何经过识别的文档（ocr_pages 全为空或 NULL）。")
            return 0
        audits = [audit_doc(conn, r) for r in rows]
    finally:
        conn.close()

    text = report(audits, db_path=db_path)
    if args.md:
        out = Path(args.md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"已写入 {out}")
    if not args.quiet:
        print(text)

    bad = [a for a in audits if a.clause_gaps or a.table_gaps or a.placeholder]
    print(f"\n{len(audits)} 份识别稿，{len(bad)} 份有具体线索（断档 / 占位符）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
