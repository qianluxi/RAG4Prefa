"""凭关键词在语料里找条文（给"先写问句再找答案"的题集定位 gold）。

刻意**不用向量检索、不用 BM25**：只用子串匹配把候选条文捞出来，再由人读正文确认。
理由是老 50 题里 31 题 gold 就是这么被"检索器当时给的候选"决定的，可见性偏差进了题集。
这里换成"我知道的规范术语 → 全文查"，问句仍是口语，两者不共享措辞。

用法：
    python scripts/find_gold_by_text.py --file eval/independent-b-questions.json
    python scripts/find_gold_by_text.py --file ... --only 3 --width 400
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data/app.db"))
    ap.add_argument("--file", required=True, help="[{question, keywords}, ...]")
    ap.add_argument("--only", type=int, nargs="*", default=None, help="只看这些序号（1 起）")
    ap.add_argument("--width", type=int, default=260, help="摘录字数")
    args = ap.parse_args()

    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    wanted = set(args.only or [])
    for i, item in enumerate(items, 1):
        if wanted and i not in wanted:
            continue
        print(f"\n===== [{i:02d}] {item['question']}")
        seen: set[str] = set()
        for kw in item["keywords"]:
            rows = conn.execute(
                "select c.id, c.text, d.filename from chunks c join documents d on d.id=c.doc_id "
                "where c.text like ? limit 8",
                (f"%{kw}%",),
            ).fetchall()
            fresh = [r for r in rows if r["id"] not in seen]
            print(f"  «{kw}» 命中 {len(rows)} 条，新 {len(fresh)} 条")
            for r in fresh[:3]:
                seen.add(r["id"])
                text = " ".join((r["text"] or "").split())
                print(f"    - {r['id']} | {r['filename'][:34]} | {text[: args.width]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
