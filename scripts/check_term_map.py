"""校验词表是否真的"从语料里长出来"，以及它在某个数据集上的覆盖率。

两类检查，各有各的用处：

1. **准入检查**（不需要数据集）：
   - 每个术语必须**逐字**出现在语料正文里。这一条挡的是"模型意译的近义词"——
     而近义词正是第四十一轮被证伪的那一路（LLM 改写 15 题跑两次，0 题相同）。
   - 公众表达侧（`groups` 的子串）应该**几乎不在条文里出现**。如果一个"日常说法"
     在语料里命中几十次，那它本身就是条文用语，不需要映射——写进去只会误命中。
   - `source` 指向的文档必须真的在库里。

2. **覆盖率**（给 `--dataset` 时）：逐条统计命中几题、哪些题被多条同时命中。
   多条同时命中会拼成一条重排查询，串越长越模糊，所以要看得见。

用法：
    python scripts/check_term_map.py rag4prefa/terms/public_to_standard.v2.json
    python scripts/check_term_map.py <词表> --dataset eval/dataset.paraphrase.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa.adapters.term_map import load_term_map  # noqa: E402

DB = ROOT / "data" / "app.db"

# 一个子串在 2560 条片段里出现超过这个次数，就认为它是"条文也这么说"的词。
# 只对 3 字以上的子串生效：单字（修/换/漏）与双字（管线/验收）在任何中文语料里
# 都高频，用频次判它们是不是"日常说法"没有意义，只会把整张表都标红。
COMMON = 30
MIN_LEN_FOR_FREQ = 3


def _corpus(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    files = {r["id"]: r["filename"] for r in conn.execute("select id, filename from documents")}
    return [
        (files.get(r["doc_id"], ""), re.sub(r"\s+", "", r["text"]))
        for r in conn.execute("select doc_id, text from chunks")
    ]


def _count(docs: list[tuple[str, str]], needle: str) -> tuple[int, list[str]]:
    hits = 0
    names: list[str] = []
    for name, text in docs:
        n = text.count(needle)
        if n:
            hits += n
            names.append(name)
    return hits, names


def check(path: Path) -> bool:
    tm = load_term_map(path)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    docs = _corpus(conn)
    ok = True

    print(f"词表 {path.name}：{len(tm)} 条，语料 {len(docs)} 条片段\n")
    print(f"{'id':<22}{'术语命中':>8}  {'文档':>4}  {'来源在库':>6}  判定")
    for entry in tm.entries:
        per_term = [(t, *_count(docs, t)) for t in entry.terms]
        empty = [t for t, n, _ in per_term if n == 0]
        total = sum(n for _, n, _ in per_term)
        doc_names = sorted({d for _, _, ds in per_term for d in ds})
        # source 指向的文档要在库里——判的是"这个出处存不存在"，
        # 不是"术语命中过它"。混成一件事会把「出处正确但术语恰好在别的文档里」判成错。
        src_doc = entry.source.split(" ")[0].split("「")[0].strip()
        all_names = {name for name, _ in docs}
        src_ok = not src_doc or any(src_doc in name for name in all_names)
        verdict = "OK"
        if empty:
            verdict = f"术语不在语料：{empty}"
            ok = False
        elif not src_ok:
            verdict = f"source 未指向命中文档（写成 {src_doc!r}）"
            ok = False
        print(f"{entry.id:<22}{total:>8}  {len(doc_names):>4}  {str(src_ok):>6}  {verdict}")

    print(f"\n共有词（{MIN_LEN_FOR_FREQ} 字以上、条文里 ≥{COMMON} 次）：")
    shared = []
    for entry in tm.entries:
        for group in entry.groups:
            for token in group:
                if len(token) < MIN_LEN_FOR_FREQ:
                    continue
                n, _ = _count(docs, token)
                if n >= COMMON:
                    shared.append((entry.id, token, n))
    if not shared:
        print("  无（所有日常说法都是单侧词）")
    else:
        # 不判错：`卫生间`、`吊顶板` 这类词双方都说，用它当匹配键无害。
        # 真正要担心的是「词条太宽、在无关问题上也命中」，那件事用下面的误命中率衡量。
        for eid, token, n in sorted(shared, key=lambda x: -x[2]):
            print(f"  {eid:<22}{token:<12}{n:>5} 次")
    return ok


def coverage(path: Path, dataset: Path) -> None:
    tm = load_term_map(path)
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]
    print(f"\n覆盖率（{dataset.name}，{len(rows)} 题）：")
    per_entry: dict[str, list[str]] = {e.id: [] for e in tm.entries}
    multi = []
    covered = 0
    for row in rows:
        q = row["question"]
        hits = tm.hits(q)
        if hits:
            covered += 1
            for e in hits:
                per_entry[e.id].append(q)
            if len(hits) > 1:
                multi.append((q, [e.id for e in hits]))
    print(f"  命中词表 {covered}/{len(rows)} = {covered / len(rows):.3f}")
    print(f"  被多条同时命中 {len(multi)} 题"
          + ("（串会变长，重排查询会被稀释）" if multi else ""))
    for q, ids in multi[:8]:
        print(f"    {q[:30]}  ← {ids}")
    print("\n  逐条命中：")
    for eid, qs in sorted(per_entry.items(), key=lambda kv: -len(kv[1])):
        mark = "  ← 死重" if not qs else ""
        print(f"    {eid:<22}{len(qs):>3} 题{mark}")


def main() -> None:
    ap = argparse.ArgumentParser(description="校验受控词表与语料的一致性")
    ap.add_argument("term_map", type=Path)
    ap.add_argument("--dataset", type=Path, default=None, help="顺带算覆盖率")
    args = ap.parse_args()
    ok = check(args.term_map)
    if args.dataset:
        coverage(args.term_map, args.dataset)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
