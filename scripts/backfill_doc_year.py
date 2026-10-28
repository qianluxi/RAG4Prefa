"""给缺年份的文档补上 `doc_year`。

跑法：
    python scripts/backfill_doc_year.py            # 只看推断结果，不写库
    python scripts/backfill_doc_year.py --apply    # 写库（只改年份，不重算向量）

**为什么必须补。** `doc_year` 不是展示字段，是**过滤条件**：问题里提到年份时
（`ask.build_filter` 会解析出 `year_from=year_to=2025`），SQL 侧是
`d.doc_year >= 2025 AND d.doc_year <= 2025`——**NULL 一条都过不去**。于是"2025 年补贴
包含什么"这类问题对着一个装着 2025 年文件的库，召回结果可以是空的，而界面上一切正常。
2026-10-28 实测：52 份文档的 `doc_year` 全为 NULL，评测集里 2 道带年份的题金标被
判成"不在筛选范围内"剔除。

**为什么不改入库逻辑自动识别。** 入库时"不带 `--year`"现在的语义是**沿用**已有年份
（不清空），这是刻意的（`--force` 补字段时不该把年份抹掉）。自动识别会把这个语义
偷偷改成"按文件名猜"，猜错的后果是过滤条件错，比缺年份更难发现。所以推断放在这里、
一次跑完、结果可核对。

**推断顺序**：文件名里的四位年份（取**最后**一个，即发文/公示年）→ 正文前
`_BODY_SCAN` 字里第一次出现的「20XX年」或「20XX-XX-XX」→ 都没有就留 NULL 并明确报告。
留 NULL 是诚实的：宁可"不知道"，也不要为凑数写一个会让过滤出错的年份。

**改年份走的是入库那条路**（`ingest_file(doc_year=...)`），不是直接 UPDATE：
年份同时存在于 `documents`、`vectors` 归档与后端标量列三处，那条路会先置
`unsynced:<doc_id>`、三处对齐后才撤，中途失败也不会留下"过滤结果按年份漏一半"的库。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa import repo, snapshot  # noqa: E402
from rag4prefa.adapters.parsers import parse_document  # noqa: E402
from rag4prefa.config import Settings  # noqa: E402
from rag4prefa.ingest import ingest_file  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

_FILENAME_YEAR = re.compile(r"(20\d{2})")
# 只认**完整的发文日期**（2024 年4 月30 日）。不认"正文里第一个年份"：政策文本开头
# 常写目标年（「到2027年…」），实测这样会把 03 判成 2023（真值是 2024 年4 月30 日印发）、
# 把 05/06 判成 2027（那句是引用《…行动方案（2023—2027 年）》）。
_BODY_DATE = re.compile(r"(20\d{2})\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日")
# 抓下来的网页会把来源 URL 留在正文末尾（`来源：https://jst.zj.gov.cn/art/2024/4/30/…`），
# 其中的日期就是发文日期，而且比正文里的"到2024年底"硬得多。
_SOURCE_URL = re.compile(r"/(20\d{2})/\d{1,2}/\d{1,2}/")
_BODY_SCAN = 40000  # 发文日期常排在内文最后，扫到 4 万字足够覆盖最长的那份


def year_from_filename(filename: str) -> int | None:
    """文件名里的年份。取**最后**一个：`..._国办发2016-71号` 是发文年，
    `..._2021vs2025.md` 里最后那个才是这份文件讲的版本。"""
    hits = _FILENAME_YEAR.findall(filename)
    return int(hits[-1]) if hits else None


def year_from_body(path: Path) -> int | None:
    """正文里的第一个年份。解析不出来就返回 None，不猜。"""
    try:
        sections = parse_document(path)
    except Exception:  # 解析器拒收的文件不该在这一步把整批拦住
        return None
    text = ""
    for section in sections:
        text = f"{text}\n{section.text}" if text else section.text
        if len(text) > _BODY_SCAN:
            break
    head = text[:_BODY_SCAN]
    match = _BODY_DATE.search(head) or _SOURCE_URL.search(head)
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description="补 doc_year")
    parser.add_argument("--apply", action="store_true", help="写库（默认只看推断结果）")
    parser.add_argument("--corpus", default="../工程文档", help="语料目录")
    parser.add_argument(
        "--set",
        action="append",
        metavar="文件名=年份",
        help="人工指定（重复给多次）。用于推断不出的少数几份",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = Settings()
    given = Path(args.corpus)
    corpus_dir = given if given.is_absolute() else (root / given).resolve()

    overrides: dict[str, int] = {}
    for item in args.set or []:
        name, _, value = item.rpartition("=")
        if not name or not value.isdigit():
            parser.error(f"--set 要写成 文件名=年份，收到 {item!r}")
        overrides[name] = int(value)

    services = build_services(settings)
    conn = services.conn
    plans: list[tuple[str, str, int | None, str]] = []
    for project in repo.list_projects(conn):
        for doc in repo.list_documents(conn, project.name):
            if doc.doc_year is not None:
                continue
            if doc.filename in overrides:
                plans.append((project.name, doc.filename, overrides[doc.filename], "人工"))
                continue
            year = year_from_filename(doc.filename)
            source = "文件名"
            if year is None:
                path = snapshot.locate_corpus_file(corpus_dir, doc.filename)
                if path is None:
                    plans.append((project.name, doc.filename, None, "缺源文件"))
                    continue
                year = year_from_body(path)
                source = "发文日期"
            plans.append((project.name, doc.filename, year, source if year else "未知"))

    if not plans:
        print("所有文档都有年份，不需要补")
        return 0

    width = max(len(name) for _, name, _, _ in plans)
    for project, filename, year, source in plans:
        mark = str(year) if year else "  -  "
        print(f"{mark}  [{source:4s}] [{project}] {filename:<{width}}")
    unknown = [p for p in plans if p[2] is None]
    print(f"\n可补 {len(plans) - len(unknown)} 份，推断不出 {len(unknown)} 份")
    for _, filename, _, source in unknown:
        print(f"  - {source}：{filename}")

    if not args.apply:
        print("这是预览。确认无误后加 --apply 写库（只改年份，不重算向量）")
        return 0
    if unknown:
        # 留 NULL 不是遗漏，是"没有证据"。写一个猜的年份比留空更危险：过滤条件会
        # 悄悄收窄，问"2024 年的规定"时把一份 2023 的文件放进结果，且没有任何提示。
        print("\n推断不出的保持 NULL（带年份的问法召回不到它们，这是已知取舍）")

    failed = 0
    for project, filename, year, _ in plans:
        if year is None:
            continue
        path = snapshot.locate_corpus_file(corpus_dir, filename)
        if path is None:
            print(f"  跳过（缺源文件）：{filename}")
            failed += 1
            continue
        outcome = ingest_file(
            conn,
            services.store,
            services.require_embedder(),
            settings,
            path=path,
            project_name=project,
            doc_year=year,
        )
        flag = "OK" if outcome.status in {"indexed", "skipped"} else "失败"
        why = outcome.note or outcome.error or ""
        print(f"  {flag} {year} {filename[:50]} {why}")
        if flag == "失败":
            failed += 1

    print(f"\n完成：改写 {len(plans) - len(unknown) - failed} 份，失败 {failed} 份")
    print("接着跑：python -m rag4prefa verify 与 scripts/snapshot_corpus.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
