"""重建 `eval/` 下的两份记录：语料清单（从库）与解析器金丝雀（从源文件）。

跑法：
    python scripts/build_manifest.py                       # 只写 eval/corpus.json
    python scripts/build_manifest.py --canary              # 顺带写 eval/parser-canary.json
    python scripts/build_manifest.py --db data/app.db --quiet

三份记录各答一个问题，别混：

| 文件 | 数据来源 | 回答什么 |
| --- | --- | --- |
| `eval/corpus.json` | 数据库 | 标注时那批语料是什么（文件名 + 分块参数） |
| `eval/corpus-snapshot.json` | 数据库 | 上次出报告时库里有什么（含全量 id） |
| `eval/parser-canary.json` | 源文件 | 当前解析器对这 7 份样本输出什么 |

读它们的地方：`corpus.json` 给 `relabel_eval.py` 重建临时库、给 `test_snapshot.py`
与快照交叉校验；`corpus-snapshot.json` 给 `eval.py` 判漂移、给 `test_eval_dataset.py`
离线判 gold 是否过期；`parser-canary.json` 只给 `test_eval_dataset.py` 断言解析器没变。

前两份都从数据库导出而不重新解析：补语料时旧文件会被删掉（2026-10-28 实测：旧清单里
6 份文件，本地一份都不剩），而"当时那批片段"只有库还认得出。

第三份反过来——**必须**重新解析，因为它要回答的正是"换个时间再解析，结果还是不是同一批"。
代价是全量 52 份要 106 秒，且其中 5 份当初是开着 OCR 入库的、纯离线解析本就复现不出来，
所以它只取 7 份文字层完好、形态各异、合计约 3 秒的样本。金丝雀报的是"解析器变了没有"，
不是"哪份文档变了"。

**什么时候要跑：** 语料换血或标注之后跑 `corpus.json`（`--canary` 一起更省事）；
有意改了解析判据之后跑 `--canary`，再跑 `relabel_eval.py --db ... --write` 重算 gold。

为什么按文件名索引会拒绝同名文件：清单用文件名当键，两个项目下的"计算书.pdf"只能存
一份，少记的那份等于没校验。与其悄悄丢一份，不如要求先改名（金丝雀同理）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa import repo, snapshot  # noqa: E402
from rag4prefa.adapters.parsers import parse_document  # noqa: E402
from rag4prefa.chunking import build_chunks  # noqa: E402
from rag4prefa.config import Settings  # noqa: E402
from rag4prefa.db import connect  # noqa: E402

DEFAULT_OUT = Path("eval") / "corpus.json"
DEFAULT_CANARY = Path("eval") / "parser-canary.json"

# 金丝雀样本：按"格式 × 结构"挑，不按"哪份最常被问到"。合计解析约 3 秒。
#   - X1：309 块的长文档、多级编号，分块阈值一动就变
#   - 01 / 29：长公文与短公文，目录页、编号标题两条判据都在里面
#   - A6：docx 一路（大纲级别 / 样式 / 编号三条来源）
#   - Z3 / 23：政策 md（`key: value` 模板头、`#` 标题剥离）
#   - 17：标准信息卡 md，短、几乎全是表格行
CANARY_DOCS = (
    "X1_浙江省装配式建筑施工质量安全监督要点_浙建管发2024-71号.pdf",
    "01_装配式内装修设计导则_浙建管发2024-72号.pdf",
    "29_2025年浙江省旧房装修和厨卫局部改造补贴实施细则.pdf",
    "A6_宁波市装配化装修试点工作实施方案_甬建发2023-53号.docx",
    "Z3_浙江省十五五住房和城乡建设规划_浙政办发_2026.md",
    "23_浙江省开展装配化装修试点工作方案_浙建管发2023-50号.md",
    "17_GBT44546-2024_建筑用装配式集成吊顶通用技术要求_标准信息卡.md",
)


def build(conn, *, chunk_min_tokens: int, chunk_max_tokens: int) -> dict:
    """从库导出清单。键是**原始文件名**，直接读表而不是套快照的扁平结果。

    快照为了"同名不互相覆盖"会把键改成 `名字 [项目]`；清单这种格式按文件名去拼
    `CORPUS_DIR / 名字`，套用那套键会得到一条指向不存在文件的死路径。所以这里自己
    遍历 `documents`，遇到同名就记下来报错。
    """
    grouped = repo.chunk_ids_by_doc(conn)
    documents: dict[str, dict] = {}
    clashes: list[str] = []
    for project in repo.list_projects(conn):
        for doc in repo.list_documents(conn, project.name):
            if doc.filename in documents:
                clashes.append(
                    f"{doc.filename}（{documents[doc.filename]['project']} / {project.name}）"
                )
                continue
            ids = grouped.get(doc.id, [])
            documents[doc.filename] = {
                "project": project.name,
                "chunks": len(ids),
                "ids_sha": snapshot.ids_digest(ids),
            }
    return {
        "chunk_min_tokens": chunk_min_tokens,
        "chunk_max_tokens": chunk_max_tokens,
        "documents": documents,
        "_clashes": clashes,
    }


def build_canary(
    conn,
    corpus_dir: Path,
    manifest: dict,
    *,
    chunk_min_tokens: int,
    chunk_max_tokens: int,
) -> dict:
    """重新解析样本，记下"此刻的解析器会产出什么"。

    记 `doc_id` 而不是让测试自己算：`doc_id` 是 `chunk_id` 的输入，两边算法不同
    （比如以后把项目名归一化）就会得到两批 id，而"两批 id 都自洽"这种错误最难看出来。
    """
    documents: dict[str, dict] = {}
    for filename in CANARY_DOCS:
        meta = manifest["documents"].get(filename)
        if meta is None:
            raise SystemExit(f"金丝雀样本不在库里：{filename}（先入库，或从名单里删掉）")
        doc = repo.get_document(conn, meta["project"], filename)
        if doc is None:  # pragma: no cover - 上面刚查过 manifest，正常到不了
            raise SystemExit(f"库里查不到文档记录：{filename}")
        path = snapshot.locate_corpus_file(corpus_dir, filename)
        if path is None:
            raise SystemExit(f"语料缺文件：{corpus_dir / filename}（或在子目录里找不到）")
        chunks = build_chunks(
            parse_document(path),
            doc.id,
            max_tokens=chunk_max_tokens,
            min_tokens=chunk_min_tokens,
        )
        ids = sorted(chunk.id for chunk in chunks)
        if not ids:
            raise SystemExit(f"样本一个片段都解析不出来：{filename}（判据是不是坏了）")
        documents[filename] = {
            "project": meta["project"],
            "doc_id": doc.id,
            "chunks": len(ids),
            "ids_sha": snapshot.ids_digest(ids),
        }
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chunk_min_tokens": chunk_min_tokens,
        "chunk_max_tokens": chunk_max_tokens,
        "documents": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="重建语料清单与解析器金丝雀")
    parser.add_argument("--db", default="", help="数据库路径（默认取配置里的 db_path）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="清单输出路径")
    parser.add_argument(
        "--canary",
        nargs="?",
        const=str(DEFAULT_CANARY),
        default="",
        help=f"重新解析样本并写金丝雀记录（默认 {DEFAULT_CANARY}）",
    )
    parser.add_argument("--corpus", default="../工程文档", help="语料目录（写金丝雀时用）")
    parser.add_argument("--quiet", action="store_true", help="只打印一行摘要")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    settings = Settings()
    db_path = Path(args.db) if args.db else Path(settings.db_path)
    if not db_path.is_file():
        print(f"数据库不存在：{db_path.resolve()}", file=sys.stderr)
        print("先把文档入库（或确认是不是在仓库根目录下运行）", file=sys.stderr)
        return 2

    conn = connect(db_path)
    try:
        manifest = build(
            conn,
            chunk_min_tokens=settings.chunk_min_tokens,
            chunk_max_tokens=settings.chunk_max_tokens,
        )
        clashes = manifest.pop("_clashes")
        if not manifest["documents"]:
            print(f"库里没有任何文档：{db_path.resolve()}", file=sys.stderr)
            print("空清单会让所有标注校验失去意义，拒绝写出", file=sys.stderr)
            return 2
        if clashes:
            print("有同名文件，清单格式表达不了（同名只能存一份）：", file=sys.stderr)
            for item in clashes:
                print(f"  - {item}", file=sys.stderr)
            print("先把其中一份改名，再重跑本脚本", file=sys.stderr)
            return 2

        empty = [
            name for name, meta in manifest["documents"].items() if not meta["chunks"]
        ]
        if empty:
            print(f"有 {len(empty)} 份文档一个片段都没有，不该记进清单：", file=sys.stderr)
            for name in empty[:5]:
                print(f"  - {name}", file=sys.stderr)
            print("先把这些文档从库里删掉或重新入库", file=sys.stderr)
            return 2

        canary = None
        if args.canary:
            given = Path(args.corpus)
            corpus_dir = given if given.is_absolute() else (root / given).resolve()
            canary = build_canary(
                conn,
                corpus_dir,
                manifest,
                chunk_min_tokens=settings.chunk_min_tokens,
                chunk_max_tokens=settings.chunk_max_tokens,
            )
    finally:
        conn.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    total = sum(meta["chunks"] for meta in manifest["documents"].values())

    if canary is not None:
        canary_path = Path(args.canary)
        canary_path.write_text(
            json.dumps(canary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        samples = sum(meta["chunks"] for meta in canary["documents"].values())
        if not args.quiet:
            print(
                f"金丝雀：{len(canary['documents'])} 份样本 · {samples} 个片段 → "
                f"{canary_path}"
            )

    if args.quiet:
        print(f"{len(manifest['documents'])} 份 / {total} 片段 → {out}")
        return 0
    print(f"数据库：{db_path.resolve()}")
    print(
        f"清单：{len(manifest['documents'])} 份文档 · {total} 个片段 · "
        f"chunk {manifest['chunk_min_tokens']}~{manifest['chunk_max_tokens']} token"
    )
    print(f"\n已写入 {out}")
    print("接着跑：pytest tests/test_eval_dataset.py tests/test_snapshot.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
