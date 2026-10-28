"""把当前知识库导出为语料快照（`eval/corpus-snapshot.json`）。

跑法：
    python scripts/snapshot_corpus.py
    python scripts/snapshot_corpus.py --out eval/corpus-snapshot.json

**什么时候要跑：** 每次入库/删除文档之后，以及每次出评测报告之前。
它不做判断、不改数据，只把"此刻库里有几份文档、几个片段、id 整体指纹是什么"记下来。
评估报告会读这份文件并核对，对不上就明确写出"与旧报告不可比"。

为什么不用 `eval/corpus.json`：那份是**标注时**的语料清单（6 份 / 66 块），
是手写的，且只在"源文件还在"时才校验得了。这份从数据库导出，覆盖全部文档，
不依赖源文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa import snapshot  # noqa: E402
from rag4prefa.config import Settings  # noqa: E402
from rag4prefa.db import connect  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="导出语料快照")
    parser.add_argument("--out", default=str(snapshot.DEFAULT_PATH), help="输出路径")
    parser.add_argument("--db", default="", help="数据库路径（默认取配置里的 db_path）")
    parser.add_argument("--quiet", action="store_true", help="只打印一行摘要")
    args = parser.parse_args()

    settings = Settings()
    db_path = Path(args.db) if args.db else Path(settings.db_path)
    # 报绝对路径：相对路径在"从别的目录调用"时会指向另一个库，
    # 而快照最怕的就是记录了一份不是你想要的那批语料。
    if not db_path.is_file():
        print(f"数据库不存在：{db_path.resolve()}", file=sys.stderr)
        print("先把文档入库（或检查是不是在仓库根目录下运行）", file=sys.stderr)
        return 2

    conn = connect(db_path)
    try:
        snap = snapshot.build_snapshot(
            conn,
            chunk_min_tokens=settings.chunk_min_tokens,
            chunk_max_tokens=settings.chunk_max_tokens,
        )
    finally:
        conn.close()

    if not snap["document_count"]:
        print(f"库里没有任何文档：{db_path.resolve()}", file=sys.stderr)
        print("空库的快照没有意义，拒绝写出", file=sys.stderr)
        return 2

    out = snapshot.write_snapshot(snap, Path(args.out))
    if args.quiet:
        print(
            f"{snap['document_count']} 份 / {snap['chunk_count']} 片段 / "
            f"ids_sha {snap['ids_sha']} → {out}"
        )
        return 0

    print(f"数据库：{db_path.resolve()}")
    print(
        f"语料：{snap['document_count']} 份文档 · {snap['chunk_count']} 个片段 · "
        f"ids_sha `{snap['ids_sha']}`"
    )
    print(
        f"坐标系：embedding `{snap['embedding_model'] or '未登记'}`"
        f"（{snap['embedding_dim'] or '?'} 维）· "
        f"chunk {snap['chunk_min_tokens']}~{snap['chunk_max_tokens']} token"
    )
    print()
    for name, meta in sorted(
        snap["documents"].items(), key=lambda kv: -kv[1]["chunks"]
    ):
        print(f"  {meta['chunks']:4d} 块  [{meta['project']}] {name[:60]}")
    print(f"\n已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
