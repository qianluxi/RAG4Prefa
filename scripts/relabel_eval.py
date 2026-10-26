"""评估集重标注：用「正文锚点」重算 `gold_chunk_ids`。

为什么需要这个工具：

`chunk_id` 是内容寻址（`doc_id + 章节路径 + 正文哈希`），解析器任何一处判据调整都会
让 id 整体换掉。dataset 里存的是具体 id，改完解析器不同步，评估**照样能跑完**，
只是 Recall 低一截，看不出是标注过期还是检索退化。

所以每题额外存一个 `anchor`（正文里一段唯一的原文）。重标注时按 anchor 去新库里
重新定位，id 变了也不影响。没有 anchor 的老标注走"用现有 id 反推 anchor"这条路，
跑一次就补齐。

**chunk_id 与 embedding 无关**，所以这里用确定性假向量建临时库，
不需要任何 API Key，也不碰 `data/`。

用法：
    python scripts/relabel_eval.py                       # 补齐 anchor 并校验
    python scripts/relabel_eval.py --write               # 顺便写回 dataset
    python scripts/relabel_eval.py --corpus ../工程文档
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa.adapters.vectors import NumpyStore  # noqa: E402
from rag4prefa.config import Settings  # noqa: E402
from rag4prefa.db import connect, init_db  # noqa: E402
from rag4prefa.ingest import ingest_file  # noqa: E402

ANCHOR_LEN = 24


class _FakeEmbedder:
    """确定性假向量。chunk_id 与向量无关，这里只是为了让入库流程跑得完。"""

    model = "fake-embedder"
    dim = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dim]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)


def build_corpus(corpus_dir: Path, manifest: dict) -> sqlite3.Connection:
    """按 manifest 记录的项目与分块参数重建一份临时库（真实 SQLite，假向量）。"""
    tmp = Path(tempfile.mkdtemp(prefix="r4p-relabel-"))
    settings = Settings(
        data_dir=tmp,
        embedding_api_key="test",
        llm_api_key="test",
        chunk_max_tokens=manifest.get("chunk_max_tokens", 480),
        chunk_min_tokens=manifest.get("chunk_min_tokens", 80),
        vector_backend="numpy",
    )
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)
    embedder = _FakeEmbedder()
    store = NumpyStore(conn, embedder.dim)
    for filename, meta in manifest["documents"].items():
        path = corpus_dir / filename
        if not path.is_file():
            raise SystemExit(f"语料缺文件：{path}")
        ingest_file(
            conn,
            store,
            embedder,
            settings,
            path=path,
            project_name=meta["project"],
            doc_year=2025,
        )
    return conn


def pick_anchor(target: str, rows: list[tuple[str, str, str]]) -> str | None:
    """从目标正文里挑一段"只出现在它自己里面"的片段。

    rows 是 (chunk_id, project, text)。窗口从正文中部开始向前试，
    太靠前的往往是标题（同一章的相邻块都带），不如中间那段有区分度。
    """
    text = next((t for cid, _, t in rows if cid == target), "")
    if len(text) < ANCHOR_LEN:
        return None
    starts = []
    middle = max(0, len(text) // 2 - ANCHOR_LEN // 2)
    starts.append(middle)
    starts.extend(range(0, max(1, len(text) - ANCHOR_LEN), 8))
    for start in starts:
        window = text[start : start + ANCHOR_LEN].strip()
        if len(window) < ANCHOR_LEN:
            continue
        hits = [cid for cid, _, body in rows if window in body]
        if hits == [target]:
            return window
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="按正文锚点重算评估集金标")
    parser.add_argument("--corpus", default="../工程文档", help="语料目录")
    parser.add_argument("--dataset", action="append", help="评估集路径，可重复")
    parser.add_argument("--write", action="store_true", help="写回文件")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    given = Path(args.corpus)
    corpus_dir = given if given.is_absolute() else (root / given).resolve()
    manifest = json.loads((root / "eval" / "corpus.json").read_text(encoding="utf-8"))
    defaults = ["eval/dataset.jsonl", "eval/dataset.paraphrase.jsonl"]
    datasets = [Path(p) for p in (args.dataset or defaults)]

    conn = build_corpus(corpus_dir, manifest)
    rows = [
        (row["id"], row["project"], row["text"])
        for row in conn.execute(
            "SELECT c.id AS id, d.project_name AS project, c.text AS text "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id"
        )
    ]
    print(f"语料：{len(rows)} 块")

    problems = 0
    for path in datasets:
        full = path if path.is_absolute() else root / path
        lines = full.read_text(encoding="utf-8").splitlines()
        out: list[str] = []
        for line in lines:
            if not line.strip() or line.startswith("//"):
                out.append(line)
                continue
            row = json.loads(line)
            scope = row.get("filter", {}).get("projects")
            scoped = [r for r in rows if not scope or r[1] in scope]

            anchors = row.get("anchor") or []
            if anchors:
                # 有锚点：按锚点重新定位，id 变了也能跟上
                gold = []
                for anchor in anchors:
                    hits = [cid for cid, _, body in scoped if anchor in body]
                    if len(hits) != 1:
                        print(
                            f"  [!] 锚点多义/丢失（{len(hits)} 处）："
                            f"{anchor[:24]} | {row['question'][:22]}"
                        )
                        problems += 1
                    gold.extend(hits)
            else:
                # 老标注：用现有 id 反推锚点
                gold, anchors = list(dict.fromkeys(row.get("gold_chunk_ids") or [])), []
                known = {cid for cid, _, _ in scoped}
                for cid in gold:
                    if cid not in known:
                        continue
                    picked = pick_anchor(cid, scoped)
                    if picked is None:
                        print(f"  [!] 找不到唯一锚点：{row['question'][:30]}")
                        problems += 1
                    else:
                        anchors.append(picked)

            if not gold:
                print(f"  [!] 金标为空：{row['question'][:40]}")
                problems += 1
            row["gold_chunk_ids"] = sorted(set(gold))
            if anchors:
                row["anchor"] = anchors
            out.append(json.dumps(row, ensure_ascii=False))

        if args.write:
            full.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"{full.name}: {len(out)} 行，问题 {problems} 处")

    print("OK" if problems == 0 else f"有 {problems} 处需要人工处理")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
