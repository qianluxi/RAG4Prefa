"""评估集与语料的契约：解析器一改，`chunk_id` 就全变，标注会静默失效。

`chunk_id` 是内容寻址（`doc_id + 章节路径 + 正文哈希 + 同段序号`），所以任何一处
解析判据调整——多认一个标题、少认一行目录、分块阈值变化——都会让 id 整体换掉。
而 `eval/dataset*.jsonl` 里的 gold 是**具体 id**：不同步的话评估照样能跑完，
只是 Recall 会莫名其妙地低一截，看不出是标注过期还是检索退化。

这份测试把"过期"变成一次明确的失败，并指出要重新生成哪两样东西。
语料目录不在仓库里（体积大、且含真实项目名），所以 CI 上没有这个目录，自动跳过；
它保护的是本机那套语料 + 解析器的组合。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.db import connect, init_db
from rag4prefa.ingest import ingest_file
from tests.conftest import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT.parent / "工程文档"
MANIFEST = ROOT / "eval" / "corpus.json"
DATASETS = [ROOT / "eval" / "dataset.jsonl", ROOT / "eval" / "dataset.paraphrase.jsonl"]

pytestmark = pytest.mark.skipif(
    not CORPUS_DIR.is_dir(),
    reason="本地语料目录不存在（工程文档/ 不进仓库，CI 上跳过）",
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> sqlite3.Connection:
    """把 manifest 里的文档按标注时分块参数重新入库，返回连接。"""
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tmp = tmp_path_factory.mktemp("eval-corpus")
    settings = Settings(
        data_dir=tmp,
        embedding_api_key="test",
        llm_api_key="test",
        chunk_max_tokens=spec["chunk_max_tokens"],
        chunk_min_tokens=spec["chunk_min_tokens"],
        vector_backend="numpy",
    )
    settings.ensure_dirs()
    init_db(settings.db_path)
    connection = connect(settings.db_path)
    embedder = FakeEmbedder()
    store = NumpyStore(connection, embedder.dim)
    for filename, meta in spec["documents"].items():
        path = CORPUS_DIR / filename
        if not path.is_file():
            pytest.skip(f"语料缺文件：{filename}")
        ingest_file(
            connection,
            store,
            embedder,
            settings,
            path=path,
            project_name=meta["project"],
            doc_year=2025,
        )
    yield connection
    connection.close()


def _chunk_ids(conn: sqlite3.Connection, project: str, filename: str) -> list[str]:
    doc_id = hashlib.sha1(f"{project}|{filename}".encode()).hexdigest()[:16]
    rows = conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
    return sorted(row[0] for row in rows)


def test_corpus_matches_manifest(corpus: sqlite3.Connection) -> None:
    """语料指纹必须对得上：对不上说明解析器改了，dataset 也要重生成。

    失败时先看这里的两条出路：确实是有意的解析改进 → 跑一遍
    `python scripts/build_manifest.py` 与评估集生成脚本；不是有意的 → 是回归。
    """
    spec = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for filename, meta in spec["documents"].items():
        ids = _chunk_ids(corpus, meta["project"], filename)
        digest = hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
        assert len(ids) == meta["chunks"], (
            f"{filename} 的 chunk 数从 {meta['chunks']} 变成了 {len(ids)}："
            "解析器或分块参数改动过，eval/corpus.json 与 dataset*.jsonl 都要重新生成"
        )
        assert digest == meta["ids_sha"], (
            f"{filename} 的 chunk_id 集合变了（数量未变）：章节路径或正文哈希变了，"
            "dataset*.jsonl 里的 gold 已失效，需要重新生成"
        )


@pytest.mark.parametrize("dataset", DATASETS, ids=lambda p: p.name)
def test_dataset_gold_ids_exist(corpus: sqlite3.Connection, dataset: Path) -> None:
    """每题的金标都必须能在库里查到——查不到就是标注过期，评估数字会失真。"""
    rows = [
        json.loads(line)
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("//")
    ]
    assert rows, f"{dataset.name} 是空的"

    known = {row[0] for row in corpus.execute("SELECT id FROM chunks")}
    missing: list[str] = []
    for row in rows:
        for chunk_id in row.get("gold_chunk_ids") or []:
            if chunk_id not in known:
                missing.append(f"{row['question'][:20]} → {chunk_id[:12]}")
    assert not missing, (
        f"{dataset.name} 里 {len(missing)} 个 gold_chunk_id 在语料中不存在："
        f"{missing[:3]}。标注与语料已经脱节，重新生成评估集"
    )
