"""reindex 的回归测试。

两条底线：
1. 局部重建只能影响指定项目——曾经 `store.clear()` 会把全库（含归档）一起删掉；
2. 维度变化只能整体重建，且必须明确报错，而不是留下两套维度的向量。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag4prefa import repo
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.domain import Filter
from rag4prefa.ingest import ReindexError, ingest_file, rebuild_store, reindex
from tests.conftest import FakeEmbedder


class AltEmbedder(FakeEmbedder):
    """同维度、不同向量：模拟换了一个维度相同的模型。"""

    model = "alt-embedder"

    def _vec(self, text: str) -> list[float]:
        return super()._vec("alt|" + text)


class SmallEmbedder(FakeEmbedder):
    """维度更小：模拟换了一个维度不同的模型。"""

    model = "small-embedder"
    dim = 8


def _store(conn: sqlite3.Connection) -> NumpyStore:
    return NumpyStore(conn, FakeEmbedder.dim)


def _seed(
    conn: sqlite3.Connection, store: NumpyStore, settings: Settings, docx_file: Path
) -> int:
    """建两个项目，返回总向量数。"""
    embedder = FakeEmbedder()
    for project, year in (("甲项目", 2024), ("乙项目", 2020)):
        outcome = ingest_file(
            conn,
            store,
            embedder,
            settings,
            path=docx_file,
            project_name=project,
            doc_year=year,
        )
        assert outcome.status == "indexed"
    return store.count()


def test_partial_reindex_only_touches_target_project(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)
    jia_chunks = repo.get_project(conn, "甲项目").chunk_count

    result = reindex(conn, store, AltEmbedder(), settings, project_name="甲项目")

    assert result["chunks"] == jia_chunks
    assert result["rebuilt"] is False
    assert store.count() == total, "重建一个项目不该让其他项目的向量消失"

    # 乙项目仍可检索
    hits = store.search(
        FakeEmbedder().embed_one("悬挑"), limit=50, filt=Filter(projects=["乙项目"])
    )
    assert hits, "乙项目的向量被误删了"

    models = {
        r["model"]
        for r in conn.execute("SELECT model FROM vectors WHERE project_name = '甲项目'")
    }
    assert models == {"alt-embedder"}
    others = {
        r["model"]
        for r in conn.execute("SELECT model FROM vectors WHERE project_name = '乙项目'")
    }
    assert others == {"fake-embedder"}, "非目标项目的向量不该被改写"
    assert repo.get_meta(conn, "embedding_model") == "alt-embedder"


def test_partial_reindex_rejects_dim_change(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    with pytest.raises(ReindexError, match="整体重建"):
        reindex(conn, store, SmallEmbedder(), settings, project_name="甲项目")

    assert store.count() == total, "拒绝执行时必须保持原样"
    dims = {r["dim"] for r in conn.execute("SELECT dim FROM vectors")}
    assert dims == {FakeEmbedder.dim}


def test_full_reindex_allows_dim_change(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    result = reindex(conn, store, SmallEmbedder(), settings)

    assert result["rebuilt"] is True
    assert result["dim"] == SmallEmbedder.dim
    assert store.dim == SmallEmbedder.dim
    assert store.count() == total
    assert {r["dim"] for r in conn.execute("SELECT dim FROM vectors")} == {SmallEmbedder.dim}
    assert repo.get_meta(conn, "embedding_dim") == str(SmallEmbedder.dim)
    assert repo.get_meta(conn, "embedding_model") == "small-embedder"


def test_rebuild_store_preserves_archive(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    rebuilt = rebuild_store(conn, store)

    assert rebuilt == total
    assert store.count() == total
    models = {r["model"] for r in conn.execute("SELECT model FROM vectors")}
    assert models == {"fake-embedder"}, "重建索引不该把归档里的模型名抹成空串"


def test_reindex_on_empty_store_returns_zero(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    store = _store(conn)
    result = reindex(conn, store, FakeEmbedder(), settings)
    assert result["chunks"] == 0
    assert result["rebuilt"] is False
