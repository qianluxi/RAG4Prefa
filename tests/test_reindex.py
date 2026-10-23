"""reindex 的回归测试。

三条底线：
1. 局部重建只能影响指定项目——曾经 `store.clear()` 会把全库（含归档）一起删掉；
2. 局部重建必须沿用索引现有的模型——**维度相同也不行**，混用两套向量不会报错，
   只会让检索结果悄悄变差，这是最难在验收里发现的一类故障；
3. 换模型/换维度只能全量重建，且全量重建后归档里不能留下孤儿向量。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag4prefa import repo
from rag4prefa.adapters.vectors import NumpyStore, VectorItem, to_archive_row
from rag4prefa.config import Settings
from rag4prefa.domain import Filter
from rag4prefa.ingest import ReindexError, ingest_file, rebuild_store, reindex
from tests.conftest import FakeEmbedder


class AltEmbedder(FakeEmbedder):
    """同维度、不同向量空间：模拟换了一个 16 维的模型。"""

    model = "alt-embedder"

    def _vec(self, text: str) -> list[float]:
        return super()._vec("alt|" + text)


class SmallEmbedder(FakeEmbedder):
    """维度更小、模型名也不同。"""

    model = "small-embedder"
    dim = 8


class SameNameSmallEmbedder(FakeEmbedder):
    """模型名不变但维度变了（服务端换了权重的极端情况），用来单独验证维度分支。"""

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


def _archive_models(conn: sqlite3.Connection, project: str | None = None) -> set[str]:
    sql = "SELECT DISTINCT model FROM vectors"
    params: tuple = ()
    if project:
        sql += " WHERE project_name = ?"
        params = (project,)
    return {r["model"] for r in conn.execute(sql, params)}


def test_partial_reindex_same_model_only_touches_target_project(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)
    jia_chunks = repo.get_project(conn, "甲项目").chunk_count

    result = reindex(conn, store, FakeEmbedder(), settings, project_name="甲项目")

    assert result["chunks"] == jia_chunks
    assert result["rebuilt"] is False
    assert store.count() == total, "重建一个项目不该让其他项目的向量消失"

    hits = store.search(
        FakeEmbedder().embed_one("悬挑"), limit=50, filt=Filter(projects=["乙项目"])
    )
    assert hits, "乙项目的向量被误删了"
    assert _archive_models(conn) == {"fake-embedder"}


def test_partial_reindex_rejects_model_switch_at_same_dim(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """同维度换模型必须被拒绝：这是"能跑但结果无意义"的典型入口。"""
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    with pytest.raises(ReindexError, match="沿用"):
        reindex(conn, store, AltEmbedder(), settings, project_name="甲项目")

    assert store.count() == total
    assert _archive_models(conn) == {"fake-embedder"}, "被拒绝时不能改写任何向量"
    assert repo.get_meta(conn, "embedding_model") == "fake-embedder"


def test_partial_reindex_rejects_dim_change(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    with pytest.raises(ReindexError, match="维度"):
        reindex(conn, store, SameNameSmallEmbedder(), settings, project_name="甲项目")

    assert store.count() == total, "拒绝执行时必须保持原样"
    assert {r["dim"] for r in conn.execute("SELECT dim FROM vectors")} == {FakeEmbedder.dim}


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


def test_full_reindex_is_the_only_way_to_switch_model(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    result = reindex(conn, store, AltEmbedder(), settings)

    assert result["model"] == "alt-embedder"
    assert result["dim"] == FakeEmbedder.dim
    assert _archive_models(conn) == {"alt-embedder"}, "全量重建后整个索引只能有一套房型"
    assert repo.get_meta(conn, "embedding_model") == "alt-embedder"
    assert store.count() == total
    # 检索自洽：用新模型查得到，用旧模型算出来的向量已不该再被当作同一个空间
    hits = store.search(AltEmbedder().embed_one("悬挑"), limit=5)
    assert hits


def test_full_reindex_drops_orphan_archive_rows(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """没有对应片段的归档行必须被清掉，否则 rebuild_store 会把旧向量灌回来。"""
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)
    orphan = VectorItem(
        "chunk-已删除", [0.5] * FakeEmbedder.dim, "甲项目", "", 2024, "fake-embedder"
    )
    repo.archive_vectors(conn, [to_archive_row(orphan)])
    assert repo.vector_stats(conn)["count"] == total + 1
    assert "chunk-已删除" not in {r["chunk_id"] for r in repo.iter_archived_vectors(conn)}

    reindex(conn, store, FakeEmbedder(), settings)

    assert repo.vector_stats(conn)["count"] == total, "孤儿归档行应被物理删除"
    assert store.count() == total, "孤儿不该进入正式索引"


def test_rebuild_store_ignores_orphans_and_preserves_model(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)
    orphan = VectorItem(
        "chunk-已删除", [0.5] * FakeEmbedder.dim, "甲项目", "", 2024, "fake-embedder"
    )
    repo.archive_vectors(conn, [to_archive_row(orphan)])

    rebuilt = rebuild_store(conn, store)

    assert rebuilt == total
    assert store.count() == total
    assert _archive_models(conn) == {"fake-embedder"}, "重建索引不该把归档里的模型名抹成空串"


def test_reindex_on_empty_store_returns_zero(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    store = _store(conn)
    result = reindex(conn, store, FakeEmbedder(), settings)
    assert result["chunks"] == 0
    assert result["rebuilt"] is False
