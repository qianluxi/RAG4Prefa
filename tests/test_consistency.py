"""模型一致性：一个索引里只允许一套向量空间。

这组测试防的不是"程序崩"，而是**程序正常但结果错**：
两个模型同为 16 维，写进同一个索引不会报任何错，只有检索质量悄悄下降。
所以每一条都同时断言"拒绝发生"和"数据没被改动"。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag4prefa import ask, repo
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.consistency import IndexMismatch, inspect
from rag4prefa.domain import Filter
from rag4prefa.ingest import ReindexError, ingest_file
from tests.conftest import FakeEmbedder


class AltEmbedder(FakeEmbedder):
    """同维度（16）、不同向量空间。"""

    model = "alt-embedder"

    def _vec(self, text: str) -> list[float]:
        return super()._vec("alt|" + text)


@pytest.fixture
def seeded(conn: sqlite3.Connection, settings: Settings, docx_file: Path):
    store = NumpyStore(conn, FakeEmbedder.dim)
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="甲项目", doc_year=2024
    )
    assert outcome.status == "indexed"
    return store, outcome


def test_index_state_reads_declaration_and_fact(seeded, conn: sqlite3.Connection) -> None:
    store, outcome = seeded
    state = inspect(conn)
    assert state.model == "fake-embedder"
    assert state.dim == FakeEmbedder.dim
    assert state.count == outcome.chunks
    assert state.known_models == {"fake-embedder"}
    assert state.issues == []


def test_ingest_with_other_model_is_refused(seeded, conn: sqlite3.Connection, settings, docx_file):
    """增量入库也必须校验，不能只在 reindex 里检查。"""
    store, outcome = seeded
    before = store.count()

    failed = ingest_file(
        conn, store, AltEmbedder(), settings, path=docx_file, project_name="乙项目"
    )

    assert failed.status == "failed"
    assert "reindex" in (failed.error or "")
    assert store.count() == before, "被拒绝时不能写入任何向量"
    assert {r["model"] for r in conn.execute("SELECT DISTINCT model FROM vectors")} == {
        "fake-embedder"
    }
    assert repo.get_meta(conn, "embedding_model") == "fake-embedder", "元数据不能被改写"
    assert repo.get_project(conn, "乙项目") is None, "被拒绝时不该留下空项目"


def test_partial_reindex_with_other_model_is_refused(
    seeded, conn: sqlite3.Connection, settings
) -> None:
    """维度相同也不行——这正是上一轮测试把 bug 当成功用例的那条路径。"""
    store, _ = seeded
    with pytest.raises(ReindexError, match="沿用"):
        from rag4prefa.ingest import reindex

        reindex(conn, store, AltEmbedder(), settings, project_name="甲项目")
    assert inspect(conn).known_models == {"fake-embedder"}


@pytest.mark.asyncio
async def test_retrieval_refuses_when_query_model_differs(
    seeded, conn: sqlite3.Connection, settings
) -> None:
    """配置换了模型但没重建时，检索必须报错，而不是返回一份看起来正常的引用。"""
    store, _ = seeded
    with pytest.raises(IndexMismatch, match="向量空间"):
        await ask.retrieve(
            conn, store, AltEmbedder(), settings, question="悬挑做法", filt=Filter()
        )


def test_health_reports_inconsistency(seeded, conn: sqlite3.Connection, settings) -> None:
    from rag4prefa.adapters.rerank import NoopReranker
    from rag4prefa.services import Services

    store, _ = seeded
    services = Services(
        settings=settings,
        conn=conn,
        store=store,
        embedder=AltEmbedder(),
        llm=None,
        reranker=NoopReranker(),
    )
    health = services.health()
    assert health["index"]["consistent"] is False
    assert health["ok"] is False
    assert any("不一致" in issue for issue in health["index"]["issues"])


def test_health_is_clean_for_matching_model(seeded, conn: sqlite3.Connection, settings) -> None:
    from rag4prefa.adapters.rerank import NoopReranker
    from rag4prefa.services import Services

    store, _ = seeded
    services = Services(
        settings=settings,
        conn=conn,
        store=store,
        embedder=FakeEmbedder(),
        llm=None,
        reranker=NoopReranker(),
    )
    health = services.health()
    assert health["index"]["consistent"] is True
    assert health["index"]["issues"] == []


def test_mixed_archive_is_reported_and_blocks_writes(
    seeded, conn: sqlite3.Connection, settings, docx_file: Path
) -> None:
    """手工构造"混用状态"（历史遗留），必须被识别出来并阻止继续写入。"""
    store, _ = seeded
    # 直接把一半向量标成另一个模型，模拟有人绕过代码写进来的情况
    conn.execute("UPDATE vectors SET model = 'ghost-embedder' WHERE rowid % 2 = 1")
    conn.commit()

    state = inspect(conn)
    assert state.known_models == {"fake-embedder", "ghost-embedder"}
    assert any("混有" in issue for issue in state.issues)

    failed = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="丙项目"
    )
    assert failed.status == "failed"
    assert "混有" in (failed.error or "")


def test_index_lagging_behind_facts_is_reported(seeded, conn: sqlite3.Connection) -> None:
    """片段已落库、向量没写完的中间状态必须被识别出来（否则只是"少召回"不报错）。"""
    store, _ = seeded
    ids = [r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM vectors LIMIT 2")]
    conn.executemany("DELETE FROM vectors WHERE chunk_id = ?", [(i,) for i in ids])
    conn.commit()

    state = inspect(conn)
    assert state.count < state.chunks
    assert any("索引与事实不一致" in issue for issue in state.issues)


def test_orphan_archive_rows_are_not_read_back(conn: sqlite3.Connection) -> None:
    """没有正文的归档行不该被 rebuild_store 读回去。"""
    from rag4prefa.adapters.vectors import VectorItem, to_archive_row

    repo.archive_vectors(
        conn, [to_archive_row(VectorItem("孤儿", [0.1] * 4, "甲项目", "", 2024, "m"))]
    )
    assert repo.iter_archived_vectors(conn) == []
    assert repo.delete_orphan_vectors(conn) == 1
