"""模型一致性：一个索引里只允许一套向量空间。

这组测试防的不是"程序崩"，而是**程序正常但结果错**：
两个模型同为 16 维，写进同一个索引不会报任何错，只有检索质量悄悄下降。
所以每一条都同时断言"拒绝发生"和"数据没被改动"。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag4prefa import ask, consistency, repo
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
    with pytest.raises(ReindexError, match="向量空间"):
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
    assert state.missing == 2, "缺几条要能精确报出来，而不是只说总数对不上"
    assert any("没有归档向量" in issue for issue in state.blockers)


# ── 故障注入：这些状态以前只是"能报告"，现在是"必须拒绝" ──────────────


@pytest.mark.asyncio
async def test_retrieval_refuses_when_archive_is_empty_but_facts_exist(
    seeded, conn: sqlite3.Connection, settings: Settings
) -> None:
    """情况 A：片段都在，归档向量被人清空了。

    这是最会骗人的一种状态——稠密召回一条都返回不了，BM25 还在正常出结果，
    用户看到一份格式完整的答案，完全不知道向量索引已经废了。
    """
    store, _ = seeded
    conn.execute("DELETE FROM vectors")
    conn.commit()

    state = inspect(conn)
    assert state.count == 0 and state.chunks > 0
    assert state.blockers, "归档为空但事实还在，必须被判定为不一致"

    with pytest.raises(IndexMismatch, match="没有归档向量"):
        await ask.retrieve(conn, store, FakeEmbedder(), settings, question="悬挑", filt=Filter())


@pytest.mark.asyncio
async def test_retrieval_refuses_when_part_of_vectors_lost(
    seeded, conn: sqlite3.Connection, settings: Settings
) -> None:
    """情况 B：只丢了一部分向量，剩下的模型和维度都正确。

    只比"模型对不对"是查不出来的：这里模型只有一个、维度只有一个，但索引已经残缺。
    """
    store, _ = seeded
    ids = [r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM vectors LIMIT 2")]
    conn.executemany("DELETE FROM vectors WHERE chunk_id = ?", [(i,) for i in ids])
    conn.commit()

    with pytest.raises(IndexMismatch, match="没有归档向量"):
        await ask.search(conn, store, FakeEmbedder(), settings, question="悬挑", filt=Filter())


def test_meta_mismatch_blocks_both_retrieval_and_ingest(
    seeded, conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """情况 C：meta 声明与归档事实不符（上次重建没走完），读写都必须拒绝。"""
    store, _ = seeded
    repo.set_meta(conn, "embedding_model", "ghost-embedder")

    state = inspect(conn)
    assert any("上次重建没走完" in issue for issue in state.blockers)

    with pytest.raises(IndexMismatch):
        consistency.ensure_queryable(conn, "fake-embedder", store.dim, store_count=store.count())

    failed = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="丁项目"
    )
    assert failed.status == "failed"
    assert repo.get_project(conn, "丁项目") is None, "被拒绝时不该留下空项目"


def test_store_lagging_behind_archive_is_reported_and_blocks(
    seeded, conn: sqlite3.Connection
) -> None:
    """情况 D：SQLite 归档是对的，后端索引少写了几条。

    只查 SQLite 内部两个数字永远发现不了：chunks == vectors 归档，看起来完全自洽。
    """
    store, _ = seeded
    state = inspect(conn, store_count=store.count() - 1)
    assert state.count == state.chunks, "归档与事实是一致的"
    assert state.blockers, "但后端条数少于归档，仍必须报不一致"

    with pytest.raises(IndexMismatch, match="向量索引实际"):
        consistency.ensure_queryable(
            conn, "fake-embedder", store.dim, store_count=store.count() - 1
        )


def test_unlabeled_vectors_are_not_treated_as_consistent(
    seeded, conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """不知道向量是谁算的，就不能声称它和当前模型同一套空间。"""
    store, _ = seeded
    conn.execute("UPDATE vectors SET model = ''")
    conn.commit()

    state = inspect(conn)
    assert state.unlabeled == state.count > 0
    assert any("没有记录产生它的模型" in issue for issue in state.blockers)

    failed = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="戊项目"
    )
    assert failed.status == "failed"


def test_health_counts_each_kind_of_gap(
    seeded, conn: sqlite3.Connection, settings: Settings
) -> None:
    """health 要把三种差额分开报：缺向量、孤儿、后端条数，便于定位。"""
    from rag4prefa.adapters.rerank import NoopReranker
    from rag4prefa.services import Services

    store, outcome = seeded
    conn.execute("DELETE FROM vectors WHERE chunk_id = (SELECT chunk_id FROM vectors LIMIT 1)")
    conn.commit()

    services = Services(
        settings=settings, conn=conn, store=store, embedder=FakeEmbedder(), llm=None,
        reranker=NoopReranker(),
    )
    health = services.health()
    assert health["index"]["missing_vectors"] == 1
    assert health["index"]["consistent"] is False
    assert health["ok"] is False
    assert health["index"]["chunks"] == outcome.chunks


def test_orphan_archive_rows_are_not_read_back(conn: sqlite3.Connection) -> None:
    """没有正文的归档行不该被 rebuild_store 读回去。"""
    from rag4prefa.adapters.vectors import VectorItem, to_archive_row

    repo.archive_vectors(
        conn, [to_archive_row(VectorItem("孤儿", [0.1] * 4, "甲项目", "", 2024, "m"))]
    )
    assert repo.iter_archived_vectors(conn) == []
    assert repo.delete_orphan_vectors(conn) == 1
