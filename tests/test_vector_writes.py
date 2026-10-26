"""向量写入的两条硬要求：批次原子、缓存跟得上别的进程。

这两条都不是"异常处理写错了"，而是**边界画错了**：

1. `NumpyStore.upsert` 曾经逐条 `INSERT OR REPLACE`、在循环里校验。第 2 条不合格时，
   第 1 条已经躺在一个不属于任何显式事务的隐式事务里——随后任何一次 `commit()`
   （比如另一个线程更新任务进度）都会把它一起提交。结果是"入库失败、旧向量却被
   新版覆盖"，而回滚逻辑只删本次新增的 id，删不掉被覆盖的旧行。
2. numpy 后端的索引是**进程内的内存矩阵**。维护锁能让 CLI 与 API 不同时写，但挡不住
   "CLI 重建完了，API 里的矩阵还是旧的"——锁释放之后它既不报错也不会自己变。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag4prefa import repo
from rag4prefa.adapters.vectors import NumpyStore, VectorItem, to_archive_row
from rag4prefa.config import Settings
from rag4prefa.db import connect
from rag4prefa.ingest import ingest_file
from tests.conftest import FakeEmbedder


def _archived_bytes(conn: sqlite3.Connection, chunk_id: str) -> bytes:
    row = conn.execute("SELECT vec FROM vectors WHERE chunk_id = ?", (chunk_id,)).fetchone()
    assert row is not None
    return bytes(row["vec"])


def test_a_batch_upsert_leaves_the_previous_vectors_intact_when_a_later_item_is_invalid(
    conn: sqlite3.Connection,
) -> None:
    """一批里第 2 条维度不对：第 1 条也不能落盘，旧值必须原样保留。"""
    victim = "chunk-已有"
    repo.archive_vectors(
        conn,
        [
            to_archive_row(
                VectorItem(victim, [0.1] * FakeEmbedder.dim, "甲项目", "", 2024, "fake-embedder")
            )
        ],
    )
    store = NumpyStore(conn, FakeEmbedder.dim)
    before = _archived_bytes(conn, victim)
    assert store.count() == 1

    items = [
        # 第 1 条是中规中矩的覆盖写，它会先被写进事务。
        # 注意向量必须是"非等值"的：归一化之后 [0.1]*16 与 [0.9]*16 都是 0.25，
        # 字节完全相同，用它做判据等于什么都没测。
        VectorItem(
            victim,
            [0.9] * (FakeEmbedder.dim // 2) + [0.1] * (FakeEmbedder.dim // 2),
            "甲项目",
            "",
            2024,
            "fake-embedder",
        ),
        # 第 2 条维度不对，校验在写入之前就该拦住整批
        VectorItem(
            "chunk-新", [0.5] * (FakeEmbedder.dim // 2), "甲项目", "", 2024, "fake-embedder"
        ),
    ]
    with pytest.raises(ValueError, match="维度"):
        store.upsert(items)

    # 关键的一步：让另一个写事务提交一次。旧实现里正是这一下把半批写带了进来
    # （任务进度更新、状态记录都会触发），修复后没有东西可提交。
    repo.set_meta(conn, "probe", "1")

    assert _archived_bytes(conn, victim) == before, "旧向量被覆盖了，等于旧版本已经不可用"
    assert repo.vector_stats(conn)["count"] == 1, "半批不能留下任何一行"
    assert store.count() == 1


def test_the_numpy_cache_follows_a_change_made_by_another_process(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """另一个进程改了索引之后，本进程的缓存必须先刷新再用于检索。

    现实版本：服务在跑（numpy 后端），运维另开一个终端跑了 `reindex` 或 `rebuild`。
    维护锁保证了两边不同时写，但锁一释放，服务进程里那份内存矩阵就是旧的——
    条数可以对得上（重建换了模型、片段数没变），于是"检查通过、查的是旧内容"。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="甲项目", doc_year=2024
    )
    assert outcome.status == "indexed"
    before = store.count()
    assert before >= 2

    victim = sorted(repo.archived_chunk_ids(conn))[0]

    # 另一个进程：自己一条连接，删掉一条归档向量并推代次（相当于 reindex 换了索引）
    other = connect(settings.db_path)
    try:
        other.execute("DELETE FROM vectors WHERE chunk_id = ?", (victim,))
        other.commit()
        repo.bump_index_generation(other)
    finally:
        other.close()

    assert victim not in store.ids(), "缓存没刷新就会继续认为这条还在这里"
    assert store.count() == before - 1
    assert all(cid != victim for cid, _ in store.search([0.5] * FakeEmbedder.dim, limit=10))
