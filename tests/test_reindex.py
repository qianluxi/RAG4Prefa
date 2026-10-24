"""reindex 的回归测试。

三条底线：
1. 局部重建只能影响指定项目——曾经 `store.clear()` 会把全库（含归档）一起删掉；
2. 局部重建必须沿用索引现有的模型——**维度相同也不行**，混用两套向量不会报错，
   只会让检索结果悄悄变差，这是最难在验收里发现的一类故障；
3. 换模型/换维度只能全量重建，且全量重建后归档里不能留下孤儿向量。
"""

from __future__ import annotations

import sqlite3
from array import array
from pathlib import Path

import pytest

from rag4prefa import consistency, repo
from rag4prefa.adapters.vectors import (
    NumpyStore,
    RebuildError,
    VectorItem,
    to_archive_row,
)
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

    with pytest.raises(ReindexError, match="向量空间"):
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


# ── 跨组件失败窗口：归档写完了、向量索引没跟上 ─────────────────────


class FailingRebuildStore(NumpyStore):
    """全量重建直接失败：模拟向量后端在切换前报错。"""

    def rebuild(self, items) -> None:  # type: ignore[override]
        raise RuntimeError("模拟向量后端重建失败")


class FailingUpsertStore(NumpyStore):
    """局部重建写入失败：模拟覆盖本次范围时后端报错。"""

    def upsert(self, items) -> None:  # type: ignore[override]
        raise RuntimeError("模拟向量后端写入失败")


def _query(conn: sqlite3.Connection, store: NumpyStore) -> None:
    consistency.ensure_queryable(
        conn, FakeEmbedder.model, store.dim, store_count=store.count()
    )


def test_full_reindex_failure_blocks_queries_until_recovered(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """全量重建失败：归档已是新版、索引还是旧版，此时**必须**拒绝检索。

    这个场景的可怕之处是它看起来毫无问题：归档写完了，索引重建失败但旧集合还在，
    两边的条数、模型名、维度完全一致——逐项校验一条都挑不出来。所以判断依据只能
    是写归档之前留的那条标记。
    """
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)
    before_stats = repo.vector_stats(conn)

    with pytest.raises(RuntimeError):
        reindex(conn, FailingRebuildStore(conn, FakeEmbedder.dim), FakeEmbedder(), settings)

    # 先把"这个场景确实无法靠比对发现"钉死：数字全都一样
    assert repo.vector_stats(conn) == before_stats, "归档内容与重建前完全相同"
    assert store.count() == total, "旧索引也还在，条数一样"
    assert repo.get_meta(conn, "embedding_model") == FakeEmbedder.model

    # 唯一的证据是标记
    assert repo.get_index_state(conn) == "rebuilding:*"
    with pytest.raises(consistency.IndexMismatch) as excinfo:
        _query(conn, store)
    assert "重建索引没有完成" in str(excinfo.value)

    # verify 命令也不能因为"id 全对"就报 ok——否则运维拿到这个结论就不会再查了
    report = consistency.verify_ids(conn, store)
    assert report["pending_rebuild"] == "rebuilding:*"
    assert report["ok"] is False
    assert not any(report["counts"].values()), "这个场景下三方 id 确实一个不差"

    # 恢复一：归档是完整的，直接按归档重建索引即可（不必重新 embedding、不花钱）
    assert rebuild_store(conn, _store(conn)) == total
    assert repo.get_index_state(conn) == repo.INDEX_READY
    _query(conn, store)  # 恢复之后必须立刻可检索


def test_partial_reindex_failure_is_retryable_for_the_same_project(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """局部重建失败：标记带项目名，重跑**同一个项目**就能清掉它。

    挡住自己是不行的：局部重建正是消除这条标记的方式，若它也要求库先自洽，
    用户就只能去做代价更大的全量重建。
    """
    store = _store(conn)
    _seed(conn, store, settings, docx_file)

    with pytest.raises(RuntimeError):
        reindex(
            conn,
            FailingUpsertStore(conn, FakeEmbedder.dim),
            FakeEmbedder(),
            settings,
            project_name="甲项目",
        )

    assert repo.get_index_state(conn) == "rebuilding:甲项目"
    with pytest.raises(consistency.IndexMismatch):
        _query(conn, store)

    # 重跑同一个项目：前置检查放过同范围的标记，成功后恢复 ready
    result = reindex(conn, store, FakeEmbedder(), settings, project_name="甲项目")
    assert result["rebuilt"] is False
    assert repo.get_index_state(conn) == repo.INDEX_READY
    _query(conn, store)


def test_partial_reindex_cannot_clear_a_full_rebuild_marker(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """全量重建的坑，局部重建填不上——所以它会明确拒绝，而不是假装修好。

    如果放开这一步，只重建一个项目就会把 `rebuilding:*` 恢复成 ready，
    其余项目仍旧是新归档配旧索引，而且再没有任何标记能发现它。
    """
    store = _store(conn)
    _seed(conn, store, settings, docx_file)
    repo.set_index_state(conn, "rebuilding:*")

    with pytest.raises(ReindexError) as excinfo:
        reindex(conn, store, FakeEmbedder(), settings, project_name="甲项目")
    assert "全量" in str(excinfo.value)
    assert repo.get_index_state(conn) == "rebuilding:*", "标记不能被局部重建抹掉"

    # 全量重建是唯一出口
    reindex(conn, store, FakeEmbedder(), settings)
    assert repo.get_index_state(conn) == repo.INDEX_READY


# ── 恢复动作本身必须可验证：空范围不能清标记，rebuild 必须同步元数据 ──────


def test_local_reindex_on_a_missing_scope_keeps_the_full_rebuild_marker(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """范围里没有片段 ≠ 全库已经一致。

    这是最阴的一条：`reindex --project 不存在的项目` 会返回 0 条，如果顺手把状态
    改成 ready，那笔没走完的全量重建就再没有任何痕迹能发现——归档和旧索引的条数、
    模型、维度本来就可能完全一样。
    """
    store = _store(conn)
    _seed(conn, store, settings, docx_file)
    repo.set_index_state(conn, "rebuilding:*")

    result = reindex(conn, store, FakeEmbedder(), settings, project_name="不存在的项目")

    assert result["chunks"] == 0
    assert repo.get_index_state(conn) == "rebuilding:*", "空范围不能清掉全量的标记"
    with pytest.raises(consistency.IndexMismatch):
        _query(conn, store)


def test_local_reindex_on_a_missing_scope_keeps_other_projects_markers(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """同上，但标记属于另一个项目：局部操作只该管自己那摊事。"""
    store = _store(conn)
    _seed(conn, store, settings, docx_file)
    repo.set_index_state(conn, "rebuilding:乙项目")

    reindex(conn, store, FakeEmbedder(), settings, project_name="不存在的项目")

    assert repo.get_index_state(conn) == "rebuilding:乙项目"


def test_full_reindex_on_an_empty_database_clears_the_marker(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """真的什么都没有时，标记才是可以清的——这是"空"唯一站得住的含义。"""
    store = _store(conn)
    repo.set_index_state(conn, "rebuilding:*")

    result = reindex(conn, store, FakeEmbedder(), settings)

    assert result["chunks"] == 0
    assert repo.get_index_state(conn) == repo.INDEX_READY
    _query(conn, store)


def test_full_reindex_on_an_empty_database_clears_a_stale_index(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """片段没了、索引里还留着向量：这不算"空"，清干净之后才能报 ready。"""
    repo.archive_vectors(
        conn,
        [
            to_archive_row(
                VectorItem("孤儿", [0.1] * FakeEmbedder.dim, "甲项目", "", 2024, "fake-embedder")
            )
        ],
    )
    repo.set_index_state(conn, "rebuilding:*")
    store = _store(conn)  # 重新加载才看得到刚写进去的残留
    assert store.count() == 1

    result = reindex(conn, store, FakeEmbedder(), settings)

    assert result["chunks"] == 0
    assert store.count() == 0, "没有正文的残留向量必须被清掉"
    assert repo.get_index_state(conn) == repo.INDEX_READY


def test_full_reindex_refuses_when_facts_cannot_be_read_back(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """片段表有正文、却一条都取不到重建输入：这是数据缺失，不是可以清理的残留。

    若把它当成"空库"处理，全量重建会顺手清掉索引和标记——正文还在，向量却没了，
    而且再没有任何标记提示这件事发生过。

    构造办法：临时关掉外键约束，塞一条指向不存在文档的片段（外键开着时级联删除会让
    这种状态根本存不下来，真实环境里它来自"手工清理数据库""老版本逻辑"这类路径）。
    """
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO chunks (id, doc_id, ordinal, chapter_path, title, text, token_count,"
        " content_sha) VALUES ('孤儿片段', '不存在的文档', 0, '[]', '', '正文', 2, 'x')"
    )
    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    repo.archive_vectors(
        conn,
        [
            to_archive_row(
                VectorItem(
                    "孤儿片段", [0.1] * FakeEmbedder.dim, "甲项目", "", 2024, "fake-embedder"
                )
            )
        ],
    )
    store = _store(conn)
    assert repo.chunk_count(conn) == 1
    assert repo.chunks_for_reindex(conn) == []

    with pytest.raises(ReindexError, match="取不到重建输入"):
        reindex(conn, store, FakeEmbedder(), settings)

    assert store.count() == 1, "被拒绝时不能清索引"
    assert repo.get_index_state(conn) == repo.INDEX_READY, "被拒绝时也不该动标记"


def test_rebuild_store_restores_model_metadata_after_a_switch_failure(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """换模型重建中断后，归档是新模型、元数据还是旧模型。

    只重建索引、不清标记与元数据的话，命令会报成功，但一致性检查仍然拒绝检索——
    因为"元数据模型 ≠ 归档模型"。用户的感受是"恢复命令说好了，可还是查不了"。
    """
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    with pytest.raises(RuntimeError):
        reindex(conn, FailingRebuildStore(conn, FakeEmbedder.dim), AltEmbedder(), settings)

    # 中断现场：归档已经换成 alt 的向量，元数据还停在 fake-embedder
    assert _archive_models(conn) == {"alt-embedder"}
    assert repo.get_meta(conn, "embedding_model") == "fake-embedder"
    assert repo.get_index_state(conn) == "rebuilding:*"

    fresh = _store(conn)
    assert rebuild_store(conn, fresh) == total

    assert repo.get_meta(conn, "embedding_model") == "alt-embedder", "元数据必须跟着归档走"
    assert repo.get_index_state(conn) == repo.INDEX_READY
    # 恢复的判据不是"命令返回了"，而是"按当前配置真的能查"
    consistency.ensure_queryable(
        conn, AltEmbedder.model, fresh.dim, store_count=fresh.count()
    )
    assert fresh.search(AltEmbedder().embed_one("悬挑"), limit=5)


def test_rebuild_store_restores_dim_metadata_after_a_dimension_change(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """维度变了同样要同步：只改索引不改 meta，检索照样被拒。"""
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    with pytest.raises(RuntimeError):
        reindex(conn, FailingRebuildStore(conn, FakeEmbedder.dim), SmallEmbedder(), settings)

    assert repo.get_meta(conn, "embedding_dim") == str(FakeEmbedder.dim)
    assert {r["dim"] for r in conn.execute("SELECT dim FROM vectors")} == {SmallEmbedder.dim}

    fresh = NumpyStore(conn, FakeEmbedder.dim)  # 构造函数按归档加载，维度会跟过去
    assert fresh.dim == SmallEmbedder.dim
    assert rebuild_store(conn, fresh) == total

    assert repo.get_meta(conn, "embedding_dim") == str(SmallEmbedder.dim)
    assert repo.get_meta(conn, "embedding_model") == "small-embedder"
    assert repo.get_index_state(conn) == repo.INDEX_READY
    consistency.ensure_queryable(
        conn, SmallEmbedder.model, fresh.dim, store_count=fresh.count()
    )


def test_rebuild_store_refuses_an_incomplete_archive_and_keeps_the_marker(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """归档本身缺向量时不能"恢复"——重建出来的索引仍然是残缺的，却会宣称 ready。"""
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)
    repo.set_index_state(conn, "rebuilding:*")

    victim = sorted(repo.archived_chunk_ids(conn))[0]
    conn.execute("DELETE FROM vectors WHERE chunk_id = ?", (victim,))
    conn.commit()

    with pytest.raises(ReindexError, match="恢复源"):
        rebuild_store(conn, _store(conn))

    assert repo.get_index_state(conn) == "rebuilding:*", "恢复失败必须把标记留着"
    assert repo.vector_stats(conn)["count"] == total - 1, "被拒绝时不该再改动归档"


class BrokenTargetStore:
    """站在"另一个后端"（如 Milvus）的位置上：不与 SQLite 共享存储。

    与 `FailingRebuildStore` 的区别在于它不继承 NumpyStore：重建失败时，
    numpy 的内存索引和 `vectors` 表是不是还对得上，完全取决于调用方有没有提前改表。
    """

    name = "broken-target"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.notes: list[str] = []

    def rebuild(self, items) -> None:  # type: ignore[override]
        raise RuntimeError("create_collection failed: code=1")

    def count(self) -> int:  # pragma: no cover
        raise AssertionError("重建已经失败，不该再问条数")


def test_a_failed_rebuild_leaves_the_archive_untouched(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """目标后端重建失败时，归档必须原样——numpy 的内存索引就是这张表的缓存。

    旧实现是"先删孤儿行、再重建索引"：删完、重建失败、调用方继续用原来的 numpy
    对象，而它的内存索引里还留着那些行，条数从此对不上，一致性检查开始拒绝检索。
    一次可选迁移于是把本来状态明确的库弄成了不可用。
    """
    store = _store(conn)
    _seed(conn, store, settings, docx_file)
    repo.archive_vectors(
        conn,
        [
            to_archive_row(
                VectorItem(
                    "chunk-已删除", [0.5] * FakeEmbedder.dim, "甲项目", "", 2024, "fake-embedder"
                )
            )
        ],
    )
    before = repo.vector_stats(conn)

    with pytest.raises(RuntimeError):
        rebuild_store(conn, BrokenTargetStore(FakeEmbedder.dim))

    assert repo.vector_stats(conn) == before, "目标后端没成功之前不许删归档行"
    # 源后端按归档重新加载后条数仍然对得上，才叫"能继续用"
    assert _store(conn).count() == before["count"]


def test_a_failed_rebuild_keeps_the_source_backend_queryable(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """归档本来就健康时，迁移失败不该让检索也停下来。"""
    store = _store(conn)
    total = _seed(conn, store, settings, docx_file)

    with pytest.raises(RuntimeError):
        rebuild_store(conn, BrokenTargetStore(FakeEmbedder.dim))

    assert store.count() == total
    _query(conn, store)
    assert store.search(FakeEmbedder().embed_one("悬挑"), limit=5)


def test_rebuild_store_rejects_vectors_shorter_than_their_declared_dim(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """`dim` 是声明、`vec` 的字节长度才是事实，两者必须交叉验证。

    只比声明的话，"每行都写 16 维、实际只存 8 个浮点"的归档能通过全部检查，
    恢复出来的索引与 meta 对不上：命令报成功，检索却继续被拒。
    """
    store = _store(conn)
    _seed(conn, store, settings, docx_file)
    repo.set_index_state(conn, "rebuilding:*")

    victim = sorted(repo.archived_chunk_ids(conn))[0]
    conn.execute(
        "UPDATE vectors SET vec = ? WHERE chunk_id = ?",
        (array("f", [0.1] * (FakeEmbedder.dim // 2)).tobytes(), victim),
    )
    conn.commit()
    before = repo.vector_stats(conn)

    with pytest.raises(ReindexError, match="实际长度"):
        rebuild_store(conn, store)

    assert repo.get_index_state(conn) == "rebuilding:*", "恢复失败必须留着标记"
    assert repo.vector_stats(conn) == before, "被拒绝时不该改动归档"


def test_a_corrupt_archive_fails_with_a_readable_message_not_a_numpy_traceback(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """字节长度坏掉时连 numpy 都拼不出矩阵；报错要指向"从原文重新生成"。"""
    store = _store(conn)
    _seed(conn, store, settings, docx_file)
    victim = sorted(repo.archived_chunk_ids(conn))[0]
    conn.execute(
        "UPDATE vectors SET vec = ? WHERE chunk_id = ?",
        (array("f", [0.1] * (FakeEmbedder.dim // 2)).tobytes(), victim),
    )
    conn.commit()

    with pytest.raises(RebuildError, match="字节长度"):
        _store(conn)


def test_ingest_is_refused_while_a_rebuild_is_pending(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """重建没走完时不许再入库：新向量会按当前模型写进一个来路不明的索引。"""
    store = _store(conn)
    _seed(conn, store, settings, docx_file)
    repo.set_index_state(conn, "rebuilding:*")

    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="丙项目"
    )

    assert outcome.status == "failed"
    assert "重建索引没有完成" in (outcome.error or "")
    assert repo.get_project(conn, "丙项目") is None, "被拒绝时不该留下空项目"

