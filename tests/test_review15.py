"""第十五轮复审：把"活动后端"和"集合打开"这两条状态转换做成可证明的协议。

这一轮的问题有个共同点：**每个动作单独看都是对的，错在"它凭什么认为自己可以动"**。

- 能打开一个 Milvus 集合，不等于它就是归档这一套（条数可以相同、id 全不同）；
- 看到"干净布局"，不等于真正 load 的那一刻还是干净的（中间可能被 rename 掉）；
- 一个维护命令失败返回 failed，不等于检索就停了（按年份过滤还在拿旧列回答）；
- 拿着维护锁，不等于手里这个后端就是库里记着的那个。

所以这里的断言都不看条数，只看"哪个 id 在不在""登记的是谁""检索还答不答"。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag4prefa import admin, consistency, ingest, repo
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.consistency import IndexMismatch
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services, open_store
from tests.conftest import FakeEmbedder


class MemoryStore:
    """不碰 SQLite 的"另一个后端"：内容自己拿着，用来模拟遗留的 Milvus 集合。"""

    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dim = dim
        self.notes: list[str] = []
        self.rows: dict[str, tuple[str, int | None]] = {}
        self.rebuilds = 0
        self.fail_rebuild = False

    def rebuild(self, items) -> None:  # type: ignore[override]
        self.rebuilds += 1
        if self.fail_rebuild:
            raise RuntimeError("create_collection failed: code=1, message=lite is busy")
        self.rows = {i.chunk_id: (i.project_name, i.doc_year) for i in items}

    def upsert(self, items) -> None:  # type: ignore[override]
        for item in items:
            self.rows[item.chunk_id] = (item.project_name, item.doc_year)

    def delete(self, chunk_ids) -> None:  # type: ignore[override]
        for chunk_id in chunk_ids:
            self.rows.pop(chunk_id, None)

    def clear(self) -> None:  # type: ignore[override]
        self.rows.clear()

    def count(self) -> int:  # type: ignore[override]
        return len(self.rows)

    def ids(self) -> set[str]:  # type: ignore[override]
        return set(self.rows)

    def search(self, vector, *, limit, filt=None):  # type: ignore[override]
        return []


def _services(settings: Settings, conn: sqlite3.Connection, store) -> Services:
    return Services(
        settings=settings,
        conn=conn,
        store=store,
        embedder=FakeEmbedder(),
        llm=None,
        reranker=NoopReranker(),
    )


def _seed(
    conn: sqlite3.Connection, store, settings: Settings, docx: Path, year: int = 2024
) -> int:
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx, project_name="甲项目", doc_year=year
    )
    assert outcome.status == "indexed", outcome.error
    return store.count()


def _stale_milvus(like: int, dim: int) -> MemoryStore:
    """一个条数与归档相同、chunk_id 全不同的"遗留集合"。

    这正是最难发现的一种：`health` 看条数对得上，`verify_ids` 之外没有任何检查会反对
    把它当成当前索引。
    """
    store = MemoryStore("milvus", dim)
    store.rows = {f"别的-{i}": ("甲项目", 2024) for i in range(like)}
    return store


# ── P1：能打开 ≠ 就是当前版本 ─────────────────────────────


def test_settle_rebuilds_before_registering_a_backend_whose_ids_differ(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """遗留集合的条数与归档一样，也不能直接登记为活动后端。

    登记的含义是"以后所有写入和检索都算在这份索引上"。只凭"集合能打开"就登记，等于把
    一份内容完全不同的索引当成当前版本——而且条数一致时，日常计数校验一条也发现不了。
    """
    archive = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, archive, settings, docx_file)
    stale = _stale_milvus(total, FakeEmbedder.dim)
    assert stale.count() == archive.count(), "构造前提：条数必须相同，否则这条测试没有鉴别力"

    services = _services(settings, conn, stale)
    assert repo.get_vector_backend(conn) is None

    services.settle_backend()

    assert stale.rebuilds == 1, "内容对不上就得先从归档重建，不能直接登记"
    assert stale.ids() == repo.recoverable_chunk_ids(conn), "登记的前提是内容真的对齐了"
    assert repo.get_vector_backend(conn) == "milvus"
    assert any("逐 id 校验" in note for note in services.notes), (
        "要留下'为什么重建了一遍'的说明，否则运维只看到启动慢"
    )


def test_settle_does_not_register_a_backend_it_could_not_align(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """重建失败就不能登记：宁可这次没有活动后端，也不要承认一份没验证过的索引。"""
    archive = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, archive, settings, docx_file)
    stale = _stale_milvus(total, FakeEmbedder.dim)
    stale.fail_rebuild = True

    services = _services(settings, conn, stale)
    services.settle_backend()

    assert repo.get_vector_backend(conn) is None, "没对齐就不许登记"
    assert any("rebuild" in note for note in services.notes), "要说清怎么修（rebuild）"


def test_settle_refuses_to_register_an_index_that_has_no_facts(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """归档为空、索引里有东西：那不是"确认"，是把上一版的残留认成当前索引。"""
    leftover = MemoryStore("milvus", FakeEmbedder.dim)
    leftover.rows = {"上一版的-1": ("甲项目", 2024)}

    services = _services(settings, conn, leftover)
    services.settle_backend()

    assert repo.get_vector_backend(conn) is None
    assert leftover.count() == 1, "归档是空的，凭什么替人决定这些向量是垃圾"

def test_settle_is_a_no_op_when_the_backend_is_already_recorded(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """已经登记过的后端不该每次启动都重建一遍——那会让启动时间随库变大。"""
    archive = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, archive, settings, docx_file)
    repo.set_vector_backend(conn, "milvus")
    fake = MemoryStore("milvus", FakeEmbedder.dim)
    fake.rows = dict.fromkeys(repo.recoverable_chunk_ids(conn), ("甲项目", 2024))

    _services(settings, conn, fake).settle_backend()

    assert fake.rebuilds == 0


# ── P1 的另一半：显式切后端要真的能走通 ──────────────────


def test_switching_to_numpy_is_registered_so_rebuild_is_possible(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """文档写的"改成 numpy 再 rebuild"，以前在这条路上被自己挡住。

    库里记着 milvus、配置改成 numpy，`ensure_backend_current` 会拒绝一切维护动作——
    包括唯一能把活动后端改过来的 `rebuild`。于是"显式切换"根本没有出路。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, store, settings, docx_file)
    repo.set_vector_backend(conn, "milvus")

    with pytest.raises(IndexMismatch):
        consistency.ensure_backend_current(conn, store)

    _services(settings, conn, store).settle_backend()

    assert repo.get_vector_backend(conn) == "numpy"
    consistency.ensure_backend_current(conn, store)  # 不抛即通过
    assert ingest.rebuild_store(conn, store) == total, "切换完成后 rebuild 必须能跑"


def test_auto_does_not_reopen_milvus_after_an_explicit_switch_to_numpy(
    conn: sqlite3.Connection, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """库已经确认用 numpy 时，`auto` 不该每次启动都去试 Milvus。

    试成功就等于把上一个人显式做出的选择推翻——而它连一句"我改了"都不会说。
    """
    repo.set_vector_backend(conn, "numpy")

    def explode(*_args, **_kwargs):
        raise AssertionError("库已确认用 numpy，auto 不该再去构造 MilvusStore")

    monkeypatch.setattr("rag4prefa.services.MilvusStore", explode)
    auto = settings.model_copy(update={"vector_backend": "auto"})

    store, notes, fell_back = open_store(auto, conn)

    assert store.name == "numpy"
    assert fell_back is False, "这不是'回退'，是按库里记着的选择打开"
    assert notes == []


def test_open_store_does_not_register_the_backend_it_opened(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """打开只证明"能用"，登记要等逐 id 核对过。"""
    archive = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, archive, settings, docx_file)
    repo.set_vector_backend(conn, "numpy")
    fake = _stale_milvus(total, FakeEmbedder.dim)
    # 让 open_store 直接返回这个假后端，模拟"Milvus 能打开"
    import rag4prefa.services as services_module

    real_milvus = services_module.MilvusStore
    services_module.MilvusStore = lambda *a, **k: fake  # type: ignore[assignment]
    try:
        store, notes, _fell_back = open_store(
            settings.model_copy(update={"vector_backend": "milvus"}), conn
        )
    finally:
        services_module.MilvusStore = real_milvus

    assert store is fake
    assert repo.get_vector_backend(conn) == "numpy", "记录还是上一次的值，没被这次打开改写"
    assert any("不会把它登记" in note for note in notes)


# ── P2：改年份失败必须让检索停下来 ────────────────────────


class _UpsertFails(NumpyStore):
    """归档写得进、向量索引写不进：Milvus 超时、集合被锁都是这一类。"""

    def upsert(self, items) -> None:  # type: ignore[override]
        raise RuntimeError("milvus upsert failed: deadline exceeded")


def test_a_failed_year_update_blocks_queries_instead_of_serving_old_filters(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """只更新年份失败时，"任务 failed"是不够的：检索得停下来。

    归档已是 2025、后端还是 2024，条数一条不差。此时若检索照常放行，按年份过滤会返回
    错的那一批——比报错难发现得多。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file, year=2024)

    outcome = ingest_file(
        conn,
        _UpsertFails(conn, FakeEmbedder.dim),
        FakeEmbedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
        doc_year=2025,
    )

    assert outcome.status == "failed"
    marker = repo.get_index_state(conn)
    assert marker == f"unsynced:{ingest._doc_id('甲项目', docx_file.name)}"
    with pytest.raises(IndexMismatch, match="检索"):
        consistency.ensure_queryable(
            conn, FakeEmbedder.model, store.dim, store_count=store.count()
        )


def test_rebuild_clears_the_year_marker_and_keeps_the_new_year(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """恢复走 rebuild：归档里已经是新年份，重建完就一致了，标记必须清掉。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file, year=2024)
    ingest_file(
        conn,
        _UpsertFails(conn, FakeEmbedder.dim),
        FakeEmbedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
        doc_year=2025,
    )
    assert repo.get_index_state(conn).startswith("unsynced:")

    total = ingest.rebuild_store(conn, NumpyStore(conn, FakeEmbedder.dim))

    assert repo.get_index_state(conn) == repo.INDEX_READY
    info = repo.get_document(conn, "甲项目", docx_file.name)
    assert info is not None and info.doc_year == 2025
    consistency.ensure_queryable(
        conn, FakeEmbedder.model, store.dim, store_count=total
    )


def test_a_successful_year_update_restores_the_previous_marker(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """改年份成功时把标记撤成"上一次的状态"，不是无条件 ready。

    上一次欠着一笔没走完的重建，不能被这次顺手擦掉——那正是标记存在的意义。
    这里直接调改年份那一层：正常入口在 `rebuilding:*` 上就会被 `ensure_writable` 拦住，
    而"标记怎么撤"正是要单独验的那件事。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file, year=2024)
    repo.set_index_state(conn, "rebuilding:*")
    doc_id = ingest._doc_id("甲项目", docx_file.name)
    existing = repo.get_document(conn, "甲项目", docx_file.name)
    assert existing is not None

    outcome = ingest._update_doc_year(conn, store, doc_id, docx_file.name, 2025, existing)

    assert outcome.status == "indexed", outcome.error
    assert outcome.note and "2025" in outcome.note
    assert repo.get_index_state(conn) == "rebuilding:*"


# ── P2：删除项目也要确认后端身份 ──────────────────────────


def test_deleting_a_project_refuses_a_store_that_is_not_the_recorded_backend(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """维护锁挡不住"手里的后端是装配那一刻的"，所以删除前要自己对一遍。

    少了这一步：先删掉 SQLite 的事实与归档，再往一个已经不是活动后端的索引发一批 id。
    删除既不完整，事后也说不清该以哪边为准。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, store, settings, docx_file)
    repo.set_vector_backend(conn, "milvus")

    with pytest.raises(IndexMismatch):
        admin.delete_project(conn, store, "甲项目")

    assert repo.get_project(conn, "甲项目") is not None, "被拒绝时一个字节都不许改"
    assert store.count() == total
    assert repo.chunk_count(conn) > 0
    assert repo.archived_chunk_ids(conn), "归档也不许动"


def test_deleting_a_project_works_when_the_backend_matches(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """正向路径不能被上面那条检查误伤。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    repo.set_vector_backend(conn, "numpy")

    result = admin.delete_project(conn, store, "甲项目")

    assert result["deleted"] is True
    assert store.count() == 0
    assert repo.chunk_count(conn) == 0
