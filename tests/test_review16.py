"""第十六轮复审：把"证明"从"身份"推进到"内容"，并要求读者也证明自己手里的东西。

上一轮判定"后端可以登记"的依据是 id 集合。这轮指出它只能证明**记录的身份**一致：
Milvus 里还有向量值、`doc_year`、`project_code`，从 Milvus 切到 numpy、在 numpy 上
重建过向量或改过年份之后，旧集合的 id 可以一个不差，内容却整整差一代。

另外两条边界同样是"没证明就放行"：

- 查询侧只看代次，不看"这个进程手里的后端是不是库里登记的那个"；
- 回滚失败只写进错误信息、不落盘，于是任务 failed 了、检索还在回答；
- 空归档时把"读不到后端条数"当成"确认为空"。

所以这里的断言一律落在**内容**与**能不能查**上，不落在条数上。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from rag4prefa import admin, ask, consistency, ingest, repo
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.concurrency import maintenance_lock
from rag4prefa.config import Settings
from rag4prefa.consistency import IndexMismatch
from rag4prefa.domain import Filter
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services
from tests.conftest import FakeEmbedder


class MemoryStore:
    """不碰 SQLite 的"另一个后端"：内容自己拿着，才能造出"id 相同、内容过期"。

    真 Milvus 上这个场景太容易发生了——切到 numpy 之后旧集合没人管，再切回来时
    id 集合一个不差，向量和过滤列却全停在切换前。
    """

    def __init__(self, name: str, dim: int) -> None:
        self.name = name
        self.dim = dim
        self.notes: list[str] = []
        # chunk_id -> (project_name, doc_year, 向量首分量)
        self.rows: dict[str, tuple[str, int | None, float]] = {}
        self.rebuilds = 0
        self.count_error: Exception | None = None
        self.fail_delete = False

    def rebuild(self, items) -> None:  # type: ignore[override]
        self.rebuilds += 1
        self.rows = {
            i.chunk_id: (i.project_name, i.doc_year, i.vector[0]) for i in items
        }

    def upsert(self, items) -> None:  # type: ignore[override]
        for item in items:
            self.rows[item.chunk_id] = (item.project_name, item.doc_year, item.vector[0])

    def delete(self, chunk_ids) -> None:  # type: ignore[override]
        if self.fail_delete:
            raise RuntimeError("milvus delete failed: deadline exceeded")
        for chunk_id in chunk_ids:
            self.rows.pop(chunk_id, None)

    def clear(self) -> None:  # type: ignore[override]
        self.rows.clear()

    def count(self) -> int:  # type: ignore[override]
        if self.count_error is not None:
            raise self.count_error
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


def _stale_but_same_ids(conn: sqlite3.Connection, dim: int) -> MemoryStore:
    """id 集合与归档完全相同、内容却是上一代的"遗留集合"。"""
    store = MemoryStore("milvus", dim)
    archive = repo.iter_archived_vectors(conn)
    store.rows = {
        r["chunk_id"]: (r["project_name"], 2019, 0.5)  # 年份与向量都故意不是这一版
        for r in archive
    }
    return store


# ── P1-1：id 相同证明不了内容相同 ──────────────────────────


def test_settle_rewrites_content_even_when_the_id_sets_are_identical(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """条数与 id 全都对得上，也必须重写一遍才允许登记。

    这是"只在换后端时需要证明"的另一半：id 一致只能说明是同一批**片段**，说明不了
    索引用的是这一版的向量和过滤列。遗留集合里向量是旧的、`doc_year` 还是 2019，
    登记之后每次按年份过滤都会查错，而日常校验一条也不反对。
    """
    archive = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, archive, settings, docx_file, year=2024)
    repo.set_vector_backend(conn, "numpy")
    stale = _stale_but_same_ids(conn, FakeEmbedder.dim)
    assert stale.ids() == repo.recoverable_chunk_ids(conn), (
        "构造前提：id 集合必须完全一致，否则这条测试退化成上一轮那条"
    )

    services = _services(settings, conn, stale)
    services.settle_backend()

    assert stale.rebuilds == 1, "id 一致不是证据，换后端必须从归档重写内容"
    years = {row[1] for row in stale.rows.values()}
    assert years == {2024}, f"过滤列必须被改写成归档里的年份，实际是 {years}"
    assert {row[2] for row in stale.rows.values()} != {0.5}, "向量也必须重写，不能留旧的"
    assert repo.get_vector_backend(conn) == "milvus"
    assert services.backend_confirmed is True


def test_settle_bumps_the_generation_so_in_flight_queries_notice(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """换后端要推进索引代次，否则在途查询看不出"索引已经换成另一份"。

    上一轮这条分支走的是 `set_vector_backend`，它只写一行 meta、不推代次。结果是：
    索引换了、`index_generation` 没变，正在算查询向量的那次请求两次确认都过。
    """
    archive = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, archive, settings, docx_file)
    repo.set_vector_backend(conn, "numpy")
    before = repo.get_index_generation(conn)

    _services(settings, conn, _stale_but_same_ids(conn, FakeEmbedder.dim)).settle_backend()

    assert repo.get_index_generation(conn) > before, "换索引必须作废别人的缓存"


# ── P1-2：登记对了，不等于在跑的服务用的是它 ─────────────


async def test_a_long_running_service_notices_its_backend_was_switched_away(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """别的进程换了活动后端，本进程手里的旧索引必须拒绝服务。

    代次检查挡不住这一种：本次查询的开始就在切换之后，整段期间代次一个数都没变，
    两次确认自然都通过——而这个进程手里拿的还是切换之前那份索引，它不会报错，
    只会安静地返回旧内容。
    """
    old = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, old, settings, docx_file)
    repo.set_vector_backend(conn, "numpy")
    services = _services(settings, conn, old)
    assert services.settle_backend() is True
    assert services.backend_confirmed is True

    # 另一个进程：把归档重建进 Milvus，活动后端随之改成 milvus
    milvus = MemoryStore("milvus", FakeEmbedder.dim)
    ingest.rebuild_store(conn, milvus)
    assert repo.get_vector_backend(conn) == "milvus"

    with pytest.raises(IndexMismatch, match="重启服务"):
        await ask.retrieve(
            conn, old, FakeEmbedder(), settings, question="悬挑", filt=Filter()
        )


def test_query_ready_refuses_a_service_that_could_not_confirm_its_backend(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """启动时没确认成功（比如抢不到维护锁），这次的检索就不能当成没问题。

    以前这种情况只在 notes 里留一句"下次启动会重新核对"，检索照旧回答——
    而"没确认"的具体含义正是"我不知道这份索引是不是这一版"。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    services = _services(settings, conn, store)

    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with maintenance_lock(conn, purpose="另一个进程的重建", timeout=None):
            held.set()
            release.wait(10)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert held.wait(10)
    try:
        assert services.settle_backend() is False, "抢不到锁就不能假装确认过"
    finally:
        release.set()
        worker.join(10)

    assert services.backend_confirmed is False
    with pytest.raises(IndexMismatch, match="没能确认活动后端"):
        services.assert_query_ready()


# ── P2-2：读不到条数 ≠ 确认为空 ───────────────────────────


def test_settle_does_not_register_a_backend_whose_count_cannot_be_read(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """归档为空、后端 `count()` 抛异常：不许登记。

    以前这里 `except Exception: leftover = 0`，于是 SDK 超时、连接被占都被读成
    "后端是空的"，接着登记。之后 `open_store` 又把它当"已有活动后端"，这个从没被
    证实过的后端就一路被信任下去了。
    """
    broken = MemoryStore("milvus", FakeEmbedder.dim)
    broken.count_error = RuntimeError("milvus count failed: deadline exceeded")

    services = _services(settings, conn, broken)
    services.settle_backend()

    assert repo.get_vector_backend(conn) is None, "证明不了就不能登记"
    assert services.backend_confirmed is False
    assert any("读不到" in note for note in services.notes), "要说清是读不到、不是空"


# ── P2-1：回滚失败必须落盘，而不是只写进错误信息 ──────────


class _UpsertDies(NumpyStore):
    """第一批写得进，之后每次写都失败：模拟"upsert 抛了，但批次可能已经生效"。"""

    def __init__(self, conn: sqlite3.Connection, dim: int) -> None:
        super().__init__(conn, dim)
        self.broken = False

    def upsert(self, items) -> None:  # type: ignore[override]
        if self.broken:
            raise RuntimeError("milvus upsert failed: deadline exceeded")
        super().upsert(items)


def test_a_failed_rollback_leaves_a_marker_so_queries_stop(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """upsert 失败、把旧行写回也失败：条数 id 可能都对得上，检索必须停。

    这条路径以前只把"还原失败"写进返回文本：任务 failed，索引停在"一半旧行、一半新行"，
    而 id 集合、条数、模型、维度全都可以一致——计数校验看不出任何异常，检索照常回答。
    """
    store = _UpsertDies(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file, year=2024)
    before = repo.chunk_ids_of_doc(conn, ingest._doc_id("甲项目", docx_file.name))

    store.broken = True
    outcome = ingest_file(
        conn,
        store,
        FakeEmbedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
        doc_year=2025,
        force=True,
    )

    assert outcome.status == "failed"
    assert "检索已停用" in (outcome.error or ""), outcome.error
    marker = repo.get_index_state(conn)
    assert marker == f"unsynced:{ingest._doc_id('甲项目', docx_file.name)}"
    # id 集合一条没变，正是"计数校验发现不了"的那种状态
    assert set(repo.chunk_ids_of_doc(conn, ingest._doc_id("甲项目", docx_file.name))) == set(
        before
    )
    with pytest.raises(IndexMismatch, match="检索"):
        consistency.ensure_queryable(
            conn, FakeEmbedder.model, store.dim, store_count=store.count()
        )


def test_a_successful_rollback_keeps_the_library_queryable(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """回滚成功就不该置标记：网络抖一下导致 upsert 立刻失败，没必要让整个库停服。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file, year=2024)

    class _FailsOnce(NumpyStore):
        broken = True

        def upsert(self, items) -> None:  # type: ignore[override]
            if self.broken:
                self.broken = False
                raise RuntimeError("milvus upsert failed: transient")
            super().upsert(items)

    flaky = _FailsOnce(conn, FakeEmbedder.dim)
    outcome = ingest_file(
        conn,
        flaky,
        FakeEmbedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
        doc_year=2025,
        force=True,
    )

    assert outcome.status == "failed"
    assert repo.get_index_state(conn) == repo.INDEX_READY, "还原成功了就别留停机标记"
    consistency.ensure_queryable(
        conn, FakeEmbedder.model, store.dim, store_count=store.count()
    )


def test_deleting_a_project_leaves_a_marker_when_the_index_delete_fails(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """事实删掉了、索引没删掉：这也是一次跨组件修改，必须留痕。

    计数校验多半能发现（索引多于归档），但那只是"多半"——同一个事务窗口里若有别的
    写入凑巧补齐差额，就只能靠标记。而且标记里写清了修复方式。

    用独立存储的假后端：numpy 的"索引"就是 `vectors` 表，事实一删它就跟着空了，
    表达不出"索引里还留着没有正文的向量"这个状态。
    """
    store = MemoryStore("numpy", FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    repo.set_vector_backend(conn, "numpy")
    store.fail_delete = True

    with pytest.raises(IndexMismatch, match="rebuild"):
        admin.delete_project(conn, store, "甲项目")

    assert repo.get_project(conn, "甲项目") is None, "事实删除已经提交，这是不可回退的"
    assert repo.get_index_state(conn) == "unsynced:project:甲项目"
    assert store.count() > 0, "索引里还留着没有正文的向量"


def test_a_successful_delete_clears_the_marker(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """正常删除要把标记撤回，否则删一次项目就得 rebuild 一次。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    repo.set_vector_backend(conn, "numpy")

    result = admin.delete_project(conn, store, "甲项目")

    assert result["deleted"] is True
    assert repo.get_index_state(conn) == repo.INDEX_READY
    assert store.count() == 0


def test_project_marker_says_which_project_and_how_to_fix() -> None:
    """标记文案要能让人照着做，不能只说"有东西没对齐"。"""
    text = consistency.pending_message("unsynced:project:甲项目")

    assert "甲项目" in text
    assert "rebuild" in text
    assert "检索一律拒绝" in text
