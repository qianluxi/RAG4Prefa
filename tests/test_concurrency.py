"""维护锁：重建、入库、后端切换之间必须互斥。

为什么这一组比"再造一个异常分支"重要：单次调用里的异常处理都能写对，但两个动作
同时跑时，"每一步都对"仍然可以拼出一个错的库——

- 重建在算 embedding（几分钟），期间来了一个新文档：重建用的是**旧快照**，
  算完写回去，条数可能和归档一样、id 却不一样；
- 一个上传任务刚触发 numpy → Milvus 迁移，另一个任务还握着旧的 numpy 对象继续写，
  根服务已经查 Milvus 了。

所以这里验证的不是"抛了哪个异常"，而是**两件事不能同时发生**，以及发生之后
归档、后端、根服务指针三者仍然指向同一批 id。
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from rag4prefa import concurrency, repo
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.concurrency import (
    MaintenanceBusy,
    maintenance_lock,
)
from rag4prefa.config import Settings
from rag4prefa.ingest import ingest_file, reindex
from rag4prefa.services import Services
from tests.conftest import FakeEmbedder


def _seed(conn: sqlite3.Connection, store: NumpyStore, settings: Settings, docx: Path) -> int:
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx, project_name="甲项目", doc_year=2024
    )
    assert outcome.status == "indexed"
    return store.count()


# ── 锁本身 ──────────────────────────────────────────────


def test_a_second_maintenance_action_is_refused_with_the_holder_named(
    conn: sqlite3.Connection,
) -> None:
    """抢不到锁要说清是谁占着——"另一个动作正在运行"等于没说。

    必须是**另一个线程**：同一线程的嵌套调用是合法的重入（见下一条测试）。
    """
    caught: list[BaseException] = []

    def rival() -> None:
        try:
            with maintenance_lock(conn, purpose="入库 方案.docx", timeout=0.0):
                pass
        except BaseException as exc:  # noqa: BLE001 - 线程里的异常要带回主线程
            caught.append(exc)

    with maintenance_lock(conn, purpose="全量重建", timeout=0.0):
        worker = threading.Thread(target=rival, daemon=True)
        worker.start()
        worker.join(10)

    assert len(caught) == 1
    assert isinstance(caught[0], MaintenanceBusy)
    message = str(caught[0])
    assert "全量重建" in message, "要能看出是在等哪个动作"
    assert "入库 方案.docx" in message


def test_the_lock_is_reentrant_for_the_same_thread(conn: sqlite3.Connection) -> None:
    """嵌套调用是常态（`ingest_batch`→`ingest_file`、`maybe_promote`→`rebuild_store`）。

    这条曾经真的把线程锁死：重入深度记在锁对象实例上，而每次调用都新建一个实例，
    第二层于是去抢自己线程已经占着的锁，timeout=None 时直接挂住。
    """
    with (
        maintenance_lock(conn, purpose="外层", timeout=0.0),
        maintenance_lock(conn, purpose="内层", timeout=0.0),
    ):
        assert concurrency.is_busy(conn) is None, "自己占着不算别人在跑"

    with maintenance_lock(conn, purpose="退出后可再取", timeout=0.0):
        pass


def test_the_lock_is_released_and_leaves_no_holder_behind(conn: sqlite3.Connection) -> None:
    path = concurrency.lock_path_for(conn)
    assert path is not None
    with maintenance_lock(conn, purpose="一次动作", timeout=0.0):
        assert path.exists()
        assert path.with_name(path.name + ".holder").exists()
    assert not path.with_name(path.name + ".holder").exists()
    assert concurrency.is_busy(conn) is None, "锁必须随动作结束释放，否则下次永远抢不到"


# ── 场景：重建算 embedding 期间来了新文档 ────────────────────


class BlockingEmbedder(FakeEmbedder):
    """第一次 embed 就卡住，直到测试放行——模拟"重建正在算向量"的那几分钟。"""

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.started.set()
        assert self.release.wait(20), "测试没有放行"
        return super().embed(texts)


def test_reindex_blocks_ingest_while_it_is_computing_vectors(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """重建在算向量时，新文档必须被拒绝——否则重建会拿着旧快照覆盖新数据。

    验收条件不是"入库返回 failed"，而是失败之后：没有留下半个项目、重建正常完成、
    归档与后端条数一致。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, store, settings, docx_file)

    started, release = threading.Event(), threading.Event()
    failures: list[BaseException] = []

    def run_reindex() -> None:
        try:
            reindex(conn, store, BlockingEmbedder(started, release), settings)
        except BaseException as exc:  # noqa: BLE001 - 线程里的异常要带回主线程
            failures.append(exc)

    worker = threading.Thread(target=run_reindex, daemon=True)
    worker.start()
    assert started.wait(20), "重建没进入 embedding 阶段"

    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="乙项目"
    )
    assert outcome.status == "failed"
    assert "全量" in (outcome.error or ""), "错误里要写出被谁挡住"
    assert repo.get_project(conn, "乙项目") is None, "被拒绝时不能留下空项目"

    release.set()
    worker.join(30)
    assert not failures, f"重建本身不该失败：{failures}"

    assert repo.chunk_count(conn) == total
    assert store.count() == total
    assert concurrency.is_busy(conn) is None

    # 放行之后同一份文档可以正常入库
    again = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="乙项目"
    )
    assert again.status == "indexed"


# ── 场景：后端切换期间另一个上传 ────────────────────────────


class BlockingMilvus:
    """迁移目标：rebuild 卡住，直到测试放行。不共享 SQLite，和真 Milvus 一样。"""

    name = "milvus"
    started = threading.Event()
    release = threading.Event()
    latest: BlockingMilvus | None = None

    def __init__(self, *args, **kwargs) -> None:
        self.dim = int(args[2]) if len(args) > 2 else 0
        self.notes: list[str] = []
        self._ids: set[str] = set()
        BlockingMilvus.latest = self

    def rebuild(self, items) -> None:  # type: ignore[override]
        BlockingMilvus.started.set()
        assert BlockingMilvus.release.wait(20), "测试没有放行"
        self._ids = {i.chunk_id for i in items}

    def count(self) -> int:
        return len(self._ids)

    def ids(self) -> set[str]:
        return set(self._ids)


def test_an_upload_during_the_backend_switch_is_refused_and_the_backend_stays_consistent(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """迁移搬的是归档快照；同一个窗口里再写一笔，新后端就落后于归档。"""
    BlockingMilvus.started.clear()
    BlockingMilvus.release.clear()

    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    services = Services(
        settings=settings.model_copy(update={"vector_backend": "auto"}),
        conn=conn,
        store=store,
        embedder=FakeEmbedder(),
        llm=None,
        reranker=NoopReranker(),
        fell_back=True,
    )
    monkeypatch.setattr("rag4prefa.services.MilvusStore", BlockingMilvus)

    promoted: list[bool] = []
    worker = threading.Thread(target=lambda: promoted.append(services.maybe_promote()), daemon=True)
    worker.start()
    assert BlockingMilvus.started.wait(20), "迁移没进入重建阶段"

    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="乙项目"
    )
    assert outcome.status == "failed"
    assert "迁移到 Milvus" in (outcome.error or "")

    BlockingMilvus.release.set()
    worker.join(30)
    assert promoted == [True]

    # 迁移成功之后，根服务、后端、归档必须是同一批 id
    assert services.store is BlockingMilvus.latest
    assert services.fell_back is False
    assert services.store.ids() == repo.archived_chunk_ids(conn)
    assert concurrency.is_busy(conn) is None


# ── 入口层：抢不到锁要给出"稍后重试"，不是 500 ──────────────


def test_upload_is_refused_with_a_readable_status_while_maintenance_runs(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """重建跑着的时候收下文件是坑：任务会在后台失败，用户十分钟后才看到 failed。"""
    from fastapi.testclient import TestClient

    from rag4prefa.api import create_app

    store = NumpyStore(conn, FakeEmbedder.dim)
    services = Services(
        settings=settings, conn=conn, store=store, embedder=FakeEmbedder(), llm=None,
        reranker=NoopReranker(),
        # 连接归测试夹具管：lifespan 结束时不能把它关掉，后面还要断言
        owns_conn=False,
    )
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with maintenance_lock(conn, purpose="全量重建", timeout=0.0):
            held.set()
            release.wait(10)

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert held.wait(10)

    try:
        with TestClient(create_app(services)) as client:
            resp = client.post(
                "/api/projects/甲项目/documents",
                files={"files": ("方案.docx", b"not-a-real-docx", "application/octet-stream")},
            )
    finally:
        release.set()
        worker.join(10)

    assert resp.status_code == 503, "不是服务器错误，是「现在不行」"
    assert "全量重建" in resp.json()["detail"]
    assert repo.get_project(conn, "甲项目") is None, "被拒绝时不该留下空项目"
