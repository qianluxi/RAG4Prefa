"""自动迁移到 Milvus 只是**可选升级**：失败必须原地降级。

为什么要单独盯这条路径：它挂在"入库成功之后"（`api.py` 与 `cli.py` 都紧跟
`ingest_batch` 调用），而 api 那个 `try` 块把两步包在一起。迁移抛出的异常会被当成
入库失败写进任务状态——数据其实已经写好了，用户看到 failed，还会照着它重导一遍。

两条曾经的缺口：
1. `rebuild_store` 在目标后端重建成功之前就删归档里的孤儿行，重建一失败，numpy 的
   内存索引（它就是 `vectors` 表的缓存）与表脱节，条数对不上，检索被拒；
2. `maybe_promote` 只捕获 `ReindexError` / `RebuildError`，pymilvus 自己的异常
   （造集合、写向量、加载集合）会原样冒到调用方。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from rag4prefa import consistency, repo
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.credentials import UserCredentials
from rag4prefa.domain import JobState
from rag4prefa.ingest import ingest_batch, ingest_file
from rag4prefa.services import Services
from tests.conftest import FakeEmbedder


class BrokenTarget:
    """站在"另一个后端"的位置上：不与 SQLite 共享存储，重建时抛 SDK 原始异常。

    刻意不抛 `RebuildError`：这正是本轮要覆盖的那一类——以前它不会被捕获。
    """

    name = "broken-milvus"

    def __init__(self, *args, **kwargs) -> None:
        self.dim = int(args[2]) if len(args) > 2 else 0
        self.notes: list[str] = []

    def rebuild(self, items) -> None:  # type: ignore[override]
        raise RuntimeError("create_collection failed: code=1, message=lite is busy")

    def count(self) -> int:  # pragma: no cover - 走到这里说明调用顺序错了
        raise AssertionError("重建已经失败，不该再问条数")


class BrokenConstructor:
    """连集合都打不开：抛的是 pymilvus 自己的异常，而不是 `BackendUnavailable`。"""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError("lite file is locked by another process")


def _services(settings: Settings, conn: sqlite3.Connection, store: NumpyStore) -> Services:
    """构造一个"启动时回退到 numpy、正等着首次入库后迁回 Milvus"的服务对象。"""
    return Services(
        settings=settings.model_copy(update={"vector_backend": "auto"}),
        conn=conn,
        store=store,
        embedder=FakeEmbedder(),
        llm=None,
        reranker=NoopReranker(),
        fell_back=True,
    )


def _seed(conn: sqlite3.Connection, store: NumpyStore, settings: Settings, docx: Path) -> int:
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx, project_name="甲项目", doc_year=2024
    )
    assert outcome.status == "indexed"
    return store.count()


def test_promotion_failure_keeps_the_numpy_backend_usable(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """迁移失败只记 notes：后端不换、检索照常、异常不外溢。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, store, settings, docx_file)
    services = _services(settings, conn, store)
    monkeypatch.setattr("rag4prefa.services.MilvusStore", BrokenTarget)

    assert services.maybe_promote() is False

    assert services.store is store, "没迁移成功就不能把后端换成一个半成品"
    assert services.fell_back is True, "下次入库还要再试，不能把机会关掉"
    assert any("暂不迁移" in n and "create_collection" in n for n in services.notes)

    # 判据不是"没抛异常"，而是"检索链路真的还能用"
    consistency.ensure_queryable(
        conn, FakeEmbedder.model, store.dim, store_count=store.count()
    )
    assert store.count() == total
    assert store.search(FakeEmbedder().embed_one("悬挑"), limit=5)


def test_promotion_failure_does_not_delete_anything_from_the_archive(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """目标后端没成功之前，归档一个字节都不许动。

    归档里有一条孤儿向量时，旧实现先把它物理删掉、再重建索引；重建失败后调用方
    继续用原来的 numpy 对象，而它的内存索引里还留着那一行——条数一对不上，一致性
    检查就开始拒绝检索。一次"可选升级"把状态搞得更糟。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    from rag4prefa.adapters.vectors import VectorItem, to_archive_row

    orphan = VectorItem(
        "chunk-已删除", [0.5] * FakeEmbedder.dim, "甲项目", "", 2024, "fake-embedder"
    )
    repo.archive_vectors(conn, [to_archive_row(orphan)])
    before = repo.vector_stats(conn)
    services = _services(settings, conn, store)
    monkeypatch.setattr("rag4prefa.services.MilvusStore", BrokenTarget)

    assert services.maybe_promote() is False

    assert repo.vector_stats(conn) == before, "重建失败就不许删归档行"
    # 源后端重新按归档加载时，条数必须仍然对得上——这是"能继续用"的实际含义
    assert NumpyStore(conn, FakeEmbedder.dim).count() == before["count"]


def test_promotion_swallows_a_raw_sdk_error_while_opening_the_collection(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """构造阶段也可能抛非 `BackendUnavailable` 的异常（文件锁、版本不兼容）。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    services = _services(settings, conn, store)
    monkeypatch.setattr("rag4prefa.services.MilvusStore", BrokenConstructor)

    assert services.maybe_promote() is False
    assert services.store is store
    assert any("仍无法启用 Milvus" in n and "lite file is locked" in n for n in services.notes)


def test_a_failed_promotion_does_not_turn_an_indexed_job_into_a_failure(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """入库任务已经 done，可选的迁移不许把它改写成 failed。

    这就是 api.py 里 `ingest_batch` 与 `maybe_promote` 共用一个 try 的后果：
    迁移抛异常 → 任务 failed，而 chunk 与向量其实都写好了。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    job_id = "job-promote"
    repo.create_job(conn, job_id, "ingest", {"project": "甲项目"})
    ingest_batch(
        conn,
        store,
        FakeEmbedder(),
        settings,
        job_id=job_id,
        files=[docx_file],
        project_name="甲项目",
        doc_year=2024,
    )
    services = _services(settings, conn, store)
    monkeypatch.setattr("rag4prefa.services.MilvusStore", BrokenTarget)

    services.maybe_promote()  # 不抛异常＝本用例的被测行为

    assert repo.get_job(conn, job_id).state is JobState.done
    assert repo.chunk_count(conn) > 0, "数据确实已经写进去了"


class OkTarget:
    """迁移目标：写入成功，只记下 id 集合，用来核对"根服务、后端、归档"三者。"""

    name = "milvus"

    def __init__(self, *args, **kwargs) -> None:
        self.dim = int(args[2]) if len(args) > 2 else 0
        self.notes: list[str] = []
        self._ids: set[str] = set()

    def rebuild(self, items) -> None:  # type: ignore[override]
        self._ids = {i.chunk_id for i in items}

    def count(self) -> int:
        return len(self._ids)

    def ids(self) -> set[str]:
        return set(self._ids)


def test_promotion_through_a_request_view_updates_the_root_service(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """迁移只能由根服务拥有。

    请求级视图是 `dataclasses.replace()` 出来的临时对象，一次请求结束就扔。在它上面
    迁移会出现一种很别扭的"成功"：Milvus 建好了、`meta` 也更新了，根服务却还在用
    numpy，而根服务的 `fell_back` 仍是 True——下一个请求会再迁一遍。

    触发条件也不苛刻：只带 `X-R4P-LLM-Model` 的请求同样会走到视图分支。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    services = _services(settings, conn, store)
    view = services.for_credentials(UserCredentials("张三", llm_model="mock-chat"))
    assert view is not services
    monkeypatch.setattr("rag4prefa.services.MilvusStore", OkTarget)

    assert view.maybe_promote() is True

    assert services.store is not store, "根服务必须换成新后端，否则下一个请求还在查 numpy"
    assert services.fell_back is False
    assert services.owner is services, "根服务的 owner 就是自己"
    assert view.owner is services, "视图必须指回根服务"
    assert any("迁移到 Milvus" in n for n in services.notes), "/health 读的是根服务的 notes"
    assert services.store.ids() == repo.archived_chunk_ids(conn)
