"""清空整库与删单个文件：两条破坏性入口的共同要求是「要么全做完，要么一个字节都没动」。

这里刻意不复用 `ingest` 的断言方式，而是围着**失败路径**写：

- 清空时后端删不掉 → SQLite 里项目、文档、片段、向量归档必须原样都在（否则
  `rebuild` 失去恢复源，"清空失败"就变成了"数据被悄悄扔掉"）；
- 后端说删完了、条数却没归零 → 同样不许往下删；
- 删单个文件时 `doc_id` 属于别的项目 → 404，且两份数据都不能少一块。

顺便钉住"清空之后还能不能重新开始"：模型/维度声明要清掉、索引状态要回到 ready，
否则下次换个模型入库会被"配置与索引不一致"挡住，而这个不一致其实已经不存在了。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag4prefa import admin, consistency, repo
from rag4prefa.adapters.llm import LLM
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import create_app
from rag4prefa.config import Settings
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services
from tests.conftest import FakeEmbedder

PROJECT = "测试项目"


def _store(conn: sqlite3.Connection) -> NumpyStore:
    return NumpyStore(conn, FakeEmbedder.dim)


def _ingest(conn: sqlite3.Connection, store: NumpyStore, settings: Settings, path: Path):
    return ingest_file(
        conn, store, FakeEmbedder(), settings, path=path, project_name=PROJECT, doc_year=2024
    )


class _StubStore:
    """只实现 clear/count：用来注入「删不掉」「条数没归零」两种失败。

    不实现 delete/upsert：本文件只关心清空路径，多实现一个方法就可能掩盖
    "清空其实绕过了某个后端能力"这类问题。
    """

    name = "numpy"
    identity = None
    dim = FakeEmbedder.dim

    def __init__(self, *, raises: Exception | None = None, remain: int = 0) -> None:
        self._raises = raises
        self._remain = remain
        self.cleared = 0

    def clear(self) -> None:
        if self._raises:
            raise self._raises
        self.cleared += 1

    def count(self) -> int:
        return self._remain


def _facts(conn: sqlite3.Connection) -> dict[str, int]:
    return repo.data_counts(conn)


# ── 清空整库 ────────────────────────────────────────────────


def test_clear_all_empties_facts_vectors_and_declarations(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    outcome = _ingest(conn, store, settings, docx_file)
    assert outcome.chunks > 0

    result = admin.clear_all(conn, store)

    assert result["cleared"] is True
    assert result["before"]["documents"] == 1
    assert result["after"] == {"projects": 0, "documents": 0, "chunks": 0, "vectors": 0, "jobs": 0}
    assert _facts(conn) == {"projects": 0, "documents": 0, "chunks": 0, "vectors": 0, "jobs": 0}
    assert store.count() == 0, "向量索引没清干净：重启后检索会读到没有正文的向量"
    # 声明必须一起清掉：留着会让 health 报出一个不存在的坐标系，
    # 也让下次换个模型入库时被"配置与索引不一致"无故挡住
    assert repo.get_meta(conn, "embedding_model") is None
    assert repo.get_meta(conn, "embedding_dim") is None
    assert repo.get_index_state(conn) == repo.INDEX_READY
    # 清完必须是"能直接重新开始"的状态
    assert consistency.inspect(conn, store_count=0).blockers == []


def test_clear_all_is_idempotent_on_empty_db(conn: sqlite3.Connection) -> None:
    store = _store(conn)
    first = admin.clear_all(conn, store)
    second = admin.clear_all(conn, store)
    assert first["before"] == second["before"] == {
        "projects": 0,
        "documents": 0,
        "chunks": 0,
        "vectors": 0,
        "jobs": 0,
    }


def test_clear_all_keeps_embedding_cache(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """缓存不是事实：留着只省钱，删了要多花一次 embedding 的钱。

    `FakeEmbedder` 不写缓存（真实链路才写），所以这里直接落一条缓存行——
    断言的是"清空不碰这张表"，不是"入库会写缓存"。
    """
    store = _store(conn)
    _ingest(conn, store, settings, docx_file)
    conn.execute(
        "INSERT OR REPLACE INTO embed_cache (model, sha, dim, vec) VALUES (?, ?, ?, ?)",
        ("fake-embedder", "sha-1", FakeEmbedder.dim, b"\x00" * (4 * FakeEmbedder.dim)),
    )
    conn.commit()

    admin.clear_all(conn, store)

    assert conn.execute("SELECT count(*) FROM embed_cache").fetchone()[0] == 1


def test_clear_all_leaves_every_byte_when_backend_clear_fails(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    store = _store(conn)
    _ingest(conn, store, settings, docx_file)
    before = _facts(conn)

    with pytest.raises(consistency.IndexMismatch) as exc:
        admin.clear_all(conn, _StubStore(raises=RuntimeError("boom")))

    assert "没有任何改动" in str(exc.value)
    # 顺序的目的就是这个：先清索引失败时 SQLite 必须完好，才能重试
    assert _facts(conn) == before


def test_clear_all_refuses_when_backend_count_does_not_reach_zero(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """`clear()` 返回了不等于清干净了——条数才是判据。"""
    store = _store(conn)
    _ingest(conn, store, settings, docx_file)
    before = _facts(conn)

    with pytest.raises(consistency.IndexMismatch) as exc:
        admin.clear_all(conn, _StubStore(remain=3))

    assert "还剩 3 条" in str(exc.value)
    assert _facts(conn) == before


# ── 删单个文件（HTTP 入口）────────────────────────────────────


@pytest.fixture
def api(conn: sqlite3.Connection, settings: Settings, docx_file: Path):
    store = _store(conn)
    _ingest(conn, store, settings, docx_file)
    services = Services(
        settings=settings,
        conn=conn,
        store=store,
        embedder=FakeEmbedder(),
        llm=LLM(settings),
        reranker=NoopReranker(),
    )
    with TestClient(create_app(services)) as client:
        yield client, services, store


def test_delete_document_route_removes_chunks_and_vectors(api) -> None:
    client, services, store = api
    doc = repo.list_documents(services.conn, PROJECT)[0]
    assert doc.chunk_count > 0

    resp = client.delete(f"/api/projects/{PROJECT}/documents/{doc.id}")

    assert resp.status_code == 200, resp.text
    assert resp.json()["filename"] == doc.filename
    assert repo.list_documents(services.conn, PROJECT) == []
    assert repo.chunk_count(services.conn) == 0
    assert store.count() == 0, "删除只删了事实没删向量：检索会命中没有正文的片段"
    # 删完仍是"能自证一致"的状态，不该给后面留一个要手工修的理由
    assert consistency.inspect(services.conn, store_count=0).blockers == []


def test_delete_document_route_rejects_a_doc_of_another_project(api) -> None:
    client, services, store = api
    doc = repo.list_documents(services.conn, PROJECT)[0]
    before = _facts(services.conn)

    resp = client.delete(f"/api/projects/另一个项目/documents/{doc.id}")

    assert resp.status_code == 404
    assert _facts(services.conn) == before, "项目名对不上却把文档删了，这是最难解释的事故"
    assert store.count() == before["vectors"]


def test_delete_document_route_is_404_on_second_call(api) -> None:
    client, services, _ = api
    doc = repo.list_documents(services.conn, PROJECT)[0]
    assert client.delete(f"/api/projects/{PROJECT}/documents/{doc.id}").status_code == 200
    assert client.delete(f"/api/projects/{PROJECT}/documents/{doc.id}").status_code == 404


# ── 清空整库（HTTP 入口）────────────────────────────────────


def test_clear_route_requires_the_exact_confirm_word(api) -> None:
    client, services, store = api
    before = _facts(services.conn)

    for payload in ({}, {"confirm": ""}, {"confirm": "清"}, {"confirm": "clear"}):
        resp = client.post("/api/admin/clear", json=payload)
        assert resp.status_code == 400, payload

    assert _facts(services.conn) == before, "确认词不对却动了数据"


def test_clear_route_clears_and_reports_the_impact(api) -> None:
    client, services, store = api

    resp = client.post("/api/admin/clear", json={"confirm": "清空"})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cleared"] is True
    assert body["before"]["documents"] == 1
    assert body["before"]["chunks"] > 0
    # 影响面与结果要分成两个字段：同一个字段里混着"删前"和"删后"的数，
    # 前端显示出来就是自相矛盾的（例如 2 个项目、0 个向量）
    assert body["after"] == {"projects": 0, "documents": 0, "chunks": 0, "vectors": 0, "jobs": 0}
    assert repo.list_projects(services.conn) == []
    assert store.count() == 0
    # 清空之后 health 要能自证"干净"，而不是留一堆需要人解释的红字
    health = client.get("/api/health").json()
    assert health["index"]["issues"] == []
    assert health["index"]["vectors"] == 0
