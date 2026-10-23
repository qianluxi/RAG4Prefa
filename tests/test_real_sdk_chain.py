"""离线但"用真的"：真实 openai SDK + 真实 Embedder 缓存 + 真实 SSE。

FakeEmbedder 绕过了 HTTP、缓存和 SSE，所以它能全绿而线上第一秒就崩
（`embed_cache` 缺表就是这么漏掉的）。这里用一个本地假服务顶住网络边界，
把真实的适配器、SQLite 缓存、FastAPI 与 SSE 串起来跑一遍，不产生费用。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag4prefa.adapters.embed import Embedder
from rag4prefa.adapters.llm import LLM
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import create_app
from rag4prefa.config import Settings
from rag4prefa.db import connect, init_db
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services
from tests import mock_openai


@pytest.fixture
def services(tmp_path: Path) -> Iterator[tuple[Services, Path]]:
    server = mock_openai.serve()
    base = mock_openai.base_url(server)
    settings = Settings(
        data_dir=tmp_path,
        embedding_base_url=base,
        embedding_api_key="sk-mock",
        embedding_model="mock-embed",
        embedding_dim=mock_openai.DIM,
        embedding_batch=4,
        llm_base_url=base,
        llm_api_key="sk-mock",
        llm_model="mock-chat",
        vector_backend="numpy",
        chunk_max_tokens=120,
        chunk_min_tokens=30,
    )
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)
    services = Services(
        settings=settings,
        conn=conn,
        store=NumpyStore(conn, mock_openai.DIM),
        embedder=Embedder(settings, conn),
        llm=LLM(settings),
        reranker=NoopReranker(),
    )
    try:
        yield services, tmp_path
    finally:
        conn.close()
        server.shutdown()


def test_ingest_through_real_sdk_writes_cache(
    services: tuple[Services, Path], docx_file: Path
) -> None:
    svc, _ = services
    outcome = ingest_file(
        svc.conn,
        svc.store,
        svc.require_embedder(),
        svc.settings,
        path=docx_file,
        project_name="冒烟项目",
        doc_year=2026,
    )

    assert outcome.status == "indexed"
    assert svc.store.count() == outcome.chunks > 0

    cache_rows = svc.conn.execute("SELECT COUNT(*) AS c FROM embed_cache").fetchone()["c"]
    assert cache_rows == outcome.chunks, "真实 Embedder 必须把向量写进缓存"

    archived = svc.conn.execute("SELECT DISTINCT model FROM vectors").fetchall()
    assert {r["model"] for r in archived} == {"mock-embed"}, "归档不该被后端写成空模型"


def test_sse_chat_streams_tokens_and_citations(
    services: tuple[Services, Path], docx_file: Path
) -> None:
    svc, _ = services
    ingest_file(
        svc.conn,
        svc.store,
        svc.require_embedder(),
        svc.settings,
        path=docx_file,
        project_name="冒烟项目",
        doc_year=2026,
    )

    with (
        TestClient(create_app(svc)) as client,
        client.stream(
            "POST", "/api/chat", json={"question": "张拉控制要点有哪些？", "top_k": 3}
        ) as resp,
    ):
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    events = [ln.split(":", 1)[1].strip() for ln in body.splitlines() if ln.startswith("event:")]
    assert events[0] == "meta"
    assert events[-1] == "done"
    assert "token" in events
    assert "citations" in events
    assert "error" not in events
    assert "缓粘结预应力" in body, "大模型 token 流应原样出现在 SSE 里"


def test_health_reports_real_backend(
    services: tuple[Services, Path], docx_file: Path
) -> None:
    svc, _ = services
    ingest_file(
        svc.conn,
        svc.store,
        svc.require_embedder(),
        svc.settings,
        path=docx_file,
        project_name="冒烟项目",
    )
    health = svc.health()
    assert health["ok"] is True
    assert health["credentials_missing"] == []
    assert health["backend"]["vectors"] > 0
    assert health["index"]["model"] == "mock-embed"


def test_embed_cache_is_reused_across_calls(services: tuple[Services, Path]) -> None:
    svc, _ = services
    embedder = svc.require_embedder()
    embedder.embed(["同一段文本"])
    rows_after_first = _cache_count(svc.conn)

    embedder.embed(["同一段文本"])
    assert _cache_count(svc.conn) == rows_after_first, "重复文本不该再写一条缓存"


def _cache_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM embed_cache").fetchone()["c"]
