"""上传时「格式不支持」的文件必须被点名，不能安静消失。

现场问题：用户把 9 个文件放进同一个目录（5 个 pdf / 3 个 docx / 1 个 doc），
页面只多出 6 份，剩下 3 份在任何地方都不出现——不是删除、不是失败，是**从结果里
消失**。原来上传路由对不支持的后缀直接 `continue`，连任务消息里都不会留痕。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag4prefa.adapters.llm import LLM
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import create_app
from rag4prefa.config import Settings
from rag4prefa.services import Services
from tests.conftest import FakeEmbedder


@pytest.fixture
def client(conn: sqlite3.Connection, settings: Settings):
    services = Services(
        settings=settings,
        conn=conn,
        store=NumpyStore(conn, FakeEmbedder.dim),
        embedder=FakeEmbedder(),
        llm=LLM(settings),
        reranker=NoopReranker(),
    )
    with TestClient(create_app(services)) as c:
        yield c


def test_only_unsupported_files_is_400_with_the_reason(client: TestClient) -> None:
    resp = client.post(
        "/api/projects/甲项目/documents",
        files={"files": ("表格.doc", b"legacy-binary", "application/msword")},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "表格.doc" in detail, "只说'格式不支持'不够，用户不知道是哪一份"
    assert "docx" in detail, "要给出怎么办（另存为 .docx），否则用户只能猜"


def test_mixed_upload_reports_skipped_files(
    client: TestClient, docx_file: Path
) -> None:
    body = docx_file.read_bytes()
    resp = client.post(
        "/api/projects/甲项目/documents",
        files=[
            ("files", (docx_file.name, body, "application/octet-stream")),
            ("files", ("表格.doc", b"legacy-binary", "application/msword")),
        ],
    )

    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert data["files"] == [docx_file.name]
    # 关键断言：没进队列的那些要带着理由一起回来，而不是被 continue 吃掉
    assert [s["filename"] for s in data["skipped"]] == ["表格.doc"]
    assert data["skipped"][0]["reason"]
