"""管理口令：三条不可撤销的入口（清空库 / 删项目 / 删文件）必须有第二道门。

在这之前它们**没有任何身份校验**——`confirm: "清空"` 只防手滑，不证明"这个人有权删"。
额度那一层早就收口了（不代付时匿名请求拿不到服务端 Key），"谁能删库"这层一直是空的。

四种组合都要钉住，少一种就会留下一条没人守的岔路：

| 部署 | 口令 | 期望 |
|---|---|---|
| 单机演示 `fallback=true` | 没配 | 放行（打开页面的人就是主人） |
| 各付各的 `fallback=false` | 没配 | **503**，宁可删不了，也不让访客清库 |
| 任意 | 配了 | 缺头/错头 403；对了才进业务逻辑 |
| 单机演示 + 显式配了口令 | 配了 | 同样校验（显式设了就该生效） |

另外钉住两件事：拒绝时**一个字节都不能动**；口令校验要在 `confirm` 之前——
先告诉调用方"你没权限"，而不是"你的确认词写错了"。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import create_app
from rag4prefa.config import Settings
from rag4prefa.credentials import HEADER_ADMIN_TOKEN
from rag4prefa.db import connect, init_db
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services
from tests.conftest import FakeEmbedder

PROJECT = "上海文件"


def _settings(tmp_path: Path, **over: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "data_dir": tmp_path,
        "embedding_api_key": "test",
        "llm_api_key": "llm-key",
        "embedding_model": "fake-embed",
        "embedding_dim": FakeEmbedder.dim,
        "vector_backend": "numpy",
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def _services(tmp_path: Path, **over: object) -> Services:
    settings = _settings(tmp_path, **over)
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)
    return Services(
        settings=settings,
        conn=conn,
        store=NumpyStore(conn, FakeEmbedder.dim),
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        llm=None,
        reranker=NoopReranker(),
    )


def _counts(conn: sqlite3.Connection) -> tuple[int, int]:
    docs = conn.execute("select count(*) from documents").fetchone()[0]
    chunks = conn.execute("select count(*) from chunks").fetchone()[0]
    return docs, chunks


def _ingest_one(services: Services, docx_file: Path) -> None:
    outcome = ingest_file(
        services.conn,
        services.store,
        services.embedder,  # type: ignore[arg-type]
        services.settings,
        path=docx_file,
        project_name=PROJECT,
    )
    assert outcome.status == "indexed", outcome.error


# ── 单机演示：不配就不校验 ──────────────────────────────


def test_demo_deployment_needs_no_token(tmp_path: Path) -> None:
    """`fallback=true` 且没配口令时不拦。

    那时能打开页面的人就是这台机器的主人，强行要求口令只会让本地删一个文件
    也要先改 `.env` 再重启——安全收益为零，摩擦是实的。
    """
    services = _services(tmp_path)
    try:
        with TestClient(create_app(services)) as client:
            resp = client.delete(f"/api/projects/{PROJECT}")
        # 项目本来就不存在：404 说明已经越过口令这一关
        assert resp.status_code == 404, resp.text
    finally:
        services.close()


# ── 各付各的部署：没配口令就停用 ────────────────────────


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda c: c.post("/api/admin/clear", json={"confirm": "清空"}), id="clear"),
        pytest.param(lambda c: c.delete(f"/api/projects/{PROJECT}"), id="delete-project"),
        pytest.param(
            lambda c: c.delete(f"/api/projects/{PROJECT}/documents/whatever"),
            id="delete-document",
        ),
    ],
)
def test_strict_deployment_refuses_without_token(tmp_path: Path, call) -> None:
    """没配口令的多人部署里，三条入口全部停用，并且要说清怎么配。

    不选"放行 + 打日志"：删库没有第二次机会，日志也不会在误删之后把数据还回来。
    """
    services = _services(tmp_path, allow_server_key_fallback=False)
    try:
        with TestClient(create_app(services)) as client:
            resp = call(client)
        assert resp.status_code == 503, resp.text
        detail = resp.json()["detail"]
        assert "R4P_ADMIN_TOKEN" in detail, "只说停用不说怎么开，等于让人来问"
        assert "不可撤销" in detail
    finally:
        services.close()


def test_refusal_changes_nothing(tmp_path: Path, docx_file: Path) -> None:
    """被拒之后库必须原样：清空是最危险的一条，不能在"检查权限"之前动手。"""
    services = _services(tmp_path, allow_server_key_fallback=False)
    try:
        _ingest_one(services, docx_file)
        before = _counts(services.conn)
        assert before[0] > 0

        with TestClient(create_app(services)) as client:
            resp = client.post("/api/admin/clear", json={"confirm": "清空"})
            # 断言必须留在 with 里面：退出 TestClient 会触发 lifespan 关停，
            # 连接随之关闭，之后连 count 都查不了
            assert resp.status_code == 503
            assert _counts(services.conn) == before
    finally:
        services.close()


# ── 配了口令：缺头、错头、对头 ──────────────────────────


def test_missing_or_wrong_token_is_403(tmp_path: Path) -> None:
    services = _services(
        tmp_path, allow_server_key_fallback=False, admin_token="s3cret-token"
    )
    try:
        with TestClient(create_app(services)) as client:
            missing = client.delete(f"/api/projects/{PROJECT}")
            wrong = client.delete(
                f"/api/projects/{PROJECT}", headers={HEADER_ADMIN_TOKEN: "s3cret-tokeN"}
            )
        assert missing.status_code == 403, missing.text
        assert wrong.status_code == 403, "只差一个字符也必须拒"
        assert "管理口令" in missing.json()["detail"]
    finally:
        services.close()


def test_missing_and_wrong_token_say_different_things(tmp_path: Path) -> None:
    """没带和带错要分开说，因为下一步动作不同。

    合成一句"管理口令不对"时，**根本没填过**的人会去核对自己填了什么（他什么都没填），
    最后把"页面上还没有这一栏"当成后端的 bug——2026-10-28 用户就是这么被拦住的。
    """
    services = _services(
        tmp_path, allow_server_key_fallback=False, admin_token="s3cret-token"
    )
    try:
        with TestClient(create_app(services)) as client:
            missing = client.post("/api/admin/clear", json={"confirm": "清空"})
            wrong = client.post(
                "/api/admin/clear",
                json={"confirm": "清空"},
                headers={HEADER_ADMIN_TOKEN: "nope"},
            )
        assert missing.status_code == wrong.status_code == 403
        missing_detail = missing.json()["detail"]
        wrong_detail = wrong.json()["detail"]
        assert "保存管理口令" in missing_detail, "没带时要说清去哪儿填、怎么保存"
        assert "不一致" in wrong_detail, "带错时要指向 .env 里那份值"
        assert missing_detail != wrong_detail
    finally:
        services.close()


def test_page_is_never_served_from_cache(tmp_path: Path) -> None:
    """改了前端却拿到缓存里的旧页面时，新控件不存在，用户会误判成后端拒绝。

    静态页就是这里的客户端，所以它必须每次都回来问一次（没改动仍是 304）。
    """
    services = _services(tmp_path)
    try:
        with TestClient(create_app(services)) as client:
            resp = client.get("/")
        assert resp.status_code == 200
        assert "no-cache" in resp.headers.get("cache-control", "")
    finally:
        services.close()


def test_correct_token_reaches_the_business_layer(tmp_path: Path) -> None:
    """口令对了就往下走：这里用"项目不存在→404"证明越过了口令这一关。"""
    services = _services(
        tmp_path, allow_server_key_fallback=False, admin_token="s3cret-token"
    )
    try:
        with TestClient(create_app(services)) as client:
            resp = client.delete(
                f"/api/projects/{PROJECT}", headers={HEADER_ADMIN_TOKEN: "s3cret-token"}
            )
        assert resp.status_code == 404, resp.text
    finally:
        services.close()


def test_correct_token_can_actually_clear(tmp_path: Path, docx_file: Path) -> None:
    """口令对、确认词对，就要真的能清——否则"修数据入口"变成了"谁都进不来"。"""
    services = _services(
        tmp_path, allow_server_key_fallback=False, admin_token="s3cret-token"
    )
    try:
        _ingest_one(services, docx_file)
        with TestClient(create_app(services)) as client:
            resp = client.post(
                "/api/admin/clear",
                json={"confirm": "清空"},
                headers={HEADER_ADMIN_TOKEN: "s3cret-token"},
            )
            assert resp.status_code == 200, resp.text
            assert _counts(services.conn) == (0, 0)
    finally:
        services.close()


def test_token_is_checked_before_the_confirm_word(tmp_path: Path) -> None:
    """没权限的人不该从 `400 确认词不对` 里读出"只要打对字就能删"。

    所以这里**故意用错的确认词**：守卫在前 → 403；确认词在前 → 400。
    用"清空"去测是分不出顺序的（两条路都回 403），这一点是变异测试逼出来的。
    """
    services = _services(
        tmp_path, allow_server_key_fallback=False, admin_token="s3cret-token"
    )
    try:
        with TestClient(create_app(services)) as client:
            denied = client.post("/api/admin/clear", json={"confirm": "打错字了"})
            after_gate = client.post(
                "/api/admin/clear",
                json={"confirm": "打错字了"},
                headers={HEADER_ADMIN_TOKEN: "s3cret-token"},
            )
        assert denied.status_code == 403, denied.text
        assert after_gate.status_code == 400, "过了守卫才轮到确认词"
    finally:
        services.close()


def test_explicit_token_is_enforced_even_in_demo(tmp_path: Path) -> None:
    """单机演示下显式配了口令，也要校验：配了却不生效比不配更让人迷惑。"""
    services = _services(tmp_path, allow_server_key_fallback=True, admin_token="s3cret-token")
    try:
        with TestClient(create_app(services)) as client:
            denied = client.delete(f"/api/projects/{PROJECT}")
            allowed = client.delete(
                f"/api/projects/{PROJECT}", headers={HEADER_ADMIN_TOKEN: "s3cret-token"}
            )
        assert denied.status_code == 403, denied.text
        assert allowed.status_code == 404, "口令对了就该走到业务层（项目不存在）"
    finally:
        services.close()
