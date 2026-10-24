"""按用户 Key 的端到端行为：真实 openai SDK + 本地假服务，不联网、不花钱。

重点不是"接口能返回 200"，而是：
- 请求头里的 Key 真的被拿去调了厂商接口（靠假服务记录 Authorization 来证明）；
- 错 Key 会被明确拒绝，且错误信息不回显 Key；
- 每个请求的视图不会把全局 SQLite 连接一起关掉。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from rag4prefa import credentials
from rag4prefa.adapters.embed import Embedder
from rag4prefa.adapters.llm import LLM
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import _explain_error, _redact, _with_model_hint, create_app
from rag4prefa.config import Settings
from rag4prefa.credentials import UserCredentials
from rag4prefa.db import connect, init_db
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services
from tests import mock_openai


@pytest.fixture
def api_env(tmp_path: Path) -> Iterator[tuple[Services, Settings]]:
    mock_openai.reset()
    server = mock_openai.serve()
    base = mock_openai.base_url(server)
    settings = Settings(
        data_dir=tmp_path,
        embedding_base_url=base,
        embedding_api_key="sk-server",
        embedding_model="mock-embed",
        embedding_dim=mock_openai.DIM,
        embedding_batch=4,
        llm_base_url=base,
        llm_api_key="sk-server",
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
        yield services, settings
    finally:
        services.close()
        server.shutdown()


def _user_headers(name: str, embedding: str = "", llm: str = "") -> dict[str, str]:
    headers = {"X-R4P-User": quote(name)}
    if embedding:
        headers["X-R4P-Embedding-Key"] = embedding
    if llm:
        headers["X-R4P-LLM-Key"] = llm
    return headers


def test_health_reports_server_source_without_headers(
    api_env: tuple[Services, Settings],
) -> None:
    services, _ = api_env
    with TestClient(create_app(services)) as client:
        data = client.get("/api/health").json()

    assert data["user"]["source"] == "server"
    assert data["user"]["label"] == "服务端公共 Key"
    assert data["models"]["embedding_model"] == "mock-embed"


def test_health_reflects_the_header_user(api_env: tuple[Services, Settings]) -> None:
    services, _ = api_env
    with TestClient(create_app(services)) as client:
        data = client.get("/api/health", headers=_user_headers("张三", "sk-a")).json()

    assert data["user"]["source"] == "user", "请求头没被认到，前端会以为配了其实没用上"
    assert data["user"]["label"] == "张三"
    # 用户提供了 embedding Key，公共配置里缺 LLM Key 的情况不该再算"缺凭据"
    assert "R4P_EMBEDDING_API_KEY" not in data["credentials_missing"]


def test_verify_calls_provider_with_the_user_key(
    api_env: tuple[Services, Settings],
) -> None:
    services, _ = api_env
    with TestClient(create_app(services)) as client:
        resp = client.post(
            "/api/credentials/verify",
            json={
                "username": "alice",
                "embedding_key": "sk-alice-embed",
                "llm_key": "sk-alice-llm",
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["embedding"]["ok"] is True
    assert data["embedding"]["dim"] == mock_openai.DIM
    assert data["llm"]["ok"] is True
    assert mock_openai.LAST_AUTH["/v1/embeddings"] == "Bearer sk-alice-embed"
    assert mock_openai.LAST_AUTH["/v1/chat/completions"] == "Bearer sk-alice-llm"


def test_verify_rejects_bad_key_without_echoing_it(
    api_env: tuple[Services, Settings],
) -> None:
    services, _ = api_env
    with TestClient(create_app(services)) as client:
        resp = client.post(
            "/api/credentials/verify",
            json={"username": "bob", "embedding_key": "bad-key-123456", "llm_key": ""},
        )

    assert resp.status_code == 200, "部分失败用 200 + 字段说明，别丢掉另一项的结果"
    data = resp.json()
    assert data["embedding"]["ok"] is False
    assert data["embedding"]["source"] == "user"
    assert "bad-key-123456" not in resp.text, "错误信息回显 Key 等于把它写进浏览器日志"
    # LLM 栏留空会回落到服务端 Key：既不该报失败，也要标清来源，
    # 否则用户会以为刚才验的是自己的额度
    assert data["llm"]["ok"] is True
    assert data["llm"]["source"] == "server"


def test_search_uses_the_user_key_not_the_server_key(
    api_env: tuple[Services, Settings], docx_file: Path
) -> None:
    services, settings = api_env
    ingest_file(
        services.conn,
        services.store,
        services.require_embedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
    )

    with TestClient(create_app(services)) as client:
        resp = client.post(
            "/api/search",
            json={"question": "陶土砖饰面的施工工艺", "top_k": 3},
            headers=_user_headers("alice", "sk-alice-embed"),
        )

    assert resp.status_code == 200
    assert resp.json()["results"], "检索本身要正常返回"
    assert mock_openai.LAST_AUTH["/v1/embeddings"] == "Bearer sk-alice-embed", (
        "检索必须用请求头里的 Key，而不是服务端公共 Key"
    )


def test_chat_uses_the_user_llm_key(
    api_env: tuple[Services, Settings], docx_file: Path
) -> None:
    services, settings = api_env
    ingest_file(
        services.conn,
        services.store,
        services.require_embedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
    )

    with (
        TestClient(create_app(services)) as client,
        client.stream(
            "POST",
            "/api/chat",
            json={"question": "张拉控制要点有哪些？", "top_k": 3},
            headers=_user_headers("alice", "sk-alice-embed", "sk-alice-llm"),
        ) as resp,
    ):
        assert resp.status_code == 200
        body = "".join(resp.iter_text())

    # meta 里带 rerank_error 字段，所以不能笼统断言 "error" 不在正文里
    assert "event: error" not in body
    assert mock_openai.LAST_AUTH["/v1/chat/completions"] == "Bearer sk-alice-llm"


def test_view_does_not_close_the_shared_connection(
    api_env: tuple[Services, Settings],
) -> None:
    """视图和主 Services 共用一个 SQLite 连接，误关一次后面所有请求都会崩。"""
    services, _ = api_env
    view = services.for_credentials(UserCredentials("alice", "sk-x", ""))
    assert view is not services

    view.close()

    assert services.conn.execute("SELECT 1").fetchone()[0] == 1


def test_no_credentials_returns_the_same_instance(
    api_env: tuple[Services, Settings],
) -> None:
    services, _ = api_env
    assert services.for_credentials(None) is services


def test_redact_hides_secrets() -> None:
    message = "AuthenticationError: key sk-abcdef123456 is invalid"
    assert "sk-abcdef123456" not in _redact(message, "sk-abcdef123456")
    # 太短的串不替换，否则正常文字会被误伤
    assert _redact("the test failed", "test") == "the test failed"


def test_missing_embedder_blocks_upload_before_creating_project(
    api_env: tuple[Services, Settings], docx_file: Path
) -> None:
    """没有可用 Key 时应提前 503，而不是落一个空项目让人以为导过了。"""
    services, _ = api_env
    services.embedder = None

    with TestClient(create_app(services)) as client:
        resp = client.post(
            "/api/projects/新项目/documents",
            files={"files": (docx_file.name, docx_file.read_bytes(), "application/octet-stream")},
        )

    assert resp.status_code == 503
    rows = sqlite3.connect(services.settings.db_path).execute(
        "SELECT COUNT(*) FROM projects"
    ).fetchone()[0]
    assert rows == 0, "被拒绝的上传不该在库里留下空项目"


def test_verify_blocks_a_token_from_another_platform(
    api_env: tuple[Services, Settings],
) -> None:
    """把魔搭令牌填给 embedding 时，应当在发请求之前就说清是哪一步错了。

    端点用 `.invalid` 结尾：即便预检失效，请求也只会 DNS 失败，不会真的打到厂商
    接口——离线测试不该产生外部调用，也不该依赖网络是否可达。
    """
    services, settings = api_env
    services.settings = settings.model_copy(
        update={"embedding_base_url": "https://api.siliconflow.cn.invalid/v1"}
    )

    with TestClient(create_app(services)) as client:
        data = client.post(
            "/api/credentials/verify",
            json={"username": "alice", "embedding_key": "ms-alice-token", "llm_key": ""},
        ).json()

    assert data["embedding"]["ok"] is False
    assert data["embedding"]["source"] == "user"
    assert "魔搭" in data["embedding"]["error"]
    assert "硅基流动" in data["embedding"]["error"], "必须说清请求发往哪一家"


def test_missing_model_points_at_the_model_list() -> None:
    """模型下线时给的是 400，看起来像鉴权问题；提示必须把人引到模型名上。"""
    message = _with_model_hint(
        "BadRequestError: Error code: 400 - Model id : Qwen/X , has no provider supported",
        "https://api-inference.modelscope.cn/v1",
    )

    assert "v1/models" in message
    assert "api-inference.modelscope.cn" in message


def test_model_hint_stays_quiet_on_other_errors() -> None:
    """别的错误不能被加上模型提示，否则会指向错误的方向。"""
    assert _with_model_hint("RateLimitError: 429", "https://x/v1") == "RateLimitError: 429"


# ── 三类根因的识别与优先级 ─────────────────────────────────


def test_unverified_account_is_not_blamed_on_the_key() -> None:
    """魔搭对未验证账号报的是 400，但 Key 完全有效。

    真机上这条错误是：`For account security, please complete your profile via
    email verification.` 若不翻译，用户只会看到一串英文，然后回去重建 Key。
    """
    raw = (
        'BadRequestError: Error code: 400 - {"error":{"message":"For account security, '
        'please complete your profile via email verification. ModelScope Magicubes is '
        'only available to verified accounts. Please refer to '
        'https://modelscope.cn/my/settings/account"}}'
    )
    explained = credentials.account_blocked_hint(
        raw, "https://api-inference.modelscope.cn/v1"
    )

    assert "邮箱验证" in explained
    assert "modelscope.cn/my/settings/account" in explained, "必须给出处理入口"
    assert "已通过鉴权" in explained, "要说清 Key 本身没坏，否则用户会去重建 Key"
    # 链接夹在 JSON 里，尾部会粘上引号；不剥掉用户点不开
    assert "account'" not in explained and 'account"' not in explained


def test_account_hint_outranks_auth_hint() -> None:
    """账号问题不能被追加成「Key 未被接受」——两句话结论相反。"""
    raw = 'Error code: 403 - complete your profile via email verification'
    explained = _explain_error(raw, "https://api-inference.modelscope.cn/v1", "ms-abc")

    assert "邮箱验证" in explained
    assert "未被魔搭" not in explained, "Key 明明通过了鉴权，不能反过来说它被拒绝"


def test_quota_exhaustion_maps_to_top_up_not_to_key_rotation() -> None:
    """额度用尽与 Key 失效在日志里都像 401，但要做的事完全不同。"""
    for raw in (
        "Error code: 429 - insufficient_quota",
        "Error code: 400 - {\"code\":30001,\"message\":\"余额不足\"}",
    ):
        explained = credentials.account_blocked_hint(raw, "https://api.siliconflow.cn/v1")
        assert explained, raw


def test_transient_rate_limit_is_not_an_account_problem() -> None:
    """普通限流会自己恢复，说成账号问题会让人白跑一趟控制台。"""
    raw = "RateLimitError: Error code: 429 - rate limit reached for requests"
    assert credentials.account_blocked_hint(raw, "https://api.siliconflow.cn/v1") == ""
