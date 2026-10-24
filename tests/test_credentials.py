"""凭据解析与池化。

这里断言的不是"程序跑通了"，而是几条业务不变量：
- 没带 Key 头 = 用服务端公共配置（旧客户端不受影响）；
- 用户只能换 Key，绝不能顺手换掉模型（否则同一索引会混进两套向量）；
- 池子按 Key 复用、超额关闭，不泄漏连接。
"""

from __future__ import annotations

import sqlite3
from urllib.parse import quote

import pytest

from rag4prefa import credentials
from rag4prefa.config import Settings
from rag4prefa.credentials import CredentialPool, UserCredentials, from_headers


def _headers(**kwargs: str) -> dict[str, str]:
    mapping = {
        "user": "X-R4P-User",
        "embedding": "X-R4P-Embedding-Key",
        "llm": "X-R4P-LLM-Key",
    }
    return {mapping[k]: v for k, v in kwargs.items()}


def test_no_key_header_falls_back_to_server_config() -> None:
    """只有用户名不算凭据：否则每个请求都会白白新建一套客户端。"""
    assert from_headers({}) is None
    assert from_headers(_headers(user="alice")) is None


def test_keys_are_parsed_and_trimmed() -> None:
    creds = from_headers(
        _headers(user="alice", embedding=" sk-embed ", llm="sk-llm")
    )
    assert creds is not None
    assert creds.username == "alice"
    # 带空白会变成 "Bearer  sk-embed "，服务端直接 401，是最难查的一类错
    assert creds.embedding_key == "sk-embed"
    assert creds.llm_key == "sk-llm"


def test_single_key_is_allowed() -> None:
    """只带 embedding Key 是合法用法：入库花自己的额度，问答用公共 Key。"""
    creds = from_headers(_headers(embedding="sk-only"))
    assert creds is not None
    assert creds.username == "anonymous"
    assert creds.llm_key == ""


def test_chinese_username_survives_header_encoding() -> None:
    """HTTP 头只能是 ASCII；中文名靠 encodeURIComponent 传递，服务端要解回来。"""
    creds = from_headers(_headers(user=quote("张三"), embedding="sk-x"))
    assert creds is not None
    assert creds.username == "张三"


def test_fingerprint_separates_keys_without_leaking_them() -> None:
    a = UserCredentials("u", "sk-1", "sk-2")
    b = UserCredentials("u", "sk-1", "sk-3")
    assert a.fingerprint != b.fingerprint, "换了 LLM Key 就该是另一套客户端"
    assert "sk-1" not in a.fingerprint, "缓存键里不该出现明文 Key"


def test_settings_for_only_swaps_keys(settings: Settings, conn: sqlite3.Connection) -> None:
    pool = CredentialPool(settings, conn)
    local = pool.settings_for(UserCredentials("alice", "sk-user", ""))

    assert local.embedding_api_key == "sk-user"
    assert local.llm_api_key == settings.llm_api_key, "没提供的 Key 应沿用服务端"
    # 这条最关键：用户能换额度，不能换模型/维度，否则跨用户的向量空间会混
    assert local.embedding_model == settings.embedding_model
    assert local.embedding_dim == settings.embedding_dim
    assert local.embedding_base_url == settings.embedding_base_url


class _FakeClient:
    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(credentials, "Embedder", _FakeClient)
    monkeypatch.setattr(credentials, "LLM", _FakeClient)


def test_pool_reuses_clients_by_key(
    settings: Settings, conn: sqlite3.Connection, fake_clients: None
) -> None:
    pool = CredentialPool(settings, conn)
    first = pool.acquire(UserCredentials("alice", "sk-1", "sk-1"))
    again = pool.acquire(UserCredentials("alice", "sk-1", "sk-1"))

    assert first[0] is again[0] and first[1] is again[1], "同一个 Key 不该重复建客户端"
    assert len(pool) == 1


def test_pool_evicts_and_closes_oldest(
    settings: Settings, conn: sqlite3.Connection, fake_clients: None
) -> None:
    pool = CredentialPool(settings, conn, limit=2)
    oldest, _ = pool.acquire(UserCredentials("a", "sk-1", "sk-1"))
    pool.acquire(UserCredentials("b", "sk-2", "sk-2"))
    pool.acquire(UserCredentials("c", "sk-3", "sk-3"))

    assert len(pool) == 2
    assert oldest.closed, "淘汰的客户端必须关掉，否则 httpx 连接会一直挂着"


def test_pool_close_releases_everything(
    settings: Settings, conn: sqlite3.Connection, fake_clients: None
) -> None:
    pool = CredentialPool(settings, conn)
    one, _ = pool.acquire(UserCredentials("a", "sk-1", "sk-1"))
    two, _ = pool.acquire(UserCredentials("b", "sk-2", "sk-2"))

    pool.close()

    assert one.closed and two.closed
    assert len(pool) == 0


# ── Key 与端点是否配对 ──────────────────────────────────────


def test_modelscope_token_on_siliconflow_is_caught_before_request() -> None:
    """把魔搭令牌填进 embedding 一栏，是这套界面最容易犯的错。

    靠厂商返回的 `401 Token is invalid` 排查，会把人引去核对令牌有没有复制错；
    而真正的问题是两个输入框长得一样。这一层必须在发请求之前就说清楚。
    """
    hint = credentials.platform_mismatch("ms-abc123", "https://api.siliconflow.cn/v1")

    assert hint
    assert "魔搭" in hint and "硅基流动" in hint


def test_matching_platform_passes_silently() -> None:
    """配对正确时不能多嘴：假阳性会让正常配置看起来有毛病。"""
    assert credentials.platform_mismatch("ms-abc", "https://api-inference.modelscope.cn/v1") is None
    assert credentials.platform_mismatch("sk-abc", "https://api.siliconflow.cn/v1") is None


def test_unknown_endpoint_is_not_judged() -> None:
    """认不出的端点（自建、代理、测试假服务）一律不判断。"""
    assert credentials.platform_mismatch("ms-abc", "http://127.0.0.1:8123/v1") is None
    assert credentials.platform_mismatch("", "https://api.siliconflow.cn/v1") is None


def test_sk_keys_cannot_be_told_apart() -> None:
    """硅基流动、百炼、OpenAI 都发 sk-，这层判断不假装能区分。"""
    assert credentials.platform_mismatch("sk-bailian", "https://api.siliconflow.cn/v1") is None


def test_auth_failure_hint_names_the_platform() -> None:
    assert "硅基流动" in credentials.auth_failure_hint("https://api.siliconflow.cn/v1")
    assert credentials.auth_failure_hint("http://127.0.0.1:8123/v1") == ""
