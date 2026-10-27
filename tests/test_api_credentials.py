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

from rag4prefa import credentials, repo
from rag4prefa.adapters.embed import Embedder
from rag4prefa.adapters.llm import LLM
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import _explain_error, _redact, _with_model_hint, create_app
from rag4prefa.config import Settings
from rag4prefa.credentials import LLM_CHOICES, UserCredentials
from rag4prefa.db import connect, init_db
from rag4prefa.ingest import ingest_file
from rag4prefa.processcontrol import APP_MARKER
from rag4prefa.services import Services
from tests import mock_openai


@pytest.fixture
def api_env(tmp_path: Path) -> Iterator[tuple[Services, Settings]]:
    mock_openai.reset()
    server = mock_openai.serve()
    base = mock_openai.base_url(server)
    settings = Settings(
        # 不读本机 .env：这些用例自带全套参数，读了就"本机改配置、测试变红"
        _env_file=None,
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


def test_health_announces_identity_for_stop_command(
    api_env: tuple[Services, Settings],
) -> None:
    """`rag4prefa stop` 靠这一行认出"端口上是本程序"。

    没有它就只能按端口号杀进程，误杀别人家的服务只需要一次巧合。
    """
    services, _ = api_env
    with TestClient(create_app(services)) as client:
        assert client.get("/api/health").json()["app"] == APP_MARKER


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


def test_verify_falls_back_to_server_key_not_the_current_user(
    api_env: tuple[Services, Settings],
) -> None:
    """留空的字段必须回落到**服务端**公共 Key，不能是"当前登录用户"的那把。

    场景是真实会发生的：Alice 已登录且 Key 可用，现在给 Bob 只填 embedding、LLM 留空。
    前端的通用请求会顺手带上 Alice 的请求头，如果 verify 拿这个"按用户视图"当回退
    来源，验的其实是 Alice 的 Key，却对外标成 `source=server`。Bob 保存之后真正发
    请求走的是服务端 Key——那时才失败，而校验早就绿了。
    """
    services, _ = api_env
    with TestClient(create_app(services)) as client:
        mock_openai.reset()
        data = client.post(
            "/api/credentials/verify",
            # 关键：请求头里放一把**可用**的用户 Key，与"留空"形成对照
            headers=_user_headers("alice", embedding="sk-alice-embed", llm="sk-alice-llm"),
            json={"username": "bob", "embedding_key": "", "llm_key": ""},
        ).json()

    assert data["llm"]["ok"] is True
    assert data["llm"]["source"] == "server"
    assert data["embedding"]["source"] == "server"
    assert data["llm"]["model"] == "mock-chat", "回退的模型也该是服务端配的那个"
    # 这是全部要点：真正发出去的必须是服务端那把 sk-server，而不是 Alice 的
    assert mock_openai.LAST_AUTH["/v1/chat/completions"] == "Bearer sk-server"
    assert mock_openai.LAST_AUTH["/v1/embeddings"] == "Bearer sk-server"


def test_verify_accepts_model_switch_with_server_keys(
    api_env: tuple[Services, Settings], docx_file: Path
) -> None:
    """只换对话模型、两个 Key 都留空：界面允许保存，接口也必须能验通。

    后端早就支持只带模型头（`from_headers` 因此要连模型一起判空），但前端曾经强制
    "至少填一个 Key"，于是"用公共额度换个模型"这件事在浏览器里根本做不到。
    """
    services, settings = api_env
    ingest_file(
        services.conn,
        services.store,
        services.require_embedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
    )

    headers = {"X-R4P-User": "carol", "X-R4P-LLM-Model": LLM_CHOICES[1].id}
    with TestClient(create_app(services)) as client:
        mock_openai.reset()
        data = client.post(
            "/api/credentials/verify",
            json={"username": "carol", "embedding_key": "", "llm_key": ""},
        ).json()
        assert data["llm"]["ok"] is True

        # 这是前端保存后真正发出的请求形态：没有 Key，只有一个模型头
        mock_openai.reset()
        with client.stream(
            "POST", "/api/chat", json={"question": "张拉控制要点有哪些？", "top_k": 3},
            headers=headers,
        ) as resp:
            body = "".join(resp.iter_text())

    assert "event: error" not in body
    assert mock_openai.LAST_MODEL["/v1/chat/completions"] == LLM_CHOICES[1].id, (
        "只带模型头时，真正调用的必须是选中的那个模型"
    )
    assert mock_openai.LAST_AUTH["/v1/chat/completions"] == f"Bearer {settings.llm_api_key}", (
        "没填 Key 就该用服务端公共额度，而不是报错"
    )


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


# ── 对话模型可选 ───────────────────────────────────────────


def test_llm_model_list_is_served_for_the_dropdown(
    api_env: tuple[Services, Settings],
) -> None:
    """下拉内容由服务端给，前端不硬编码——加模型只改一处。"""
    services, _ = api_env
    with TestClient(create_app(services)) as client:
        data = client.get("/api/llm-models").json()

    ids = [option["id"] for option in data["options"]]
    assert all(option.id in ids for option in LLM_CHOICES)
    # 测试 fixture 的模型名不在内置清单里，仍必须出现——否则前端一项都选不中
    assert data["current"] in ids
    selected = [option for option in data["options"] if option["selected"]]
    assert len(selected) == 1 and selected[0]["id"] == data["current"]
    # 每一项都要有可读说明，否则用户只能凭名字猜
    assert all(option["note"] for option in data["options"])
    assert data["embedding_fixed"] is True, "要能告诉前端 embedding 为何不可选"


def test_verify_calls_the_chosen_llm_model(
    api_env: tuple[Services, Settings],
) -> None:
    """选了模型就必须真的用它去调，而不是界面显示切换、背后还在跑默认模型。"""
    services, _ = api_env
    chosen = LLM_CHOICES[1].id
    with TestClient(create_app(services)) as client:
        data = client.post(
            "/api/credentials/verify",
            json={"username": "alice", "llm_key": "sk-alice-llm", "llm_model": chosen},
        ).json()

    assert data["llm"]["ok"] is True
    assert data["llm"]["model"] == chosen
    assert mock_openai.LAST_MODEL["/v1/chat/completions"] == chosen


def test_unknown_llm_model_falls_back_instead_of_failing(
    api_env: tuple[Services, Settings],
) -> None:
    """旧浏览器里可能还存着已下线的模型名，不能因此让校验整体失败。"""
    services, settings = api_env
    with TestClient(create_app(services)) as client:
        data = client.post(
            "/api/credentials/verify",
            json={"username": "a", "llm_key": "sk-a", "llm_model": "some/retired-model"},
        ).json()

    assert data["llm"]["ok"] is True
    assert data["llm"]["model"] == settings.llm_model


def test_choosing_an_llm_model_does_not_touch_embedding(
    api_env: tuple[Services, Settings],
) -> None:
    """embedding 决定向量空间，绝不能跟着用户的选择变。"""
    services, settings = api_env
    with TestClient(create_app(services)) as client:
        data = client.post(
            "/api/credentials/verify",
            json={
                "username": "a",
                "embedding_key": "sk-a",
                "llm_model": LLM_CHOICES[-1].id,
            },
        ).json()

    assert data["server"]["embedding_model"] == settings.embedding_model
    assert data["server"]["embedding_base_url"] == settings.embedding_base_url


# ---------------------------------------------------------------- 服务端 Key 回落开关


def _strict_env(tmp_path: Path, allow_fallback: bool) -> tuple[Services, Settings]:
    """与 api_env 相同的假服务，但可控制是否允许匿名请求回落到服务端公共 Key。"""
    from rag4prefa.adapters.embed import Embedder
    from rag4prefa.adapters.llm import LLM

    mock_openai.reset()
    server = mock_openai.serve()
    base = mock_openai.base_url(server)
    settings = Settings(
        # 不读本机 .env：这些用例自带全套参数，读了就"本机改配置、测试变红"
        _env_file=None,
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
        allow_server_key_fallback=allow_fallback,
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
    return services, settings


def test_anonymous_request_is_rejected_when_fallback_is_off(
    tmp_path: Path, docx_file: Path
) -> None:
    """关掉回落时，不带 Key 的请求不能白嫖服务端公共额度，应直接 401。

    这不是"厂商会报 401 所以在这里拦也一样"——等厂商报的时候额度已经烧掉一部分，
    而且用户看到的报错是 "Token is invalid"，会去反复核对那把**他根本没填过**的 Key。
    """
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        ingest_file(
            services.conn,
            services.store,
            services.require_embedder(),
            settings,
            path=docx_file,
            project_name="甲项目",
        )
        with TestClient(create_app(services)) as client:
            resp = client.post("/api/search", json={"question": "要点", "top_k": 3})
            assert resp.status_code == 401
            assert "X-R4P-Embedding-Key" in resp.json()["detail"]
    finally:
        services.close()


def test_anonymous_chat_is_rejected_when_fallback_is_off(
    tmp_path: Path, docx_file: Path
) -> None:
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        ingest_file(
            services.conn,
            services.store,
            services.require_embedder(),
            settings,
            path=docx_file,
            project_name="甲项目",
        )
        with TestClient(create_app(services)) as client, client.stream(
            "POST", "/api/chat", json={"question": "要点", "top_k": 3}
        ) as resp:
            assert resp.status_code == 401
    finally:
        services.close()


def test_user_key_still_works_when_fallback_is_off(
    tmp_path: Path, docx_file: Path
) -> None:
    """关掉回落只是拒绝匿名，不是拒绝一切。带上自己的 Key 仍然该放行。

    判据必须落在"请求头真的被拿去调了厂商"上，而不是只看 200——否则一个
    把所有请求都放行的实现也能通过这条测试。
    """
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        ingest_file(
            services.conn,
            services.store,
            services.require_embedder(),
            settings,
            path=docx_file,
            project_name="甲项目",
        )
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            resp = client.post(
                "/api/search",
                json={"question": "要点", "top_k": 3},
                headers=_user_headers("alice", embedding="sk-alice-embed"),
            )
        assert resp.status_code == 200
        assert mock_openai.LAST_AUTH["/v1/embeddings"] == "Bearer sk-alice-embed"
    finally:
        services.close()


def test_anonymous_request_still_allowed_by_default(
    tmp_path: Path, docx_file: Path
) -> None:
    """默认（回落开）时匿名请求照常走服务端公共 Key——CLI / curl 行为不能变。"""
    services, settings = _strict_env(tmp_path, allow_fallback=True)
    try:
        ingest_file(
            services.conn,
            services.store,
            services.require_embedder(),
            settings,
            path=docx_file,
            project_name="甲项目",
        )
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            resp = client.post("/api/search", json={"question": "要点", "top_k": 3})
        assert resp.status_code == 200
        assert mock_openai.LAST_AUTH["/v1/embeddings"] == "Bearer sk-server"
    finally:
        services.close()


def test_health_reports_whether_fallback_is_on(
    tmp_path: Path,
) -> None:
    """前端要据这个字段把"用自己的 Key"从可选改成必填，而不是让用户点一次失败一次。"""
    services, _ = _strict_env(tmp_path, allow_fallback=False)
    try:
        with TestClient(create_app(services)) as client:
            data = client.get("/api/health").json()
        assert data["server_key_fallback"] is False
    finally:
        services.close()


# ---------------------------------------------- 服务端 Key 回落：按角色逐个校验
#
# 2026-10-27 复审 P1-1：`svc_billable()` 原来只判 `creds is None`，而
# `from_headers()` 只要带任意一个头就返回非空凭据，`settings_for()` 又会把缺的那把
# Key 静默补成服务端那两把。下面四类请求因此在"关掉回落"的部署上照样花服务端的钱。
# 每条都断言"厂商没收到请求 / 收到的是用户自己的 Key"，而不是只看状态码——
# 只看状态码的话，一个把请求放行、再让厂商报错的实现也能通过。


def _ingest_one(services: Services, settings: Settings, docx_file: Path) -> None:
    ingest_file(
        services.conn,
        services.store,
        services.require_embedder(),
        settings,
        path=docx_file,
        project_name="甲项目",
    )


def test_model_header_only_cannot_chat_when_fallback_is_off(
    tmp_path: Path, docx_file: Path
) -> None:
    """只传模型选择头（不传任何 Key）也是匿名请求，不能白嫖两把服务端 Key。"""
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        _ingest_one(services, settings, docx_file)
        mock_openai.reset()
        with TestClient(create_app(services)) as client, client.stream(
            "POST",
            "/api/chat",
            json={"question": "要点", "top_k": 3},
            headers={"X-R4P-LLM-Model": LLM_CHOICES[1].id},
        ) as resp:
            assert resp.status_code == 401
            resp.read()  # 流式响应必须先读完才能取 body
            detail = resp.json()["detail"]
        assert "X-R4P-Embedding-Key" in detail
        assert "X-R4P-LLM-Key" in detail
        assert mock_openai.LAST_AUTH == {}, "被拒绝的请求不该打到厂商"
    finally:
        services.close()


def test_embedding_key_alone_cannot_chat_when_fallback_is_off(
    tmp_path: Path, docx_file: Path
) -> None:
    """只填 embedding Key 也能出答案，就等于用服务端 LLM Key 生成内容。

    检索照旧该放行：那个接口只用 embedding 那一把，用户已经付了。
    """
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        _ingest_one(services, settings, docx_file)
        headers = _user_headers("alice", embedding="sk-alice-embed")

        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            ok = client.post("/api/search", json={"question": "要点", "top_k": 3}, headers=headers)
            assert ok.status_code == 200
            assert mock_openai.LAST_AUTH["/v1/embeddings"] == "Bearer sk-alice-embed"

            mock_openai.reset()
            with client.stream(
                "POST", "/api/chat", json={"question": "要点", "top_k": 3}, headers=headers
            ) as resp:
                assert resp.status_code == 401
                resp.read()
                detail = resp.json()["detail"]
        assert "X-R4P-LLM-Key" in detail
        assert "X-R4P-Embedding-Key" not in detail, "embedding Key 已经提供了，不该再要"
        assert "/v1/chat/completions" not in mock_openai.LAST_AUTH
    finally:
        services.close()


def test_llm_key_alone_cannot_search_when_fallback_is_off(
    tmp_path: Path, docx_file: Path
) -> None:
    """反过来也一样：只填 LLM Key 不能让检索引擎用服务端的 embedding Key 算向量。"""
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        _ingest_one(services, settings, docx_file)
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            resp = client.post(
                "/api/search",
                json={"question": "要点", "top_k": 3},
                headers=_user_headers("alice", llm="sk-alice-llm"),
            )
        assert resp.status_code == 401
        assert "X-R4P-Embedding-Key" in resp.json()["detail"]
        assert mock_openai.LAST_AUTH == {}, "被拒绝的检索不该算一次 embedding"
    finally:
        services.close()


def test_upload_requires_the_users_own_embedding_key(
    tmp_path: Path, docx_file: Path
) -> None:
    """入库同样只花 embedding 额度：只带 LLM Key 不能建项目、不能落盘。"""
    services, _ = _strict_env(tmp_path, allow_fallback=False)
    try:
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            resp = client.post(
                "/api/projects/甲项目/documents",
                headers=_user_headers("alice", llm="sk-alice-llm"),
                files={
                    "files": (
                        docx_file.name,
                        docx_file.read_bytes(),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
            assert resp.status_code == 401
            assert "X-R4P-Embedding-Key" in resp.json()["detail"]
            # 拒绝发生在落盘与建项目之前：一个字节都不该动
            assert repo.get_project(services.conn, "甲项目") is None
        assert mock_openai.LAST_AUTH == {}
        assert not (services.settings.upload_dir / "甲项目").exists()
    finally:
        services.close()


def test_verify_does_not_probe_with_server_key_when_fallback_is_off(
    tmp_path: Path,
) -> None:
    """空字段不许回落到服务端 Key 真探测——否则这个接口就是一条免费探测通道。"""
    services, _ = _strict_env(tmp_path, allow_fallback=False)
    try:
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            data = client.post("/api/credentials/verify", json={}).json()
        assert mock_openai.LAST_AUTH == {}, "空凭据不该产生任何厂商调用"
        assert data["embedding"]["ok"] is False
        assert data["llm"]["ok"] is False
        assert data["embedding"]["source"] == "missing"
        assert data["llm"]["source"] == "missing"
        assert "你自己的 Key" in data["embedding"]["error"]
    finally:
        services.close()


def test_verify_still_probes_the_key_the_user_did_fill(
    tmp_path: Path,
) -> None:
    """严格模式只是"不代付"，不是"什么都不验"：填了的那一把要真的验一遍。"""
    services, _ = _strict_env(tmp_path, allow_fallback=False)
    try:
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            data = client.post(
                "/api/credentials/verify",
                json={"embedding_key": "sk-alice-embed", "llm_model": LLM_CHOICES[0].id},
            ).json()
        assert data["embedding"]["ok"] is True
        assert data["embedding"]["source"] == "user"
        assert mock_openai.LAST_AUTH["/v1/embeddings"] == "Bearer sk-alice-embed"
        assert data["llm"]["ok"] is False
        assert data["llm"]["source"] == "missing"
        assert "/v1/chat/completions" not in mock_openai.LAST_AUTH
    finally:
        services.close()


def test_project_summary_is_not_generated_on_the_server_budget(
    tmp_path: Path, docx_file: Path
) -> None:
    """项目详情会顺手调一次大模型写摘要，这条路径不走 `svc_billable`。

    判据不能是"用户名非空"：只传一个模型头也会被解析成 anonymous，那样这道侧门
    从来没关上过。
    """
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        _ingest_one(services, settings, docx_file)
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            resp = client.get(
                "/api/projects/甲项目", headers={"X-R4P-LLM-Model": LLM_CHOICES[1].id}
            )
        assert resp.status_code == 200
        assert resp.json()["summary"] == ""
        assert "/v1/chat/completions" not in mock_openai.LAST_AUTH
    finally:
        services.close()


def test_project_summary_still_works_with_the_users_own_llm_key(
    tmp_path: Path, docx_file: Path
) -> None:
    """带了自己的 LLM Key 就该照常生成摘要——严格模式只拦"代付"，不拦功能。"""
    services, settings = _strict_env(tmp_path, allow_fallback=False)
    try:
        _ingest_one(services, settings, docx_file)
        mock_openai.reset()
        with TestClient(create_app(services)) as client:
            resp = client.get(
                "/api/projects/甲项目", headers=_user_headers("alice", llm="sk-alice-llm")
            )
        assert resp.status_code == 200
        assert resp.json()["summary"] != ""
        assert mock_openai.LAST_AUTH["/v1/chat/completions"] == "Bearer sk-alice-llm"
    finally:
        services.close()
