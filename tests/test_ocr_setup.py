"""扫描件识别的准入策略：配置齐了才识别，没配就明确拒绝并引导去配置。

2026-10-27 的实测结论决定了这套策略（细节见 `ocr_setup` 模块头）：免费视觉模型会
在部分页面上输出**伪造正文**，而伪造正文会被切块、写向量、被引用成答案，且不报错。
所以现在只有一条路径——用配置好的付费视觉模型；没配就不处理扫描件。

这些用例守的是那条边界：
- 没有 Key 时**不许**开始识别（更不能悄悄回落到 LLM / embedding 的 Key）；
- 拒绝时的消息必须能让普通用户看懂"去哪儿配"，并且先说"文件没入库"；
- `health` 必须把"能不能用"如实报出来，前端才能提前提示而不是等失败。

全部离线：只碰配置与消息，不发任何请求。
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pymupdf = pytest.importorskip("pymupdf", reason="pdf 解析需要 pymupdf")

from rag4prefa import ingest, repo  # noqa: E402
from rag4prefa.adapters.ocr import (  # noqa: E402
    OCR_SETUP_HINT,
    OcrError,
    VlmOcr,
)
from rag4prefa.adapters.parsers import parse_document_report  # noqa: E402
from rag4prefa.adapters.rerank import NoopReranker  # noqa: E402
from rag4prefa.adapters.vectors import NumpyStore  # noqa: E402
from rag4prefa.api import create_app  # noqa: E402
from rag4prefa.config import Settings  # noqa: E402
from rag4prefa.credentials import UserCredentials, settings_for_ocr  # noqa: E402
from rag4prefa.db import connect, init_db  # noqa: E402
from rag4prefa.ocr_setup import effective  # noqa: E402
from rag4prefa.services import Services  # noqa: E402
from tests.conftest import FakeEmbedder  # noqa: E402

_HINT_BARE = OCR_SETUP_HINT.strip("【】")


def _settings(tmp_path: Path, **over: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "data_dir": tmp_path,
        "embedding_api_key": "test",
        "llm_api_key": "llm-key",
        "embedding_model": "fake-embed",
        "embedding_dim": 16,
        "vector_backend": "numpy",
    }
    base.update(over)
    return Settings(**base)  # type: ignore[arg-type]


def _blank_pdf(path: Path, pages: int = 2) -> Path:
    """造一份"扫描件"：每页一个字都不写，`get_text()` 返回空串。

    在 `get_text()` 这一层，它和真实扫描件没有任何区别——判定的入口就是这里。
    """
    doc = pymupdf.open()
    for i in range(pages):
        page = doc.new_page()
        # 画一道细线：真实的两页扫描图不会逐字节相同，而完全相同的页图会被
        # "同图只识别一次"合并，测出来的哈希与调用次数就跟真实情况对不上。
        page.draw_line((10, 10 + i), (100, 10 + i))
    doc.save(path)
    return path


# ── 参数解析 ────────────────────────────────────────────


def test_effective_reads_env_and_reports_ready(tmp_path: Path) -> None:
    s = _settings(
        tmp_path,
        ocr_base_url="https://api.example.com/v1",
        ocr_api_key="sk-server",
        ocr_model="Qwen/Qwen3-VL-8B-Instruct",
    )
    eff = effective(s)
    assert eff.ready is True
    assert eff.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert eff.base_url == "https://api.example.com/v1"
    assert eff.api_key == "sk-server"


def test_effective_user_key_wins_but_never_invents_one(tmp_path: Path) -> None:
    """用户 Key 压过服务端；两处都没有时 `ready` 必须是假，而不是"能凑合用"。"""
    s = _settings(tmp_path, ocr_base_url="https://api.example.com/v1", ocr_model="m")
    assert effective(s).ready is False, "没 Key 还说 ready，扫描件会在半路失败"
    assert effective(s, user_key="sk-mine").ready is True


def test_effective_does_not_fall_back_to_other_keys(tmp_path: Path) -> None:
    """OCR 绝不借用 embedding / LLM 的 Key。

    它们是另外两个平台的账号：借用只会得到 401，而"为什么 OCR 一直失败"会变成
    一个要翻三个文件才能回答的问题。
    """
    s = _settings(
        tmp_path,
        embedding_api_key="sk-embed",
        llm_api_key="sk-llm",
        ocr_base_url="https://api.example.com/v1",
        ocr_model="m",
    )
    eff = effective(s)
    assert eff.api_key == ""
    assert eff.ready is False


def test_cache_key_keeps_legacy_format(tmp_path: Path) -> None:
    """缓存键必须还是 `模型名#提示词版本`。

    已有 `ocr_cache` 全是这个键；换写法等于让所有文档重新付费识别一遍，
    而提示词一个字都没改——那种"优化"是要花钱的。
    """
    s = _settings(
        tmp_path,
        ocr_base_url="https://api.example.com/v1",
        ocr_api_key="sk-server",
        ocr_model="Qwen/Qwen3-VL-8B-Instruct",
    )
    assert VlmOcr(s).model == "Qwen/Qwen3-VL-8B-Instruct#v2"


# ── 拒绝时的行为 ────────────────────────────────────────


def test_missing_key_refuses_before_any_page(tmp_path: Path) -> None:
    s = _settings(
        tmp_path,
        ocr_mode="auto",
        ocr_base_url="https://api.example.com/v1",
        ocr_model="Qwen/Qwen3-VL-8B-Instruct",
    )
    with pytest.raises(OcrError) as err:
        VlmOcr(s)

    msg = str(err.value)
    assert msg.startswith(OCR_SETUP_HINT), "前端靠这个前缀决定要不要弹配置引导"
    assert "R4P_OCR_API_KEY" in msg
    assert "扫描件无法入库" in msg, "必须说清文件与索引都没被动过"


def test_scanned_pdf_without_ocr_says_where_to_configure(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path / "扫描件.pdf")
    with pytest.raises(ValueError) as err:
        parse_document_report(pdf, ocr=None)

    msg = str(err.value)
    assert msg.startswith(OCR_SETUP_HINT)
    assert "扫描件" in msg
    assert "R4P_OCR_API_KEY" in msg, "只讲现象不讲怎么修，用户只能来问人"
    assert "索引与归档没有任何改动" in msg


# ── 状态上报 ────────────────────────────────────────────


def _services(tmp_path: Path, **over: object) -> Services:
    settings = _settings(tmp_path, **over)
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)
    return Services(
        settings=settings,
        conn=conn,
        store=NumpyStore(conn, 16),
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        llm=None,
        reranker=NoopReranker(),
    )


def test_health_reports_ocr_not_ready_when_key_missing(tmp_path: Path) -> None:
    services = _services(
        tmp_path,
        ocr_mode="auto",
        ocr_base_url="https://api.example.com/v1",
        ocr_model="Qwen/Qwen3-VL-8B-Instruct",
    )
    try:
        with TestClient(create_app(services)) as client:
            ocr = client.get("/api/health").json()["ocr"]
    finally:
        services.close()

    assert ocr["ready"] is False
    assert ocr["key_source"] == "missing"
    assert ocr["mode"] == "auto", "关了和没配是两件事，不能混成一个值"


def test_health_reports_ready_with_server_key(tmp_path: Path) -> None:
    services = _services(
        tmp_path,
        ocr_mode="auto",
        ocr_base_url="https://api.example.com/v1",
        ocr_api_key="sk-server",
        ocr_model="Qwen/Qwen3-VL-8B-Instruct",
    )
    try:
        with TestClient(create_app(services)) as client:
            ocr = client.get("/api/health").json()["ocr"]
    finally:
        services.close()

    assert ocr["ready"] is True
    assert ocr["key_source"] == "server"
    assert ocr["model"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_health_reports_not_ready_when_mode_off(tmp_path: Path) -> None:
    """Key 配好了但功能关着，扫描件同样进不来——`ready` 必须是假。"""
    services = _services(
        tmp_path,
        ocr_mode="off",
        ocr_base_url="https://api.example.com/v1",
        ocr_api_key="sk-server",
        ocr_model="Qwen/Qwen3-VL-8B-Instruct",
    )
    try:
        with TestClient(create_app(services)) as client:
            ocr = client.get("/api/health").json()["ocr"]
    finally:
        services.close()

    assert ocr["ready"] is False
    assert ocr["model"] is None


def test_profile_list_route_is_gone(tmp_path: Path) -> None:
    """档位接口随免费档一起删掉。

    留着它（哪怕只返回一档）会让前端继续渲染一个没有意义的"选择"，
    而用户会以为自己能在两种模型之间挑。
    """
    services = _services(tmp_path)
    try:
        with TestClient(create_app(services)) as client:
            assert client.get("/api/ocr-profiles").status_code == 404
    finally:
        services.close()


# ── 各付各的：不代付时扫描件必须进不来 ──────────────────


def _strict(tmp_path: Path, **over: object) -> Services:
    """`.env` 里明明有服务端 Key，但这个部署不接受代付。

    这才是"给别人用"的配置：`.env` 那把只对服务端主人有意义，别人的请求一律
    不许拿它去花。判据不能只看"配置里有没有 Key"——有，不等于这次能用。
    """
    base: dict[str, object] = {
        "allow_server_key_fallback": False,
        "ocr_mode": "auto",
        "ocr_base_url": "https://api.example.com/v1",
        "ocr_api_key": "sk-server",
        "ocr_model": "Qwen/Qwen3-VL-8B-Instruct",
    }
    base.update(over)
    return _services(tmp_path, **base)


def test_strict_mode_refuses_scanned_pdf_without_own_key(tmp_path: Path) -> None:
    """访客不填自己的视觉模型 Key，扫描件必须进不来，而且一个字节都不留。

    这条守的是账单：视觉模型按页计费，`KEY_ROLES` 里早有 `ocr` 这个角色，但上传
    接口的 `need` 只有 embedding——上传时并不知道这批文件里有没有扫描件，硬加进去
    会让只传 docx 的人也被要求填视觉 Key。所以闸门落在"派生视图里的 key 被清空"，
    再由 `VlmOcr` 在真要识别某一页时拒绝。
    """
    services = _strict(tmp_path)
    pdf = _blank_pdf(tmp_path / "扫描件.pdf")
    viewer = services.for_credentials(UserCredentials(username="访客"))
    try:
        assert viewer.settings.ocr_api_key == "", "不代付时不能把服务端那把带下去"

        outcome = ingest.ingest_file(
            services.conn,
            services.store,
            services.embedder,  # type: ignore[arg-type]
            viewer.settings,
            path=pdf,
            project_name="上海文件",
        )

        assert outcome.status == "failed"
        msg = outcome.error or ""
        assert msg.startswith(OCR_SETUP_HINT), "前端按这个前缀决定弹不弹配置引导"
        assert "扫描件无法入库" in msg, "要直接说结果，不能只说'缺少配置'"
        assert "你自己的视觉模型 Key" in msg, "得告诉他去填自己的，不是去找管理员"
        for table in ("documents", "chunks"):
            got = services.conn.execute(f"select count(*) from {table}").fetchone()[0]
            assert got == 0, f"{table} 不该留下任何东西"
    finally:
        services.close()


def test_strict_mode_still_accepts_docx(tmp_path: Path, docx_file: Path) -> None:
    """判据是"这一页要不要识别"，不是"有没有 Key"。

    docx 与带文字层的 pdf 从不调用视觉模型。原来在 `_build_ocr` 里构造识别器，
    缺 Key 会让整批文件（包括 docx）一起失败——那是把"扫描件进不来"扩大成了
    "什么都进不来"。
    """
    services = _strict(tmp_path)
    viewer = services.for_credentials(UserCredentials(username="访客"))
    try:
        outcome = ingest.ingest_file(
            services.conn,
            services.store,
            services.embedder,  # type: ignore[arg-type]
            viewer.settings,
            path=docx_file,
            project_name="方案",
        )
        assert outcome.status == "indexed", outcome.error
        assert outcome.chunks > 0
    finally:
        services.close()


def test_strict_mode_uses_the_users_own_ocr_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """填了自己的 Key 就必须用他那把，而不是"服务端有就用服务端的"。"""
    seen: list[str] = []

    class FakeVlm:
        def __init__(self, settings: Settings) -> None:
            seen.append(settings.ocr_api_key)

        @property
        def model(self) -> str:
            return "fake-vlm#v1"

        def __call__(self, png: bytes, page: int) -> str:
            return f"# 第1章 总则\n\n第 {page} 页的正文，由用户自己的 Key 识别。"

    monkeypatch.setattr(ingest, "VlmOcr", FakeVlm)
    services = _strict(tmp_path)
    viewer = services.for_credentials(UserCredentials(username="张三", ocr_key="sk-mine"))
    try:
        outcome = ingest.ingest_file(
            services.conn,
            services.store,
            services.embedder,  # type: ignore[arg-type]
            viewer.settings,
            path=_blank_pdf(tmp_path / "扫描件.pdf"),
            project_name="上海文件",
        )
        assert outcome.status == "indexed", outcome.error
        assert seen and set(seen) == {"sk-mine"}, "必须全程用用户自己的 Key"
    finally:
        services.close()


def test_cache_hit_needs_no_key_at_all(conn: sqlite3.Connection) -> None:
    """整页都命中缓存时，不该要求任何付费能力。

    结果是库里现成的文本，重新识别只会得到"另一版"正文、换一批 chunk_id。
    构造器在这里直接抛错，就是要把"有没有构造过识别器"变成可断言的事实。
    """

    def never() -> VlmOcr:
        raise OcrError("这一页已经有结果了，不该构造识别器")

    call, close = ingest._cached_ocr(conn, "m#v1", never)
    repo.put_ocr_cache(conn, hashlib.sha256(b"page-1").hexdigest(), "m#v1", "已经识别过的正文")

    assert call(b"page-1", 1) == "已经识别过的正文"
    close()


def test_settings_for_ocr_follows_the_billing_mode(tmp_path: Path) -> None:
    """三条规则，少一条就会把钱记错人。"""
    loose = _settings(tmp_path, ocr_api_key="sk-server")
    strict = _settings(tmp_path, ocr_api_key="sk-server", allow_server_key_fallback=False)

    assert settings_for_ocr(loose, "").ocr_api_key == "sk-server", "单机演示沿用服务端那把"
    assert settings_for_ocr(strict, "").ocr_api_key == "", "不代付就必须清空"
    assert settings_for_ocr(strict, "sk-mine").ocr_api_key == "sk-mine", "用户填的优先"


def test_health_says_missing_when_strict_and_no_own_key(tmp_path: Path) -> None:
    """页头不能显示「OCR 就绪」：`.env` 里有 Key 不代表访客能用。

    这条与 `test_health_reports_ready_with_server_key` 是一对：同一个 `.env` 配置，
    只差 `allow_server_key_fallback`，结论必须相反。
    """
    services = _strict(tmp_path)
    try:
        with TestClient(create_app(services)) as client:
            ocr = client.get("/api/health").json()["ocr"]
    finally:
        services.close()

    assert ocr["ready"] is False
    assert ocr["key_source"] == "missing"


def test_health_sees_the_users_own_key(tmp_path: Path) -> None:
    services = _strict(tmp_path)
    try:
        with TestClient(create_app(services)) as client:
            ocr = client.get(
                "/api/health", headers={"X-R4P-OCR-Key": "sk-mine"}
            ).json()["ocr"]
    finally:
        services.close()

    assert ocr["ready"] is True
    assert ocr["key_source"] == "user"


# ── 前后端约定 ──────────────────────────────────────────


def test_frontend_shares_the_setup_hint() -> None:
    """前端那条判断用的是同一句话。

    两个进程没有共享常量的办法，只能靠这条用例守住：改了后端前缀却忘了改前端，
    表现是"扫描件被拒了，但引导不弹"——最难查的一类不一致。
    """
    html = (Path(__file__).resolve().parents[1] / "rag4prefa/web/index.html").read_text(
        encoding="utf-8"
    )
    assert _HINT_BARE in html
