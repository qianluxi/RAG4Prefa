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

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

pymupdf = pytest.importorskip("pymupdf", reason="pdf 解析需要 pymupdf")

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
    assert "不会被入库" in msg, "必须说清文件与索引都没被动过"


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
