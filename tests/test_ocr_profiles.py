"""OCR 档位的离线测试。

与 embedding 的取舍完全不同：embedding 换模型会污染向量空间，OCR 不会——它只把
像素变成文本，不产生任何向量。所以「可以换」是合理的，但要守住三件事：

1. 选了免费档就**真的**打到免费那个端点与模型，否则"零成本"只是界面上两个字；
2. 两档的缓存键必须不同——同一个 `chunk_id` 不能对应两份格式不同的正文；
3. 用户填的 Key 必须落到**这一档要读的那个字段**上，落错了就会"显示已填、
   实际仍在用服务端那把"，而两种情况下界面都显示成功。

全部离线：不联网、不花钱。免费档的端点写死在代码里，测试不去打它，
只断言"要打的是它"。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag4prefa import ingest
from rag4prefa.adapters.ocr import DOCUMENT_PRESET, VLM_PRESET, VlmOcr
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import create_app
from rag4prefa.config import Settings
from rag4prefa.credentials import (
    HEADER_OCR_KEY,
    HEADER_OCR_PROFILE,
    from_headers,
    settings_for_ocr,
    user_key,
)
from rag4prefa.db import connect, init_db
from rag4prefa.ocr_profiles import (
    DEFAULT_PROFILE,
    FREE_PROFILE,
    PAID_PROFILE,
    effective,
    resolve_profile,
)
from rag4prefa.services import Services

_DIM = 8


def _settings(tmp_path: Path, **over: object) -> Settings:
    """一份"两档都配齐"的配置。

    `embedding_api_key` / `llm_api_key` 默认给空串：`/api/credentials/verify`
    在留空时会直接标"未配置"而不真发请求，测试因此不会意外联网。
    """
    base: dict[str, object] = {
        "_env_file": None,
        "data_dir": tmp_path,
        "embedding_api_key": "",
        "llm_api_key": "",
        "vector_backend": "numpy",
        "ocr_mode": "auto",
        "ocr_base_url": "https://api.siliconflow.com/v1",
        "ocr_api_key": "sk-intl",
        "ocr_model": "Qwen/Qwen3-VL-8B-Instruct",
        "ocr_free_api_key": "sk-cn",
        "ocr_concurrency": 4,
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
        store=NumpyStore(conn, _DIM),
        embedder=None,
        llm=None,
        reranker=NoopReranker(),
    )


# ── 档位解析 ────────────────────────────────────────────


@pytest.mark.parametrize("value", ["", "   ", "premium", "freee", "free2"])
def test_unknown_profile_falls_back_to_default(value: str) -> None:
    assert resolve_profile(value) == DEFAULT_PROFILE


def test_profile_name_is_case_and_space_insensitive() -> None:
    """浏览器里存过 " FREE " 这种值也不该把它当成未知档位。"""
    assert resolve_profile(" FREE ") == "free"
    assert resolve_profile("Paid") == "paid"


def test_fallback_argument_is_used_when_value_is_unknown() -> None:
    assert resolve_profile("nope", "free") == "free"
    # 连 fallback 也是垃圾时，落到默认档而不是抛错——请求不该因此失败
    assert resolve_profile("nope", "also-nope") == DEFAULT_PROFILE


# ── 每档真正用到什么 ─────────────────────────────────────


def test_free_profile_overrides_endpoint_model_preset_and_concurrency(
    tmp_path: Path,
) -> None:
    """免费档的四个参数全部来自代码常量，`.env` 一个都不参与。

    这四个值就是"零成本"的全部依据：端点换成国内站、模型换成标价 ¥0 的文档解析
    模型、提示词换成它的原生指令、并发降到免费池扛得住的量。少盖任何一个，
    界面上写着免费、账单上照样扣钱。
    """
    eff = effective(_settings(tmp_path, ocr_profile="free"))
    assert eff.profile is FREE_PROFILE
    assert eff.base_url == "https://api.siliconflow.cn/v1"
    assert eff.model == "PaddlePaddle/PaddleOCR-VL-1.5"
    assert eff.preset == "document"
    assert eff.concurrency == FREE_PROFILE.concurrency
    assert eff.concurrency < 4, "免费池并发必须低于付费档，否则会持续 429"


def test_paid_profile_uses_everything_from_env(tmp_path: Path) -> None:
    """付费档不许覆盖任何一项：换厂商只改 `.env`，不用回来改代码。"""
    eff = effective(_settings(tmp_path, ocr_profile="paid", ocr_concurrency=7))
    assert eff.profile is PAID_PROFILE
    assert eff.base_url == "https://api.siliconflow.com/v1"
    assert eff.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert eff.preset == "vlm"
    assert eff.concurrency == 7


def test_free_profile_never_reuses_the_paid_key(tmp_path: Path) -> None:
    """免费档没配 Key 时必须报"没有"，不能回落到付费档那把。

    两站是两套独立账号，回落的结果是国际站的 Key 打到国内站，收到
    `30014 Token is invalid`——与"Key 被删"同一个错误码，用户会去反复重建一把
    本来就没错的 Key。
    """
    eff = effective(_settings(tmp_path, ocr_profile="free", ocr_free_api_key=""))
    assert eff.api_key == ""
    assert not eff.ready


def test_free_profile_uses_the_free_key_not_the_paid_one(tmp_path: Path) -> None:
    eff = effective(_settings(tmp_path, ocr_profile="free"))
    assert eff.api_key == "sk-cn"


def test_paid_profile_uses_the_paid_key(tmp_path: Path) -> None:
    eff = effective(_settings(tmp_path, ocr_profile="paid"))
    assert eff.api_key == "sk-intl"


def test_user_key_wins_in_either_profile(tmp_path: Path) -> None:
    for profile in ("free", "paid"):
        eff = effective(_settings(tmp_path, ocr_profile=profile), user_key="sk-mine")
        assert eff.api_key == "sk-mine", profile


# ── 缓存键 ──────────────────────────────────────────────


def test_paid_cache_key_is_unchanged(tmp_path: Path) -> None:
    """付费档的缓存键必须还是老写法。

    已有部署的 `ocr_cache` 全是按 `模型名#v2` 存的。这次改动如果顺手把格式
    统一成 `模型名#vlm-v2`，所有已识别过的扫描件都要重新付一遍钱，
    而提示词一个字都没改。
    """
    assert VlmOcr(_settings(tmp_path)).model == "Qwen/Qwen3-VL-8B-Instruct#v2"


def test_two_profiles_never_share_a_cache_entry(tmp_path: Path) -> None:
    """同一页在两档下格式完全不同，绝不能共用一份缓存。

    共用的话，先跑免费档再切付费档（或反过来）会读到"格式对不上的旧文本"，
    而且 `verify` 查不出来——它只比 id 集合是否自洽，不比内容是不是这一版。
    """
    free_key = VlmOcr(_settings(tmp_path, ocr_profile="free")).model
    paid_key = VlmOcr(_settings(tmp_path, ocr_profile="paid")).model
    assert free_key != paid_key
    assert free_key == "PaddlePaddle/PaddleOCR-VL-1.5#document-d1"


# ── 提示词形状 ───────────────────────────────────────────


def test_document_preset_sends_no_system_message(tmp_path: Path) -> None:
    """专用解析模型只收一条 user 消息。

    给它套那段 7 条中文规则（含"宁可多标"）会和它自己的版面判断打架：
    它本来就知道哪些行是标题，额外的文字规则只会把正文推广成标题。
    """
    messages = VlmOcr(_settings(tmp_path, ocr_profile="free"))._messages(b"\x00", 1)
    assert [m["role"] for m in messages] == ["user"]
    assert messages[0]["content"][0]["text"] == DOCUMENT_PRESET.user


def test_vlm_preset_keeps_system_prompt_and_page_number(tmp_path: Path) -> None:
    """通用视觉模型那一档保持原样：带 system，且 user 里说明是第几页。"""
    messages = VlmOcr(_settings(tmp_path, ocr_profile="paid"))._messages(b"\x00", 7)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == VLM_PRESET.system
    assert "第 7 页" in messages[1]["content"][0]["text"]


def test_document_preset_allows_more_output_tokens(tmp_path: Path) -> None:
    """密排页 + 多表格会超过 4096 token；专用模型免费，没有理由掐着给。"""
    assert DOCUMENT_PRESET.max_tokens > VLM_PRESET.max_tokens


# ── 凭据归位 ────────────────────────────────────────────


def test_user_key_lands_in_the_field_each_profile_reads(tmp_path: Path) -> None:
    """免费档的 Key 必须写进 `ocr_free_api_key`。

    写进 `ocr_api_key` 的话，`effective()` 在免费档下读的是另一个字段，
    用户填的 Key 就静默失效了——界面上显示"已保存"，请求实际还在用服务端那把。
    """
    base = _settings(tmp_path)
    free = settings_for_ocr(base, "free", "sk-mine")
    assert free.ocr_free_api_key == "sk-mine"
    assert free.ocr_api_key == "sk-intl", "付费档那把不能被免费档的 Key 顶掉"

    paid = settings_for_ocr(base, "paid", "sk-mine")
    assert paid.ocr_api_key == "sk-mine"
    assert paid.ocr_free_api_key == "sk-cn"


def test_switching_profiles_does_not_destroy_env_values(tmp_path: Path) -> None:
    """一个 `Settings` 里两档的配置必须同时活着。

    如果免费档在构造期把 `ocr_base_url` 覆盖成国内站，同进程再切回付费档就
    取不回国际站那个地址了——`Settings` 是唯一一份，覆盖之后原值就没了。
    """
    base = _settings(tmp_path)
    assert effective(base.model_copy(update={"ocr_profile": "free"})).base_url.endswith(
        "siliconflow.cn/v1"
    )
    assert effective(base).base_url.endswith("siliconflow.com/v1")


def test_from_headers_reads_the_ocr_headers() -> None:
    creds = from_headers({HEADER_OCR_KEY: "sk-x", HEADER_OCR_PROFILE: "free"})
    assert creds is not None
    assert creds.ocr_key == "sk-x"
    assert creds.ocr_profile == "free"
    assert creds.has_keys, "带了 OCR Key 就算带了 Key，额度来源要跟着变"


def test_profile_only_request_is_still_a_valid_credential() -> None:
    """只切档位、不带任何 Key 也是有效请求。

    判空只看 Key 的话，这种请求会被当成"未登录"，档位丢失 →
    界面显示免费档、服务端按 .env 跑付费档。
    """
    assert from_headers({HEADER_OCR_PROFILE: "free"}) is not None
    assert from_headers({}) is None


def test_user_key_for_ocr_role_does_not_fall_through_to_llm() -> None:
    """角色查表必须是三路，不能用「不是 embedding 就是 llm」的二分。

    二分时 strict 模式查 OCR 的 Key 会读到 LLM 那把，于是"用户没填 OCR Key"
    被误判成"填了"，匿名请求照样能烧服务端额度。
    """
    from rag4prefa.credentials import UserCredentials

    creds = UserCredentials("u", embedding_key="", llm_key="sk-llm", ocr_key="")
    assert user_key(creds, "ocr") == ""
    assert user_key(creds, "llm") == "sk-llm"


# ── 对外可见的状态 ───────────────────────────────────────


def test_health_reports_tier_and_key_source(tmp_path: Path) -> None:
    paid = _services(tmp_path)._ocr_status()
    assert paid["profile"] == "paid"
    assert paid["free"] is False
    assert paid["key_source"] == "server"
    assert paid["model"] == "Qwen/Qwen3-VL-8B-Instruct"


def test_health_marks_a_keyless_free_tier_as_missing(tmp_path: Path) -> None:
    """免费档但没配 Key：必须报 missing，而不是让用户拖进扫描件才发现。

    "报 missing" 与 "报 server" 的差别就是：前者用户去填 Key，后者用户以为能跑。
    """
    status = _services(tmp_path, ocr_profile="free", ocr_free_api_key="")._ocr_status()
    assert status["profile"] == "free"
    assert status["free"] is True
    assert status["key_source"] == "missing"


def test_health_reports_key_from_user_when_supplied(tmp_path: Path) -> None:
    from rag4prefa.credentials import UserCredentials

    services = _services(tmp_path)
    services.user_creds = UserCredentials("u", ocr_key="sk-mine", ocr_profile="paid")
    assert services._ocr_status()["key_source"] == "user"


def test_ocr_profile_endpoint_lists_both_tiers(tmp_path: Path) -> None:
    services = _services(tmp_path)
    with TestClient(create_app(services)) as client:
        data = client.get("/api/ocr-profiles").json()
    assert {option["id"] for option in data["options"]} == {"free", "paid"}
    assert data["current"] == "paid"
    # 两档的模型名必须各算各的：只回一个 effective() 会让没选中那档显示成同一个名字
    models = {option["id"]: option["model"] for option in data["options"]}
    assert models["free"] != models["paid"]
    assert models["free"] == "PaddlePaddle/PaddleOCR-VL-1.5"


def test_verify_reports_missing_ocr_key_instead_of_pretending_ok(tmp_path: Path) -> None:
    """没配 Key 的 OCR 项必须是 ok=False。

    "没验过"报成通过，是这类面板最常见的假象：用户点完"保存并验证"看到三行绿色，
    第一次上传扫描件才发现根本调不通。
    """
    services = _services(tmp_path, ocr_profile="free", ocr_free_api_key="")
    with TestClient(create_app(services)) as client:
        result = client.post(
            "/api/credentials/verify",
            json={"username": "tester", "ocr_profile": "free"},
        ).json()
    assert result["ocr"]["ok"] is False
    assert result["ocr"]["profile"] == "free"
    assert result["ocr"]["source"] == "missing"


def test_build_ocr_takes_concurrency_from_the_profile(tmp_path: Path, conn) -> None:
    """并发按档位取，不是照抄 `.env` 的 4：免费池撑不住 4 并发。"""
    free = ingest._build_ocr(_settings(tmp_path, ocr_profile="free"), conn)
    assert free is not None
    assert free.workers == FREE_PROFILE.concurrency

    paid = ingest._build_ocr(_settings(tmp_path, ocr_profile="paid", ocr_concurrency=7), conn)
    assert paid is not None
    assert paid.workers == 7


def test_build_ocr_returns_none_when_mode_is_off(tmp_path: Path, conn) -> None:
    assert ingest._build_ocr(_settings(tmp_path, ocr_mode="off"), conn) is None
