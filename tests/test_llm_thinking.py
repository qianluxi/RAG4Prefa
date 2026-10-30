"""思考开关的三态行为。

这个开关不是"顺手加个参数"：同一个模型名在两个平台上默认行为相反——
魔搭的 `Qwen/Qwen3.5-35B-A3B` 默认不思考，硅基流动的**同名**模型默认思考。
后者在判官这种要求严格两行输出的调用里，200 token 会全花在思维链上
（实测 completion 202 / reasoning 200，正文为空，单次 105 秒），
在评测报告里表现成"模型不听话"，而实际原因是没关思考。

所以这里钉两件事：
1. 三态的解析（留空 = 没配，不能看成 False）；
2. 没配时**一个字节都不多发**——给不认识这个字段的服务端乱发 body 可能 400。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag4prefa.adapters.llm import LLM
from rag4prefa.config import Settings


def _fake(settings: Settings) -> tuple[LLM, list[dict]]:
    """装配一个只记录请求参数的 LLM，不走网络。"""
    calls: list[dict] = []

    class _Completions:
        def create(self, **kwargs):  # noqa: ANN003, ANN202 - 假的 SDK 方法
            calls.append(kwargs)
            message = SimpleNamespace(content="可\n依据片段 1。")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    llm = LLM.__new__(LLM)  # 跳过 __init__ 里创建真实客户端
    llm._settings = settings
    llm._sync = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    return llm, calls


def _settings(**kwargs) -> Settings:
    # `_env_file=None` 是为了不读本机真实 .env：这些断言只关于传进来的值。
    base = {"llm_api_key": "sk-test", "_env_file": None}
    return Settings(**{**base, **kwargs})


parametrize = pytest.mark.parametrize


@parametrize("raw, expected", [
    ("", None),
    ("   ", None),
    ("none", None),
    ("null", None),
    ("false", False),
    ("False", False),
    ("0", False),
    ("true", True),
    ("1", True),
])
def test_thinking_env_parsing(monkeypatch, raw: str, expected: bool | None) -> None:
    """空串/占位词当"没配"，不是 False。

    从 `.env.example` 复制出来的文件常常保留 `R4P_LLM_ENABLE_THINKING=` 这一行，
    若把空串解析成 False，魔搭（不认识这个字段）会收到未知 body 而 400；
    若不解析直接抛 ValidationError，启动就失败，报错还指向一个无关字段。
    """
    monkeypatch.setenv("R4P_LLM_ENABLE_THINKING", raw)
    assert Settings(_env_file=None).llm_enable_thinking is expected


def test_thinking_absent_by_default() -> None:
    """默认必须是 None（三态），不能是 False。"""
    assert _settings().llm_enable_thinking is None


def test_unset_sends_no_extra_body() -> None:
    """没配就不发：魔搭那边多传未知字段可能 400。"""
    llm, calls = _fake(_settings())
    assert llm.complete("s", "u") == "可\n依据片段 1。"
    assert "extra_body" not in calls[0]


def test_false_is_sent_explicitly() -> None:
    """显式 False 要真的发出去——硅基流动靠它关掉默认开启的思考。"""
    llm, calls = _fake(_settings(llm_enable_thinking=False))
    llm.complete("s", "u")
    assert calls[0]["extra_body"] == {"enable_thinking": False}


def test_probe_also_carries_the_switch() -> None:
    """校验按钮也要带开关，否则开思考的部署上点一次要等 100 秒，像卡死。"""
    llm, calls = _fake(_settings(llm_enable_thinking=False))
    assert llm.probe() == "可\n依据片段 1。"
    assert calls[0]["extra_body"] == {"enable_thinking": False}
