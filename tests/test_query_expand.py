"""查询术语扩展的单元测试：不联网，用带 `complete` 的假 LLM 对象。"""

from __future__ import annotations

from rag4prefa.adapters.query_expand import QueryExpander, build_expander


class FakeLLM:
    """只实现 QueryExpander 用到的 `complete`；可注入返回值或异常。"""

    def __init__(self, result: str | None = None, exc: Exception | None = None) -> None:
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, str, int | None]] = []

    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        self.calls.append((system, user, max_tokens))
        if self._exc is not None:
            raise self._exc
        return self._result or ""


def _settings():
    # 只需要一个能构造 expander 的最小对象；QueryExpander 不读 settings 的字段。
    class _S:
        pass

    return _S()


def test_expand_appends_terms_to_original_question() -> None:
    """扩展是「原话 + 术语」，不是替换：原话必须完整保留在最前面。"""
    llm = FakeLLM(result="悬挂较重物体 专用连接件")
    exp = QueryExpander(_settings(), llm)
    out = exp.expand("挂电视吊柜会不会掉下来？")
    assert out == "挂电视吊柜会不会掉下来？ 悬挂较重物体 专用连接件"
    assert exp.available is True


def test_expand_without_llm_is_identity() -> None:
    """没有 LLM 时是 no-op：原样返回，且 available=False。"""
    exp = QueryExpander(_settings(), None)
    assert exp.expand("问句") == "问句"
    assert exp.available is False


def test_expand_empty_result_falls_back_to_question() -> None:
    """LLM 返回空串，降级回原话，不拼出「原话 + 空格」这种坏串。"""
    llm = FakeLLM(result="   ")
    exp = QueryExpander(_settings(), llm)
    assert exp.expand("问句") == "问句"


def test_expand_llm_exception_falls_back_to_question() -> None:
    """LLM 抛异常（超时、鉴权失败等）也降级回原话，绝不让检索中断。"""
    llm = FakeLLM(exc=RuntimeError("boom"))
    exp = QueryExpander(_settings(), llm)
    assert exp.expand("问句") == "问句"


def test_build_expander_without_llm_is_identity() -> None:
    """build_expander(None) 得到一个 no-op，available=False。"""
    exp = build_expander(_settings(), None)
    assert exp.expand("问句") == "问句"
    assert exp.available is False


def test_build_expander_with_llm_is_active() -> None:
    llm = FakeLLM(result="术语")
    exp = build_expander(_settings(), llm)
    assert exp.available is True
    assert exp.expand("问句") == "问句 术语"
