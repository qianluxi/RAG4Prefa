"""判官层：解析必须可信，重试必须只换格式。

判官的全部价值在"输出可核对"。如果解析把「输出不合格式」当成「判了不可」，
拒答率会因为模型话多而虚高，而报告上看不出任何异常——所以 `None` 与 `False`
必须分成两个值，且这两条要有断言钉住。

另一条同样容易被写坏的是重试：重试若带上新的判定口径，同一个问题两次判断就
不是同一把尺子，"判官"这件事本身失去意义。测试只允许它改格式。
"""

from __future__ import annotations

from rag4prefa import prompts, sufficiency
from rag4prefa.domain import Hit


def _hit(
    i: int = 1, *, filename: str = "05_验收暂行规定.pdf", path=None, text: str = "正文"
) -> Hit:
    return Hit(
        chunk_id=f"c{i}",
        filename=filename,
        chapter_path=path if path is not None else ["第5章", "5.1 一般规定"],
        text=text,
    )


class _Recorder:
    """假模型：按顺序吐预设回复，并记下每次的 user 提示。"""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


# ── 解析 ────────────────────────────────────────────────────


def test_parse_yes_and_no() -> None:
    assert sufficiency.parse_judge_reply("可\n片段给出了 5 年").can_answer is True
    assert sufficiency.parse_judge_reply("不可\n没有给年限").can_answer is False


def test_parse_negative_variants() -> None:
    """三字变体不能被「可」吃掉，也不能落进 None——否则会多一次无意义的重试。"""
    cases = ("不可以\n缺依据", "不能\n缺依据", "不足以\n缺依据",
             "无法判断\n缺依据", "不行\n缺依据")
    for text in cases:
        reply = sufficiency.parse_judge_reply(text)
        assert reply.can_answer is False, text
        assert reply.reason == "缺依据"


def test_parse_wrapped_marker() -> None:
    """模型爱加粗、爱加书名号：包装层不该改变结论。"""
    assert sufficiency.parse_judge_reply("**不可**\n缺具体取值").can_answer is False
    assert sufficiency.parse_judge_reply("「可」\n片段有 30 个工作日").can_answer is True


def test_parse_unparseable_is_none_not_false() -> None:
    """首行不是判断词 → None。若退化成 False，格式故障会被记成一次成功的拒答。"""
    assert sufficiency.parse_judge_reply("").can_answer is None
    assert sufficiency.parse_judge_reply("这段资料看下来算是可以回答的").can_answer is None
    assert sufficiency.parse_judge_reply("根据片段[1]，可以回答").can_answer is None


def test_parse_ignores_reason_line() -> None:
    """结论只看首行：理由里出现「不可」不能推翻首行的「可」。"""
    reply = sufficiency.parse_judge_reply("可\n虽然第三段不可用，前两段给了 5 年")
    assert reply.can_answer is True
    assert "第三段" in reply.reason


# ── 判官调用 ────────────────────────────────────────────────


def test_no_hits_does_not_call_model() -> None:
    """没有候选是规则层能判的事，多调一次模型只多一个出错的机会。"""
    rec = _Recorder("可")
    reply = sufficiency.judge_by_llm("问题", [], rec)
    assert reply.can_answer is False
    assert rec.calls == []


def test_retry_only_when_unparseable() -> None:
    rec = _Recorder("我觉得差不多可以", "不可\n缺具体年限")
    reply = sufficiency.judge_by_llm("保修期几年？", [_hit()], rec)
    assert reply.can_answer is False
    assert reply.attempts == 2
    assert len(rec.calls) == 2
    # 第二次调用必须是"收紧格式"，不是换判据：判据段落两次要一致。
    assert rec.calls[0][0] == rec.calls[1][0]
    assert rec.calls[0][1] in rec.calls[1][1]


def test_no_retry_when_parsed() -> None:
    rec = _Recorder("不可\n缺依据", "可\n不该被用到")
    assert sufficiency.judge_by_llm("问题", [_hit()], rec).can_answer is False
    assert len(rec.calls) == 1


def test_retry_disabled_keeps_none() -> None:
    rec = _Recorder("看不太出来")
    reply = sufficiency.judge_by_llm("问题", [_hit()], rec, retry=False)
    assert reply.can_answer is None
    assert len(rec.calls) == 1


# ── 提示词 ──────────────────────────────────────────────────


def test_prompt_shows_where_the_snippet_comes_from() -> None:
    """库外题的主要失败形态是"片段来自别处"，出处被抹掉判官就失去判据。"""
    user = prompts.build_judge_prompt("北京市的补贴标准？", [_hit(filename="A3_宁波市鄞州区.pdf")])
    assert "A3_宁波市鄞州区.pdf" in user
    assert "第5章 > 5.1 一般规定" in user
    assert "北京市的补贴标准？" in user


def test_prompt_falls_back_to_title() -> None:
    hit = Hit(chunk_id="c1", title="工程保修单", chapter_path=[], text="x")
    user = prompts.build_judge_prompt("q", [hit])
    assert "工程保修单" in user


def test_prompt_truncates() -> None:
    hits = [_hit(i) for i in range(1, 8)]
    user = prompts.build_judge_prompt("q", hits, top_n=3)
    assert "[3]" in user and "[4]" not in user
    user = prompts.build_judge_prompt("q", [_hit(text="甲" * 1000)], per_snippet=50)
    assert "甲" * 50 in user and "甲" * 51 not in user


# ── 版本切换 ────────────────────────────────────────────────


def test_judge_prompt_registry_has_two_distinct_versions() -> None:
    """v1/v2 必须是两套判据，不是同一段话加了句提示。

    v2 的由来是 v1 在口语题上的系统性误拒（`eval/refusal-judge-20261030.md`）：
    取值型与做法型不能用同一把尺子。所以断言落在"分型的标志性文字"上，
    而不是"两段字符串不相等"——后者改个错字也会通过。
    """
    assert set(prompts.JUDGE_PROMPTS) == {"v1", "v2"}
    v1, v2 = prompts.JUDGE_PROMPTS["v1"], prompts.JUDGE_PROMPTS["v2"]
    assert "取值型" not in v1 and "做法/结论型" not in v1
    assert "取值型" in v2 and "做法/结论型" in v2
    # 共同的那条不能丢：库外拦截全靠"别的地区/年份/项目"
    for text in (v1, v2):
        assert "别的地区" in text


def test_judge_by_llm_passes_chosen_system_to_both_attempts() -> None:
    """重试要沿用同一个版本，否则判了两次却是两把尺子。"""
    rec = _Recorder("格式不对", "不可\n缺依据")
    sufficiency.judge_by_llm("问题", [_hit()], rec, system=prompts.SYSTEM_JUDGE_V2)
    assert len(rec.calls) == 2
    assert all(call[0] is prompts.SYSTEM_JUDGE_V2 for call in rec.calls)
