"""词表的**误触发**约束：词出现在句子里 ≠ 用户在问这件事。

这个文件和 `test_term_map.py` 分工不同。那边钉的是"该命中的要命中"（召回侧），
这边钉的是"不该命中的别命中"（精确侧）——两边都会让指标变好或变坏，但错了以后
表现出来的症状完全不同：漏命中是"词表没用"，误命中是"注入的术语把检索带歪"，
而后者在报告上看起来像"混合召回变差"。

三条约束，每条都对应一类真实的错法：

1. **单字不算证据**。`修` 会命中「装修 / 检修 / 维修」，`板` 会命中「楼板 / 板材」。
   组里只要还有多字备选，仅靠单字命中就不成立——v2 在专业问法集上 6 次触发里
   有 2 次是这么来的。
2. **意图词与对象词不能写在同一个 OR 组**。「包含/算不算」是意图，「灯具/浴霸」是对象。
   混在一起，`吊顶安装灯具的构造要求` 会被判成品类边界题，注入「不含电器设备」。
3. **负例集是数据，不是注释**。`eval/term-negatives.jsonl` 是这条约束的唯一来源，
   审计脚本与测试共用；加规则时先加负例。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag4prefa.adapters.term_map import INTENTS, TermEntry, TermMap, load_term_map

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "rag4prefa" / "terms" / "public_to_standard.v2.json"
V3 = ROOT / "rag4prefa" / "terms" / "public_to_standard.v3.json"
NEGATIVES = ROOT / "eval" / "term-negatives.jsonl"


def _rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            out.append(json.loads(line))
    return out


# ── 单字规则 ──────────────────────────────────────────────


def test_single_char_alone_is_not_evidence_when_group_has_multichar() -> None:
    """「装修」里的「修」不该被当成「维修」——这是 v2 在专业集上误触发的原形。"""
    entry = TermEntry(
        id="e",
        groups=(("电线", "管线"), ("埋", "墙里", "换", "修")),
        terms=("管线分离",),
        source="s",
    )
    assert not entry.matches("装修完多久能入住？")
    assert entry.matches("墙里的电线以后坏了能换吗？")


def test_group_with_only_single_chars_still_matches() -> None:
    """规则不能一刀切：整组都是单字时（敲/叩），没有"多字备选"可比，照常命中。"""
    entry = TermEntry(id="e", groups=(("敲", "叩"), ("砖",)), terms=("空鼓",), source="s")
    assert entry.matches("敲一敲瓷砖有没有空响")


def test_relaxed_mode_exposes_what_the_rule_dropped() -> None:
    """收紧必须**可测量**：否则"词表没命中"与"被规则挡了"就分不开。"""
    entry = TermEntry(
        id="e", groups=(("电线",), ("修", "墙里")), terms=("维修更换",), source="s"
    )
    assert not entry.matches("装修时电线怎么处理？")
    assert entry.matches("装修时电线怎么处理？", strict=False)
    assert entry.dropped_singles("装修时电线怎么处理？") == ["修"]
    assert entry.dropped_singles("墙里的电线") == []


def test_explain_reports_dropped_entries_without_faking_a_match() -> None:
    """被挡下的条目可以出现在诊断里，但 `matched_by` 必须为空——它没生效。"""
    entry = TermEntry(id="e", groups=(("电线",), ("修", "墙里")), terms=("维修更换",), source="s")
    rows = TermMap([entry]).explain("装修时电线怎么处理？")
    assert rows == [
        {
            "id": "e",
            "intent": "unlabelled",
            "matched_by": "",
            "dropped": ["修"],
            "terms": ["维修更换"],
        }
    ]
    assert TermMap([entry]).match("装修时电线怎么处理？") == []


# ── 意图类别 ──────────────────────────────────────────────


def test_intent_defaults_to_unlabelled_for_old_maps(tmp_path: Path) -> None:
    """v1 词表没有 intent，必须还能载入——默认值只给老词表用。"""
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {"entries": [{"id": "e", "groups": [["挂"]], "terms": ["连接件"], "source": "s"}]}
        ),
        "utf-8",
    )
    assert load_term_map(path).entries[0].intent == "unlabelled"


def test_illegal_intent_is_rejected_not_defaulted(tmp_path: Path) -> None:
    """拼错的 intent 会静默落进 unlabelled，而那正是"分类没做"和"分类写错"分不开的地方。"""
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"id": "e", "groups": [["挂"]], "terms": ["连接件"], "source": "s",
                     "intent": "Practise"}
                ]
            }
        ),
        "utf-8",
    )
    with pytest.raises(ValueError, match="intent 非法"):
        load_term_map(path)


def test_every_v3_entry_is_labelled() -> None:
    tm = load_term_map(V3)
    assert tm is not None
    assert {e.intent for e in tm.entries} <= set(INTENTS)
    assert "unlabelled" not in {e.intent for e in tm.entries}


# ── 负例集：真正的闸门 ────────────────────────────────────


def test_v3_has_no_single_char_exemption() -> None:
    """v3 一条都不许开豁免。开了就等于把"拿语素当词"重新写回数据里。"""
    tm = load_term_map(V3)
    assert tm is not None
    assert [e.id for e in tm.entries if e.single_char_evidence] == []


def test_frozen_maps_keep_their_historical_loose_matching() -> None:
    """v1/v2 的豁免是**技术债登记**，不是推荐做法：旧报告必须还能复现。

    这条锁的是"别为了让新规则好看，顺手把历史词表也收紧"——那会让 15 题与 50 题
    两组数字同时失去可比性，而报告里没有任何一行会提示这件事。
    """
    for path in (ROOT / "rag4prefa" / "terms" / "public_to_standard.json", V2):
        tm = load_term_map(path)
        assert tm is not None
        exempt = {e.id for e in tm.entries if e.single_char_evidence}
        assert len(exempt) == 2, (path.name, exempt)
    v1 = load_term_map(ROOT / "rag4prefa" / "terms" / "public_to_standard.json")
    assert v1 is not None
    # 「会不会掉下来」靠的是单字「掉」——这就是豁免存在的理由。
    assert v1.match("墙上想挂电视和吊柜，会不会掉下来？") == ["墙面悬挂较重物体", "专用连接件"]


def test_negative_set_has_no_violation() -> None:
    """v3 必须全过。这条一旦红，说明改词表把"词同意不同"的情况又放回来了。"""
    tm = load_term_map(V3)
    assert tm is not None
    cases = _rows(NEGATIVES)
    assert len(cases) >= 5, "负例集空掉等于闸门不存在"
    fired_bad = []
    for case in cases:
        hits = {entry.id for entry in tm.hits(case["question"])}
        for entry_id in case["forbid"]:
            if entry_id in hits:
                fired_bad.append((case["kind"], case["question"], entry_id))
    assert not fired_bad


def test_negative_set_ids_exist_in_map() -> None:
    """forbid 写了不存在的 id，闸门会因为拼错而永远通过。"""
    tm = load_term_map(V3)
    assert tm is not None
    known = {entry.id for entry in tm.entries}
    for case in _rows(NEGATIVES):
        missing = set(case["forbid"]) - known
        assert not missing, f"{case['question']} 的 forbid 有词表里不存在的 id：{missing}"


def test_negative_set_would_have_caught_v2() -> None:
    """负例集必须有鉴别力：拿冻结的 v2 跑，它得抓出至少 3 条。

    只在 v3 上全绿、在 v2 上也全绿的闸门，是"没接上"的闸门。
    """
    tm = load_term_map(V2)
    assert tm is not None
    violated = [
        (case["question"], entry_id)
        for case in _rows(NEGATIVES)
        for entry_id in case["forbid"]
        if entry_id in {entry.id for entry in tm.hits(case["question"])}
    ]
    assert len(violated) >= 3, violated


# ── 词表演化：只准改触发条件，不准偷偷换词 ────────────────


def test_v3_keeps_every_term_and_source_of_v2() -> None:
    """v3 = v2 加标签 + 收紧触发条件。术语和出处必须逐条相同。

    没有这条，"收紧误触发"很容易顺手变成"换几个更好用的术语"——那已经是另一组实验，
    而且会让 v2 上跑出来的所有数字失去可比性。
    """
    v2 = load_term_map(V2)
    v3 = load_term_map(V3)
    assert v2 is not None and v3 is not None
    before = {e.id: (e.terms, e.source, e.public_side) for e in v2.entries}
    after = {e.id: (e.terms, e.source, e.public_side) for e in v3.entries}
    assert before == after


def test_v3_only_retunes_two_groups() -> None:
    """触发条件只动了两条（ceiling-scope / accessibility），其余分组一字未改。

    这条是给下一次改词表的人看的：要是 diff 里冒出第三条，说明改动的范围失控了。
    """
    v2 = load_term_map(V2)
    v3 = load_term_map(V3)
    assert v2 is not None and v3 is not None
    changed = {
        a.id for a, b in zip(v2.entries, v3.entries, strict=True) if a.groups != b.groups
    }
    assert changed == {"ceiling-scope", "accessibility"}


def test_ceiling_scope_separates_intent_from_object() -> None:
    """同一个「吊顶 + 灯具」，问品类边界要触发，问构造做法不许触发。"""
    tm = load_term_map(V3)
    assert tm is not None
    assert tm.match("集成吊顶包不包含灯具？") == ["不含电器设备"]
    assert tm.match("吊顶安装灯具需要满足什么构造要求？") == []


def test_accessibility_requires_a_passage_intent() -> None:
    """「门槛石怎么做」与「门口有坎怎么过」是两件事，注入的术语只能给后者。"""
    tm = load_term_map(V3)
    assert tm is not None
    assert tm.match("卫生间想装成轮椅能进去的，门口那个坎怎么做？") == ["过渡斜坡", "乘轮椅者"]
    assert tm.match("集成卫生间有无障碍设计要求时，门槛石位置应如何处理？") == []
