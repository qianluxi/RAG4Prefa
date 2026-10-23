from __future__ import annotations

from rag4prefa.chunking import build_chunks, contextual_text
from rag4prefa.domain import Section, SectionKind
from rag4prefa.tokens import estimate_tokens, split_by_tokens


def _section(title: str, text: str, path: list[str], level: int = 2) -> Section:
    return Section(
        title=title,
        kind=SectionKind.section if level == 2 else SectionKind.chapter,
        level=level,
        path=path,
        text=text,
    )


def test_empty_sections_do_not_produce_chunks() -> None:
    chunks = build_chunks([_section("空标题", "   ", [])], "doc", max_tokens=100, min_tokens=20)
    assert chunks == []


def test_long_section_is_split_within_budget() -> None:
    text = "。".join(["悬挑构件需要控制挠度与裂缝宽度"] * 60) + "。"
    chunks = build_chunks(
        [_section("1.1 悬挑", text, ["第1章"])], "doc", max_tokens=100, min_tokens=20
    )
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    assert all(c.chapter_path == ["第1章", "1.1 悬挑"] for c in chunks)


def test_small_siblings_merge_within_same_parent() -> None:
    sections = [
        _section("1.1 甲", "短内容。", ["第1章"]),
        _section("1.2 乙", "另一段短内容。", ["第1章"]),
    ]
    chunks = build_chunks(sections, "doc", max_tokens=200, min_tokens=50)
    assert len(chunks) == 1
    assert "甲 / 1.2 乙" in chunks[0].title  # 合并后标题保留双方，不丢信息


def test_small_sections_do_not_merge_across_parents() -> None:
    sections = [
        _section("1.1 甲", "短内容。", ["第1章"]),
        _section("2.1 乙", "另一段短内容。", ["第2章"]),
    ]
    chunks = build_chunks(sections, "doc", max_tokens=200, min_tokens=50)
    assert len(chunks) == 2


def test_chunk_ids_are_deterministic() -> None:
    sections = [_section("1.1 甲", "内容。" * 20, ["第1章"])]
    first = build_chunks(sections, "doc-a", max_tokens=100, min_tokens=20)
    second = build_chunks(sections, "doc-a", max_tokens=100, min_tokens=20)
    assert [c.id for c in first] == [c.id for c in second]
    other = build_chunks(sections, "doc-b", max_tokens=100, min_tokens=20)
    assert other[0].id != first[0].id


def test_chunk_id_follows_content_not_position() -> None:
    """内容寻址：改一段只该让那一段换号，别的段不受影响。

    这是"先写索引、后切事实"能安全的前提。位置寻址（doc_id|ordinal）下，
    在中间插一段会让后面所有片段换号，而换号前那些位置可能已经指向别的正文——
    索引与事实各说各话，且不报任何错。
    """
    head = _section("1.1 甲", "甲段内容，关于悬挑。" * 8, ["第1章"])
    tail = _section("1.2 乙", "乙段内容，关于幕墙。" * 8, ["第1章"])
    before = build_chunks([head, tail], "doc", max_tokens=100, min_tokens=20)
    assert len(before) == 2

    # 在中间插入一段完全不同的内容
    middle = _section("1.1b 丙", "丙段内容，关于防水。" * 8, ["第1章"])
    after = build_chunks([head, middle, tail], "doc", max_tokens=100, min_tokens=20)

    ids_before = {c.id for c in before}
    ids_after = {c.id for c in after}
    assert len(ids_after - ids_before) == 1, "插入一段只该新增一个 id"
    assert len(ids_before & ids_after) == 2, "原有两段的 id 必须保持不变"


def test_identical_paragraphs_get_distinct_ids() -> None:
    """同一章节里出现完全相同的两段（合并单元格解析后常见）不能撞成一个。"""
    text = "同一段文字。" * 10
    chunks = build_chunks([_section("1.1 甲", f"{text}\n{text}", ["第1章"])], "doc",
                          max_tokens=60, min_tokens=10)
    assert len(chunks) > 1
    assert len({c.id for c in chunks}) == len(chunks), "id 必须唯一，否则会丢片段"


def test_contextual_text_carries_path() -> None:
    sections = [_section("1.1 悬挑", "正文。", ["第1章"])]
    chunk = build_chunks(sections, "doc", max_tokens=100, min_tokens=10)[0]
    assert contextual_text(chunk).startswith("第1章 > 1.1 悬挑")


def test_split_by_tokens_respects_sentence_boundaries() -> None:
    text = "第一句话。第二句话。第三句话。"
    pieces = split_by_tokens(text, 6)
    assert all(p.endswith("。") for p in pieces)
    assert "".join(pieces) == text
    assert sum(estimate_tokens(p) for p in pieces) >= estimate_tokens(text) - len(pieces)
