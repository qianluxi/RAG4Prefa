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


def test_chunk_falls_back_to_section_page_without_line_pages() -> None:
    """没有逐行页码时退回节起始页——docx 与外部构造的 Section 走这条路。

    **不参与 chunk_id**：页码是展示字段（见 test_page_is_not_part_of_chunk_identity）。
    """
    section = Section(
        title="1.1 结构概况",
        kind=SectionKind.section,
        level=2,
        path=["第一章"],
        text="正文内容。",
        page=7,
    )
    (chunk,) = build_chunks([section], "doc", max_tokens=100, min_tokens=20)
    assert chunk.page == 7


def test_chunk_reports_its_own_page_not_the_section_title_page() -> None:
    """一节跨 4~7 页时，每个片段要报**自己**的起始页，不能全标成标题页。

    这是 2026-10-27 复审 P1-2 的核心：`Section.page` 是标题所在页，而"4.2 悬挑构造"
    从第 4 页开始、正文排到第 7 页。拿标题页给这一节所有片段做页码，用户按"第 4 页"
    翻过去找不到那段话——工程文档知识库里这等于引用不可核。
    """
    lines = [f"第{i}条：本段用于把这一节撑到跨页。" for i in range(1, 13)]
    # 每 3 行一页：4,4,4,5,5,5,6,6,6,7,7,7
    pages = [page for page in (4, 5, 6, 7) for _ in range(3)]
    section = Section(
        title="4.2 悬挑构造",
        kind=SectionKind.section,
        level=2,
        path=["第四章"],
        text="\n".join(lines),
        page=4,  # 标题在第 4 页
        line_pages=pages,
    )
    chunks = build_chunks([section], "doc", max_tokens=40, min_tokens=1)

    assert len(chunks) > 1, "样本必须真的被切成多块，否则断言恒真"
    assert [c.page for c in chunks] == sorted(c.page for c in chunks), "页码应随正文单调不减"
    assert {c.page for c in chunks} == {4, 5, 6, 7}, "每一页的片段都要出现，不能全被标成 4"
    # 逐块核对：片段的页码 = 它起始那一行所在页。
    # 片段里的换行已被切句丢弃，所以拿前缀去认行，而不是按 "\n" 取首行。
    line_page = dict(zip(lines, pages, strict=True))
    for chunk in chunks:
        lead = chunk.text[:8]
        first_line = next(line for line in lines if line.startswith(lead))
        assert chunk.page == line_page[first_line], f"{lead} 的页码标错了"


def test_page_lookup_survives_pieces_that_carry_inner_newlines() -> None:
    """片段内部带着换行时页码仍然要对，游标不能按"带换行的长度"推进。

    切句是按 `(?<=[。！？；\\n])` 做的，行尾没有句号的行会连换行一起成为一句，
    拼回的片段里就留着换行。而定位是在**去掉换行**的扁平文本上做的——游标多走
    那几个换行字符，后面几块就再也找不到，页码会静默退回节起始页：
    看起来有页码，全是第一页（2026-10-27 真实语料上就是这样交错跳回去的）。
    """
    # 故意不带句号结尾：这样换行才会留在片段内部
    lines = [f"第{i}段没有句号结尾的正文" for i in range(1, 13)]
    pages = [page for page in (1, 2, 3, 4) for _ in range(3)]
    section = Section(
        title="1.1 甲",
        kind=SectionKind.section,
        level=2,
        path=["第一章"],
        text="\n".join(lines),
        page=1,
        line_pages=pages,
    )
    chunks = build_chunks([section], "doc", max_tokens=40, min_tokens=1)

    assert any("\n" in c.text for c in chunks), "样本没造出带换行的片段，这条用例就白写了"
    assert [c.page for c in chunks] == [1, 2, 3, 4]


def test_mismatched_line_pages_fall_back_instead_of_guessing() -> None:
    """行数与页码条数对不上时退回节起始页——错位的页码比没有页码更容易误导。

    外部构造的 Section、或将来某个解析器改了换行处理，都可能让两者不同长；
    这时按行号硬取会取到隔壁行的页码，而它看起来是可信的。
    """
    section = Section(
        title="1.1 甲",
        kind=SectionKind.section,
        level=2,
        text="第一行。\n第二行。\n第三行。",
        page=9,
        line_pages=[3, 4],  # 少一条
    )
    chunks = build_chunks([section], "doc", max_tokens=100, min_tokens=1)
    assert [c.page for c in chunks] == [9]


def test_page_is_not_part_of_chunk_identity() -> None:
    """页码变了（排版微调）不该让同一段正文换 id——否则内容寻址就退化成位置寻址。"""
    def build(page: int | None) -> str:
        section = Section(
            title="1.1 甲",
            kind=SectionKind.section,
            level=2,
            path=["第1章"],
            text="同一段正文。",
            page=page,
        )
        return build_chunks([section], "doc", max_tokens=100, min_tokens=20)[0].id

    assert build(7) == build(8)
    assert build(None) == build(3)


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
