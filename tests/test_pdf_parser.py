"""pdf 解析器的回归测试。

pdf 与 docx 的根本差别是**没有结构**：只有字号、加粗、坐标。所以这里的用例都围绕
"在信息更少的情况下，章节树还能不能立起来"，以及"会不会把不是标题的东西当标题"。

样本用 pymupdf 现造，不用 reportlab：多一个测试依赖不值得，而且能精确控制每个字的
字号、坐标与字体，比"找一份真实 pdf 期望它长什么样"稳定得多。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="pdf 解析需要 pymupdf")

from rag4prefa.adapters.parsers import parse_document  # noqa: E402
from rag4prefa.adapters.parsers.headings import (  # noqa: E402
    MAX_LEVEL,
    MAX_SIZE_LEVELS,
    build_sections,
    text_heading_level,
)
from rag4prefa.adapters.parsers.pdf import parse_pdf  # noqa: E402
from rag4prefa.adapters.vectors import NumpyStore  # noqa: E402
from rag4prefa.chunking import build_chunks  # noqa: E402
from rag4prefa.config import Settings  # noqa: E402
from rag4prefa.ingest import ingest_file  # noqa: E402
from tests.conftest import FakeEmbedder  # noqa: E402

_CN = "china-s"  # pymupdf 内置简体中文字体
_CN_BODY = 10.0

_INTRO = "本次方案覆盖主体结构与外立面，重点说明悬挑部位的施工工艺与质量控制要点。"
_FRAME = "本工程为装配式框架结构，地上十二层，标准层层高四点二米，抗震设防烈度六度。"
_CANTILEVER = "悬挑长度最大九米，采用缓粘结预应力梁，需控制张拉时机与摩阻损失。"
_EN_BODY = "This document describes the construction method for large cantilevers."
_EN_SCOPE = "The scope covers the main structure and the facade system only."


def _make_pdf(path: Path, pages: list[list[tuple[float, str, float, str]]]) -> Path:
    """按 (y, 文本, 字号, 字体) 精确落字，坐标可控才能测"页眉区"。"""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()  # 默认 A4：595 x 842
        for y, text, size, font in lines:
            page.insert_text((72, y), text, fontname=font, fontsize=size)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def structure_pdf(tmp_path: Path) -> Path:
    """三级结构：章（18pt）/ 节（14pt）/ 相对级别（（一）12pt）+ 正文（10pt）。"""
    return _make_pdf(
        tmp_path / "方案.pdf",
        [
            [
                (60, "美睿金座装配式计算书", 20, _CN),
                (110, "第一章 工程概况", 18, _CN),
                (140, _INTRO, _CN_BODY, _CN),
                (170, "1.1 结构设计概况", 14, _CN),
                (200, _FRAME, _CN_BODY, _CN),
                (240, "（一）悬挑部位", 12, _CN),
                (270, _CANTILEVER, _CN_BODY, _CN),
            ]
        ],
    )


def test_pdf_rebuilds_heading_tree_from_font_size(structure_pdf: Path) -> None:
    """字号是 pdf 里唯一可靠的结构线索；层级与路径都要对得上。"""
    sections = parse_pdf(structure_pdf)
    by_title = {s.title: s for s in sections}

    assert "第一章 工程概况" in by_title
    assert by_title["第一章 工程概况"].level == 1
    assert by_title["第一章 工程概况"].path == []

    assert by_title["1.1 结构设计概况"].level == 2
    assert by_title["1.1 结构设计概况"].path == ["第一章 工程概况"]

    # （一）是相对级别：挂在"最近一个绝对级别标题"的下一层，而不是固定第 1 级
    assert by_title["（一）悬挑部位"].level == 3
    assert by_title["（一）悬挑部位"].path == ["第一章 工程概况", "1.1 结构设计概况"]

    assert "张拉时机" in by_title["（一）悬挑部位"].text, "正文要归到它前面的标题下"


def test_pdf_header_and_page_number_do_not_become_content(tmp_path: Path) -> None:
    """页眉与页码每页都有，漏进正文会污染正文并让 BM25 的文档频率失真。"""
    header = "美睿金座项目装配式计算书（内部资料）"
    body = "悬挑梁采用缓粘结预应力体系，张拉端设置于柱侧，锚固端封锚后做防腐处理。"
    pages = [
        [
            (30, header, 9, _CN),
            (120, "第一章 施工准备", 16, _CN),
            (150, body, _CN_BODY, _CN),
            (800, f"第 {n} 页 共 2 页", 9, _CN),
        ]
        for n in (1, 2)
    ]
    path = _make_pdf(tmp_path / "两页.pdf", pages)

    sections = parse_pdf(path)
    joined = "\n".join(s.title + "\n" + s.text for s in sections)

    assert "第一章 施工准备" in joined
    assert "张拉端设置于柱侧" in joined
    assert "内部资料" not in joined, "页眉不该进正文"
    assert "共 2 页" not in joined, "页码不该进正文"


def test_pdf_section_carries_physical_page(tmp_path: Path) -> None:
    """标题所在的物理页（1-based）要透传到 Section.page，供引用侧栏标注「第 N 页」。"""
    pages = [
        [(80, "第一章 工程概况", 16, _CN), (140, _INTRO, _CN_BODY, _CN)],
        [(80, "第二章 结构计算", 16, _CN), (140, _FRAME, _CN_BODY, _CN)],
    ]
    path = _make_pdf(tmp_path / "跨页.pdf", pages)
    sections = parse_pdf(path)
    by_title = {s.title: s for s in sections}

    assert by_title["第一章 工程概况"].page == 1
    assert by_title["第二章 结构计算"].page == 2


def test_pdf_chunks_of_a_cross_page_section_report_their_own_pages(tmp_path: Path) -> None:
    """一节跨多页时，每个片段报**自己**的起始页，而不是章节标题页。

    2026-10-27 复审 P1-2 的原始缺陷：解析器已经把每行的物理页透传下来了，但
    `build_sections()` 在正文分支只存文本、把页码丢了，于是整节只剩"标题页"，
    引用卡片一律显示第 1 页——用户翻过去找不到原文，引用不可核。
    这条用例必须真的跨页并真的切出多块，否则"断言等于标题页"仍然成立。
    """
    body = [
        "预制柱的纵向受力钢筋采用套筒灌浆连接，接缝处坐浆层厚度控制在二十毫米。",
        "叠合梁后浇段混凝土强度等级比预制构件提高一级，浇筑前需清除浮浆。",
        "外墙挂板的连接件需做防腐处理，安装完成后逐块检查螺栓扭矩。",
    ]
    # y=140 起：页面顶部 10%（约 84pt）是页眉区，正文落在那里会被当页眉丢掉
    pages = [
        [(80, "第一章 工程概况", 16, _CN), (140, body[0], _CN_BODY, _CN)],
        [(140, body[1], _CN_BODY, _CN)],
        [(140, body[2], _CN_BODY, _CN)],
    ]
    path = _make_pdf(tmp_path / "节跨三页.pdf", pages)

    (section,) = [s for s in parse_pdf(path) if s.page == 1 and s.text]

    assert section.text.count("\n") >= 2, "正文要分布在不同页，否则这条用例证明不了什么"
    # max_tokens 要略大于单行 token 数：太小会把一行硬切成两块，
    # 用例就变成在数碎片，而不是在验页码。
    chunks = build_chunks([section], "doc", max_tokens=45, min_tokens=1)
    assert len(chunks) == 3, f"应切出 3 块（每页一块），实际 {len(chunks)}"
    assert [c.page for c in chunks] == [1, 2, 3], "片段页码必须是它自己所在的物理页"


def test_pdf_paragraph_first_line_is_not_treated_as_heading(tmp_path: Path) -> None:
    """（一）开头的**段落首行**不是标题。

    这是最容易写错的一处：看到编号就提级，整份文档会变成"每段一节"——
    章节路径全错，chunk_id 跟着全变。判据必须是"先有视觉证据（更大或加粗）"。
    """
    para = "（一）第一申报单位应为在本省注册的独立法人单位，多个单位联合申报的，一般不超过五家"
    path = _make_pdf(
        tmp_path / "公文.pdf",
        [[
            (60, "第一章 总则", 16, _CN),
            (100, para, _CN_BODY, _CN),  # 与正文同字号、不加粗 → 不是标题
            (125, "（含第一申报单位）。申报单位应拥有相关技术的自主知识产权。", _CN_BODY, _CN),
        ]],
    )

    sections = parse_pdf(path)
    titles = [s.title for s in sections]

    assert "第一章 总则" in titles
    assert not any(t.startswith("（一）") for t in titles), f"段落首行被当成了标题：{titles}"
    assert "不超过五家" in "\n".join(s.text for s in sections)


def test_pdf_bold_at_body_size_still_counts_as_heading(tmp_path: Path) -> None:
    """标题与正文同字号、只靠加粗区分时也要认出来（英文技术文档的常见排版）。"""
    path = _make_pdf(
        tmp_path / "en.pdf",
        [[
            (100, "1 Introduction", 10, "hebo"),  # Helvetica-Bold
            (130, _EN_BODY, 10, "helv"),
            (160, "2 Scope", 10, "hebo"),
            (190, _EN_SCOPE, 10, "helv"),
        ]],    )

    titles = [s.title for s in parse_pdf(path)]
    assert "1 Introduction" in titles
    assert "2 Scope" in titles


def test_pdf_without_any_heading_degrades_to_single_section(tmp_path: Path) -> None:
    """认不出标题时退化成一节，而不是抛出异常或每行一节。"""
    path = _make_pdf(
        tmp_path / "无标题.pdf",
        [[
            (100, "本文件说明施工过程中的注意事项与质量验收要求。", _CN_BODY, _CN),
            (130, "现场应按图施工，隐蔽工程须经监理验收后方可覆盖。", _CN_BODY, _CN),
            (160, "材料进场应提供合格证与检测报告，不合格品严禁使用。", _CN_BODY, _CN),
        ]],
    )

    sections = parse_pdf(path)
    assert len(sections) == 1
    assert sections[0].title == "前言"
    assert "隐蔽工程" in sections[0].text


def test_pdf_without_text_layer_is_rejected_with_reason(tmp_path: Path) -> None:
    """扫描件没有文字层：明确报错，而不是静默入库一个空文档。

    静默的后果比报错严重：用户看到"导入成功"、检索却什么都搜不到，
    会去怀疑检索参数，而不是怀疑源文件。
    """
    doc = pymupdf.open()
    doc.new_page()
    path = tmp_path / "扫描件.pdf"
    doc.save(str(path))
    doc.close()

    with pytest.raises(ValueError, match="扫描件"):
        parse_pdf(path)


def test_pdf_garbled_text_layer_is_rejected_with_reason(tmp_path: Path) -> None:
    """取得到字符、取出来的却不是汉字：必须拒绝，且说清"换版本"而不是"做 OCR"。

    这是真实撞到的局面：方正书版导出的 pdf 缺 ToUnicode 映射，文字层里存的是字形
    编号，`get_text()` 会返回 `!"#$%&'()*` 这样的串。它"有 1 万多个字符"，
    只查字符数的闸门完全拦不住——那些乱码会一路入库，把 BM25 的文档频率
    和向量空间一起拖坏，而且全程没有任何报错。
    """
    doc = pymupdf.open()
    noisy = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for _ in range(4):
        page = doc.new_page()
        for row in range(20):
            page.insert_text((60, 100 + row * 18), noisy, fontsize=10)
    path = tmp_path / "方正书版乱码.pdf"
    doc.save(str(path))
    doc.close()

    with pytest.raises(ValueError, match="ToUnicode") as exc:
        parse_pdf(path)
    # 用户能做的事和扫描件不同：原件多半是可复制的，换一份能选中文字的版本即可
    assert "OCR" in str(exc.value)


def test_pdf_english_document_passes_readability_check(tmp_path: Path) -> None:
    """英文文档不能被可读性判据误伤：汉字占比为 0 是正常的，靠拉丁词密度放行。"""
    doc = pymupdf.open()
    page = doc.new_page()
    for row in range(25):
        page.insert_text(
            (60, 80 + row * 16),
            "The precast column shall be erected after the grout sleeve is verified.",
            fontsize=10,
        )
    path = tmp_path / "english.pdf"
    doc.save(str(path))
    doc.close()

    sections = parse_pdf(path)  # 不抛异常即通过
    assert sum(len(s.text) for s in sections) > 300


def test_pdf_short_document_skips_readability_check(tmp_path: Path) -> None:
    """样本太短时不做可读性判断：几十个字撑不起一个占比，误判成乱码更糟。"""
    doc = pymupdf.open()
    page = doc.new_page()
    lines = [
        "工程概况：本工程地上十二层，标准层层高四点二米。",
        "1.1 结构体系为装配式框架，预制率不低于百分之四十。",
        "1.2 抗震设防烈度六度，设计使用年限五十年。",
        "1.3 预制构件包括叠合板、预制楼梯与预制外墙挂板。",
        "编制依据：GB 50010、GB 50204 及地方标准。",
    ]
    for row, line in enumerate(lines):
        page.insert_text((60, 100 + row * 18), line, fontname=_CN, fontsize=10)
    path = tmp_path / "短文档.pdf"
    doc.save(str(path))
    doc.close()

    sections = parse_pdf(path)  # 只有一百多字，没过可读性闸门的门槛，照样放行
    assert sum(len(s.text) for s in sections) < 300


def test_pdf_toc_page_is_skipped_entirely(tmp_path: Path) -> None:
    """目录页整页丢弃：它的条目就是"编号一行、标题一行"，会凭空长出假章节树。

    YJK 手册（368 页）的目录有 5 页、上百条，全部被当成标题——它们是 461 节里
    绝大多数空节点的来源，而且因为编号逐级递增，会构造出一棵和正文无关的树。
    """
    path = _make_pdf(
        tmp_path / "带目录.pdf",
        [
            [  # 目录页：点线引导符
                (60, "目 录", 18, _CN),
                (100, "1.1 装配式设计要点............................. 1", 10, _CN),
                (120, "1.2 结构计算................................... 5", 10, _CN),
                (140, "1.3 构件拆分................................... 9", 10, _CN),
                (160, "1.4 施工图出图................................ 12", 10, _CN),
            ],
            [  # 正文页
                (60, "第1 章 概述", 18, _CN),
                (100, _FRAME, _CN_BODY, _CN),
            ],
        ],
    )

    sections = parse_pdf(path)
    titles = [s.title for s in sections]
    assert "第1 章 概述" in titles
    # 目录文本一个字都不该进正文：点线行、目录里的编号都会污染检索。
    # 不跳目录时它们会被当成正文吸进"前言"节——所以断言的是"文本不在"，
    # 而不是"标题不在"（目录行与正文同字号时，它本来也成不了标题）。
    body = "\n".join(s.title + s.text for s in sections)
    assert "....." not in body, "目录页的点线内容不能进入正文"
    assert "装配式设计要点" not in body, "目录条目不能当正文"


def test_pdf_number_only_line_is_merged_with_following_heading(tmp_path: Path) -> None:
    """编号与标题名分两行的排版要拼回一条标题。

    pdf 里 `1.1` 单独一行、`装配式设计要点` 下一行很常见。不合并会得到两个节点：
    前一个只有编号、正文为空，后一个层级由字号排名决定——章节树长成
    `1.1 > 装配式设计要点`，引用时只能显示"第 1.1 节"。
    YJK 手册实测 461 节里有 195 个这样的空节点。
    """
    path = _make_pdf(
        tmp_path / "编号分行.pdf",
        [
            [
                (60, "第1 章 概述", 18, _CN),
                (100, "1.1", 16, _CN),
                (140, "装配式设计要点", 14, _CN),
                (180, _FRAME, _CN_BODY, _CN),
            ]
        ],
    )

    sections = parse_pdf(path)
    titles = [s.title for s in sections]
    assert "1.1 装配式设计要点" in titles, titles
    assert "1.1" not in titles, "编号不能单独成节"
    merged = next(s for s in sections if s.title == "1.1 装配式设计要点")
    assert merged.level == 2, "层级该由编号决定，而不是由下一行的字号排名决定"
    assert _FRAME in merged.text


def test_pdf_chapter_heading_with_space_before_unit_is_level_one(tmp_path: Path) -> None:
    """「第1 章」中间的空格不能让它失去"章"的身份。

    中英混排的排版软件常在这种地方插空格；一旦认不出 `第X章`，一级标题会退回
    "按字号排名"的结果，层级跟着整棵树错位（YJK 手册的 `第1 章概述` 就落在 L3）。

    这里的字号是**故意反过来**排的（节比章更大）：只有认编号才能得到正确的 1/2，
    按字号排名会给出相反的 2/1。字号与逻辑层级一致时两种判据结果相同，
    那样的样例证明不了任何事。
    """
    path = _make_pdf(
        tmp_path / "空格章节.pdf",
        [
            [
                (60, "第1 章 概述", 14, _CN),  # 章：字号较小
                (100, _FRAME, _CN_BODY, _CN),
                (140, "第 2 节 计算参数", 18, _CN),  # 节：字号反而更大
                (180, _INTRO, _CN_BODY, _CN),
            ]
        ],
    )

    by_title = {s.title: s for s in parse_pdf(path)}
    assert by_title["第1 章 概述"].level == 1, "认编号才给得出 1 级"
    assert by_title["第 2 节 计算参数"].level == 2


def test_formula_number_with_closing_paren_is_not_a_heading() -> None:
    """`1）` 是公式编号，不是标题；`（1）` 成对出现才算。

    上海计算细则实测：`1）预制柱比例 = ...` 这类行被判成一级标题，
    22 个只有几十字的公式碎片各自成章，把章节树撑成一片噪声。
    """
    assert text_heading_level("1）预制柱比例") is None
    assert text_heading_level("（1）预制柱比例") == 0  # 成对括号 = 相对级别标题
    assert text_heading_level("1.1 装配式设计要点") == 2


def test_build_sections_clamps_level_to_domain_limit() -> None:
    """解析器给出的级别不能把章节树撑破 `Section.level <= 9`。

    YJK 手册（368 页）实测撞出过 level=10：pdf 的层级来自"字号排名"，
    而一份真实文档里的字号远不止 9 种，全换算成层级就会排出十几层。
    夹紧点放在组装层，这样 docx（Word 大纲级别最多 9）与 pdf 共用一道口子。
    """
    sections = build_sections([(12, "过深的标题"), (None, "正文")])
    assert [s.level for s in sections] == [MAX_LEVEL]


def test_pdf_font_size_ranks_are_capped_to_max_levels(tmp_path: Path) -> None:
    """字号排名只取前若干个字号：多出来的字号落进最深一层，而不是继续往下钻。"""
    sizes = [20, 18, 16, 14, 13, 12, 11]  # 7 种"比正文大"的字号，正文 10
    doc = pymupdf.open()
    page = doc.new_page()
    y = 90
    for index, size in enumerate(sizes):
        page.insert_text((60, y), f"标题层{index}", fontname=_CN, fontsize=size)
        y += size + 14
    for _ in range(6):
        page.insert_text((60, y), _FRAME, fontname=_CN, fontsize=_CN_BODY)
        y += 18
    path = tmp_path / "多字号.pdf"
    doc.save(str(path))
    doc.close()

    sections = parse_pdf(path)
    assert sections
    assert max(s.level for s in sections) <= MAX_SIZE_LEVELS


def test_pdf_table_cells_are_kept_as_rows(tmp_path: Path) -> None:
    """表格按 "|" 保留成行：工程文档里的参数表是主要检索对象之一。"""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 80), "第一章 构件参数", fontname=_CN, fontsize=16)

    left, top, width, height = 72.0, 120.0, 120.0, 24.0
    cells = [
        ["构件", "截面"],
        ["悬挑梁", "400x800"],
        ["屋面梁", "300x700"],
        ["框架柱", "600x600"],
    ]
    rows, cols = len(cells), 2
    for r in range(rows + 1):
        page.draw_line((left, top + r * height), (left + cols * width, top + r * height))
    for c in range(cols + 1):
        page.draw_line((left + c * width, top), (left + c * width, top + rows * height))
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            page.insert_text(
                (left + c * width + 6, top + r * height + 16),
                text,
                fontname=_CN,
                fontsize=9,
            )
    path = tmp_path / "表格.pdf"
    doc.save(str(path))
    doc.close()

    sections = parse_pdf(path)
    joined = "\n".join(s.title + "\n" + s.text for s in sections)
    assert "构件 | 截面" in joined
    assert "悬挑梁 | 400x800" in joined
    # 表格不参与"正文字号"统计时，这个标题会被当成正文、接着被当页边短行删掉——
    # 整篇退化成"前言"一节。锁定这条链：标题必须还在，而且是标题。
    assert any(s.title == "第一章 构件参数" and s.level == 1 for s in sections)
    # 表格与正文按 y 排回原顺序：先收表格再收文本会让整页表格都跑到标题前面
    assert joined.index("第一章 构件参数") < joined.index("构件 | 截面")


def test_pdf_split_paragraph_is_stitched_back(tmp_path: Path) -> None:
    """同一段被 pdf 拆进两个 block 时要拼回去，真正的两段之间要保留分隔。

    不拼的后果不是"难看"，而是词被切断："住房城乡" / "建设科技" 中间插换行，
    分词与向量都会把它当成两个词。
    """
    path = _make_pdf(
        tmp_path / "断行.pdf",
        [[
            (100, "第一章 施工方案", 16, _CN),
            (130, "悬挑梁采用缓粘结预应力体系，张拉端设置于柱侧，", 10, _CN),
            (143, "锚固端封锚后做防腐处理。", 10, _CN),  # 紧贴上一行：同一段
            (200, "材料进场应提供合格证与检测报告。", 10, _CN),  # 间距大：另一段
        ]],
    )

    text = "\n".join(s.text for s in parse_pdf(path))
    assert "柱侧，锚固端封锚" in text, "同一段的换行没有拼回去"
    assert "防腐处理。锚固" not in text
    assert "防腐处理。\n材料进场" in text, "段与段之间不该被拼成一行"


def test_parse_document_dispatches_by_suffix_and_rejects_others(
    structure_pdf: Path, tmp_path: Path
) -> None:
    """.pdf 走 pdf 解析器；不认识的格式明确拒绝。"""
    assert parse_document(structure_pdf), "pdf 应该被分派到 parse_pdf"

    other = tmp_path / "说明.txt"
    other.write_text("纯文本", encoding="utf-8")
    with pytest.raises(ValueError, match="不支持的文档格式"):
        parse_document(other)


def test_pdf_ingest_produces_chunks_with_chapter_path(
    conn: sqlite3.Connection, settings: Settings, structure_pdf: Path
) -> None:
    """整条链路：pdf → 章节 → 分块 → 向量。章节路径要真的进到 chunk 里。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    outcome = ingest_file(
        conn,
        store,
        FakeEmbedder(),
        settings,
        path=structure_pdf,
        project_name="PDF项目",
    )

    assert outcome.status == "indexed"
    assert outcome.chunks > 0
    assert store.count() == outcome.chunks

    rows = conn.execute("SELECT chapter_path FROM chunks").fetchall()
    paths = " ".join(str(r[0]) for r in rows)
    assert "第一章 工程概况" in paths, "章节路径必须进 chunk，否则引用只会指到整篇"


@pytest.mark.skipif(
    sys.platform != "win32", reason="这份样本是本机路径下的真实文件，仅用于本地冒烟"
)
def test_real_pdf_smoke() -> None:
    """真实 pdf 冒烟：确认在真实排版上也能立起章节树，而不只是在造的样本上。

    刻意 skip 非 Windows：仓库里没有这份文件，CI 上跑不到；它的价值是本地改动解析器后
    立刻能验证，而不是作为回归门槛。
    """
    sample = Path("F:/project/课题/Ref/省建设厅关于组织申报2026年度建设科研项目的通知.pdf")
    if not sample.exists():
        pytest.skip("本地样本不存在")
    titles = [s.title for s in parse_pdf(sample)]
    assert any("申报内容" in t for t in titles)
    assert not any(t.strip().isdigit() for t in titles)
