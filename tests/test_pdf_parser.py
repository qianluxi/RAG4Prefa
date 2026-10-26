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
from rag4prefa.adapters.parsers.pdf import parse_pdf  # noqa: E402
from rag4prefa.adapters.vectors import NumpyStore  # noqa: E402
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
