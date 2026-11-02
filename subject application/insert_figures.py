"""把 figs 下的三张图插入申请书 docx 指定位置，并加居中图题。

插入点：
  图 1  一、1.1 研究目的 末尾（总目标段之后）
  图 2  三、技术路线 主链路介绍段之后、①知识治理层之前
  图 3  二、(一) 4. 多轮交互智能体与效果评价 末尾
"""

from copy import deepcopy
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

HERE = Path(__file__).parent
DOC = HERE / "面向公众的装配式内装知识服务与交互智能体关键技术研究-申请书26.10.31.docx"
FIGS = HERE / "figs"
WIDTH = Cm(15.5)


def new_par_after(anchor: Paragraph) -> Paragraph:
    """在 anchor 之后插入一个空段落，沿用其段落格式（缩进另置零）。"""
    anchor._p.addnext(deepcopy(anchor._p))
    p = Paragraph(anchor._p.getnext(), anchor._parent)
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    return p


def add_figure(anchor: Paragraph, png: Path, caption: str) -> None:
    # 图片段：居中、无首行缩进、段前段后各 6 磅
    pic_p = new_par_after(anchor)
    pf = pic_p.paragraph_format
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.first_line_indent = Cm(0)
    pf.left_indent = Cm(0)
    pf.space_before = Pt(6)
    pf.space_after = Pt(3)
    pic_p.add_run().add_picture(str(png), width=WIDTH)

    # 图题段：五号仿宋居中
    cap_p = new_par_after(pic_p)
    cf = cap_p.paragraph_format
    cf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cf.first_line_indent = Cm(0)
    cf.left_indent = Cm(0)
    cf.space_before = Pt(0)
    cf.space_after = Pt(8)
    run = cap_p.add_run(caption)
    run.font.size = Pt(10.5)
    run.font.name = "仿宋"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "仿宋")


def main() -> None:
    d = docx.Document(str(DOC))
    tbl = d.tables[0]

    # 断言三处锚点，防止插错位置
    c1 = tbl.rows[1].cells[0].paragraphs
    assert c1[5].text.startswith("总目标"), c1[5].text[:20]
    assert c1[6].text.startswith("1.2"), c1[6].text[:20]

    c3 = tbl.rows[3].cells[0].paragraphs
    assert "评价方法" in c3[17].text, c3[17].text[:20]
    assert c3[18].text.startswith("（二）"), c3[18].text[:20]

    c5 = tbl.rows[5].cells[0].paragraphs
    assert "四阶段" in c5[1].text, c5[1].text[:20]
    assert c5[2].text.startswith("①知识治理层"), c5[2].text[:20]

    add_figure(c1[5], FIGS / "figure1_research_framework.png", "图 1  项目总体研究框架")
    add_figure(c5[1], FIGS / "figure2_system_architecture.png", "图 2  系统总体架构与技术路线")
    add_figure(c3[17], FIGS / "figure3_public_interaction_flow.png",
               "图 3  典型公众咨询的多轮交互与可信处置流程")

    d.save(str(DOC))

    # 复检
    d2 = docx.Document(str(DOC))
    for ri, name in ((1, "图1"), (5, "图2"), (3, "图3")):
        ps = d2.tables[0].rows[ri].cells[0].paragraphs
        caps = [p.text.strip() for p in ps if p.text.strip().startswith("图 ")]
        shapes = sum(1 for p in ps if p._p.findall(qn("w:r") + "/" + qn("w:drawing")))
        print(name, "图题:", caps, "| 内嵌图片段:", shapes)


if __name__ == "__main__":
    main()
