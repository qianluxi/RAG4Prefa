"""按省级申请书（26.11.02）填写杭州市级《建设科研项目立项申报书》。

两份表结构一一对应（一~九），市级模板只是把大段内容留成空白单元格。这个脚本做的是
「整段搬运」而不是「重新排版」：省级那份的段落元素整体深拷贝过去，字体、字号、
首行缩进、行距、对齐全部照旧；三张插图按原尺寸重新挂到目标文档（换 rId）；
表头、边框、章节编号、十~十二的意见栏一个字都不动。

三步用法（脚本不碰二进制 .doc，Word/WPS 的格式转换手工做）：

  1. 把 (市级).doc 用 Word/WPS 另存为 city_tpl.docx（文件→另存为→.docx）；
  2. python "subject application/fill_city.py" 省级.docx city_tpl.docx 输出.docx
  3. 把输出.docx 再另存回 .doc 即可。

搬运过程中顺手修掉四处「模板与内容不匹配」的地方（都只动 hRule / 断行 / 列内字号，
不改模板的边框与网格）：

  - 参考文献 [25] 那条 95 字符的公积金链接会把整张表从 16.2cm 撑到 23cm、右半边跑到
    纸外 → 给含超长西文串的段落开「允许西文在单词中间换行」，并把表格设成固定列宽；
  - 图片段落同时设了「居中」和「首行缩进 0.74cm」，15.5cm 的图会被推宽 → 去掉缩进；
  - 正文行有 9~13cm 的最小高度，填满后章节末尾会留半页空白 → 正文行改 hRule=auto；
  - 组员行的 0.7cm 固定高度会把两行字裁掉 → 改 atLeast，并把窄列字号降到 9pt
    （「浙江绿城建筑设计有限公司」在 4.4cm 列里才排得下一行）。
"""

from __future__ import annotations

import re
import sys
from copy import deepcopy
from io import BytesIO
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "面向公众的装配式内装知识服务与交互智能体关键技术研究-申请书26.11.02.docx"
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
TPL = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "city_tpl.docx"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "city_filled.docx"

R_EMBED = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"


def paras(tc) -> list:
    return tc.findall(qn("w:p"))


def _next_docpr_id(doc) -> int:
    ids = [
        int(el.get("id"))
        for el in doc.element.body.iter(f"{{{WP_NS}}}docPr")
        if el.get("id", "").isdigit()
    ]
    return max(ids, default=0) + 1


def copy_cell(src_doc, src_tc, dst_doc, dst_tc, skip_placeholder: bool = False) -> int:
    """把 src_tc 的全部段落搬进 dst_tc（先清空）。返回写入的段落数。"""
    for p in paras(dst_tc):
        dst_tc.remove(p)

    docpr = _next_docpr_id(dst_doc)
    written = 0
    for p in paras(src_tc):
        text = "".join(t.text or "" for t in p.iter(qn("w:t")))
        if skip_placeholder and text.strip() in {"【传真】", "【协作单位一】", ""}:
            continue
        new_p = deepcopy(p)
        # 图片：rId 在目标文档里不存在，换成目标文档自己的（同一份 blob 去重）
        for blip in new_p.iter(f"{{{A_NS}}}blip"):
            old_rid = blip.get(R_EMBED)
            if not old_rid:
                continue
            blob = src_doc.part.related_parts[old_rid].blob
            new_rid, _ = dst_doc.part.get_or_add_image(BytesIO(blob))
            blip.set(R_EMBED, new_rid)
        for doc_pr in new_p.iter(f"{{{WP_NS}}}docPr"):
            doc_pr.set("id", str(docpr))
            docpr += 1
        dst_tc.append(new_p)
        written += 1

    if not paras(dst_tc):  # 单元格至少要有一个段落；被跳过的占位符留一个空段
        blank = deepcopy(paras(src_tc)[0])
        for r in blank.findall(qn("w:r")):
            blank.remove(r)
        dst_tc.append(blank)
    return written


def set_text(dst_tc, text: str, rpr_template, align="center") -> None:
    """在 dst_tc 里写一段居中文字，字体照 rpr_template。"""
    for p in paras(dst_tc):
        dst_tc.remove(p)
    new_p = dst_tc.makeelement(qn("w:p"), {})
    ppr = new_p.makeelement(qn("w:pPr"), {})
    jc = ppr.makeelement(qn("w:jc"), {qn("w:val"): align})
    ppr.append(jc)
    new_p.append(ppr)
    run = new_p.makeelement(qn("w:r"), {})
    if rpr_template is not None:
        run.append(deepcopy(rpr_template))
    t = run.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.append(t)
    new_p.append(run)
    dst_tc.append(new_p)


def prepend_para(tc, template_p, text: str) -> None:
    """在单元格最前面插一段，格式照 template_p 的正文（首行缩进、行距一并抄）。"""
    new_p = deepcopy(template_p)
    for r in new_p.findall(qn("w:r")):
        new_p.remove(r)
    src_run = template_p.find(qn("w:r"))
    run = new_p.makeelement(qn("w:r"), {})
    if src_run is not None:
        rpr = src_run.find(qn("w:rPr"))
        if rpr is not None:
            run.append(deepcopy(rpr))
    t = run.makeelement(qn("w:t"), {})
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.append(t)
    new_p.append(run)
    tc.insert(list(tc).index(paras(tc)[0]), new_p)


def allow_char_break(tc, min_token: int = 22) -> int:
    """给含超长西文串（URL/DOI）的段落开「允许西文在单词中间换行」。

    不开的话 Word 按内容自动调整列宽：参考文献 [25] 那条 95 字符的宁波公积金链接
    会把整张表从 16.2cm 撑到 23cm，右半边直接跑到纸外。
    """
    fixed = 0
    for p in tc.iter(qn("w:p")):
        text = "".join(t.text or "" for t in p.iter(qn("w:t")))
        tokens = re.split(r"[\s，。；：、（）“”《》\[\]]", text)
        if not any(len(t) >= min_token and not re.search(r"[\u4e00-\u9fff]", t) for t in tokens):
            continue
        ppr = p.find(qn("w:pPr"))
        if ppr is None:
            ppr = p.makeelement(qn("w:pPr"), {})
            p.insert(0, ppr)
        if ppr.find(qn("w:wordWrap")) is not None:
            continue
        ww = ppr.makeelement(qn("w:wordWrap"), {qn("w:val"): "0"})
        # pPr 子元素有固定顺序：wordWrap 排在 spillover 之后、snapToGrid/spacing/ind/jc/rPr 之前
        later = {
            qn(t)
            for t in (
                "w:snapToGrid",
                "w:spacing",
                "w:ind",
                "w:contextualSpacing",
                "w:jc",
                "w:textDirection",
                "w:outlineLvl",
                "w:rPr",
                "w:sectPr",
            )
        }
        anchor = next((c for c in ppr if c.tag in later), None)
        if anchor is not None:
            anchor.addprevious(ww)
        else:
            ppr.append(ww)
        fixed += 1
    return fixed


def fix_image_para(tc) -> int:
    """图片段落：居中、去掉首行缩进。

    省表里图片段落同时设了「居中」和「首行缩进 0.74cm」，两者叠加会把 15.5cm 宽的
    图推到 16.24cm，超出单元格可用宽度（15.8cm），Word 于是自动加宽整张表。
    """
    n = 0
    for p in tc.iter(qn("w:p")):
        if p.find(qn("w:r") + "/" + qn("w:drawing")) is None and "w:drawing" not in p.xml:
            continue
        ppr = p.find(qn("w:pPr"))
        if ppr is None:
            ppr = p.makeelement(qn("w:pPr"), {})
            p.insert(0, ppr)
        ind = ppr.find(qn("w:ind"))
        if ind is not None:
            ppr.remove(ind)
        n += 1
    return n


def set_table_fixed(dst_doc) -> None:
    """把表格设成固定列宽。

    模板本身没写 tblLayout，Word 默认按内容自动调整列宽：表格里出现一条超长 URL
    或一张贴边图，整张表就会被撑宽、右半边跑到纸外。固定列宽＝沿用模板的网格，
    内容在格子里换行，这才叫"格式不变"。
    """
    tbl_pr = dst_doc.tables[0]._tbl.tblPr
    if tbl_pr.find(qn("w:tblLayout")) is not None:
        return
    layout = tbl_pr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
    anchor = None
    for tag in ("w:tblCellMar", "w:tblLook", "w:tblCaption", "w:tblDescription"):
        anchor = tbl_pr.find(qn(tag))
        if anchor is not None:
            break
    if anchor is not None:
        anchor.addprevious(layout)
    else:
        tbl_pr.append(layout)


def relax_row_heights(dst_doc, rows_sel) -> None:
    """放宽被填满的行的高度规则。

    模板把正文行设成「最小高度 9~13cm」、组员行设成「固定 0.7cm」——那是给空白表
    用的。填满之后：正文行的最后一段会被撑到最小高度，章节末尾留出半页空白；
    组员行则会把两行文字直接裁掉。这里只改 hRule（auto / atLeast），不改数值。
    """
    drows = dst_doc.tables[0]._tbl.tr_lst
    for row in rows_sel:
        h = drows[row].trPr.find(qn("w:trHeight"))
        if h is not None:
            h.set(qn("w:hRule"), "auto")
    for row in range(32, 41):
        h = drows[row].trPr.find(qn("w:trHeight"))
        if h is not None:
            h.set(qn("w:hRule"), "atLeast")


def main() -> None:
    src = Document(str(SRC))
    dst = Document(str(TPL))
    srows = src.tables[0]._tbl.tr_lst
    drows = dst.tables[0]._tbl.tr_lst

    # ── 一、二、三、五：整段搬运 ─────────────────────────────
    rows_sel = (1, 3, 5, 9)  # 一、二、三、五
    for row in rows_sel:
        n = copy_cell(src, srows[row].tc_lst[0], dst, drows[row].tc_lst[0])
        print(f"第{row}行正文：{n} 段")

    # 市级表头写的是「技术、经济指标和社会效益」，技术指标在第二部分（三），
    # 开头加一句指路，免得评审在本节找不到技术指标。
    if 9 in rows_sel:
        prepend_para(
            drows[9].tc_lst[0],
            paras(drows[9].tc_lst[0])[1],
            "本项目的技术指标（脏文档解析成功率、Recall@8、引用来源准确率、答案证据支持率、"
            "误拒率与多轮交互能力等）已按统一口径逐项列于第二部分（三）量化考核指标，"
            "本节说明经济效益与社会效益。",
        )

    # ── 六、承担单位 ────────────────────────────────────────
    for d_row, s_row in ((11, 11), (12, 12)):  # 单位名称 / 所在地
        copy_cell(src, srows[s_row].tc_lst[2], dst, drows[d_row].tc_lst[2])
    # 省表把公司名写成「浙江绿城建筑设计有有限公司」，按封面订正
    for t in drows[11].tc_lst[2].iter(qn("w:t")):
        if t.text and "有有限公司" in t.text:
            t.text = t.text.replace("有有限公司", "有限公司")
    copy_cell(src, srows[13].tc_lst[2], dst, drows[13].tc_lst[2])  # 详细地址
    copy_cell(src, srows[13].tc_lst[4], dst, drows[13].tc_lst[4])  # 邮编
    copy_cell(src, srows[14].tc_lst[2], dst, drows[14].tc_lst[2])  # 联系人
    copy_cell(src, srows[14].tc_lst[4], dst, drows[14].tc_lst[4])  # 联系电话
    # 传真：省表里只是「【传真】」占位符，留空
    copy_cell(src, srows[15].tc_lst[2], dst, drows[15].tc_lst[2], skip_placeholder=True)
    # 电子邮箱：省表空着，这里按封面的邮箱补上，字体照联系电话
    rpr = paras(srows[14].tc_lst[4])[0].find(qn("w:r")).find(qn("w:rPr"))
    set_text(drows[15].tc_lst[4], "qianluxi@gad.com.cn", rpr)

    # ── 七、负责人 ──────────────────────────────────────────
    for d_row, s_row, d_ci, s_ci in (
        (23, 22, 2, 2),  # 姓名
        (24, 23, 2, 2),  # 身份证
        (25, 24, 2, 2),  # 电话
        (26, 25, 2, 2),  # E-mail
        (27, 26, 2, 2),  # 学历
        (27, 26, 4, 4),  # 学位
        (28, 27, 2, 2),  # 职称
        (28, 27, 4, 4),  # 专业
        (29, 28, 2, 2),  # 分工
        (30, 29, 2, 2),  # 工作单位
    ):
        copy_cell(src, srows[s_row].tc_lst[s_ci], dst, drows[d_row].tc_lst[d_ci])

    # ── 七、项目组成员（省表 5 人，市表 9 行） ───────────────
    for i in range(5):
        for ci in range(1, 7):
            copy_cell(src, srows[31 + i].tc_lst[ci], dst, drows[32 + i].tc_lst[ci])
        # 「工作单位」列只有 4.4cm，14pt 的公司全称要占 5.9cm；降到 9pt 才排得下一行
        # 「专业」列只有 1.67cm，「建筑结构」14pt 会折成「建筑结／构」，同样降到 9pt
        for ci, size in ((4, "18"), (5, "18")):
            for run in drows[32 + i].tc_lst[ci].iter(qn("w:r")):
                rpr = run.find(qn("w:rPr"))
                if rpr is None:
                    rpr = run.makeelement(qn("w:rPr"), {})
                    run.insert(0, rpr)
                for tag in ("w:sz", "w:szCs"):
                    el = rpr.find(qn(tag))
                    if el is None:
                        el = rpr.makeelement(qn(tag), {})
                        rpr.append(el)
                    el.set(qn("w:val"), size)

    # ── 八、计划进度目标（省表 4 阶段，市表 6 行） ────────────
    for i in range(4):
        copy_cell(src, srows[43 + i].tc_lst[0], dst, drows[43 + i].tc_lst[0])
        copy_cell(src, srows[43 + i].tc_lst[1], dst, drows[43 + i].tc_lst[1])

    # ── 九、经费及来源 ─────────────────────────────────────
    copy_cell(src, srows[48].tc_lst[0], dst, drows[50].tc_lst[0])

    n = sum(allow_char_break(drows[i].tc_lst[0]) for i in rows_sel)
    print(f"开「允许西文断行」的段落：{n}")
    m = sum(fix_image_para(drows[i].tc_lst[0]) for i in rows_sel)
    print(f"去掉首行缩进的图片段落：{m}")
    set_table_fixed(dst)
    relax_row_heights(dst, rows_sel)
    dst.save(str(OUT))
    print("saved", OUT)


if __name__ == "__main__":
    main()
