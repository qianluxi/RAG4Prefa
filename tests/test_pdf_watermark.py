"""水印型扫描件：有文字层，但那层字不是正文。

住建部「信息公开浏览专用」的 pdf 全是这种形态：整页一张扫描图，上面叠一行
「住房城乡建设部信息公开　浏览专用」。实测 45 页、每页 37 个字符、汉字占比 100%，
字符数与可读性两道闸门**一道都拦不住**——它会被拼成 719 字的一节塞进库里，
检索时这串水印还会被当成正文命中。

判据只能看版面：水印页的图片**合计**覆盖接近整页（实测 1.00），正常文档（含图集）
是 0.00。所以这里既钉"能认出来"，也钉"不要把正常文档认错"——后者代价更大：
一份 120 页的设计示例被误判成扫描件，就要白花 120 页的识别费。

另一个实测细节：水印页由**几张图拼成**（一页 8 张、单张只占 15%），
判"单张最大覆盖"会全部漏掉。用例 4 专门守这一条。
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from rag4prefa.adapters.ocr import OCR_SETUP_HINT
from rag4prefa.adapters.parsers import parse_document_report
from rag4prefa.adapters.parsers.pdf import _image_cover, _page_is_scanned

MARK = "住房城乡建设部信息公开　浏览专用"


def _blank_pixmap() -> pymupdf.Pixmap:
    return pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4), 0)


def _watermark_pdf(tmp_path: Path, pages: int = 3, tiles: int = 4) -> Path:
    """整页图（tiles 张拼满）+ 一行水印文字。"""
    path = tmp_path / "水印.pdf"
    doc = pymupdf.open()
    pix = _blank_pixmap()
    for index in range(pages):
        page = doc.new_page()
        rect = page.rect
        if tiles == 1:
            boxes = [rect]
        else:
            # 2×2 拼满：单张只覆盖 25%，合计 100%
            hw, hh = rect.width / 2, rect.height / 2
            boxes = [pymupdf.Rect(x, y, x + hw, y + hh) for x in (0, hw) for y in (0, hh)]
        for box in boxes:
            page.insert_image(box, pixmap=pix)
        page.insert_text((72, 120 + index), MARK, fontname="china-s", fontsize=10)
    doc.save(str(path))
    doc.close()
    return path


def _text_pdf(tmp_path: Path, with_small_image: bool = False) -> Path:
    """正常文字页；可选插一张小图（logo），覆盖远低于阈值。"""
    path = tmp_path / "正文.pdf"
    doc = pymupdf.open()
    for index in range(3):
        page = doc.new_page()
        if with_small_image:
            page.insert_image(pymupdf.Rect(60, 40, 120, 70), pixmap=_blank_pixmap())
        page.insert_text((72, 120), f"第{index + 1}章 总则", fontname="china-s", fontsize=18)
        page.insert_text(
            (72, 160),
            "装配式内装修应采用干式工法，部品部件在工厂生产、现场组合安装。",
            fontname="china-s",
            fontsize=11,
        )
    doc.save(str(path))
    doc.close()
    return path


def test_watermark_only_pdf_gets_its_own_refusal(tmp_path: Path) -> None:
    """两种坏法要说成两句不同的话。

    说成"没有文字层"会让用户以为自己文件是空的——他明明在阅读器里能看见字。
    用户能做的事也不同：一个是"去开 OCR 或换版本"，一个是"这版本来就是浏览专用"。
    """
    with pytest.raises(ValueError) as exc:
        parse_document_report(_watermark_pdf(tmp_path))
    message = str(exc.value)
    assert message.startswith(OCR_SETUP_HINT), "前端靠这个前缀弹配置引导"
    assert "水印" in message
    assert "没有文字层" not in message


def test_tiled_images_are_counted_together(tmp_path: Path) -> None:
    """水印页是 4 张图拼起来的：单张只占 25%，判单张最大值会全部漏掉。"""
    path = _watermark_pdf(tmp_path, pages=1, tiles=4)
    with pymupdf.open(path) as doc:
        page = doc[0]
        assert _image_cover(page) > 0.99
        assert _page_is_scanned(page, MARK) is True


def test_normal_text_pages_are_not_mistaken_for_scan(tmp_path: Path) -> None:
    """正常文档（哪怕带一张 logo 小图）不能被判成扫描件——误判要白花识别费。"""
    path = _text_pdf(tmp_path, with_small_image=True)
    with pymupdf.open(path) as doc:
        assert _image_cover(doc[0]) < 0.05
        assert _page_is_scanned(doc[0], doc[0].get_text()) is False
    sections, report = parse_document_report(path)
    assert sections, "正常文档必须能解析出章节"
    assert report.missing_pages == ()


def test_text_rich_page_over_a_full_page_image_is_body_not_scan(tmp_path: Path) -> None:
    """整页图 + 大量正文（扫描背景上盖了文字层）按正文处理。

    这条是短路的守护：`_page_is_scanned` 先看字数，够多就不量几何。
    少了这个短路，一份"每页都有背景图"的排版文档会被整体送去识别，白花钱。
    """
    path = tmp_path / "背景图.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, pixmap=_blank_pixmap())
    body = "装配式内装修工程应按本导则进行设计与施工，部品部件应满足模数协调要求。"
    # 用 textbox 自动换行：单行 insert_text 超出页面宽度的部分会被直接裁掉
    page.insert_textbox(
        pymupdf.Rect(60, 100, page.rect.width - 60, page.rect.height - 100),
        body * 6,
        fontname="china-s",
        fontsize=11,
    )
    doc.save(str(path))
    doc.close()

    with pymupdf.open(path) as doc:
        page = doc[0]
        assert _image_cover(page) > 0.9, "确实是整页图"
        assert len(page.get_text().strip()) > 120
        assert _page_is_scanned(page, page.get_text()) is False
