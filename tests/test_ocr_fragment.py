"""送识别前的页图渲染 + 识别结果"停早了"的检测。

两个判据都是被同一份文档逼出来的：GB 55037 的页面是 1653×2397 pt（普通 A4 的
2.8 倍），150 dpi 渲出来 3444×4994，模型对这张图只还原第一行就停，不报错、不超时。
66 页里 44 页是这么废掉的：识别结果照样进库、照样有向量、照样出现在引用里。
"""

from __future__ import annotations

import sqlite3

import pymupdf
import pytest

from rag4prefa import ingest
from rag4prefa.adapters.ocr import page_ink
from rag4prefa.adapters.parsers.pdf import _OCR_MAX_EDGE, _render_for_ocr

_CN = "china-s"


def _inked_png(paragraphs: int = 20, dpi: int = 60) -> bytes:
    """一页真有字的小图（够触发"有墨迹"判据，但不必是 A4）。"""
    doc = pymupdf.open()
    page = doc.new_page()
    for row in range(paragraphs):
        page.insert_text(
            (40, 50 + row * 16),
            "消防车道应满足消防车安全通行的要求",
            fontname=_CN,
            fontsize=10,
        )
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    doc.close()
    return png


# ── 渲染：大页面必须被压到长边上限 ─────────────────────────


def test_oversized_page_is_scaled_down_before_it_goes_out() -> None:
    """GB 55037 那种 1653×2397 pt 的页面：150 dpi 会是 3444×4994。"""
    doc = pymupdf.open()
    page = doc.new_page(width=1653, height=2397)
    page.insert_text((100, 200), "10.1.10 建筑内疏散照明", fontname=_CN, fontsize=60)
    png = _render_for_ocr(page, 150)
    raw = doc[0].get_pixmap(dpi=150).tobytes("png")
    doc.close()

    assert len(png) < len(raw)
    pix = pymupdf.Pixmap(png)
    assert max(pix.width, pix.height) <= _OCR_MAX_EDGE


def test_normal_a4_page_is_not_resized() -> None:
    """普通 A4 在 150 dpi 下是 1240×1754，够不着上限——已有语料的页图不能变。"""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((60, 100), "1 总则", fontname=_CN, fontsize=12)
    png = _render_for_ocr(page, 150)
    plain = doc[0].get_pixmap(dpi=150).tobytes("png")
    doc.close()
    assert png == plain


# ── 墨迹：区分"本来没字"和"有字没认出来" ───────────────────


def test_page_ink_separates_a_blank_page_from_a_written_one() -> None:
    doc = pymupdf.open()
    blank = doc.new_page()
    blank_png = blank.get_pixmap(dpi=60).tobytes("png")
    doc.close()

    assert page_ink(blank_png) < 0.01
    assert page_ink(_inked_png()) > ingest._OCR_INK_MIN
    # 解不开的字节当作没墨迹（我们自己的 PNG 一定能解开）
    assert page_ink(b"not-a-png") == 0.0


def test_short_result_on_an_inked_page_is_asked_again(conn: sqlite3.Connection) -> None:
    """碎片判据：短 + 有墨迹 → 再问一次，取长的那个。"""
    answers = iter(["1", "1 消防车道应满足消防车安全通行的要求；" * 6])
    calls: list[int] = []

    class Fragment:
        model = "m#v1"

        def __call__(self, png: bytes, page: int) -> str:
            calls.append(page)
            return next(answers)

    ocr, close = ingest._cached_ocr(conn, "m#v1", Fragment)
    text = ocr(_inked_png(), 15)
    close()

    assert calls == [15, 15]
    assert len(text) > 100


def test_short_result_on_a_blank_page_is_not_asked_again(conn: sqlite3.Connection) -> None:
    """空白页回空串是**正确**结果，不该再花钱问一遍。"""
    doc = pymupdf.open()
    blank_png = doc.new_page().get_pixmap(dpi=60).tobytes("png")
    doc.close()

    calls: list[int] = []

    class Blank:
        model = "m#v1"

        def __call__(self, png: bytes, page: int) -> str:
            calls.append(page)
            return ""

    ocr, close = ingest._cached_ocr(conn, "m#v1", Blank)
    assert ocr(blank_png, 3) == ""
    close()
    assert calls == [3]


def test_a_shorter_retry_does_not_replace_the_first_answer(
    conn: sqlite3.Connection,
) -> None:
    """重试只取"更长的"：第二次什么都没回，第一次那半句仍然比空强。

    这条必须让**第一次真的触发重试**，否则测不到"覆盖"这一步——第一版用了一个
    够长的第一次回答，重试根本没发生，把覆盖改成无脑赋值也全绿（变异存活）。
    """
    answers = iter(["1", ""])
    calls: list[int] = []

    class Shrinking:
        model = "m#v1"

        def __call__(self, png: bytes, page: int) -> str:
            calls.append(page)
            return next(answers)

    ocr, close = ingest._cached_ocr(conn, "m#v1", Shrinking)
    text = ocr(_inked_png(), 9)
    close()
    assert text == "1"
    assert calls == [9, 9]


@pytest.mark.parametrize("text", ["", "1", "图 4-1 悬挑节点大样"])
def test_short_results_are_kept_when_the_page_has_no_ink(text: str) -> None:
    """没有墨迹的页不重试（判据本身就不该触发）。"""
    doc = pymupdf.open()
    blank_png = doc.new_page().get_pixmap(dpi=60).tobytes("png")
    doc.close()
    assert ingest._looks_cut_short(blank_png, text) is False
