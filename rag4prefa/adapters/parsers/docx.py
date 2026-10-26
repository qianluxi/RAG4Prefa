"""docx 解析：把 Word 文档还原成「标题层级 + 正文」的线性序列。

与原实现的关键差异：
- 不再依赖 Windows 的 Word COM，纯 python-docx，跨平台、可进容器、可单测；
- 层级判定走「大纲级别(XML) → 标题样式 → 编号正则」三级优先，而不是只用正则猜；
- **会走进表格**：申报表/验收申请书这类公文，正文全在表格单元格里，原实现直接丢弃。

编号正则与正文清洗放在 `headings.py`，与 pdf 解析器共用——同一份文档换个格式，
章节树必须是同一棵，否则内容寻址的 `chunk_id` 全部对不上。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from ...domain import Section
from .headings import build_sections, clean_text, text_heading_level

_STYLE_HEADING = re.compile(r"(?:heading|标题)\s*(\d)", re.IGNORECASE)

_ROW_CELL_MAX = 80  # 单元格都短于这个长度，按"数据行"保留成表格行


def _outline_level(element) -> int | None:
    """从 OOXML 读大纲级别（1 起）。不同 python-docx 版本对它的封装不一致，直接读 XML 最稳。"""
    p_pr = element.find(qn("w:pPr"))
    if p_pr is None:
        return None
    outline = p_pr.find(qn("w:outlineLvl"))
    if outline is None:
        return None
    value = outline.get(qn("w:val"))
    try:
        level = int(value) + 1
    except (TypeError, ValueError):
        return None
    return level if 1 <= level <= 9 else None


def heading_level(paragraph: Paragraph) -> int | None:
    """返回标题级别；不是标题返回 None。

    返回 0 表示"相对级别"——作为当前所在标题的子级。具体语义见
    `headings.text_heading_level`。Word 的大纲级别与标题样式比文字编号可靠，
    所以先问它们，都没有线索时才退回编号正则。
    """
    if level := _outline_level(paragraph._p):
        return level

    style = paragraph.style
    if style is not None:
        if level := _outline_level(style.element):
            return level
        if m := _STYLE_HEADING.search(style.name or ""):
            return int(m.group(1))

    return text_heading_level(clean_text(paragraph.text))


def _walk(element, document) -> Iterator[tuple[str, object]]:
    """按文档顺序产出 ('p', Paragraph) 或 ('row', [cell_text...])。

    合并单元格会让同一段文字在多个 cell 里重复出现，这里按行去重，
    否则一份 13 列的申报表会被复制 13 遍。
    """
    for child in element.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            yield "p", Paragraph(child, document)
        elif tag == "tbl":
            for row in Table(child, document).rows:
                logical: list[tuple[str, _Cell]] = []
                seen: set[str] = set()
                for cell in row.cells:
                    text = clean_text(cell.text)
                    if text and text in seen:
                        continue
                    seen.add(text)
                    logical.append((text, cell))
                filled = [(t, c) for t, c in logical if t]
                if len(filled) >= 2 and all(len(t) <= _ROW_CELL_MAX for t, _ in filled):
                    yield "row", [t for t, _ in filled]
                    continue
                for _, cell in filled:
                    yield from _walk(cell._element, document)
                    for nested in cell.tables:
                        yield from _walk(nested._element, document)


def parse_docx(path: str | Path) -> list[Section]:
    """解析为 Section 列表；正文归到最近标题之下，数据行按 "|" 保留。

    组装（栈、相对级别、前言）交给 `headings.build_sections`，与 pdf 共用同一份。
    """
    document = Document(str(path))
    events: list[tuple[int | None, str]] = []
    for kind, payload in _walk(document.element.body, document):
        if kind == "row":
            events.append((None, " | ".join(payload)))  # type: ignore[arg-type]
            continue

        paragraph: Paragraph = payload  # type: ignore[assignment]
        text = clean_text(paragraph.text)
        if not text:
            continue
        events.append((heading_level(paragraph), text))
    return build_sections(events)
