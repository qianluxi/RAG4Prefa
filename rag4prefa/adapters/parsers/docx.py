"""docx 解析：把 Word 文档还原成「标题层级 + 正文」的线性序列。

与原实现的关键差异：
- 不再依赖 Windows 的 Word COM，纯 python-docx，跨平台、可进容器、可单测；
- 层级判定走「大纲级别(XML) → 标题样式 → 编号正则」三级优先，而不是只用正则猜；
- **会走进表格**：申报表/验收申请书这类公文，正文全在表格单元格里，原实现直接丢弃。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from ...domain import Section, SectionKind

_CN_NUM = "一二三四五六七八九十百零〇两"
_CHAPTER = re.compile(rf"^第[{_CN_NUM}\d]+[章篇]")
_SECTION = re.compile(rf"^第[{_CN_NUM}\d]+[节条]")
_CN_DUN = re.compile(rf"^[{_CN_NUM}]+\s*、")  # 一、二、—— 公文顶层标题
_NUMBERED = re.compile(r"^(\d+(?:[.．]\d+)*)[.．]?\s*\S")  # 1 / 1.1 / 1.1.1（可与标题紧贴）
_PAREN_CN = re.compile(rf"^[（(][{_CN_NUM}]+[)）]")
_PAREN_NUM = re.compile(r"^[（(]?\d+[)）]")
_CIRCLED = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]")
_STYLE_HEADING = re.compile(r"(?:heading|标题)\s*(\d)", re.IGNORECASE)
_WS = re.compile(r"[\s\u3000]+")

_MAX_TITLE_LEN = 60  # 超过这个长度基本是正文句子
_ROW_CELL_MAX = 80  # 单元格都短于这个长度，按"数据行"保留成表格行


def clean_text(text: str) -> str:
    return _WS.sub(" ", text.replace("\x0b", " ").replace("\x07", " ")).strip()


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

    返回 0 表示"相对级别"——作为当前所在标题的子级。
    中文括号序号（（一）/（1）/①）的绝对层级在不同文档里差别很大
    （有的直接挂在一级标题下，有的挂在 1.3.1 之下），用相对级别才不会错位。
    """
    if level := _outline_level(paragraph._p):
        return level

    style = paragraph.style
    if style is not None:
        if level := _outline_level(style.element):
            return level
        if m := _STYLE_HEADING.search(style.name or ""):
            return int(m.group(1))

    # 文本编号兜底（无样式文档的唯一线索）
    text = clean_text(paragraph.text)
    if not text:
        return None
    if len(text) > _MAX_TITLE_LEN or text.endswith(("。", "；", "，", ".")):
        return None
    if "。" in text[:30]:
        return None

    if _CHAPTER.match(text) or _CN_DUN.match(text):
        return 1
    if _SECTION.match(text):
        return 2
    if m := _NUMBERED.match(text):
        depth = m.group(1).count(".") + m.group(1).count("．")
        return min(4, depth + 1)
    if _PAREN_CN.match(text) or _PAREN_NUM.match(text) or _CIRCLED.match(text):
        return 0  # 相对级别
    return None


def _kind_for(level: int) -> SectionKind:
    if level <= 1:
        return SectionKind.chapter
    if level == 2:
        return SectionKind.section
    return SectionKind.subsection


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
    """解析为 Section 列表；正文归到最近标题之下，数据行按 "|" 保留。"""
    document = Document(str(path))
    sections: list[Section] = []
    preamble: list[str] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    base_level = 0  # 最近一个"绝对级别"标题的层级，用于给（一）/（1）这类相对标题定位
    current: Section | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            current.text = "\n".join(buffer).strip()
            sections.append(current)
        current, buffer = None, []

    for kind, payload in _walk(document.element.body, document):
        if kind == "row":
            line = " | ".join(payload)  # type: ignore[arg-type]
            (buffer if current is not None else preamble).append(line)
            continue

        paragraph: Paragraph = payload  # type: ignore[assignment]
        text = clean_text(paragraph.text)
        if not text:
            continue
        level = heading_level(paragraph)
        if level is None:
            (buffer if current is not None else preamble).append(text)
            continue
        if level == 0:  # 相对级别：与「最近一个绝对级别标题的下一层」同级并列
            level = min(9, base_level + 1)
        else:
            base_level = level

        flush()
        while stack and stack[-1][0] >= level:
            stack.pop()
        current = Section(
            title=text, kind=_kind_for(level), level=level, path=[t for _, t in stack]
        )
        stack.append((level, text))

    flush()

    if preamble:  # 封面/摘要等无标题内容单独成节，而不是被静默丢弃
        sections.insert(
            0,
            Section(
                title="前言",
                kind=SectionKind.body,
                level=0,
                path=[],
                text="\n".join(preamble),
            ),
        )
    return sections
