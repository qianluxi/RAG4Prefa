"""解析器共用的标题判定规则。

docx 与 pdf 的差别只在"怎么拿到一行字"：docx 有大纲级别和样式，pdf 只有字号与坐标。
但「这段文字像不像标题、它是第几级」必须用同一套规则——否则同一份文档转成 pdf 之后
章节路径会换成一棵不同的树，`chunk_id`（内容寻址，含章节路径）跟着全变，
引用和评估金标都对不上。

所以这里只放**与格式无关**的判据：正文清洗、中文编号正则、纯文本判级。
拿到标题的线索（Word 的 outlineLvl、PDF 的字号）由各自的解析器提供。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ...domain import Section, SectionKind

_CN_NUM = "一二三四五六七八九十百零〇两"
_CHAPTER = re.compile(rf"^第[{_CN_NUM}\d]+[章篇]")
_SECTION = re.compile(rf"^第[{_CN_NUM}\d]+[节条]")
_CN_DUN = re.compile(rf"^[{_CN_NUM}]+\s*、")  # 一、二、—— 公文顶层标题
_NUMBERED = re.compile(r"^(\d+(?:[.．]\d+)*)[.．]?\s*\S")  # 1 / 1.1 / 1.1.1（可与标题紧贴）
_PAREN_CN = re.compile(rf"^[（(][{_CN_NUM}]+[)）]")
_PAREN_NUM = re.compile(r"^[（(]?\d+[)）]")
_CIRCLED = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]")
_WS = re.compile(r"[\s\u3000]+")

MAX_TITLE_LEN = 60  # 超过这个长度基本是正文句子

# 以这些标点收尾的是句子，不是标题。PDF 尤其需要：正文里的行尾常常正好是逗号。
# 冒号**不能**列进来——"附件："、"说明：" 是 PDF 里很常见的标题。
_SENTENCE_END = ("。", "；", "，", ".", ",", ";", "？", "?", "！", "!")


def clean_text(text: str) -> str:
    return _WS.sub(" ", text.replace("\x0b", " ").replace("\x07", " ")).strip()


def looks_like_title(text: str) -> bool:
    """长度与标点层面的标题粗筛，不看字号也不看样式。

    只回答"这段文字有没有可能是标题"，回答 False 就一定不是；
    回答 True 仍需要调用方提供证据（样式、字号或编号）。
    """
    if not text or len(text) > MAX_TITLE_LEN:
        return False
    if text.endswith(_SENTENCE_END):
        return False
    return "。" not in text[:30]


def text_heading_level(text: str) -> int | None:
    """只看文字，判断编号隐含的级别；不是标题返回 None。

    返回 0 表示"相对级别"——作为当前所在标题的子级。中文括号序号
    （（一）/（1）/①）的绝对层级在不同文档里差别很大（有的直接挂在一级标题下，
    有的挂在 1.3.1 之下），用相对级别才不会错位。
    """
    if not looks_like_title(text):
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


def kind_for(level: int) -> SectionKind:
    if level <= 1:
        return SectionKind.chapter
    if level == 2:
        return SectionKind.section
    return SectionKind.subsection


def build_sections(
    events: Iterable[tuple[int | None, str]], *, preamble_title: str = "前言"
) -> list[Section]:
    """把「(级别, 文本)」流折成章节树。

    级别为 None 表示正文行，归到最近一个标题名下；为 0 表示相对级别（见
    `text_heading_level`），由这里挂到最近一个绝对级别标题的下一层——这件事必须由
    组装方做，因为只有它知道"最近一个绝对级别"是多少。

    两个解析器共用这一份：章节树是 `chunking._chunk_id` 的输入之一，
    如果两份实现各写一遍栈，pdf 与 docx 的层级偏差会以"同一文档两个 md5"的形式出现。

    无标题的前导内容（封面、签发日期、摘要）单独成节而不是丢弃——那正是 pdf 里
    最常见的一段正文。
    """
    sections: list[Section] = []
    preamble: list[str] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    base_level = 0  # 最近一个"绝对级别"标题的层级，用于给相对级别标题定位
    current: Section | None = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal current, buffer
        if current is not None:
            current.text = "\n".join(buffer).strip()
            sections.append(current)
        current, buffer = None, []

    for level, text in events:
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
            title=text, kind=kind_for(level), level=level, path=[t for _, t in stack]
        )
        stack.append((level, text))

    flush()

    if preamble:  # 封面/摘要等无标题内容单独成节，而不是被静默丢弃
        sections.insert(
            0,
            Section(
                title=preamble_title,
                kind=SectionKind.body,
                level=0,
                path=[],
                text="\n".join(preamble),
            ),
        )
    return sections
