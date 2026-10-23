"""把自然语言问题里的检索条件抽出来（规则优先，不依赖大模型）。

为什么不用 LLM 做这一步：
- 规则在这里足够准确（年份、项目名、引号内关键词都是确定性模式）；
- 规则可单测、零延迟、零成本；
- LLM 一旦抽错，用户看到的"范围缩小"是错误的静默行为，比"不缩小"更糟。
LLM 只在规则完全没抽到时兜底（可选，见 `llm_fallback`）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .domain import Filter

_RANGE = re.compile(r"(20\d{2})\s*(?:[-~－—至到]|年\s*[-~至到])\s*(20\d{2})\s*年?")
_AFTER = re.compile(r"(20\d{2})\s*年?\s*(?:以后|之后|起)")
_BEFORE = re.compile(r"(20\d{2})\s*年?\s*(?:以前|之前|前)")
_RECENT = re.compile(r"近\s*([一二三四五六七八九十\d]+)\s*年")
_SINGLE = re.compile(r"(20\d{2})\s*年")
_QUOTED = re.compile(r"[“\"'「『]([^”\"'」』]{1,40})[”\"'」』]")

_CN_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _cn_to_int(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if text in _CN_DIGITS:
        return _CN_DIGITS[text]
    if text.startswith("十") and len(text) == 2:
        return 10 + _CN_DIGITS.get(text[1], 0)
    return None


def parse_filter(
    question: str, project_names: Sequence[str] = (), *, current_year: int = 2026
) -> Filter:
    projects: list[str] = []
    for name in project_names:
        if name and name in question:
            projects.append(name)
    # 项目编号也当作可精确命中的别名（如 kc2015）
    for name in project_names:
        code_match = re.search(r"\b(kc\d{3,6})\b", question, re.IGNORECASE)
        if code_match and name.lower().startswith(code_match.group(1).lower()):
            projects.append(name)

    year_from: int | None = None
    year_to: int | None = None
    if m := _RANGE.search(question):
        year_from, year_to = sorted((int(m.group(1)), int(m.group(2))))
    elif m := _AFTER.search(question):
        year_from = int(m.group(1))
    elif m := _BEFORE.search(question):
        year_to = int(m.group(1))
    elif m := _RECENT.search(question):
        n = _cn_to_int(m.group(1)) or 3
        year_from, year_to = current_year - n, current_year
    elif m := _SINGLE.search(question):
        year_from = year_to = int(m.group(1))

    keyword = None
    if m := _QUOTED.search(question):
        keyword = m.group(1).strip()

    return Filter(
        projects=projects or None,
        year_from=year_from,
        year_to=year_to,
        keyword=keyword,
    )


def describe(filt: Filter, parsed: Filter | None = None) -> str:
    """给用户/提示词看的一句话范围说明。"""
    parts: list[str] = []
    if filt.projects:
        parts.append("项目：" + "、".join(filt.projects))
    if filt.year_from and filt.year_to and filt.year_from == filt.year_to:
        parts.append(f"年份：{filt.year_from}")
    elif filt.year_from and filt.year_to:
        parts.append(f"年份：{filt.year_from}-{filt.year_to}")
    elif filt.year_from:
        parts.append(f"年份：{filt.year_from} 年起")
    elif filt.year_to:
        parts.append(f"年份：截至 {filt.year_to}")
    if filt.keyword:
        parts.append(f"关键词：{filt.keyword}")
    return "；".join(parts)
