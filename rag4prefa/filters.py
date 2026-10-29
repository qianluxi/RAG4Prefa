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
_AFTER = re.compile(r"(20\d{2})\s*年?\s*(?:以后|之后|以来|起)")
_BEFORE = re.compile(r"(20\d{2})\s*年?\s*(?:以前|之前|前)")
_RECENT = re.compile(r"近\s*([一二三四五六七八九十\d]+)\s*年")

# 单个年份只有在明确指向"某份资料是哪一年的"时才当检索条件。
#
# 之前的写法是 `(20\d{2})\s*年` 一见到年份就筛，于是"到2027年，老旧住宅电梯要怎么处理"
# 被理解成"只看 2027 年的文件"，而正文其实在 2024 年的文件里——检索前的静默否决，
# 后面的混合召回、重排、词表全都救不回来。政策文件里"到2027年""2027年底前"说的是
# 目标年，不是发文年，这类问法在真实用户里很常见。
#
# 判据是年份后面紧跟的东西：
# - 强标记（年版 / 年文件）：本身就是"版本"的意思，不要求中间有"的"；
# - 弱标记（标准 / 通知 / 方案 …）：必须写成"A年的X"，否则"2027年标准体系"这种
#   目标年读法会被误判成资料年；
# - 发文动词（年发布 / 年印发）：明确在说这份东西是哪年出的。
_DOC_STRONG = r"(?:版|版本|文件|资料)"
_DOC_NOUN = (
    r"(?:标准|规范|规程|导则|图集|通知|政策|意见|方案|报告|规定|办法|细则|要点"
    r"|项目|数据|统计|公报|答复|清单|定额)"
)
_DOC_VERB = r"(?:发布|印发|出台|施行|实施|修订|编制|批准|公布|下发)"
_YEAR_TOKEN = re.compile(r"(20\d{2})\s*年(?:度)?")
# 年份后面紧跟的资料标记："的X"、直接是版本类词、或者是发文动作。
_DOC_AFTER = re.compile(rf"\s*(?:的\s*{_DOC_NOUN}|的?\s*{_DOC_STRONG}|{_DOC_VERB})")
# 两个年份之间的连接词："2021年和2023年的文件" 是一个时间短语，共享同一个资料标记。
_YEAR_JOIN = re.compile(r"\s*(?:和|与|、|及|或|,|，)\s*")

# "到 / 截至 2027 年……"——目标年份。紧跟在年份前面的这几个词一旦出现，
# 哪怕后面写的是"的文件"，也不按资料年份筛（宁可放宽，不可错杀）。
_TARGET_PREFIX = re.compile(r"(?:到|至|截至|截止到|目标|力争|预计|实现)\s*$")
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


def _single_year_span(question: str) -> tuple[int, int] | None:
    """把问句里"指向资料年份"的年份收成一段区间；没有就返回 None。

    一句话点了两个资料年份（"2021 年和 2023 年的文件"）时取跨度而不是只认第一个：
    放宽只会多带进候选，错杀才是不可恢复的。
    """
    tokens = [(int(m.group(1)), m.start(), m.end()) for m in _YEAR_TOKEN.finditer(question)]
    if not tokens:
        return None
    # 从右往左：年份自己带资料标记，或者通过"和/与/、"连到右边那个带标记的年份。
    carried = [_DOC_AFTER.match(question, end) is not None for _, _, end in tokens]
    for i in range(len(tokens) - 2, -1, -1):
        between = question[tokens[i][2] : tokens[i + 1][1]]
        if _YEAR_JOIN.fullmatch(between):
            carried[i] = carried[i] or carried[i + 1]

    years = [
        year
        for (year, start, _), ok in zip(tokens, carried, strict=True)
        if ok and not _TARGET_PREFIX.search(question[:start])
    ]
    if not years:
        return None
    return min(years), max(years)


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
    elif span := _single_year_span(question):
        year_from, year_to = span

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
