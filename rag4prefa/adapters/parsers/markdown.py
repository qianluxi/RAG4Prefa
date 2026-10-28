"""Markdown 解析：把 `.md` 还原成与 docx / pdf 同一棵章节树。

为什么要有它：课题语料里政策原文、试点案例、造价摘要有大半是网页抓下来整理的 md。
在它之前这些文件**一律被拒**（`_UNSUPPORTED_HINTS` 写着"Markdown 暂无解析器"），
而拒绝的理由并不成立——md 的标题是显式语法，比 pdf 的"字号大一点"可靠得多。

两种标题来源，按可靠程度取：

1. ATX 标题 `# / ## / ###`：语法层面就是结构，直接定级；
2. 没有 `#` 的政策原文（网页正文常见）：退回 `text_heading_level`，与 pdf 共用
   "第 X 章 / 一、 / （一） / 1.1" 那套编号判据——同一份文件换个格式，章节树不该变。

**清洗只做两件事，且都可解释**：去掉 `**`、`反引号`、图片语法这类行内标记
（它们进 BM25 与向量只会变成噪声字符），以及丢掉**整行**命中网页模板词表的行
（索引号 / 发布机构 / 浏览次数……）。不做"智能去导航"：判错的代价是正文消失，
而正文消失在这套系统里是不可见的——用户只会觉得"检索没抓到"。

不解析的语法（故意）：表格转成逐行正文而不是结构化表；脚注、HTML 块原样保留。
它们都不影响"这一段讲什么"，而按结构重建它们要额外一套渲染，收益不在检索上。
"""

from __future__ import annotations

import re
from pathlib import Path

from ...domain import Section
from .headings import build_sections, clean_text, text_heading_level

_ATX = re.compile(r"^(#{1,6})\s+(\S.*)$")
# 少数文件写 `#标题`（不带空格）。只在 # 后面紧跟非 #、非空格时容错，
# 否则 `###` 这种纯分隔线会被读成标题。
_ATX_TIGHT = re.compile(r"^(#{1,6})([^\s#].*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")
_TABLE_DIVIDER = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_RULE = re.compile(r"^\s*([-*_])\s*(\1\s*){2,}$")
_LIST_MARK = re.compile(r"^\s{0,3}[-*+]\s+")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_INLINE = re.compile(r"(\*\*|__|~~|`)")
_QUOTE = re.compile(r"^\s*>\s?")

# 「看着像编号、其实是元信息」的两类行。md 没有字号可依，`text_heading_level`
# 会把它们读成一级标题——抓下来的政策页面里，索引号 / 发布时间恰好排在正文之前，
# 于是第一个"章"变成了 `002482242/2023-79897`，整篇的章节树从这里歪掉。
_DATE_LINE = re.compile(
    r"^\d{4}\s*[-/.年]\s*\d{1,2}(\s*[-/.月]\s*\d{1,2}\s*日?)?"
    r"(\s+\d{1,2}:\d{2}(:\d{2})?)?$"
)
_CODE_LINE = re.compile(r"^[\d\s\-/._]+$")

# 网页模板行。判据是"整行就是这个标签"或"标签：值"——只丢模板本身，
# 不碰同一行里的正文（有些页把标题和"发布时间"写在一行）。
_NOISE_LABELS = frozenset(
    {
        "首页",
        "政府信息公开",
        "政府信息公开目录",
        "法定主动公开内容",
        "政策文件及解读",
        "其他文件",
        "工作信息",
        "安全更新",
        "当前位置",
        "索引号",
        "发布机构",
        "发布单位",
        "公开方式",
        "公开范围",
        "主题分类",
        "体裁分类",
        "文号",
        "文　　号",
        "文 号",
        "发文机关",
        "发布时间",
        "发布日期",
        "公开日期",
        "成文日期",
        "信息来源",
        "浏览次数",
        "打印",
        "打印关闭",
        "关闭",
        "扫一扫",
        "分享",
        "主动公开",
        "依申请公开",
        "不予公开",
    }
)


def _read_text(path: Path) -> str:
    """网页抓下来的 md 编码不一定干净：先按 utf-8（含 BOM）读，再退 gbk。

    两个都失败时用 replace 而不是抛错：一份混了乱码的文件里，能读的部分仍然有
    检索价值，而"整份拒收"会把 90% 可用的正文一起扔掉。
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _norm_label(text: str) -> str:
    return text.replace(" ", "").replace("\u3000", "").strip()


_NOISE_NORM = frozenset(_norm_label(label) for label in _NOISE_LABELS)


def _is_noise(line: str) -> bool:
    """这一行是不是"不该进正文"的元信息。

    两类，都只判**整行**：网页模板标签（索引号 / 发布机构 / 浏览次数……），
    以及模板标签的值——日期与流水号。后者是上一类被丢之后的残留（`索引号` 那行没了，
    `002482242/2023-79897` 还留着），留着只会让前言片段变成一串编号。
    """
    text = line.strip()
    for sep in (":", "："):
        head, found, _ = text.partition(sep)
        if found and _norm_label(head) in _NOISE_NORM:
            return True
    if _norm_label(text) in _NOISE_NORM:
        return True
    return bool(_DATE_LINE.match(text) or _CODE_LINE.match(text))


def _clean_inline(line: str) -> str:
    line = _IMAGE.sub("", line)
    line = _INLINE.sub("", line)
    return clean_text(line)


def _skip_front_matter(lines: list[str]) -> int:
    """YAML front matter（首行 `---` 包起来的一段）不是正文，返回起始行号。

    判据加了"块内至少一行 `key: value`"：单独两行 `---` 也可能只是分隔线，
    没有这个条件会把紧随其后的标题一起吃掉。
    """
    if not lines or lines[0].strip() != "---":
        return 0
    for idx in range(1, min(len(lines), 60)):
        if lines[idx].strip() in {"---", "..."}:
            body = lines[1:idx]
            if any(re.match(r"^[\w\u4e00-\u9fff-]+\s*:", row) for row in body):
                return idx + 1
            return 0
    return 0


def _split_heading(line: str) -> tuple[int, str] | None:
    """返回 (级别, 标题文本)；不是标题返回 None。

    ATX 的 `#` 必须在这里剥掉再当标题：写进 `Section.title` 的话，章节路径会变成
    `# 目次`，而章节路径参与 `chunk_id`——同一份文件换个标题级别，整份文档的 id 全变。

    日期与流水号**不在这里判**：它们在 `_is_noise` 那一步就整行丢掉了，判据只该有一处。
    （列表项 `- 2024-09-10` 也躲不过去：`_CODE_LINE` 的字符集里就有 `-` 与空格。）
    """
    if m := _ATX.match(line):
        return len(m.group(1)), clean_text(m.group(2))
    if m := _ATX_TIGHT.match(line):
        return len(m.group(1)), clean_text(m.group(2))
    level = text_heading_level(line)
    return None if level is None else (level, line)


def parse_markdown(path: str | Path) -> list[Section]:
    """解析为 Section 列表。`page` 恒为 None——md 没有物理页。"""
    lines = _read_text(Path(path)).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    events: list[tuple[int | None, str]] = []
    in_code = False

    for line in lines[_skip_front_matter(lines) :]:
        stripped = line.strip()
        if not stripped:
            continue

        if _FENCE.match(stripped):
            in_code = not in_code
            continue
        if in_code:
            # 代码块内容当正文保留：有些抓取结果把数据表塞在代码块里
            events.append((None, stripped))
            continue

        if _RULE.match(stripped):
            continue
        if _TABLE_DIVIDER.match(stripped):
            continue
        if _is_noise(stripped):
            continue

        text = _QUOTE.sub("", stripped) if stripped.startswith(">") else stripped
        text = _LIST_MARK.sub("", text)
        text = _clean_inline(text)
        if not text:
            continue

        if heading := _split_heading(text):
            events.append(heading)
        else:
            events.append((None, text))

    return build_sections(events)
