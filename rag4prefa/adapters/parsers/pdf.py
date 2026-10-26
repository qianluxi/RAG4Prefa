"""pdf 解析：把 pdf 还原成与 docx 同形的「标题层级 + 正文」序列。

为什么不能只调 `page.get_text()`：那样拿到的是一整页没有结构的文字，章节树就没了，
而章节路径是 `chunking._chunk_id` 的组成部分——没有路径，引用只能指到"某文档某段"，
指不到"第 3.2 节"。

pdf 没有样式表，能用的线索只有三样：**字号、加粗、位置**。所以策略是"先统计、后判级"：

1. 全文按字符数加权求**正文字号**（占字符最多的那个）——不用绝对磅值，
   因为同一份文档可能 9.1pt，另一份 12pt，写死阈值必然错一半；
2. block 内按「字号 + 加粗」相同的连续行切成 run。一个 block 里既有标题又有正文时，
   不切就会把标题拼进正文，长度一超 60 字就再也认不出来；
3. 比正文大或加粗的短行是标题候选，**编号（一、/ 第X章 / 1.1）比字号可信**：
   同一份文档里"一、"和"（一）"常常同字号，靠编号才能分出层级；
4. 没有编号的候选按字号降序排名给层级。

页眉页脚单独处理（见 `_drop_furniture`）：页边区域里的短行、跨页重复的行、页码。
漏掉它们的代价比想象中大——一份 60 页的 pdf 会多出 120 条"第 3 页"，
既污染正文，又让 BM25 的文档频率失真。

**不做 OCR**：没有文字层的扫描件直接报错并说明原因。这条路要么依赖本地模型，
要么依赖外部服务，两条都超出当前部署约束；静默返回空正文会让用户以为文件有问题。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from ...domain import Section
from .headings import build_sections, clean_text, looks_like_title, text_heading_level

try:  # pymupdf 1.24 起同时提供 pymupdf 与 fitz 两个名字，早于它的只有 fitz
    import pymupdf
except ImportError:  # pragma: no cover - 取决于安装的版本
    try:
        import fitz as pymupdf  # type: ignore[no-redef]
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "解析 PDF 需要 pymupdf：pip install 'pymupdf>=1.24,<2'"
            "（不要装 2.x——它会把 onnxruntime 版面模型一并拉进来）"
        ) from exc

_BOLD_FLAG = 16  # span.flags 的第 5 位；不按字体名猜，因为不同厂商命名差异太大

_EDGE_TOP = 0.10  # 页面顶部 10% 视为页眉区
_EDGE_BOTTOM = 0.93  # 底部 7% 视为页脚区
_FURNITURE_MAX_LEN = 40  # 页边区域里超过这个长度的行按正文保留
_FURNITURE_MAX_GROW = 0.8  # 页边区域里字号不超过"正文 + 0.8"才可能是页眉页脚
_FURNITURE_BUDGET = 0.30  # 被判定为页眉页脚的行超过这个比例，说明判据失效，整体放行

_MIN_TEXT_CHARS = 50  # 全文可提取字符少于这个数，视为没有文字层
_FURNITURE_RATIO_MIN = 20  # 少于这么多段时不用比例保护：短文档里页眉本来占比就高
_CONTINUATION_GAP = 1.0  # 行间空隙不超过一个字号，就认为是同一段内的换行

_PAGE_NO = re.compile(r"^[-—–\s]*\d{1,4}[-—–\s]*$")
_PAGE_NO_CN = re.compile(r"^第\s*\d+\s*页(\s*[,，/]?\s*共\s*\d+\s*页)?$")
# 落款日期常被加粗居中排版，视觉上跟标题一样，但它是文末内容而不是章节
_DATE_LINE = re.compile(
    r"^[〇零一二三四五六七八九十\d]{2,4}\s*[年./\-]\s*\d{1,2}\s*[月./\-]\s*\d{1,2}\s*日?$"
)


@dataclass(frozen=True)
class _Item:
    """一段候选内容：一个同款式的行组，或者一张表。"""

    kind: str  # "text" | "table"
    text: str
    size: float
    bold: bool
    y0: float
    y1: float
    page: int


def _needs_space(left: str, right: str) -> bool:
    """拼接断行时要不要补空格：中文直接接，西文之间补一个。"""
    return bool(left and right) and left.isascii() and right.isascii()


def _join_lines(parts: list[str]) -> str:
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and _needs_space(out[-1], part[0]):
            out += " "
        out += part
    return out


def _line_of(line: dict) -> tuple[str, float, bool]:
    """一行文本 + 该行的代表字号 + 是否加粗。"""
    spans = [s for s in line["spans"] if s["text"].strip()]
    if not spans:
        return "", 0.0, False
    text = "".join(s["text"] for s in spans)
    weight: Counter[float] = Counter()
    bold_chars = 0
    for span in spans:
        size = round(float(span["size"]), 1)
        weight[size] += len(span["text"])
        if span["flags"] & _BOLD_FLAG:
            bold_chars += len(span["text"])
    size = weight.most_common(1)[0][0]
    return text, size, bold_chars * 2 >= len(text)


def _runs(lines: list[tuple[str, float, bool]]) -> list[tuple[str, float, bool, float, float]]:
    """按「字号 + 加粗」把连续行并成 run，并给出该 run 的 y 范围。"""
    out: list[tuple[str, float, bool, float, float]] = []
    buf: list[str] = []
    key: tuple[float, bool] | None = None
    top = bottom = 0.0

    def flush() -> None:
        nonlocal buf, key, top, bottom
        if buf:
            out.append((_join_lines(buf), key[0], key[1], top, bottom))  # type: ignore[index]
        buf, key, top, bottom = [], None, 0.0, 0.0

    for text, size, bold, y0, y1 in lines:
        k = (size, bold)
        if key is not None and k != key:
            flush()
        if key is None:
            top, bottom = y0, y1
        else:
            bottom = y1
        key = k
        buf.append(text)
    flush()
    return out


def _overlaps(bbox: tuple, box: tuple) -> bool:
    """文本块是否落在表格里（重叠面积占自身一半以上）。"""
    x0, y0 = max(bbox[0], box[0]), max(bbox[1], box[1])
    x1, y1 = min(bbox[2], box[2]), min(bbox[3], box[3])
    if x1 <= x0 or y1 <= y0:
        return False
    own = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if own <= 0:
        return False
    return (x1 - x0) * (y1 - y0) / own > 0.5


def _table_rows(table) -> list[str]:
    rows: list[str] = []
    for raw in table.extract():
        cells = [clean_text(c) for c in raw if c]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def _collect(doc) -> list[_Item]:
    items: list[_Item] = []
    for pno, page in enumerate(doc):
        try:
            found = list(page.find_tables().tables)
        except Exception:  # noqa: BLE001 - 表格识别是增强项，失败就当没有表
            found = []
        boxes = [t.bbox for t in found]
        page_items: list[_Item] = []
        for table in found:
            rows = _table_rows(table)
            if len(rows) >= 2:
                page_items.append(
                    _Item("table", "\n".join(rows), 0.0, False, table.bbox[1], table.bbox[3], pno)
                )

        # sort=True 让 pymupdf 按阅读顺序（而非内部对象顺序）给块，多栏排版才不会串栏
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block["type"] != 0:
                continue
            if any(_overlaps(block["bbox"], box) for box in boxes):
                continue
            raw: list[tuple[str, float, bool, float, float]] = []
            for line in block["lines"]:
                text, size, bold = _line_of(line)
                if text.strip():
                    raw.append((text, size, bold, line["bbox"][1], line["bbox"][3]))
            for text, size, bold, y0, y1 in _runs(raw):
                page_items.append(_Item("text", text, size, bold, y0, y1, pno))

        # 表格与正文必须按 y 排回原顺序：先收表格再收文本会让整页的表格都跑到正文前面
        page_items.sort(key=lambda item: (item.y0, item.y1))
        items.extend(page_items)
    return items


def _body_size(doc) -> float:
    """占字符最多的字号就是正文。

    统计的是**原始 span**，不是过滤后的条目：表格里的字同样是正文，而表格被识别成
    `table` 后就没有字号了。只按"文本条目"统计，会让"一页表格 + 一个标题"这种页面
    把标题字号当成正文——接着标题就不再"比正文大"，被当成页边短行删掉，整篇退化成
    "前言"一节（2026-10-26 实测）。

    页眉页脚也在统计里，但它们的字数相对全文可以忽略；短文档里页眉占比高一些，
    只要正文段落稍长就不会翻盘。
    """
    weight: Counter[float] = Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        weight[round(float(span["size"]), 1)] += len(span["text"])
    return weight.most_common(1)[0][0] if weight else 0.0


def _skeleton(text: str) -> str:
    """去掉数字与空白，用于识别"只差页码"的重复行。"""
    return re.sub(r"[\d\s]", "", text)


def _drop_furniture(items: list[_Item], doc, body: float) -> list[_Item]:
    """去掉页眉、页脚、页码。

    三条判据，都要求"在页边区域"（正文段落极少起于页面最上方或止于最下方）：

    - 纯页码：`3` / `- 3 -` / `第 3 页`；
    - **跨页重复**：去掉数字后同一串出现在六成以上页面（"XX项目施工方案"这类页眉）；
    - 页边短行且字号不超过正文 + 0.8：打印水印、日期戳、联系方式。

    最后一条故意允许"比正文略大"：Chrome 打印出来的页眉常比正文大 0.5~0.6pt，
    只按"不大于正文"判会全部漏掉。真正要保护的是文档大标题，它一般比正文大 3pt 以上，
    仍会留下。

    误判保护有两条：**丢完一行不剩就全部放回**（判据在这份文档上显然失效了），
    以及长文档里丢弃比例超过 `_FURNITURE_BUDGET` 时放回（整页表格、红头文件这类排版
    会让"页边"落在正文里）。比例保护只对长文档生效——一页两页的短文件里，
    页眉页脚本来就占去一半的段数，按比例卡会把该丢的全留下。
    """
    pages = max(doc.page_count, 1)
    seen: dict[str, set[int]] = defaultdict(set)
    for item in items:
        if item.kind == "text":
            seen[_skeleton(item.text)].add(item.page)
    repeat_threshold = max(2, pages * 0.6)
    repeated = {key for key, hits in seen.items() if len(hits) >= repeat_threshold}

    kept: list[_Item] = []
    dropped = 0
    for item in items:
        if item.kind == "text" and _is_furniture(item, doc, body, repeated):
            dropped += 1
            continue
        kept.append(item)

    if not kept:
        return items
    if len(items) >= _FURNITURE_RATIO_MIN and dropped > len(items) * _FURNITURE_BUDGET:
        return items
    return kept


def _is_furniture(item: _Item, doc, body: float, repeated: set[str]) -> bool:
    height = doc[item.page].rect.height or 1.0
    in_edge = item.y0 < height * _EDGE_TOP or item.y1 > height * _EDGE_BOTTOM
    if not in_edge:
        return False
    if item.size >= body + 1.0:
        # 页边的大字是文档标题（首页顶部那种），不是页眉。页眉最多比正文大半磅左右。
        return False
    text = item.text.strip()
    if _PAGE_NO.match(text) or _PAGE_NO_CN.match(text):
        return True
    if _skeleton(text) in repeated:
        return True
    return len(text) <= _FURNITURE_MAX_LEN and item.size <= body + _FURNITURE_MAX_GROW


def _continues(prev: _Item, cur: _Item) -> bool:
    """`cur` 是不是 `prev` 同一段的下一截。

    pdf 里一段话常被拆进多个 block（生成器按文本对象切，不按段落切），于是
    "住房城乡 / 建设科技创新发展" 这样的词会在中间插一个换行。分词与向量都会被它干扰。

    判据只用**行距**：段内换行是紧贴的（实测这份文档 gap≈7.5，字号 9.1），
    段与段之间留白更大（gap≈13.6）。加上"上一句没说完"——不以句末标点收尾。
    不用首行缩进：中文首行缩进两字符，与 x 坐标体系绑得太死，换一份文档就得重调。
    """
    if prev.kind != "text" or cur.kind != "text":
        return False
    if prev.page != cur.page or abs(prev.size - cur.size) > 0.2 or prev.bold != cur.bold:
        return False
    if prev.text.rstrip().endswith(("。", "！", "？", "；", "…", ".", "!", "?")):
        return False
    return (cur.y0 - prev.y1) <= prev.size * _CONTINUATION_GAP


def _merge_continuations(items: list[_Item]) -> list[_Item]:
    """把同一段被拆开的几截拼回去。合并只影响本段，不跨标题。"""
    out: list[_Item] = []
    for item in items:
        prev = out[-1] if out else None
        if prev is not None and _continues(prev, item):
            out[-1] = replace(
                prev, text=_join_lines([prev.text, item.text]), y1=max(prev.y1, item.y1)
            )
        else:
            out.append(item)
    return out


def _size_ranks(items: list[_Item], body: float) -> dict[float, int]:
    """「看着像标题」的字号按降序排名，给没有编号的标题一个层级。"""
    weight: Counter[float] = Counter()
    for item in items:
        if item.kind != "text":
            continue
        if item.size >= body + 0.3 and (item.bold or item.size >= body + 1.0):
            weight[item.size] += len(item.text)
    return {size: rank + 1 for rank, size in enumerate(sorted(weight, reverse=True))}


def _level_of(
    item: _Item, body: float, ranks: dict[float, int], *, trust_numbers: bool = False
) -> int | None:
    """这一段是不是标题；是的话第几级（0 表示相对级别）。

    关键取舍：**编号只用来定级，不用来认定"这是标题"**。

    看起来反直觉，但写成"看到（一）就认为是标题"会毁掉整份文档：pdf 里段落首行
    常常正好以（一）/（2）开头，前半截不到 60 字又不以标点收尾，于是被提成标题，
    真正的正文挂在它下面，章节树变成每段一节。现实中的标题一定还有别的痕迹——
    字体更大或加粗——所以要求"先有视觉证据，再看编号叫什么名字"。

    `trust_numbers=True` 是兜底：整份文档一个视觉标题都没有（正文与标题同字号、
    都不加粗，只靠编号分层的公文），此时退回"只信编号"，总比整篇塞进一节强。
    """
    if item.kind != "text" or not looks_like_title(item.text):
        return None
    if _DATE_LINE.match(item.text.strip()):
        return None  # 落款日期常常加粗居中，但它不是章节
    big = item.size >= body + 1.0
    bold_enough = item.bold and item.size >= body - 0.1
    if not (big or bold_enough or trust_numbers):
        return None
    by_number = text_heading_level(item.text)
    if by_number is not None:  # 1..4 为绝对级别，0 为相对级别
        return by_number
    if big or bold_enough:
        return ranks.get(item.size, 1)
    return None


def _events(
    items: list[_Item], body: float, ranks: dict[float, int], *, trust_numbers: bool
) -> list[tuple[int | None, str]]:
    out: list[tuple[int | None, str]] = []
    for item in items:
        if item.kind == "table":
            out.extend((None, line) for line in item.text.split("\n"))
            continue
        level = _level_of(item, body, ranks, trust_numbers=trust_numbers)
        out.append((level, item.text))
    return out


def parse_pdf(path: str | Path) -> list[Section]:
    """解析为 Section 列表。认不出标题的文档退化成一节，不报错。"""
    with pymupdf.open(str(path)) as doc:
        if doc.page_count == 0:
            raise ValueError("这个 pdf 没有任何页面，无法解析")
        items = _collect(doc)
        total_chars = sum(len(item.text) for item in items)
        if total_chars < _MIN_TEXT_CHARS:
            raise ValueError(
                f"这个 pdf 几乎取不到文字（可提取字符 {total_chars} 个），"
                "大概率是扫描件或图片导出的。当前管线不做 OCR，请提供带文字层的版本，"
                "或先用外部工具（如 Acrobat、abbyy）做一次文字识别。"
                "文件未被入库，索引与归档没有任何改动。"
            )

        body = _body_size(doc)
        items = _drop_furniture(items, doc, body)
        items = _merge_continuations(items)
        ranks = _size_ranks(items, body)

        events = _events(items, body, ranks, trust_numbers=False)
        if items and not any(level is not None for level, _ in events):
            # 整篇没有一个"看着像标题"的行：多半是标题与正文同字号、都不加粗，
            # 只能靠编号分层的老公文。退回"只信编号"，总比整篇塞进一节强。
            events = _events(items, body, ranks, trust_numbers=True)

    return build_sections(events)
