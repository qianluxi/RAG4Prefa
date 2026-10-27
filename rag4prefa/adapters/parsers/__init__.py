"""文档解析器：按扩展名分派，输出统一的 `list[Section]`。

分派放在这里而不是 `ingest.py`：入库流程只想知道"这份文件能不能变成章节"，
不该关心是哪一家的解析库。新增格式（pdf / pptx / xlsx）只动这个文件加一个新模块。

解析器按需导入（延迟到真正用到那一刻）：pdf 依赖 pymupdf，没装它的人
只要不传 pdf 就完全不受影响，而不是一 import 整个包就报缺依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...domain import Section
from ..ocr import OcrOptions

SUPPORTED_SUFFIXES = (".docx", ".pdf")


@dataclass(frozen=True)
class ParseReport:
    """解析过程里"没进正文的部分"的清单。

    存在的理由：之前只有 `list[Section]` 一个返回值，于是"这份 pdf 里第 3、7 页没有
    文字层，一个字都没进来"这件事**在接口上无处表达**，只能静默丢弃——用户以为
    入库了 24 页，实际只有 17 页参与检索，问到了就答不上，还会被当成"检索没做好"。

    与"不支持的格式静默 continue"是同一类问题（2026-10-27 已修一次），只是发生在
    页这一级。区别是这里**不能拒绝整份文件**：17 页能用、7 页缺，整份拒掉比缺页更糟。
    """

    pages: int = 0
    ocr_pages: tuple[int, ...] = ()  # 走 OCR 读出来的页（1-based）
    missing_pages: tuple[int, ...] = ()  # 没有文字层、也没做 OCR 的页
    empty_pages: tuple[int, ...] = ()  # 做了 OCR，但模型说这页没有可读文字

    def note(self) -> str | None:
        """给用户看的一句话。没有异常情况就返回 None，别在界面上刷"一切正常"。"""
        parts: list[str] = []
        if self.ocr_pages:
            parts.append(f"{_page_list(self.ocr_pages)}经 OCR 识别")
        if self.empty_pages:
            parts.append(f"{_page_list(self.empty_pages)}识别后仍无文字，内容缺失")
        if self.missing_pages:
            parts.append(
                f"{_page_list(self.missing_pages)}没有文字层、未做 OCR，内容缺失"
                "（设置 R4P_OCR_MODE=auto 可补上）"
            )
        return "；".join(parts) or None


def _page_list(pages: tuple[int, ...]) -> str:
    if len(pages) <= 6:
        body = "、".join(str(p) for p in pages)
    else:
        body = "、".join(str(p) for p in pages[:6]) + f" 等共 {len(pages)}"
    return f"第 {body} 页"

# 常见但明确不支持的格式：给出"该怎么做"，而不是只说"不支持"。
# `.doc` 单独列出来是因为它看上去跟 `.docx` 只差一个字母，实际是两套完全不同的
# 二进制格式（OLE2 复合文档 vs ZIP 包），python-docx 打不开。用户拿到的报错如果只有
# "不支持 .doc"，第一反应会是"那就改个后缀名吧"——那只会得到一份打不开的 docx。
_UNSUPPORTED_HINTS = {
    ".doc": "老式 Word 二进制格式（不是 zip 包），请用 Word 另存为 .docx",
    ".wps": "WPS 自有格式，请用 WPS 另存为 .docx",
    ".ppt": "只支持文档，不支持演示文稿",
    ".pptx": "只支持文档，不支持演示文稿",
    ".xls": "只支持文档，不支持电子表格",
    ".xlsx": "只支持文档，不支持电子表格",
    ".csv": "只支持文档，不支持电子表格",
    ".txt": "纯文本没有章节结构，无法按标题树切块",
    ".md": "Markdown 暂无解析器",
}


def explain_unsupported(suffix: str) -> str:
    """给一个不支持的后缀，返回"为什么 + 怎么办"。用于目录导入时逐个说明被跳过的文件。"""
    suffix = suffix.lower()
    if not suffix:
        return "没有扩展名，无法判断格式"
    return _UNSUPPORTED_HINTS.get(suffix, f"不支持的格式 {suffix}")


def parse_document_report(
    path: str | Path, *, ocr: OcrOptions | None = None
) -> tuple[list[Section], ParseReport]:
    """解析并附上"哪些页没进来"。`ocr=None` 表示不做 OCR。

    docx 没有物理页码（分页是渲染期决定的），所以它的报告恒为"无异常"——
    这不是偷懒：docx 里不存在"某页没文字层"这件事，正文一定在文件里。
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        from .docx import parse_docx

        return parse_docx(path), ParseReport()
    if suffix == ".pdf":
        from .pdf import parse_pdf_report

        return parse_pdf_report(path, ocr=ocr)
    raise ValueError(_unsupported_message(suffix, path))


def parse_document(path: str | Path) -> list[Section]:
    """只要章节，不要报告。给脚本、测试与"我不关心缺哪几页"的调用方。"""
    return parse_document_report(path)[0]


def _unsupported_message(suffix: str, path: str | Path) -> str:
    return (
        f"不支持的文档格式「{suffix or Path(path).name}」：目前只能处理 "
        f"{' / '.join(SUPPORTED_SUFFIXES)}。文件未被入库。"
    )
