"""文档解析器：按扩展名分派，输出统一的 `list[Section]`。

分派放在这里而不是 `ingest.py`：入库流程只想知道"这份文件能不能变成章节"，
不该关心是哪一家的解析库。新增格式（pdf / pptx / xlsx）只动这个文件加一个新模块。

解析器按需导入（延迟到真正用到那一刻）：pdf 依赖 pymupdf，没装它的人
只要不传 pdf 就完全不受影响，而不是一 import 整个包就报缺依赖。
"""

from __future__ import annotations

from pathlib import Path

from ...domain import Section

SUPPORTED_SUFFIXES = (".docx", ".pdf")

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


def parse_document(path: str | Path) -> list[Section]:
    """按扩展名选解析器。不支持的格式直接报错，不做"尽力而为"。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        from .docx import parse_docx

        return parse_docx(path)
    if suffix == ".pdf":
        from .pdf import parse_pdf

        return parse_pdf(path)
    raise ValueError(
        f"不支持的文档格式「{suffix or Path(path).name}」：目前只能处理 "
        f"{' / '.join(SUPPORTED_SUFFIXES)}。文件未被入库。"
    )
