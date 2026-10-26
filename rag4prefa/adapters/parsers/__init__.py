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
