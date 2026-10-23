from __future__ import annotations

from pathlib import Path

from rag4prefa.adapters.parsers.docx import parse_docx
from rag4prefa.domain import SectionKind


def test_heading_hierarchy_and_preamble(docx_file: Path) -> None:
    sections = parse_docx(docx_file)
    titles = [s.title for s in sections]

    assert titles[0] == "前言"  # 封面文字不能被丢弃
    chapters = [s for s in sections if s.kind is SectionKind.chapter]
    assert [s.title for s in chapters] == ["第1章 研究背景", "第2章 技术路线"]

    section = next(s for s in sections if s.title == "1.1 悬挑结构")
    assert section.kind is SectionKind.section
    assert section.path == ["第1章 研究背景"]
    assert section.full_path == ["第1章 研究背景", "1.1 悬挑结构"]
    assert "缓粘结预应力" in section.text


def test_body_text_attaches_to_nearest_heading(docx_file: Path) -> None:
    sections = parse_docx(docx_file)
    chapter = next(s for s in sections if s.title == "第1章 研究背景")
    # 章标题与其后、下一个子标题之前的正文应归到这一章
    assert "立体聚落" in chapter.text
    assert "缓粘结" not in chapter.text
