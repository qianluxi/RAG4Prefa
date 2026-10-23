"""测试夹具：不联网、不依赖 Milvus，全部用内存 SQLite + 确定性假向量。"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from docx import Document

from rag4prefa.config import Settings
from rag4prefa.db import connect, init_db


class FakeEmbedder:
    """确定性向量：同一文本永远得到同一向量，便于断言检索行为。"""

    model = "fake-embedder"
    dim = 16

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[: self.dim]]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        embedding_api_key="test",
        llm_api_key="test",
        chunk_max_tokens=120,
        chunk_min_tokens=30,
        vector_backend="numpy",
    )


@pytest.fixture
def conn(settings: Settings) -> sqlite3.Connection:
    settings.ensure_dirs()
    init_db(settings.db_path)
    connection = connect(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    """构造一份有层级、有长正文的 docx，模拟真实的课题文档。"""
    doc = Document()
    doc.add_paragraph("课题研究方案（封面）")
    doc.add_heading("第1章 研究背景", level=1)
    doc.add_paragraph("随着百年建筑理念推进，立体聚落形态逐步应用于公共建筑。" * 6)
    doc.add_heading("1.1 悬挑结构", level=2)
    doc.add_paragraph("缓粘结预应力技术适用于大悬挑构件，需控制张拉时机与摩阻损失。" * 5)
    doc.add_heading("1.2 外立面", level=2)
    doc.add_paragraph("陶土砖饰面存在反打、粘贴、叠涩错砌等多种工艺。" * 3)
    doc.add_heading("第2章 技术路线", level=1)
    doc.add_paragraph("采用设计施工协同的质量总控体系，覆盖主体、围护与机电。" * 4)
    path = tmp_path / "方案.docx"
    doc.save(path)
    return path
