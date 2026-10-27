"""测试夹具：不联网、不依赖 Milvus，全部用内存 SQLite + 确定性假向量。"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from docx import Document

from rag4prefa.config import Settings
from rag4prefa.db import connect, init_db

# 这个快照必须在**收集测试之前**取：pytest 会把所有测试模块先 import 一遍，
# 而 `test_milvus_store.py` 顶部的 `import pymilvus` 会顺手把 `.env` 灌进
# `os.environ`。所以在每个用例里现取快照是没用的——那时已经被污染了。
_ENV_BEFORE_COLLECTION = dict(os.environ)
_ENV_PREFIX = "R4P_"


def _restore_environ() -> None:
    """把 `R4P_*` 恢复到收集测试之前的快照。"""
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIX) and key not in _ENV_BEFORE_COLLECTION:
            del os.environ[key]
    for key, value in _ENV_BEFORE_COLLECTION.items():
        if key.startswith(_ENV_PREFIX) and os.environ.get(key) != value:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _isolate_environ() -> Iterator[None]:
    """每个用例**前后**都把 `R4P_*` 恢复成收集前的快照。

    **pymilvus 在 import 时用 environs 把仓库根目录的 `.env` 读进 `os.environ`**
    （2026-10-27 实测）。这不是"某个用例改了配置"，而是"导入一个包就改了全局状态"，
    于是同一进程里后跑的用例看到的配置取决于**文件执行顺序**——它们在
    `Settings(_env_file=None)` 之下也躲不开，因为值已经在环境变量里了。

    这次就是这么翻车的：打开 OCR 之后，排在 `test_milvus_store` 之后的用例读到
    `R4P_OCR_MODE=auto`，"默认关闭"那条断言变红，而单独跑又是绿的。

    只在收尾恢复是不够的（第二十九轮）：那只保证"用例之间不互相串"，而**每个用例
    都还带着本机 `.env` 在跑**。把 `.env` 的 `R4P_ALLOW_SERVER_KEY_FALLBACK` 改成
    false 之后，7 条断言"匿名请求应该能过"的用例集体变红——它们与这个开关毫无关系，
    只是读了开发机上的真实配置。开始前也恢复一次，用例看到的配置才只由它自己决定。

    只动 `R4P_*`：那是本项目的全部配置面，而**清空整个 os.environ 会踩到 pytest
    自己**——它在用例收尾时还要 pop `PYTEST_CURRENT_TEST`，被我们删掉就是每个用例
    一条 KeyError。
    """
    _restore_environ()
    yield
    _restore_environ()


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
    # `_env_file=None`：测试不能受本机 .env 影响。否则开发者一开 OCR 或换向量后端，
    # 用例行为就跟着变——更糟的是真去调付费接口，而"离线全绿"是这套测试的底线。
    return Settings(
        _env_file=None,
        data_dir=tmp_path,
        embedding_api_key="test",
        llm_api_key="test",
        chunk_max_tokens=120,
        chunk_min_tokens=30,
        vector_backend="numpy",
        ocr_mode="off",
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
