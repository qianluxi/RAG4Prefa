"""检索不能跨过索引切换。

复审给的时序：查询用模型 A 通过一致性检查 → 正在等 embedding 返回 → 另一个进程
做完全量重建、索引换成模型 B → 原查询恢复执行，仍拿 A 的查询向量去查 B 的索引。
维度一样时 Milvus 与 numpy 都会"查得出结果"——不报错，只是相似度算在两种语义之间。

维护锁保护的是"写"，不能把查询也塞进去排队（一次重建几分钟，提问不能陪着等）。
所以这里验证的是**代次确认**：算完向量确认一次、检索完确认一次，跨代就作废重来。
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

from rag4prefa import ask
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.consistency import IndexMismatch
from rag4prefa.domain import Filter
from rag4prefa.ingest import ingest_file, reindex
from tests.conftest import FakeEmbedder


class BlockingQueryEmbedder(FakeEmbedder):
    """`embed_one` 卡住：模拟"查询正等着 embedding 服务返回那几百毫秒"。"""

    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.started = started
        self.release = release

    def embed_one(self, text: str) -> list[float]:
        self.started.set()
        assert self.release.wait(20), "测试没有放行"
        return super().embed_one(text)


class OtherModelEmbedder(FakeEmbedder):
    """同一套假向量，只改模型名——模拟"换模型做全量重建"。"""

    model = "fake-embedder-v2"


def _seed(conn: sqlite3.Connection, store: NumpyStore, settings: Settings, docx: Path) -> None:
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx, project_name="甲项目", doc_year=2024
    )
    assert outcome.status == "indexed"


def test_a_query_is_void_when_the_index_is_switched_while_it_waits_for_its_vector(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """切换发生在"算查询向量"期间：这次检索必须作废，而不是接着查新索引。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)

    started, release = threading.Event(), threading.Event()
    outcome: dict[str, object] = {}

    def run_query() -> None:
        try:
            outcome["hits"] = asyncio.run(
                ask.search(
                    conn,
                    store,
                    BlockingQueryEmbedder(started, release),
                    settings,
                    question="悬挑结构的张拉控制要点",
                    filt=Filter(),
                )
            )
        except BaseException as exc:  # noqa: BLE001 - 线程里的异常要带回主线程
            outcome["error"] = exc

    worker = threading.Thread(target=run_query, daemon=True)
    worker.start()
    assert started.wait(20), "查询没进入 embedding 阶段"

    # 查询卡住的时候，另一个进程把索引整体换成了另一个模型
    reindex(conn, NumpyStore(conn, FakeEmbedder.dim), OtherModelEmbedder(), settings)

    release.set()
    worker.join(30)

    error = outcome.get("error")
    assert isinstance(error, IndexMismatch), f"应该拒绝而不是返回结果：{outcome}"
    assert "代次" in str(error), "要说清是索引被换过，而不是一句通用的不一致"


def test_a_query_is_not_void_just_because_someone_else_ingested(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """内容变了不等于向量空间变了：在途查询的结果依然成立，不能报错。

    如果两个计数器合成一个，"有人上传文档时别人就问不了问题"——这比原问题更烦人，
    而且会让人以为系统坏了。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)

    started, release = threading.Event(), threading.Event()
    outcome: dict[str, object] = {}

    def run_query() -> None:
        try:
            outcome["hits"] = asyncio.run(
                ask.search(
                    conn,
                    store,
                    BlockingQueryEmbedder(started, release),
                    settings,
                    question="悬挑结构的张拉控制要点",
                    filt=Filter(),
                )
            )
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(target=run_query, daemon=True)
    worker.start()
    assert started.wait(20)

    # 另一个项目入库（内容变了，模型没变）
    other = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="乙项目", doc_year=2024
    )
    assert other.status == "indexed"

    release.set()
    worker.join(30)

    assert "error" not in outcome, f"内容变更不该把在途查询判死：{outcome.get('error')}"
    hits, _, _ = outcome["hits"]  # type: ignore[misc]
    assert hits, "索引与模型都没换，这次查询应当正常返回结果"
