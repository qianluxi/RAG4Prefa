from __future__ import annotations

import sqlite3
from pathlib import Path

from rag4prefa import repo
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.ingest import ingest_file
from tests.conftest import FakeEmbedder


def _store(conn: sqlite3.Connection) -> NumpyStore:
    return NumpyStore(conn, FakeEmbedder.dim)


def test_ingest_builds_index_and_facts(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    embedder = FakeEmbedder()
    store = _store(conn)
    outcome = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )

    assert outcome.status == "indexed"
    assert outcome.chunks > 0
    assert store.count() == outcome.chunks

    project = repo.get_project(conn, "测试项目")
    assert project is not None
    assert project.document_count == 1
    assert project.chunk_count == outcome.chunks  # 统计来自 COUNT(*)，不再是硬编码 0


def test_reingest_same_file_is_skipped(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    embedder = FakeEmbedder()
    store = _store(conn)
    first = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )
    second = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )

    assert first.status == "indexed"
    assert second.status == "skipped"
    assert store.count() == first.chunks  # 没有重复向量


def test_force_reingest_does_not_duplicate(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    embedder = FakeEmbedder()
    store = _store(conn)
    first = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )
    ingest_file(
        conn,
        store,
        embedder,
        settings,
        path=docx_file,
        project_name="测试项目",
        doc_year=2024,
        force=True,
    )
    assert store.count() == first.chunks


def test_delete_project_removes_vectors_and_rows(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    embedder = FakeEmbedder()
    store = _store(conn)
    ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )
    chunk_ids = repo.delete_project(conn, "测试项目")
    store.delete(chunk_ids)

    assert repo.get_project(conn, "测试项目") is None
    assert store.count() == 0
    assert repo.fts_search(conn, "悬挑", limit=5) == []


def test_search_filters_by_project(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    embedder = FakeEmbedder()
    store = _store(conn)
    ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="甲项目", doc_year=2024
    )
    ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="乙项目", doc_year=2020
    )

    from rag4prefa.domain import Filter

    all_hits = store.search(embedder.embed_one("悬挑"), limit=50)
    only_jia = store.search(embedder.embed_one("悬挑"), limit=50, filt=Filter(projects=["甲项目"]))
    assert len(only_jia) < len(all_hits)
    assert len(only_jia) > 0

    docs = repo.get_chunks(conn, [cid for cid, _ in only_jia])
    assert {d["project_name"] for d in docs.values()} == {"甲项目"}


def test_fts_search_finds_chinese_terms(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    embedder = FakeEmbedder()
    store = _store(conn)
    ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )
    hits = repo.fts_search(conn, "缓粘结预应力", limit=10)
    assert hits, "BM25 应该能命中中文专有名词"


def test_index_can_be_rebuilt_from_sqlite(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    from rag4prefa.ingest import reindex

    embedder = FakeEmbedder()
    store = _store(conn)
    ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )
    before = store.count()

    store.clear()
    assert store.count() == 0

    result = reindex(conn, store, embedder, settings)
    assert result["chunks"] == before
    assert store.count() == before


class FailingEmbedder(FakeEmbedder):
    """模拟限流 / 超时 / 密钥失效：远程调用一定会抛。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("429 rate limited")


def _update_docx(path: Path, extra: str) -> None:
    from docx import Document

    doc = Document(path)
    doc.add_paragraph(extra)
    doc.save(path)


def test_embedding_failure_keeps_previous_version_searchable(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """更新文档时 embedding 失败，旧版本必须还能被检索到。

    旧实现先删旧片段再调 embedding，一次限流就把原先能用的版本弄没了。
    """
    store = _store(conn)
    first = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="测试项目"
    )
    assert first.status == "indexed"
    before_vectors = store.count()
    before_chunks = repo.chunk_count(conn)
    assert before_vectors == before_chunks > 0

    _update_docx(docx_file, "新增一段关于屋面防水的补充说明。")
    failed = ingest_file(
        conn, store, FailingEmbedder(), settings, path=docx_file, project_name="测试项目"
    )

    assert failed.status == "failed"
    assert "rate limited" in (failed.error or "")
    assert store.count() == before_vectors, "失败不该删掉旧向量"
    assert repo.chunk_count(conn) == before_chunks, "失败不该删掉旧片段"
    assert repo.fts_search(conn, "缓粘结预应力", limit=5), "旧版本仍应可检索"
    assert repo.get_document(conn, "测试项目", "方案.docx").chunk_count == before_chunks


def test_retry_after_failure_is_not_skipped(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """失败过再重试必须真的重跑，而不是因为 content_sha 相同被跳过。"""
    store = _store(conn)
    ingest_file(conn, store, FakeEmbedder(), settings, path=docx_file, project_name="测试项目")
    _update_docx(docx_file, "补充：陶土砖反打工艺的验收要点。")
    failed = ingest_file(
        conn, store, FailingEmbedder(), settings, path=docx_file, project_name="测试项目"
    )
    assert failed.status == "failed"

    retried = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="测试项目"
    )

    assert retried.status == "indexed", "失败后的重试被误判成 skipped"
    assert repo.get_document(conn, "测试项目", "方案.docx").status == "indexed"
    assert repo.fts_search(conn, "陶土砖反打", limit=5), "新版本内容应已入库"


def test_reingest_smaller_version_drops_stale_chunks(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    """新版本片段更少时，多出来的旧片段与旧向量都要被清掉。"""
    from docx import Document

    long_doc = tmp_path / "长文.docx"
    doc = Document()
    doc.add_heading("第1章 结构", level=1)
    for i in range(6):
        doc.add_paragraph(f"第{i}段：悬挑构件的预应力张拉控制要点与摩阻损失。" * 4)
    doc.save(long_doc)

    store = _store(conn)
    big = ingest_file(
        conn, store, FakeEmbedder(), settings, path=long_doc, project_name="测试项目"
    )
    assert big.status == "indexed"
    assert big.chunks > 1

    short_doc = tmp_path / "长文.docx"
    doc = Document()
    doc.add_heading("第1章 结构", level=1)
    doc.add_paragraph("只剩一段：悬挑构件的预应力张拉控制要点。")
    doc.save(short_doc)

    small = ingest_file(
        conn, store, FakeEmbedder(), settings, path=short_doc, project_name="测试项目"
    )

    assert small.status == "indexed"
    assert small.chunks < big.chunks
    assert store.count() == small.chunks, "多出来的旧向量必须被删除"
    assert repo.chunk_count(conn) == small.chunks
    # 索引里剩下的每个 id 都还能在事实表里找到正文，不留指向已删片段的历史向量
    hits = store.search(FakeEmbedder().embed_one("悬挑"), limit=50)
    assert {cid for cid, _ in hits} == set(repo.get_chunks(conn, [c for c, _ in hits]))
