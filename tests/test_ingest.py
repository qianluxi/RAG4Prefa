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
