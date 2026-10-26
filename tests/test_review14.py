"""第十四轮复审（基线 04d122c）指出的边界。

这一轮的问题都不是"忘记加锁"，而是**动作与它依赖的状态错位**：

- 崩溃恢复会删集合，却发生在维护锁之外——别人正在重建，它删的是人家的半成品；
- 内存快照读行与读代次不是同一瞬间，旧行可以配上新代次，从此不再刷新；
- 装配后端发生在拿锁之前，等锁的几分钟里后端可能已经被换成另一个；
- 入库失败只删"新增"的 id，被覆盖的旧行（doc_year）留在新版上。

共同点是**这些窗口都不会让条数变**：条数、模型、维度全都对得上，所以计数校验
一条也发现不了。测试因此都要断言"内容/过滤结果"，而不是"条数没变"。
"""

from __future__ import annotations

import sqlite3
from array import array
from pathlib import Path

import pytest
from docx import Document

from rag4prefa import consistency, repo
from rag4prefa.adapters import vectors as vectors_module
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.domain import Filter
from rag4prefa.ingest import ingest_file
from tests.conftest import FakeEmbedder


def _store(conn: sqlite3.Connection) -> NumpyStore:
    return NumpyStore(conn, FakeEmbedder.dim)


def _query_vector() -> list[float]:
    """任意非零查询向量即可：这里只关心过滤有没有生效，不关心排序。"""
    return [0.3] * FakeEmbedder.dim


# ── 1. 快照不能在归档变动期间被"追认"为新版本 ─────────────────


def test_snapshot_is_not_published_when_archive_changes_mid_load(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path, monkeypatch
) -> None:
    """加载途中别的进程提交了一条新向量：这一份快照必须丢掉重读。

    旧实现是"读行 → 组矩阵 → 读代次"。中途提交的那条不会出现在这一份矩阵里，
    但代次已经是新的，于是 `_refresh_if_stale` 之后一直认为"我就是最新"。
    表现是条数、模型、维度全对，检索却永远少一条——计数校验看不见这种差异。
    """
    embedder = FakeEmbedder()
    store = _store(conn)
    first = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )
    assert first.status == "indexed"

    calls = {"n": 0}
    real = vectors_module._snapshot_of

    def fake(rows, dim):
        snap = real(rows, dim)
        calls["n"] += 1
        if calls["n"] == 1:
            # 模拟另一个进程：在"读行"与"读代次"之间写入一条向量并推进代次
            conn.execute(
                "INSERT OR REPLACE INTO vectors"
                " (chunk_id, project_name, project_code, doc_year, dim, model, vec)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "中途写入的片段",
                    "测试项目",
                    "",
                    2024,
                    FakeEmbedder.dim,
                    FakeEmbedder.model,
                    array("f", [0.5] * FakeEmbedder.dim).tobytes(),
                ),
            )
            conn.commit()
            repo.bump_index_generation(conn)
        return snap

    monkeypatch.setattr(vectors_module, "_snapshot_of", fake)

    rebuilt = NumpyStore(conn, FakeEmbedder.dim)  # 相当于新进程首次加载
    assert calls["n"] >= 2, "代次变了必须整份重读，而不是采用这一次的结果"
    assert rebuilt.count() == first.chunks + 1, (
        "重读之后必须能看到中途写入的那一条；只采用第一次的结果会永久少一条"
    )
    assert "中途写入的片段" in rebuilt.ids()


# ── 2. 内容没变但年份变了，不能被当成"跳过" ─────────────────────


def test_reingest_with_new_year_updates_document_archive_and_index(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """同一份文件改年份：文档行、归档、索引的过滤列必须一起变。

    曾经这种情况直接返回 `skipped`——用户改了年份、界面显示"已存在"，
    按新年份过滤却查不到任何片段，而且没有任何报错提示这件事。
    """
    embedder = FakeEmbedder()
    store = _store(conn)
    first = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2024
    )
    assert first.status == "indexed"

    second = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="测试项目", doc_year=2025
    )
    assert second.status == "indexed"
    assert second.note and "2025" in second.note, "要说清这次只改了年份，不是重新入库"

    doc = repo.get_document(conn, "测试项目", docx_file.name)
    assert doc is not None and doc.doc_year == 2025

    ids = repo.chunk_ids_of_doc(conn, doc.id)
    archived = repo.archived_vectors_by_ids(conn, ids)
    assert len(archived) == len(ids)
    assert {r["doc_year"] for r in archived} == {2025}

    assert store.search(_query_vector(), limit=20, filt=Filter(year_from=2025)), (
        "按新年份必须能筛到，否则说明索引里的过滤列没跟上"
    )
    assert not store.search(_query_vector(), limit=20, filt=Filter(year_to=2024)), (
        "按旧年份不该再筛到这些片段"
    )


# ── 3. 事实事务失败时，被覆盖的旧行要还原 ────────────────────────


def _docx_with_shared_section(path: Path, second: str) -> Path:
    """两版共享"第1章"，第二版另外多一节。共享那几段的 chunk_id 会因为内容
    寻址而保持不变——这正是"被 upsert 覆盖"的那批行。"""
    doc = Document()
    doc.add_heading("第1章 研究背景", level=1)
    doc.add_paragraph("这一段两版完全相同，因此 chunk_id 不变，会被第二次写入整行覆盖。" * 6)
    doc.add_heading("第2章 技术路线", level=1)
    doc.add_paragraph(second * 6)
    doc.save(path)
    return path


def test_failed_fact_commit_restores_overwritten_vectors(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, monkeypatch
) -> None:
    """事实事务失败后，新旧版本共用的 chunk_id 必须还是上一版的过滤元数据。

    旧实现只删"本次新增"的 id。共用 id 上的 `doc_year` 早被 upsert 改成新值，
    于是正文是旧版、年份是新版：条数一条不差，`year_from` 过滤却错了一年。
    """
    embedder = FakeEmbedder()
    store = _store(conn)

    path = _docx_with_shared_section(tmp_path / "方案.docx", "旧版的技术路线")
    first = ingest_file(
        conn, store, embedder, settings, path=path, project_name="测试项目", doc_year=2024
    )
    assert first.status == "indexed"

    doc = repo.get_document(conn, "测试项目", path.name)
    assert doc is not None
    old_ids = set(repo.chunk_ids_of_doc(conn, doc.id))
    assert len(old_ids) >= 2

    # 第二版：共享章节原样保留，另一节换内容（chunk_id 因此一变一不变）
    _docx_with_shared_section(path, "新版的技术路线，正文已经改过")

    real_commit = repo.commit_document_version

    def boom(*args, **kwargs):
        raise RuntimeError("模拟事实事务提交失败")

    monkeypatch.setattr(repo, "commit_document_version", boom)
    try:
        outcome = ingest_file(
            conn, store, embedder, settings, path=path, project_name="测试项目", doc_year=2025
        )
    finally:
        monkeypatch.setattr(repo, "commit_document_version", real_commit)

    assert outcome.status == "failed"
    assert "还原" in (outcome.error or ""), "失败说明要交代旧向量有没有被还原"

    # 事实没变、索引也必须回到上一版：新增的 id 被删掉，共用的行被写回旧值
    assert set(repo.chunk_ids_of_doc(conn, doc.id)) == old_ids
    assert store.ids() == old_ids, "新增的向量必须撤回，旧向量一条都不能少"
    restored = repo.archived_vectors_by_ids(conn, sorted(old_ids))
    assert {r["doc_year"] for r in restored} == {2024}
    assert not store.search(_query_vector(), limit=20, filt=Filter(year_from=2025)), (
        "索引里的过滤列不能停在新版本上"
    )
    assert store.search(_query_vector(), limit=20, filt=Filter(year_to=2024))


# ── 4. 装配之后后端被换掉，维护动作必须拒绝 ─────────────────────


def test_stale_backend_binding_is_rejected(conn: sqlite3.Connection) -> None:
    """装配发生在拿锁之前：库里记着 Milvus、手里还是 numpy，就必须拒绝。

    继续写下去的后果是"归档更新了、在线服务查的 Milvus 没更新"，
    而两边条数一致时连交叉校验都发现不了。
    """

    class _FakeNumpy:
        name = "numpy"

    class _FakeMilvus:
        name = "milvus"

    repo.set_vector_backend(conn, "milvus")
    with pytest.raises(consistency.IndexMismatch, match="milvus"):
        consistency.ensure_backend_current(conn, _FakeNumpy())

    # 对得上就放行，而且不改变任何状态
    consistency.ensure_backend_current(conn, _FakeMilvus())
    assert repo.get_vector_backend(conn) == "milvus"


def test_backend_check_is_skipped_when_nothing_recorded(conn: sqlite3.Connection) -> None:
    """还没确认过后端的库（首次部署 / 一直用 numpy）不该被这道检查挡住。"""

    class _Fake:
        name = "numpy"

    assert repo.get_vector_backend(conn) is None
    consistency.ensure_backend_current(conn, _Fake())  # 不抛即通过
