"""入库用例：解析 → 分块 → 向量化 → 写库存索引，全过程幂等。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import repo
from .adapters.embed import Embedder
from .adapters.parsers.docx import parse_docx
from .adapters.vectors import (
    VectorItem,
    VectorStore,
    from_archive_row,
    to_archive_row,
)
from .chunking import build_chunks, contextual_text
from .config import Settings
from .domain import Chunk, JobState
from .tokens import estimate_tokens


@dataclass
class FileOutcome:
    filename: str
    status: str  # indexed / skipped / failed
    sections: int = 0
    chunks: int = 0
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "status": self.status,
            "sections": self.sections,
            "chunks": self.chunks,
            "error": self.error,
        }


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _doc_id(project_name: str, filename: str) -> str:
    return hashlib.sha1(f"{project_name}|{filename}".encode()).hexdigest()[:16]


def ingest_file(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    path: Path,
    project_name: str,
    project_code: str = "",
    doc_year: int | None = None,
    force: bool = False,
) -> FileOutcome:
    filename = path.name
    doc_id = _doc_id(project_name, filename)
    try:
        repo.upsert_project(conn, project_name, project_code)
        content_sha = _sha_file(path)
        existing = repo.get_document(conn, project_name, filename)
        if existing and existing.content_sha == content_sha and not force:
            return FileOutcome(filename, "skipped", existing.section_count, existing.chunk_count)

        sections = parse_docx(path)
        chunks = build_chunks(
            sections,
            doc_id,
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
        )
        if not chunks:
            repo.upsert_document(
                conn,
                doc_id=doc_id,
                project_name=project_name,
                filename=filename,
                content_sha=content_sha,
                doc_year=doc_year,
                status="empty",
                error="未能从文档中提取到正文",
            )
            return FileOutcome(filename, "failed", len(sections), 0, "文档没有可用正文")

        # 重新入库：先清旧片段，避免残留孤儿向量
        stale = repo.delete_chunks_of_doc(conn, doc_id)
        if stale:
            store.delete(stale)

        repo.upsert_document(
            conn,
            doc_id=doc_id,
            project_name=project_name,
            filename=filename,
            content_sha=content_sha,
            doc_year=doc_year,
            section_count=len(sections),
            chunk_count=len(chunks),
            status="indexing",
        )

        vectors = embedder.embed([contextual_text(c) for c in chunks])
        items = [
            VectorItem(
                chunk_id=c.id,
                vector=v,
                project_name=project_name,
                project_code=project_code,
                doc_year=doc_year,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]
        repo.archive_vectors(conn, [to_archive_row(i, embedder.model) for i in items])
        store.upsert(items)
        repo.insert_chunks(conn, chunks)
        repo.set_document_status(conn, doc_id, "indexed", None)
        return FileOutcome(filename, "indexed", len(sections), len(chunks))
    except Exception as exc:  # 单文件失败不该拖垮整批
        repo.set_document_status(conn, doc_id, "failed", str(exc))
        return FileOutcome(filename, "failed", 0, 0, f"{type(exc).__name__}: {exc}")


def ingest_batch(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    job_id: str,
    files: list[Path],
    project_name: str,
    project_code: str = "",
    doc_year: int | None = None,
) -> list[FileOutcome]:
    repo.update_job(conn, job_id, state=JobState.running, progress=0.0, message="开始解析")

    outcomes: list[FileOutcome] = []
    total = max(1, len(files))
    for i, path in enumerate(files, start=1):
        outcome = ingest_file(
            conn,
            store,
            embedder,
            settings,
            path=path,
            project_name=project_name,
            project_code=project_code,
            doc_year=doc_year,
        )
        outcomes.append(outcome)
        repo.update_job(
            conn,
            job_id,
            progress=round(i / total, 3),
            message=f"{path.name} → {outcome.status}"
            + (f"（{outcome.error}）" if outcome.error else ""),
        )

    ok = sum(1 for o in outcomes if o.status == "indexed")
    failed = [o for o in outcomes if o.status == "failed"]
    state = JobState.failed if failed and not ok else JobState.done
    summary = {"outcomes": [o.as_dict() for o in outcomes], "indexed": ok, "failed": len(failed)}
    repo.update_job(
        conn,
        job_id,
        state=state,
        progress=1.0,
        message=json.dumps(summary, ensure_ascii=False),
    )
    return outcomes


def rebuild_store(conn: sqlite3.Connection, store: VectorStore) -> int:
    """从 SQLite 归档重建向量索引（不重新 embedding，换后端/索引损坏时用）。"""
    items = [from_archive_row(row) for row in repo.iter_archived_vectors(conn)]
    store.clear()
    if items:
        store.upsert(items)
    return len(items)


def reindex(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    project_name: str | None = None,
) -> dict:
    """按 SQLite 里的事实重建向量索引（换 embedding 模型后必做）。"""
    where, params = "", []
    if project_name:
        where = "WHERE d.project_name = ?"
        params.append(project_name)
    rows = conn.execute(
        f"""
        SELECT c.id, c.text, c.title, c.chapter_path, d.project_name,
               COALESCE(p.code, '') AS project_code, d.doc_year
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        JOIN projects  p ON p.name = d.project_name
        {where}
        ORDER BY c.doc_id, c.ordinal
        """,
        params,
    ).fetchall()
    if not rows:
        return {"chunks": 0}

    store.clear()
    batch_size = max(1, settings.embedding_batch)
    done = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        texts = []
        for r in batch:
            path = " > ".join(json.loads(r["chapter_path"]))
            texts.append(f"{path}\n{r['text']}" if path else r["text"])
        vectors = embedder.embed(texts)
        store.upsert(
            [
                VectorItem(
                    chunk_id=r["id"],
                    vector=v,
                    project_name=r["project_name"],
                    project_code=r["project_code"],
                    doc_year=r["doc_year"],
                )
                for r, v in zip(batch, vectors, strict=True)
            ]
        )
        done += len(batch)
    return {"chunks": done}


def doc_year_from(value: str | None, fallback: int | None = None) -> int | None:
    if not value:
        return fallback
    for token in (value[:4], value[:4][::-1]):
        if token.isdigit() and 1900 <= int(token) <= 2200:
            return int(token)
    return fallback


def chunk_tokens(chunks: list[Chunk]) -> int:
    return sum(estimate_tokens(c.text) for c in chunks)
