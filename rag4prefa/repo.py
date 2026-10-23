"""数据访问层：所有 SQL 集中在此。上层只处理领域对象，不写 SQL。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence

from .db import now_iso, transaction
from .domain import Chunk, DocumentInfo, Filter, Job, JobState, ProjectInfo


def _fts_tokens(text: str) -> str:
    """jieba 预分词：索引与查询必须用同一套分词，否则中文召回必然错位。"""
    import jieba

    tokens = [t.strip() for t in jieba.lcut_for_search(text)]
    return " ".join(t for t in tokens if t and not t.isspace())


# ── projects ────────────────────────────────────────────────


def upsert_project(
    conn: sqlite3.Connection, name: str, code: str = "", started_on: str | None = None
) -> None:
    ts = now_iso()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO projects (name, code, started_on, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                code = COALESCE(NULLIF(excluded.code, ''), projects.code),
                started_on = COALESCE(excluded.started_on, projects.started_on),
                updated_at = excluded.updated_at
            """,
            (name, code, started_on, ts, ts),
        )


def list_projects(conn: sqlite3.Connection) -> list[ProjectInfo]:
    rows = conn.execute(
        """
        SELECT p.name, p.code, p.started_on,
               COUNT(DISTINCT d.id)          AS document_count,
               COALESCE(SUM(d.chunk_count), 0) AS chunk_count,
               MAX(d.uploaded_at)            AS last_updated
        FROM projects p
        LEFT JOIN documents d ON d.project_name = p.name
        GROUP BY p.name
        ORDER BY p.name
        """
    ).fetchall()
    return [
        ProjectInfo(
            name=r["name"],
            code=r["code"] or "",
            started_on=r["started_on"],
            document_count=r["document_count"],
            chunk_count=r["chunk_count"],
            last_updated=r["last_updated"],
        )
        for r in rows
    ]


def get_project(conn: sqlite3.Connection, name: str) -> ProjectInfo | None:
    return next((p for p in list_projects(conn) if p.name == name), None)


def project_names(conn: sqlite3.Connection) -> list[str]:
    return [r["name"] for r in conn.execute("SELECT name FROM projects ORDER BY name")]


def delete_project(conn: sqlite3.Connection, name: str) -> list[str]:
    """级联删除并返回被删掉的 chunk_id（供向量库同步清理）。"""
    chunk_ids = [
        r["id"]
        for r in conn.execute(
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id "
            "WHERE d.project_name = ?",
            (name,),
        )
    ]
    with transaction(conn):
        if chunk_ids:
            marks = ",".join("?" * len(chunk_ids))
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({marks})", chunk_ids)
            conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({marks})", chunk_ids)
        conn.execute("DELETE FROM projects WHERE name = ?", (name,))
    return chunk_ids


# ── documents ───────────────────────────────────────────────


def get_document(conn: sqlite3.Connection, project_name: str, filename: str) -> DocumentInfo | None:
    row = conn.execute(
        "SELECT * FROM documents WHERE project_name = ? AND filename = ?",
        (project_name, filename),
    ).fetchone()
    return _document(row) if row else None


def get_document_by_id(conn: sqlite3.Connection, doc_id: str) -> DocumentInfo | None:
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    return _document(row) if row else None


def list_documents(conn: sqlite3.Connection, project_name: str) -> list[DocumentInfo]:
    rows = conn.execute(
        "SELECT * FROM documents WHERE project_name = ? ORDER BY uploaded_at DESC",
        (project_name,),
    ).fetchall()
    return [_document(r) for r in rows]


def _document(row: sqlite3.Row) -> DocumentInfo:
    return DocumentInfo(
        id=row["id"],
        project_name=row["project_name"],
        filename=row["filename"],
        content_sha=row["content_sha"],
        section_count=row["section_count"],
        chunk_count=row["chunk_count"],
        status=row["status"],
        uploaded_at=row["uploaded_at"],
        error=row["error"],
    )


def upsert_document(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    project_name: str,
    filename: str,
    content_sha: str,
    doc_year: int | None,
    section_count: int = 0,
    chunk_count: int = 0,
    status: str = "indexed",
    error: str | None = None,
) -> None:
    ts = now_iso()
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO documents (id, project_name, filename, content_sha, doc_year,
                                   section_count, chunk_count, status, error, uploaded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_name, filename) DO UPDATE SET
                id = excluded.id,
                content_sha = excluded.content_sha,
                doc_year = excluded.doc_year,
                section_count = excluded.section_count,
                chunk_count = excluded.chunk_count,
                status = excluded.status,
                error = excluded.error,
                uploaded_at = excluded.uploaded_at
            """,
            (
                doc_id,
                project_name,
                filename,
                content_sha,
                doc_year,
                section_count,
                chunk_count,
                status,
                error,
                ts,
            ),
        )


def set_document_status(
    conn: sqlite3.Connection, doc_id: str, status: str, error: str | None
) -> None:
    with transaction(conn):
        conn.execute(
            "UPDATE documents SET status = ?, error = ? WHERE id = ?", (status, error, doc_id)
        )


def delete_chunks_of_doc(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
    with transaction(conn):
        conn.execute(
            "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({marks})", ids)
    return ids


# ── chunks ──────────────────────────────────────────────────


def insert_chunks(conn: sqlite3.Connection, chunks: Sequence[Chunk]) -> None:
    with transaction(conn):
        for c in chunks:
            conn.execute(
                """
                INSERT INTO chunks (id, doc_id, ordinal, chapter_path, title, text,
                                    token_count, content_sha)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    ordinal = excluded.ordinal, chapter_path = excluded.chapter_path,
                    title = excluded.title, text = excluded.text,
                    token_count = excluded.token_count, content_sha = excluded.content_sha
                """,
                (
                    c.id,
                    c.doc_id,
                    c.ordinal,
                    json.dumps(c.chapter_path, ensure_ascii=False),
                    c.title,
                    c.text,
                    c.token_count,
                    c.content_sha,
                ),
            )
            conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (c.id,))
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, title, body) VALUES (?, ?, ?)",
                (c.id, _fts_tokens(f"{c.title} {' '.join(c.chapter_path)}"), _fts_tokens(c.text)),
            )
        # 注：vectors 表由向量后端自己维护，这里不碰，避免两条写路径互相覆盖


def get_chunks(conn: sqlite3.Connection, chunk_ids: Iterable[str]) -> dict[str, dict]:
    ids = list(chunk_ids)
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"""
        SELECT c.id, c.title, c.text, c.chapter_path, c.doc_id,
               d.filename, d.doc_year, d.project_name, p.code AS project_code
        FROM chunks c
        JOIN documents d ON d.id = c.doc_id
        JOIN projects  p ON p.name = d.project_name
        WHERE c.id IN ({marks})
        """,
        ids,
    ).fetchall()
    return {
        r["id"]: {
            "chunk_id": r["id"],
            "title": r["title"],
            "text": r["text"],
            "chapter_path": json.loads(r["chapter_path"]),
            "filename": r["filename"],
            "doc_year": r["doc_year"],
            "project_name": r["project_name"],
            "project_code": r["project_code"],
        }
        for r in rows
    }


def fts_search(
    conn: sqlite3.Connection, query: str, *, limit: int, filt: Filter | None = None
) -> list[tuple[str, float]]:
    """BM25 稀疏召回。返回 (chunk_id, 分数)，分数越大越相关。"""
    tokens = [t for t in _fts_tokens(query).split() if t]
    if not tokens:
        return []
    expr = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)

    where = ["chunks_fts MATCH ?"]
    params: list[object] = [expr]
    if filt:
        if filt.projects:
            where.append(f"d.project_name IN ({','.join('?' * len(filt.projects))})")
            params.extend(filt.projects)
        if filt.year_from is not None:
            where.append("d.doc_year >= ?")
            params.append(filt.year_from)
        if filt.year_to is not None:
            where.append("d.doc_year <= ?")
            params.append(filt.year_to)

    sql = f"""
        SELECT chunks_fts.chunk_id AS chunk_id, bm25(chunks_fts) AS rank
        FROM chunks_fts
        JOIN chunks     c ON c.id = chunks_fts.chunk_id
        JOIN documents  d ON d.id = c.doc_id
        WHERE {" AND ".join(where)}
        ORDER BY rank
        LIMIT ?
    """
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    # bm25() 越小越相关，转成"越大越相关"
    return [(r["chunk_id"], -float(r["rank"])) for r in rows]


# ── 向量归档（与后端无关的持久记录）────────────────────────


def archive_vectors(
    conn: sqlite3.Connection,
    rows: Iterable[tuple[str, str, str, int | None, int, str, bytes]],
) -> None:
    """把向量写入 SQLite 归档表：换后端/重建索引不必重新调用 embedding。"""
    with transaction(conn):
        conn.executemany(
            "INSERT OR REPLACE INTO vectors (chunk_id, project_name, project_code, doc_year,"
            " dim, model, vec) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def iter_archived_vectors(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT chunk_id, project_name, project_code, doc_year, dim, model, vec FROM vectors"
    ).fetchall()
    return [
        {
            "chunk_id": r["chunk_id"],
            "project_name": r["project_name"],
            "project_code": r["project_code"],
            "doc_year": r["doc_year"],
            "dim": r["dim"],
            "model": r["model"],
            "vec": r["vec"],
        }
        for r in rows
    ]


def chunks_for_reindex(
    conn: sqlite3.Connection, project_name: str | None = None
) -> list[dict]:
    """重建索引的输入：片段正文 + 归属信息。project_name 为空表示全量。"""
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
    return [
        {
            "chunk_id": r["id"],
            "text": r["text"],
            "title": r["title"],
            "chapter_path": r["chapter_path"],
            "project_name": r["project_name"],
            "project_code": r["project_code"],
            "doc_year": r["doc_year"],
        }
        for r in rows
    ]


# ── meta / 索引元信息 ──────────────────────────────────────


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    with transaction(conn):
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


# ── jobs ────────────────────────────────────────────────────


def create_job(conn: sqlite3.Connection, job_id: str, kind: str, payload: dict) -> Job:
    ts = now_iso()
    with transaction(conn):
        conn.execute(
            "INSERT INTO jobs (id, kind, state, progress, message, payload, created_at)"
            " VALUES (?, ?, ?, 0, '', ?, ?)",
            (job_id, kind, JobState.queued.value, json.dumps(payload, ensure_ascii=False), ts),
        )
    return Job(id=job_id, kind=kind, state=JobState.queued, payload=payload, created_at=ts)


def update_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    state: JobState | None = None,
    progress: float | None = None,
    message: str | None = None,
) -> None:
    sets, params = [], []
    if state is not None:
        sets.append("state = ?")
        params.append(state.value)
        if state in (JobState.done, JobState.failed):
            sets.append("finished_at = ?")
            params.append(now_iso())
    if progress is not None:
        sets.append("progress = ?")
        params.append(progress)
    if message is not None:
        sets.append("message = ?")
        params.append(message)
    if not sets:
        return
    params.append(job_id)
    with transaction(conn):
        conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)


def get_job(conn: sqlite3.Connection, job_id: str) -> Job | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row:
        return None
    return Job(
        id=row["id"],
        kind=row["kind"],
        state=JobState(row["state"]),
        progress=row["progress"],
        message=row["message"],
        payload=json.loads(row["payload"]),
        created_at=row["created_at"],
        finished_at=row["finished_at"],
    )
