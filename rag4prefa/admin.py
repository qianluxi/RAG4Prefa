"""管理用例：项目/文档的查询与删除，全部以 SQLite 为准计数。"""

from __future__ import annotations

import json
import sqlite3

from . import repo
from .adapters.llm import LLM
from .adapters.vectors import VectorStore
from .domain import DocumentInfo, Hit, ProjectInfo
from .prompts import SYSTEM_SUMMARY, build_summary_prompt


def list_projects(conn: sqlite3.Connection) -> list[ProjectInfo]:
    return repo.list_projects(conn)


def list_documents(conn: sqlite3.Connection, project_name: str) -> list[DocumentInfo]:
    return repo.list_documents(conn, project_name)


def chapter_distribution(conn: sqlite3.Connection, project_name: str) -> list[dict]:
    rows = conn.execute(
        """
        SELECT c.chapter_path, c.token_count
        FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE d.project_name = ?
        """,
        (project_name,),
    ).fetchall()
    buckets: dict[str, dict] = {}
    for row in rows:
        path = json.loads(row["chapter_path"])
        chapter = path[0] if path else "未分章"
        bucket = buckets.setdefault(chapter, {"chapter": chapter, "chunks": 0, "tokens": 0})
        bucket["chunks"] += 1
        bucket["tokens"] += row["token_count"]
    return sorted(buckets.values(), key=lambda b: b["tokens"], reverse=True)


def sample_hits(conn: sqlite3.Connection, project_name: str, limit: int = 5) -> list[Hit]:
    """取该项目的代表性片段（按序取，不调用 embedding，避免摘要也要联网）。"""
    rows = conn.execute(
        """
        SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
        WHERE d.project_name = ? ORDER BY c.token_count DESC LIMIT ?
        """,
        (project_name, limit),
    ).fetchall()
    ids = [r["id"] for r in rows]
    details = repo.get_chunks(conn, ids)
    return [Hit(score=0.0, **details[i]) for i in ids if i in details]


def summarize_project(
    conn: sqlite3.Connection, llm: LLM, project_name: str, *, refresh: bool = False
) -> str:
    if not refresh:
        row = conn.execute(
            "SELECT summary FROM projects WHERE name = ?", (project_name,)
        ).fetchone()
        if row and row["summary"]:
            return row["summary"]
    hits = sample_hits(conn, project_name)
    if not hits:
        return ""
    text = llm.complete(
        SYSTEM_SUMMARY, build_summary_prompt(project_name, hits), max_tokens=300
    ).strip()
    if text:
        conn.execute("UPDATE projects SET summary = ? WHERE name = ?", (text, project_name))
        conn.commit()
    return text


def delete_project(conn: sqlite3.Connection, store: VectorStore, name: str) -> dict:
    if not repo.get_project(conn, name):
        return {"deleted": False, "reason": "项目不存在"}
    chunk_ids = repo.delete_project(conn, name)
    store.delete(chunk_ids)
    return {"deleted": True, "chunks": len(chunk_ids)}
