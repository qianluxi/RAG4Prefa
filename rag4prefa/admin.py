"""管理用例：项目/文档的查询与删除，全部以 SQLite 为准计数。"""

from __future__ import annotations

import json
import sqlite3

from . import repo
from .adapters.llm import LLM
from .adapters.vectors import VectorStore
from .concurrency import maintenance_lock
from .domain import DocumentInfo, Hit, ProjectInfo
from .prompts import SYSTEM_SUMMARY, build_summary_prompt


def list_projects(conn: sqlite3.Connection) -> list[ProjectInfo]:
    return repo.list_projects(conn)


def list_documents(conn: sqlite3.Connection, project_name: str) -> list[DocumentInfo]:
    return repo.list_documents(conn, project_name)


def chapter_distribution(conn: sqlite3.Connection, project_name: str) -> list[dict]:
    buckets: dict[str, dict] = {}
    for row in repo.project_chapter_rows(conn, project_name):
        path = json.loads(row["chapter_path"])
        chapter = path[0] if path else "未分章"
        bucket = buckets.setdefault(chapter, {"chapter": chapter, "chunks": 0, "tokens": 0})
        bucket["chunks"] += 1
        bucket["tokens"] += row["token_count"]
    return sorted(buckets.values(), key=lambda b: b["tokens"], reverse=True)


def sample_hits(conn: sqlite3.Connection, project_name: str, limit: int = 5) -> list[Hit]:
    """取该项目的代表性片段（按长度取，不调用 embedding，避免摘要也要联网）。"""
    ids = repo.longest_chunk_ids(conn, project_name, limit)
    details = repo.get_chunks(conn, ids)
    return [Hit(score=0.0, **details[i]) for i in ids if i in details]


def summarize_project(
    conn: sqlite3.Connection, llm: LLM, project_name: str, *, refresh: bool = False
) -> str:
    if not refresh:
        cached = repo.get_project_summary(conn, project_name)
        if cached:
            return cached
    hits = sample_hits(conn, project_name)
    if not hits:
        return ""
    text = llm.complete(
        SYSTEM_SUMMARY, build_summary_prompt(project_name, hits), max_tokens=300
    ).strip()
    if text:
        repo.set_project_summary(conn, project_name, text)
    return text


def delete_project(conn: sqlite3.Connection, store: VectorStore, name: str) -> dict:
    """删项目：事实与索引一起删，所以和入库/重建是同一类动作，必须互斥。

    不互斥的话，一次重建能在"片段已删、向量还在"的中间态读快照，算完再写回去；
    条数可能仍然对得上，id 却对不上。
    """
    with maintenance_lock(conn, purpose=f"删除项目 {name}", timeout=0.0):
        if not repo.get_project(conn, name):
            return {"deleted": False, "reason": "项目不存在"}
        chunk_ids = repo.delete_project(conn, name)
        store.delete(chunk_ids)
        return {"deleted": True, "chunks": len(chunk_ids)}
