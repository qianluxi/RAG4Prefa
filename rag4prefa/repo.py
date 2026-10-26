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
        _bump(conn, DATA_REVISION_KEY)
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
        doc_year=row["doc_year"],
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


def mark_document_indexing(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    project_name: str,
    filename: str,
    doc_year: int | None,
) -> None:
    """占位 + 标记"正在重建"。

    存在的文档只改 status，**不动 content_sha / chunk_count**：新的摘要要等
    这一版真的写成功才记下来，否则下次重试会被误判成"内容没变"而跳过。
    """
    with transaction(conn):
        conn.execute(
            """
            INSERT INTO documents (id, project_name, filename, content_sha, doc_year,
                                   section_count, chunk_count, status, error, uploaded_at)
            VALUES (?, ?, ?, '', ?, 0, 0, 'indexing', NULL, ?)
            ON CONFLICT(project_name, filename) DO UPDATE SET
                status = 'indexing', error = NULL
            """,
            (doc_id, project_name, filename, doc_year, now_iso()),
        )


def chunk_ids_of_doc(conn: sqlite3.Connection, doc_id: str) -> list[str]:
    return [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]


def set_document_year(
    conn: sqlite3.Connection, doc_id: str, doc_year: int | None
) -> list[str]:
    """只改年份，不碰正文与向量：文档行、归档行、代次在同一个事务里。

    为什么单独一条路径：`doc_year` 是**过滤条件**，它同时存在于 documents、vectors
    归档和后端的标量列三处。只改一处，就会出现"文档说 2025、按 2025 过滤却查不到
    这批片段"——而且条数、模型、维度全都对得上，计数校验发现不了。

    返回受影响的 chunk_id：归档是事实，后端索引要由调用方按这份事实去覆盖
    （numpy 靠代次自加载，Milvus 只能整行 upsert）。
    """
    ids = chunk_ids_of_doc(conn, doc_id)
    with transaction(conn):
        conn.execute("UPDATE documents SET doc_year = ? WHERE id = ?", (doc_year, doc_id))
        if ids:
            marks = ",".join("?" * len(ids))
            conn.execute(
                f"UPDATE vectors SET doc_year = ? WHERE chunk_id IN ({marks})",
                [doc_year, *ids],
            )
        # 过滤列变了 ⇒ 别的进程的缓存必须刷新。只推"内容代次"不推"索引代次"：
        # 向量空间没变，在途查询的向量仍然有效，不该因此把一次检索判死。
        _bump(conn, DATA_REVISION_KEY)
    return ids



def replace_document_chunks(
    conn: sqlite3.Connection, doc_id: str, chunks: Sequence[Chunk]
) -> list[str]:
    """用一个事务把该文档的片段换成新的一版，返回被淘汰的 chunk_id。

    逐条 delete 再 insert 会让"删完、插一半"的窗口暴露给检索；这里整个过程
    在同一个事务里，读取方要么看到旧版、要么看到新版。
    """
    return commit_document_version(conn, doc_id, chunks, [])


def commit_document_version(
    conn: sqlite3.Connection,
    doc_id: str,
    chunks: Sequence[Chunk],
    archive_rows: Iterable[tuple[str, str, str, int | None, int, str, bytes]],
) -> list[str]:
    """片段与向量归档在**同一个事务**里切换到新版本，返回被淘汰的 chunk_id。

    为什么必须同事务：如果片段先提交、归档后写，中间崩溃就得到"正文是新版、归档是旧版"。
    如果两个 chunk_id 恰好相同（内容寻址下即内容相同，通常不是问题，但位置寻址时是常态），
    检索会拿旧向量配新正文。放在一个事务里，这种状态在结构上就不存在。

    向量索引（Milvus / numpy 后端）不在这里写——它是派生数据，由调用方先写好再切事实。
    """
    keep = {c.id for c in chunks}
    stale = [i for i in chunk_ids_of_doc(conn, doc_id) if i not in keep]
    rows = list(archive_rows)
    with transaction(conn):
        if stale:
            marks = ",".join("?" * len(stale))
            conn.execute(f"DELETE FROM chunks_fts WHERE chunk_id IN ({marks})", stale)
            conn.execute(f"DELETE FROM chunks WHERE id IN ({marks})", stale)
            # 归档里也不该留下已经没有正文的向量
            conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({marks})", stale)
        for c in chunks:
            _write_chunk(conn, c)
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (chunk_id, project_name, project_code, doc_year,"
                " dim, model, vec) VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        # 归档和片段一起变了：别的进程手里的 numpy 缓存（它就是这两张表的视图）
        # 从此刻起是旧的。这里只推"内容代次"而不是"索引代次"——向量空间没变，
        # 在途查询的结果仍然成立，不该因为它把一次检索判死。
        _bump(conn, DATA_REVISION_KEY)
    return stale


# ── chunks ──────────────────────────────────────────────────


def insert_chunks(conn: sqlite3.Connection, chunks: Sequence[Chunk]) -> None:
    with transaction(conn):
        for c in chunks:
            _write_chunk(conn, c)
        # 注：vectors 表由向量后端自己维护，这里不碰，避免两条写路径互相覆盖


def _write_chunk(conn: sqlite3.Connection, c: Chunk) -> None:
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
    """归档向量，**只取仍有正文的那些**。

    孤儿向量（chunk 已被删除）没有正文，检索命中它也只能跳过，
    留着只会让重建后的索引比事实表更大、并且把旧维度带回来。
    """
    rows = conn.execute(
        """
        SELECT v.chunk_id, v.project_name, v.project_code, v.doc_year, v.dim, v.model, v.vec
        FROM vectors v JOIN chunks c ON c.id = v.chunk_id
        ORDER BY v.chunk_id
        """
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


def archived_vectors_by_ids(
    conn: sqlite3.Connection, chunk_ids: Sequence[str]
) -> list[dict]:
    """按 id 取归档行（含 vec 字节），用于"写坏了要还原成上一版"这类恢复动作。

    与 `iter_archived_vectors` 的区别：这个不做 JOIN、也不排序——它要的是
    "这批 id 当时是什么样"，id 可能已经在事实表里不存在了（正是要还原的那种情况）。
    """
    ids = list(chunk_ids)
    if not ids:
        return []
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        "SELECT chunk_id, project_name, project_code, doc_year, dim, model, vec"
        f" FROM vectors WHERE chunk_id IN ({marks})",
        ids,
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


def delete_orphan_vectors(conn: sqlite3.Connection) -> int:
    """清掉没有对应片段的归档行；返回删除条数。"""
    with transaction(conn):
        cur = conn.execute("DELETE FROM vectors WHERE chunk_id NOT IN (SELECT id FROM chunks)")
    return int(cur.rowcount or 0)


def vector_stats(conn: sqlite3.Connection) -> dict:
    """归档里实际存在的模型与维度分布——判断"有没有混用两套向量"的唯一依据。"""
    models = {
        (r["model"] or ""): int(r["c"])
        for r in conn.execute("SELECT model, COUNT(*) AS c FROM vectors GROUP BY model")
    }
    dims = {
        int(r["dim"]): int(r["c"])
        for r in conn.execute("SELECT dim, COUNT(*) AS c FROM vectors GROUP BY dim")
    }
    return {"models": models, "dims": dims, "count": sum(models.values())}


def missing_vector_count(conn: sqlite3.Connection) -> int:
    """有正文、但没有归档向量的片段数。

    比"chunks 总数 vs vectors 总数"更可诊断：总数相等也可能是"A 缺了、B 多了"，
    而 A 缺了正是最危险的一种——稠密召回静默少一路，只靠 BM25 还能返回结果。
    """
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM chunks c"
        " LEFT JOIN vectors v ON v.chunk_id = c.id WHERE v.chunk_id IS NULL"
    ).fetchone()
    return int(row["c"])


def orphan_vector_count(conn: sqlite3.Connection) -> int:
    """有归档向量、但没有对应正文的条数（片段已删、向量没跟上）。"""
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM vectors v"
        " LEFT JOIN chunks c ON c.id = v.chunk_id WHERE c.id IS NULL"
    ).fetchone()
    return int(row["c"])


def unlabeled_vector_count(conn: sqlite3.Connection) -> int:
    """模型名为空的历史归档行。

    不知道向量是谁算的，就无法证明它和当前模型同一套空间；把它当"一致"是自欺。
    """
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM vectors WHERE model IS NULL OR model = ''"
    ).fetchone()
    return int(row["c"])


def malformed_vector_count(conn: sqlite3.Connection) -> int:
    """声明维度与 float32 字节数对不上的归档行数。

    这是唯一一种"条数、模型、维度三项全对得上、却组不出索引"的损坏：`dim` 写 1024，
    `vec` 里只有 768 个浮点。SQLite 的 `length()` 对 BLOB 返回字节数，正好用来核对
    `dim * 4`——只看 `dim` 字段是查不出来的。
    """
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM vectors WHERE dim <= 0 OR LENGTH(vec) != dim * 4"
    ).fetchone()
    return int(row["c"])


def all_chunk_ids(conn: sqlite3.Connection) -> set[str]:
    return {r["id"] for r in conn.execute("SELECT id FROM chunks")}


def archived_chunk_ids(conn: sqlite3.Connection) -> set[str]:
    return {r["chunk_id"] for r in conn.execute("SELECT chunk_id FROM vectors")}


def recoverable_chunk_ids(conn: sqlite3.Connection) -> set[str]:
    """能作为"恢复源"的归档 id：仍然有正文的那些。

    与 `iter_archived_vectors` 同一套 JOIN，但**不取 vec 列**——确认活动后端时要比的是
    id 集合，把几千条 1024 维向量捞进内存只为丢进集合，纯属浪费。

    不直接用 `archived_chunk_ids`：孤儿行（正文已删、向量还在）不该参与"这个后端是不是
    当前版本"的判断——它们本来就要由 rebuild 清掉，拿它们去要求索引必须一致，会让每次
    启动都判定"不一致"然后重建一遍。
    """
    return {
        r["chunk_id"]
        for r in conn.execute("SELECT v.chunk_id FROM vectors v JOIN chunks c ON c.id = v.chunk_id")
    }


def select_chunk_ids(
    conn: sqlite3.Connection, ids: Iterable[str], *, filt: Filter | None = None
) -> set[str]:
    """从给定 id 里筛出**存在且满足筛选条件**的那些。

    两件事必须一起做：不存在的 id 是标注写错了（当成有效标注会让 Recall 永远差一截），
    不满足筛选条件的 id 是标注越界了（那条片段本题根本召回不到）。只筛存在的、
    不筛范围的，会让 gold 里混进"不可能命中"的项，把分数压低却看不出原因。
    """
    wanted = list(dict.fromkeys(ids))
    found: set[str] = set()
    step = 400  # SQLite 参数上限是 999，这里按 400 一批，给筛选参数留余量
    for start in range(0, len(wanted), step):
        batch = wanted[start : start + step]
        marks = ",".join("?" * len(batch))
        sql = (
            "SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id"
            f" WHERE c.id IN ({marks})"
        )
        params: list[object] = list(batch)
        if filt:
            if filt.projects:
                sql += f" AND d.project_name IN ({','.join('?' * len(filt.projects))})"
                params.extend(filt.projects)
            if filt.year_from is not None:
                sql += " AND d.doc_year >= ?"
                params.append(filt.year_from)
            if filt.year_to is not None:
                sql += " AND d.doc_year <= ?"
                params.append(filt.year_to)
        found.update(r["id"] for r in conn.execute(sql, params))
    return found


def chunk_ids_containing(
    conn: sqlite3.Connection,
    snippet: str,
    *,
    filt: Filter | None = None,
    limit: int | None = None,
) -> list[str]:
    """找出正文包含某段原文的片段——评估集用片段标注时，靠它换算成 chunk_id。

    `filt` 必须传该题实际生效的筛选条件：题面只允许检索甲项目时，别的项目里出现
    同一段文字并不算"本题的相关片段"，把它们算进 gold 会让 Recall 永远达不到 1。

    `limit` 默认不截断。截断会让 gold 悄悄变小、指标看起来更好看，
    这比慢一点危险得多；确实要限制时由调用方显式给值并自己报告。
    """
    escaped = snippet.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    where = ["c.text LIKE ? ESCAPE '\\'"]
    params: list[object] = [f"%{escaped}%"]
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
    sql = (
        "SELECT DISTINCT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id"
        f" WHERE {' AND '.join(where)} ORDER BY c.id"
    )
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [r["id"] for r in rows]


def chunk_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"])


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


# 活动向量后端。**落库而不是只读配置**，因为同一份库会被 uvicorn 和 CLI 同时打开：
# 两边各按自己的配置（或各自回退的结果）选后端，就会一边写 numpy 索引、一边查 Milvus，
# 归档是共享的、派生索引却成了两份。条数恰好相同时，连"索引 vs 归档"的计数校验都
# 发现不了差异（2026-10-24 第十四轮复审）。后端是**这个库**的属性，不是每个进程的偏好。
#
# 只记录被确认可用的后端（目前只有 milvus 会写）：它的作用是让 `auto` 在"库里已经
# 启用 Milvus"之后不再静默回退 numpy——回退本身不报错，但会让本进程的写入只更新
# 一半（归档 + numpy 索引），在线服务的 Milvus 索引停在旧版。
VECTOR_BACKEND_KEY = "vector_backend"


def get_vector_backend(conn: sqlite3.Connection) -> str | None:
    """上一次确认可用的向量后端。None 表示还没有任何后端被确认过。"""
    return get_meta(conn, VECTOR_BACKEND_KEY)


def set_vector_backend(conn: sqlite3.Connection, name: str) -> None:
    set_meta(conn, VECTOR_BACKEND_KEY, name)


# 索引切换标记。SQLite 归档与向量库不在同一个事务里，跨组件的"两阶段提交"没法靠
# BEGIN/COMMIT 表达，只能显式记一条状态：写归档之前置位，全部切换成功后才恢复 ready。
# 只要这个值不是 ready，归档与向量索引就可能分属两个版本——此时**检索必须拒绝**，
# 因为两边条数、模型名、维度都可能完全一致，逐项校验根本发现不了差异。
INDEX_STATE_KEY = "index_state"
INDEX_READY = "ready"


def get_index_state(conn: sqlite3.Connection) -> str:
    """读索引切换状态。历史库没有这条记录，按 ready 处理（等价于旧行为）。"""
    return get_meta(conn, INDEX_STATE_KEY) or INDEX_READY


def set_index_state(conn: sqlite3.Connection, state: str) -> None:
    with transaction(conn):
        _write_index_state(conn, state)
        _bump_generation(conn)


# ── 索引代次 ────────────────────────────────────────────────
#
# 归档与向量索引每被改一次，代次 +1。它解决的是**进程之间**的那类不一致：
# 维护锁能挡住两个进程同时写，但挡不住"CLI 重建完了，API 进程里 numpy 的内存矩阵
# 还是旧的"——锁释放之后，API 手里那一份既没有报错、也没有任何标记说明它过期了。
# 代次是个单调计数器，只增不减：读方记住自己加载时的值，发现变了就重新加载，
# 或者拒绝用这份缓存做检索。
#
# 为什么不能用"条数"代替：条数相同、id 不同是这类事故的常态（改一版文档换了几个
# chunk、重建换了模型但片段数没变），条数完全看不出来。
#
# 为什么分成两个计数器：**换索引**和**换内容**对读方的影响不同。
# - `index_generation` 变了 ⇒ 向量空间可能变了（换了模型/换了后端）。在途查询的
#   查询向量可能属于旧模型，必须作废重来，不能"接着查"；
# - `data_revision` 变了 ⇒ 只是多了/少了内容。缓存要刷新，但在途查询的结果依然成立
#   （查询向量还是同一个模型算的），不该因此把一次正常的检索判死——否则"有人上传
#   文档时别人就问不了问题"，比问题本身更烦人。
INDEX_GENERATION_KEY = "index_generation"
DATA_REVISION_KEY = "data_revision"


def _counter(conn: sqlite3.Connection, key: str) -> int:
    raw = get_meta(conn, key)
    return int(raw) if raw and raw.isdigit() else 0


def get_index_generation(conn: sqlite3.Connection) -> int:
    """向量空间那一代。只有换索引（模型、后端、整体重建）才会推它。"""
    return _counter(conn, INDEX_GENERATION_KEY)


def get_data_revision(conn: sqlite3.Connection) -> int:
    return _counter(conn, DATA_REVISION_KEY)


def get_store_revision(conn: sqlite3.Connection) -> tuple[int, int]:
    """内存型后端（numpy）判断"我这份缓存过没过期"用的一对数。"""
    return get_index_generation(conn), get_data_revision(conn)


def _bump(conn: sqlite3.Connection, key: str) -> int:
    """在调用方**已有**的事务里把计数器 +1，返回新值。

    刻意不自己开事务：所有调用点都在 `with transaction(conn)` 里面，嵌套开事务会在
    内层退出时提前 `commit()`，把外层还没写完的东西一起提交掉——那正是写锁要防的事。
    """
    value = _counter(conn, key) + 1
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
    )
    return value


def _bump_generation(conn: sqlite3.Connection) -> int:
    return _bump(conn, INDEX_GENERATION_KEY)


def bump_index_generation(conn: sqlite3.Connection) -> int:
    """外部主动作废别人的缓存（例如测试模拟"另一个进程换了索引"）。"""
    with transaction(conn):
        return _bump_generation(conn)


def _write_index_state(conn: sqlite3.Connection, state: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (INDEX_STATE_KEY, state)
    )


def set_index_meta(
    conn: sqlite3.Connection,
    *,
    model: str,
    dim: str,
    state: str = INDEX_READY,
    backend: str | None = None,
) -> None:
    """模型、维度、切换状态（可含活动后端）一次提交。

    分开写会留下"模型已更新、状态还是 rebuilding"的中间态，而读方看到模型对得上
    就会放行检索——那正是这个标记要消除的窗口，所以三者必须在同一个事务里落地。
    代次一起 +1：这是在换索引，任何读过旧索引的人都必须知道。

    `backend` 也走这个事务：换后端与"新索引已经就绪"必须同时成立。分两次写的话，
    中间那一刻库里记着新后端、`index_state` 还是旧的未完成标记（或反过来），
    两边都读到一半就各自得出相反结论（2026-10-24 第十五轮复审）。
    """
    with transaction(conn):
        rows = [
            ("embedding_model", model),
            ("embedding_dim", dim),
            (INDEX_STATE_KEY, state),
        ]
        if backend:
            rows.append((VECTOR_BACKEND_KEY, backend))
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", rows
        )
        _bump_generation(conn)


# ── 管理视图（给 admin 用，避免上层写 SQL）─────────────────


def project_chapter_rows(conn: sqlite3.Connection, project_name: str) -> list[dict]:
    return [
        {"chapter_path": r["chapter_path"], "token_count": r["token_count"]}
        for r in conn.execute(
            """
            SELECT c.chapter_path, c.token_count
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE d.project_name = ?
            """,
            (project_name,),
        )
    ]


def longest_chunk_ids(conn: sqlite3.Connection, project_name: str, limit: int) -> list[str]:
    return [
        r["id"]
        for r in conn.execute(
            """
            SELECT c.id FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE d.project_name = ? ORDER BY c.token_count DESC LIMIT ?
            """,
            (project_name, limit),
        )
    ]


def get_project_summary(conn: sqlite3.Connection, project_name: str) -> str | None:
    row = conn.execute("SELECT summary FROM projects WHERE name = ?", (project_name,)).fetchone()
    return (row["summary"] or None) if row else None


def set_project_summary(conn: sqlite3.Connection, project_name: str, summary: str) -> None:
    with transaction(conn):
        conn.execute("UPDATE projects SET summary = ? WHERE name = ?", (summary, project_name))


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


def mark_interrupted_jobs(conn: sqlite3.Connection) -> int:
    """进程重启后，把上次遗留的 running/queued 任务标记为失败。

    入库任务跑在进程内，进程一死它们永远不会前进；不处理就会永远显示"运行中"，
    让人无法判断到底是排队还是已经死了。
    """
    with transaction(conn):
        cur = conn.execute(
            "UPDATE jobs SET state = ?, finished_at = ?,"
            " message = CASE WHEN message = '' THEN ? ELSE message || ' / ' || ? END"
            " WHERE state IN (?, ?)",
            (
                JobState.failed.value,
                now_iso(),
                "进程重启，任务中断",
                "进程重启，任务中断",
                JobState.running.value,
                JobState.queued.value,
            ),
        )
    return int(cur.rowcount or 0)


def mark_interrupted_documents(conn: sqlite3.Connection) -> int:
    """进程被杀时正处在 `indexing` 的文档，重启后必须有个结论。

    这类文档的片段可能已经切换、也可能没有——没有结论时它会永远显示"正在入库"，
    而 health 里的片段/向量计数又可能正好相等（因为事务是原子的），于是看起来一切正常。
    统一标成 failed，让人知道"这一版没走完，重跑一次即可"；重跑不会被 content_sha 判成跳过。
    """
    with transaction(conn):
        cur = conn.execute(
            "UPDATE documents SET status = 'failed', error = ? WHERE status = 'indexing'",
            ("上次进程在入库过程中退出，这一版没有走完；重新导入该文件即可",),
        )
    return int(cur.rowcount or 0)


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
