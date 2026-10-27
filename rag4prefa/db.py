"""SQLite：系统唯一事实来源（projects / documents / chunks / jobs / vectors）。

向量库只是可重建的派生索引。这样"统计为 0""元数据对不上"这类问题在结构上不可能发生。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .concurrency import sqlite_write

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS projects (
    name        TEXT PRIMARY KEY,
    code        TEXT NOT NULL DEFAULT '',
    started_on  TEXT,
    summary     TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    project_name  TEXT NOT NULL REFERENCES projects(name) ON DELETE CASCADE,
    filename      TEXT NOT NULL,
    content_sha   TEXT NOT NULL,
    doc_year      INTEGER,
    section_count INTEGER NOT NULL DEFAULT 0,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,
    error         TEXT,
    uploaded_at   TEXT NOT NULL,
    UNIQUE (project_name, filename)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal      INTEGER NOT NULL,
    chapter_path TEXT NOT NULL,          -- JSON 数组
    title        TEXT NOT NULL,
    text         TEXT NOT NULL,
    token_count  INTEGER NOT NULL,
    content_sha  TEXT NOT NULL,
    page         INTEGER                 -- 起始物理页码（1-based），docx 为 NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- 全文索引：body 存 jieba 预分词结果（空格分隔），用 FTS5 内置 bm25() 排序
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    title,
    body,
    tokenize = 'unicode61'
);

-- 索引元信息（embedding 模型与维度），用于换模型后的显式校验
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    state       TEXT NOT NULL,
    progress    REAL NOT NULL DEFAULT 0,
    message     TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    finished_at TEXT
);

-- embedding 结果缓存：同一 (模型, 文本 sha) 只付一次费。
-- 注意：这张表是「省钱的加速层」，丢了最多重算，不承载业务事实。
CREATE TABLE IF NOT EXISTS embed_cache (
    model TEXT NOT NULL,
    sha   TEXT NOT NULL,
    dim   INTEGER NOT NULL,
    vec   BLOB NOT NULL,
    PRIMARY KEY (model, sha)
);

-- OCR 结果缓存：同一张页图（同哈希、同模型）只识别一次。
--
-- 这张表不只是省钱。视觉模型的输出**不确定**：同一页识别两次，标点、断行甚至
-- 标题层级都可能差一点。而 `chunk_id` 是内容寻址的，正文差一个字就是另一批 id，
-- `--force` 重入库会得到"内容相同、id 全变"的一份新文档——旧向量原地成孤儿，
-- `verify` 却一切正常，因为两侧都自洽。有缓存才谈得上"同一份文件重跑结果一致"。
--
-- 与 `embed_cache` 一样属于加速层，不是业务事实：清空数据库时保留。
CREATE TABLE IF NOT EXISTS ocr_cache (
    page_sha   TEXT NOT NULL,
    model      TEXT NOT NULL,   -- 含提示词版本，改提示词等于换模型
    text       TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (page_sha, model)
);

-- numpy 向量后端使用；milvus 后端不使用
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id     TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    project_code TEXT NOT NULL DEFAULT '',
    doc_year     INTEGER,
    dim          INTEGER NOT NULL,
    model        TEXT NOT NULL,
    vec          BLOB NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """就地补列。`CREATE TABLE IF NOT EXISTS` 对已有库**不会**加列，所以历史库要单独
    补。这里只做幂等的小步迁移：缺哪列补哪列，绝不重建表（重建会丢数据、还要重新调
    embedding）。page 列是纯展示字段，补出来是 NULL 也不影响检索——旧库要拿到页码，
    重新入库对应文档即可。
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "page" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN page INTEGER")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """一个写事务。

    **必须串行**：整进程共用一条连接，`check_same_thread=False` 只让它"能用"，不保证
    "能交错"。两个线程同时开事务时，先到的那个 `COMMIT` 会把另一个还没写完的事务一起
    提交，回滚同样会互相踩——这类问题的表现是"数据偶尔少几条"，最难复现。
    锁在 `concurrency`，和 NumpyStore 的直写共用同一把。
    """
    with sqlite_write():
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
