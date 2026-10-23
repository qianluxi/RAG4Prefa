"""入库用例：解析 → 分块 → 向量化 → 写库存索引，全过程幂等。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from . import consistency, repo
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
from .consistency import IndexMismatch
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
    """把一个 docx 变成可检索的片段。

    步骤顺序是按"失败后用户还剩什么"排的，不是按"最短代码路径"排的：

    1. 解析 + 分块（纯本地，失败无副作用）；
    2. **先把整份文档算成向量**——这一步要走网络、最可能失败，此时索引里还是旧版本，
       用户照样能检索，只是内容还没更新；
    3. 再在一个事务里切换片段（事实），随后覆盖向量（派生）与归档；
    4. 最后才把新的 content_sha 与 indexed 状态记下来。

    旧实现把"删旧片段"放在调 embedding 之前，于是限流/超时/密钥失效一次，
    原先能用的版本就没了，只剩一个 failed 的空壳。
    """
    filename = path.name
    doc_id = _doc_id(project_name, filename)
    try:
        # 第一步就是一致性检查：模型对不上时，连项目/文档记录都不要建，
        # 否则库里会多出一个空项目，让人误以为"导过了"。
        try:
            consistency.ensure_writable(conn, embedder.model, embedder.dim)
        except IndexMismatch as exc:
            return FileOutcome(filename, "failed", 0, 0, str(exc))

        repo.upsert_project(conn, project_name, project_code)
        content_sha = _sha_file(path)
        existing = repo.get_document(conn, project_name, filename)
        # 只有"上一版确实建成功了"才允许跳过；失败过的文档必须能重试，
        # 否则一次限流就把这份文件永久钉在 failed 上。
        if (
            existing
            and existing.status == "indexed"
            and existing.content_sha == content_sha
            and not force
        ):
            return FileOutcome(filename, "skipped", existing.section_count, existing.chunk_count)

        sections = parse_docx(path)
        chunks = build_chunks(
            sections,
            doc_id,
            max_tokens=settings.chunk_max_tokens,
            min_tokens=settings.chunk_min_tokens,
        )
        if not chunks:
            if existing is None:
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

        repo.mark_document_indexing(
            conn, doc_id=doc_id, project_name=project_name, filename=filename, doc_year=doc_year
        )
        vectors = embedder.embed([contextual_text(c) for c in chunks])

        items = [
            VectorItem(
                chunk_id=c.id,
                vector=v,
                project_name=project_name,
                project_code=project_code,
                doc_year=doc_year,
                model=embedder.model,
            )
            for c, v in zip(chunks, vectors, strict=True)
        ]

        # 事实先落在一个事务里；向量随后覆盖，再删掉这一版淘汰的片段
        stale = repo.replace_document_chunks(conn, doc_id, chunks)
        repo.archive_vectors(conn, [to_archive_row(i) for i in items])
        store.upsert(items)
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
            status="indexed",
        )
        # 元数据由"真正写过向量的人"记，而不是指望每个调用方都记得调一次
        repo.set_meta(conn, "embedding_model", embedder.model)
        repo.set_meta(conn, "embedding_dim", str(embedder.dim or len(items[0].vector)))
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


class ReindexError(RuntimeError):
    """重建条件不满足，必须由人决定怎么继续——不静默降级，也不删掉现有索引。"""


def rebuild_store(conn: sqlite3.Connection, store: VectorStore) -> int:
    """从 SQLite 归档重建向量索引（不重新 embedding，换后端/索引损坏时用）。"""
    items = [from_archive_row(row) for row in repo.iter_archived_vectors(conn)]
    store.rebuild(items)
    return len(items)


def _contextual_text(row: dict) -> str:
    """与入库同一套上下文化规则：章节路径 + 正文，换模型重建时才不会改变检索语义。"""
    path = " > ".join(json.loads(row["chapter_path"]))
    return f"{path}\n{row['text']}" if path else row["text"]


def reindex(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    project_name: str | None = None,
) -> dict:
    """按 SQLite 里的事实重新 embedding，并重建指定范围的向量索引（换模型后必做）。

    执行顺序刻意如此：
    1. 先做前置校验（模型、维度），不满足就直接拒绝，一个字节都不改；
    2. 再把整个范围的向量算完放在内存里——中途失败时索引与归档都还是旧的一套，不留半成品；
    3. 然后写 SQLite 归档（事实），最后写向量索引（派生）——索引写坏还能用
       rebuild_store 从归档补回，反过来则会把已经把新向量算完的成果弄丢；
    4. 全部成功后才更新 meta，避免"元数据说新模型、索引里还是旧模型"。

    范围语义：
    - 不给 project_name → 全量重建，允许换模型、换维度（唯一的换模型入口）；
    - 指定项目 → 只动这个项目，**绝不 clear 全库**；且必须沿用索引现有的模型——
      维度相同不代表向量空间相同，局部换模型会得到"能查、但结果悄悄变差"的索引。
    """
    rows = repo.chunks_for_reindex(conn, project_name)
    if not rows:
        return {"scope": project_name or "all", "chunks": 0, "rebuilt": False}

    if project_name:
        _ensure_same_model(conn, embedder.model)

    items: list[VectorItem] = []
    batch_size = max(1, settings.embedding_batch)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        vectors = embedder.embed([_contextual_text(r) for r in batch])
        for r, v in zip(batch, vectors, strict=True):
            items.append(
                VectorItem(
                    chunk_id=r["chunk_id"],
                    vector=v,
                    project_name=r["project_name"],
                    project_code=r["project_code"],
                    doc_year=r["doc_year"],
                    model=embedder.model,
                )
            )

    new_dim = len(items[0].vector)
    if any(len(i.vector) != new_dim for i in items):
        raise ReindexError("embedding 服务返回的向量维度不一致，已中止（索引未做任何改动）")

    dim_changed = new_dim != store.dim
    if dim_changed and project_name:
        raise ReindexError(
            f"embedding 维度由 {store.dim} 变为 {new_dim}，向量索引只能整体重建，"
            f"无法只重建项目「{project_name}」。请去掉 --project 执行全量 reindex。"
        )

    repo.archive_vectors(conn, [to_archive_row(i) for i in items])
    if project_name:
        store.upsert(items)  # 局部：只覆盖本次范围内的 chunk_id
    else:
        # 全量：归档、索引、片段三者要对齐，先把归档里的孤儿清掉，
        # 否则 rebuild_store 会把已经没有正文的旧向量重新灌回索引。
        repo.delete_orphan_vectors(conn)
        store.rebuild(items)
    repo.set_meta(conn, "embedding_model", embedder.model)
    repo.set_meta(conn, "embedding_dim", str(new_dim))
    return {
        "scope": project_name or "all",
        "chunks": len(items),
        "model": embedder.model,
        "dim": new_dim,
        "rebuilt": project_name is None,  # 全量才是"整体替换"，局部只是覆盖本次范围
        "dim_changed": dim_changed,
    }


def _ensure_same_model(conn: sqlite3.Connection, model: str) -> None:
    """局部重建只允许沿用索引现有模型；要换模型就必须全量。"""
    state = consistency.inspect(conn)
    if not state.count:
        return
    if len(state.known_models) > 1:
        listed = "、".join(sorted(state.known_models))
        raise ReindexError(
            f"索引里已混有两套模型的向量（{listed}），只重建一个项目修不好这个问题。"
            "请去掉 --project 执行全量 reindex。"
        )
    if state.known_models and model not in state.known_models:
        index_model = "、".join(sorted(state.known_models))
        raise ReindexError(
            f"局部重建必须沿用索引现有的 embedding 模型：索引用的是「{index_model}」，"
            f"当前配置是「{model}」。维度相同也不代表向量空间相同，混用会让检索静默变差。"
            "确实要换模型，请去掉 --project 执行全量 reindex。"
        )


def doc_year_from(value: str | None, fallback: int | None = None) -> int | None:
    if not value:
        return fallback
    for token in (value[:4], value[:4][::-1]):
        if token.isdigit() and 1900 <= int(token) <= 2200:
            return int(token)
    return fallback


def chunk_tokens(chunks: list[Chunk]) -> int:
    return sum(estimate_tokens(c.text) for c in chunks)
