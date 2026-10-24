"""入库用例：解析 → 分块 → 向量化 → 写库存索引，全过程幂等。"""

from __future__ import annotations

import contextlib
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
    2. **先把整份文档算成向量**——这一步要走网络、最可能失败，此时索引与事实都还是旧版本；
    3. **再写向量索引**：chunk_id 是内容寻址的（见 `chunking._chunk_id`），所以新写入的
       只可能是"新内容"或"内容相同的老片段"，旧版本的正文不会被新向量配上；
    4. 最后用一个事务把片段与归档一起切换，再删掉这一版淘汰掉的旧向量；
    5. 全部成功后才记 content_sha、`indexed` 状态与索引元数据。

    为什么不是"先切事实再写索引"：那种顺序下，一旦写索引失败，事实已经是新版、索引还是
    旧版，chunk_id 相同的位置会出现"新向量配旧正文"或"新正文配旧向量"，两路召回各说各话。
    现在改成索引先行，失败时旧版本（事实 + 归档 + 索引）完整可用，只需把多写的向量撤回；
    撤回也失败时，一致性检查会发现"索引条数多于归档"并拒绝检索，不会静默给错答案。
    """
    filename = path.name
    doc_id = _doc_id(project_name, filename)
    stage = "预检查"
    # 本次"可能新增"的 chunk_id。为什么是"可能"：它在调用向量后端**之前**就算好了，
    # 因为后端中途失败时不会返回，事后无法知道写进去过哪些。删多算的几个是无害的
    # （id 不存在就是空操作），删少了才会留下没有正文的向量。
    planned: list[str] = []
    facts_committed = False
    doc_row_exists = False
    try:
        # 第一步就是一致性检查：索引不能自证一致时，连项目/文档记录都不要建，
        # 否则库里会多出一个空项目，让人误以为"导过了"。
        try:
            consistency.ensure_writable(
                conn, embedder.model, embedder.dim, store_count=store.count()
            )
        except IndexMismatch as exc:
            return FileOutcome(filename, "failed", 0, 0, str(exc))

        repo.upsert_project(conn, project_name, project_code)
        content_sha = _sha_file(path)
        existing = repo.get_document(conn, project_name, filename)
        doc_row_exists = existing is not None
        # 只有"上一版确实建成功了"才允许跳过；失败过的文档必须能重试，
        # 否则一次限流就把这份文件永久钉在 failed 上。
        if (
            existing
            and existing.status == "indexed"
            and existing.content_sha == content_sha
            and not force
        ):
            return FileOutcome(filename, "skipped", existing.section_count, existing.chunk_count)

        stage = "解析与分块"
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
        doc_row_exists = True

        stage = "计算向量"
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

        # 派生索引先落：此时事实还是旧版本，检索照常可用。
        # 记下切换前该文档已有的 id——chunk_id 是内容寻址的，改一版时未变的片段会沿用
        # 原 id，撤回时只该删"这一版新增的"，否则会把旧版本的向量一起抹掉。
        #
        # **这一版可能新增哪些 id，必须在写入之前算出来**：`upsert()` 是一批批写的，
        # 中途抛异常就不会返回，那时再算 `written` 只会得到空列表，撤回逻辑等于不存在
        # ——残留的向量条数多于归档，检索因此被拒绝（不会给错答案，但"旧版本照常可用"
        # 这个承诺没兑现）。所以先算后写：删多算的几个 id 是无害的，删少了才留垃圾。
        stage = "写入向量索引"
        preexisting = set(repo.chunk_ids_of_doc(conn, doc_id))
        planned = [i.chunk_id for i in items if i.chunk_id not in preexisting]
        store.upsert(items)

        # 事实与归档在同一个事务里切换：读方要么看到旧版、要么看到新版
        stage = "切换文档片段"
        stale = repo.commit_document_version(
            conn, doc_id, chunks, [to_archive_row(i) for i in items]
        )
        facts_committed = True

        stage = "清理淘汰向量"
        if stale:
            store.delete(stale)

        stage = "记录文档状态"
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
        detail = f"{type(exc).__name__}: {exc}"
        if planned and not facts_committed:
            # 索引已（可能部分）写、事实没切换：索引里多出的是"没有正文的向量"，
            # 检索会跳过它们，但会把召回名额占掉。撤回失败也不算灾难——一致性检查
            # 会因此报错并拒绝检索。
            try:
                store.delete(planned)
                detail += f"（{stage}失败，已撤回本次新增的 {len(planned)} 条向量，库仍是上一版）"
            except Exception as rollback_exc:
                detail += (
                    f"（{stage}失败，且撤回写入的向量也失败：{rollback_exc}；"
                    "索引条数会多于归档，检索将被拒绝，重试同一文件或用 reindex 修复）"
                )
        elif facts_committed:
            detail += (
                f"（{stage}失败，但新版本已生效且可检索；"
                "重新导入同一文件即可清理残留，不要只依赖这次的结果）"
            )
        if doc_row_exists:
            with contextlib.suppress(Exception):
                # 状态都写不进去时只能靠返回值说明，别让收尾动作盖掉原始错误
                repo.set_document_status(conn, doc_id, "failed", detail)
        return FileOutcome(filename, "failed", 0, 0, detail)


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
    """从 SQLite 归档重建向量索引（不重新 embedding，换后端/索引损坏时用）。

    成功后顺手把切换标记恢复成 ready：重建做的事就是"索引 := 归档"，做完之后
    两边必然一致，无论上次中断的是全量还是某个项目。它也是被阻断之后除 `reindex`
    之外的一条更便宜的恢复路径（不重新调 embedding，不花钱）。
    """
    items = [from_archive_row(row) for row in repo.iter_archived_vectors(conn)]
    store.rebuild(items)
    repo.set_index_state(conn, repo.INDEX_READY)
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
    3. **写归档之前先置 `index_state` 标记**。SQLite 与向量库不在同一个事务里，这笔
       "两阶段提交"没法用 BEGIN 表达，只能显式留痕；
    4. 然后写 SQLite 归档（事实），最后写向量索引（派生）——索引写坏还能用
       rebuild_store 从归档补回，反过来则会把已经把新向量算完的成果弄丢；
    5. 全部成功后才把模型、维度、状态一起提交，避免"元数据说新模型、索引里还是旧模型"。

    第 3 步是关键：归档写完、索引没跟上的那个窗口里，两边的条数、模型名、维度可以
    完全一致，事后逐项校验一条都挑不出毛病。只有写之前留的标记能让检索在这段时间里
    停住，而不是拿旧向量去配新正文。

    范围语义：
    - 不给 project_name → 全量重建，允许换模型、换维度（唯一的换模型入口）；
    - 指定项目 → 只动这个项目，**绝不 clear 全库**；且必须沿用索引现有的模型——
      维度相同不代表向量空间相同，局部换模型会得到"能查、但结果悄悄变差"的索引。
    """
    rows = repo.chunks_for_reindex(conn, project_name)
    if not rows:
        # 没有可重建的内容：此时不存在"新归档 vs 旧索引"的差异，遗留标记可以直接清掉，
        # 否则用户会被一次早已没有对象的失败永久挡在门外。
        repo.set_index_state(conn, repo.INDEX_READY)
        return {"scope": project_name or "all", "chunks": 0, "rebuilt": False}

    # 标记里带范围：`rebuilding:*` 表示全量没走完，`rebuilding:<项目>` 只表示那个项目。
    # 带范围才能判断"这次重建能不能把上次的坑填上"——局部重建填不了全量的坑。
    marker = "rebuilding:*" if project_name is None else f"rebuilding:{project_name}"
    previous = repo.get_index_state(conn)

    if project_name:
        _ensure_same_model(
            conn, embedder.model, store.count(), retry_scope=project_name
        )

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

    # 从这一行起开始改真正的数据：先留痕，再写归档。
    repo.set_index_state(conn, marker)
    try:
        repo.archive_vectors(conn, [to_archive_row(i) for i in items])
    except Exception:
        # 归档写入本身是一个事务，失败就是一个字节都没变；索引也还没动，
        # 所以状态可以原样退回，不必让用户去跑一次全量重建。
        repo.set_index_state(conn, previous)
        raise
    if project_name:
        store.upsert(items)  # 局部：只覆盖本次范围内的 chunk_id
    else:
        # 全量：归档、索引、片段三者要对齐，先把归档里的孤儿清掉，
        # 否则 rebuild_store 会把已经没有正文的旧向量重新灌回索引。
        repo.delete_orphan_vectors(conn)
        store.rebuild(items)
    # 全量重建重写了所有范围，一定能恢复 ready；局部重建只能抵消**同范围**的标记，
    # 若上一笔失败是全量的，状态要原样留着，否则会把全库的坑说成已经填了。
    settled = (
        repo.INDEX_READY
        if project_name is None or previous in (repo.INDEX_READY, marker)
        else previous
    )
    repo.set_index_meta(conn, model=embedder.model, dim=str(new_dim), state=settled)
    return {
        "scope": project_name or "all",
        "chunks": len(items),
        "model": embedder.model,
        "dim": new_dim,
        "rebuilt": project_name is None,  # 全量才是"整体替换"，局部只是覆盖本次范围
        "dim_changed": dim_changed,
    }


def _ensure_same_model(
    conn: sqlite3.Connection,
    model: str,
    store_count: int | None = None,
    *,
    retry_scope: str | None = None,
) -> None:
    """局部重建只允许在"索引本身自洽"的前提下沿用现有模型。

    局部重建修不了全库的不一致（缺向量、混模型、索引与归档对不上），只会让状态更复杂，
    所以这类前提不满足时直接拒绝，把"全量 reindex"指给用户。

    `retry_scope` 让"上一次同项目的局部重建失败"不挡自己：那种标记正是重跑要消除的，
    挡住它只会逼用户去做代价更大的全量重建。
    """
    try:
        consistency.ensure_writable(
            conn,
            model,
            store_count=store_count,
            action="局部重建",
            retry_scope=retry_scope,
        )
    except IndexMismatch as exc:
        raise ReindexError(
            f"{exc}\n局部重建只能覆盖一个项目，修不了这类问题；"
            "请去掉 --project 执行全量 reindex。"
        ) from exc


def doc_year_from(value: str | None, fallback: int | None = None) -> int | None:
    if not value:
        return fallback
    for token in (value[:4], value[:4][::-1]):
        if token.isdigit() and 1900 <= int(token) <= 2200:
            return int(token)
    return fallback


def chunk_tokens(chunks: list[Chunk]) -> int:
    return sum(estimate_tokens(c.text) for c in chunks)
