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
from .concurrency import MaintenanceBusy, maintenance_lock
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
    # 附带说明。用于"这次确实做了事，但不是重新入库"——例如内容没变只改了年份，
    # 或者重导入时发现调度被别的维护动作挡过。放进 error 会被当成失败。
    note: str | None = None

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "status": self.status,
            "sections": self.sections,
            "chunks": self.chunks,
            "error": self.error,
            "note": self.note,
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
    """入库一个文件。真正干活的是 `_ingest_locked`，这里是维护锁的入口。

    为什么分成两层：抢不到锁和"这份文件有问题"要说成两件事。前者是任务级拒绝
    （别人正在重建索引），换一份文件、多试几次都一样；后者才是文件/网络的问题。
    把它们混成同一个失败原因，用户会去检查一份没问题的文档。
    """
    try:
        with maintenance_lock(conn, purpose=f"入库 {path.name}", timeout=0.0):
            return _ingest_locked(
                conn,
                store,
                embedder,
                settings,
                path=path,
                project_name=project_name,
                project_code=project_code,
                doc_year=doc_year,
                force=force,
            )
    except MaintenanceBusy as exc:
        return FileOutcome(path.name, "failed", 0, 0, str(exc))


def _ingest_locked(
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
    overwritten: list[dict] = []
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
            if doc_year is None or doc_year == existing.doc_year:
                return FileOutcome(
                    filename, "skipped", existing.section_count, existing.chunk_count
                )
            # 内容一模一样、只有年份不同。年份是**过滤条件**：当"没变化"跳过，
            # 用户改的那一年就永远不生效，而且界面上还显示成功（skipped）。
            return _update_doc_year(conn, store, doc_id, filename, doc_year, existing)

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
        # 新旧版本**共用**的 chunk_id 也会被这次 upsert 整行覆盖（doc_year、project_code）。
        # 事实事务万一失败，只删新增会让这些行停在"新版的过滤元数据 + 旧版的正文"：
        # 条数完全对得上，按年份过滤却错了一年。所以先留一份旧行，失败时原样写回去
        # （2026-10-24 第十四轮复审）。
        kept_ids = [i.chunk_id for i in items if i.chunk_id in preexisting]
        overwritten = repo.archived_vectors_by_ids(conn, kept_ids)
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
        if not facts_committed and (planned or overwritten):
            detail += _restore_vectors(store, planned, overwritten, stage)
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


def _restore_vectors(
    store: VectorStore, planned: list[str], overwritten: list[dict], stage: str
) -> str:
    """把索引恢复到"这次入库之前"的样子：删掉新增的 id，并把被覆盖的旧行写回去。

    两步都不能省。

    - 只删新增：新旧版本共用的 chunk_id 上，`doc_year` / `project_code` 已经被写成
      这一版的值。条数一条不差、模型维度也对，只有"按年份过滤"会错——计数校验看不见。
    - 只写回旧行：本次新增的向量留在索引里却没有正文，检索会跳过它们，但它们占掉
      召回名额，稠密召回因此少一条。

    还原本身失败时要说清"哪一步没成、现在是什么状态"，因为这两种残留的修复方式不同：
    条数多出来（新增没删掉）由一致性检查拦住，内容不一致只能靠 rebuild。
    """
    parts: list[str] = []
    failed: list[str] = []
    if overwritten:
        try:
            store.upsert([from_archive_row(r) for r in overwritten])
            parts.append(f"已把被覆盖的 {len(overwritten)} 条向量还原成上一版")
        except Exception as exc:  # noqa: BLE001 - 收尾动作失败只影响说明文字
            failed.append(f"还原 {len(overwritten)} 条旧向量失败（{type(exc).__name__}: {exc}）")
    if planned:
        try:
            store.delete(planned)
            parts.append(f"已撤回本次新增的 {len(planned)} 条向量")
        except Exception as exc:  # noqa: BLE001 - 同上
            failed.append(f"撤回新增的 {len(planned)} 条向量失败（{type(exc).__name__}: {exc}）")

    if not failed:
        return f"（{stage}失败，{'，'.join(parts)}，库仍是上一版）"
    return (
        f"（{stage}失败，{'，'.join([*parts, *failed])}；"
        "索引可能与归档不一致——条数相同也可能内容不同，"
        "执行 `python -m rag4prefa rebuild` 从归档恢复后再重试同一文件）"
    )


def _update_doc_year(
    conn: sqlite3.Connection,
    store: VectorStore,
    doc_id: str,
    filename: str,
    doc_year: int,
    existing,
) -> FileOutcome:
    """内容没变、只改了年份：不重算向量，只把三处过滤列对齐。

    为什么不走"重新入库一遍"：那要重新解析 docx、重算 embedding（有缓存也只是少花钱）、
    把一份本来正确的索引整表重写一遍——为一个只影响过滤的整数付这些代价，而且重写
    过程还有失败窗口。这条路径一个向量都不用算，只覆盖该文档那几十条。

    **先验证能不能改，再动数据**：归档不全时（有片段没向量）改年份会把关系弄得更乱，
    那种情况应当先 `reindex` 补齐，而不是在这里"尽力而为"。
    """
    ids = repo.chunk_ids_of_doc(conn, doc_id)
    if not ids:
        return FileOutcome(
            filename,
            "failed",
            existing.section_count,
            0,
            "文档状态是已入库，但库里查不到它的片段；用 --force 重新导入，或执行 reindex",
        )
    if len(repo.archived_vectors_by_ids(conn, ids)) != len(ids):
        return FileOutcome(
            filename,
            "failed",
            existing.section_count,
            len(ids),
            "该文档有片段缺少归档向量，改年份前需要先把归档补全"
            "（执行 `python -m rag4prefa reindex`）；本次未做任何改动",
        )

    repo.set_document_year(conn, doc_id, doc_year)
    try:
        store.upsert([from_archive_row(r) for r in repo.archived_vectors_by_ids(conn, ids)])
    except Exception as exc:  # noqa: BLE001 - 文档与归档已是新值，索引没跟上要说清
        return FileOutcome(
            filename,
            "failed",
            existing.section_count,
            len(ids),
            f"年份已写入文档与归档，但更新向量索引失败（{type(exc).__name__}: {exc}）；"
            "执行 `python -m rag4prefa rebuild` 从归档恢复索引即可（归档里已是新年份）",
        )
    return FileOutcome(
        filename,
        "indexed",
        existing.section_count,
        len(ids),
        note=f"内容未变，只把年份更新为 {doc_year}",
    )


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
    started_on: str | None = None,
    doc_year: int | None = None,
    timeout: float | None = 0.0,
) -> list[FileOutcome]:
    """入库一批文件。**整批一次占住维护锁**，而不是每个文件各抢一次。

    为什么粒度是"整批"：

    - 项目记录、任务记录一旦落库，用户就会把它当成"已经导过了"。如果只在文件级别
      抢锁，第一个文件成功、第二个撞上别人的重建，项目照样存在、任务照样是 failed，
      库里留下一个只有一半内容的项目；更糟的是调用方（API）在**批次开始之前**就把
      项目建好了，一次被拒绝就能留下一个空项目。
    - 批次期间后端指针可能被迁移换掉。把"当前后端"锁在批次里读一次（`StoreRef`），
      就不会出现"前两个文件写进 numpy、根服务已经在查 Milvus"。

    抢锁失败在这里抛出 `MaintenanceBusy`，此时任务、项目、文档都还没被动过。
    """
    with maintenance_lock(
        conn, purpose=f"入库 {project_name}（{len(files)} 个文件）", timeout=timeout
    ):
        return _ingest_batch_locked(
            conn,
            store,
            embedder,
            settings,
            job_id=job_id,
            files=files,
            project_name=project_name,
            project_code=project_code,
            started_on=started_on,
            doc_year=doc_year,
        )


def _ingest_batch_locked(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    job_id: str,
    files: list[Path],
    project_name: str,
    project_code: str = "",
    started_on: str | None = None,
    doc_year: int | None = None,
) -> list[FileOutcome]:
    # 装配发生在锁外，手里这个后端可能已经不是库里记着的那个了（见函数说明）。
    # 必须排在项目/任务写入**之前**：被拦住时要做到一个字节都没改。
    consistency.ensure_backend_current(conn, store)
    # 项目与任务状态都在锁内写：锁外的"预创建"一旦被拒绝就是一条假记录
    repo.upsert_project(conn, project_name, project_code, started_on)
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

    "恢复完成"= 索引、归档、元数据三者一起回到可服务状态，而不只是"重建函数没抛异常"。
    这三步缺任何一步，命令都会报成功而系统仍然查不了：

    1. **先证归档能当恢复源**：有正文就必须有归档向量，归档行必须带模型名，全库只有
       一套模型、一种维度，且每行向量的**实际字节长度**与声明的维度相符。不合格就抛错
       并**保留**未完成标记，别在这里反复重试；
    2. 再从归档重建索引，核对后端实际条数**和实际 id 集合**——"写进去了"要以 id 为准。
       条数相同只是必要条件：两个版本的条数可以一模一样，只有 chunk_id 不同（重建期间
       有并发写入时就是这样），那种库能检索、只是检索的是另一批内容；
    3. 最后清掉孤儿归档行，并把归档对应的模型、维度连同 `index_state=ready` 在
       **同一个事务**里提交。

    第 3 步是必须的：换模型重建中断后，归档已经是新模型、元数据还停在旧模型。只恢复
    索引和标记的话，一致性检查会因为"元数据模型 ≠ 归档模型"继续拒绝检索——用户看到
    的是"恢复命令说成功，可还是查不了"。

    顺序同理不能反：**目标后端重建成功之前，归档一个字节都不许动**。numpy 后端的内存
    索引就是 `vectors` 表的缓存，先删孤儿行再失败，它的条数就和归档对不上了——一次
    "可选迁移失败"会把本来能用的后端变成不可检索。旧实现正是在这个位置先删后建。

    整个过程持有维护锁：从"读归档快照"到"切完索引"之间不许有别的写入插进来。锁的
    时间范围比函数体本身更早生效的只有调用方（例如 `reindex` 先算 embedding 再进来），
    那种情况由外层锁负责。
    """
    with maintenance_lock(conn, purpose="从归档重建索引", timeout=None):
        return _rebuild_store_locked(conn, store)


def _rebuild_store_locked(conn: sqlite3.Connection, store: VectorStore) -> int:
    try:
        consistency.ensure_backend_current(conn, store)
    except IndexMismatch as exc:
        raise ReindexError(str(exc)) from exc
    # 只有仍有正文的行才算恢复源；孤儿行留着会让重建后的索引比事实表更大
    rows = repo.iter_archived_vectors(conn)
    models = {r["model"] for r in rows}
    dims = {int(r["dim"] or 0) for r in rows}

    problems: list[str] = []
    missing = repo.missing_vector_count(conn)
    if missing:
        problems.append(f"{missing} 个片段没有归档向量，这份归档是残缺的，无法作为恢复源")
    if "" in models:
        problems.append("有归档向量没有记录产生它的模型，无法确认向量空间")
    if len(models) > 1:
        problems.append(f"归档里混有多个模型的向量（{'、'.join(sorted(models))}）")
    if len(dims) > 1:
        problems.append(f"归档里混有多种维度（{'、'.join(str(d) for d in sorted(dims))}）")
    elif rows and 0 in dims:
        problems.append("有归档向量记录的维度是 0，无法确认向量空间")
    bad = [r for r in rows if len(r["vec"]) != int(r["dim"] or 0) * 4]
    if bad:
        # `dim` 字段只是声明，float32 字节长度才是事实。只比声明的话，"每行都写 1024、
        # 实际只有 768 个浮点"的归档照样通过校验，恢复出来的索引和 meta 对不上，
        # 命令报成功、检索却继续被拒。
        first = bad[0]
        problems.append(
            f"有 {len(bad)} 条归档向量的实际长度与声明的维度对不上"
            f"（例如 {first['chunk_id']}：声明 {first['dim']} 维、"
            f"实际 {len(first['vec']) / 4:g} 个浮点数）——归档已被写坏"
        )
    if problems:
        raise ReindexError(
            "不能从归档恢复（索引与重建标记均未改动）：\n- "
            + "\n- ".join(problems)
            + "\n执行 `python -m rag4prefa reindex` 全量重算，别在这份归档上反复重试。"
        )

    items = [from_archive_row(r) for r in rows]
    store.rebuild(items)
    _assert_backend_matches(store, {i.chunk_id for i in items}, "归档")

    # 到这里目标后端已经是新的一套，再清孤儿行就伤不到任何一方：
    # 归档里剩下的孤儿本来也不在 items 里，删掉只是让"归档 = 索引"成立。
    repo.delete_orphan_vectors(conn)
    model = next(iter(models), "") or (repo.get_meta(conn, "embedding_model") or "")
    dim = str(next(iter(dims), 0) or store.dim)
    repo.set_index_meta(conn, model=model, dim=dim, state=repo.INDEX_READY)
    return len(items)


def _assert_backend_matches(
    store: VectorStore, expected: set[str], source: str
) -> None:
    """核对后端里的 id 集合与本次应当写入的那一套完全一致。

    为什么条数不够：重建期间若有并发写入，两边可以条数相同、chunk_id 不同。那种库
    能检索、`health` 也看不出异常，只是检索到的是另一批内容。`consistency.verify_ids`
    早就在做逐 id 比对，但它不在主路径上——恢复动作在宣布 `ready` 之前必须自己过一遍，
    否则"恢复成功"的判据比日常校验还弱。
    """
    got = store.count()
    if got != len(expected):
        raise ReindexError(
            f"向量后端实际 {got} 条、{source} {len(expected)} 条，恢复未完成，"
            "重建标记保持原样。可直接重试 rebuild。"
        )
    try:
        actual = store.ids()
    except Exception as exc:  # 后端读不出主键时不能假设它写对了
        raise ReindexError(
            f"读不到向量后端的主键，无法确认恢复结果（{type(exc).__name__}: {exc}）；"
            "重建标记保持原样。"
        ) from exc
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ReindexError(
            f"后端条数与{source}相同（{got} 条），但 id 集合不同："
            f"缺 {len(missing)} 个、多 {len(extra)} 个"
            f"（例如缺 {missing[:2]}、多 {extra[:2]}）。"
            "这说明重建期间有别的写入改了索引，重建标记保持原样；"
            "确认没有并发的入库/删除后再重试。"
        )


def _contextual_text(row: dict) -> str:
    """与入库同一套上下文化规则：章节路径 + 正文，换模型重建时才不会改变检索语义。"""
    path = " > ".join(json.loads(row["chapter_path"]))
    return f"{path}\n{row['text']}" if path else row["text"]


def _reindex_nothing(
    conn: sqlite3.Connection, store: VectorStore, project_name: str | None
) -> dict:
    """重建范围里没有片段时的收尾。

    这里最容易写错的一步是"没数据就等于已经一致"。两种情形必须分开：

    - **局部重建**空范围：只能说明这个项目没内容，和"全库那笔重建没走完"毫无关系。
      顺手指把 `rebuilding:*` 改成 ready，那笔失败就再没有任何痕迹能发现——归档与
      旧索引的条数、模型、维度本来就可能一模一样。所以局部一律不动标记。
    - **全量重建**空范围：先确认"真的空"再清。片段表非空却一条重建输入都取不到，
      是数据缺失（JOIN 不上项目/文档），不是可以清理的残留；此时清索引等于把正文
      留在库里、向量全丢掉，而且从此没有任何标记提示这件事发生过。
    """
    scope = project_name or "all"
    if project_name:
        pending = repo.get_index_state(conn)
        note = "该项目没有可重建的片段，未做任何改动"
        if pending != repo.INDEX_READY:
            note += (
                f"；索引仍处于未完成状态（{pending}），"
                "需要全量 reindex（不带 --project）或 rebuild 才能清除"
            )
        return {"scope": scope, "chunks": 0, "rebuilt": False, "note": note}

    chunks = repo.chunk_count(conn)
    if chunks:
        raise ReindexError(
            f"片段表里有 {chunks} 条正文，却一条都取不到重建输入——"
            "多半是所属项目或文档记录缺失（JOIN 不上），这不是可以清理的残留。"
            "先跑 `python -m rag4prefa verify` 定位并修好数据，再重建。"
            "索引与标记均未改动。"
        )
    # 片段真为空：归档里的行必然是孤儿，索引里也可能留着上一版的向量。
    # 顺序与普通重建一致——**先清后端、确认它真的空了，再动归档**。
    # 反过来的话，删完归档行、后端清理失败，就会停在"归档已改、索引还在"的中间态；
    # 而这条路径未必带着 `rebuilding:*` 标记，事后无从判断。
    if store.count():
        store.rebuild([])
    left = store.count()
    if left:
        raise ReindexError(
            f"向量后端仍有 {left} 条残留向量，清空没有成功；归档与索引标记均未改动。"
            "先处理后端（例如确认没有别的进程在用同一个集合），再重试 reindex。"
        )
    removed = repo.delete_orphan_vectors(conn)
    repo.set_index_state(conn, repo.INDEX_READY)
    return {
        "scope": scope,
        "chunks": 0,
        "rebuilt": False,
        "orphans_removed": removed,
    }


def reindex(
    conn: sqlite3.Connection,
    store: VectorStore,
    embedder: Embedder,
    settings: Settings,
    *,
    project_name: str | None = None,
) -> dict:
    """按 SQLite 里的事实重新 embedding，并重建指定范围的向量索引（换模型后必做）。

    **全程持有维护锁**，从"读片段快照"一直到"写完元数据"——因为中间那段 embedding
    可能要跑几分钟。只锁写入的话，算完才发现事实表已经变了：那时条数可能仍然一样，
    只有 chunk_id 不一样，`rebuild_store` 的条数校验发现不了。锁期间别的入库会被
    直接拒绝（`MaintenanceBusy`），而不是排队等到一个说不清的失败。
    """
    with maintenance_lock(conn, purpose=f"重建索引（{project_name or '全量'}）", timeout=None):
        return _reindex_locked(
            conn, store, embedder, settings, project_name=project_name
        )


def _reindex_locked(
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
      维度相同不代表向量空间相同，局部换模型会得到"能查、但结果悄悄变差"的索引；
    - 范围内没有片段时见 `_reindex_nothing`：局部一律不动标记，全量要先确认"真的空"。
    """
    # 空范围也要先校验：`_reindex_nothing` 同样会清后端（rebuild([])），
    # 写到"不是当前活动后端"的那一份上，等于把别人的索引清空。
    try:
        consistency.ensure_backend_current(conn, store)
    except IndexMismatch as exc:
        raise ReindexError(str(exc)) from exc
    rows = repo.chunks_for_reindex(conn, project_name)
    if not rows:
        return _reindex_nothing(conn, store, project_name)

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
        # 只核对"这次写进去的那些"确实在后端里：全量核对是 verify 的活，
        # 局部重建没资格要求别的项目也一致。
        missing = {i.chunk_id for i in items} - store.ids()
        if missing:
            raise ReindexError(
                f"局部重建后有 {len(missing)} 条向量没有真正写进后端"
                f"（例如 {sorted(missing)[:2]}），标记仍是 {marker}，可重跑同一个项目修复。"
            )
    else:
        # 全量：先把索引整体换掉，再清没有任何正文的归档行——顺序不能反。
        # 先清的话，重建一旦失败，归档已经少了行而索引还停在旧的一套；numpy 后端
        # 的内存索引更是直接与表脱节，后面的修复动作反而更难判断起点。
        store.rebuild(items)
        _assert_backend_matches(store, {i.chunk_id for i in items}, "本次重建")
        repo.delete_orphan_vectors(conn)
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
