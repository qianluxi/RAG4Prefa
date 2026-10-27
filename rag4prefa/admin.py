"""管理用例：项目/文档的查询与删除，全部以 SQLite 为准计数。"""

from __future__ import annotations

import json
import sqlite3

from . import consistency, repo
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

    **锁内还要确认后端身份**（2026-10-24 第十五轮复审）：维护锁只保证"没有别人同时在
    改"，锁不住"手里这个后端是不是库里记着的那个"——那是装配那一刻定死的，而装配发生在
    拿锁之前。少了这一步，就会先删掉 SQLite 的事实与归档，再往一个已经不是活动后端的
    索引发一批 chunk_id：删除既不完整，事后也说不清该以哪边为准。

    **删除也是一次跨组件修改**（第十六轮）：事实与归档在 SQLite 里删，向量在后端里删，
    两者不在同一个事务。后端删除失败时，归档已经没了、索引里还留着那些 id——这时计数
    校验多半能发现（索引多于归档），但如果同一个事务窗口里还有别的写入凑巧补齐了差额，
    就看不出来了。所以动手之前先留 `unsynced:project:<名字>` 标记：成功才撤，失败留着，
    检索继续拒绝，直到 `rebuild` 从归档重来一遍。
    """
    with maintenance_lock(conn, purpose=f"删除项目 {name}", timeout=0.0):
        # 排在 `repo.delete_project` 之前：被拦住时要做到一个字节都没改。
        consistency.ensure_backend_current(conn, store)
        if not repo.get_project(conn, name):
            return {"deleted": False, "reason": "项目不存在"}

        previous = repo.get_index_state(conn)
        marker = consistency.mark_unsynced(
            conn, f"{consistency.UNSYNCED_PROJECT_PREFIX}{name}"
        )
        try:
            chunk_ids = repo.delete_project(conn, name)
        except Exception:
            # 事实没删成，索引也没动：这次改动没有发生，标记要退回去
            repo.set_index_state(conn, previous)
            raise
        try:
            store.delete(chunk_ids)
        except Exception as exc:
            raise consistency.IndexMismatch(
                f"项目「{name}」的事实与归档已删除，但向量索引里的 {len(chunk_ids)} 条没能删掉"
                f"（{type(exc).__name__}: {exc}）。检索已停用（标记 {marker}）："
                "索引里现在留着没有正文的向量，按条数也不一定看得出差异。"
                "执行 `python -m rag4prefa rebuild` 从归档重建一份索引即可清除该标记"
            ) from exc
        repo.set_index_state(conn, previous)
        return {"deleted": True, "chunks": len(chunk_ids)}


def delete_document(conn: sqlite3.Connection, store: VectorStore, doc_id: str) -> dict:
    """删单个文档。与 `delete_project` 同一套互斥与留痕约束（见其 docstring）。"""
    with maintenance_lock(conn, purpose=f"删除文档 {doc_id}", timeout=0.0):
        consistency.ensure_backend_current(conn, store)
        if not repo.get_document_by_id(conn, doc_id):
            return {"deleted": False, "reason": "文档不存在"}

        previous = repo.get_index_state(conn)
        marker = consistency.mark_unsynced(
            conn, f"{consistency.UNSYNCED_PROJECT_PREFIX}doc:{doc_id}"
        )
        try:
            chunk_ids = repo.delete_document(conn, doc_id)
        except Exception:
            repo.set_index_state(conn, previous)
            raise
        try:
            store.delete(chunk_ids)
        except Exception as exc:
            raise consistency.IndexMismatch(
                f"文档「{doc_id}」的事实与归档已删除，但向量索引里的 {len(chunk_ids)} 条没能删掉"
                f"（{type(exc).__name__}: {exc}）。检索已停用（标记 {marker}）。"
                "执行 `python -m rag4prefa rebuild` 从归档重建索引即可清除该标记"
            ) from exc
        repo.set_index_state(conn, previous)
        return {"deleted": True, "chunks": len(chunk_ids)}


def clear_all(conn: sqlite3.Connection, store: VectorStore) -> dict:
    """清空整个库：项目、文档、片段、全文索引、向量（索引与归档）、任务记录。

    **这是修复入口，所以不要求"索引自证一致"**。别的破坏性动作都在 `assert_write_ready`
    之后才动手，理由是不能往一个说不清状态的地方继续写；清空的终点是"什么都没有"，
    那个约束在这里没有意义，而卡在 `resynced`/`unsynced`/半途劫持状态里的库恰恰最需要
    一个从头来过的出口。仍然要求**后端身份对得上**：认不出手里这个索是不是库里的那个，
    就可能把别人家的数据当自己的擦掉。

    顺序是「先清索引、后清事实」，与入库相反：
    - 反过来先删归档时，若后端删不掉，归档已经没了、索引里还留着向量，`rebuild` 失去
      恢复源，只剩 `reindex` 出一条空索引——结果同样是空，但那不叫"清干净了"，叫
      "数据被悄悄扔掉"。
    - 先清索引则失败时 SQLite 一个字节都没动，可以直接重试。

    完成判据（缺一条都不算清空）：后端条数确认为 0（读不到不算 0）+ 五张业务表归 0 +
    `meta` 里的模型/维度声明清掉、状态回到 ready。
    """
    with maintenance_lock(conn, purpose="清空数据库", timeout=0.0):
        # 排在所有删除之前：被拦住时一个字节都不许改。
        consistency.ensure_backend_current(conn, store)

        before = repo.data_counts(conn)
        try:
            store.clear()
        except Exception as exc:
            raise consistency.IndexMismatch(
                f"向量索引没能清空（{type(exc).__name__}: {exc}）。数据库没有任何改动，"
                "可以重试；持续失败就先看 /api/health 的索引状态，或重启服务后再试"
            ) from exc

        # 「读不到 ≠ 空」：后端条数读不出来时不能当 0 继续往下删。
        try:
            left = store.count()
        except Exception as exc:
            raise consistency.IndexMismatch(
                f"清空了向量索引，但读不到它的实际条数（{type(exc).__name__}: {exc}），"
                "无法确认是否清干净。数据库没有任何改动，请先排除后端问题再重试"
            ) from exc
        if left:
            raise consistency.IndexMismatch(
                f"向量索引里还剩 {left} 条没删掉（`clear()` 返回了，但条数没归零）。"
                "数据库没有任何改动。请重启服务后重试，或先重建索引"
            )

        repo.clear_all_data(conn)
        repo.clear_index_meta(conn)
        after = repo.data_counts(conn)
        # 完成判据的最后一条。删除语句失败会抛异常，但"执行了却没删干净"不会——
        # 而一份"显示已清空、实际还有片段"的库会一直参与检索，且没人会想到去查。
        if any(after.values()):
            raise consistency.IndexMismatch(
                f"业务表没有清干净（剩余 {after}）。索引已清空，数据库还剩这些记录，"
                "请重跑 `python -m rag4prefa clear --yes`"
            )
        return {"cleared": True, "before": before, "after": after}
