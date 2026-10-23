"""组件装配：所有依赖在这里创建一次，通过显式注入往下传。

没有模块级单例、没有可变全局状态——除了 SQLite 连接（单进程、WAL、串行写）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from .adapters.embed import Embedder, EmbeddingError
from .adapters.llm import LLM, LLMError
from .adapters.rerank import NoopReranker, RemoteReranker, Reranker
from .adapters.vectors import (
    BackendUnavailable,
    MilvusStore,
    NumpyStore,
    VectorStore,
)
from .config import Settings, get_settings
from .consistency import health_issues, inspect
from .db import connect, init_db
from .repo import chunk_count, get_meta, mark_interrupted_jobs


@dataclass
class Services:
    settings: Settings
    conn: sqlite3.Connection
    store: VectorStore
    embedder: Embedder | None
    llm: LLM | None
    reranker: Reranker
    notes: list[str] = field(default_factory=list)
    fell_back: bool = False

    @property
    def index_dim(self) -> int | None:
        value = get_meta(self.conn, "embedding_dim")
        return int(value) if value else None

    @property
    def index_model(self) -> str | None:
        return get_meta(self.conn, "embedding_model")

    def require_embedder(self) -> Embedder:
        if not self.embedder:
            raise EmbeddingError("未配置 embedding 服务（R4P_EMBEDDING_API_KEY）")
        return self.embedder

    def require_llm(self) -> LLM:
        if not self.llm:
            raise LLMError("未配置大模型服务（R4P_LLM_API_KEY）")
        return self.llm

    def maybe_promote(self) -> bool:
        """启动时若因维度未知回退到 numpy，首次入库拿到维度后迁回 Milvus。

        避免"先写进 numpy、下次启动换成 Milvus 却查不到数据"的静默不一致。
        """
        if not self.fell_back or self.settings.vector_backend != "auto":
            return False
        dim = (self.embedder.dim if self.embedder else 0) or self.index_dim or 0
        if not dim:
            return False
        try:
            milvus = MilvusStore(self.settings.milvus_uri, self.settings.milvus_collection, dim)
        except BackendUnavailable as exc:
            self.notes.append(f"仍无法启用 Milvus：{exc}")
            return False

        from .ingest import rebuild_store  # 局部导入，避免循环依赖

        moved = rebuild_store(self.conn, milvus)
        self.store = milvus
        self.fell_back = False
        self.notes = [n for n in self.notes if "Milvus 不可用" not in n]
        self.notes.append(f"已把 {moved} 条归档向量迁移到 Milvus")
        return True

    def health(self) -> dict:
        state = inspect(self.conn)
        issues = health_issues(
            self.conn, self.embedder.model if self.embedder else None, self.store.dim
        )
        return {
            # 索引与配置对不上时不能报"健康"：那种状态能跑，但检索结果不可信
            "ok": bool(self.embedder and self.llm) and not issues,
            "backend": {
                "name": self.store.name,
                "vectors": self.store.count(),
                "dim": self.store.dim,
                "fell_back": self.fell_back,
            },
            "index": {
                "model": state.model,
                "dim": state.dim,
                "vectors": state.count,
                # vectors 与 chunks 数量对不上，说明索引落后于事实（重建没走完）
                "chunks": chunk_count(self.conn),
                "consistent": not issues,
                "issues": issues,
            },
            "credentials_missing": self.settings.missing_credentials(),
            # 后端可能把恢复动作（如"已从上一版集合还原索引"）记在 notes 里；
            # 用 getattr 是为了让 health 在任何 VectorStore 实现上都不会因为缺属性而 500
            "notes": [*self.notes, *getattr(self.store, "notes", [])],
        }

    def close(self) -> None:
        self.conn.close()


def open_store(settings: Settings, conn: sqlite3.Connection) -> tuple[VectorStore, list[str], bool]:
    notes: list[str] = []
    dim = settings.embedding_dim or int(get_meta(conn, "embedding_dim") or 0)
    if settings.vector_backend in ("auto", "milvus"):
        try:
            return MilvusStore(settings.milvus_uri, settings.milvus_collection, dim), notes, False
        except BackendUnavailable as exc:
            if settings.vector_backend == "milvus":
                raise BackendUnavailable(
                    f"{exc}。当前显式指定了 R4P_VECTOR_BACKEND=milvus，不会自动回退。"
                    "处理办法二选一：① 把 R4P_EMBEDDING_DIM 填成模型的真实维度"
                    "（如 gte-base-chinese 为 768）；② 改用 R4P_VECTOR_BACKEND=auto。"
                ) from exc
            notes.append(f"Milvus 不可用，已回退到 numpy 精确检索：{exc}")
            return NumpyStore(conn, dim), notes, True
    return NumpyStore(conn, dim), notes, False


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)

    store, notes, fell_back = open_store(settings, conn)

    # 入库任务跑在进程内，上次进程死掉留下的 running/queued 永远不会前进
    interrupted = mark_interrupted_jobs(conn)
    if interrupted:
        notes.append(f"已把上次进程遗留的 {interrupted} 个任务标记为失败")

    embedder: Embedder | None
    try:
        embedder = Embedder(settings, conn)
    except EmbeddingError as exc:
        embedder = None
        notes.append(str(exc))

    llm: LLM | None
    try:
        llm = LLM(settings)
    except LLMError as exc:
        llm = None
        notes.append(str(exc))

    reranker: Reranker = RemoteReranker(settings) if settings.rerank_url else NoopReranker()

    return Services(
        settings=settings,
        conn=conn,
        store=store,
        embedder=embedder,
        llm=llm,
        reranker=reranker,
        notes=notes,
        fell_back=fell_back,
    )


def default_upload_dir(settings: Settings) -> Path:
    return settings.upload_dir
