"""组件装配：所有依赖在这里创建一次，通过显式注入往下传。

没有模块级单例、没有可变全局状态——除了 SQLite 连接（单进程、WAL、串行写）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field, replace
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
from .consistency import collect_issues, inspect
from .credentials import CredentialPool, UserCredentials
from .db import connect, init_db
from .repo import get_meta, mark_interrupted_documents, mark_interrupted_jobs


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
    # 多用户自带 Key：池子按指纹复用客户端，视图共享同一个池
    credentials: CredentialPool | None = None
    user_label: str | None = None
    # 视图不拥有连接：一次请求结束若能关掉全局 SQLite 连接，下一个请求就全崩
    owns_conn: bool = True

    def __post_init__(self) -> None:
        if self.credentials is None:
            self.credentials = CredentialPool(self.settings, self.conn)

    def for_credentials(self, creds: UserCredentials | None) -> Services:
        """换上某个用户 Key 之后的视图，共享 conn / store / reranker。

        没有凭据时原样返回 `self`，未登录访问继续用 `.env` 里的公共 Key——
        CLI、测试和已有部署的行为完全不变。
        """
        if creds is None or self.credentials is None:
            return self
        embedder, llm = self.credentials.acquire(creds)
        return replace(
            self,
            settings=self.credentials.settings_for(creds),
            embedder=embedder,
            llm=llm,
            notes=list(self.notes),
            user_label=creds.username,
            owns_conn=False,
        )

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
        # 后端条数是"索引 vs 归档"交叉校验的一半，读不到就如实说读不到，
        # 不要拿 None 当 0 —— 那会把"查不了"伪装成"索引是空的"。
        store_count: int | None
        store_error: str | None = None
        try:
            store_count = self.store.count()
        except Exception as exc:  # pragma: no cover - 后端不可用时才会走到
            store_count = None
            store_error = f"{type(exc).__name__}: {exc}"

        state = inspect(self.conn, store_count=store_count)
        issues = collect_issues(
            state, self.embedder.model if self.embedder else None, self.store.dim
        )
        if store_error:
            issues.append(f"读不到向量索引的实际条数，无法完成交叉校验：{store_error}")
        return {
            # 索引与配置对不上时不能报"健康"：那种状态能跑，但检索结果不可信
            "ok": bool(self.embedder and self.llm) and not issues,
            "backend": {
                "name": self.store.name,
                "vectors": store_count,
                "dim": self.store.dim,
                "fell_back": self.fell_back,
            },
            "index": {
                "model": state.model,
                "dim": state.dim,
                # 归档（事实）：片段的权威向量副本
                "vectors": state.count,
                "chunks": state.chunks,
                # 三个差额分开报，比"总数对不上"更能定位问题
                "missing_vectors": state.missing,
                "orphan_vectors": state.orphan,
                "unlabeled_vectors": state.unlabeled,
                "consistent": not issues,
                "issues": issues,
            },
            "credentials_missing": self.settings.missing_credentials(),
            # 前端要拿它提示"你自己的 Key 必须属于同一家服务商"——模型由服务端统一决定
            "models": {
                "embedding_model": self.settings.embedding_model,
                "embedding_dim": self.settings.embedding_dim or self.store.dim,
                "llm_model": self.settings.llm_model,
            },
            # 让前端能证明"请求头确实被认到了"，而不是以为自己配了其实没用上
            "user": {
                "label": self.user_label or "服务端公共 Key",
                "source": "user" if self.user_label else "server",
            },
            # 后端可能把恢复动作（如"已从上一版集合还原索引"）记在 notes 里；
            # 用 getattr 是为了让 health 在任何 VectorStore 实现上都不会因为缺属性而 500
            "notes": [*self.notes, *getattr(self.store, "notes", [])],
        }

    def close(self) -> None:
        # 视图（owns_conn=False）可能在一堆请求之间共享，关它等于把全局连接掐了
        if not self.owns_conn:
            return
        if self.credentials is not None:
            self.credentials.close()
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
    # 同一类问题在文档维度也要收尾，否则它会永远显示"正在入库"
    dangling = mark_interrupted_documents(conn)
    if dangling:
        notes.append(f"已把上次进程中断时处于入库中的 {dangling} 个文档标记为失败，可重新导入")

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
