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
from .concurrency import MaintenanceBusy, maintenance_lock, worker_alive
from .config import Settings, get_settings
from .consistency import collect_issues, inspect
from .credentials import CredentialPool, UserCredentials
from .db import connect, init_db
from .repo import get_meta, mark_interrupted_documents, mark_interrupted_jobs


def _describe(exc: BaseException) -> str:
    """异常摘要。有些异常（裸 `RuntimeError()`）没有消息，那就只留类型名，
    否则 notes 里会出现一条以冒号结尾的空话。"""
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


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
    # 请求级视图指回根服务。后端指针（store / fell_back）只能由根服务改：
    # 视图是 `replace()` 出来的临时对象，请求结束就扔，在它上面迁移等于"迁移成功、
    # 根服务还在用旧后端"。见 `owner` 与 `maybe_promote`。
    root: Services | None = None

    def __post_init__(self) -> None:
        if self.credentials is None:
            self.credentials = CredentialPool(self.settings, self.conn)

    @property
    def owner(self) -> Services:
        """后端指针的所有者。视图返回根服务，根服务返回自己。"""
        return self.root if self.root is not None else self

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
            root=self.owner,
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

        三件事必须一起成立，历史上各错一次：

        1. **只能由根服务执行**。请求级视图是 `replace()` 出来的临时对象，在它上面
           迁移会让"迁移成功、根服务还在用旧后端"，而且根服务的 `fell_back` 仍是
           True，下一个请求再迁一遍；
        2. **整个过程持有维护锁**，和入库、重建互斥——迁移要把归档搬进新后端，
           期间又有人写入，新后端拿到的就是过期快照；
        3. **异常一律截住**。上传接口是在 `ingest_batch` 之后调它的，一次没接住的
           异常会把已经 done 的任务改写成 failed，数据其实已经写好。
        """
        owner = self.owner
        if owner is not self:
            return owner.maybe_promote()
        if not self.fell_back or self.settings.vector_backend != "auto":
            return False
        try:
            with maintenance_lock(self.conn, purpose="迁移到 Milvus", timeout=0.0):
                return self._promote()
        except MaintenanceBusy as exc:
            # 有别的动作在改索引时不迁移：这次不是"失败"，是让路。下次入库还会再来。
            self.notes.append(f"暂不迁移到 Milvus：{exc}")
            return False

    def _promote(self) -> bool:
        dim = (self.embedder.dim if self.embedder else 0) or self.index_dim or 0
        if not dim:
            return False
        from .ingest import rebuild_store  # 局部导入，避免循环依赖

        try:
            milvus = MilvusStore(self.settings.milvus_uri, self.settings.milvus_collection, dim)
        except Exception as exc:  # noqa: BLE001 - 见方法说明：可选升级不能外溢
            # 以前只认 BackendUnavailable，但 pymilvus 在 Lite 文件被占用、版本不匹配时
            # 抛的是它自己的异常，那类失败同样只该降级、不该中断调用方。
            self.notes.append(f"仍无法启用 Milvus：{_describe(exc)}")
            return False

        try:
            moved = rebuild_store(self.conn, milvus)
        except Exception as exc:  # noqa: BLE001 - 同上
            # 迁移到 Milvus 只是"换个更快的后端"，是可选项。归档本身不合格、或 SDK 在
            # 建集合/写向量时报错，都留在原后端继续用，把原因记进 notes。
            # rebuild_store 只在校验通过后才动归档，所以这里不需要回收任何东西。
            self.notes.append(
                f"暂不迁移到 Milvus（继续用 {self.store.name} 检索）：{_describe(exc)}"
            )
            return False
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
                hint = (
                    f"{exc}。当前显式指定了 R4P_VECTOR_BACKEND=milvus，不会自动回退："
                    "按上面的报错处理，或改成 auto / numpy 先把服务跑起来。"
                )
                # "维度未知"只是 Milvus 打不开的众多原因之一。路径不合法、缺 pymilvus、
                # 中文路径，报的也是这一条——一律往"改维度"上引，会让人在不相关的
                # 配置项上反复试，越试越远。
                if "维度" in str(exc):
                    hint += "（报的是维度未知时：把 R4P_EMBEDDING_DIM 填成模型的真实维度。）"
                raise BackendUnavailable(hint) from exc
            notes.append(f"Milvus 不可用，已回退到 numpy 精确检索：{exc}")
            return NumpyStore(conn, dim), notes, True
    return NumpyStore(conn, dim), notes, False


def build_services(
    settings: Settings | None = None, *, recover_interrupted: bool | None = None
) -> Services:
    """装配服务。

    `recover_interrupted` 决定要不要把库里遗留的 `running` / `queued` 任务标成中断。
    默认值是"没别人在跑才标"：这个库现在同时被 uvicorn 和 CLI 使用，光看数据库
    分不出"上一个进程已经死了"和"另一个进程正在写"。以前每次装配都无条件清理，
    于是跑一次 `doctor` 就能把在线入库任务标成 failed，而数据其实还在写。

    判活的依据是 `concurrency.WORKER_LEASE`——服务进程活着就一直占着的文件锁，
    进程退出由内核释放，不需要心跳，也不会留下残留。
    """
    settings = settings or get_settings()
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)

    store, notes, fell_back = open_store(settings, conn)

    if recover_interrupted is None:
        recover_interrupted = not worker_alive(conn)
    if recover_interrupted:
        # 入库任务跑在进程内，上次进程死掉留下的 running/queued 永远不会前进
        interrupted = mark_interrupted_jobs(conn)
        if interrupted:
            notes.append(f"已把上次进程遗留的 {interrupted} 个任务标记为失败")
        # 同一类问题在文档维度也要收尾，否则它会永远显示"正在入库"
        dangling = mark_interrupted_documents(conn)
        if dangling:
            notes.append(
                f"已把上次进程中断时处于入库中的 {dangling} 个文档标记为失败，可重新导入"
            )
    else:
        # 说清"为什么这次不动它们"，否则运维会以为清理逻辑失效了
        notes.append(
            "检测到另一个服务进程正在使用同一个数据库，已跳过对未完成任务的中断清理"
            "（那些任务可能正在别的进程里跑）"
        )

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
