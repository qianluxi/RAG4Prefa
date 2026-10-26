"""向量索引：可替换后端。

约定——向量库只是**派生索引**，SQLite 里的 chunks 才是事实。
因此这里所有方法都是幂等的，任何时刻都可以从 SQLite 重建。
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from array import array
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Protocol

import numpy as np

from .. import repo
from ..concurrency import MaintenanceBusy, maintenance_lock, sqlite_write
from ..consistency import IndexMismatch
from ..domain import Filter


class BackendUnavailable(RuntimeError):
    pass


class RebuildError(RuntimeError):
    """重建索引失败。错误信息里必须写清"现在处于什么状态、怎么恢复"。"""


NON_ASCII_PATH_HINT = (
    "Milvus Lite 在 Windows 上无法在含非 ASCII 字符的路径下建立索引："
    "faiss 用窄字符 fopen 打开索引文件，中文目录名会直接失败；"
    "一旦触发 flush / release+load（重建、换维度、重启后重开都算），集合就再也加载不起来。"
    "两条出路：① 把 R4P_DATA_DIR（或 R4P_MILVUS_URI）指到纯 ASCII 路径，"
    "例如 C:\\rag4prefa-data；② 用 R4P_VECTOR_BACKEND=numpy，"
    "十万级片段以内它做精确余弦检索，结果与 Milvus 完全一致，只是 O(n) 扫描。"
)


def _normalize_uri(uri: str) -> str:
    """把 Milvus 地址归一成可以"比字面量"的形式。

    两种形态要分开处理：Lite 是本地文件路径（`./data/vectors.db` 与 `data/vectors.db`
    指的是同一个文件，只有绝对路径能比），standalone 是 URL（对 URL 做 `abspath` 会把
    它改成一个本地路径，两个真 URL 反而可能"看起来相同"）。
    """
    text = str(uri).strip()
    if "://" in text:
        return text.rstrip("/").lower()
    return os.path.abspath(text)


def _reject_non_ascii_uri(uri: str) -> None:
    """在开集合之前就拒绝，别让人在建索引那一刻才收到一条看不懂的 faiss 报错。

    这条限制是实测出来的，不是猜的：同一份数据放 `C:/tmp/...` 一切正常，
    放到 `F:/project/课题/...` 下 rebuild 立即失败。

    必须按**绝对路径**判断。默认配置是相对路径 `data/vectors.db`，字面量干净，
    真正落盘的位置却由工作目录决定；只看字面量的话，工程放在中文目录下时这道检查
    形同虚设，要等 MilvusClient(uri) 打开才报 "Open local milvus failed"——
    那条报错既不提 faiss 也不提中文，根因完全看不出来。
    """
    if os.name != "nt":
        return
    absolute = os.path.abspath(str(uri))
    if any(ord(ch) > 127 for ch in absolute):
        raise BackendUnavailable(f"{NON_ASCII_PATH_HINT}（实际路径：{absolute}）")


@dataclass(slots=True)
class VectorItem:
    chunk_id: str
    vector: list[float]
    project_name: str
    project_code: str
    doc_year: int | None
    model: str = ""  # 产生该向量的模型；numpy 后端的归档表也靠它记模型


class VectorStore(Protocol):
    name: str
    dim: int
    notes: list[str]
    # 身份 = 类型 + **实际存储位置**。`name` 只说类型：两个进程共用一份 SQLite、却各自
    # 配了不同的 Milvus 地址或集合名时，两边都叫 "milvus"。
    identity: str

    def probe(self) -> None: ...
    def upsert(self, items: Sequence[VectorItem]) -> None: ...
    def delete(self, chunk_ids: Sequence[str]) -> None: ...
    def search(
        self, vector: list[float], *, limit: int, filt: Filter | None = None
    ) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...
    def ids(self) -> set[str]: ...
    def clear(self) -> None: ...
    def rebuild(self, items: Sequence[VectorItem]) -> None: ...


def _normalize(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm else arr


@dataclass(frozen=True)
class _Snapshot:
    """一次加载得到的完整索引视图。

    为什么整体换而不是逐个字段赋值：`search` 要同时用到 ids、矩阵和筛选列，
    分几次赋值时读者可能拿到"上一版的矩阵 + 这一版的 ids"，索引越界或张冠李戴。
    Python 里给一个属性赋值是原子的，所以把这几样捆成一个不可变对象，
    读者取一次引用就拿到一致的一套。
    """

    ids: tuple[str, ...] = ()
    projects: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()
    years: tuple[int | None, ...] = ()
    mat: np.ndarray = dc_field(default_factory=lambda: np.zeros((0, 1), dtype=np.float32))


_ARCHIVE_SQL = (
    "SELECT chunk_id, project_name, project_code, doc_year, dim, vec FROM vectors"
)

# 加载期间归档被反复改动时的重试次数。取 5 次是因为正常情况第一次就成功：
# 本进程自己写完就已经是唯一写者，能撞上的只有"另一个进程恰好在提交"这种短窗口。
_SNAPSHOT_ATTEMPTS = 5

# "布局判断完了、真正打开之前布局又变了"的重试次数。一次重建里有两次 rename，
# 撞上的窗口很短；给它三次机会，之后就走持锁路径或推迟打开，不再猜。
_OPEN_ATTEMPTS = 3


class SnapshotUnstable(IndexMismatch):
    """归档在加载过程中被反复改动，拿不到一份能自称"最新"的快照。

    继承 `IndexMismatch` 而不是自己造一个异常族：对调用方来说这和"索引不能自证一致"
    是同一类事——既不能拿它回答问题，也不能在它之上继续写入，只能等人重试。
    """


def _snapshot_of(
    rows: Sequence[sqlite3.Row], fallback_dim: int
) -> tuple[_Snapshot, int, list[str]]:
    """把归档行组成一份快照。纯函数（不碰 self），因为重试时要能丢弃整份重来。"""
    notes: list[str] = []

    # dim 字段是声明，float32 字节数才是事实。两者对不上的行组不进矩阵，
    # 硬拼的话 vstack 会抛一句既不说是哪行、也不说怎么办的 dimensions must match。
    bad = {
        r["chunk_id"]
        for r in rows
        if int(r["dim"] or 0) <= 0 or len(r["vec"]) != int(r["dim"]) * 4
    }
    usable = [r for r in rows if r["chunk_id"] not in bad]
    if not usable:
        if rows:
            notes.append(
                f"归档里 {len(rows)} 条向量的字节长度与声明维度都不符，索引为空；"
                "执行 `python -m rag4prefa reindex` 从原文重新生成向量"
            )
        empty = _Snapshot(mat=np.zeros((0, fallback_dim or 1), dtype=np.float32))
        return empty, fallback_dim, notes

    # 声明维度本身不一致（历史库混过多套）时，只能取条数最多的那一套：
    # 剩下的照样跳过并如实说明，不假装它们在同一套向量空间里。
    counts = Counter(int(r["dim"]) for r in usable)
    dim = counts.most_common(1)[0][0]
    kept = [r for r in usable if int(r["dim"]) == dim]
    skipped = len(rows) - len(kept)
    if skipped:
        notes.append(
            f"归档里有 {skipped} 条向量与主流维度 {dim} 不符（或字节长度与声明维度不符），"
            "已跳过；索引因此少于归档，检索会被拒绝，执行 `python -m rag4prefa reindex` 修复"
        )

    snap = _Snapshot(
        ids=tuple(r["chunk_id"] for r in kept),
        projects=tuple(r["project_name"] for r in kept),
        codes=tuple(r["project_code"] for r in kept),
        years=tuple(r["doc_year"] for r in kept),
        mat=np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in kept]),
    )
    return snap, dim, notes


class NumpyStore:
    """精确余弦检索（暴力遍历）。

    优点：零外部依赖、Windows 可跑、结果可复现（无近似召回误差）；
    代价：O(n) 扫描。十万级 chunk 以内延迟仍在毫秒级，超出请切 Milvus。

    与 Milvus 最大的结构差别：这里的索引**就是内存里的一份拷贝**，而 Milvus 是
    独立存储。所以这个后端额外要管两件 Milvus 不需要管的事：

    1. 同进程多线程写时，内存矩阵和 `vectors` 表可能短暂脱节（`_load` 期间）；
    2. 别的进程（CLI reindex/rebuild）改了归档，本进程的内存矩阵不会自己变新。
       靠 `meta.index_generation` 发现，见 `_refresh_if_stale`。
    """

    name = "numpy"

    @property
    def identity(self) -> str:
        """numpy 后端的"存储位置"就是这份 SQLite 本身。

        共用同一个库文件的进程必然指向同一份向量，所以这里没有第二个位置可填——
        写成常量是**结论**，不是省事：它让"两个进程打开的是不是同一份索引"这个问题
        在 numpy 上永远只有一个答案。
        """
        return "numpy"

    def probe(self) -> None:
        """numpy 后端没有"打开"这一步：索引就是 `vectors` 表的缓存，构造完就可用。

        归档暂时读不稳（`SnapshotUnstable`）时也**不在这里拒绝**：那是"别人正在写"，
        下一次 `count()`/`search()` 会自己重载。把它当成"后端不可用"会让一次正常的
        并发写入变成需要重启的故障。
        """
        return None

    def __init__(self, conn: sqlite3.Connection, dim: int) -> None:
        self._conn = conn
        self.dim = dim
        self.notes: list[str] = []
        self._revision: tuple[int, int] = (-1, -1)
        self._snap = _Snapshot()
        try:
            self._load()
        except SnapshotUnstable as exc:
            # 构造阶段不因为"别人正好在写"就把服务挡在门外：留空快照、**不记代次**。
            # 代次停在 (-1, -1) 与真实值必然不同，下一次 count/search 会再试一轮。
            self.notes.append(f"暂时没有可用的内存索引：{exc}")

    # ── 内部 ────────────────────────────────────────

    @property
    def _ids(self) -> list[str]:
        """兼容旧读法（测试里按条数构造假 id 集合）。"""
        return list(self._snap.ids)

    @property
    def _mat(self) -> np.ndarray:
        return self._snap.mat

    def _load(self) -> None:
        """按归档重建内存矩阵，**并保证这份快照与它记下的代次是同一版**。

        **不在构造阶段拒绝坏数据**：这个后端在 `build_services()` 里就被创建，
        构造即抛错意味着归档一旦损坏，服务起不来、`reindex` 也进不去——而那条命令
        正是唯一的修复入口（它按正文重算向量、整表重写，根本不需要读旧索引）。
        所以这里只跳过组不成的行、把原因记进 notes，让"能不能检索"由
        `consistency` 统一判定：跳过的行会让后端条数少于归档，那就是阻断项。

        **但"读到旧行、记下新代次"必须杜绝**（2026-10-24 第十四轮复审）。原来的顺序是
        "读行 → 组矩阵 → 读代次"，中间那段别的进程可以提交一整次重建：本进程拿着旧行、
        记上新代次，此后 `_refresh_if_stale` 认为缓存已是最新，再也不刷新。表现是
        "条数、模型、维度全对得上，检索结果却是上一版内容"——一致性检查一条也挑不出来，
        因为差异在内容里，不在计数里。

        所以改成 seqlock 式的双读：**读行之前取一次代次、读完再取一次，两次相同才发布**。
        不同就整份丢掉重来（`_snapshot_of` 是纯函数，丢弃无副作用）。
        """
        for attempt in range(_SNAPSHOT_ATTEMPTS):
            before = repo.get_store_revision(self._conn)
            rows = self._conn.execute(_ARCHIVE_SQL).fetchall()
            # 组快照本身也在"读代次"之前完成：这段计算同样属于"我需要它一致"的窗口，
            # 把它放到比较之后，等于承认"比较完之后再变就不算变"。
            snap, dim, notes = _snapshot_of(rows, self.dim)
            after = repo.get_store_revision(self._conn)
            if before == after:
                self._snap, self.dim = snap, dim
                # 反复重载时不要把同一条说明叠成一长串
                self.notes = list(dict.fromkeys([*self.notes, *notes]))
                self._revision = after
                return
            time.sleep(0.005 * (attempt + 1))
        raise SnapshotUnstable(
            f"读取归档期间索引被改动 {_SNAPSHOT_ATTEMPTS} 次，拿不到一份能自称最新的快照；"
            "本次不使用内存索引。等写入结束后重试即可——"
            "若持续出现，说明有进程在反复重建，先停掉它。"
        )

    def _refresh_if_stale(self) -> None:
        """归档被别的进程改过（或本进程换过一代）时，重新加载内存矩阵。

        为什么一定要有这一步：维护锁只保证"同一时刻只有一个人在写"，写完就把锁放开了。
        CLI 跑一次 `reindex` 之后，API 进程里的矩阵还是旧的，而一致性检查看的是
        SQLite 里的条数与模型——两边可以完全对得上（重建换了模型但片段数没变、
        改一版文档换了几个 chunk），于是"检查通过、检索用的是旧内容"。

        代次是只增不减的计数器，不承担任何语义，只回答"我看的这一版还是最新的吗"。
        换索引和换内容推的是两个不同的计数器，这里都看——缓存过期就是过期，
        至于"要不要因此否决一次在途查询"是 `ask` 的判断，不是缓存的。
        """
        if repo.get_store_revision(self._conn) == self._revision:
            return
        with sqlite_write():
            if repo.get_store_revision(self._conn) == self._revision:
                return
            self._load()

    # ── 对外 ────────────────────────────────────────

    def upsert(self, items: Sequence[VectorItem]) -> None:
        """一批向量一次性写入。

        **先全部校验、再一次性提交**：旧实现逐条 `INSERT OR REPLACE`，校验也在循环里，
        于是第 2 条不合格时第 1 条已经写进事务了。这个事务不属于任何显式边界，随后
        任何一次 `commit()`——比如另一个线程更新任务进度——都会把它一起提交。结果是
        "入库失败、旧向量却已经被新版覆盖"，而回滚逻辑只会删掉本次新增的 id，
        删不掉被覆盖的那几行。
        """
        if not items:
            return
        dim = self.dim or len(items[0].vector)
        rows: list[tuple] = []
        for item in items:
            arr = _normalize(item.vector)
            if arr.shape[0] != dim:
                raise ValueError(f"向量维度 {arr.shape[0]} 与索引维度 {dim} 不一致")
            rows.append(
                (
                    item.chunk_id,
                    item.project_name,
                    item.project_code,
                    item.doc_year,
                    dim,
                    item.model,
                    array("f", arr.tolist()).tobytes(),
                )
            )
        with sqlite_write():
            with self._conn:  # 整个批次要么全进、要么全不进
                self._conn.executemany(
                    "INSERT OR REPLACE INTO vectors (chunk_id, project_name, project_code,"
                    " doc_year, dim, model, vec) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            self.dim = dim
            self._load()

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        marks = ",".join("?" * len(chunk_ids))
        with sqlite_write():
            self._conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({marks})", list(chunk_ids))
            self._conn.commit()
            self._load()

    def search(
        self, vector: list[float], *, limit: int, filt: Filter | None = None
    ) -> list[tuple[str, float]]:
        self._refresh_if_stale()
        snap = self._snap  # 取一次引用：矩阵、ids、筛选列必须是同一版
        if not snap.ids:
            return []
        query = _normalize(vector)
        if query.shape[0] != snap.mat.shape[1]:
            raise ValueError(f"查询维度 {query.shape[0]} 与索引维度 {snap.mat.shape[1]} 不一致")
        scores = snap.mat @ query

        mask = np.ones(len(snap.ids), dtype=bool)
        if filt:
            if filt.projects:
                wanted = set(filt.projects)
                mask &= np.array([p in wanted for p in snap.projects], dtype=bool)
            if filt.year_from is not None:
                mask &= np.array([y is not None and y >= filt.year_from for y in snap.years])
            if filt.year_to is not None:
                mask &= np.array([y is not None and y <= filt.year_to for y in snap.years])
        if not mask.any():
            return []

        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:limit]
        return [(snap.ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def count(self) -> int:
        self._refresh_if_stale()
        return len(self._snap.ids)

    def ids(self) -> set[str]:
        self._refresh_if_stale()
        return set(self._snap.ids)

    def clear(self) -> None:
        with sqlite_write():
            self._conn.execute("DELETE FROM vectors")
            self._conn.commit()
            self._load()

    def rebuild(self, items: Sequence[VectorItem]) -> None:
        """整体替换全部向量，允许维度变化（换模型后用）。

        与 clear() 的区别：clear() 是"清空"，rebuild() 是"换成另一套"。
        先构造并校验全部行，再在一个事务里删旧写新，中途失败不会留下半套数据。
        """
        rows: list[tuple] = []
        dim = len(items[0].vector) if items else self.dim
        for item in items:
            arr = _normalize(item.vector)
            if arr.shape[0] != dim:
                raise ValueError(f"向量维度 {arr.shape[0]} 与本次重建维度 {dim} 不一致")
            rows.append(
                (
                    item.chunk_id,
                    item.project_name,
                    item.project_code,
                    item.doc_year,
                    dim,
                    item.model,
                    array("f", arr.tolist()).tobytes(),
                )
            )
        with sqlite_write():
            with self._conn:
                self._conn.execute("DELETE FROM vectors")
                if rows:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO vectors (chunk_id, project_name, project_code,"
                        " doc_year, dim, model, vec) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        rows,
                    )
            self.dim = dim
            self._load()


class StoreRef:
    """指向"当前后端"的引用，而不是某个具体后端对象。

    为什么需要这一层：入批处理任务在开始时就拿到了 `services.store`，而那个指针可能
    在批次执行期间被迁移换掉（numpy → Milvus）。把对象本身传下去，任务就会一直往
    旧后端写：根服务已经查 Milvus，归档也在 SQLite 里更新了，只有新数据躺在没人读的
    numpy 表里。维护锁能保证"写的时候没有别人在迁移"，但锁不住"手里这个对象是不是
    迁移之前的那一个"——那是取参数那一刻就定死的。

    每次访问都重新解析，指针变了这里就跟上。
    """

    def __init__(self, resolve) -> None:
        self._resolve = resolve

    def _target(self):
        return self._resolve()

    def __getattr__(self, name: str):
        return getattr(self._target(), name)

    def __repr__(self) -> str:  # pragma: no cover - 只在调试输出里出现
        try:
            return f"StoreRef({self._target()!r})"
        except Exception:
            return "StoreRef(<无法解析>)"


class MilvusStore:
    """Milvus（Lite 或 standalone，同一个 pymilvus 客户端，只改 URI）。

    注意：Milvus Lite 不支持 HNSW 参数，统一用 AUTOINDEX；
    切到 standalone 时把 index_type 换成 HNSW 才需要调参。

    集合命名约定（用于崩溃恢复，见 `_recover`）：
    - `<collection>`            正式索引，检索只读它；
    - `<collection>__staging`   重建时的暂存集合，写完并校验后才切换；
    - `<collection>__retired`   被换下来的上一版，切换成功后立即删除。

    **构造与"打开"是两件事**（2026-10-24 第十四轮复审）。构造只连客户端；收拾残留、
    建集合、load 归 `_open()`。原因是 `_recover()` 会**删掉** `__staging`，而正在重建
    的进程恰恰把半成品放在那里：构造函数是每个进程启动都走的路，又发生在任何维护锁
    之外，它在别人重建到一半时删一次，那笔重建就白跑了，而且两边条数、模型、维度
    全都能对得上，事后查不出来。
    """

    name = "milvus"

    def __init__(
        self, uri: str, collection: str, dim: int, *, conn: sqlite3.Connection | None = None
    ) -> None:
        """`conn` 用来定位维护锁。传了它，"打开"就被纳入跨进程互斥；
        不传（测试、独立脚本）说明没有共享库，也就不存在"别人正在重建"。
        """
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise BackendUnavailable("未安装 pymilvus：pip install 'rag4prefa[milvus]'") from exc

        _reject_non_ascii_uri(os.path.join(str(uri), collection))
        self._collection = collection
        self._uri = str(uri)
        self.dim = dim
        self.notes: list[str] = []
        self._ready = False
        self._deferred: str | None = None
        try:
            self._client = MilvusClient(uri=uri)
        except Exception as exc:  # pragma: no cover - 平台相关
            raise BackendUnavailable(f"无法打开 Milvus 存储 {uri}: {exc}") from exc

        self._open(conn)

    @property
    def identity(self) -> str:
        """类型 + 实际存储位置。同名不同集合时，只有它能分出两个进程在不在同一份索引上。"""
        return f"milvus|{_normalize_uri(self._uri)}|{self._collection}"

    def probe(self) -> None:
        """证明"这个后端现在真的能回答问题"，而不只是"对象构造出来了"。

        为什么需要（2026-10-24 第十七轮复审）：`_open()` 在发现重建残留却抢不到维护锁
        时会 `_defer()`，留下一个 `_ready=False` 的对象。它的 `name` 照样是 "milvus"，
        于是"库里登记的也是 milvus"这条快速路径就把它认成已确认——`/api/health` 报正常，
        直到某次查询在 `search()` 里才炸出来，而那里通常只处理 `IndexMismatch`。
        """
        self._require_ready()
        # 只 load 过不等于查得动：集合可能已被别的进程换掉。真读一次条数才算数。
        self._count_of(self._collection)

    # ── 打开（有副作用，必须由锁来划边界）────────────────

    def _layout(self) -> tuple[bool, bool, bool]:
        """只读地看一眼三个集合在不在，不做任何改动。"""
        return (
            self._client.has_collection(self._collection),
            self._client.has_collection(self._retired),
            self._client.has_collection(self._staging),
        )

    def _open(self, conn: sqlite3.Connection | None) -> None:
        if conn is None:
            self._open_all()
            return

        for attempt in range(_OPEN_ATTEMPTS):
            has_final, has_retired, has_staging = self._layout()
            if has_final and not has_retired and not has_staging:
                # 干净状态：建集合（已存在就跳过）与 load 都不删东西，因此不需要独占。
                # 这条路径必须是无锁的——启动不能因为"另一个进程正在入库"而卡几分钟。
                if self._open_clean():
                    return
                # 判断之后布局变了：有进程正在两次 rename 之间切换索引。重看一次布局，
                # 这时通常已经能看到 __staging / __retired，落到下面的持锁路径。
                time.sleep(0.05 * (attempt + 1))
                continue
            if not (has_final or has_retired or has_staging):
                # 全新库：建集合是写操作，两个进程同时启动会撞在一起。这里没有"别人的
                # 半成品"可破坏，排队等待是安全的（秒级）。
                with maintenance_lock(
                    conn, purpose=f"创建向量集合 {self._collection}", timeout=None
                ):
                    self._open_all()
                return
            try:
                with maintenance_lock(
                    conn, purpose=f"打开向量索引 {self._collection}", timeout=0.0
                ):
                    self._open_all()
            except MaintenanceBusy as exc:
                # 有残留却拿不到锁 ⇒ 那些残留很可能正是别人**正在用**的（重建中的暂存
                # 集合）。这时一个集合都不许碰，把索引标成未打开：宁可这次查不了，
                # 也不能删掉别人重建到一半的成果。
                self._defer(str(exc))
            return

        self._defer("集合布局在打开过程中反复变化（另一个进程正在切换索引）")

    def _open_all(self) -> None:
        """收拾残留 + 建集合 + load。调用方必须已持有维护锁（或明确没有共享库）。"""
        self._recover()
        self._ensure_collection(self.dim)
        self._load_collection()
        self._ready = True

    def _open_clean(self) -> bool:
        """干净布局下的无锁打开：**只加载此刻仍然存在的正式集合，绝不创建**。

        为什么要把"看过布局"和"能建集合"拆开（2026-10-24 第十五轮复审）：
        `_open()` 看到"正式集合在、没有残留"之后，到真正 load 之间隔着一段时间。另一个
        进程可能正好在这段时间里重建——它把正式集合 rename 成 `__retired`、再把
        `__staging` 换成正式名。旧实现这时调 `_ensure_collection`，发现"正式集合不
        存在"，于是**建了一个空集合**：那个进程的第二次 rename 撞名失败，而它的回滚
        条件（`__retired` 在、正式集合不在）已经不成立，旧索引被留在 `__retired` 里。
        一次重建白跑，还得人工恢复——而两边条数、模型、维度全都能对得上。

        无锁路径能做的事只有一件：加载一个此刻仍然存在的正式集合。它没有锁，
        就没有资格创建、删除、重命名任何集合。返回 False 表示"布局在打开过程中变了"，
        由调用方重看布局或改走持锁路径。
        """
        if not self._client.has_collection(self._collection):
            return False
        try:
            self._adopt_existing(self.dim)
            self._load_collection()
        except BackendUnavailable:
            # rename 是原子元数据操作，失败通常就是"它已经不在这个名字下了"。
            # 别的错误（faiss 路径、权限）不能咽掉，那类问题重试也还是错。
            if not self._client.has_collection(self._collection):
                return False
            raise
        self._ready = True
        return True

    def _defer(self, reason: str) -> None:
        self._deferred = (
            f"另一个维护动作正在进行（{reason}），本进程没有打开向量索引——"
            "宁可暂时查不了，也不能动到它正在重建的集合。等它结束后重启服务即可；"
            "若确认那个进程已不存在，直接重启。"
        )
        self.notes.append(self._deferred)

    def _require_ready(self) -> None:
        if not self._ready:
            raise BackendUnavailable(self._deferred or "Milvus 索引尚未打开")

    # ── 集合名 ──────────────────────────────────────────

    @property
    def _staging(self) -> str:
        return f"{self._collection}__staging"

    @property
    def _retired(self) -> str:
        return f"{self._collection}__retired"

    # ── 崩溃恢复 ────────────────────────────────────────

    def _recover(self) -> None:
        """把上一次重建留下的中间状态收拾干净，让启动后的行为是确定的。

        正常结束时不会留下任何残留；只有进程在切换过程中被杀掉才会走到这里。
        """
        has_final = self._client.has_collection(self._collection)
        has_retired = self._client.has_collection(self._retired)
        has_staging = self._client.has_collection(self._staging)

        if not has_final and has_retired:
            # 切换进行到一半：旧版还在 retired 里，先还原成可用状态
            self._client.rename_collection(old_name=self._retired, new_name=self._collection)
            self.notes.append("检测到未完成的重建，已从上一版集合恢复索引")
            has_final, has_retired = True, False

        if has_staging:
            # 正式集合可用 → 暂存的一定是半成品；正式集合不在 → 半成品不能当正式索引用
            self._drop_if_exists(self._staging)
            self.notes.append(
                "清理了上次未完成的重建暂存集合"
                if has_final
                else "丢弃了上次未完成的暂存集合，请执行 rebuild_store 或 reindex 从归档重建"
            )
        if has_retired:
            self._drop_if_exists(self._retired)

    # ── 集合管理 ────────────────────────────────────────

    def _load_collection(self) -> None:
        """新进程打开已有集合时它处于 released 状态，不 load 就无法 search。"""
        try:
            self._client.load_collection(self._collection)
        except Exception as exc:  # pragma: no cover - 平台相关
            message = str(exc)
            if "faiss" in message.lower():
                # 走到这里说明路径检查没拦住（例如符号链接绕过了判断），
                # 把 faiss 的原始报错换成能照着做的话
                raise BackendUnavailable(NON_ASCII_PATH_HINT) from exc
            if "already loaded" not in message.lower():
                raise BackendUnavailable(f"无法加载 Milvus 集合：{message}") from exc

    def _vector_dim(self, collection: str) -> int:
        """从集合 schema 里读出向量字段的实际维度（0 表示读不到）。"""
        try:
            info = self._client.describe_collection(collection)
        except Exception as exc:  # pragma: no cover - 平台相关
            raise BackendUnavailable(f"无法读取集合 {collection} 的结构：{exc}") from exc
        for field in info.get("fields", []):
            if field.get("name") == "vector":
                dim = field.get("params", {}).get("dim")
                return int(dim) if dim else 0
        return 0

    def _create_collection(self, collection: str, dim: int) -> None:
        from pymilvus import DataType, MilvusClient

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field("project_name", DataType.VARCHAR, max_length=256)
        schema.add_field("project_code", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_year", DataType.INT32)
        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")
        self._client.create_collection(
            collection_name=collection, schema=schema, index_params=index_params
        )

    def _drop_if_exists(self, collection: str) -> None:
        if self._client.has_collection(collection):
            self._client.drop_collection(collection)

    def _adopt_existing(self, dim: int) -> None:
        """已存在的集合：只核对维度，不创建、不改名。

        单独拆出来是为了让无锁的快速打开路径能复用维度校验，却拿不到"建集合"的能力
        ——那两件事以前写在同一个函数里，于是"只读地看一眼"和"可以创建"被混成了一句。
        """
        actual = self._vector_dim(self._collection)
        if dim and actual and dim != actual:
            # 配置说是 N 维、集合里其实是 M 维：写入会直接报错，检索会给出无意义结果
            raise BackendUnavailable(
                f"Milvus 集合 {self._collection} 的向量维度是 {actual}，"
                f"当前配置是 {dim}。改配置或换集合名都改不了已写入的向量，"
                "请用 `python -m rag4prefa reindex` 按当前模型整体重建。"
            )
        self.dim = actual or dim

    def _ensure_collection(self, dim: int) -> None:
        if self._client.has_collection(self._collection):
            self._adopt_existing(dim)
            return
        if not dim:
            raise BackendUnavailable("首次创建 Milvus 集合需要已知向量维度")
        self._create_collection(self._collection, dim)
        self.dim = dim

    # ── 对外 ────────────────────────────────────────────

    def _upsert_into(self, collection: str, items: Sequence[VectorItem]) -> None:
        rows = [
            {
                "id": item.chunk_id,
                "vector": _normalize(item.vector).tolist(),
                "project_name": item.project_name,
                "project_code": item.project_code,
                "doc_year": item.doc_year if item.doc_year is not None else -1,
            }
            for item in items
        ]
        self._client.upsert(collection_name=collection, data=rows)

    def upsert(self, items: Sequence[VectorItem]) -> None:
        self._require_ready()
        if not items:
            return
        self._upsert_into(self._collection, items)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        self._require_ready()
        if chunk_ids:
            self._client.delete(collection_name=self._collection, ids=list(chunk_ids))

    def search(
        self, vector: list[float], *, limit: int, filt: Filter | None = None
    ) -> list[tuple[str, float]]:
        self._require_ready()
        expr = _milvus_expr(filt)
        res = self._client.search(
            collection_name=self._collection,
            data=[_normalize(vector).tolist()],
            limit=limit,
            filter=expr or "",
            output_fields=["project_name"],
        )
        hits = res[0] if res else []
        return [(str(h["id"]), float(h["distance"])) for h in hits]

    def count(self) -> int:
        self._require_ready()
        return self._count_of(self._collection)

    def ids(self) -> set[str]:
        self._require_ready()
        return self._all_ids(self._collection)

    def _count_of(self, collection: str) -> int:
        """优先用 count(*) 拿精确条数。

        `get_collection_stats` 的 row_count 语义随版本而异（可能是统计估算或含已删未压实
        的记录）。一致性检查拿它当阻断依据时，宁可慢一点也要一个确定的数字——
        误报 503 和漏报脏索引一样糟。
        """
        try:
            res = self._client.query(
                collection_name=collection, filter="", output_fields=["count(*)"]
            )
            if res:
                return int(res[0].get("count(*)", 0))
        except Exception:  # pragma: no cover - 老版本不支持聚合查询时回退
            pass
        stats = self._client.get_collection_stats(collection)
        return int(stats.get("row_count", 0))

    def clear(self) -> None:
        self._require_ready()
        self._client.delete(collection_name=self._collection, filter="id != ''")

    def _all_ids(self, collection: str) -> set[str]:
        """列出集合里的全部主键。用于精确比对与清理孤儿，所以要一次性取完、不能截断。"""
        total = self._count_of(collection)
        ids: set[str] = set()
        page = 1024
        offset = 0
        while offset < total:
            batch = self._client.query(
                collection_name=collection,
                filter="id != ''",
                output_fields=["id"],
                limit=page,
                offset=offset,
            )
            rows = list(batch)
            if not rows:
                break
            ids.update(str(r["id"]) for r in rows)
            offset += len(rows)
        return ids

    def rebuild(self, items: Sequence[VectorItem]) -> None:
        """整体替换全部向量：准备一整套新的，校验通过后再切换。

        **维度变不变都走同一条路径**，不再有"同维度就原地 upsert"的分支。
        原地覆盖在正常情况下确实更快、也不需要删集合，但它不是原子操作：覆盖到一半
        进程被杀，正式集合就是新旧混杂的一堆向量，而检索看不出任何异常。既然"少一次
        rename"换不来任何业务收益，就不该为它保留一条更脆弱的路径。

        代价：重建期间暂存集合占一份额外空间；两次 rename 之间有一个极短窗口，
        此刻的检索会失败而不是返回混杂结果——这是刻意的取舍。

        语义：
        1. 在 `__staging` 里建同维度集合、写全、load、按条数校验；
        2. 正式集合 rename 成 `__retired` 让位，暂存 rename 成正式名（元数据操作，不是删了重建）；
        3. 成功后才删掉 `__retired`；
        4. 中途失败：暂存直接丢弃，正式集合原样可用；`_recover()` 负责收拾进程被杀留下的残留。
        """
        self._require_ready()
        if not items:
            self.clear()
            return

        dim = len(items[0].vector)
        if any(len(i.vector) != dim for i in items):
            raise ValueError("待重建的向量维度不一致，已中止（索引未做任何改动）")

        self._drop_if_exists(self._staging)
        try:
            self._create_collection(self._staging, dim)
            self._upsert_into(self._staging, items)
            self._client.load_collection(self._staging)
            got = self._count_of(self._staging)
            if got != len(items):
                raise RebuildError(
                    f"暂存集合写入 {got} 条，期望 {len(items)} 条。正式索引未受影响，可直接重试。"
                )
        except Exception:
            self._drop_if_exists(self._staging)
            raise

        self._drop_if_exists(self._retired)
        had_final = self._client.has_collection(self._collection)
        try:
            if had_final:
                self._client.rename_collection(
                    old_name=self._collection, new_name=self._retired
                )
            self._client.rename_collection(old_name=self._staging, new_name=self._collection)
        except Exception as exc:
            # 尽力回滚：把旧集合改回正式名字，别让系统停在没有索引的状态
            if self._client.has_collection(self._retired) and not self._client.has_collection(
                self._collection
            ):
                self._client.rename_collection(
                    old_name=self._retired, new_name=self._collection
                )
            raise RebuildError(
                f"切换索引失败：{exc}。已回滚到上一版，检索可继续；"
                "完整的新向量仍在暂存集合中，可重试 reindex。"
            ) from exc

        self._drop_if_exists(self._retired)
        self.dim = dim
        self._load_collection()


def to_archive_row(item: VectorItem, model: str = "") -> tuple:
    """归档行：归一化后以 float32 存，换后端重建时不必重新 embedding。"""
    arr = _normalize(item.vector)
    return (
        item.chunk_id,
        item.project_name,
        item.project_code,
        item.doc_year,
        int(arr.shape[0]),
        item.model or model,
        array("f", arr.tolist()).tobytes(),
    )


def from_archive_row(row: dict) -> VectorItem:
    vec = array("f")
    vec.frombytes(row["vec"])
    return VectorItem(
        chunk_id=row["chunk_id"],
        vector=vec.tolist(),
        project_name=row["project_name"],
        project_code=row["project_code"],
        doc_year=row["doc_year"],
        model=row.get("model") or "",
    )


def _milvus_expr(filt: Filter | None) -> str:
    if not filt:
        return ""
    parts: list[str] = []
    if filt.projects:
        parts.append(f"project_name in {json.dumps(filt.projects, ensure_ascii=False)}")
    if filt.year_from is not None:
        parts.append(f"doc_year >= {int(filt.year_from)}")
    if filt.year_to is not None:
        parts.append(f"doc_year <= {int(filt.year_to)}")
    return " and ".join(parts)
