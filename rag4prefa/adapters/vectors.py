"""向量索引：可替换后端。

约定——向量库只是**派生索引**，SQLite 里的 chunks 才是事实。
因此这里所有方法都是幂等的，任何时刻都可以从 SQLite 重建。
"""

from __future__ import annotations

import json
import os
import sqlite3
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

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


class NumpyStore:
    """精确余弦检索（暴力遍历）。

    优点：零外部依赖、Windows 可跑、结果可复现（无近似召回误差）；
    代价：O(n) 扫描。十万级 chunk 以内延迟仍在毫秒级，超出请切 Milvus。
    """

    name = "numpy"

    def __init__(self, conn: sqlite3.Connection, dim: int) -> None:
        self._conn = conn
        self.dim = dim
        self.notes: list[str] = []
        self._ids: list[str] = []
        self._projects: list[str] = []
        self._codes: list[str] = []
        self._years: list[int | None] = []
        self._mat: np.ndarray = np.zeros((0, dim or 1), dtype=np.float32)
        self._load()

    # ── 内部 ────────────────────────────────────────

    def _load(self) -> None:
        self._ids = []
        self._projects = []
        self._codes = []
        self._years = []
        self._mat = np.zeros((0, self.dim or 1), dtype=np.float32)
        rows = self._conn.execute(
            "SELECT chunk_id, project_name, project_code, doc_year, dim, vec FROM vectors"
        ).fetchall()
        if not rows:
            return
        self._ids = [r["chunk_id"] for r in rows]
        self._projects = [r["project_name"] for r in rows]
        self._codes = [r["project_code"] for r in rows]
        self._years = [r["doc_year"] for r in rows]
        dims = {int(r["dim"]) for r in rows}
        if len(dims) > 1:
            # 归档里混了两种维度，vstack 会直接崩在一个看不懂的错误上
            raise RebuildError(
                f"归档里混有多种向量维度 {sorted(dims)}，无法组成一个矩阵。"
                "请执行 `python -m rag4prefa reindex` 全量重建。"
            )
        dim = dims.pop() if dims else 0
        if dim:
            self.dim = dim
        self._mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])

    def _rebuild_slice(self) -> None:
        self._load()

    # ── 对外 ────────────────────────────────────────

    def upsert(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
        if not self.dim:
            self.dim = len(items[0].vector)
        new_rows = []
        for item in items:
            arr = _normalize(item.vector)
            if arr.shape[0] != self.dim:
                raise ValueError(f"向量维度 {arr.shape[0]} 与索引维度 {self.dim} 不一致")
            self._conn.execute(
                "INSERT OR REPLACE INTO vectors (chunk_id, project_name, project_code, doc_year,"
                " dim, model, vec) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    item.chunk_id,
                    item.project_name,
                    item.project_code,
                    item.doc_year,
                    self.dim,
                    item.model,
                    array("f", arr.tolist()).tobytes(),
                ),
            )
            new_rows.append((item, arr))
        self._conn.commit()
        self._rebuild_slice()

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        marks = ",".join("?" * len(chunk_ids))
        self._conn.execute(f"DELETE FROM vectors WHERE chunk_id IN ({marks})", list(chunk_ids))
        self._conn.commit()
        self._rebuild_slice()

    def search(
        self, vector: list[float], *, limit: int, filt: Filter | None = None
    ) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        query = _normalize(vector)
        if query.shape[0] != self._mat.shape[1]:
            raise ValueError(f"查询维度 {query.shape[0]} 与索引维度 {self._mat.shape[1]} 不一致")
        scores = self._mat @ query

        mask = np.ones(len(self._ids), dtype=bool)
        if filt:
            if filt.projects:
                wanted = set(filt.projects)
                mask &= np.array([p in wanted for p in self._projects], dtype=bool)
            if filt.year_from is not None:
                mask &= np.array([y is not None and y >= filt.year_from for y in self._years])
            if filt.year_to is not None:
                mask &= np.array([y is not None and y <= filt.year_to for y in self._years])
        if not mask.any():
            return []

        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:limit]
        return [(self._ids[i], float(scores[i])) for i in order if np.isfinite(scores[i])]

    def count(self) -> int:
        return len(self._ids)

    def ids(self) -> set[str]:
        return set(self._ids)

    def clear(self) -> None:
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


class MilvusStore:
    """Milvus（Lite 或 standalone，同一个 pymilvus 客户端，只改 URI）。

    注意：Milvus Lite 不支持 HNSW 参数，统一用 AUTOINDEX；
    切到 standalone 时把 index_type 换成 HNSW 才需要调参。

    集合命名约定（用于崩溃恢复，见 `_recover`）：
    - `<collection>`            正式索引，检索只读它；
    - `<collection>__staging`   重建时的暂存集合，写完并校验后才切换；
    - `<collection>__retired`   被换下来的上一版，切换成功后立即删除。
    """

    name = "milvus"

    def __init__(self, uri: str, collection: str, dim: int) -> None:
        try:
            from pymilvus import MilvusClient
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise BackendUnavailable("未安装 pymilvus：pip install 'rag4prefa[milvus]'") from exc

        _reject_non_ascii_uri(os.path.join(str(uri), collection))
        self._collection = collection
        self.dim = dim
        self.notes: list[str] = []
        try:
            self._client = MilvusClient(uri=uri)
        except Exception as exc:  # pragma: no cover - 平台相关
            raise BackendUnavailable(f"无法打开 Milvus 存储 {uri}: {exc}") from exc

        self._recover()
        self._ensure_collection(dim)
        self._load_collection()

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

    def _ensure_collection(self, dim: int) -> None:
        if self._client.has_collection(self._collection):
            actual = self._vector_dim(self._collection)
            if dim and actual and dim != actual:
                # 配置说是 N 维、集合里其实是 M 维：写入会直接报错，检索会给出无意义结果
                raise BackendUnavailable(
                    f"Milvus 集合 {self._collection} 的向量维度是 {actual}，"
                    f"当前配置是 {dim}。改配置或换集合名都改不了已写入的向量，"
                    "请用 `python -m rag4prefa reindex` 按当前模型整体重建。"
                )
            self.dim = actual or dim
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
        if not items:
            return
        self._upsert_into(self._collection, items)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        if chunk_ids:
            self._client.delete(collection_name=self._collection, ids=list(chunk_ids))

    def search(
        self, vector: list[float], *, limit: int, filt: Filter | None = None
    ) -> list[tuple[str, float]]:
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
        return self._count_of(self._collection)

    def ids(self) -> set[str]:
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
