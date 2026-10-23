"""向量索引：可替换后端。

约定——向量库只是**派生索引**，SQLite 里的 chunks 才是事实。
因此这里所有方法都是幂等的，任何时刻都可以从 SQLite 重建。
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..domain import Filter


class BackendUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class VectorItem:
    chunk_id: str
    vector: list[float]
    project_name: str
    project_code: str
    doc_year: int | None


class VectorStore(Protocol):
    name: str
    dim: int

    def upsert(self, items: Sequence[VectorItem]) -> None: ...
    def delete(self, chunk_ids: Sequence[str]) -> None: ...
    def search(
        self, vector: list[float], *, limit: int, filt: Filter | None = None
    ) -> list[tuple[str, float]]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


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
        dim = rows[0]["dim"] or self.dim
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
                " dim, model, vec) VALUES (?, ?, ?, ?, ?, '', ?)",
                (
                    item.chunk_id,
                    item.project_name,
                    item.project_code,
                    item.doc_year,
                    self.dim,
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

    def clear(self) -> None:
        self._conn.execute("DELETE FROM vectors")
        self._conn.commit()
        self._load()


class MilvusStore:
    """Milvus（Lite 或 standalone，同一个 pymilvus 客户端，只改 URI）。

    注意：Milvus Lite 不支持 HNSW 参数，统一用 AUTOINDEX；
    切到 standalone 时把 index_type 换成 HNSW 才需要调参。
    """

    name = "milvus"

    def __init__(self, uri: str, collection: str, dim: int) -> None:
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise BackendUnavailable("未安装 pymilvus：pip install 'rag4prefa[milvus]'") from exc

        self._collection = collection
        self.dim = dim
        try:
            self._client = MilvusClient(uri=uri)
        except Exception as exc:  # pragma: no cover - 平台相关
            raise BackendUnavailable(f"无法打开 Milvus 存储 {uri}: {exc}") from exc

        if not self._client.has_collection(collection):
            if not dim:
                raise BackendUnavailable("首次创建 Milvus 集合需要已知向量维度")
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("project_name", DataType.VARCHAR, max_length=256)
            schema.add_field("project_code", DataType.VARCHAR, max_length=64)
            schema.add_field("doc_year", DataType.INT32)
            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self._client.create_collection(
                collection_name=collection, schema=schema, index_params=index_params
            )

    def upsert(self, items: Sequence[VectorItem]) -> None:
        if not items:
            return
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
        self._client.upsert(collection_name=self._collection, data=rows)

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
        stats = self._client.get_collection_stats(self._collection)
        return int(stats.get("row_count", 0))

    def clear(self) -> None:
        self._client.delete(collection_name=self._collection, filter="id != ''")


def to_archive_row(item: VectorItem, model: str) -> tuple:
    """归档行：归一化后以 float32 存，换后端重建时不必重新 embedding。"""
    arr = _normalize(item.vector)
    return (
        item.chunk_id,
        item.project_name,
        item.project_code,
        item.doc_year,
        int(arr.shape[0]),
        model,
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
