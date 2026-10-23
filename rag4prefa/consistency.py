"""索引一致性：一个向量空间只允许一个 embedding 模型。

为什么需要独立一层：维度相同不代表向量空间相同。模型 A 和模型 B 都输出 768 维，
余弦相似度照样算得出来、接口照样返回 200，只是**语义悄悄错位**——这类故障不报错、
不掉数据，只让检索慢慢变差，是最难在验收里发现的一种。

所以这里把"能不能写 / 能不能查"的判断集中起来，条件不满足就明确拒绝，
而不是让调用方各自记得要检查。判断依据有两个，缺一不可：

- `meta.embedding_model`：上一次成功建索引用的是哪个模型（声明）；
- `vectors.model`：归档里每一行实际是哪个模型产生的（事实）。

两者不一致说明有过一次没走完的重建，同样拒绝继续，让人来决定怎么修。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import repo


class IndexMismatch(RuntimeError):
    """索引与当前配置的 embedding 模型对不上，必须人工处理后再继续。"""


@dataclass(frozen=True)
class IndexState:
    model: str | None  # meta 里记录的模型
    dim: int | None  # meta 里记录的维度
    count: int  # 归档向量条数
    models: dict[str, int]  # 归档里实际出现的模型 → 条数
    dims: dict[int, int]  # 归档里实际出现的维度 → 条数
    chunks: int = 0  # 事实表里的片段数

    @property
    def known_models(self) -> set[str]:
        """归档里出现过的、有名字的模型。空串是历史遗留的未标注行，不能当依据。"""
        return {m for m in self.models if m}

    @property
    def issues(self) -> list[str]:
        """不影响继续用、但必须让人看见的问题。"""
        problems: list[str] = []
        if len(self.known_models) > 1:
            listed = "、".join(sorted(self.known_models))
            problems.append(f"归档里混有两套模型的向量（{listed}），检索语义不可信")
        if len(self.dims) > 1:
            listed = "、".join(str(d) for d in sorted(self.dims))
            problems.append(f"归档里混有多种向量维度（{listed}）")
        if self.count and self.model and self.known_models and self.model not in self.known_models:
            actual = "、".join(sorted(self.known_models))
            problems.append(
                f"元数据记录模型 {self.model}，归档实际是 {actual}——上次重建没走完"
            )
        if self.dim and self.known_dims and self.dim not in self.dims:
            problems.append(f"元数据记录维度 {self.dim}，归档实际是 {sorted(self.dims)}")
        if self.chunks != self.count:
            problems.append(
                f"片段 {self.chunks} 条、向量 {self.count} 条，索引与事实不一致"
                "（入库或重建没走完），可执行 `python -m rag4prefa reindex` 重建"
            )
        return problems

    @property
    def known_dims(self) -> set[int]:
        return set(self.dims)


def inspect(conn: sqlite3.Connection) -> IndexState:
    stats = repo.vector_stats(conn)
    dim_value = repo.get_meta(conn, "embedding_dim")
    return IndexState(
        model=repo.get_meta(conn, "embedding_model"),
        dim=int(dim_value) if dim_value and dim_value.isdigit() else None,
        count=int(stats["count"]),
        models=stats["models"],
        dims=stats["dims"],
        chunks=repo.chunk_count(conn),
    )


def _mixing_message(state: IndexState, model: str) -> str:
    index_model = "、".join(sorted(state.known_models)) or state.model or "未知"
    return (
        f"当前配置的 embedding 模型是「{model}」，但索引是用「{index_model}」建的。"
        "维度相同也不能混用：两个模型的向量空间不同，混在一起检索不会报错，只会让结果变差。"
        "处理办法二选一：① 把 R4P_EMBEDDING_MODEL 改回索引所用的模型；"
        "② 确认要换模型，就执行 `python -m rag4prefa reindex`（不带 --project）做全量重建。"
    )


def ensure_writable(conn: sqlite3.Connection, model: str, dim: int = 0) -> IndexState:
    """入库/重建前的检查：索引里已有的向量必须和当前模型同一套。"""
    state = inspect(conn)
    if not state.count:
        return state  # 空索引，谁先写谁定调

    if len(state.known_models) > 1:
        listed = "、".join(sorted(state.known_models))
        raise IndexMismatch(
            f"索引里已经混有两套模型的向量（{listed}），任何写入都会让问题更糟。"
            "请先执行 `python -m rag4prefa reindex` 用当前模型的配置整体重建。"
        )
    if state.known_models and model not in state.known_models:
        raise IndexMismatch(_mixing_message(state, model))
    if dim and state.dims and dim not in state.dims:
        actual = "、".join(str(d) for d in sorted(state.dims))
        raise IndexMismatch(
            f"当前模型输出 {dim} 维，索引里是 {actual} 维。"
            "换维度只能整体重建：执行 `python -m rag4prefa reindex`。"
        )
    return state


def ensure_queryable(conn: sqlite3.Connection, model: str, store_dim: int = 0) -> IndexState:
    """检索前的检查：查询向量必须和文档向量出自同一个模型。

    宁可让用户看到一条能读懂的错误，也不要给出一份看起来正常、实则无关的引用。
    """
    state = ensure_writable(conn, model, store_dim)
    if len(state.known_models) > 1:
        listed = "、".join(sorted(state.known_models))
        raise IndexMismatch(
            f"索引里混有两套模型的向量（{listed}），检索结果不可信。"
            "请执行 `python -m rag4prefa reindex` 重建。"
        )
    return state


def health_issues(conn: sqlite3.Connection, model: str | None, store_dim: int = 0) -> list[str]:
    """供 /api/health 报告：不抛异常，只把问题列出来。"""
    state = inspect(conn)
    issues = list(state.issues)
    if model and state.count and state.known_models and model not in state.known_models:
        index_model = "、".join(sorted(state.known_models))
        issues.append(f"配置模型 {model} 与索引模型 {index_model} 不一致，检索会被拒绝")
    if store_dim and state.dims and store_dim not in state.dims:
        issues.append(f"向量后端维度 {store_dim} 与归档维度 {sorted(state.dims)} 不一致")
    return issues
