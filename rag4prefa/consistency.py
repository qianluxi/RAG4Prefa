"""索引一致性：能不能写、能不能查，只看这一处判断。

为什么需要独立一层：维度相同不代表向量空间相同。模型 A 和模型 B 都输出 768 维，
余弦相似度照样算得出来、接口照样返回 200，只是**语义悄悄错位**——这类故障不报错、
不掉数据，只让检索慢慢变差，是最难在验收里发现的一种。

更隐蔽的是"数量对不上"：片段在库里、向量只写了一半，稠密召回静默少一路，BM25 还在
返回结果，用户看到一份格式完整、引用也对得上、内容却漏了的答案。所以这里的判断是
**阻断式**的：只要索引不能自证一致，检索与增量写入都拒绝，而不是给一份"看起来正常"的答案。

判断依据（缺一不可）：

- `meta.embedding_model` / `meta.embedding_dim`：上一次成功建索引时声明的模型与维度；
- `vectors.model` / `vectors.dim`：归档里每一行实际是谁算的（事实）；
- 片段与归档的**逐 id 对应**，以及后端实际条数与归档的交叉校验。

分两级：
- `blockers`：索引不能自证一致，检索/写入必须拒绝；
- 后端的回退说明、恢复日志属于提示，不算损坏（记在 `VectorStore.notes`，不进这里）。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import repo


class IndexMismatch(RuntimeError):
    """索引与当前配置或自身事实对不上，必须人工处理后再继续。"""


@dataclass(frozen=True)
class IndexState:
    model: str | None  # meta 里记录的模型（声明）
    dim: int | None  # meta 里记录的维度（声明）
    count: int  # 归档向量条数
    models: dict[str, int]  # 归档里实际出现的模型 → 条数
    dims: dict[int, int]  # 归档里实际出现的维度 → 条数
    chunks: int = 0  # 事实表里的片段数
    missing: int = 0  # 有正文但没有归档向量
    orphan: int = 0  # 有归档向量但没有正文
    unlabeled: int = 0  # model 为空的历史行
    store_count: int | None = None  # 后端实际条数（None 表示没查）

    @property
    def known_models(self) -> set[str]:
        """归档里出现过的、有名字的模型。空串是历史遗留的未标注行，不能当依据。"""
        return {m for m in self.models if m}

    @property
    def known_dims(self) -> set[int]:
        return set(self.dims)

    @property
    def is_empty(self) -> bool:
        """全空（新库）：此时没有任何东西可以不一致。"""
        return not self.chunks and not self.count and not self.store_count

    @property
    def blockers(self) -> list[str]:
        """索引不能自证一致——出现任意一条，检索和增量写入都必须拒绝。"""
        if self.is_empty:
            return []
        problems: list[str] = []

        if len(self.known_models) > 1:
            listed = "、".join(sorted(self.known_models))
            problems.append(f"归档里混有两套模型的向量（{listed}），检索语义不可信")
        if self.unlabeled:
            problems.append(
                f"归档里有 {self.unlabeled} 条向量没有记录产生它的模型，无法确认向量空间"
            )
        if len(self.dims) > 1:
            listed = "、".join(str(d) for d in sorted(self.dims))
            problems.append(f"归档里混有多种向量维度（{listed}）")
        if self.model and self.known_models and self.model not in self.known_models:
            actual = "、".join(sorted(self.known_models))
            problems.append(
                f"元数据记录模型 {self.model}，归档实际是 {actual}——上次重建没走完"
            )
        if self.dim and self.dims and self.dim not in self.dims:
            problems.append(f"元数据记录维度 {self.dim}，归档实际是 {sorted(self.dims)}")
        if self.missing:
            problems.append(
                f"{self.missing} 个片段没有归档向量（稠密召回会静默漏掉这些内容），"
                "执行 `python -m rag4prefa reindex` 可补齐"
            )
        if self.orphan:
            problems.append(
                f"归档里有 {self.orphan} 条向量找不到对应正文，"
                "执行 `python -m rag4prefa reindex` 会清掉"
            )
        if self.store_count is not None and self.store_count != self.count:
            problems.append(
                f"向量索引实际 {self.store_count} 条、归档 {self.count} 条，"
                "两者不一致；执行 `python -m rag4prefa verify` 可定位差异"
            )
        return problems

    @property
    def issues(self) -> list[str]:
        """兼容旧名：就是 `blockers`。"""
        return self.blockers


def inspect(conn: sqlite3.Connection, *, store_count: int | None = None) -> IndexState:
    """读取一次完整状态。

    `store_count` 是向量后端的实际条数。传进来才做"索引 vs 归档"的交叉校验——
    只比 SQLite 内部的数字会漏掉"归档对、Milvus 少写了几条"这种情况。
    """
    stats = repo.vector_stats(conn)
    dim_value = repo.get_meta(conn, "embedding_dim")
    return IndexState(
        model=repo.get_meta(conn, "embedding_model"),
        dim=int(dim_value) if dim_value and dim_value.isdigit() else None,
        count=int(stats["count"]),
        models=stats["models"],
        dims=stats["dims"],
        chunks=repo.chunk_count(conn),
        missing=repo.missing_vector_count(conn),
        orphan=repo.orphan_vector_count(conn),
        unlabeled=repo.unlabeled_vector_count(conn),
        store_count=store_count,
    )


def _mixing_message(state: IndexState, model: str) -> str:
    index_model = "、".join(sorted(state.known_models)) or state.model or "未知"
    return (
        f"当前配置的 embedding 模型是「{model}」，但索引是用「{index_model}」建的。"
        "维度相同也不能混用：两个模型的向量空间不同，混在一起检索不会报错，只会让结果变差。"
        "处理办法二选一：① 把 R4P_EMBEDDING_MODEL 改回索引所用的模型；"
        "② 确认要换模型，就执行 `python -m rag4prefa reindex`（不带 --project）做全量重建。"
    )


def _dim_message(state: IndexState, dim: int) -> str:
    actual = "、".join(str(d) for d in sorted(state.dims))
    return (
        f"当前模型输出 {dim} 维，索引里是 {actual} 维。"
        "换维度只能整体重建：执行 `python -m rag4prefa reindex`。"
    )


def _check(
    conn: sqlite3.Connection,
    model: str,
    dim: int,
    store_count: int | None,
    *,
    action: str,
) -> IndexState:
    """检索与写入共用的唯一入口：不自洽就抛，绝不返回"可能还行"的结果。"""
    state = inspect(conn, store_count=store_count)
    problems = list(state.blockers)
    # 索引自身自洽还不够，还要和**当前配置**对得上：换模型/换维度都属于这一类
    if state.count or state.chunks:
        if state.known_models and model not in state.known_models:
            problems.append(_mixing_message(state, model))
        if dim and state.dims and dim not in state.dims:
            problems.append(_dim_message(state, dim))
    if problems:
        raise IndexMismatch(f"{action}已拒绝：索引当前不能自证一致。\n- " + "\n- ".join(problems))
    return state


def ensure_writable(
    conn: sqlite3.Connection,
    model: str,
    dim: int = 0,
    *,
    store_count: int | None = None,
    action: str = "写入",
) -> IndexState:
    """入库 / 重建前的检查。

    以前这里在"归档条数为 0"时直接放行，理由是"空索引，谁先写谁定调"。但"归档为 0"
    不等于"库是空的"：片段表可能有几百条、向量全丢了。那种情况下继续写入只会让
    不一致越滚越大，所以现在只有**真的全空**才放行。
    """
    return _check(conn, model, dim, store_count, action=action)


def ensure_queryable(
    conn: sqlite3.Connection,
    model: str,
    store_dim: int = 0,
    *,
    store_count: int | None = None,
) -> IndexState:
    """检索前的检查：宁可让用户看到一条能读懂的错误，也不要给一份看似正常的引用。"""
    return _check(conn, model, store_dim, store_count, action="检索")


def collect_issues(
    state: IndexState, model: str | None, store_dim: int = 0
) -> list[str]:
    """把"配置 vs 索引"的冲突并进 blockers，供 /api/health 一次性报告。"""
    issues = list(state.blockers)
    if model and state.count and state.known_models and model not in state.known_models:
        index_model = "、".join(sorted(state.known_models))
        issues.append(f"配置模型 {model} 与索引模型 {index_model} 不一致，检索会被拒绝")
    if store_dim and state.dims and store_dim not in state.dims:
        issues.append(f"向量后端维度 {store_dim} 与归档维度 {sorted(state.dims)} 不一致")
    return issues


def health_issues(
    conn: sqlite3.Connection,
    model: str | None,
    store_dim: int = 0,
    *,
    store_count: int | None = None,
) -> list[str]:
    """供 /api/health 报告：不抛异常，只把问题列出来（与检索用的判断同源）。"""
    return collect_issues(inspect(conn, store_count=store_count), model, store_dim)


def verify_ids(conn: sqlite3.Connection, store) -> dict:
    """逐 id 比对"片段 / 归档 / 实际索引"三方，返回可打印的差异报告。

    为什么每天要跑的不是这个：它要把索引里的主键全部拉出来，成本随条数线性增长，
    放在每次请求里不现实。日常用计数做必要条件（`IndexState.blockers`），
    这个命令做充分条件——条数相同但 id 不同（写坏了、切换只成功一半）只有它能看出来。
    """
    facts = repo.all_chunk_ids(conn)
    archived = repo.archived_chunk_ids(conn)
    try:
        indexed = store.ids()
    except Exception as exc:  # 后端读不出来也要给出可读结论，而不是抛栈
        return {
            "ok": False,
            "error": f"读不到向量索引的主键：{type(exc).__name__}: {exc}",
            "chunks": len(facts),
            "archived": len(archived),
        }

    missing_in_archive = facts - archived  # 有正文、没有归档向量
    orphan_archive = archived - facts  # 归档向量没有正文
    missing_in_index = archived - indexed  # 归档里有、索引里没写进去
    extra_in_index = indexed - archived  # 索引里有、归档里没有
    problems = {
        "片段没有归档向量": missing_in_archive,
        "归档向量没有正文": orphan_archive,
        "归档有但索引缺": missing_in_index,
        "索引有但归档缺": extra_in_index,
    }
    return {
        "ok": not any(problems.values()),
        "chunks": len(facts),
        "archived": len(archived),
        "indexed": len(indexed),
        "counts": {k: len(v) for k, v in problems.items()},
        # 只给样例，避免几万条 id 把终端刷满；条数在 counts 里是全的
        "samples": {k: sorted(v)[:10] for k, v in problems.items() if v},
    }
