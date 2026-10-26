"""索引一致性：能不能写、能不能查，只看这一处判断。

为什么需要独立一层：维度相同不代表向量空间相同。模型 A 和模型 B 都输出 768 维，
余弦相似度照样算得出来、接口照样返回 200，只是**语义悄悄错位**——这类故障不报错、
不掉数据，只让检索慢慢变差，是最难在验收里发现的一种。

更隐蔽的是"数量对不上"：片段在库里、向量只写了一半，稠密召回静默少一路，BM25 还在
返回结果，用户看到一份格式完整、引用也对得上、内容却漏了的答案。所以这里的判断是
**阻断式**的：只要索引不能自证一致，检索与增量写入都拒绝，而不是给一份"看起来正常"的答案。

判断依据（缺一不可）：

- `meta.embedding_model` / `meta.embedding_dim`：上一次成功建索引时声明的模型与维度；
- `meta.index_state`：上一次索引切换有没有走完（不是 `ready` 就一律阻断）；
- `vectors.model` / `vectors.dim`：归档里每一行实际是谁算的（事实）；
- 片段与归档的**逐 id 对应**，以及后端实际条数与归档的交叉校验。

为什么要单独记那条切换状态：SQLite 归档与向量库不在同一个事务里。归档写完、
向量库还没跟上时，两边条数、模型名、维度可以**完全一致**，逐项校验一条都挑不出
毛病，检索却会拿旧向量去配新正文。这类窗口没法靠事后比对发现，只能靠写之前留痕。

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


# "归档改了、索引没跟上"的标记前缀，后接受影响的文档 id。
#
# 为什么不能只报一句错：改年份这条路径不重算向量，只覆盖过滤列。归档（事实）先改、
# 向量索引后改，中间或之后就有一个窗口——文档表说 2025、检索按 2025 过滤却召回不到
# 这批片段。条数、模型、维度全都对得上，日常计数校验一条也发现不了，只有"写之前留痕"
# 能让检索在这段时间里停住。与 `rebuilding:*` 分开是因为修复动作不同：那个要重建索引，
# 这个从归档覆盖一遍就行（`rebuild` 一条命令两类标记都能清）。
UNSYNCED_PREFIX = "unsynced:"

# `unsynced:` 后面跟的作用域前缀。文档级直接写 doc_id，项目级走这个前缀
# （删项目一次影响几十上百个 id，逐个列出来没有意义）。
UNSYNCED_PROJECT_PREFIX = "project:"


def mark_unsynced(conn: sqlite3.Connection, scope: str) -> str:
    """在"归档与索引要分头改"之前留痕，返回**当前生效**的标记。

    只在状态是 `ready` 时覆盖：已经欠着一笔（例如 `rebuilding:*`、另一个文档的
    `unsynced:`）时不改写——那条标记同样能阻断检索，覆盖它只会让后来的人以为
    只有这一处没对齐，把更早的那笔彻底忘掉。

    有了它，调用方不必自己拼字符串：撤销时把 `get_index_state()` 在动手前取到的旧值
    写回去即可，不需要猜"刚才是 ready 还是另有标记"。
    """
    previous = repo.get_index_state(conn)
    if previous != repo.INDEX_READY:
        return previous
    marker = f"{UNSYNCED_PREFIX}{scope}"
    repo.set_index_state(conn, marker)
    return marker


def pending_message(marker: str) -> str:
    """「上次索引切换没走完」的说明。

    为什么单独一种表述：这种情况和"库坏了"不一样，它的特征是**所有数字都对得上**。
    SQLite 归档已经写进新向量，向量索引还停在旧的一版；条数、模型名、维度全都一致，
    逐项校验一条都挑不出毛病，检索却可能返回旧向量配上新正文。所以这里不能靠比对
    发现，只能靠写之前留下的标记。
    """
    if marker.startswith(UNSYNCED_PREFIX):
        scope = marker[len(UNSYNCED_PREFIX) :]
        if scope.startswith(UNSYNCED_PROJECT_PREFIX):
            where = f"项目「{scope[len(UNSYNCED_PROJECT_PREFIX) :]}」"
        elif scope:
            where = f"文档 {scope}"
        else:
            where = "索引"
        return (
            f"上一次改动{where}没有把归档与向量索引同步完（标记 {marker}）："
            "归档（事实）与后端索引的内容可能已经不一致——条数、模型、维度可以完全一致，"
            "按内容或按年份过滤却会对不上。同步完成前检索一律拒绝。"
            "执行 `python -m rag4prefa rebuild` 从归档恢复索引即可清除该标记。"
        )
    scope = "全量" if marker.endswith(":*") else f"项目「{marker.split(':', 1)[-1]}」"
    return (
        f"上次{scope}重建索引没有完成（标记 {marker}）：向量索引可能还停在上一版，"
        "而归档已经更新——两边条数、模型名、维度完全一致也证明不了内容一致。"
        "重建完成前检索一律拒绝。执行 `python -m rag4prefa reindex`（不带 --project）"
        "重建即可清除该标记；只想用归档里现成的向量，也可以用 rebuild。"
    )


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
    malformed: int = 0  # 声明维度与 vec 实际字节数对不上的行
    store_count: int | None = None  # 后端实际条数（None 表示没查）
    pending: str | None = None  # 上次索引切换未完成时的标记（meta.index_state）

    @property
    def known_models(self) -> set[str]:
        """归档里出现过的、有名字的模型。空串是历史遗留的未标注行，不能当依据。"""
        return {m for m in self.models if m}

    @property
    def known_dims(self) -> set[int]:
        return set(self.dims)

    @property
    def is_empty(self) -> bool:
        """全空（新库）：没有可以互相比对的数据。

        注意它**不等于"一致"**：空库上仍可能留着一条没走完的重建标记，那一条
        与数据无关、不能被"空"抵消，所以 `blockers` 会先判标记再判空。
        """
        return not self.chunks and not self.count and not self.store_count

    @property
    def blockers(self) -> list[str]:
        """索引不能自证一致——出现任意一条，检索和增量写入都必须拒绝。"""
        problems: list[str] = []

        # 未完成的重建标记必须排在"空库"判断**之前**：
        # 空库只说明没有可以互相比对的数据，它证明不了那笔重建已经走完。反过来，
        # 空库上带着 `rebuilding:*` 还放行写入，新内容一落地就会撞上这条标记——
        # 等于先把数据收进来，再拒绝检索，比一开始就报错更难收拾。
        # 唯一例外仍留在 `_check(retry_scope=...)`：同项目的局部重建可以消自己的标记。
        if self.pending:
            problems.append(pending_message(self.pending))
        if self.is_empty:
            return problems

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
        if self.malformed:
            # 这条挡的是一种"所有数字都对得上"的损坏：条数、模型名、dim 字段全都一致，
            # 只有 vec 的字节数不对。numpy 后端组不出矩阵，Milvus 侧则会把坏向量当好的。
            problems.append(
                f"归档里有 {self.malformed} 条向量的字节长度与声明的维度对不上（归档已损坏），"
                "执行 `python -m rag4prefa reindex` 从原文重新生成向量"
            )
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
    state = repo.get_index_state(conn)
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
        malformed=repo.malformed_vector_count(conn),
        store_count=store_count,
        pending=None if state == repo.INDEX_READY else state,
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
    retry_scope: str | None = None,
) -> IndexState:
    """检索与写入共用的唯一入口：不自洽就抛，绝不返回"可能还行"的结果。

    `retry_scope` 只给「重跑同一个项目的重建」用：上一次局部重建失败会留下
    `rebuilding:<项目名>` 标记，而重跑它正是消除该标记的方式。若连它也被自己挡住，
    用户就只能全量重建——那比必要的代价大得多。只放过**完全同名**的标记，
    全量标记（`rebuilding:*`）不在此列：一个项目的重建修不了全库。
    """
    state = inspect(conn, store_count=store_count)
    problems = list(state.blockers)
    if retry_scope and state.pending == f"rebuilding:{retry_scope}":
        problems = [p for p in problems if p != pending_message(state.pending)]
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
    retry_scope: str | None = None,
) -> IndexState:
    """入库 / 重建前的检查。

    以前这里在"归档条数为 0"时直接放行，理由是"空索引，谁先写谁定调"。但"归档为 0"
    不等于"库是空的"：片段表可能有几百条、向量全丢了。那种情况下继续写入只会让
    不一致越滚越大，所以现在只有**真的全空**才放行。
    """
    return _check(
        conn, model, dim, store_count, action=action, retry_scope=retry_scope
    )


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


def ensure_backend_current(conn: sqlite3.Connection, store) -> None:
    """本进程手里的后端，是不是这个库**当前**记着的那个。

    为什么需要单独一道：装配（`open_store`）发生在拿到维护锁**之前**。CLI 装配时
    库还记着 numpy，排队等锁的那几分钟里另一个进程可能把活动后端切到 Milvus；
    这时用手里的 numpy 继续写，归档更新了、Milvus 索引却没更新——而两边条数一致时
    连"索引 vs 归档"的计数校验都发现不了（2026-10-24 第十四轮复审）。

    判断放在维护锁里做：那时不会有第二个进程同时改后端，一次比较就是确定的。

    **只有两个地方允许换后端**（它们不调这个函数，改走自己的路径）：
    `rebuild`（从归档整体重建到目标后端，校验通过后连同 `ready` 一起登记）和
    `Services.settle_backend`（启动时确认）。其余动作一律拒绝——拒绝比"用错后端
    默默写一份没人查的索引"便宜得多。
    """
    recorded = repo.get_vector_backend(conn)
    if not recorded:
        return
    actual = getattr(store, "name", recorded)
    if actual == recorded:
        return
    raise IndexMismatch(
        f"库里记录的向量后端是「{recorded}」，本进程手里的是「{actual}」——"
        "装配之后活动后端被别的进程换过了。本次没有改动任何数据；"
        f"重新执行这条命令即可（新进程会装配到当前后端）。"
        f"确实要改用「{actual}」，用 `python -m rag4prefa rebuild`：它从归档重建，"
        "校验通过后才把活动后端改成新的那个。"
    )


def ensure_active_backend(conn: sqlite3.Connection, store) -> None:
    """检索准入：手里这个后端必须是库里登记的活动后端。

    为什么读路径也要单独一条：`index_generation` 只能证明"这次查询期间索引没被换过"。
    一个长期运行的服务装配时持有 Milvus，之后别的进程把活动后端换成 numpy——本次查询
    的起点已经在切换之后，代次从头到尾没变，两次确认都通过，可这个进程手里的索引早就
    是另一份了。它不会报错，只会安静地返回切换之前的内容（2026-10-24 第十六轮复审）。

    库里没有登记值（从没确认过）时不在这里拒绝：那属于"本进程没跑确认流程"，由服务层
    用 `Services.backend_confirmed` 判断。把两种情况混成一条，会让人分不清"该重启"
    还是"该重建"。
    """
    recorded = repo.get_vector_backend(conn)
    if not recorded:
        return
    actual = getattr(store, "name", recorded)
    if actual == recorded:
        return
    raise IndexMismatch(
        f"检索已拒绝：本进程持有的向量后端是「{actual}」，而库里登记的活动后端已经是"
        f"「{recorded}」——别的进程换过后端，这个进程手里的索引停在切换之前。"
        "继续用它检索不会报错，只会返回切换之前的内容。"
        "重启服务（重新装配后端）后再问；确实就该用这个后端，"
        "用 `python -m rag4prefa rebuild` 显式切回去。"
    )


def verify_ids(conn: sqlite3.Connection, store) -> dict:
    """逐 id 比对"片段 / 归档 / 实际索引"三方，返回可打印的差异报告。

    为什么每天要跑的不是这个：它要把索引里的主键全部拉出来，成本随条数线性增长，
    放在每次请求里不现实。日常用计数做必要条件（`IndexState.blockers`），
    这个命令做充分条件——条数相同但 id 不同（写坏了、切换只成功一半）只有它能看出来。
    """
    facts = repo.all_chunk_ids(conn)
    archived = repo.archived_chunk_ids(conn)
    # 切换标记要在这里一并报出来：逐 id 比对只能证明"三条链的 id 一样"，证明不了
    # "索引里的向量就是归档里的那些向量"。重建中断时 id 可以完全一致，若此处仍报 ok，
    # 运维人员拿着这个结论就再也不会去看真正的差距在哪。
    state = repo.get_index_state(conn)
    pending = None if state == repo.INDEX_READY else state
    try:
        indexed = store.ids()
    except Exception as exc:  # 后端读不出来也要给出可读结论，而不是抛栈
        return {
            "ok": False,
            "error": f"读不到向量索引的主键：{type(exc).__name__}: {exc}",
            "chunks": len(facts),
            "archived": len(archived),
            "pending_rebuild": pending,
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
        "ok": not any(problems.values()) and not pending,
        "chunks": len(facts),
        "archived": len(archived),
        "indexed": len(indexed),
        "counts": {k: len(v) for k, v in problems.items()},
        # 只给样例，避免几万条 id 把终端刷满；条数在 counts 里是全的
        "samples": {k: sorted(v)[:10] for k, v in problems.items() if v},
        "pending_rebuild": pending,
    }
