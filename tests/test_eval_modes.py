"""评测的召回方式：每一行之间"唯一的变量"必须真的只有一个。

报告的价值全在"两行相减就是那个变量的收益"这句话上。一旦某一行顺手换了候选池深度、
或者干脆没走重排，减出来的差值就会被读成另一个结论——而报告里看起来完全正常。

2026-10-28 第一版基线撞到的正是这件事：`rerank` 恒等于 `hybrid` 再重排，于是口语集上
`dense 0.467 > hybrid 0.333 = rerank 0.333`，分不清"融合把候选池拉低了"还是"重排没用"。
加了 `dense_rerank` 之后，这两件事各自有了对照，这里钉住它的接线：
**池深与 `rerank` 一样、"被重排的"确实是整个池、`dense` 那一行不许碰重排器。**
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag4prefa import repo
from rag4prefa.db import connect, init_db
from rag4prefa.domain import Chunk, Filter
from tests.conftest import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("eval_script_modes", ROOT / "scripts" / "eval.py")
assert _spec and _spec.loader
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

# 池子要比 k 深得多，才能区分"整池重排"与"只重排截断后的 k 条"。
POOL = 60
CHUNK_IDS = [f"c{i:02d}" for i in range(POOL)]


class _Store:
    """只记录"被要了多少条"，这正是两份报告可比性的关键。"""

    def __init__(self, pairs: list[tuple[str, float]]) -> None:
        self.pairs = pairs
        self.limits: list[int] = []

    def count(self) -> int:
        return len(self.pairs)

    def search(self, vector, *, limit: int, filt=None):  # noqa: ANN001, ARG002
        self.limits.append(limit)
        return self.pairs[:limit]


class _Reranker:
    def __init__(self) -> None:
        self.sizes: list[int] = []

    async def rerank(self, question: str, candidates, k: int):  # noqa: ANN001, ARG002
        self.sizes.append(len(candidates))
        return candidates[:k]


@pytest.fixture()
def services(tmp_path: Path):
    db = tmp_path / "modes.db"
    init_db(db)
    conn = connect(db)
    repo.upsert_project(conn, "甲项目")
    repo.upsert_document(
        conn,
        doc_id="d1",
        project_name="甲项目",
        filename="a.docx",
        content_sha="sha",
        doc_year=2024,
        status="indexed",
    )
    repo.insert_chunks(
        conn,
        [
            Chunk(
                id=chunk_id,
                doc_id="d1",
                ordinal=i,
                chapter_path=["第1章"],
                title="t",
                text=f"正文 {i}",
                token_count=5,
                content_sha=f"s{i}",
            )
            for i, chunk_id in enumerate(CHUNK_IDS)
        ],
    )
    store = _Store([(chunk_id, 0.9 - i / 100) for i, chunk_id in enumerate(CHUNK_IDS)])
    reranker = _Reranker()
    fake = SimpleNamespace(
        conn=conn,
        store=store,
        reranker=reranker,
        require_embedder=lambda: FakeEmbedder(),
    )
    yield fake
    conn.close()


def test_dense_rerank_feeds_the_whole_pool_to_the_reranker(services) -> None:
    """`dense_rerank` 的候选池深度必须与 `rerank` 一致，且整池都过一遍重排。

    池深不同，"有没有重排"就不是唯一变量了：拿 50 条候选重排、与拿 5 条候选重排，
    差值里混着"多给了模型多少选择"。这正是旧实现里 `+0.033` 假收益的来路。
    """
    hits = asyncio.run(
        _eval.retrieve_for(services, "dense_rerank", "问一句", Filter(), 5, None)
    )

    assert services.store.limits == [_eval._hybrid_pool_size(5)]
    assert services.reranker.sizes == [_eval._hybrid_pool_size(5)], (
        "交给重排的应当是整池，不是截断后的 k 条"
    )
    assert len(hits) == 5, "重排后的结果要被截到 k"


def test_dense_alone_never_touches_the_reranker(services) -> None:
    """纯向量那一行不许碰重排器。

    两个都不动重排，差值才等于"召回方式本身的差别"；`dense` 若顺手也重排一次，
    `dense` 与 `dense_rerank` 两行就都成了"重排 + 某召回"，对照失效。
    """
    hits = asyncio.run(_eval.retrieve_for(services, "dense", "问一句", Filter(), 5, None))

    assert services.reranker.sizes == []
    assert services.store.limits == [5], "纯向量只要 top-k，不需要深池"
    assert len(hits) == 5


def test_modes_are_all_dispatchable() -> None:
    """`MODES` 与分派表必须一一对应。

    两边的错法各有各的难发现：`MODES` 里多一项而没有实现，报告里会多出一行、
    静默串到兜底分支，数字与 `rerank` 一模一样；实现多一个而 `MODES` 里没有，
    那段代码永远不跑，改坏了也没人知道。
    """
    assert set(_eval.MODES) == set(_eval._HANDLERS)
    assert len(_eval.MODES) == len(set(_eval.MODES)), "MODES 里有重复项"
