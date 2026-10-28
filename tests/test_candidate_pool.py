"""候选池深度：声明的深度必须等于实际拿到的条数。

2026-10-28 专家复审 P1：`eval._hybrid_pool_size(5)` 声明融合池 50 深，但
`ask.retrieve` 当时在返回前又按 `max(recall // 2, top_k * 3)` 截了一刀（50 → 25）。
`hybrid` 与 `rerank` 两行因此都只有 25 条候选，而 `dense_rerank` 走的是稠密直出的
50 条——"融合 vs 重排"的对照里混进了池深这个变量。

这里钉住三件事：
1. **生产默认不变**：不传 `candidate_k` 时仍是旧行为（半个 recall）；
2. **显式指定就足额返回**：`candidate_k=50` 必须拿到 50 条，不能被再截；
3. **评测的 hybrid / rerank 走足额池**：真实 `ask.retrieve` 路径下也必须是 50 条，
   而不是只有个参数写着 50。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag4prefa import ask, repo
from rag4prefa.adapters.vectors import NumpyStore, VectorItem
from rag4prefa.config import Settings
from rag4prefa.domain import Chunk, Filter
from tests.conftest import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("eval_script_pool", ROOT / "scripts" / "eval.py")
assert _spec and _spec.loader
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)

N = 60  # 必须比池深大，否则"足额"与"不够"分不开


def _seed(conn, dim: int = FakeEmbedder.dim) -> tuple[NumpyStore, FakeEmbedder]:
    """直接写 60 个片段 + 对应向量，不经过解析与 embedding 接口。"""
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
    embedder = FakeEmbedder()
    chunks = [
        Chunk(
            id=f"c{i:03d}",
            doc_id="d1",
            ordinal=i,
            chapter_path=["第1章"],
            title="t",
            text=f"正文 {i}",
            token_count=5,
            content_sha=f"s{i}",
        )
        for i in range(N)
    ]
    repo.insert_chunks(conn, chunks)
    store = NumpyStore(conn, dim)
    store.upsert(
        [
            VectorItem(
                chunk_id=c.id,
                vector=embedder.embed_one(c.text),
                project_name="甲项目",
                project_code="",
                doc_year=2024,
                model=embedder.model,
            )
            for c in chunks
        ]
    )
    # 检索准入要证明"索引就是归档这一版"：模型、维度、后端三者一起登记。
    repo.set_index_meta(
        conn, model=embedder.model, dim=str(dim), backend="numpy", backend_id="numpy"
    )
    return store, embedder


def test_default_pool_keeps_legacy_half_recall(conn, settings: Settings) -> None:
    """不传 `candidate_k` 时保持旧行为（半个 recall）。

    这条不是"期望 25 条"的偏好，而是把**生产链路的既有行为**钉住：
    想改默认值可以，但那是一次会影响线上问答的改动，必须先看见这条变红。
    """
    store, embedder = _seed(conn)

    hits, _, _ = asyncio.run(
        ask.retrieve(
            conn, store, embedder, settings, question="正文 3", filt=Filter(), recall_k=50
        )
    )

    assert len(hits) == max(50 // 2, settings.top_k * 3) == 25


def test_candidate_k_returns_the_requested_depth(conn, settings: Settings) -> None:
    """显式给了 `candidate_k`，融合结果就必须足额返回。"""
    store, embedder = _seed(conn)

    hits, _, _ = asyncio.run(
        ask.retrieve(
            conn,
            store,
            embedder,
            settings,
            question="正文 3",
            filt=Filter(),
            recall_k=50,
            candidate_k=50,
        )
    )

    assert len(hits) == 50, "声明 50 深却拿回不到 50 条，等于报告上的池深是假的"


def test_hybrid_pool_reaches_declared_depth(conn, settings: Settings) -> None:
    """评测里 `hybrid`/`rerank` 的真实候选池必须等于 `_hybrid_pool_size(k)`。

    这条走的是**真实的** `ask.retrieve`（上面那条只验参数、这条验行为）。
    修 P1 之前它返回 25：参数写着 50，实际被内部默认截断砍半。
    """
    store, embedder = _seed(conn)
    services = SimpleNamespace(
        conn=conn,
        store=store,
        settings=settings,
        require_embedder=lambda: embedder,
    )

    pool = asyncio.run(_ev.hybrid_pool(services, "正文 3", Filter(), 5))

    assert _ev._hybrid_pool_size(5) == 50
    assert len(pool) == _ev._hybrid_pool_size(5)


class _Expander:
    """把口语问句扩成「原话 + 术语」，并记录被问了什么。"""

    available = True

    def __init__(self, terms: str) -> None:
        self.terms = terms
        self.seen: list[str] = []

    def expand(self, question: str) -> str:
        self.seen.append(question)
        return f"{question} {self.terms}"


class _RecordingReranker:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def rerank(self, question: str, candidates, k: int):  # noqa: ANN001
        self.questions.append(question)
        return candidates[:k]


def _services(conn, settings: Settings, reranker) -> SimpleNamespace:
    store, embedder = _seed(conn)
    return SimpleNamespace(
        conn=conn,
        store=store,
        settings=settings,
        reranker=reranker,
        require_embedder=lambda: embedder,
    )


def test_expand_touches_recall_only_by_default(conn, settings: Settings) -> None:
    """实验 B：扩展只改召回串，重排仍读原话。

    这是"改写着召回"与"改写起重排"两件事能被分开的唯一前提。修之前
    `--expand` 把扩展串一路喂到 reranker，rerank 那行的变化就没法归因。
    """
    reranker = _RecordingReranker()
    services = _services(conn, settings, reranker)
    expander = _Expander("悬挂 专用连接件")
    rows = [{"question": "挂电视会不会掉", "gold_snippets": ["正文 3"]}]

    result = asyncio.run(
        _ev._evaluate(services, rows, 5, ["rerank"], expander=expander, expand_rerank=False)
    )

    assert reranker.questions == ["挂电视会不会掉"], "重排必须读原话"
    assert result["diagnostics"][0]["search_question"] == "挂电视会不会掉 悬挂 专用连接件", (
        "召回串必须是扩展后的，否则这条实验什么都没改"
    )


def test_expand_rerank_feeds_expanded_query(conn, settings: Settings) -> None:
    """实验 C：显式要求时，重排也读扩展串。"""
    reranker = _RecordingReranker()
    services = _services(conn, settings, reranker)
    expander = _Expander("悬挂 专用连接件")
    rows = [{"question": "挂电视会不会掉", "gold_snippets": ["正文 3"]}]

    asyncio.run(
        _ev._evaluate(services, rows, 5, ["rerank"], expander=expander, expand_rerank=True)
    )

    assert reranker.questions == ["挂电视会不会掉 悬挂 专用连接件"]


def _diag_file(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "diag.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"question": q, "search_question": s}, ensure_ascii=False)
            for q, s in rows
        ),
        encoding="utf-8",
    )
    return path


def test_replay_expander_reuses_recorded_terms(tmp_path: Path) -> None:
    """固定扩展串：读回来的必须与落盘的一字不差，且不再问 LLM。

    为什么重要：同一份数据集跑两次，LLM 给的扩展词可以完全不同（实测 15 题 0 题一致）。
    不复用的话，"只扩召回"与"连重排一起扩"两次实验比的其实是两个查询集。
    """
    path = _diag_file(tmp_path, [("挂电视会不会掉", "挂电视会不会掉 悬挂 专用连接件")])

    expander = _ev._replay_expander(path)

    assert expander.available
    assert expander.expand("挂电视会不会掉") == "挂电视会不会掉 悬挂 专用连接件"


def test_replay_expander_rejects_incomplete_file(tmp_path: Path) -> None:
    """数据集里的题在扩展文件里缺一条，就必须报错。

    静默退回原话会让"扩展组"里混进未扩展的题，而均值上完全看不出来——
    这正是"看起来做了对照、其实两组条件不同"的典型。
    """
    path = _diag_file(tmp_path, [("Q1", "Q1 术语")])

    with pytest.raises(ValueError, match="缺少"):
        _ev._replay_expander(path, ["Q1", "Q2"])
