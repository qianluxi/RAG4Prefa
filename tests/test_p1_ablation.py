"""2×2 消融脚本的判据：池深能派生、顺序无关、未覆盖题不许编池。

这份脚本的全部结论都建立在两条**可以被打破**的前提上，所以必须钉住：

1. **池深 25 的结果可以从 50 深池派生**——融合排序与 `candidate_k` 无关，25 池就是
   50 池的前 25 条；重排对池内每条独立打分、与池内顺序无关。两条里任何一条不成立，
   "一次调用算出两档池深"就会悄悄算错，而报告上只是两个看起来正常的数。
2. **没命中词表的题，四格退化成两格**（P0Q0 = P1Q0，Q1 不存在）。给它们编一个术语池
   会让"未覆盖题四格全同"变成脚本自己造的事实，而不是词表的事实。

另外钉住两件小事：多数票怎么算、`NoopReranker` 下必须拒绝跑（否则"重排查询没用"
是被造出来的结论）。
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from rag4prefa import repo
from rag4prefa.adapters.vectors import NumpyStore, VectorItem
from rag4prefa.domain import Chunk, Filter
from tests.conftest import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_abl = _load("term_ablation_under_test", ROOT / "scripts" / "term_ablation.py")
_ev = _load("eval_script_for_ablation", ROOT / "scripts" / "eval.py")


class _TableReranker:
    """按 id 查固定分。用来精确控制"哪条排第几"，不受文本内容干扰。

    排序用 stable sort 且**只按分数**：同分时保持池内顺序。这不是偷懒，而是真实
    远程重排的行为——分差 0.001~0.02 的两个候选先后会翻，正是它让同一池子连跑三次
    结果不同。
    """

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, int, int]] = []

    async def rerank(self, query: str, hits, top_k: int):  # noqa: ANN001
        self.calls.append((query, len(hits), top_k))
        ordered = sorted(hits, key=lambda h: -self.scores.get(h.chunk_id, 0.0))
        return ordered[:top_k]


class _OverlapReranker:
    """按 query 与正文的公共字符数打分。对 query 敏感，用来验证 Q 因子不是摆设。"""

    async def rerank(self, query: str, hits, top_k: int):  # noqa: ANN001
        def score(hit) -> int:  # noqa: ANN001
            return len(set(query) & set(hit.text))

        ordered = sorted(hits, key=lambda h: -score(h))
        return ordered[:top_k]


class _NoHitTermMap:
    """一条都不命中的词表：模拟"口语题没被覆盖"。"""

    def rewrite(self, question: str, *, form: str = "terms") -> list[str]:  # noqa: ARG002
        return []

    def explain(self, question: str, *, strict: bool = True) -> list[dict]:  # noqa: ARG002
        return []


def _seed(conn, texts: list[str]) -> tuple[NumpyStore, FakeEmbedder]:
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
            text=text,
            token_count=8,
            content_sha=f"s{i}",
        )
        for i, text in enumerate(texts)
    ]
    repo.insert_chunks(conn, chunks)
    store = NumpyStore(conn, embedder.dim)
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
    repo.set_index_meta(
        conn, model=embedder.model, dim=str(embedder.dim), backend="numpy", backend_id="numpy"
    )
    return store, embedder


def _services(conn, store, embedder, reranker, settings) -> SimpleNamespace:
    return SimpleNamespace(
        conn=conn,
        store=store,
        settings=settings,
        reranker=reranker,
        require_embedder=lambda: embedder,
    )


TEXTS = [f"第{i}条 墙面悬挂较重物体应采用专用连接件" for i in range(30)]


def test_pool_25_is_the_prefix_of_pool_50(conn, settings) -> None:
    """融合排序与 candidate_k 无关——这是"两档池深只算一次"的地基。"""
    store, embedder = _seed(conn, TEXTS)
    services = _services(conn, store, embedder, _TableReranker({}), settings)

    pool50 = asyncio.run(_abl._retrieve_pool(services, TEXTS[0], Filter(), (), 50, 0.5))
    pool25 = asyncio.run(_abl._retrieve_pool(services, TEXTS[0], Filter(), (), 25, 0.5))

    assert len(pool25) == 25
    assert [h.chunk_id for h in pool25] == [h.chunk_id for h in pool50[:25]]


def test_derived_depth_excludes_candidates_outside_shallow_pool() -> None:
    """浅池派生的核心判据：**进不了浅池的候选，不管重排给多高的分，都不许出现**。

    这一条必须直接调脚本的 `_ranks_at_depths`。第一版测试在用例里把派生逻辑又写了
    一遍，于是"把 `pool[:depth]` 改成 `pool`"这个变异从断言下面溜了过去——9 条全绿。
    测试自己重写实现，等于没测实现。
    """

    def hits(ids: list[str]) -> list[SimpleNamespace]:
        return [SimpleNamespace(chunk_id=i, text=i) for i in ids]

    pool = hits([f"c{i:02d}" for i in range(50)])
    full = list(reversed(pool))  # 重排全序：c49 第 1、c00 第 50
    ranks = _abl._ranks_at_depths(full, pool, {"c49"}, (25, 50))
    assert ranks[50] == 1
    assert ranks[25] is None, "c49 在池 26~50 条的位置上，浅池里根本没有它"

    ranks = _abl._ranks_at_depths(full, pool, {"c00"}, (25, 50))
    assert ranks[50] == 50
    assert ranks[25] == 25


def test_derived_depth_matches_a_direct_run(conn, settings) -> None:
    """从 50 深池过滤出的名次，必须与"直接拿 25 池重排"一致。

    这一条测的是另一层前提——融合排序与 `candidate_k` 无关，深池的前 N 条就是浅池。
    与上一条合起来，才是"两档池深只跑一次重排"的完整依据。
    """
    store, embedder = _seed(conn, TEXTS)
    scores = {f"c{i:03d}": float(i) for i in range(30)}
    reranker = _TableReranker(scores)
    services = _services(conn, store, embedder, reranker, settings)

    question = TEXTS[0]
    pool50 = asyncio.run(_abl._retrieve_pool(services, question, Filter(), (), 50, 0.5))
    pool25 = pool50[:25]
    gold = {"c028"}

    full = asyncio.run(reranker.rerank(question, pool50, len(pool50)))
    derived = _abl._ranks_at_depths(full, pool50, gold, (25, 50))

    direct_full = asyncio.run(reranker.rerank(question, pool25, len(pool25)))
    assert derived[25] == _abl._first_gold_rank(direct_full, gold)
    assert derived[50] == _abl._first_gold_rank(full, gold)


def test_rerank_order_invariance_needs_unique_scores() -> None:
    """顺序无关只在分数唯一时成立；同分就是"谁先进池谁在前"。

    把它写成用例而不是注释，是因为这条性质决定了报告里"名次翻动 N/n"那一栏怎么读：
    翻动不是模型在思考，是同分候选的先后。
    """

    class _Hit:
        def __init__(self, chunk_id: str) -> None:
            self.chunk_id = chunk_id
            self.text = chunk_id

    hits = [_Hit("a"), _Hit("b"), _Hit("c")]
    unique = _TableReranker({"a": 0.1, "b": 0.9, "c": 0.5})
    assert [h.chunk_id for h in asyncio.run(unique.rerank("q", hits, 3))] == ["b", "c", "a"]
    reversed_hits = list(reversed(hits))
    assert [h.chunk_id for h in asyncio.run(unique.rerank("q", reversed_hits, 3))] == [
        "b",
        "c",
        "a",
    ]

    tied = _TableReranker({"a": 0.5, "b": 0.5, "c": 0.5})
    assert [h.chunk_id for h in asyncio.run(tied.rerank("q", hits, 3))] == ["a", "b", "c"]
    assert [h.chunk_id for h in asyncio.run(tied.rerank("q", reversed_hits, 3))] == [
        "c",
        "b",
        "a",
    ]


def test_uncovered_row_has_no_term_pool(conn, settings) -> None:
    """没命中词表的题只能有 P0Q0 / P1Q0，不许出现 Q1 或独立的术语池。"""
    store, embedder = _seed(conn, TEXTS)
    services = _services(conn, store, embedder, _OverlapReranker(), settings)

    row = {"question": TEXTS[0], "gold_chunk_ids": ["c000"]}
    result = asyncio.run(
        _abl.measure(
            services, _ev, [row], _NoHitTermMap(), top_k=5, depths=(10,), repeat=1
        )
    )

    cells = result["records"][0]["cells"]
    assert set(cells) == {"P0Q0", "P1Q0"}
    assert result["records"][0]["covered"] is False
    # 四格里只有两格有定义；`summary` 对缺格返回 n=0，不要让它冒充"测过了"
    block = _abl.summarise(result, group="all")["all"]["10"]
    assert block["P0Q0"]["n"] == 1
    assert block["P0Q1"]["n"] == 0


def test_covered_row_fills_all_four_cells(conn, settings) -> None:
    """命中词表的题必须四格齐全，且池真的多了一路——否则消融无从谈起。"""

    class _HitMap:
        def rewrite(self, question: str, *, form: str = "terms") -> list[str]:  # noqa: ARG002
            return ["专用连接件"]

        def explain(self, question: str, *, strict: bool = True) -> list[dict]:  # noqa: ARG002
            return [{"id": "anchor", "matched_by": "悬挂", "terms": ["专用连接件"]}]

    store, embedder = _seed(conn, TEXTS)
    services = _services(conn, store, embedder, _OverlapReranker(), settings)

    row = {"question": "墙上挂重东西要用什么件", "gold_chunk_ids": ["c000"]}
    result = asyncio.run(
        _abl.measure(services, _ev, [row], _HitMap(), top_k=5, depths=(10,), repeat=2)
    )

    rec = result["records"][0]
    assert rec["covered"] is True
    assert set(rec["cells"]) == {"P0Q0", "P0Q1", "P1Q0", "P1Q1"}
    assert all(len(runs[10]) == 2 for runs in rec["cells"].values())


def test_query_factor_has_teeth(conn, settings) -> None:
    """Q 因子必须真的能改变结果，否则四格消融测到的全是池的差。

    构造：gold 靠术语串才排得上来。只换重排查询（池不动）就应该翻盘——
    这正是 P0Q1 那一格存在的理由。
    """
    texts = ["悬挂 重物", "专用 连接件", "别的 条款"]
    store, embedder = _seed(conn, texts)
    services = _services(conn, store, embedder, _OverlapReranker(), settings)

    question = "悬挂重物"
    pool = asyncio.run(_abl._retrieve_pool(services, question, Filter(), (), 10, 0.5))
    reranker = _OverlapReranker()
    plain = asyncio.run(reranker.rerank(question, pool, len(pool)))
    appended = asyncio.run(reranker.rerank(f"{question} 专用连接件", pool, len(pool)))

    gold = {h.chunk_id for h in pool if h.text == "专用 连接件"}
    assert _abl._first_gold_rank(plain, gold) is None or _abl._first_gold_rank(plain, gold) > 1
    assert _abl._first_gold_rank(appended, gold) == 1


def test_majority_hit_and_rank() -> None:
    """两次重复里命中一次算命中；三次里命中两次算。名次取众数、平局取最小。

    偶数重复下的平局（1/2）算命中——宁可把边界题记成"进过前 5"，也不记账成
    "没进"，因为后者会把一次真实的命中抹掉。默认 `--repeat 3` 时不会出现平局，
    这条用例守着的是"有人把重复改成偶数"那一天。
    """
    assert _abl._hit([1, None, 1], 5) is True
    assert _abl._hit([1, 7, 7], 5) is False
    assert _abl._hit([1, 1, 1], 5) is True
    assert _abl._hit([1, None], 5) is True
    assert _abl._hit([None, None], 5) is False
    assert _abl._majority_rank([3, 3, 7]) == 3
    assert _abl._majority_rank([3, 7]) == 3
    assert _abl._majority_rank([None, None]) is None


def test_noop_reranker_is_refused() -> None:
    """没有重排服务时 Q 因子不可测，必须拒绝而不是给出"重排查询无用"的结论。"""

    class NoopReranker:  # 只要类名对得上——守卫看的是类名，避免 import 一整个后端
        pass

    assert _abl._reranker_problem(SimpleNamespace(reranker=NoopReranker())) is not None
    assert _abl._reranker_problem(SimpleNamespace(reranker=_TableReranker({}))) is None


def test_render_handles_uncovered_only(conn, settings) -> None:
    """只有未覆盖题时，表 B 仍要出现，表 A 不能凭空出现。"""
    store, embedder = _seed(conn, TEXTS)
    services = _services(conn, store, embedder, _OverlapReranker(), settings)
    row = {"question": TEXTS[0], "gold_chunk_ids": ["c000"]}
    result = asyncio.run(
        _abl.measure(
            services, _ev, [row], _NoHitTermMap(), top_k=5, depths=(10,), repeat=1
        )
    )

    text = _abl.render(result)
    assert "A. 命中词表的题" not in text
    assert "B. 全部题" in text
