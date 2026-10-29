"""融合方式：为什么把等权 RRF 换成本路标准化后的加权和。

第四十二轮（2026-10-29）扩到 50 题才看清的问题：口语 50 题里 17 题的 gold
只有稠密这一路找得到，BM25 回的是另一批不相干的片段。两路几乎不相交时，
名次求和退化成"把展示位对半分"——dense 的第 k 名被挤到第 2k 名，`hybrid`
于是比纯 dense 低一截。这些用例钉住的是"换完之后关系真的翻过来了"，
而不是"函数能跑"。
"""

from __future__ import annotations

import asyncio

import pytest

from rag4prefa import ask, repo
from rag4prefa.adapters.vectors import NumpyStore, VectorItem
from rag4prefa.config import Settings
from rag4prefa.domain import Chunk, Filter
from tests.conftest import FakeEmbedder


def _two_runs():
    """复现口语集那一题的形状：dense 的第一名，BM25 完全查不到。

    `weak` 刻意放在两路各自的最末（第 30 名）：这样等权 RRF 里
    `1/90 + 1/90 ≈ 0.0222` 才会压过 `1/61 ≈ 0.0164`——这是"双路弱命中赢过
    单路强命中"的完整形状，少放一路就复现不出来。
    """
    dense = [("strong", 0.95)] + [(f"d{i:02d}", 0.50) for i in range(28)] + [("weak", 0.40)]
    sparse = [(f"s{i:02d}", 1.00) for i in range(29)] + [("weak", 0.90)]
    return dense, sparse


def test_strong_single_run_hit_beats_two_run_weak_match() -> None:
    """同一个构造：等权 RRF 里 weak 赢，本路标准化后 strong 赢。

    这不是"换个算法数值变好看了"，而是同一个判断翻了个方向：RRF 认为
    "被两路都命中"比"在本路排第一"更有说服力，而口语集上的实测是反的。
    """
    dense, sparse = _two_runs()

    rrf_scores = ask.rrf_multi(((dense, 1.0), (sparse, 1.0)), k=60)
    fused = ask.fuse_scores(((dense, 1.0), (sparse, 1.0)), mode="zscore")

    assert rrf_scores["weak"] > rrf_scores["strong"], "构造本身不成立"
    assert fused["strong"] > fused["weak"]


def test_standardisation_makes_fusion_immune_to_score_scale() -> None:
    """把 sparse 的所有分数乘 100，融合结果必须一模一样。

    这是 z-score 相对"直接加权原始分"的唯一区别所在：余弦在 0~1，BM25 能到几十，
    不标准化就等于让 BM25 单独决定顺序。去掉 `/std` 会让这条立刻红。
    """
    dense, sparse = _two_runs()
    scaled = [(cid, score * 100.0) for cid, score in sparse]

    a = ask.fuse_scores(((dense, 1.0), (sparse, 1.0)), mode="zscore")
    b = ask.fuse_scores(((dense, 1.0), (scaled, 1.0)), mode="zscore")

    assert a.keys() == b.keys()
    for chunk_id, value in a.items():
        assert b[chunk_id] == pytest.approx(value, rel=1e-9)


def test_candidate_below_its_run_mean_scores_zero_not_negative() -> None:
    """低于本路均值的候选得 0 分，不是负分。

    负分等于说"本路排第 50"比"本路没命中"更差。在稠密 + BM25 上无所谓，
    但并行改写通道会借它把原话的候选压到改写候选之后——那会破坏"改写只能加、
    不能减"这条不变量（第四十轮的核心结论）。两种写法在 81 题上指标相同，
    所以选不破坏不变量的这个。
    """
    dense, sparse = _two_runs()

    fused = ask.fuse_scores(((dense, 1.0), (sparse, 1.0)), mode="zscore")

    assert min(fused.values()) >= 0.0
    # dense 的最后一名、sparse 的最后一名都在本路均值以下
    assert fused["d27"] == 0.0
    assert fused["weak"] == 0.0


def test_run_with_identical_scores_gets_no_vote() -> None:
    """一路分数全同（std=0）时不参与——它说不出谁更靠前。

    BM25 只命中一批同分片段时就是这种情形。此时"名次"完全是列表顺序，
    拿它去加权等于凭空造一个判断。
    """
    dense = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    flat = [("x", 1.0), ("y", 1.0), ("z", 1.0)]

    with_flat = ask.fuse_scores(((dense, 1.0), (flat, 1.0)), mode="zscore")
    without = ask.fuse_scores(((dense, 1.0),), mode="zscore")

    # 候选仍在表里（不能整路丢掉），但一分没拿到
    assert with_flat["x"] == 0.0 and with_flat["y"] == 0.0 and with_flat["z"] == 0.0
    # 排序与"根本没有这一路"完全一致：全同分那一路确实没投票
    top = lambda scores: [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])][:3]  # noqa: E731
    assert top(with_flat) == top(without)


def test_all_runs_flat_falls_back_to_rank_fusion() -> None:
    """两路都退化时退回名次法：顺序确定、结果不为空。

    这是"全部跳过"那条分支的用途——不兜底的话融合分全是 0，下游拿到一张
    所有候选并列的分数表，顺序变成字典插入顺序，看起来有结果、其实没有排序。
    """
    dense = [("a", 1.0), ("b", 1.0)]
    sparse = [("c", 1.0), ("b", 1.0)]

    fused = ask.fuse_scores(((dense, 1.0), (sparse, 1.0)), mode="zscore")
    expected = ask.rrf_multi(((dense, 1.0), (sparse, 1.0)), k=60)

    assert fused == expected
    assert fused  # 不是空表


def test_weight_still_scales_each_runs_contribution() -> None:
    """权重仍然乘在每一路的贡献上——换刻度不等于把并行改写的权重丢了。"""
    dense = [("a", 0.9), ("b", 0.6)]
    sparse = [("b", 2.0), ("c", 1.0)]

    equal = ask.fuse_scores(((dense, 1.0), (sparse, 1.0)), mode="zscore")
    down = ask.fuse_scores(((dense, 1.0), (sparse, 0.1)), mode="zscore")

    # dense 里 a 是唯一高于均值的，sparse 里 b 是唯一高于均值的 → 同权时并列
    assert equal["a"] == pytest.approx(equal["b"])
    # 把 sparse 权重压到 0.1，b 只剩 0.1 份，排在 a 后面；c 本来就低于均值，得 0
    assert down["a"] > down["b"] > down["c"] == 0.0


def test_rrf_mode_is_exactly_the_old_behaviour() -> None:
    """`mode="rrf"` 必须与 `rrf_multi` 一模一样：它是回退开关，不能有一点点偏差。"""
    dense, sparse = _two_runs()

    assert ask.fuse_scores(((dense, 1.0), (sparse, 0.5)), mode="rrf", k=60) == (
        ask.rrf_multi(((dense, 1.0), (sparse, 0.5)), k=60)
    )


def test_unknown_mode_is_rejected() -> None:
    """写错融合方式要立刻报错，而不是悄悄按某一种算。"""
    with pytest.raises(ValueError, match="未知的融合方式"):
        ask.fuse_scores(((("a", 1.0),),), mode="borda")


def test_retrieve_reads_the_configured_fusion(conn, settings: Settings, monkeypatch) -> None:
    """端到端：`retrieve` 必须真的按 `settings.fusion` 融合，而不是写死一种。

    构造方式：让稠密只能给出 c000（BM25 完全查不到的那条），BM25 给出一批
    别的片段。等权 RRF 下 c000 会被那批弱命中挤下去，本路标准化后不会。
    """
    repo.upsert_project(conn, "甲项目")
    repo.upsert_document(
        conn, doc_id="d1", project_name="甲项目", filename="a.docx",
        content_sha="sha", doc_year=2024, status="indexed",
    )
    embedder = FakeEmbedder()
    texts = ["墙面悬挂较重物体应采用专用连接件", "无关片段甲", "无关片段乙"]
    chunks = [
        Chunk(
            id=f"c{i:03d}", doc_id="d1", ordinal=i, chapter_path=["第1章"], title="t",
            text=text, token_count=8, content_sha=f"s{i}",
        )
        for i, text in enumerate(texts)
    ]
    repo.insert_chunks(conn, chunks)
    store = NumpyStore(conn, FakeEmbedder.dim)
    query = embedder.embed_one(texts[0])
    store.upsert(
        [
            VectorItem(
                chunk_id=c.id, vector=query if i == 0 else embedder.embed_one(c.text),
                project_name="甲项目", project_code="", doc_year=2024,
                model=embedder.model,
            )
            for i, c in enumerate(chunks)
        ]
    )
    repo.set_index_meta(
        conn, model=embedder.model, dim=str(FakeEmbedder.dim), backend="numpy", backend_id="numpy"
    )

    # BM25 只回两条与问题无关的片段，且刻意不给 c000：这就是口语集的常态。
    monkeypatch.setattr(
        repo,
        "fts_search",
        lambda conn, query, limit, filt: [("c001", 5.0), ("c002", 4.0)][:limit],
    )

    def run(mode: str) -> list[str]:
        local = settings.model_copy(update={"fusion": mode})
        hits, _, _ = asyncio.run(
            ask.retrieve(
                conn, store, embedder, local, question=texts[0], filt=Filter(),
                recall_k=10, candidate_k=10,
            )
        )
        return [h.chunk_id for h in hits]

    assert run("zscore")[0] == "c000"
    assert run("rrf")[0] != "c000"
