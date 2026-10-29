"""并行改写通道：原话独立保留，改写只能往池子里加东西。

第三十八轮的首版扩展是「原话 + 术语拼成一个串」。它看起来保住了原话（字都还在），
但拼完是一个**新文本**：查询向量变了、BM25 的词频变了，原来排在头部的片段可以
被新加的干扰词挤出去。实测 dense 0.400→0.467，可 5 道目标题一道都没救回，
挤掉的与救回的数量相当——那条路走不通。

这里钉住并行形态的两条承诺，它们正是拼串做不到的：

1. **原话通道的候选一条都不丢**：改写权重再大，原话 top-k 仍在融合结果里；
2. **权重真的生效**：同一组改写串，权重改变会改变融合顺序（不是写在参数里好看）。

第 3 条是防呆：`rewrites` 与 `expander` 同时给必须报错，而不是算出一组
没人能解释的数。
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag4prefa import ask, repo
from rag4prefa.adapters.vectors import NumpyStore, VectorItem
from rag4prefa.config import Settings
from rag4prefa.domain import Chunk, Filter
from tests.conftest import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("eval_script_parallel", ROOT / "scripts" / "eval.py")
assert _spec and _spec.loader
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)


def _seed(conn, texts: list[str], dim: int = FakeEmbedder.dim) -> tuple[NumpyStore, FakeEmbedder]:
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
    repo.set_index_meta(
        conn, model=embedder.model, dim=str(dim), backend="numpy", backend_id="numpy"
    )
    return store, embedder


TEXTS = [f"墙面悬挂较重物体应采用专用连接件 第{i}条" for i in range(20)]


def test_rrf_multi_scales_each_run_by_its_weight() -> None:
    """权重是乘在每一路贡献上的，不是只影响一路。"""
    dense = [("a", 1.0), ("b", 1.0)]
    sparse = [("b", 1.0), ("c", 1.0)]

    fused = ask.rrf_multi(((dense, 1.0), (sparse, 0.5)), k=0)

    # k=0 时第 1 名 1/1、第 2 名 1/2。
    assert fused["a"] == pytest.approx(1.0)
    assert fused["b"] == pytest.approx(0.5 + 0.5 * 1.0)
    assert fused["c"] == pytest.approx(0.5 * 0.5)


def test_rrf_equal_weight_lets_two_weak_hits_beat_one_strong_hit() -> None:
    """等权 RRF 下「两路都排第 31/30」会压过「单路排第 1」——这是数学性质，不是 bug。

    口语集实测（2026-10-29，50 题）：「卫生间地面要不要往地漏那边找坡？」的 gold
    在 dense 排第 1、BM25 不进前 50，融合后掉到第 22 名，于是 `hybrid` 整行
    Recall@5 只有 0.460，比纯 dense 的 0.600 还低 0.14。

    算一遍就明白：单路第 1 得 1/61≈0.0164；两路各自排 31 得 1/91+1/91≈0.0220。
    这个用例钉住的是「别把 hybrid 低于 dense 当成实现写错了」——真正的修法只有
    两种：给强通道更高权重，或者改成按名次取并集后再统一重排。改 rrf 公式本身
    修不了它（等权就是等权）。
    """
    dense = [("strong", 0.9)] + [(f"d{i}", 0.5) for i in range(29)] + [("weak", 0.4)]
    sparse = [(f"s{i}", 1.0) for i in range(29)] + [("weak", 0.9)]

    fused = ask.rrf_multi(((dense, 1.0), (sparse, 1.0)), k=60)

    assert fused["strong"] == pytest.approx(1 / 61)
    assert fused["weak"] == pytest.approx(1 / 91 + 1 / 90)
    assert fused["weak"] > fused["strong"]


def test_rrf_weight_can_rescue_the_strong_single_run() -> None:
    """同一个构造，把 dense 权重抬起来就能翻回来——说明权重是那个可动的旋钮。

    临界值不是 2.0：`w/61 > w/91 + 1/90` 解出 w > 2.06。2.0 仍然输（0.0328 vs
    0.0331），3.0 才赢。写成测试而不是注释，是因为"抬一点权重就够了"是个很容易
    想当然的结论，而它恰好差在临界值下面一点。
    """
    dense = [("strong", 0.9)] + [(f"d{i}", 0.5) for i in range(29)] + [("weak", 0.4)]
    sparse = [(f"s{i}", 1.0) for i in range(29)] + [("weak", 0.9)]

    fused = ask.rrf_multi(((dense, 3.0), (sparse, 1.0)), k=60)

    assert fused["strong"] > fused["weak"]


def test_parallel_keeps_every_original_candidate(conn, settings: Settings) -> None:
    """改写权重再大，原话通道的 top-k 也必须一条不落地留在融合结果里。

    这是并行形态相对"拼串"的唯一硬承诺。拼串时原话的字还在，但它已经不是
    那个查询了：向量与词频都变，原话命中的片段可以被挤出去。这里构造的改写串
    与语料毫不相干（纯干扰词），拼串下原话的排序会被彻底带偏，并行下不能。
    """
    store, embedder = _seed(conn, TEXTS)

    plain, _, _ = asyncio.run(
        ask.retrieve(
            conn, store, embedder, settings, question=TEXTS[0], filt=Filter(),
            recall_k=10, candidate_k=20,
        )
    )
    fused, _, _ = asyncio.run(
        ask.retrieve(
            conn, store, embedder, settings, question=TEXTS[0], filt=Filter(),
            recall_k=10, candidate_k=20,
            rewrites=("吊柜 吊顶 灯具 电视 完全不同的话题",), rewrite_weight=1.0,
        )
    )

    plain_ids = [h.chunk_id for h in plain]
    fused_ids = {h.chunk_id for h in fused}
    missing = [cid for cid in plain_ids if cid not in fused_ids]
    assert not missing, f"原话通道的候选被改写挤掉了：{missing}"


def test_rewrite_weight_changes_the_fused_order(conn, settings: Settings) -> None:
    """权重必须真的改变融合结果。

    防的是"参数传下去了、但谁也没用它"：那种实现里权重怎么调都得到同一份结果，
    而报告上会写着 weight=0.5，看起来像是做过参数选择。
    """
    store, embedder = _seed(conn, TEXTS)
    rewrite = "吊柜 吊顶 灯具 电视 完全不同的话题"

    def _run(weight: float) -> list[str]:
        hits, _, _ = asyncio.run(
            ask.retrieve(
                conn, store, embedder, settings, question=TEXTS[0], filt=Filter(),
                recall_k=10, candidate_k=20, rewrites=(rewrite,), rewrite_weight=weight,
            )
        )
        return [h.chunk_id for h in hits]

    assert _run(1.0) != _run(0.05), "权重从 1.0 调到 0.05 却拿到同样的顺序，说明它没生效"


def test_parallel_rejects_expander_combo(conn, settings: Settings) -> None:
    """`rewrites` 与 `expander` 是两种形态，同时给必须报错。

    静默取其中之一会算出一组"能跑但无法归因"的数：报告上写着并行，
    实际召回串是拼出来的，谁也说不清结果是哪种形态的。
    """
    store, embedder = _seed(conn, TEXTS)

    class _Expander:
        available = True

        def expand(self, question: str) -> str:
            return f"{question} 专用连接件"

    with pytest.raises(ValueError, match="不能同时给"):
        asyncio.run(
            ask.retrieve(
                conn, store, embedder, settings, question=TEXTS[0], filt=Filter(),
                expander=_Expander(), rewrites=("别的说法",),
            )
        )


def test_parallel_rejects_zero_weight(conn, settings: Settings) -> None:
    """权重为 0 不等于"这一路不生效"：分数贡献是 0，但候选仍会以 0 分占住池子尾部。"""
    store, embedder = _seed(conn, TEXTS)

    with pytest.raises(ValueError, match="rewrite_weight"):
        asyncio.run(
            ask.retrieve(
                conn, store, embedder, settings, question=TEXTS[0], filt=Filter(),
                rewrites=("别的说法",), rewrite_weight=0.0,
            )
        )


class _Expander:
    available = True

    def __init__(self, terms: str) -> None:
        self.terms = terms

    def expand(self, question: str) -> str:
        return f"{question} {self.terms}"


def _services(conn, settings: Settings) -> SimpleNamespace:
    store, embedder = _seed(conn, TEXTS)
    return SimpleNamespace(
        conn=conn,
        store=store,
        settings=settings,
        require_embedder=lambda: embedder,
    )


class _OverlapProbe:
    """检测"有没有两路查询同时跑"。

    这不是性能偏好，是正确性：这个进程里所有读路径共用**一条** sqlite 连接
    （`Embedder.embed_one` 读写 `embed_cache`、`NumpyStore.search` 读代次与归档、
    `repo.fts_search` 本身就是 SQL）。同一时刻多线程碰它，会抛
    `InterfaceError: bad parameter or other API misuse`，而且只落在一路上——
    报告上表现为"有效样本 14/15"，看上去像偶发。
    """

    def __init__(self) -> None:
        self._depth = 0
        self._lock = threading.Lock()
        self.overlapped = False

    def __enter__(self) -> None:
        with self._lock:
            self._depth += 1
            if self._depth > 1:
                self.overlapped = True

    def __exit__(self, *exc: object) -> bool:
        with self._lock:
            self._depth -= 1
        return False


class _SlowEmbedder:
    """包一层 sleep，把并发窗口撑开——否则 gather 也可能看不出重叠。"""

    def __init__(self, base: FakeEmbedder, probe: _OverlapProbe) -> None:
        self.base = base
        self.probe = probe
        self.model = base.model
        self.dim = base.dim

    def embed_one(self, text: str) -> list[float]:
        with self.probe:
            time.sleep(0.03)
            return self.base.embed_one(text)


def test_channels_are_queried_serially(conn, settings: Settings) -> None:
    """多通道必须一路一路查，不能 gather。

    "并行改写"这个名字说的是**候选来源并行**，不是执行并行。真正并发会踩上面
    那条单连接限制，而失败方式极隐蔽：只丢一路、均值照算。

    实现改回 `asyncio.gather` 时这条必红（sleep 制造的重叠窗口是确定的），
    所以它是防回归而不是描述现状。
    """
    store, embedder = _seed(conn, TEXTS)
    probe = _OverlapProbe()

    asyncio.run(
        ask.retrieve(
            conn, store, _SlowEmbedder(embedder, probe), settings,
            question=TEXTS[0], filt=Filter(), recall_k=10, candidate_k=20,
            rewrites=("吊柜 吊顶", "灯具 电视"), rewrite_weight=0.5,
        )
    )

    assert not probe.overlapped, "多路查询被并发执行了——单条 sqlite 连接撑不住"


def test_baseline_modes_ignore_rewrites(conn, settings: Settings) -> None:
    """基线行（hybrid/rerank）在并行配置下仍跑原话、不吃改写串。

    如果它们也跟着改写，「基线 vs 并行」的差值就没了分母——两行都变了，
    这正是第三十八轮"看不出收益来自哪"的翻版。
    """
    services = _services(conn, settings)
    question = TEXTS[0]
    rewrites = ("吊柜 吊顶 灯具",)

    plain = asyncio.run(_ev.hybrid_hits(services, question, Filter(), 5))
    parallel = asyncio.run(
        _ev.hybrid_hits(services, question, Filter(), 5, rewrites=rewrites, rewrite_weight=1.0)
    )
    baseline_with_rw = asyncio.run(
        _ev.retrieve_for(
            services, "hybrid", question, Filter(), 5, None, rewrites=rewrites
        )
    )

    assert [h.chunk_id for h in baseline_with_rw] == [h.chunk_id for h in plain], (
        "hybrid 这一行必须忽略 rewrites，否则它就不再是对照组"
    )
    assert [h.chunk_id for h in parallel] != [h.chunk_id for h in plain], (
        "hybrid_parallel 必须吃 rewrites，否则这一行等于白跑"
    )


def test_evaluate_parallel_keeps_recall_query_original(
    conn, settings: Settings, monkeypatch
) -> None:
    """并行评估里，召回串必须是**原话**，扩展串只能进 `rewrites`。

    这条断言的是整件事的定义：一旦召回串被换成扩展串，就退回拼串的老路了，
    而报告上仍然写着"并行"。

    **必须 spy 到真正传进检索的那两个串**，不能只看诊断 JSONL：诊断里记的是
    "算出什么改写串"，不是"拿它去查了什么"。变异测试里把 recall 串换成扩展串、
    诊断字段一字不变，断言照样全绿——那就是选错了断言对象。
    """
    services = _services(conn, settings)
    rows = [{"question": TEXTS[0], "gold_snippets": [TEXTS[0]]}]

    seen: list[dict] = []
    original = _ev.retrieve_for

    async def spy(services, mode, question, filt, k, keyword, *, rerank_query=None,
                  rewrites=(), rewrite_weight=0.5):
        seen.append({
            "mode": mode,
            "question": question,
            "rewrites": tuple(rewrites),
            "rerank_query": rerank_query,
        })
        return await original(
            services, mode, question, filt, k, keyword,
            rerank_query=rerank_query, rewrites=rewrites, rewrite_weight=rewrite_weight,
        )

    monkeypatch.setattr(_ev, "retrieve_for", spy)

    result = asyncio.run(
        _ev._evaluate(
            services, rows, 5, ["hybrid", "hybrid_parallel"],
            expander=_Expander("悬挂 连接件"), parallel=True,
        )
    )

    expanded = f"{TEXTS[0]} 悬挂 连接件"
    assert [s["mode"] for s in seen] == ["hybrid", "hybrid_parallel"]
    for call in seen:
        assert call["question"] == TEXTS[0], (
            f"{call['mode']} 的召回串被换成了扩展串，退回拼串形态了：{call['question']!r}"
        )
        assert call["rerank_query"] == TEXTS[0], "并行模式下重排恒读原话"

    # 改写串必须**同时**出现在两行里（谁吃谁不吃由上层的 handler 决定），
    # 这样才能保证"基线不吃"是 handler 的显式选择，而不是这里压根没传。
    assert [s["rewrites"] for s in seen] == [(expanded,), (expanded,)]
    assert result["diagnostics"][0]["rewrites"] == [expanded]
