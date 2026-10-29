"""受控词表接进**生产链路**：命中 → 多一条术语通道，重排读「原话 + 术语串」。

评估脚本里量出来的结论，只有在用户实际跑的那条链路上成立才算数。这一层最容易
"看起来接上了"：参数一路传下去、日志也打了，术语其实没进检索；或者术语进去了，
却把原话那一路挤掉——那就退回了已被证伪的拼串形态（0.400 → 0.467，5 道目标题
一道没救回）。所以断言只落在两件可观察的事实上：

1. `fuse_scores` 收到几路候选（融合前的通道数，不是"传了几个字符串"）；
2. 重排器**真正读到的查询串**（不是调用方传下去的选项）。

还有第三件事必须钉住：配了 `R4P_TERM_MAP` 却没把词表对象交给检索，要在请求上直接
失败。静默退化成"没有词表"时，界面、日志、指标全都正常，只有口语问句悄悄拿不到
术语通道——这类失效没有任何一处会报出来。
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag4prefa import ask
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.term_map import TermEntry, TermMap
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.api import create_app
from rag4prefa.config import Settings
from rag4prefa.domain import Filter, Hit
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services, build_services
from tests.conftest import FakeEmbedder

# 问题与词表同源：命题里出现「张拉 … 要点」，条文里是「张拉控制」「预应力张拉」。
# 用同一份语料既能保证检索有候选（否则重排压根不会被调用，断言会空转），
# 也保证词表确实命中——两件事都要真发生，这条链路才算被走到。
QUESTION = "张拉控制要点有哪些？"
TERMS = ("张拉控制", "预应力张拉")
JOINED = " ".join(TERMS)


def _map() -> TermMap:
    return TermMap([TermEntry("e1", (("张拉",), ("要点",)), TERMS, "某规范 6.1")])


class _RecordingReranker:
    """只记"读到了什么"，不排序——被断言的对象是查询串，不是排序质量。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        self.queries.append(query)
        return list(hits[:top_k])


class _FakeLLM:
    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        yield "好"


class _FuseSpy:
    """记下融合前有几路候选。两路（dense/sparse）× N 条通道。"""

    def __init__(self, original) -> None:
        self.channel_counts: list[int] = []
        self._original = original

    def __call__(self, runs, *, mode: str = "zscore", k: int = 60):
        self.channel_counts.append(len(runs))
        return self._original(runs, mode=mode, k=k)


def _seed(conn: sqlite3.Connection, settings: Settings, docx: Path) -> NumpyStore:
    store = NumpyStore(conn, FakeEmbedder.dim)
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx, project_name="甲项目", doc_year=2024
    )
    assert outcome.status == "indexed", outcome.error
    return store


async def _collect(gen) -> list[tuple[str, dict]]:
    return [item async for item in gen]


# ── 单元：两个纯函数 ────────────────────────────────────────


def test_build_rerank_query_forms() -> None:
    """三种形态的差别就一句话：原话在不在。`append` 是唯一"只做加法"的那个。"""
    assert ask.build_rerank_query(QUESTION, TERMS, mode="none") == QUESTION
    assert ask.build_rerank_query(QUESTION, TERMS, mode="terms") == JOINED
    assert ask.build_rerank_query(QUESTION, TERMS, mode="append") == f"{QUESTION} {JOINED}"
    # 没命中术语时三种形态都必须退回原话：重排看到的东西不能因为"词表没覆盖"而变
    for mode in ("none", "terms", "append"):
        assert ask.build_rerank_query(QUESTION, (), mode=mode) == QUESTION
    with pytest.raises(ValueError, match="未知的重排串形态"):
        ask.build_rerank_query(QUESTION, TERMS, mode="joined")


def test_term_channels_returns_rewrites_and_terms_separately(tmp_path: Path) -> None:
    """改写串喂检索、术语列表喂重排与解释，两者不能混成一个返回值。"""
    term_map = _map()
    rewrites, terms = ask.term_channels(term_map, QUESTION, form="joined")
    assert rewrites == (JOINED,)
    assert terms == list(TERMS)

    rewrites, terms = ask.term_channels(term_map, QUESTION, form="terms")
    assert rewrites == TERMS
    assert terms == list(TERMS)

    # 没配词表 / 没命中：都返回空，**不返回原话**（返回原话会让"没覆盖"伪装成"试过了"）
    assert ask.term_channels(None, QUESTION) == ((), [])
    assert ask.term_channels(term_map, "屋面防水怎么验收？") == ((), [])
    with pytest.raises(ValueError, match="未知的改写形态"):
        ask.term_channels(term_map, QUESTION, form="append")


# ── 问答链路：通道数与重排查询 ──────────────────────────────


def test_stream_answer_adds_one_channel_and_reranks_with_original_plus_terms(
    conn, settings: Settings, docx_file: Path, monkeypatch
) -> None:
    store = _seed(conn, settings, docx_file)
    reranker = _RecordingReranker()
    spy = _FuseSpy(ask.fuse_scores)
    monkeypatch.setattr(ask, "fuse_scores", spy)

    events = asyncio.run(
        _collect(
            ask.stream_answer(
                conn,
                store,
                FakeEmbedder(),
                _FakeLLM(),
                reranker,
                settings,
                question=QUESTION,
                filt=Filter(),
                top_k=3,
                term_map=_map(),
            )
        )
    )

    meta = next(data for event, data in events if event == "meta")
    assert spy.channel_counts == [4], "joined 形态下是两条通道（原话 + 术语串）→ 4 路候选"
    assert reranker.queries == [f"{QUESTION} {JOINED}"], "重排必须读到原话，术语只做加法"
    assert meta["terms"] == list(TERMS)
    assert meta["term_entries"] == [
        {"id": "e1", "intent": "unlabelled", "matched_by": "张拉 + 要点", "terms": list(TERMS)}
    ]


def test_stream_answer_without_term_map_keeps_single_channel(
    conn, settings: Settings, docx_file: Path, monkeypatch
) -> None:
    """不配词表时行为必须与"没有这一层"完全一致：一路通道、重排读原话。"""
    store = _seed(conn, settings, docx_file)
    reranker = _RecordingReranker()
    spy = _FuseSpy(ask.fuse_scores)
    monkeypatch.setattr(ask, "fuse_scores", spy)

    events = asyncio.run(
        _collect(
            ask.stream_answer(
                conn,
                store,
                FakeEmbedder(),
                _FakeLLM(),
                reranker,
                settings,
                question=QUESTION,
                filt=Filter(),
                top_k=3,
            )
        )
    )

    meta = next(data for event, data in events if event == "meta")
    assert spy.channel_counts == [2]
    assert reranker.queries == [QUESTION]
    assert meta["terms"] == [] and meta["term_entries"] == []


def test_stream_answer_with_uncovered_question_keeps_single_channel(
    conn, settings: Settings, docx_file: Path, monkeypatch
) -> None:
    """词表没命中时不能凭空多一条通道（那会把"没覆盖"变成"改写过了"）。"""
    store = _seed(conn, settings, docx_file)
    reranker = _RecordingReranker()
    spy = _FuseSpy(ask.fuse_scores)
    monkeypatch.setattr(ask, "fuse_scores", spy)

    question = "屋面防水怎么验收？"
    asyncio.run(
        _collect(
            ask.stream_answer(
                conn, store, FakeEmbedder(), _FakeLLM(), reranker, settings,
                question=question, filt=Filter(), top_k=3, term_map=_map(),
            )
        )
    )

    assert spy.channel_counts == [2]
    assert reranker.queries == [question]


def test_search_passes_terms_as_rewrites_with_configured_weight(
    conn, settings: Settings, monkeypatch
) -> None:
    """`search` 的两个细节：术语走 `rewrites`（原话不动），权重取配置而不是写死的 0.5。"""
    seen: list[dict] = []

    async def fake_retrieve(*args, **kwargs):
        seen.append(kwargs)
        return [Hit(chunk_id="c1", score=1.0, project_name="甲", text="片段")], Filter(), {}

    monkeypatch.setattr(ask, "retrieve", fake_retrieve)
    reranker = _RecordingReranker()
    settings = settings.model_copy(update={"term_rewrite_weight": 0.4})

    asyncio.run(
        ask.search(
            conn, None, None, settings,
            question=QUESTION, filt=Filter(), top_k=3,
            reranker=reranker, term_map=_map(),
        )
    )

    assert seen[0]["rewrites"] == (JOINED,)
    assert seen[0]["rewrite_weight"] == 0.4
    assert reranker.queries == [f"{QUESTION} {JOINED}"]


def test_search_form_terms_creates_one_channel_per_term(
    conn, settings: Settings, monkeypatch
) -> None:
    seen: list[tuple[str, ...]] = []

    async def fake_retrieve(*args, **kwargs):
        seen.append(kwargs["rewrites"])
        return [Hit(chunk_id="c1", score=1.0, project_name="甲", text="片段")], Filter(), {}

    monkeypatch.setattr(ask, "retrieve", fake_retrieve)
    settings = settings.model_copy(update={"term_map_form": "terms"})

    asyncio.run(
        ask.search(
            conn, None, None, settings,
            question=QUESTION, filt=Filter(), top_k=3, term_map=_map(),
        )
    )

    assert seen == [TERMS]


def test_search_skips_term_rerank_when_rerank_is_off(
    conn, settings: Settings, monkeypatch
) -> None:
    """关掉重排时术语照旧进召回（原话没被改动，那还是净增候选）。"""

    async def fake_retrieve(*args, **kwargs):
        return [Hit(chunk_id="c1", score=1.0, project_name="甲", text="片段")], Filter(), {}

    monkeypatch.setattr(ask, "retrieve", fake_retrieve)
    reranker = _RecordingReranker()

    asyncio.run(
        ask.search(
            conn, None, None, settings,
            question=QUESTION, filt=Filter(), top_k=3,
            reranker=reranker, use_rerank=False, term_map=_map(),
        )
    )

    assert reranker.queries == []


# ── 装配漏传必须炸，不能静默降级 ────────────────────────────


def test_configured_map_without_object_is_an_error(conn, settings: Settings) -> None:
    """配了路径却没传对象 = 装配漏了，不是"这次不用词表"。

    静默退回"没有词表"时，唯一的现象是"口语问句效果不如报告里写的"，而这在
    界面上与"这次问句本来就难"完全无法区分。
    """
    configured = settings.model_copy(update={"term_map": "rag4prefa/terms/public_to_standard.json"})

    with pytest.raises(ValueError, match="装配漏传"):
        asyncio.run(
            ask.search(conn, None, None, configured, question=QUESTION, filt=Filter())
        )
    with pytest.raises(ValueError, match="装配漏传"):
        asyncio.run(
            _collect(
                ask.stream_answer(
                    conn, None, None, _FakeLLM(), NoopReranker(), configured,
                    question=QUESTION, filt=Filter(),
                )
            )
        )


def test_search_refuses_mixing_llm_rewrites_with_term_channels(
    conn, settings: Settings
) -> None:
    """两套改写串同时注入，收益归不到任何一方——与评估脚本同一条规矩。"""
    with pytest.raises(ValueError, match="无法归因"):
        asyncio.run(
            ask.search(
                conn, None, None, settings,
                question=QUESTION, filt=Filter(), rewrites=("LLM 改写串",), term_map=_map(),
            )
        )


# ── 装配与健康检查 ─────────────────────────────────────────


def test_build_services_loads_the_map_once(tmp_path: Path) -> None:
    map_path = Path("rag4prefa/terms/public_to_standard.json")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        embedding_api_key="test",
        llm_api_key="test",
        vector_backend="numpy",
        term_map=str(map_path),
    )
    services = build_services(settings)
    try:
        assert services.term_map is not None and len(services.term_map) > 0
        assert services.term_map_error is None
        block = services.health()["term_map"]
        assert block["enabled"] is True
        assert block["entries"] == len(services.term_map)
        assert block["form"] == "joined"
        assert block["rerank"] == "append"
    finally:
        services.close()


def test_broken_map_degrades_but_says_why(tmp_path: Path) -> None:
    """坏词表不让服务起不来（它是可选增强项），但绝不能静默：

    `term_map=None` 与 `term_map_error` 必须同时出现——只降级不报错，等于让人
    以为词表在生效。
    """
    bad = tmp_path / "bad.json"
    bad.write_text("{ 这不是 JSON", encoding="utf-8")
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "data",
        embedding_api_key="test",
        llm_api_key="test",
        vector_backend="numpy",
        term_map=str(bad),
    )
    services = build_services(settings)
    try:
        assert services.term_map is None
        assert services.term_map_error and "JSON" in services.term_map_error.upper()
        block = services.health()["term_map"]
        assert block["enabled"] is False and block["rerank"] == "none"
        assert block["error"] == services.term_map_error
        assert any("受控词表未能加载" in note for note in services.notes)
    finally:
        services.close()


def test_missing_map_file_degrades_but_says_why(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        embedding_api_key="test",
        llm_api_key="test",
        vector_backend="numpy",
        term_map=str(tmp_path / "nope.json"),
    )
    services = build_services(settings)
    try:
        assert services.term_map is None and services.term_map_error
        assert services.health()["term_map"]["enabled"] is False
    finally:
        services.close()


# ── 两个 HTTP 入口都真的把词表传下去了 ──────────────────────


def _api_services(conn, settings: Settings, store, term_map: TermMap) -> Services:
    configured = settings.model_copy(
        update={"term_map": "rag4prefa/terms/public_to_standard.json"}
    )
    return Services(
        settings=configured,
        conn=conn,
        store=store,
        embedder=FakeEmbedder(),
        llm=_FakeLLM(),
        reranker=NoopReranker(),
        owns_conn=False,
        term_map=term_map,
    )


def test_api_search_and_chat_pass_the_term_map(
    conn, settings: Settings, docx_file: Path, monkeypatch
) -> None:
    """删掉任一处的 `term_map=services.term_map`，这里就会红。

    判据不是"响应里出现了术语"（那可能只是碰巧同字），而是**词表对象本身**有没有
    被交给 `term_channels`；再补一条"配了却没传就炸"的守卫，两处漏传都会当场暴露。
    """
    store = _seed(conn, settings, docx_file)
    term_map = _map()
    seen: list[TermMap | None] = []
    original = ask.term_channels

    def spy(term_map_arg, question, *, form="joined"):
        seen.append(term_map_arg)
        return original(term_map_arg, question, form=form)

    monkeypatch.setattr(ask, "term_channels", spy)

    with TestClient(create_app(_api_services(conn, settings, store, term_map))) as client:
        assert client.post(
            "/api/search", json={"question": QUESTION, "top_k": 3}
        ).status_code == 200
        with client.stream("POST", "/api/chat", json={"question": QUESTION, "top_k": 3}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

    assert seen == [term_map, term_map], "两个入口都要把词表传下去"
    assert "event: error" not in body
    assert "张拉控制" in body, "meta 事件里要能看到注入了哪几个术语"
