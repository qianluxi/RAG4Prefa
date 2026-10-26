"""重排的开关语义：装了真实重排就默认用，装的是 Noop 就当没这回事。

这条规则有三处出口——`/api/chat` 的默认值、CLI 的 `--rerank`、`/api/health` 的
`rerank_configured`——它们必须由同一个判断决定。分开写会导致"界面上的开关是灰的、
后台其实在重排"这种自相矛盾的状态，而那种状态一旦出现在答辩现场，很难临场解释。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from rag4prefa import ask
from rag4prefa.adapters.rerank import NoopReranker, RemoteReranker
from rag4prefa.config import Settings
from rag4prefa.domain import Filter, Hit


class _CountingReranker:
    """记录被调用过几次、以及拿到多少条候选，不真的排序。"""

    def __init__(self) -> None:
        self.calls = 0
        self.pool_sizes: list[int] = []

    async def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        self.calls += 1
        self.pool_sizes.append(len(hits))
        return list(hits[:top_k])


class _FakeLLM:
    async def stream(self, system: str, prompt: str) -> AsyncIterator[str]:
        yield "好"


class _CountingNoop(NoopReranker):
    """是 NoopReranker 的子类，但会记录自己被调过——用来验证"压根没调"。"""

    def __init__(self) -> None:
        self.calls = 0

    async def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        self.calls += 1
        return list(hits[:top_k])


def _hit(n: int) -> Hit:
    return Hit(chunk_id=f"c{n}", score=1.0 - n / 10, project_name="甲", text=f"片段{n}")


async def _collect(reranker, use_rerank, tmp_path: Path, monkeypatch) -> list[str]:
    """跑一遍 stream_answer，返回事件名序列。"""

    async def fake_retrieve(*args, **kwargs):
        return [_hit(1), _hit(2), _hit(3)], Filter(), {}

    monkeypatch.setattr(ask, "retrieve", fake_retrieve)
    settings = Settings(data_dir=tmp_path, top_k=2, rerank_url="")
    events = []
    async for event, _payload in ask.stream_answer(
        None,  # type: ignore[arg-type]  retrieve 已被替换，用不到 conn/store/embedder
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        _FakeLLM(),  # type: ignore[arg-type]
        reranker,
        settings,
        question="张拉控制要点",
        filt=Filter(),
        use_rerank=use_rerank,
    ):
        events.append(event)
    return events


def test_rerank_available_only_for_real_rerankers() -> None:
    assert ask.rerank_available(NoopReranker()) is False
    assert ask.rerank_available(None) is False
    assert ask.rerank_available(_CountingReranker()) is True


@pytest.mark.asyncio
async def test_default_follows_the_installed_reranker(tmp_path: Path, monkeypatch) -> None:
    """不显式指定时：装了真重排就调，装的是 Noop 就一次都不调。

    Noop 那一半必须用"子类计数"来验：NoopReranker 本身被调用也不会改变顺序，
    只看最终结果的话，调没调完全看不出来——那种断言永远绿。
    """
    real = _CountingReranker()
    await _collect(real, None, tmp_path, monkeypatch)
    assert real.calls == 1, "装了真实重排器，默认就该重排"

    noop = _CountingNoop()
    await _collect(noop, None, tmp_path, monkeypatch)
    assert noop.calls == 0, "Noop 意味着没有重排服务，默认就不该白发一次调用"


@pytest.mark.asyncio
async def test_explicit_true_still_calls_the_noop(tmp_path: Path, monkeypatch) -> None:
    """显式要求重排时，Noop 也会被调一次——开关是你自己按的，代码不替你判断。"""
    noop = _CountingNoop()
    await _collect(noop, True, tmp_path, monkeypatch)
    assert noop.calls == 1


@pytest.mark.asyncio
async def test_explicit_false_still_turns_rerank_off(tmp_path: Path, monkeypatch) -> None:
    real = _CountingReranker()
    await _collect(real, False, tmp_path, monkeypatch)
    assert real.calls == 0, "显式关掉时不能偷偷重排"


@pytest.mark.asyncio
async def test_explicit_true_forces_rerank(tmp_path: Path, monkeypatch) -> None:
    real = _CountingReranker()
    events = await _collect(real, True, tmp_path, monkeypatch)
    assert real.calls == 1
    assert events[0] == "meta"


def test_remote_reranker_is_built_from_url(tmp_path: Path) -> None:
    """配置了 URL 就该装出 RemoteReranker——这是"默认开启"的另一半前提。"""
    from rag4prefa.adapters.rerank import build_reranker

    assert isinstance(build_reranker(Settings(data_dir=tmp_path, rerank_url="")), NoopReranker)
    built = build_reranker(
        Settings(data_dir=tmp_path, rerank_url="https://example.invalid/v1/rerank")
    )
    assert isinstance(built, RemoteReranker)


@pytest.mark.asyncio
async def test_search_reranks_the_full_pool_before_truncating(tmp_path: Path, monkeypatch) -> None:
    """`ask.search` 必须先重排、后截断。

    第一版 CLI 是自己先调 `search(top_k=3)` 拿到 3 条、再把这 3 条交给重排——
    重排只在一小撮里挑，日志里调用记录、分数变化样样都有，召回却一点不会变好。
    真实查询上验证过：正确答案在融合结果里排第 9，只有把 25 条候选全交给重排，
    它才能升到第 1；只给 3 条的话它根本不在里面。
    """
    pool = [_hit(i) for i in range(1, 26)]

    async def fake_retrieve(*args, **kwargs):
        return list(pool), Filter(), {}

    monkeypatch.setattr(ask, "retrieve", fake_retrieve)
    settings = Settings(data_dir=tmp_path, top_k=3, rerank_url="")
    reranker = _CountingReranker()

    hits, _applied, _timings = await ask.search(
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        settings,
        question="双面叠合墙的修正系数",
        filt=Filter(),
        top_k=3,
        reranker=reranker,
    )

    assert reranker.pool_sizes == [25], f"重排应看到完整候选池，实际 {reranker.pool_sizes}"
    assert len(hits) == 3
