"""检索链路里术语扩展的集成测试：扩展串必须喂给 dense 与 sparse 两个通道。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from rag4prefa import ask, repo
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.domain import Filter
from rag4prefa.ingest import ingest_file
from tests.conftest import FakeEmbedder
from tests.test_query_expand import FakeLLM


class RecordingEmbedder(FakeEmbedder):
    """记录 embed_one 收到的每一段文本。"""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def embed_one(self, text: str) -> list[float]:
        self.seen.append(text)
        return super().embed_one(text)


def _seed(conn, store, settings: Settings, docx: Path) -> None:
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx, project_name="甲项目", doc_year=2024
    )
    assert outcome.status == "indexed"


def test_expanded_terms_feed_both_dense_and_sparse(
    conn, settings: Settings, docx_file: Path, monkeypatch
) -> None:
    """术语扩展之后，dense 用扩展串算向量、sparse 用扩展串查 BM25。

    只验证一件事：扩展不是摆设——它真的改变了两个通道吃进去的查询。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)

    embedder = RecordingEmbedder()
    expander = ask.QueryExpander(settings, FakeLLM(result="悬挂较重物体 专用连接件"))

    # 观察 sparse 通道收到的查询串
    sparse_queries: list[str] = []
    real_fts = repo.fts_search

    def spy_fts(conn, query, *, limit, filt=None):
        sparse_queries.append(query)
        return real_fts(conn, query, limit=limit, filt=filt)

    monkeypatch.setattr(repo, "fts_search", spy_fts)

    hits, _, timings = asyncio.run(
        ask.retrieve(
            conn,
            store,
            embedder,
            settings,
            question="挂电视吊柜会不会掉下来？",
            filt=Filter(),
            expander=expander,
        )
    )

    # dense 通道：embed_one 收到的应该是「原话 + 术语」
    assert embedder.seen, "embed 一次都没被调用"
    assert embedder.seen[0] == "挂电视吊柜会不会掉下来？ 悬挂较重物体 专用连接件"
    # sparse 通道：fts_search 收到的也应该是「原话 + 术语」
    assert sparse_queries, "fts_search 一次都没被调用"
    assert sparse_queries[0] == "挂电视吊柜会不会掉下来？ 悬挂较重物体 专用连接件"
    # 扩展耗时被记进 timings
    assert "expand_ms" in timings


def test_retrieve_without_expander_is_unchanged(
    conn, settings: Settings, docx_file: Path
) -> None:
    """不传 expander 时，行为与加扩展之前完全一致：embed 吃原话，无 expand_ms。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)

    embedder = RecordingEmbedder()
    hits, _, timings = asyncio.run(
        ask.retrieve(
            conn,
            store,
            embedder,
            settings,
            question="悬挑结构的张拉控制要点",
            filt=Filter(),
        )
    )
    assert embedder.seen[0] == "悬挑结构的张拉控制要点"
    assert "expand_ms" not in timings


def test_retrieve_with_noop_expander_embeds_original(
    conn, settings: Settings, docx_file: Path
) -> None:
    """传了 expander 但它不可用（无 LLM）时，等于没扩展。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)

    embedder = RecordingEmbedder()
    expander = ask.QueryExpander(settings, None)  # available=False
    hits, _, timings = asyncio.run(
        ask.retrieve(
            conn,
            store,
            embedder,
            settings,
            question="悬挑结构的张拉控制要点",
            filt=Filter(),
            expander=expander,
        )
    )
    assert embedder.seen[0] == "悬挑结构的张拉控制要点"
    assert "expand_ms" not in timings
