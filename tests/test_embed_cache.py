"""真实 Embedder 的缓存路径必须可用——离线、不联网，只把网络调用换成确定性向量。

背景：此前 FakeEmbedder 完全绕过缓存，`embed_cache` 表漏建却没有任何测试发现，
真实链路一跑就会 "no such table"。这里覆盖命中/未命中/维度变化三种情形。
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array

import pytest

from rag4prefa.adapters.embed import Embedder, EmbeddingError
from rag4prefa.config import Settings

DIM = 8


def _vec(text: str, dim: int = DIM) -> list[float]:
    # 取整数值，保证经过 float32 缓存往返后仍然逐位相等，便于直接断言
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [float(b % 8) for b in digest[:dim]]


class CountingEmbedder(Embedder):
    """只替换 _call（网络调用），缓存与维度校验仍走真实代码。"""

    calls: list[list[str]] = []

    def _call(self, batch: list[str]) -> list[list[float]]:
        type(self).calls.append(list(batch))
        return [_vec(t) for t in batch]


@pytest.fixture
def emb_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"embedding_dim": DIM, "embedding_batch": 2})


@pytest.fixture
def embedder(conn: sqlite3.Connection, emb_settings: Settings) -> CountingEmbedder:
    CountingEmbedder.calls = []
    return CountingEmbedder(emb_settings, conn)


def test_second_call_hits_cache(embedder: CountingEmbedder, conn: sqlite3.Connection) -> None:
    first = embedder.embed(["缓粘结预应力的张拉时机"])
    second = embedder.embed(["缓粘结预应力的张拉时机"])

    assert len(CountingEmbedder.calls) == 1, "第二次应当命中缓存，不再调用模型"
    assert first == second
    count = conn.execute("SELECT COUNT(*) AS c FROM embed_cache").fetchone()["c"]
    assert count == 1


def test_only_missing_texts_are_sent(embedder: CountingEmbedder) -> None:
    embedder.embed(["陶土砖反打"])
    embedder.embed(["陶土砖反打", "悬挑结构", "机电管线"])

    assert [len(c) for c in CountingEmbedder.calls] == [1, 2], "已缓存的文本不该重复计费"


def test_cache_survives_new_instance(
    embedder: CountingEmbedder, conn: sqlite3.Connection, emb_settings: Settings
) -> None:
    embedder.embed(["百年建筑"])
    before = len(CountingEmbedder.calls)

    CountingEmbedder(emb_settings, conn).embed(["百年建筑"])
    assert len(CountingEmbedder.calls) == before, "缓存落库后应跨实例复用"


def test_dim_change_invalidates_cache(
    conn: sqlite3.Connection, emb_settings: Settings, embedder: CountingEmbedder
) -> None:
    sha = hashlib.sha256("维度变了".encode()).hexdigest()
    conn.execute(
        "INSERT INTO embed_cache (model, sha, dim, vec) VALUES (?, ?, ?, ?)",
        (emb_settings.embedding_model, sha, 4, array("f", [0.1, 0.2, 0.3, 0.4]).tobytes()),
    )
    conn.commit()

    vectors = embedder.embed(["维度变了"])

    assert len(CountingEmbedder.calls) == 1, "维度不一致的缓存必须作废重算"
    assert len(vectors[0]) == DIM


def test_dimension_mismatch_is_rejected(conn: sqlite3.Connection, emb_settings: Settings) -> None:
    class BadEmbedder(CountingEmbedder):
        def _call(self, batch: list[str]) -> list[list[float]]:
            return [[0.0] * (DIM + 1) for _ in batch]

    with pytest.raises(EmbeddingError, match="维度不一致"):
        BadEmbedder(emb_settings, conn).embed(["维度不对"])


def test_cache_table_exists_after_init(conn: sqlite3.Connection) -> None:
    """直接在 schema 层兜底：这张表必须由 init_db 建出来。"""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'embed_cache'"
    ).fetchone()
    assert row is not None
