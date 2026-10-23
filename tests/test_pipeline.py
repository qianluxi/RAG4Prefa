"""端到端：真实 docx → 解析 → 分块 → 向量（假）→ Milvus → 检索 → 组装提示词。

不联网也能验证整条流水线；有网时把 FakeEmbedder 换成真实 Embedder 即为生产路径。
"""

from __future__ import annotations

import pytest

from rag4prefa import ask, repo
from rag4prefa.adapters.vectors import BackendUnavailable, MilvusStore
from rag4prefa.config import Settings
from rag4prefa.domain import Filter
from rag4prefa.ingest import ingest_file
from rag4prefa.prompts import build_answer_prompt
from tests.conftest import FakeEmbedder


@pytest.fixture
def store(conn, tmp_path):
    try:
        return MilvusStore(str(tmp_path / "vectors.db"), "chunks_vec", FakeEmbedder.dim)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")


async def test_end_to_end_search(conn, settings: Settings, docx_file, store) -> None:
    embedder = FakeEmbedder()
    outcome = ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="课题甲", doc_year=2024
    )
    assert outcome.status == "indexed"

    hits, applied, timings = await ask.retrieve(
        conn, store, embedder, settings, question="悬挑构件的预应力做法", filt=Filter()
    )
    assert hits, "应召回片段"
    assert {"embed_ms", "dense_ms", "sparse_ms"} <= set(timings)
    assert all(h.project_name == "课题甲" for h in hits)
    assert all(h.filename == "方案.docx" for h in hits)

    prompt = build_answer_prompt("悬挑构件的预应力做法", hits[:3], "全库")
    assert prompt.count("[1]") >= 1
    assert "课题甲" in prompt


async def test_filter_limits_scope(conn, settings: Settings, docx_file, store) -> None:
    embedder = FakeEmbedder()
    ingest_file(conn, store, embedder, settings, path=docx_file, project_name="甲", doc_year=2024)
    ingest_file(conn, store, embedder, settings, path=docx_file, project_name="乙", doc_year=2019)

    hits, _, _ = await ask.retrieve(
        conn, store, embedder, settings, question="陶土砖饰面", filt=Filter(year_from=2020)
    )
    assert hits
    assert {h.project_name for h in hits} == {"甲"}


def test_question_parsing_adds_scope(conn, settings: Settings, docx_file, store) -> None:
    embedder = FakeEmbedder()
    ingest_file(
        conn, store, embedder, settings, path=docx_file, project_name="广州项目", doc_year=2024
    )
    known = repo.project_names(conn)
    assert known == ["广州项目"]

    effective = ask.build_filter("广州项目 2023年以后的做法", None, known)
    assert effective.projects == ["广州项目"]
    assert effective.year_from == 2023

    explicit = ask.build_filter("广州项目 2023年以后的做法", Filter(projects=["其他"]), known)
    assert explicit.projects == ["其他"]  # 请求显式条件优先，不被自动解析覆盖


async def test_switch_model_on_milvus_rebuilds_everything(
    conn, settings: Settings, docx_file, store
) -> None:
    """在真正的向量库上验证换模型：维度不同 → 集合被整体替换，检索立刻可用。"""
    from rag4prefa.ingest import reindex

    class Dim8Embedder(FakeEmbedder):
        model = "dim8-embedder"
        dim = 8

    ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx_file, project_name="课题甲", doc_year=2024
    )
    before = store.count()
    assert store.dim == FakeEmbedder.dim

    result = reindex(conn, store, Dim8Embedder(), settings)

    assert result["dim"] == 8
    assert store.dim == 8
    assert store.count() == before
    hits, _, _ = await ask.retrieve(
        conn, store, Dim8Embedder(), settings, question="悬挑构件的预应力做法", filt=Filter()
    )
    assert hits, "换模型重建后必须能立刻检索"
    assert {h.project_name for h in hits} == {"课题甲"}
    # 换回旧模型应被拒绝：索引已经是 8 维、8 维模型的向量空间
    with pytest.raises(Exception, match=r"维度|模型"):
        await ask.retrieve(
            conn, store, FakeEmbedder(), settings, question="悬挑", filt=Filter()
        )
