"""评估流程的集成测试：真实 Embedder + 本地假服务 + 真实 SQLite，不联网。

指标公式本身由 test_eval_metrics.py 钉住，这里验证的是**链路**：
标注怎么换算成 gold、四种召回方式怎么跑、报告里能不能看出数据问题。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag4prefa import repo
from rag4prefa.adapters.embed import Embedder
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import NumpyStore
from rag4prefa.config import Settings
from rag4prefa.db import connect, init_db
from rag4prefa.domain import Filter, Hit
from rag4prefa.ingest import ingest_file
from rag4prefa.services import Services
from tests import mock_openai

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_script2", ROOT / "scripts" / "eval.py")
assert _spec and _spec.loader
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


@pytest.fixture
def services(tmp_path: Path) -> Iterator[Services]:
    server = mock_openai.serve()
    settings = Settings(
        data_dir=tmp_path,
        embedding_base_url=mock_openai.base_url(server),
        embedding_api_key="sk-mock",
        embedding_model="mock-embed",
        embedding_dim=mock_openai.DIM,
        llm_base_url=mock_openai.base_url(server),
        llm_api_key="sk-mock",
        llm_model="mock-chat",
        vector_backend="numpy",
        chunk_max_tokens=120,
        chunk_min_tokens=30,
        # 显式关掉重排：Settings 会读本机 .env，开发机上开了重排就会让"未启用重排"
        # 那条断言语义反转。CI 上没有 .env，这个坑只在本地复现。
        rerank_url="",
        rerank_api_key="",
    )
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)
    svc = Services(
        settings=settings,
        conn=conn,
        store=NumpyStore(conn, mock_openai.DIM),
        embedder=Embedder(settings, conn),
        llm=None,
        reranker=NoopReranker(),
    )
    try:
        yield svc
    finally:
        conn.close()
        server.shutdown()


def _seed(svc: Services, docx_file: Path) -> list[str]:
    outcome = ingest_file(
        svc.conn,
        svc.store,
        svc.require_embedder(),
        svc.settings,
        path=docx_file,
        project_name="评估项目",
        doc_year=2024,
    )
    assert outcome.status == "indexed"
    return [c for c in repo.chunk_ids_containing(svc.conn, "缓粘结预应力")]


def test_snippet_labels_resolve_to_chunks(services: Services, docx_file: Path) -> None:
    _seed(services, docx_file)
    gold, warnings = ev.resolve_gold(
        services.conn,
        {"question": "x", "gold_snippets": ["缓粘结预应力", "根本不存在的句子"]},
        Filter(),
    )
    assert gold, "标注片段应能解析出 chunk"
    assert warnings and "未在语料中找到" in warnings[0]


def test_gold_resolution_respects_the_question_filter(
    services: Services, docx_file: Path
) -> None:
    """同一段原文出现在两个项目里时，题目限定甲项目就不能把乙项目的片段算进 gold。

    否则 Recall 的分母里永远有本题根本召回不到的片段，指标被系统性压低，
    而这个误差和模型好坏毫无关系。
    """
    for project, year in (("甲项目", 2024), ("乙项目", 2020)):
        outcome = ingest_file(
            services.conn,
            services.store,
            services.require_embedder(),
            services.settings,
            path=docx_file,
            project_name=project,
            doc_year=year,
        )
        assert outcome.status == "indexed"

    everything = repo.chunk_ids_containing(services.conn, "缓粘结预应力")
    assert len(everything) > 1, "两个项目都该有这段文字"

    scoped, _ = ev.resolve_gold(
        services.conn, {"question": "q", "gold_snippets": ["缓粘结预应力"]},
        Filter(projects=["甲项目"]),
    )
    assert scoped and scoped < set(everything), "范围内的命中必须少于全库命中"
    details = repo.get_chunks(services.conn, scoped)
    assert {d["project_name"] for d in details.values()} == {"甲项目"}


def test_gold_resolution_does_not_silently_truncate(services: Services) -> None:
    """片段出现次数超过 50 时不能悄悄只取前 50 条——那会让指标变好看。"""
    from rag4prefa.domain import Chunk

    repo.upsert_project(services.conn, "多片段项目")
    repo.upsert_document(
        services.conn,
        doc_id="d1",
        project_name="多片段项目",
        filename="a.docx",
        content_sha="sha",
        doc_year=2024,
        status="indexed",
    )
    repo.insert_chunks(
        services.conn,
        [
            Chunk(
                id=f"c{i:03d}",
                doc_id="d1",
                ordinal=i,
                chapter_path=["第1章"],
                title="t",
                text="共同关键词 的内容",
                token_count=5,
                content_sha=f"s{i}",
            )
            for i in range(60)
        ],
    )

    assert len(repo.chunk_ids_containing(services.conn, "共同关键词")) == 60

    gold, warnings = ev.resolve_gold(
        services.conn, {"question": "q", "gold_snippets": ["共同关键词"]}, Filter()
    )
    assert len(gold) == 60
    assert warnings == []


def test_invalid_gold_ids_are_dropped_with_warning(services: Services, docx_file: Path) -> None:
    """写错的 chunk_id 不能当成有效标注，否则分母虚高、Recall 永远差一截。"""
    gold_ids = _seed(services, docx_file)
    gold, warnings = ev.resolve_gold(
        services.conn,
        {"question": "q", "gold_chunk_ids": [gold_ids[0], "根本不存在的id"]},
        Filter(),
    )
    assert gold == {gold_ids[0]}
    assert warnings and "已剔除" in warnings[0]


def test_report_shows_per_mode_sample_count(services: Services, docx_file: Path) -> None:
    """各召回方式的有效样本数必须写进报告，不能只写一个总数。"""
    gold_ids = _seed(services, docx_file)
    rows = [
        {"question": "缓粘结预应力的张拉控制要点", "gold_chunk_ids": gold_ids[:1]},
        {"question": "陶土砖饰面的做法", "gold_snippets": ["陶土砖饰面"]},
    ]
    result = ev.evaluate(services, rows, 3, list(ev.MODES))
    assert result["evaluated"]["dense"] == 2

    text = ev.report(result, 3)
    assert "有效样本" in text
    assert "| dense | 2 |" in text


def test_all_four_modes_run_and_report(services: Services, docx_file: Path) -> None:
    gold_ids = _seed(services, docx_file)
    rows = [
        {"question": "缓粘结预应力的张拉控制要点", "gold_chunk_ids": gold_ids[:1]},
        {"question": "陶土砖饰面的做法", "gold_snippets": ["陶土砖饰面"]},
        {"question": "没有标注的问题"},  # 应被跳过并计入 skipped
    ]

    result = ev.evaluate(services, rows, 3, list(ev.MODES))

    assert result["samples"] == 2
    assert len(result["skipped"]) == 1
    for mode in ev.MODES:
        values = result["modes"][mode]
        assert len(values["recall"]) == 2, f"{mode} 应该有两条样本结果"
        assert all(0.0 <= v <= 1.0 for v in values["recall"])
    # 关键词明确命中时，稀疏召回必须找回该片段
    assert result["modes"]["sparse"]["recall"][0] >= 0.5

    text = ev.report(result, 3)
    assert "| dense |" in text and "| rerank |" in text
    assert "跳过" in text


def test_dataset_loader_reports_bad_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "ok"}\n这不是 JSON\n', encoding="utf-8")
    with pytest.raises(ValueError, match="不是合法 JSON"):
        ev.load_dataset(bad)


def test_dataset_loader_skips_comment_lines(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        "// 说明行\n" + json.dumps({"question": "q", "gold_chunk_ids": ["a"]}) + "\n",
        encoding="utf-8",
    )
    rows = ev.load_dataset(path)
    assert len(rows) == 1


def test_rerank_and_hybrid_share_one_candidate_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """rerank 行的唯一变量必须是「有没有重排」，不能连候选池一起换掉。

    曾经的实现里 rerank 把 k 换成 `max(k*4, 20)` 再走同一段融合代码：融合池深 100、
    只把前 20 条交给重排；而 hybrid 是 50 深的池子取前 5。池深和截断位置两个变量一起动，
    于是在没有配重排服务（NoopReranker）的情况下两行数字仍然不同——真实数据上差到 +0.033，
    报告读起来就像「重排有收益」，其实一个候选都没被重排过。这里钉住两件事：池深一致，
    且 rerank 拿到的是整个池而不是已经截断的 top-k。
    """
    seen: list[tuple[int, int | None]] = []
    pool = [
        Hit(chunk_id=f"c{i:03d}", score=1.0 - i / 100, project_name="p", filename="f.docx")
        for i in range(60)
    ]

    async def fake_retrieve(
        conn, store, embedder, settings, *, question, filt, recall_k, candidate_k=None,
        rewrites=(), rewrite_weight=0.5,
    ):
        # 两个深度都要记：2026-10-28 之前这里只收 recall_k，于是"声明 50、实得 25"
        # 这个 bug 被假实现挡住了——假实现按 recall_k 直接切片，看起来一直是对的。
        # `rewrites` 也必须收：签名对不上会直接 TypeError，而不是悄悄忽略改写。
        seen.append((recall_k, candidate_k))
        return pool[: (candidate_k or recall_k)], [], {}

    monkeypatch.setattr(ev.ask, "retrieve", fake_retrieve)

    class _Svc:
        conn = object()
        store = object()
        settings = object()
        reranker = NoopReranker()

        @staticmethod
        def require_embedder() -> object:
            return object()

    candidates = asyncio.run(ev.hybrid_pool(_Svc(), "q", Filter(), 5))
    hybrid = asyncio.run(ev.hybrid_hits(_Svc(), "q", Filter(), 5))
    rerank = asyncio.run(ev.rerank_hits(_Svc(), "q", Filter(), 5))

    assert seen == [(50, 50), (50, 50), (50, 50)], (
        f"两种模式必须用同一个候选池，且两段深度都要显式传，实际 {seen}"
    )
    assert len(candidates) == ev._hybrid_pool_size(5), "融合结果不许被悄悄截回一半"
    assert len(hybrid) == 5
    assert [h.chunk_id for h in rerank] == [h.chunk_id for h in hybrid], (
        "NoopReranker 下 rerank 与 hybrid 的 top-k 必须一致"
    )


def test_report_says_when_rerank_is_disabled(services: Services, docx_file: Path) -> None:
    """没配 R4P_RERANK_URL 时必须写「未启用重排」，否则那一行会被当成重排的收益。"""
    gold_ids = _seed(services, docx_file)
    rows = [{"question": "缓粘结预应力的张拉控制要点", "gold_chunk_ids": gold_ids[:1]}]

    result = ev.evaluate(services, rows, 3, list(ev.MODES))
    assert result["rerank_enabled"] is False
    assert "未启用重排" in ev.report(result, 3)

    result["rerank_enabled"] = True
    assert "未启用重排" not in ev.report(result, 3)


def test_report_compares_hybrid_against_both_dense_and_sparse(
    services: Services, docx_file: Path
) -> None:
    """只报「相对纯向量」的增益，会把「混合召回其实不如纯 BM25」这种情况藏起来。"""
    gold_ids = _seed(services, docx_file)
    rows = [{"question": "缓粘结预应力的张拉控制要点", "gold_chunk_ids": gold_ids[:1]}]
    text = ev.report(ev.evaluate(services, rows, 3, list(ev.MODES)), 3)
    assert "相对纯向量" in text
    assert "相对纯稀疏" in text
