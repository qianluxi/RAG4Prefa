"""评估流程的集成测试：真实 Embedder + 本地假服务 + 真实 SQLite，不联网。

指标公式本身由 test_eval_metrics.py 钉住，这里验证的是**链路**：
标注怎么换算成 gold、四种召回方式怎么跑、报告里能不能看出数据问题。
"""

from __future__ import annotations

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
        services.conn, {"question": "x", "gold_snippets": ["缓粘结预应力", "根本不存在的句子"]}
    )
    assert gold, "标注片段应能解析出 chunk"
    assert warnings and "未在语料中找到" in warnings[0]


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
