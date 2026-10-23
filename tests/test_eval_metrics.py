"""评估指标的定义必须可被测试钉住。

背景：旧实现 `recall = sum(rels[:k]) / len(rels)` 的分母是返回条数，
实际算的是 Precision——两个标注只找回一个时会给出 0.5，
而"全找回"也会因返回了 5 条而算成 0.4。课题验收要用这些数字，必须先钉死定义。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rag4prefa.domain import Hit

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_script", ROOT / "scripts" / "eval.py")
assert _spec and _spec.loader
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _hits(*chunk_ids: str, text: str = "") -> list[Hit]:
    return [Hit(chunk_id=cid, text=text or cid) for cid in chunk_ids]


def test_recall_one_gold_found_in_five() -> None:
    """只标了 1 条且排在首位 → Recall 必须是 1.0，而不是 1/5。"""
    hits = _hits("c1", "c2", "c3", "c4", "c5")
    assert ev.recall_at_k(hits, {"c1"}, [], 5) == 1.0


def test_recall_two_gold_both_found() -> None:
    hits = _hits("c1", "c2", "c3", "c4", "c5")
    assert ev.recall_at_k(hits, {"c1", "c2"}, [], 5) == 1.0


def test_recall_half_of_gold_found() -> None:
    hits = _hits("c1", "c9", "c8")
    assert ev.recall_at_k(hits, {"c1", "c2"}, [], 3) == 0.5


def test_recall_respects_k() -> None:
    """gold 在 top-1 之外就不该算进 Recall@1。"""
    hits = _hits("c9", "c1")
    assert ev.recall_at_k(hits, {"c1"}, [], 1) == 0.0
    assert ev.recall_at_k(hits, {"c1"}, [], 2) == 1.0


def test_recall_by_snippets_dedupes() -> None:
    """同一片段在两条命中里都出现，也只能算找回 1 个。"""
    hits = [
        Hit(chunk_id="a", text="缓粘结预应力需控制张拉时机"),
        Hit(chunk_id="b", text="缓粘结预应力需控制张拉时机（重复段落）"),
    ]
    assert ev.recall_at_k(hits, set(), ["缓粘结预应力"], 2) == 1.0


def test_recall_by_snippets_partial() -> None:
    hits = [Hit(chunk_id="a", text="缓粘结预应力需控制张拉时机")]
    assert ev.recall_at_k(hits, set(), ["缓粘结预应力", "陶土砖反打"], 2) == 0.5


def test_recall_without_gold_is_zero() -> None:
    assert ev.recall_at_k(_hits("c1"), set(), [], 5) == 0.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert ev.mrr([1.0, 0.0]) == 1.0
    assert ev.mrr([0.0, 0.0, 1.0]) == 1.0 / 3
    assert ev.mrr([0.0, 0.0, 0.0]) == 0.0


def test_ndcg_perfect_and_empty() -> None:
    assert ev.ndcg_at_k([1.0, 1.0], 2, 2) == 1.0
    assert ev.ndcg_at_k([0.0, 0.0], 2, 2) == 0.0


def test_ndcg_penalises_lower_rank() -> None:
    top1 = ev.ndcg_at_k([1.0, 0.0], 2, 1)
    top2 = ev.ndcg_at_k([0.0, 1.0], 2, 1)
    assert top1 == 1.0
    assert 0.0 < top2 < top1


def test_report_mentions_recall_definition() -> None:
    results = {
        "dense": {"recall": [1.0], "mrr": [1.0], "ndcg": [1.0]},
        "hybrid": {"recall": [0.5], "mrr": [0.5], "ndcg": [0.5]},
    }
    text = ev.report(results, 5, 1)
    assert "|Gold|" in text
    assert "混合召回相对纯向量" in text
