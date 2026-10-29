"""评估指标的定义必须可被测试钉住。

两条曾经踩过的坑：

1. 旧实现 `recall = sum(rels[:k]) / len(rels)` 的分母是返回条数，实际算的是 Precision；
2. 修好分母之后，Recall 按"原文片段"计数、MRR/nDCG 按 chunk 排序，同一个系统会同时
   算出 Recall=1.0 和 nDCG=0.61——报告里没法解释。现在统一成 chunk 单位：
   片段标注在加载时解析成包含它的 chunk 集合，三个指标共用同一个 gold。
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
    assert ev.recall_at_k(hits, {"c1"}, 5) == 1.0


def test_recall_two_gold_both_found() -> None:
    hits = _hits("c1", "c2", "c3", "c4", "c5")
    assert ev.recall_at_k(hits, {"c1", "c2"}, 5) == 1.0


def test_recall_half_of_gold_found() -> None:
    hits = _hits("c1", "c9", "c8")
    assert ev.recall_at_k(hits, {"c1", "c2"}, 3) == 0.5


def test_recall_respects_k() -> None:
    """gold 在 top-1 之外就不该算进 Recall@1。"""
    hits = _hits("c9", "c1")
    assert ev.recall_at_k(hits, {"c1"}, 1) == 0.0
    assert ev.recall_at_k(hits, {"c1"}, 2) == 1.0


def test_recall_dedupes_repeated_hits() -> None:
    hits = _hits("c1", "c1")
    assert ev.recall_at_k(hits, {"c1", "c2"}, 2) == 0.5


def test_recall_without_gold_is_zero() -> None:
    assert ev.recall_at_k(_hits("c1"), set(), 5) == 0.0


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


def test_two_snippets_in_one_chunk_score_consistently() -> None:
    """两个标注片段落在同一个 chunk 里时，三个指标必须都给出 1.0。

    这正是旧实现自相矛盾的场景：Recall 按片段计数得 1.0，nDCG 的 IDCG 按"两个相关 chunk"
    构造却只命中一个，得 0.61。统一成 chunk 单位后两者一致。
    """
    hits = [
        Hit(chunk_id="a", text="缓粘结预应力需控制张拉时机"),
        Hit(chunk_id="b", text="陶土砖反铺工艺"),
    ]
    gold = {"a"}  # 解析后：两个片段都在 chunk a 里 → gold 只有一个 chunk
    rels = [1.0 if h.chunk_id in gold else 0.0 for h in hits]
    assert ev.recall_at_k(hits, gold, 2) == 1.0
    assert ev.mrr(rels) == 1.0
    assert ev.ndcg_at_k(rels, 2, len(gold)) == 1.0


def test_report_mentions_definition_and_lists_problems() -> None:
    result = {
        "modes": {
            "dense": {"recall": [1.0], "mrr": [1.0], "ndcg": [1.0]},
            "hybrid": {"recall": [0.5], "mrr": [0.5], "ndcg": [0.5]},
        },
        "samples": 1,
        "skipped": [],
        "warnings": ["某问题：标注片段未在语料中找到：'xxx'"],
    }
    text = ev.report(result, 5)
    assert "|Gold|" in text
    assert "混合召回相对纯向量" in text
    assert "需要处理的数据问题" in text


def test_unknown_mode_is_rejected(capsys, monkeypatch) -> None:
    monkeypatch.setattr(ev.sys, "argv", ["eval.py", "--modes", "dense,魔法"])
    assert ev.main() == 2
    assert "未知的召回方式" in capsys.readouterr().err


def test_missing_dataset_returns_code(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ev.sys, "argv", ["eval.py", "--dataset", str(tmp_path / "无.jsonl")])
    assert ev.main() == 2


def test_report_name_distinguishes_datasets_on_same_corpus() -> None:
    """同一份语料、同一天、同一套配置，只换了数据集，报告不能同名。

    2026-10-29 实测：口语集从 15 题扩到 50 题后重跑，文件名一字不差，
    15 题那份被静默覆盖——而 HANDOFF 里引用的正是那个名字。语料指纹挡不住这种覆盖，
    因为它只承诺"同语料同名"，没承诺"同数据集同名"。
    """
    base = {
        "generated_at": "2026-10-29T11:17:00+08:00",
        "context": {"corpus": {"ids_sha": "208ec1576dfc11cc"}},
        "modes": {"dense": {}},
        "dataset_sha": "3da85a91a18c4a11",
    }
    other = {**base, "dataset_sha": "00000000deadbeef"}

    a = ev._report_path(base, Path("eval/dataset.paraphrase.jsonl"))
    b = ev._report_path(other, Path("eval/dataset.paraphrase.jsonl"))

    assert a != b
    assert "208ec157" in a.name and "3da85a91" in a.name
    # 同语料不同数据集必须不同名；反过来同一份数据集必须稳定同名（可复现）。
    assert a == ev._report_path(dict(base), Path("eval/dataset.paraphrase.jsonl"))
