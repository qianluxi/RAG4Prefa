"""语料快照的判据。

这些用例守的不是某个功能，而是"这份报告能不能跟上一份比"这件事本身：

- 指纹算法必须与 `eval/corpus.json` 的 `ids_sha` 一致，否则两套记录无法对照；
- 指纹必须与 id 顺序无关，否则同一个库在不同时刻导出会得到不同指纹；
- 变化检测只报**会影响指标**的三件事（生成时间天天在变，把它算进差异只会训练人忽略警告）；
- gold 是否过期要能**离线**判定——源文件删掉之后，这是唯一还守得住的判据。
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from rag4prefa import repo, snapshot
from rag4prefa.db import connect, init_db
from rag4prefa.domain import Chunk

ROOT = Path(__file__).resolve().parents[1]
REAL_SNAPSHOT = ROOT / "eval" / "corpus-snapshot.json"
MANIFEST = ROOT / "eval" / "corpus.json"

_spec = importlib.util.spec_from_file_location(
    "eval_script_snapshot", ROOT / "scripts" / "eval.py"
)
assert _spec and _spec.loader
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)


@pytest.fixture()
def conn(tmp_path: Path):
    db = tmp_path / "snap.db"
    init_db(db)
    connection = connect(db)
    yield connection
    connection.close()


def _seed(
    conn: sqlite3.Connection,
    *,
    project: str = "甲项目",
    filename: str = "a.docx",
    doc_id: str = "d1",
    ids: tuple[str, ...] = ("c1", "c2", "c3"),
) -> None:
    repo.upsert_project(conn, project)
    repo.upsert_document(
        conn,
        doc_id=doc_id,
        project_name=project,
        filename=filename,
        content_sha="sha",
        doc_year=2024,
        status="indexed",
    )
    repo.insert_chunks(
        conn,
        [
            Chunk(
                id=chunk_id,
                doc_id=doc_id,
                ordinal=i,
                chapter_path=["第1章"],
                title="t",
                text=f"正文 {i}",
                token_count=5,
                content_sha=f"s{i}",
            )
            for i, chunk_id in enumerate(ids)
        ],
    )


def test_ids_digest_ignores_order() -> None:
    """同一批 id 换个遍历顺序必须得到同一个指纹。

    库里的行序会随 VACUUM、INSERT OR REPLACE、甚至索引选择而变；
    指纹要是跟着行序走，"语料没变"会被报告成"语料变了"，警告就失去意义。

    这里给两个**顺序不同**的 list，而不是 list 与 set：set 的迭代顺序在同一进程里
    通常稳定，用 set 做对照时"不排序"的实现也能碰巧通过，断言就变成恒真的了。
    """
    assert snapshot.ids_digest(["c", "a", "b"]) == snapshot.ids_digest(["a", "b", "c"])
    assert snapshot.ids_digest(["c", "a", "b"]) == snapshot.ids_digest({"b", "c", "a"})


def test_snapshot_counts_and_digest_are_self_consistent(conn: sqlite3.Connection) -> None:
    """快照自己得先自洽：全量 id 的指纹要等于顶层 `ids_sha`。"""
    _seed(conn)
    _seed(conn, filename="b.docx", doc_id="d2", ids=("c9", "c8"))

    snap = snapshot.build_snapshot(conn)

    assert snap["document_count"] == 2
    assert snap["chunk_count"] == 5
    assert snap["chunk_ids"] == sorted(snap["chunk_ids"]), "全量 id 应当有序，便于直接 diff"
    assert snap["ids_sha"] == snapshot.ids_digest(snap["chunk_ids"])


def test_snapshot_keeps_same_named_documents_apart(conn: sqlite3.Connection) -> None:
    """两个项目下的同名文件不能互相覆盖。

    少记一份，指纹就和实际库对不上——而"对不对得上"正是这份文件唯一要回答的问题。
    """
    _seed(conn, project="甲项目", doc_id="d1", ids=("a1",))
    _seed(conn, project="乙项目", doc_id="d2", ids=("b1", "b2"))

    snap = snapshot.build_snapshot(conn)

    assert snap["document_count"] == 2
    assert len(snap["documents"]) == 2
    assert snap["chunk_count"] == 3


def test_compare_reports_material_change_only(conn: sqlite3.Connection) -> None:
    """只报影响可比性的三件事：片段集合、文档数、embedding 模型。

    时间戳变了不算语料变了。把这类噪声也报出来，读者很快就会对所有警告免疫。
    """
    _seed(conn)
    saved = snapshot.build_snapshot(conn)
    current = dict(saved, generated_at="2099-01-01T00:00:00+08:00")

    assert snapshot.compare(saved, current) == [], "只改了时间戳，不该报差异"

    changed = dict(current, ids_sha="0000000000000000")
    notes = snapshot.compare(saved, changed)
    assert any("片段集合" in n for n in notes), notes


def test_compare_says_nothing_without_saved_snapshot(conn: sqlite3.Connection) -> None:
    """没有快照文件时不该报警。

    第一次跑评估、或者快照被误删，都该照常出数字；把"没有记录"当成"记录不符"，
    会让第一次使用的人以为哪里坏了。
    """
    _seed(conn)
    assert snapshot.compare(None, snapshot.build_snapshot(conn)) == []


def test_dataset_digest_counts_rows_and_hashes_bytes(tmp_path: Path) -> None:
    """题数按非注释行算；哈希按原始字节算。

    哈希不能只算题目：注释里写着"这 10 题刻意不用原文术语"这类口径说明，
    改掉它等于换了一份评测集的含义，而题面一个字都没动。
    """
    path = tmp_path / "d.jsonl"
    path.write_text(
        '// 注释\n{"question": "q1", "gold_chunk_ids": ["a"]}\n\n'
        '{"question": "q2", "gold_chunk_ids": ["b"]}\n',
        encoding="utf-8",
    )
    first, rows = snapshot.dataset_digest(path)
    assert rows == 2

    path.write_text(
        '// 改了注释\n{"question": "q1", "gold_chunk_ids": ["a"]}\n\n'
        '{"question": "q2", "gold_chunk_ids": ["b"]}\n',
        encoding="utf-8",
    )
    second, rows = snapshot.dataset_digest(path)
    assert rows == 2
    assert first != second, "注释改了也算换了数据集版本"


def test_gold_missing_finds_stale_ids(tmp_path: Path) -> None:
    """标注里的 id 若已不在语料中，必须被点出来。

    这种失效最难发现：评估照常跑完、数字也像是真的，只是 Recall 无端低一截。
    """
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"question": "还在的", "gold_chunk_ids": ["c1"]}\n'
        '{"question": "过期的", "gold_chunk_ids": ["gone01", "gone02"]}\n',
        encoding="utf-8",
    )
    snap = {"chunk_ids": ["c1", "c2"]}

    missing = snapshot.gold_missing(snap, path)

    assert len(missing) == 2, missing
    assert all("过期的" in item for item in missing)


def test_load_snapshot_tolerates_broken_file(tmp_path: Path) -> None:
    """快照是佐证，不是运行前提：文件坏了当没有，别把整条评测拦住。"""
    path = tmp_path / "broken.json"
    path.write_text("{ not json", encoding="utf-8")
    assert snapshot.load_snapshot(path) is None
    assert snapshot.load_snapshot(tmp_path / "missing.json") is None


@pytest.mark.skipif(not REAL_SNAPSHOT.is_file(), reason="还没导出语料快照")
@pytest.mark.skipif(not MANIFEST.is_file(), reason="没有标注清单")
def test_snapshot_agrees_with_manifest_on_shared_documents() -> None:
    """快照与 `eval/corpus.json` 说的是不是同一件事。

    两套记录必须能对上：清单是**标注时**的语料指纹，快照是**运行时**的。
    同一份文档在两处给出的 chunk 数与 ids_sha 不一致，只有两种解释——解析器变了，
    或者这两份记录根本不是同一批语料。无论哪种，都不能拿它们互相印证。
    """
    snap = snapshot.load_snapshot(REAL_SNAPSHOT)
    assert snap, "快照读不出来"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    shared = 0
    for filename, meta in manifest["documents"].items():
        entry = snap["documents"].get(filename)
        if entry is None:
            continue  # 该文档已从库里撤走，无从比对
        shared += 1
        assert entry["chunks"] == meta["chunks"], (
            f"{filename} 的片段数：清单 {meta['chunks']} vs 快照 {entry['chunks']}"
        )
        assert entry["ids_sha"] == meta["ids_sha"], (
            f"{filename} 的 id 指纹：清单 {meta['ids_sha']} vs 快照 {entry['ids_sha']}"
        )
    assert shared, "清单与快照没有任何共同文档，这份用例事实上没有校验任何东西"


def _fake_result(corpus_drift: list[str] | None = None) -> dict:
    return {
        "modes": {"dense": {"recall": [1.0], "mrr": [1.0], "ndcg": [1.0]}},
        "evaluated": {"dense": 1},
        "samples": 1,
        "skipped": [],
        "warnings": [],
        "dataset": "eval/dataset.jsonl",
        "dataset_sha": "158996f8a9a5db75",
        "dataset_rows": 25,
        "generated_at": "2026-10-28T10:43:35+08:00",
        "code": {"commit": "399d0cb", "dirty": True},
        "context": {
            "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
            "embedding_dim": 1024,
            "chunk_min_tokens": 80,
            "chunk_max_tokens": 480,
            "recall_k": 50,
            "rrf_k": 60,
            "rerank_model": "",
            "corpus": {
                "documents": 9,
                "chunks": 152,
                "ids_sha": "4a096f6bd0a23d13",
                "drift": corpus_drift or [],
            },
        },
    }


def test_report_prints_dataset_and_code_identity() -> None:
    """报告要把"哪份数据集（哈希）、多少题、哪个代码版本"真的印出来。

    这几个字段最容易出现"赋值了但没渲染"：测试读的是 result 字典而不是 report 输出，
    于是一个漏拼的 f-string 可以一路绿到发布。
    """
    text = _eval.report(_fake_result(), 5)

    assert "25 题" in text, "题数没印出来"
    assert "158996f8a9a5db75" in text, "数据集哈希没印出来"
    assert "399d0cb" in text and "有未提交改动" in text, "代码版本没印出来"
    assert "152 个片段" in text and "4a096f6bd0a23d13" in text, "语料快照没印出来"


def test_report_warns_when_snapshot_is_stale() -> None:
    """快照与当前库不一致时，报告必须自己说出"不可比较"。

    没有这句，读者看到两份报告数字不同，只能猜是模型换过、还是语料换过——
    而这恰恰是记录语料快照要回答的唯一问题。
    """
    text = _eval.report(
        _fake_result(["语料片段集合变了（记录 `old` → 当前 `new`）"]), 5
    )

    assert "语料已变化" in text
    assert "不可直接比较" in text
    assert "snapshot_corpus.py" in text, "要指出怎么更新记录"


def test_report_stays_quiet_when_corpus_unchanged() -> None:
    """语料没变就不要写警告。

    常年挂着一句"注意可能不可比"，会让读者对所有警告免疫，
    真正出问题那次也就跟着被忽略。
    """
    assert "语料已变化" not in _eval.report(_fake_result(), 5)
