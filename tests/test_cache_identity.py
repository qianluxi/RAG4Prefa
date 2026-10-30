"""评测缓存身份的判据（2026-10-30 专家复审第三节）。

这一份测试要钉住的不是"函数能跑"，而是**缓存不能被静默复用**。身份字段少一个，
实验就退化成"看起来只差一个变量、其实差两个"，而报告上看不出任何异常。所以每条
断言都对着一个具体的失效方式写：

- 语料 id 变了 → 候选不能复用（重新入库之后，同一句话的片段已经不是同一批）；
- 源码变了 → 候选不能复用（改了融合或截断，取法变了）；
- 服务商/思考设置变了 → 判官结论不能复用（同名模型在两个服务商上行为相反，已实测）；
- 提示词改一个字 → 判官结论不能复用（同一批片段、另一把尺子）；
- 候选换了 → 判官结论不能复用（即使提示词与模型都没动）。

反方向的断言同样重要：**身份相同就要能复用**。只写"变了会拒绝"，把函数改成恒抛错
也能全绿——那等于缓存功能被删掉了。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rag4prefa import prompts, repo
from rag4prefa.domain import Chunk, Filter, Hit

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ci = _load("cache_identity_under_test", ROOT / "scripts" / "cache_identity.py")
_js = _load("judge_sufficiency_under_test", ROOT / "scripts" / "judge_sufficiency.py")


def _hit(chunk_id: str = "c1", *, text: str = "正文") -> Hit:
    return Hit(chunk_id=chunk_id, filename="a.pdf", text=text)


def _seed(conn, texts: list[str]) -> None:
    repo.upsert_project(conn, "甲项目")
    repo.upsert_document(
        conn, doc_id="d1", project_name="甲项目", filename="a.docx",
        content_sha="sha", doc_year=2024, status="indexed",
    )
    repo.insert_chunks(
        conn,
        [
            Chunk(
                id=f"c{i:03d}", doc_id="d1", ordinal=i, chapter_path=["第1章"],
                title="t", text=text, token_count=8, content_sha=f"s{i}",
            )
            for i, text in enumerate(texts)
        ],
    )


# ── 内容指纹 ────────────────────────────────────────────────


def test_fingerprint_ignores_line_endings_but_not_content(tmp_path: Path) -> None:
    """CRLF 与 LF 必须同指纹：否则同一份代码在 Windows 与 Linux 上算出两个身份，
    "同指纹即同代码"这句话当场失效，而且没人会想到去查。"""
    crlf_dir = tmp_path / "win"
    lf_dir = tmp_path / "linux"
    crlf_dir.mkdir()
    lf_dir.mkdir()
    # 两份**同名同内容**的文件，只有换行不同：路径进不进哈希另有一条用例管，
    # 这里要隔离的只有换行这一项。
    (crlf_dir / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")
    (lf_dir / "a.py").write_bytes(b"x = 1\ny = 2\n")
    crlf, n1 = _ci.fingerprint_files(crlf_dir, ["a.py"])
    lf, n2 = _ci.fingerprint_files(lf_dir, ["a.py"])
    assert (crlf, n1) == (lf, n2) == (crlf, 1)

    (lf_dir / "b.py").write_bytes(b"x = 1\ny = 3\n")
    other, _ = _ci.fingerprint_files(lf_dir, ["b.py"])
    assert other != lf


def test_fingerprint_counts_missing_files_out(tmp_path: Path) -> None:
    """路径不存在就跳过：指纹算不出来时宁可少算一个文件，也不该抛错让评估跑不下去。"""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    sha, count = _ci.fingerprint_files(tmp_path, ["a.py", "nope.py"])
    assert count == 1
    assert sha


def test_source_fingerprint_changes_when_a_module_changes(tmp_path: Path) -> None:
    """真在仓库上取值，并且**对内容敏感**：只写"返回了字符串"的话，改成常量也能过。"""
    before = _ci.source_fingerprint()
    assert before["src_files"] > 10 and len(before["src_sha"]) == 16

    # 临时改一份核心文件的字节，指纹必须变；写完立刻还原。
    target = ROOT / "rag4prefa" / "sufficiency.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# fingerprint probe\n")
        after = _ci.source_fingerprint()
    finally:
        target.write_bytes(original)
    assert after["src_sha"] != before["src_sha"]
    assert _ci.source_fingerprint()["src_sha"] == before["src_sha"]


# ── 语料身份 ────────────────────────────────────────────────


def test_corpus_identity_tracks_ids_not_question_text(conn) -> None:
    empty = _ci.corpus_identity(conn)
    assert empty["docs"] == 0 and empty["chunks"] == 0

    _seed(conn, ["甲", "乙"])
    two = _ci.corpus_identity(conn)
    assert two["docs"] == 1 and two["chunks"] == 2
    assert two["ids_sha"] != empty["ids_sha"]
    # 同样的语料再算一次必须相同，否则"没有变化"也会让缓存失效，缓存等于没有。
    assert _ci.corpus_identity(conn) == two

    _seed_more = Chunk(
        id="c999", doc_id="d1", ordinal=9, chapter_path=["第1章"], title="t",
        text="丙", token_count=8, content_sha="s9",
    )
    repo.insert_chunks(conn, [_seed_more])
    three = _ci.corpus_identity(conn)
    assert three["chunks"] == 3 and three["ids_sha"] != two["ids_sha"]


# ── 校验 ────────────────────────────────────────────────────


def test_check_reports_the_field_that_differs(tmp_path: Path) -> None:
    """报错要能指认字段：只说"身份不符"，人会去猜，猜错就重跑一遍全量。"""
    meta = {"mode": "rerank", "k": 8, "ids_sha": "abc"}
    _ci.check(tmp_path / "x.json", meta, meta, kind="候选", bypass="--rebuild")  # 不抛

    with pytest.raises(_ci.CacheMismatch) as exc:
        _ci.check(tmp_path / "x.json", meta, {**meta, "k": 5}, kind="候选", bypass="--rebuild")
    message = str(exc.value)
    assert "k" in message and "8" in message and "5" in message
    assert "--rebuild" in message


def test_check_refuses_meta_less_cache(tmp_path: Path) -> None:
    """旧格式（没有 meta）一律拒绝：它的 key 里没有任何身份，读到就是读到别人的结论。"""
    with pytest.raises(_ci.CacheMismatch):
        _ci.check(tmp_path / "x.json", None, {"k": 5}, kind="候选", bypass="--rebuild")
    with pytest.raises(_ci.CacheMismatch):
        _ci.check(tmp_path / "x.json", {}, {"k": 5}, kind="候选", bypass="--rebuild")


def test_load_and_save_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    meta = {"kind": "hits", "mode": "rerank", "k": 8}
    _ci.save(path, {"k1": {"hits": []}}, meta)
    assert _ci.load(path, expect=meta, kind="候选", bypass="--rebuild") == {"k1": {"hits": []}}

    # 缺文件不是错误：第一次跑就该从空开始。
    assert _ci.load(tmp_path / "nope.json", expect=meta, kind="候选", bypass="--rebuild") == {}

    # 换了参数再读同一个文件，必须抛，而不是拿到 k=8 的条目。
    with pytest.raises(_ci.CacheMismatch):
        _ci.load(path, expect={**meta, "k": 5}, kind="候选", bypass="--rebuild")


def test_saved_payload_carries_schema_and_meta(tmp_path: Path) -> None:
    """落盘结构本身就是合约：`schema` 让以后的改动有版本可依，`meta` 是校验的唯一来源。"""
    path = tmp_path / "cache.json"
    _ci.save(path, {}, {"kind": "verdicts", "judge": "v3"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == _ci.SCHEMA
    assert payload["meta"]["judge"] == "v3"
    assert payload["hits"] == {}


# ── 与判官脚本的接线 ────────────────────────────────────────


def test_hits_identity_carries_corpus_and_source() -> None:
    """候选缓存的身份必须同时带"库是什么"和"取法是什么"。少任何一个，另一维的变化
    都会静默沿用旧候选。"""
    ident = _js.hits_identity(
        mode="rerank", k=8,
        corpus={"docs": 75, "chunks": 2560, "ids_sha": "71381cdb"},
        source={"src_sha": "5305e076", "src_files": 42},
    )
    assert ident["ids_sha"] == "71381cdb"
    assert ident["src_sha"] == "5305e076"
    assert ident["mode"] == "rerank" and ident["k"] == 8


def test_verdict_key_changes_with_every_identity_field() -> None:
    """六个字段逐个单独改一次，key 都必须变。

    这不是"六条一样的断言"——它们对应六个各自独立发生过（或险些发生）的混用：
    同名模型换服务商、思考默认值不同、提示词改判据、k 从 5 改 8、候选换了、换版本。
    """
    base = dict(
        judge="v3", model="Qwen/Qwen3.5-35B-A3B", provider="api.siliconflow.com",
        thinking="关", k=8, prompt_sha="abc", temperature=0.0,
    )
    hits = [_hit("c1"), _hit("c2")]
    key = _js.verdict_key(**base, hits=hits, question="台面多高")

    variants = {
        "judge": "v2",
        "model": "Qwen/Qwen3-8B",
        "provider": "api-inference.modelscope.cn",
        "thinking": "不发送",
        "k": 5,
        "prompt_sha": "abd",
        # 温度算身份：同一套提示词、同一个模型，0.2 与 0 的判定不同（实测 11 题翻转）。
        "temperature": 0.2,
    }
    for field, value in variants.items():
        changed = _js.verdict_key(**{**base, field: value}, hits=hits, question="台面多高")
        assert changed != key, f"{field} 没有进 key"

    # 候选变了（哪怕只是名次对调）也要换 key：重排分与"最相关那条第几"都变了。
    swapped = _js.verdict_key(**base, hits=list(reversed(hits)), question="台面多高")
    assert swapped != key
    assert _js.verdict_key(**base, hits=hits, question="台面多高2") != key


def test_verdict_key_is_stable_for_identical_inputs() -> None:
    """同样的身份必须给出同样的 key，否则缓存永远不命中——那和删掉缓存没区别。"""
    args = dict(
        judge="v1", model="m", provider="p", thinking="关", k=8, prompt_sha="abc",
        temperature=0.0,
    )
    hits = [_hit("c1"), _hit("c2")]
    assert _js.verdict_key(**args, hits=hits, question="q") == _js.verdict_key(
        **args, hits=[_hit("c1"), _hit("c2")], question="q"
    )


def test_filter_key_separates_the_same_question_under_different_scopes() -> None:
    """同一个问句配不同 Filter（项目/年份/关键词）会检索出不同候选，key 必须分开。"""
    base = Filter()
    assert _ci.filter_key(base) == _ci.filter_key(Filter())
    assert _ci.filter_key(base) == _ci.filter_key(Filter(projects=None))
    assert _ci.filter_key(base) != _ci.filter_key(Filter(year_from=2021))
    assert _ci.filter_key(Filter(year_from=2021)) != _ci.filter_key(
        Filter(year_from=2021, year_to=2023)
    )


def test_prompt_signature_separates_judge_versions() -> None:
    """v1 与 v2 是两套判据，指纹必须不同，否则换版本会读到上一版的结论。

    `k` 不在指纹里是有意的（`top_n` 只截片段数、不改模板文本）：它由 verdict_meta
    与 verdict_key 带着。这条用例同时钉住"指纹稳定"——同样的版本永远同一个值，
    否则缓存永不命中。
    """
    v1 = _js._prompt_signature("v1")
    assert v1 == _js._prompt_signature("v1")
    # 对着版本表现算，而不是写死"有几个版本"：新增 v3 时这条自动覆盖到它。
    sigs = {name: _js._prompt_signature(name) for name in prompts.JUDGE_PROMPTS}
    assert len(set(sigs.values())) == len(sigs), f"有版本共用同一指纹：{sigs}"
    assert sigs["v1"] == v1
