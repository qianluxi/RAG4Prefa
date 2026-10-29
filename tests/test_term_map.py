"""受控词表：公众表达 → 规范用语。

这一层的主张很小，但必须被钉死，否则整轮实验会得出错误结论：

1. **匹配是子串判断**，命中与否可以肉眼复核，不引入分词或模型；
2. **近邻话题不许误命中**——「老房子翻新能补多少钱」问的是补贴，不是适用范围，
   注入「参照执行」只会把 gold 挤出候选池；
3. **没覆盖的题返回空列表**，不返回原话。返回原话会让"词表没覆盖"伪装成
   "词表给对了词也没用"，把结论往悲观方向带；
4. **注入只能是独立通道**：一旦拼进原话，原话那一路的排序就变了，
   `hybrid`/`rerank` 两行不再是对照组。

第 2 条是最容易被"看起来合理"的表写坏的地方——词表一旦松，所有题都命中，
报告上表现为"召回全面变差"，而真正的原因是一条本该只覆盖 A 话题的规则
把 B 话题也吃了进去。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag4prefa import repo
from rag4prefa.adapters.term_map import TermEntry, TermMap, load_term_map, merge_terms
from rag4prefa.adapters.vectors import NumpyStore, VectorItem
from rag4prefa.config import Settings
from rag4prefa.domain import Chunk
from tests.conftest import FakeEmbedder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "rag4prefa" / "terms" / "public_to_standard.json"

_spec = importlib.util.spec_from_file_location("eval_script_termmap", ROOT / "scripts" / "eval.py")
assert _spec and _spec.loader
_ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ev)


def _write(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"version": 1, "entries": entries}, ensure_ascii=False), "utf-8")
    return path


def _entry(**over: object) -> dict:
    base = {
        "id": "e1",
        "groups": [["挂电视"], ["掉"]],
        "terms": ["墙面悬挂较重物体"],
        "source": "某导则 4.1.9",
    }
    base.update(over)
    return base


def test_real_map_covers_the_five_gap_questions() -> None:
    """仓库里那份词表必须真的命中 5 道「词表鸿沟」题。

    这五道是诊断里"三种召回都捞不进池"的题。词表如果连它们都命不中，
    这轮实验就变成"测了一份没生效的配置"，而报告上只会写"用了词表"。
    """
    term_map = TermMap.load(DEFAULT_MAP)
    questions = {
        "墙上想挂电视和吊柜，会不会掉下来？": "专用连接件",
        "墙里的电线以后坏了能换吗？": "维修更换",
        "不是新盖的老房子，也能用这套做法吗？": "参照执行",
        "集成吊顶里头包不包含灯具？": "不含电器设备",
        "同一个部位装修完，怎么知道自己做得好不好？": "隐蔽工程验收",
    }
    for question, expected in questions.items():
        assert expected in term_map.match(question), f"「{question}」没命中 {expected}"


def test_real_map_does_not_fire_on_neighbouring_topics() -> None:
    """近邻话题必须不命中。这是词表最容易写坏、后果最隐蔽的地方。

    - 「老房子翻新能补多少钱」问的是补贴额度，不是适用范围 → 不能注入「参照执行」；
    - 「吊顶上面还要装灯装风机，会不会塌」问的是吊顶荷载，不是品类边界
      → 不能注入「不含电器设备」，那是补贴品类表的词，会把答案挤走；
    - 「装修完多久能住进去」问的是工期，不能命中验收。
    """
    term_map = TermMap.load(DEFAULT_MAP)
    for question in (
        "老房子翻新能补多少钱？",
        "吊顶上面还要装灯装风机，会不会塌？",
        "装修完多久能住进去？",
        "我家厨房想换新，大概要花多少钱？",
    ):
        assert term_map.match(question) == [], f"「{question}」被误命中：{term_map.match(question)}"


def test_match_is_ordered_and_deduplicated(tmp_path: Path) -> None:
    """两条规则给了同一个术语时只输出一次，顺序按词表顺序。

    重复输出会让同一个术语被注入两次——而通道列表是按串去重的，
    重复只在"term 通道"形态下才会显形，不测就发现不了。
    """
    path = _write(
        tmp_path,
        [
            _entry(id="a", terms=["专用连接件", "墙面悬挂较重物体"]),
            _entry(id="b", groups=[["挂电视"]], terms=["专用连接件"]),
        ],
    )
    assert TermMap.load(path).match("挂电视会不会掉") == ["专用连接件", "墙面悬挂较重物体"]


def test_unmatched_returns_empty_not_original() -> None:
    """没命中必须是空列表。

    返回原话（`[question]`）看着"安全"，实际会让 eval 把这道题算进"词表组"：
    两条通道里一条是原话、一条还是原话，结果与基线完全一样，于是"词表没覆盖"
    被记成"词表无效"。这一条就是这个错误的判据。
    """
    term_map = TermMap.load(DEFAULT_MAP)

    assert term_map.match("装修完多久能住进去？") == []
    assert term_map.rewrite("装修完多久能住进去？") == []


def test_rewrite_forms_differ(tmp_path: Path) -> None:
    """`terms` 每词一路，`joined` 拼成一串——两种形态必须真的不一样。

    拼串会把长术语的权重稀释掉（实测逐词总比拼串好），这是"多一个串"与
    "多一路通道"的差别，不能让一个参数看起来可调、实际两条路同结果。
    """
    path = _write(tmp_path, [_entry(terms=["专用连接件", "墙面悬挂较重物体"])])
    term_map = TermMap.load(path)

    assert term_map.rewrite("挂电视会不会掉", form="terms") == [
        "专用连接件",
        "墙面悬挂较重物体",
    ]
    assert term_map.rewrite("挂电视会不会掉", form="joined") == ["专用连接件 墙面悬挂较重物体"]
    with pytest.raises(ValueError, match="未知的改写形态"):
        term_map.rewrite("挂电视会不会掉", form="whatever")


def test_empty_group_is_rejected(tmp_path: Path) -> None:
    """空分组会让 `all()` 恒真 → 该条对所有问题都命中，是词表最坏的坏法。"""
    path = _write(tmp_path, [_entry(groups=[["挂电视"], []])])

    with pytest.raises(ValueError, match="空的分组"):
        TermMap.load(path)


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    """没有出处的条目无法区分"条文原话"与"模型意译"，而这两者效果不同。"""
    path = _write(tmp_path, [_entry(source="")])

    with pytest.raises(ValueError, match="缺 source"):
        TermMap.load(path)


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    """id 重复意味着两条规则无法在诊断里区分，出问题时找不到是哪条写的。"""
    path = _write(tmp_path, [_entry(), _entry()])

    with pytest.raises(ValueError, match="id 重复"):
        TermMap.load(path)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    """空词表等于没启用，但不该静默跑完一整轮评估。"""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"version": 1, "entries": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="没有 entries"):
        TermMap.load(path)


def test_load_none_means_disabled() -> None:
    assert load_term_map(None) is None
    assert load_term_map("") is None


def test_merge_terms_keeps_order() -> None:
    assert merge_terms(["a", "b"], ["b", "c", " "]) == ["a", "b", "c"]


def test_explain_reports_which_trigger_fired() -> None:
    """诊断要能回答"这题为什么注入了这几个词"，否则词表就是个黑箱。"""
    entry = TermEntry(
        id="e", groups=(("挂电视", "吊柜"), ("掉", "塌")), terms=("专用连接件",), source="s"
    )
    term_map = TermMap([entry])

    assert term_map.explain("墙上想挂电视和吊柜，会不会掉下来？") == [
        {"id": "e", "intent": "unlabelled", "matched_by": "挂电视 + 掉", "terms": ["专用连接件"]}
    ]


# ── 接入评估链路 ─────────────────────────────────────────────

TEXTS = [f"墙面悬挂较重物体应采用专用连接件 第{i}条" for i in range(20)]
QUESTION = "墙上想挂电视和吊柜，会不会掉下来？"


def _seed(conn, dim: int = FakeEmbedder.dim) -> tuple[NumpyStore, FakeEmbedder]:
    repo.upsert_project(conn, "甲项目")
    repo.upsert_document(
        conn,
        doc_id="d1",
        project_name="甲项目",
        filename="a.docx",
        content_sha="sha",
        doc_year=2024,
        status="indexed",
    )
    embedder = FakeEmbedder()
    chunks = [
        Chunk(
            id=f"c{i:03d}",
            doc_id="d1",
            ordinal=i,
            chapter_path=["第1章"],
            title="t",
            text=text,
            token_count=8,
            content_sha=f"s{i}",
        )
        for i, text in enumerate(TEXTS)
    ]
    repo.insert_chunks(conn, chunks)
    store = NumpyStore(conn, dim)
    store.upsert(
        [
            VectorItem(
                chunk_id=c.id,
                vector=embedder.embed_one(c.text),
                project_name="甲项目",
                project_code="",
                doc_year=2024,
                model=embedder.model,
            )
            for c in chunks
        ]
    )
    repo.set_index_meta(
        conn, model=embedder.model, dim=str(dim), backend="numpy", backend_id="numpy"
    )
    return store, embedder


def _services(conn, settings: Settings) -> SimpleNamespace:
    store, embedder = _seed(conn)
    return SimpleNamespace(
        conn=conn,
        store=store,
        settings=settings,
        require_embedder=lambda: embedder,
    )


def test_evaluate_injects_terms_as_separate_channels(conn, settings: Settings, monkeypatch) -> None:
    """词表命中的术语必须走 `rewrites`，**召回串保持原话**。

    这条 spy 到真正传进检索的两个串。只断言诊断字段是不够的：把召回串换成
    术语拼串、诊断里照旧写 `rewrites`，字段一字不变而行为已经退回老路
    （第三十八轮就是这么读错的）。
    """
    services = _services(conn, settings)
    rows = [{"question": QUESTION, "gold_snippets": [TEXTS[0]]}]
    term_map = TermMap(
        [TermEntry("e", (("挂电视",), ("掉",)), ("墙面悬挂较重物体", "专用连接件"), "s")]
    )

    seen: list[dict] = []
    original = _ev.retrieve_for

    async def spy(services, mode, question, filt, k, keyword, *, rerank_query=None,
                  rewrites=(), rewrite_weight=0.5):
        seen.append({"mode": mode, "question": question, "rewrites": tuple(rewrites)})
        return await original(
            services, mode, question, filt, k, keyword,
            rerank_query=rerank_query, rewrites=rewrites, rewrite_weight=rewrite_weight,
        )

    monkeypatch.setattr(_ev, "retrieve_for", spy)

    result = asyncio.run(
        _ev._evaluate(
            services, rows, 5, ["hybrid", "hybrid_parallel"], term_map=term_map,
        )
    )

    assert [s["mode"] for s in seen] == ["hybrid", "hybrid_parallel"]
    for call in seen:
        assert call["question"] == QUESTION, (
            f"{call['mode']} 的召回串被换成了术语串，退回拼串形态了：{call['question']!r}"
        )
    assert [s["rewrites"] for s in seen] == [
        ("墙面悬挂较重物体", "专用连接件"),
        ("墙面悬挂较重物体", "专用连接件"),
    ]
    assert result["diagnostics"][0]["term_entries"] == [
        {
            "id": "e",
            "intent": "unlabelled",
            "matched_by": "挂电视 + 掉",
            "terms": ["墙面悬挂较重物体", "专用连接件"],
        }
    ]


def test_evaluate_records_uncovered_questions(conn, settings: Settings) -> None:
    """词表一条都没命中的题必须单独记账。

    混进"词表组"会让均值看起来像"词表试过但没用"，而实际上这一题词表根本没表态。
    """
    services = _services(conn, settings)
    rows = [
        {"question": QUESTION, "gold_snippets": [TEXTS[0]]},
        {"question": "装修完多久能住进去？", "gold_snippets": [TEXTS[1]]},
    ]
    term_map = TermMap([TermEntry("e", (("挂电视",),), ("专用连接件",), "s")])

    result = asyncio.run(
        _ev._evaluate(services, rows, 5, ["hybrid_parallel"], term_map=term_map)
    )

    assert result["term_uncovered"] == ["装修完多久能住进去？"]
    # 未覆盖的那题两条通道都是原话，改写串为空——不是把原话当术语塞进去。
    assert result["diagnostics"][1]["rewrites"] == []


class _RecordingReranker:
    """记下"重排器真正读到的查询"，而不是 `_evaluate` 传下来的选项。

    两者不是一回事：`_evaluate` 对所有行都传同一个 `rerank_query`，由 handler 决定
    基线行要不要退回原话。只断言传参的话，"基线被术语串污染"这个 bug 测不出来。
    """

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(self, query: str, candidates, top_k: int):
        self.queries.append(query)
        return list(candidates)[:top_k]


def test_term_rerank_switches_only_the_rerank_query(
    conn, settings: Settings, monkeypatch
) -> None:
    """`term_rerank` 只换重排读的串，且基线行必须退回原话。

    这一条把"重排读不读得懂口语"变成一个可测的事实：第四十一轮实测，同一份
    候选池，重排读口语问句时 gold 排 19/20/5/7/12，读术语串排 3/1/1/1/1。
    基线行如果不退回原话，同一份报告里的对照组也跟着变，`rerank` 与
    `rerank_parallel` 的差值就成了笔糊涂账。
    """
    services = _services(conn, settings)
    reranker = _RecordingReranker()
    services.reranker = reranker
    rows = [{"question": QUESTION, "gold_snippets": [TEXTS[0]]}]
    term_map = TermMap(
        [TermEntry("e", (("挂电视",), ("掉",)), ("墙面悬挂较重物体", "专用连接件"), "s")]
    )

    seen: list[dict] = []
    original = _ev.retrieve_for

    async def spy(services, mode, question, filt, k, keyword, *, rerank_query=None,
                  rewrites=(), rewrite_weight=0.5):
        seen.append({"question": question})
        return await original(
            services, mode, question, filt, k, keyword,
            rerank_query=rerank_query, rewrites=rewrites, rewrite_weight=rewrite_weight,
        )

    monkeypatch.setattr(_ev, "retrieve_for", spy)

    asyncio.run(
        _ev._evaluate(
            services, rows, 5, ["rerank", "rerank_parallel"],
            term_map=term_map, term_rerank="terms",
        )
    )

    assert [c["question"] for c in seen] == [QUESTION, QUESTION], "换了重排查询，召回串不该动"
    assert reranker.queries == [QUESTION, "墙面悬挂较重物体 专用连接件"], (
        "基线行必须读原话，只有 rerank_parallel 读术语串"
    )


def test_term_rerank_defaults_to_question(conn, settings: Settings, monkeypatch) -> None:
    """不给 `term_rerank` 时重排读原话——上一轮的对照不能被新开关悄悄改掉。"""
    services = _services(conn, settings)
    rows = [{"question": QUESTION, "gold_snippets": [TEXTS[0]]}]
    term_map = TermMap([TermEntry("e", (("挂电视",),), ("专用连接件",), "s")])

    seen: list[str | None] = []
    original = _ev.retrieve_for

    async def spy(services, mode, question, filt, k, keyword, *, rerank_query=None,
                  rewrites=(), rewrite_weight=0.5):
        seen.append(rerank_query)
        return await original(
            services, mode, question, filt, k, keyword,
            rerank_query=rerank_query, rewrites=rewrites, rewrite_weight=rewrite_weight,
        )

    monkeypatch.setattr(_ev, "retrieve_for", spy)

    asyncio.run(_ev._evaluate(services, rows, 5, ["hybrid_parallel"], term_map=term_map))

    assert seen == [QUESTION]


def test_term_map_and_expander_are_mutually_exclusive(tmp_path: Path, monkeypatch) -> None:
    """两套改写串同时注入，收益归不到任何一方，必须在开跑前拒绝。"""
    dataset = tmp_path / "d.jsonl"
    dataset.write_text(
        json.dumps({"question": QUESTION, "gold_snippets": [TEXTS[0]]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    class _Expander:
        available = True

        def expand(self, question: str) -> str:
            return f"{question} 术语"

    monkeypatch.setattr(_ev, "build_expander", lambda *a, **k: _Expander())

    with pytest.raises(ValueError, match="不能同用"):
        _ev.run(
            dataset, 5, ["hybrid_parallel"], term_map_path=DEFAULT_MAP, expand=True,
        )


def test_term_rerank_append_keeps_the_original_question(
    conn, settings: Settings, monkeypatch
) -> None:
    """`append` 形态下重排读「原话 + 术语串」，原话那一段不能丢。

    为什么要有这条：2026-10-29 在 v2 词表覆盖的 15 题上实测，纯术语串把三道**原本已经
    排第 1** 的题打出了前 5（原话 11/15、纯术语串 12/15、原话+术语串 14/15）。
    机制很直白——那几道题重排本来就能读懂，换成术语串等于拿掉信号。
    所以「术语只做加法」必须是一条被钉住的性质，而不是一个可以随手改的拼法。
    """
    services = _services(conn, settings)
    reranker = _RecordingReranker()
    services.reranker = reranker
    rows = [{"question": QUESTION, "gold_snippets": [TEXTS[0]]}]
    term_map = TermMap(
        [TermEntry("e", (("挂电视",), ("掉",)), ("墙面悬挂较重物体", "专用连接件"), "s")]
    )

    asyncio.run(
        _ev._evaluate(
            services, rows, 5, ["rerank", "rerank_parallel"],
            term_map=term_map, term_rerank="append",
        )
    )

    # 基线行仍然只读原话：对照组的定义不能因为换了形态而变
    assert reranker.queries[0] == QUESTION
    # 处理行是「原话 + 术语串」：原话在前、术语在后，不能只剩术语
    handled = reranker.queries[1]
    assert handled.startswith(QUESTION), f"原话被丢了：{handled!r}"
    assert "墙面悬挂较重物体" in handled and "专用连接件" in handled


def test_unknown_public_side_is_rejected(tmp_path: Path) -> None:
    """`public_side` 写错必须报错，不能静默退回 `vernacular`。

    这个字段决定评估时怎么分组：`recalled`（动笔时想到某道题）与 `vernacular`（按条文写）
    的结论正好相反——2026-10-29 实测，前者 5 题救 3，后者 10 题救 1。写成 `recall`、
    `seen` 之类拼错的形式，会被悄悄算进"按条文写"那一组，把两组的差别抹平，
    而报告上完全看不出异常。
    """
    path = _write(tmp_path, [_entry(public_side="recall")])

    with pytest.raises(ValueError, match="public_side 非法"):
        load_term_map(path)
