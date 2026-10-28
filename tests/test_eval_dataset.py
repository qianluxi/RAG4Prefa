"""评估集与解析器的契约：解析一改，`chunk_id` 就全变，标注会静默失效。

`chunk_id` 是内容寻址（`doc_id + 章节路径 + 正文哈希 + 同段序号`），所以任何一处
解析判据调整——多认一个标题、少认一行目录、分块阈值变化——都会让 id 整体换掉。
而 `eval/dataset*.jsonl` 里的 gold 是**具体 id**：不同步的话评估照样能跑完，
只是 Recall 会莫名其妙地低一截，看不出是标注过期还是检索退化。

这里守两道，一道在线一道离线：

1. **解析器金丝雀**（`eval/parser-canary.json`，要源文件）：挑 7 份形态各异、
   解析又快（合计约 3 秒）的文档，把当前解析器对它们的输出记下来。解析器一有改动，
   这 7 份的指纹就会变——它报的是"解析器变了"，不是"哪份文档变了"；定位靠下面第 2 条。
2. **标注是否还站得住**（`eval/corpus-snapshot.json`，不要源文件）：库里存了全量
   chunk_id，gold 是否还在语料中纯离线可判。

**为什么金丝雀不是全量 52 份。** 全量重解析实测 106 秒（其中 A1 一份就 63 秒），
放进每次 `pytest` 是把开发循环拖慢一倍；而且那份文档当初是**开着 OCR** 入库的，
纯离线解析本来就复现不出同样的 id，比对必然失败——它证明不了"解析器没变",
只能证明"OCR 结果不可复现"。所以金丝雀只取文字层完好的文档，按**格式与结构**取样本：
长多级编号 pdf、短公文 pdf、docx、政策 md（带模板噪声）、标准信息卡 md。
判据落在"解析器改没改"这件事上，粒度对了，成本才是 3 秒。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag4prefa import snapshot
from rag4prefa.adapters.parsers import SUPPORTED_SUFFIXES, parse_document
from rag4prefa.chunking import build_chunks

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT.parent / "工程文档"
CANARY = ROOT / "eval" / "parser-canary.json"
SNAPSHOT = ROOT / "eval" / "corpus-snapshot.json"
DATASETS = [ROOT / "eval" / "dataset.jsonl", ROOT / "eval" / "dataset.paraphrase.jsonl"]

# 后缀 → 解析器族。`.markdown` 与 `.md` 走同一个 `parse_markdown`，
# 各要一份样本只是把同一段代码验两遍。新格式进来时这里没有条目，就按后缀本身算，
# 于是会立刻要求一份样本——这正是期望的行为。
_FAMILY = {".md": ".md", ".markdown": ".md"}


def _suffixes(canary: dict) -> set[str]:
    return {Path(name).suffix.lower() for name in canary["documents"]}


@pytest.fixture(scope="module")
def canary() -> dict:
    if not CANARY.is_file():
        pytest.skip("没有解析器金丝雀记录，先跑 python scripts/build_manifest.py --canary")
    return json.loads(CANARY.read_text(encoding="utf-8"))


def test_canary_covers_every_supported_suffix(canary: dict) -> None:
    """金丝雀必须覆盖每一种**解析器**（不是每一种后缀）。

    这类记录最容易的"维护"方式是：某天它红了，就把红的那条删掉。删到只剩 md，
    它就永远绿着，而 pdf 解析器怎么改都没人知道。所以先钉住覆盖面。

    按"解析器"而不是按后缀算：`.md` 与 `.markdown` 走的是同一个函数，各要一份样本
    等于把同一段代码验两遍；而真正新增一种格式（比如 pptx）时，这条会立刻要求样本。
    """
    covered = {_FAMILY.get(suffix, suffix) for suffix in _suffixes(canary)}
    missing = [s for s in {_FAMILY.get(s, s) for s in SUPPORTED_SUFFIXES} if s not in covered]
    assert not missing, (
        f"金丝雀没有覆盖 {missing}。别为了让它变绿而删除条目——"
        "要么补一份该格式的样本，要么说明这种格式不再支持"
    )
    assert len(canary["documents"]) >= 5, (
        f"只剩 {len(canary['documents'])} 份样本，已不足以代表解析器的行为"
    )


def test_parser_canary_matches_recorded_fingerprints(canary: dict) -> None:
    """重新解析这几份样本，输出必须与记录一致。

    红了的含义只有一个：**解析器的输出变了**。这时先判断是有意改进还是回归，然后
    ——有意改进：`python scripts/build_manifest.py --canary` 更新记录，再跑
    `python scripts/relabel_eval.py --db data/app.db --write` 重算 gold；
    ——回归：改回去。

    别直接改记录把数字凑上：那份记录的全部价值就是"我没动过解析器"。
    """
    for filename, meta in canary["documents"].items():
        try:
            path = snapshot.locate_corpus_file(CORPUS_DIR, filename)
        except LookupError as exc:
            pytest.fail(f"{exc}。金丝雀样本重名会让比对指向另一份文件")
        if path is None:
            pytest.skip(
                f"本机没有这套语料（缺 {filename}）。注意这道校验此刻是失效的："
                "它守的是「解析器有没有变」，不是「结果对不对」"
            )
        chunks = build_chunks(
            parse_document(path),
            meta["doc_id"],
            max_tokens=canary["chunk_max_tokens"],
            min_tokens=canary["chunk_min_tokens"],
        )
        ids = sorted(chunk.id for chunk in chunks)
        assert len(ids) == meta["chunks"], (
            f"{filename} 的片段数从 {meta['chunks']} 变成 {len(ids)}："
            "解析或分块判据改动过，dataset 里的 gold 已失效"
        )
        assert snapshot.ids_digest(ids) == meta["ids_sha"], (
            f"{filename} 的 chunk_id 集合变了（数量未变）：章节路径或正文哈希变了，"
            "gold 已失效——重跑 relabel_eval 之前先确认这是有意的改进"
        )


@pytest.mark.skipif(
    not SNAPSHOT.is_file(), reason="没有语料快照，先跑 scripts/snapshot_corpus.py"
)
@pytest.mark.parametrize("dataset", DATASETS, ids=lambda p: p.name)
def test_dataset_gold_survives_without_source_files(dataset: Path) -> None:
    """源文件删掉之后，仍要能发现"标注已经过期"。

    2026-10-28 实测：补语料时旧文件被全部删除，依赖源文件的用例整体跳过。它们守的正是
    "解析器或语料一动，gold 就静默失效"——跳过等于这道守护悄悄消失，而且是以
    "跳过"而不是"失败"的姿态消失，看一眼测试摘要根本注意不到。

    快照里存了全量 chunk_id，于是同一件事不依赖源文件也能判：只要 gold 还在语料里，
    标注就还有效；不在，就必须重算。
    """
    snap = snapshot.load_snapshot(SNAPSHOT)
    assert snap, "快照读不出来（文件损坏？重跑 scripts/snapshot_corpus.py）"

    missing = snapshot.gold_missing(snap, dataset)
    assert not missing, (
        f"{dataset.name} 里 {len(missing)} 个 gold 已不在语料中：{missing[:3]}。"
        "标注与语料脱节，Recall 会偏低且看不出原因——按 anchor 重新生成标注"
    )
