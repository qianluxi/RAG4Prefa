"""公众表达 → 规范用语 的受控词表。

为什么要有这一层（第四十轮，前一轮诊断逼出来的）：

口语集上真正卡住的不是融合、不是池深、不是重排，而是**词不是条文里的那个搭配**。
同样的 gold 片段，「会不会掉下来」查不到，而条文原话「专用连接件」一查就进第 2 名；
「老房子能不能用」查不到，「参照执行」第 5 名。差距不在语义相似度，在术语。

LLM 术语扩展救不回来的原因也在这里：它给的是**近义概念**（锚固、更换、既有建筑、
检验批、照明灯具），而 BM25 的分数取决于**原文里的那个搭配**。近义词再准确，
没有共现就换不成分数。

所以这一层的职责被刻意压到最小：
- 词表是人写的、每条都带 source（能追到条文原话），不由模型生成；
- 匹配是**子串判断**，不分词、不加权重，命中与否可以肉眼复核；
- 命中后只输出**术语列表**，怎么注入由调用方决定（见 `rewrite` 的两种形态）。

它不负责理解意图——那是 LLM 该做的事，但这层故意不做，因为它要能被审计：
"为什么这题注入了这几个词"必须能在 JSON 里一行行看出来。

用法：
    term_map = TermMap.load(Path("rag4prefa/terms/public_to_standard.json"))
    terms = term_map.match("墙上想挂电视和吊柜，会不会掉下来？")
    # → ["墙面悬挂较重物体", "专用连接件"]
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TermEntry:
    """一条「公众说法 → 规范术语」映射。

    `groups` 是若干组可选子串，**每组都必须至少命中一个**才认为整条成立。
    分组的理由：单个词太松（"挂" 与 "掉" 分别出现在完全无关的问句里），
    而写成一整句又太紧（换个语序就漏）。两组交叉能把「挂东西 + 会不会掉」
    和「补贴 + 多少钱」这类近邻话题区分开。
    """

    id: str
    groups: tuple[tuple[str, ...], ...]
    terms: tuple[str, ...]
    source: str
    # 公众表达侧的可信度：`vernacular` = 按该领域的日常说法写的；`recalled` =
    # 动笔时已经想到某道具体题目。评估要按它分开算——`recalled` 的那些不能算泛化证据。
    # 它默认 `vernacular` 仅仅是给 v1 词表（那时还没有这个概念）留的兼容口子，
    # 新写的词表都应当显式写明。
    public_side: str = "vernacular"

    def matches(self, question: str) -> bool:
        return all(any(alt in question for alt in group) for group in self.groups)

    def explain(self, question: str) -> str:
        """命中了哪几个子串。诊断时要回答"这题为什么注入了这几个词"。"""
        hit = [
            next((alt for alt in group if alt in question), "")
            for group in self.groups
        ]
        return " + ".join(x for x in hit if x)


class TermMap:
    """一组 TermEntry。构造即校验，坏词表在启动时炸，不在检索时静默失效。"""

    def __init__(self, entries: Sequence[TermEntry]) -> None:
        self.entries: tuple[TermEntry, ...] = tuple(entries)

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def load(cls, path: Path | str) -> TermMap:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = raw.get("entries")
        if not isinstance(items, list) or not items:
            raise ValueError(f"词表 {path} 里没有 entries（空词表等于没启用，别让它悄悄跑）")

        entries: list[TermEntry] = []
        seen: set[str] = set()
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"词表 {path} 第 {index} 条不是对象")
            entry_id = str(item.get("id") or "").strip()
            if not entry_id:
                raise ValueError(f"词表 {path} 第 {index} 条缺 id")
            if entry_id in seen:
                raise ValueError(f"词表 {path} 里 id 重复：{entry_id}")
            seen.add(entry_id)

            groups = tuple(
                tuple(str(alt).strip() for alt in group if str(alt).strip())
                for group in (item.get("groups") or [])
            )
            # 空的组会让 all() 恒真 → 该条对所有问题都命中，是词表最常见的坏法：
            # 检索被塞进无关术语，而报告上只看得出"召回变差"。
            if not groups or any(not group for group in groups):
                raise ValueError(f"词表 {path} 的 {entry_id} 有空的分组，会让它对所有问题都命中")

            terms = tuple(
                str(term).strip() for term in (item.get("terms") or []) if str(term).strip()
            )
            if not terms:
                raise ValueError(f"词表 {path} 的 {entry_id} 没有术语")

            # 写错会**静默**退回默认值，把「记得住的题」混进「按条文写的」那一组——
            # 而这两组的结论正好相反，混在一起等于把结论洗掉。所以这里直接报错。
            public_side = str(item.get("public_side") or "vernacular").strip()
            if public_side not in ("vernacular", "recalled"):
                raise ValueError(
                    f"词表 {path} 的 {entry_id} 的 public_side 非法：{public_side!r}"
                    "（只能是 vernacular / recalled）"
                )

            source = str(item.get("source") or "").strip()
            # 没有出处就无法区分"条文原话"与"模型意译的近义词"，而这两者的检索效果
            # 恰恰是这轮实验要证明有差别的东西。
            if not source:
                raise ValueError(f"词表 {path} 的 {entry_id} 缺 source（必须能追到条文原话）")

            entries.append(TermEntry(entry_id, groups, terms, source, public_side))
        return cls(entries)

    def hits(self, question: str) -> list[TermEntry]:
        return [entry for entry in self.entries if entry.matches(question)]

    def match(self, question: str) -> list[str]:
        """命中的规范术语（去重、按词表顺序）。

        没有命中时返回 `[]`，**不是**返回原话：调用方必须自己决定怎么办
        （评估里要把它记成"词表未覆盖"，否则这些题会被当成"词表试过了、没用"）。
        """
        found: list[str] = []
        for entry in self.hits(question):
            for term in entry.terms:
                if term not in found:
                    found.append(term)
        return found

    def rewrite(self, question: str, *, form: str = "terms") -> list[str]:
        """把命中的术语变成检索用的改写串。

        - `terms`：每个术语一条独立通道。**默认**，因为它不稀释——拼成一串时，
          长术语会被短术语的重复词压下去（实测逐词总比拼串好）。
        - `joined`：所有术语拼成一条通道。保留它是为了给"多一条通道"与"多一个串"
          这件事留下可比的证据，不是为了省事。
        """
        terms = self.match(question)
        if not terms:
            return []
        if form == "terms":
            return list(terms)
        if form == "joined":
            return [" ".join(terms)]
        raise ValueError(f"未知的改写形态：{form}（只支持 terms / joined）")

    def explain(self, question: str) -> list[dict[str, object]]:
        """逐条说明命中情况，写进评估的逐题证据里。"""
        return [
            {
                "id": entry.id,
                "matched_by": entry.explain(question),
                "terms": list(entry.terms),
            }
            for entry in self.hits(question)
        ]


def load_term_map(path: Path | str | None) -> TermMap | None:
    """路径为空则返回 None（= 不启用），便于 eval 直接把命令行参数透传进来。"""
    if path is None or str(path).strip() == "":
        return None
    return TermMap.load(path)


def merge_terms(*groups: Iterable[str]) -> list[str]:
    """按出现顺序合并多组术语并去重。调用方偶尔需要把 LLM 词与词表词合成一路。"""
    merged: list[str] = []
    for group in groups:
        for term in group:
            text = (term or "").strip()
            if text and text not in merged:
                merged.append(text)
    return merged
