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
- 匹配是**子串判断**，不分词、不加权重，命中与否可以肉眼复核；唯一的例外是单字——
  单字在中文里是语素不是词（`修` 会命中「装修/检修」），组内还有多字备选时不算证据，
  要破例得在数据里逐条声明 `single_char_evidence`；
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

# 意图类别：这个词条解决的是**哪一类问句**。
#
# 它不是运行时分类器——这层刻意不做语义理解（见文件头）。真正的门控仍然写在
# `groups` 里；`intent` 只干三件事：给误触发率分组、组织负例、让报告能说清
# "哪一类词条最容易错触发"。把它当分类器用的人会失望，这是有意的。
INTENTS = (
    "boundary",      # 品类 / 适用范围边界（吊顶含不含电器、农村自建房算不算）
    "practice",      # 施工做法与构造（检修口、同层排水、防火封堵）
    "performance",   # 性能指标与等级（空鼓、燃烧性能、水封高度）
    "policy",        # 政策申领与结算流程（补贴、发票、保修责任）
    "acceptance",    # 验收与交付（隐蔽工程验收）
    "prohibition",   # 禁止性行为（承重墙能不能拆）
    "unlabelled",    # 兼容 v1 老词表：那时还没有这个概念
)


def _group_hit(group: Sequence[str], question: str, *, strict: bool = True) -> str:
    """这一组命中了哪个子串；没命中返回空串。

    `strict` 管的是**单字备选**：组内只要还有多字备选，仅靠单字命中就不算这一组
    命中。理由很具体——单字在中文里是语素不是词，「修」会命中「装修 / 检修 / 维修」，
    「板」会命中「楼板 / 板材 / 踢脚板」，「级」会命中「高级 / 首级」。用它们当证据，
    等于把"词出现在句子里"当成"用户在说这件事"。

    这条规则是**靠数据补回来的**：哪道题真的需要单字语境，就把那个语境写成多字备选
    （`裂开` / `开裂`），别让引擎去猜。`strict=False` 保留老口径，只用于量化
    "收紧挡掉了多少"这类反事实，生产与评测都走默认值。
    """
    found = [alt for alt in group if alt in question]
    if not found:
        return ""
    if strict and all(len(alt) == 1 for alt in found) and any(len(alt) > 1 for alt in group):
        return ""
    return found[0]


@dataclass(frozen=True)
class TermEntry:
    """一条「公众说法 → 规范术语」映射。

    `groups` 是若干组可选子串，**每组都必须至少命中一个**才认为整条成立。
    分组的理由：单个词太松（"挂" 与 "掉" 分别出现在完全无关的问句里），
    而写成一整句又太紧（换个语序就漏）。两组交叉能把「挂东西 + 会不会掉」
    和「补贴 + 多少钱」这类近邻话题区分开。

    **分组要把"意图词"和"对象词"分开。** 第一版把
    `[包含|算不算|电器|灯具|浴霸]` 写成一个 OR 组，于是「吊顶安装灯具的构造要求」
    也会命中并注入补贴品类表里的「不含电器设备」——词对了、意错了。现在
    「包含/算不算」单列一组，「电器/灯具」单列一组，缺任何一边都不触发。
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
    intent: str = "unlabelled"
    # 「这条映射是靠单字撑起来的」——一个必须写在数据里的坏消息。
    #
    # 默认 `False`（单字不算证据）。为 `True` 等于声明"我知道这条在拿语素当词用"，
    # 只有两个合法的理由：①历史词表（v1/v2）要靠它复现旧报告，改了就复现不了；
    # ②作者逐条看过语料，确认这个字在本领域不会撞上别的词。新写的映射一律不要开。
    #
    # 为什么不在代码里按"单字/多字"一刀切，而要留这个口子：v1 的主力映射
    # （挂电视 + 会不会掉）第二组全是单字（掉/塌/坠/牢），硬规则会把它整条废掉，
    # 而它恰恰是 15 题实验里把 5 道鸿沟题全推进前 5 的那条。规则对，但不能不认历史。
    single_char_evidence: bool = False

    def matches(self, question: str, *, strict: bool = True) -> bool:
        strict = strict and not self.single_char_evidence
        return all(
            _group_hit(group, question, strict=strict) for group in self.groups
        )

    def explain(self, question: str) -> str:
        """命中了哪几个子串。诊断时要回答"这题为什么注入了这几个词"。"""
        strict = not self.single_char_evidence
        hit = [_group_hit(group, question, strict=strict) for group in self.groups]
        return " + ".join(x for x in hit if x)

    def dropped_singles(self, question: str) -> list[str]:
        """被单字规则挡掉的子串。

        少了这个，"词表为什么没触发"就只剩"没命中"一种解释，而实际有两种：
        真的没命中，或者命中的全是单字（`装修` 里的 `修`）。审计要能分开。
        声明了 `single_char_evidence` 的条目不参与这条规则，这里恒返回空。
        """
        if self.single_char_evidence:
            return []
        out: list[str] = []
        for group in self.groups:
            hit = _group_hit(group, question, strict=False)
            if hit and not _group_hit(group, question):
                out.append(hit)
        return out


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

            # 和 public_side 同样的道理：拼错的 intent 会静默落进 unlabelled，
            # 而那正是"分类没做"与"分类写错"分不开的地方。
            intent = str(item.get("intent") or "unlabelled").strip()
            if intent not in INTENTS:
                raise ValueError(
                    f"词表 {path} 的 {entry_id} 的 intent 非法：{intent!r}"
                    f"（只能是 {' / '.join(INTENTS)}）"
                )

            entries.append(
                TermEntry(
                    entry_id,
                    groups,
                    terms,
                    source,
                    public_side,
                    intent,
                    bool(item.get("single_char_evidence", False)),
                )
            )
        return cls(entries)

    def hits(self, question: str, *, strict: bool = True) -> list[TermEntry]:
        return [entry for entry in self.entries if entry.matches(question, strict=strict)]

    def match(self, question: str, *, strict: bool = True) -> list[str]:
        """命中的规范术语（去重、按词表顺序）。

        没有命中时返回 `[]`，**不是**返回原话：调用方必须自己决定怎么办
        （评估里要把它记成"词表未覆盖"，否则这些题会被当成"词表试过了、没用"）。
        """
        found: list[str] = []
        for entry in self.hits(question, strict=strict):
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

    def explain(self, question: str, *, strict: bool = True) -> list[dict[str, object]]:
        """逐条说明命中情况，写进评估的逐题证据里。

        `dropped` 非空表示"这条本该命中、被单字规则挡下了"——误触发审计要能
        把这种情况和"压根没命中"分开，否则收紧的代价看不见。
        """
        out: list[dict[str, object]] = []
        for entry in self.entries:
            if entry.matches(question, strict=strict):
                out.append(
                    {
                        "id": entry.id,
                        "intent": entry.intent,
                        "matched_by": entry.explain(question),
                        "terms": list(entry.terms),
                    }
                )
            elif entry.matches(question, strict=False):
                # 只可能是单字规则挡下的（其余条件没变）。
                out.append(
                    {
                        "id": entry.id,
                        "intent": entry.intent,
                        "matched_by": "",
                        "dropped": entry.dropped_singles(question),
                        "terms": list(entry.terms),
                    }
                )
        return out


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
