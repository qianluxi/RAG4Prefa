"""回答前的证据闸门：能说出具体理由时拒答，而不是把候选硬塞给大模型。

`prompts.SYSTEM_ANSWER` 第 3 条要求模型"片段不足就说明资料中未涉及"，但那只是提示词
约束：候选非空时结果照样被递进去，而模型很愿意基于"主题相似"的片段给一个看起来合理
的答案。库外集 6 道题在每种检索方式下都召回了非空候选（北京问出宁波的方案、日本问出
绍兴的图集），就是这个现象的直接证据。

**这一层的边界必须说清楚**：它不判断"这段话是否真的支持这个结论"——那需要语义蕴含，
不是规则能做到的。它只做一件很窄的事：在能命名一个**确定性理由**时拒答。

三类信号，只有前两类阻断：

1. `no_hits`：一条候选都没有。本来就在 `stream_answer` 里处理，这里收拢到同一处，
   免得"拒答"有两个互相不知道的入口。
2. `geo_absent`：问句点名了某个地域，而库里**没有一份文件是关于它的**（文件名不含）。
   判据是文件名而不是正文，也不是"候选像不像"：向量检索永远会返回最近的一条，
   而正文提到地名往往是参编单位名单、企业简介、区域规划里的顺带一提——本库正文有
   17 条片段提到"北京"，却没有一份文件是北京的资料。用正文判断，库外问题会被当成
   "库里有"。
3. `weak_top`：最相关的那条候选，重排分低到连"沾边"都算不上。门槛必须实测标定
   （`scripts/calibrate_refusal.py`），因为重排服务的绝对值会随版本漂：同一配置
   跨时段 0.839→0.806（`eval/report-identity-20261029.md`）。拿一个拍脑袋的 0.5
   写在代码里，等于把某一天的远程服务状态固化成了产品行为。

**故意不做的**：拿词形覆盖判"证据不足"。口语题里词形对不上是常态（"有味道上来" ≠
"返味"、"看到水印" ≠ "渗漏"），拿覆盖率阻断会把大量本来答得出的题一起拒掉——
那是把一种静默错误换成另一种。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from . import prompts, repo
from .domain import Hit

# 地域名表。用途只有一个：判断"库里有没有这个地域的资料"，不参与任何语义判断。
#
# 港澳台是中国的一部分，放在同一张表里没有别的含义：这里既不做国家/地区分类，
# 也不据此排序，只是问"这批文件里提没提过这四个字"。
PLACES: tuple[str, ...] = (
    # 省级行政区
    "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南",
    "广东", "海南", "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "内蒙古", "广西", "西藏", "宁夏", "新疆", "香港", "澳门", "台湾",
    # 浙江省内城市（语料主体所在，写全是为了让"浙江的库里有没有这个市"能判）
    "杭州", "宁波", "温州", "绍兴", "湖州", "嘉兴", "金华", "台州", "丽水", "衢州", "舟山",
    # 其他常被拿来对比的城市
    "苏州", "南京", "无锡", "常州", "深圳", "广州", "珠海", "佛山", "东莞",
    "青岛", "济南", "烟台", "大连", "沈阳", "长春", "哈尔滨", "西安", "成都",
    "武汉", "长沙", "合肥", "福州", "厦门", "南昌", "郑州", "石家庄", "太原",
    "昆明", "贵阳", "南宁", "兰州", "乌鲁木齐", "呼和浩特", "银川", "西宁",
    "拉萨", "海口", "三亚",
    # 常被引用的境外体系（同样只用来判"库里有没有"）
    "日本", "美国", "德国", "新加坡", "英国", "法国", "韩国",
)


# 城市 → 所属省级行政区。存在的唯一理由：**省级文件的适用范围覆盖省内城市**。
#
# 2026-10-30 专家复审第四节：文件名里没有"杭州"就拒答，会把一个明明答得出的问题拒掉
# ——库里只有《浙江省…标准》，而省标适用于省内城市。这是误拒，比漏拒更贵。
#
# 直辖市（北京/天津/上海/重庆）故意**不在这里**：它们的上级是全国，而"全国标准符合要求"
# 不等于"某市的申报口径就是这么定的"。把它们挂到全国会让 7/7 的地域拒答全部失效——
# 库外集里问"北京市的补贴标准"，语料里有国家标准，但没有一份关于北京的文件。
PROVINCE_OF: dict[str, str] = {
    "浙江": "浙江",
    "杭州": "浙江", "宁波": "浙江", "温州": "浙江", "绍兴": "浙江", "湖州": "浙江",
    "嘉兴": "浙江", "金华": "浙江", "台州": "浙江", "丽水": "浙江", "衢州": "浙江",
    "舟山": "浙江",
    "南京": "江苏", "苏州": "江苏", "无锡": "江苏", "常州": "江苏",
    "广州": "广东", "深圳": "广东", "珠海": "广东", "佛山": "广东", "东莞": "广东",
    "济南": "山东", "青岛": "山东", "烟台": "山东",
    "沈阳": "辽宁", "大连": "辽宁",
    "长春": "吉林", "哈尔滨": "黑龙江", "西安": "陕西", "成都": "四川",
    "武汉": "湖北", "长沙": "湖南", "合肥": "安徽", "福州": "福建", "厦门": "福建",
    "南昌": "江西", "郑州": "河南", "石家庄": "河北", "太原": "山西",
    "昆明": "云南", "贵阳": "贵州", "南宁": "广西", "兰州": "甘肃",
    "乌鲁木齐": "新疆", "呼和浩特": "内蒙古", "银川": "宁夏", "西宁": "青海",
    "拉萨": "西藏", "海口": "海南", "三亚": "海南",
}


@dataclass(frozen=True)
class Signal:
    """一条判据。`blocking` 为真才会导致拒答，其余只写进审计信息。"""

    name: str
    blocking: bool
    detail: str


@dataclass(frozen=True)
class Verdict:
    refuse: bool
    signals: list[Signal] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [s.detail for s in self.signals if s.blocking]

    def as_dict(self) -> dict:
        return {
            "refuse": self.refuse,
            "reasons": self.reasons,
            "signals": [
                {"name": s.name, "blocking": s.blocking, "detail": s.detail} for s in self.signals
            ],
        }


def find_places(question: str) -> list[str]:
    """问句里点到的地域，按表内顺序返回（同一地域只算一次）。"""
    return [name for name in PLACES if name in question]


def coverage_place(name: str) -> str | None:
    """这个名字靠哪一级文件才是"有资料"。自己就是省级/直辖市级时返回 None。"""
    parent = PROVINCE_OF.get(name)
    return parent if parent and parent != name else None


def _top_score(hits: Sequence[Hit]) -> tuple[float | None, str]:
    """最相关那条候选的分数，以及它是哪种分。

    有重排分就用重排分（它是对"这段能不能回答这个问题"的判断），没有才退回稠密
    相似度（它只说"这两段话像不像"）。两者量纲不同，门槛必须分开配，不能共用一个数。
    """
    top = hits[0]
    if top.rerank_score is not None:
        return top.rerank_score, "rerank"
    if top.dense_score is not None:
        return top.dense_score, "dense"
    return None, "none"


def judge(
    conn: sqlite3.Connection,
    question: str,
    hits: Sequence[Hit],
    *,
    min_rerank: float | None = None,
    min_dense: float | None = None,
    corpus_mentions: Callable[[str], int] | None = None,
) -> Verdict:
    """给出拒答判断。纯函数式：除了那次"库里提没提过"的查询，不动任何状态。"""
    signals: list[Signal] = []
    if not hits:
        signals.append(Signal("no_hits", True, "当前检索范围内没有候选片段"))
        return Verdict(True, signals)

    mentions = corpus_mentions or (lambda needle: repo.documents_mentioning(conn, needle))
    # 城市问题先看本市，再看省：省级文件适用于省内城市，这一条不成立就会误拒一类
    # 本来答得出的问题（见 `PROVINCE_OF`）。
    by_upper: list[tuple[str, str]] = []
    absent: list[str] = []
    for name in find_places(question):
        if mentions(name) > 0:
            continue
        parent = coverage_place(name)
        if parent and mentions(parent) > 0:
            by_upper.append((name, parent))
        else:
            absent.append(name)
    if absent:
        listed = "、".join(absent)
        signals.append(
            Signal(
                "geo_absent",
                True,
                f"问句问的是「{listed}」，而库里没有一份关于它的文件，也没有覆盖它的上级文件；"
                "只凭主题相近的片段作答，会把别处的规定当成这个地域的规定",
            )
        )
    for name, parent in by_upper:
        # 非阻断信号：拒答是错的，但"这是省级口径"必须让用户看见——否则答案会被读成
        # 该市自己的规定。
        signals.append(
            Signal(
                "geo_upper_level",
                False,
                f"库里没有关于{name}的文件，只有{parent}的资料；下面这些片段是省级口径，"
                f"未必是{name}本地的具体规定",
            )
        )

    score, kind = _top_score(hits)
    floor = min_rerank if kind == "rerank" else min_dense
    if score is not None and floor is not None and score < floor:
        signals.append(
            Signal(
                "weak_top",
                True,
                f"最相关候选的{kind}分 {score:.4f} 低于门槛 {floor:.4f}，"
                "连沾边都算不上",
            )
        )

    return Verdict(any(s.blocking for s in signals), signals)


def explain(verdict: Verdict) -> str:
    """给用户看的一句话。不说"我拒绝回答"，说清楚缺的是什么。"""
    if not verdict.refuse:
        return ""
    return "资料不足，暂不能据此作答：" + "；".join(verdict.reasons) + "。"


# ── 第二层：让模型自己判"这批片段够不够" ─────────────────────
#
# 规则层只能判"能命名理由"的三类，`near_topic`（主题相近、缺的正是被问的那个事实）
# 在规则层做不到——那需要语义蕴含。所以补一个判官：不回答，只回答"够不够"。
#
# **它比"在回答提示词里求模型别猜"强的地方只有一个，但很关键**：判官的输出是结构化
# 的两行，可以逐题核对、误判能归因；提示词的效果只能从最终答案里间接体会。
#
# 代价同样明确：每题多一次模型调用，且判官本身会错。所以它默认**不参与生产判定**
# （`refuse_mode` 不含 judge），先由 `scripts/judge_sufficiency.py` 在冻结候选上量
# 漏拒与误拒，数字过关才谈接入。

_NO = ("不可", "不能", "不可以", "不足以", "不够", "无法", "否")
_YES = ("可", "能", "是")


@dataclass(frozen=True)
class JudgeReply:
    """判官的一次回答。`can_answer` 为 None 表示格式不可解析（不是"判了不可"）。"""

    can_answer: bool | None
    reason: str
    raw: str = ""
    attempts: int = 1


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped.strip("*#「」『』[]【】（）() -—·\t　")
    return ""


def parse_judge_reply(text: str, *, attempts: int = 1) -> JudgeReply:
    """解析成 (能不能答, 理由)。

    `can_answer is None` 与 `False` 必须分开：前者是"输出不合格式，什么都没判"，
    后者是"判了不可"。把两者合成一个布尔，格式故障会被当成一次成功的拒答，
    报告上的拒答率就会因为模型话多而虚高。

    先判否定词再判肯定词：`不可` 与 `可`、`不能` 与 `能` 在 `str.startswith` 下本来
    不会互相吃掉（前者的首字是「不」），但这层顺序加上 `不…` 兜底，才挡得住
    「不可以」「不足」这类三个字的变体。
    """
    head = _first_line(text)
    if not head:
        return JudgeReply(None, "", text, attempts)
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    reason = lines[1] if len(lines) > 1 else ""

    if head.startswith(_NO):
        return JudgeReply(False, reason, text, attempts)
    if head.startswith(_YES):
        return JudgeReply(True, reason, text, attempts)
    # 首行不是判断词（模型先说了别的），退回整段找判断词——但只认开头那几个词，
    # 免得「不可」出现在理由里被当成否定结论。
    if head.startswith("不"):
        return JudgeReply(False, reason, text, attempts)
    return JudgeReply(None, "", text, attempts)


def judge_by_llm(
    question: str,
    hits: Sequence[Hit],
    complete: Callable[[str, str], str],
    *,
    top_n: int = 5,
    per_snippet: int = 400,
    retry: bool = True,
    system: str | None = None,
) -> JudgeReply:
    """问一次判官。`complete(system, user) -> str` 由调用方注入，便于离线测试。

    没有候选片段时不问模型：这是规则层已经能判的事，多一次调用只会多一个出错的机会。

    `system` 用来切换判官版本（`prompts.JUDGE_PROMPTS`）：v1 只有一条口径，
    v2 按"取值型/做法型"分开判。**重试沿用同一个 system**，否则判了两次却是两把尺子。
    """
    if not hits:
        return JudgeReply(False, "没有候选片段")
    system = system or prompts.SYSTEM_JUDGE
    reply = parse_judge_reply(
        complete(
            system,
            prompts.build_judge_prompt(question, list(hits), top_n=top_n,
                                      per_snippet=per_snippet),
        )
    )
    if reply.can_answer is None and retry:
        # 只重问格式，不换判据：换了判据，两次回答就不再是同一把尺子。
        again = parse_judge_reply(
            complete(
                system,
                prompts.build_judge_retry_prompt(question, list(hits), top_n=top_n,
                                                 per_snippet=per_snippet),
            ),
            attempts=2,
        )
        return JudgeReply(again.can_answer, again.reason, again.raw, again.attempts)
    return reply
