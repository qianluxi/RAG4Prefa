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

from . import repo
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
    absent = [name for name in find_places(question) if mentions(name) <= 0]
    if absent:
        listed = "、".join(absent)
        signals.append(
            Signal(
                "geo_absent",
                True,
                f"问句问的是「{listed}」，而库里没有一份资料是关于它的；"
                "只凭主题相近的片段作答，会把别处的规定当成这个地域的规定",
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
