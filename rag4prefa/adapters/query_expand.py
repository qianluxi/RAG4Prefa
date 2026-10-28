"""查询术语扩展：把口语问句扩成「原话 + 规范术语」，缩小口语与原文的词表鸿沟。

为什么存在（2026-10-28 口语集逐题归因）：15 道口语题里 5 道三种召回都捞不进
候选池，gold 都在库里、内容也对得上，只是用户的大白话和规范术语之间隔着一条
**知识**鸿沟——「会不会掉下来」↔「悬挂较重物体应采用专用连接件」。向量和 BM25
都靠「词面/语义相似」匹配，跨不过「掉下来 = 悬挂较重物体」这层推理；换 embedding
模型、调 rrf_k 都救不了，只能在召回前把口语翻成术语变体。

设计要点：
- **扩展不是替换**。输出是「原话 + 术语」，两个通道一起吃。替换有风险——LLM 改偏了，
  原来能命中的也丢了；扩展最坏情况是回到现在的表现，不会更差。
- **失败降级回原话**。LLM 不可用、超时、返回空，都当没这回事，检索照常。
- **输出是关键词串，不是新问句**。BM25 吃词面，向量吃语义锚点，都不需要自然语言
  问句；关键词串更短、更稳、也更便宜。
"""

from __future__ import annotations

from ..config import Settings
from .llm import LLM

# 提示词固定写死，不随模型变：术语扩展的语义就是「把大白话翻译成规范术语」，
# 换模型不该改变这个契约。要求输出「纯关键词、空格分隔、一行」，省掉解析。
_SYSTEM = (
    "你是建筑工程规范检索的术语专家。把用户口语化的装修/建筑问题，翻译成规范、"
    "标准、政策文件里会用的术语关键词。只输出关键词，用空格分隔，不要句子，"
    "不要解释，不要标点。如果没有把握，输出原句里的关键词即可。"
)


class QueryExpander:
    """把问句扩成「原话 + 术语」的扩展查询。

    `expand(question) -> str` 返回扩展后的查询串；任何失败都返回原问句。
    """

    def __init__(self, settings: Settings, llm: LLM | None = None) -> None:
        self._settings = settings
        # 不在这里 new LLM：LLM 构造会因缺 Key 抛错，而扩展是可选项，
        # 不该因为没配 Key 就让整个服务起不来。用的时候再拿，拿不到就降级。
        self._llm = llm

    @property
    def available(self) -> bool:
        """扩展是否真的会做事——没有 LLM 就一定是 no-op。"""
        return self._llm is not None

    def expand(self, question: str) -> str:
        """返回「原话 + 术语」的扩展查询；失败/无 LLM 时原样返回 question。"""
        if self._llm is None:
            return question
        try:
            terms = self._llm.complete(_SYSTEM, question, max_tokens=96).strip()
        except Exception:  # noqa: BLE001 - 扩展失败一律降级，不影响检索
            return question
        if not terms:
            return question
        return f"{question} {terms}"


def build_expander(settings: Settings, llm: LLM | None) -> QueryExpander:
    """按配置构造扩展器；无 LLM 时返回一个 no-op（available=False）。"""
    return QueryExpander(settings, llm)
