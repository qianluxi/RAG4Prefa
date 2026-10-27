"""把没有文字层的页面送去识别（远端视觉模型，不装本地模型）。

## 为什么不用传统 OCR

项目约束是不引入本地模型（见 HANDOFF 的不变量）。Tesseract 要装系统二进制加
`chi_sim` 训练数据，PaddleOCR / RapidOCR 要落一份 ONNX 模型到磁盘——都算"本地模型"。
更要紧的是它们**只给文字**：一行行认出来，但不知道哪一行是一级标题、哪几行构成
一张表。而这两样恰恰是本项目切块的输入（章节树决定 `chapter_path`，`chunk_id` 又
包含章节路径）。

视觉模型一次调用能同时给出文字、层级与表格结构，代价是每页一次网络请求、几秒钟。

## 输出不可复现，这是本模块最重要的性质

同一页识别两次，标点、断行、甚至标题层级都可能差一点。而 `chunk_id` 是内容寻址的：
正文差一个字就是另一批 id。所以 `--force` 重入库会得到"内容相同、id 全变"的一份
新文档，旧向量原地成孤儿——`verify` 还显示两边都自洽，因为它只比 id 集合，不比
"这份文档是不是上次那份"。

因此**必须配 `ocr_cache`（按页图哈希缓存）**：缓存不是加速层，是这一步可重入的前提。
缓存键里带模型名和提示词版本，改了提示词就等于换了模型，旧缓存自动失效。

## 层级不交给模型

实测两个模型都在层级上翻过车：32B 把「独立建筑单体可不实施装配式建筑；」这种
上一段的续行标成了 `# 一级标题`——那是句子，不是标题。所以分工是：

* **视觉模型**回答"这一行像不像标题"（它看得见字号与位置，这是它擅长的）；
* **编号规则**（`headings.text_heading_level`）回答"是第几级"（编号是文档自带的，
  比模型数 `#` 可靠）。

这与 pdf 解析器的判据是同一个思路：视觉证据决定"是不是"，文字规则决定"第几级"。
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings
from ..ocr_profiles import EffectiveOcr, effective

# (页面 PNG 字节, 1-based 页码) -> 该页的 Markdown
OcrFn = Callable[[bytes, int], str]

# 提示词版本。改提示词必须 +1：输出格式变了，旧缓存里的文本就不能再当作"同一份"
# ——缓存键里带着它，不 +1 会静默沿用旧格式的结果。
#
# v2（2026-10-27）：v1 的规则 1 写了"承接上一段的续行一律不要加 #"，模型据此
# 过度保守——实测 Qwen3-VL-8B 在一份 24 页公文的第 2、8、9 页上**一条 # 都没给**，
# 于是 17 636 个字全落进同一节，章节路径全空。改成正面清单 + 例子 + "宁可多标"。
PROMPT_VERSION = "v2"

# 模型判定"这一页没有可读文字"时的约定回复。用它而不是空串，是因为空串没法
# 和"模型什么都没返回（截断、拒答）"区分开。
EMPTY_MARK = "<EMPTY>"

_SYSTEM = """你是工程文档的版面还原工具。把给定的这一页图片转成 Markdown，只输出 Markdown 本身。

规则：
1. 标题必须用 # 标记，# 的个数表示层级（# 一级，## 二级，### 三级）。
   满足任一条就是标题：独立成行且字号明显更大、整行加粗、以编号开头
   （一、/ 第X章 / 1.1 / （一）/ 1）。
   例如「（二）公共建筑类项目」「二、建筑单体预制率计算」「4.2.1 悬挑构造」都是标题。
   **宁可多标，不要漏标**：漏掉一个标题，这一节的内容就会并进上一节，
   章节归属全错；多标一个，最多是多一个短小节。
2. 以句号、分号、逗号、问号结尾的是句子，不是标题，即使以数字开头也不要加 #。
3. 表格转成竖线分隔的 Markdown 表格，保留表头；合并单元格按重复填写内容。
4. 不要输出页眉、页脚、页码、"第 X 页"、水印、印章上的文字。
5. 数字、百分比、编号必须照抄，不许改写、不许估算。认不出的字用 ▢ 占位，
   不要猜。
6. 如果整页没有任何可读文字（纯图片、纯表格线、空白页），只输出一行 <EMPTY>。
7. 不要任何解释、前言、结语，不要用代码块包裹输出。"""

_FENCE = re.compile(r"^```[a-zA-Z]*\s*$", re.MULTILINE)

# `probe()` 用的一张 64×64 纯白图，写死字节而不是现渲染：这条路径只回答
# 「Key 通不通」，不该为它拉起一个 PDF 文档（那要几十毫秒加一次文件句柄）。
#
# 尺寸是有讲究的：**不能太小**。实测 Qwen3-VL 对 16×16 直接回
# `400 height(16) or width(16) must be larger than 28`，于是「Key 完全正常」
# 被显示成「验证失败」——用户会去重建一把本来就没错的 Key。64 是两档都吃得下的尺寸。
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAS0lEQVR42u3PMQ0AAAwDoPo33UrYvQQc"
    "kD4XAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAYHLAMpT0sIcNbcE"
    "AAAAAElFTkSuQmCC"
)


@dataclass(frozen=True)
class PromptPreset:
    """一套提示词。换 preset 等于换了「怎么问」，缓存必须跟着失效。

    `version` 单独记，是因为同一套规则也会被改：v1→v2 那次改动让 8B 从
    「一整份 24 页公文一条 # 都不给」变成正常标层级，但缓存键没变的话，
    重跑会拿到旧格式的结果，看起来像"改了提示词却没生效"。
    """

    name: str
    version: str
    system: str
    user: str
    max_tokens: int
    # 由 `_ask` 决定要不要把页码塞进 user 消息。文档解析专用模型对额外文本敏感，
    # 给它的消息越短越接近它的训练分布。
    mention_page: bool = True


# 通用视觉模型的提示词：本项目逐条实测调出来的，规则都对应一个具体踩坑。
VLM_PRESET = PromptPreset(
    name="vlm",
    version=PROMPT_VERSION,
    system=_SYSTEM,
    user="这是第 {page} 页，请还原。",
    # 一页密排 A4 的中文正文在 2000 字上下，表格再多一点。给足余量，
    # 截断的表现是"后半页莫名消失"，比报错难查得多。
    max_tokens=4096,
)

# 文档解析专用模型（PaddleOCR-PaddleOCR-VL / DeepSeek-OCR 这类）用的是它们的原生指令。
#
# 不套 VLM 那套 7 条规则，是因为分工不同：专用模型自带版面分析，标题层级与表格结构
# 是它的输出格式本身；再叠一层"什么样的行才算标题"的文字规则，只会和它的版面判断
# 打架——尤其"宁可多标"这一条，会把它本来正确识别的正文推广成标题。
# 层级仍然不交给它一家说了算：`parsers/pdf.py` 拿到 markdown 后照样过
# `looks_like_title()` 与编号定级，与付费档走同一条后处理。
DOCUMENT_PRESET = PromptPreset(
    name="document",
    version="d1",
    system="",
    user="Convert the document to markdown.",
    # 专用解析模型单页输出可能到 8K：密排页 + 多张表格，4096 会截断。
    # 免费档不计费，给足余量没有代价。
    max_tokens=8192,
    mention_page=False,
)

PRESETS: dict[str, PromptPreset] = {
    VLM_PRESET.name: VLM_PRESET,
    DOCUMENT_PRESET.name: DOCUMENT_PRESET,
}


def preset_for(name: str) -> PromptPreset:
    """未知名字一律回落到 `vlm`——它是与旧缓存键一致的那一档。"""
    return PRESETS.get((name or "").strip().lower(), VLM_PRESET)



class OcrError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrOptions:
    """pdf 解析器需要知道的全部 OCR 参数。

    打包成一个对象而不是四个关键字参数，是因为它们总是一起出现、一起缺省；
    拆开传会让 `parse_document_report()` 的签名每加一个调整项就长一截。
    """

    fn: OcrFn
    mode: str = "auto"  # auto=只识别没有文字层的页；all=整份重做
    dpi: int = 150
    # 账单闸门，见 config.ocr_max_pages
    max_pages: int = 200
    workers: int = 4


@dataclass
class VlmOcr:
    """视觉模型识别器。可调用：`(png, page) -> markdown`。

    客户端延迟到第一次调用才建：`ingest` 每份文件都会构造一次这个对象，
    提前建客户端等于给批次里每份文件都开一遍连接池。
    """

    settings: Settings
    _client: OpenAI | None = field(default=None, init=False, repr=False)
    _eff: EffectiveOcr | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # 档位决定端点、模型、提示词、并发与 Key 的来源，集中在一处算一次。
        # 用户自己在界面上填的 OCR Key 已经由 `CredentialPool.settings_for()`
        # 按档位写进对应的 `.env` 字段，所以这里不需要知道 Key 是谁的。
        self._eff = effective(self.settings)
        self._preset = preset_for(self._eff.preset)
        missing = [
            name
            for name, value in (
                ("R4P_OCR_BASE_URL", self._eff.base_url),
                ("R4P_OCR_API_KEY", self._eff.api_key),
            )
            if not value
        ]
        if not self._eff.model:
            missing.append("R4P_OCR_MODEL")
        if missing:
            # 直接报错而不是偷偷回落到 LLM 那组配置：LLM 与 embedding 的 Key 是
            # 两个平台的两套账号，粘错了只会得到 401，而"为什么 OCR 一直失败"
            # 会变成一个要翻三个文件才能回答的问题。
            profile = self._eff.profile
            where = (
                "免费档读的是 R4P_OCR_FREE_API_KEY——硅基流动国内站（siliconflow.cn）"
                "与国际站是两套独立账号，国际站那把 Key 填进来会报 30014。"
                if profile.free
                else "付费档读的是 R4P_OCR_API_KEY。"
            )
            raise OcrError(
                f"扫描件识别已开启（{profile.label}），但缺少 {' / '.join(missing)}。"
                f"{where}"
                "OCR 走的是独立的视觉模型接口，和 embedding、LLM 的 Key 都不通用。"
                "在 .env 里补上，或在界面「API 设置」里填自己的 OCR Key；"
                "不想用 OCR 就把 R4P_OCR_MODE 设成 off。"
            )

    @property
    def eff(self) -> EffectiveOcr:
        """`__post_init__` 之后必定非空；这层断言只是给类型检查看的。"""
        if self._eff is None:  # pragma: no cover - 仅防御构造被绕过
            raise OcrError("OCR 识别器未初始化")
        return self._eff

    @property
    def model(self) -> str:
        """写进缓存的模型标识。

        通用视觉模型那一档**保持旧格式** `模型名#v2`：已有缓存按这个键存的，
        一旦换了写法，全部文档重跑都要重新付费识别一遍，而提示词一个字都没改。

        其他档位把 preset 名字与版本一起写进去——专用模型的输出格式与通用模型
        完全不同，两者绝不能共用一份缓存，否则换了档位会拿到「格式对不上的旧文本」，
        而且 `verify` 查不出来（它只比 id 自洽，不比内容是不是这一版）。
        """
        if self._preset is VLM_PRESET:
            return f"{self.eff.model}#{PROMPT_VERSION}"
        return f"{self.eff.model}#{self._preset.name}-{self._preset.version}"

    def _open(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(
                base_url=self.eff.base_url,
                api_key=self.eff.api_key,
                timeout=180.0,
            )
        return self._client

    def _messages(self, png: bytes, page: int) -> list[dict]:
        b64 = base64.b64encode(png).decode()
        preset = self._preset
        messages: list[dict] = []
        # 专用解析模型不带 system：它的行为由训练分布决定，多一段中文规则
        # 只会引出"解释一下我为什么这么标"这类跑偏输出。
        if preset.system:
            messages.append({"role": "system", "content": preset.system})
        text = preset.user.format(page=page) if preset.mention_page else preset.user
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        )
        return messages

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=15), reraise=True)
    def _ask(self, png: bytes, page: int) -> str:
        resp = self._open().chat.completions.create(
            model=self.eff.model,
            messages=self._messages(png, page),
            temperature=0,
            # 一页密排 A4 的中文正文在 2000 字上下，表格再多一点。给足余量，
            # 截断的表现是"后半页莫名消失"，比报错难查得多。
            max_tokens=self._preset.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def probe(self) -> str:
        """真发一次最小请求，返回模型给的文本（可能为空）。

        刻意**不走 `_ask`**：那条路径带指数退避重试，错 Key 要等十几秒，
        点「保存并验证」的人会以为按钮卡死。这里只试一次，失败就如实报。

        用一张 64×64 的纯白图，而不是凭空构造文本请求：专用解析模型只吃图，
        纯文本请求会被它以 400 拒掉，而 400 与 401 在用户眼里长得一样。
        """
        resp = self._open().chat.completions.create(
            model=self.eff.model,
            messages=self._messages(_TINY_PNG, 1),
            temperature=0,
            max_tokens=8,
        )
        return resp.choices[0].message.content or ""

    def close(self) -> None:
        """关掉底层连接池。`ingest` 每份文件构造一次，用完要还。"""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __call__(self, png: bytes, page: int) -> str:
        try:
            raw = self._ask(png, page)
        except Exception as exc:  # noqa: BLE001 - 统一成一种错误，页码必须带上
            raise OcrError(
                f"第 {page} 页识别失败（{type(exc).__name__}: {exc}）。"
                "已识别过的页在缓存里，重跑同一份文件只会补没识别的那些页。"
            ) from exc
        return clean_markdown(raw)


def clean_markdown(text: str) -> str:
    """去掉模型爱加的包裹与解释，并把"这一页没字"归成空串。

    不能只 `strip()`：实测模型会自己套一层 ```markdown 围栏，围栏行保留下来会变成
    正文的两个片段（"```markdown" 成为一行），进而是两条向量、两个引用角标。
    """
    out = text.strip()
    if EMPTY_MARK in out:
        return ""
    out = _FENCE.sub("", out).strip()
    if out.startswith("```"):
        out = out[3:].strip()
    if out.endswith("```"):
        out = out[:-3].strip()
    return out
