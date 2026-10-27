"""扫描件识别用哪一组参数、以及"现在到底能不能用"。

为什么只剩一条路：免费档被实测否决
==================================
2026-10-27 用一份 24 页扫描件（上海 250 号文）把两条免费路线都跑了一遍，结论是
**都不能用**，于是整个移除：

* 硅基流动国内站 `PaddlePaddle/PaddleOCR-VL-1.5`（官方标价 ¥0）
  - 8 页里 2 页输出**伪造内容**：第 3 页是纯中文正文（无表格无公式），却整页编成
    英文的"圆的面积是 πr²"，150/120/100 dpi 三个分辨率全部复现；第 8 页退化成
    `1. 1. 2. 3. 4. …` 的数字重复，一路顶到 max_tokens。
  - 单页 55 秒，公式页 250 秒；2 并发跑 17 分钟只完成 8/24 页。
    对照付费档：同样 24 页 3 分 39 秒（4 并发）。
* `deepseek-ai/DeepSeek-OCR`：一页满屏 `}}}]`，另一页 0.9 秒只返回 3 个字符。

伪造正文比识别失败危险得多——它会被切块、写向量、被引用成答案，而且**不报错**。
所以现在只有一条路：`.env` 的 `R4P_OCR_*`（默认国际站 Qwen3-VL-8B，一份 24 页约 1.4 角）。
没配好就**不处理扫描件**，错误消息前面带 `OCR_SETUP_HINT`，前端据此弹出配置引导。

为什么 OCR 能换模型，embedding 不能
====================================
embedding 换模型等于换一套语义坐标：同一个集合里混两套向量，余弦相似度照样算得出来，
只是结果不可信。所以它必须全库唯一（见 `credentials` 模块头）。

OCR 不产生向量。它把像素变成文本，文本再走同一条切块与向量化流程——换 OCR 模型只影响
「这次进来的文本长什么样」，不改变任何一条已有向量。真正要防的是**同一份文档两次入库
得到不同 `chunk_id`**，那由 `ocr_cache`（键里带模型名与提示词版本）挡住。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EffectiveOcr:
    """这次真正要用的那组参数。

    单独有一层，是因为有两个来源要合并：`.env` 里的服务端配置，和用户按请求头带过来的
    Key。`health`、`verify`、入库、CLI 都从这里取，避免各算各的。
    """

    base_url: str
    model: str
    concurrency: int
    api_key: str

    @property
    def ready(self) -> bool:
        """三件套齐了才算配好。缺一样都不许开始识别——见模块头。"""
        return bool(self.base_url and self.model and self.api_key)


def effective(settings, user_key: str = "") -> EffectiveOcr:
    """按配置算出实际参数。`user_key` 非空时压过服务端那把。

    Key **不回落**到 LLM 或 embedding：那是另外两个平台的账号，粘错了只会得到 401，
    而"为什么 OCR 一直失败"会变成一个要翻三个文件才能回答的问题。
    """
    return EffectiveOcr(
        base_url=getattr(settings, "ocr_base_url", "") or "",
        model=getattr(settings, "ocr_model", "") or "",
        # 页与页互相独立，并发只影响等待时间；.env 没填就给 4。
        concurrency=getattr(settings, "ocr_concurrency", 0) or 4,
        api_key=(user_key or "").strip() or (getattr(settings, "ocr_api_key", "") or ""),
    )
