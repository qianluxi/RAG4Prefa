"""OCR 档位：让「零成本」与「高精度」各自成立，而不是让用户自己拼配置。

为什么 OCR 能换模型，embedding 不能
====================================
embedding 换模型等于换一套语义坐标：同一个集合里混两套向量，余弦相似度照样算得出来，
只是结果不可信。所以它必须全库唯一（见 `credentials` 模块头）。

OCR 不产生向量。它把像素变成文本，文本再走同一条切块与向量化流程——换 OCR 模型只影响
「这次进来的文本长什么样」，不改变任何一条已有向量。真正要防的是**同一份文档两次入库
得到不同 `chunk_id`**，那由 `ocr_cache`（键里带模型名与提示词版本）挡住，不靠锁死模型。

两档各自解决什么
================
`free`：硅基流动**国内站**的文档解析专用模型 `PaddlePaddle/PaddleOCR-VL-1.5`，官方计价项
        是 ¥0。0.9B，快；输出原生 markdown，`#` 标题与竖线表格都是它自带的能力，正好对上
        本项目切块要的三样东西。代价是免费池并发受限，页数多时更慢。
`paid`：完全沿用 `.env` 的 `R4P_OCR_*`（默认国际站 Qwen3-VL-8B，约 1.4 角/24 页）。
        通用视觉模型，指令遵循更好、能按我们的规则标层级。

为什么免费档要单独一把 Key
==========================
硅基流动的国内站与国际站是**两套独立账号**，Key 不通用，两站的模型目录也不同：免费那两个
OCR 模型只在 `api.siliconflow.cn` 有，而默认部署的 `R4P_OCR_API_KEY` 是国际站的 Key。
一把 Key 配不了两个站，所以免费档读 `R4P_OCR_FREE_API_KEY`，**不回落**到
`R4P_OCR_API_KEY`——回落只会让国际站的 Key 被发到国内站，收到 `30014 Token is invalid`，
与「Key 被删」同一个错误码，用户会去反复重建一把本来就没错的 Key。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfileId = Literal["free", "paid"]


@dataclass(frozen=True)
class OcrProfile:
    """一个可选的识别档位。

    `base_url` / `model` / `preset` / `concurrency` 为空表示「不覆盖，用 .env 里的值」——
    付费档就是这样，管理员换厂商只改 `.env`，不必回来改代码。
    """

    id: ProfileId
    label: str
    note: str
    free: bool
    base_url: str = ""
    model: str = ""
    preset: str = ""
    concurrency: int = 0


# 免费档的端点与模型写死在代码里，而不是读 .env：这两个值是该站「这一步零成本」的全部
# 依据（计价项 free-*、模型专用文档解析）。放进 .env 迟早被人改成付费模型，界面上却
# 还挂着「免费」两个字——那比不给免费档更糟。
FREE_PROFILE = OcrProfile(
    id="free",
    label="免费档（PaddleOCR-VL 文档解析）",
    note=(
        "硅基流动国内站，官方标价 ¥0；文档解析专用模型，原生输出 Markdown 标题与表格。"
        "免费池并发受限，页数多时比付费档慢；需要国内站（siliconflow.cn）签发的 Key。"
    ),
    free=True,
    base_url="https://api.siliconflow.cn/v1",
    model="PaddlePaddle/PaddleOCR-VL-1.5",
    preset="document",
    # 免费池并发给太大会持续 429。两页并行仍比串行快，又不容易撞限流。
    concurrency=2,
)

PAID_PROFILE = OcrProfile(
    id="paid",
    label="付费档（通用视觉模型）",
    note=(
        "端点、模型与并发全部取自 .env 的 R4P_OCR_*；默认国际站 Qwen3-VL-8B，"
        "一份 24 页文档约 1.4 角。通用模型指令遵循更好，能按本项目的规则标标题层级。"
    ),
    free=False,
)

OCR_PROFILES: dict[str, OcrProfile] = {
    FREE_PROFILE.id: FREE_PROFILE,
    PAID_PROFILE.id: PAID_PROFILE,
}

# 不填就沿用旧行为：`.env` 里怎么配就怎么跑。改默认为 free 会让老部署在一次重启后
# 突然换了识别模型，已入库的文档不受影响，但重跑一份就会得到另一批 chunk_id。
DEFAULT_PROFILE: ProfileId = "paid"


def resolve_profile(value: str, fallback: str = "") -> ProfileId:
    """把任意输入收敛到两档之一。

    与 `credentials.resolve_llm_model` 同样的取舍：浏览器里可能存着旧名字或手滑输入，
    为此让整个请求失败不值得，静默回落到已知档位即可——`health` 会把实际用的档位报出来。
    """
    name = (value or "").strip().lower()[:16]
    if name in OCR_PROFILES:
        return name  # type: ignore[return-value]
    back = (fallback or "").strip().lower()
    if back in OCR_PROFILES:
        return back  # type: ignore[return-value]
    return DEFAULT_PROFILE


def key_for(settings, profile_id: str = "") -> str:
    """该档位该用哪把 Key。

    免费档只认国内的 `R4P_OCR_FREE_API_KEY`（理由见模块头），付费档沿用
    `R4P_OCR_API_KEY`。用户自己在界面上填的 Key 优先级最高，由调用方传入。
    """
    if resolve_profile(profile_id, getattr(settings, "ocr_profile", "")) == "free":
        return getattr(settings, "ocr_free_api_key", "") or ""
    return getattr(settings, "ocr_api_key", "") or ""


@dataclass(frozen=True)
class EffectiveOcr:
    """某一档位下真正要用的那组参数。

    为什么不把档位「写回」`Settings`：付费档的端点、模型、并发在 `.env` 里，免费档
    写在上面那两个常量里。如果构造期用免费档覆盖掉 `settings.ocr_base_url`，同一进程
    里再切回付费档就取不回国际站那个地址了——`Settings` 是**唯一一份**，覆盖之后
    原始值就没了。所以覆盖只发生在这一层，`Settings` 始终保留 `.env` 原样。
    """

    profile: OcrProfile
    base_url: str
    model: str
    preset: str
    concurrency: int
    api_key: str

    @property
    def ready(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)


def effective(settings, profile_id: str = "", user_key: str = "") -> EffectiveOcr:
    """按档位算出实际参数。`user_key` 非空时压过服务端那两把。"""
    name = resolve_profile(profile_id, getattr(settings, "ocr_profile", ""))
    profile = OCR_PROFILES[name]
    return EffectiveOcr(
        profile=profile,
        base_url=profile.base_url or getattr(settings, "ocr_base_url", "") or "",
        model=profile.model or getattr(settings, "ocr_model", "") or "",
        preset=profile.preset or getattr(settings, "ocr_preset", "") or "vlm",
        concurrency=profile.concurrency or getattr(settings, "ocr_concurrency", 0) or 4,
        api_key=(user_key or "").strip() or key_for(settings, name),
    )
