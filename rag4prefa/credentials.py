"""按用户解析 API 凭据：Key 由浏览器保管，服务端不落库。

为什么用户只能换 Key、不能换模型
================================
索引的向量空间是全库唯一的（见 `consistency.py`）。如果允许每人自选 embedding 模型，
A 用 bge-m3、B 用 text-embedding-v3，同一个集合里就并存两套向量——余弦相似度照样
算得出来，只是语义错位，检索结果看起来正常但不可信。

所以 `base_url` / `model` / `dim` 一律由服务端 `.env` 统一决定，用户只提供 Key
（解决"用谁的额度"），不参与"用哪个模型"。这不是省事，是多用户共享一个索引的必要条件。

安全边界（写清楚，别让人误以为是鉴权）
====================================
- 这**不是身份认证**：知道用户名的人可以填自己的 Key 顶替，服务端不校验密码。
- Key 存在浏览器 localStorage，服务端只在请求期间持有，不落库、不写日志。
- 因此它只适合「可信内网 / 单机演示」。公网部署必须换成真正的登录 + 服务端密钥托管
  （加密存储、可吊销、可审计），届时还应该顺便把上传与删除接口一起收进鉴权。
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import unquote

from .adapters.embed import Embedder, EmbeddingError
from .adapters.llm import LLM, LLMError
from .config import Settings

HEADER_USER = "X-R4P-User"
HEADER_EMBEDDING_KEY = "X-R4P-Embedding-Key"
HEADER_LLM_KEY = "X-R4P-LLM-Key"
HEADER_LLM_MODEL = "X-R4P-LLM-Model"
HEADER_OCR_KEY = "X-R4P-OCR-Key"

_MAX_NAME = 64
_MAX_KEY = 4096
_MAX_MODEL = 128


@dataclass(frozen=True)
class LLMOption:
    """一个可选的对话模型。`note` 写实测出来的差异，不写营销词。"""

    id: str
    label: str
    note: str


# 为什么 embedding 不能选、LLM 可以选
# ================================
# embedding 决定向量空间：换模型等于换一套语义坐标，同一个集合里混两套向量，
# 余弦相似度照样算得出来，只是结果不可信。所以它必须锁死。
# LLM 只读「已检索到的片段」并组织语言，不参与索引，换它只影响措辞与推理深度，
# 不影响库里任何一条向量——因此可以放开。
#
# 这份清单是 2026-10-24 逐个真调出来的，不是照抄 /v1/models：那个目录里
# DeepSeek V4 全系、MiniMax-M3、Step-3.5-Flash 都在，但调用一律返回
# "has no provider supported"。列在目录里 ≠ 真的能调通，只收实测通过的。
#
# 被排除的候选与原因（留着，免得下次又去试）：
#   ZhipuAI/GLM-5.2            先能调通，几次之后就 429 "exceeded your current quota"
#                              —— 它吃付费配额，不符合"免费额度"的定位
#   ZhipuAI/GLM-4.7-Flash      持续 429「访问量过大」，免费池排队严重
#   deepseek-ai/DeepSeek-V4-*  目录里有，调用全部 "has no provider supported"
#   MiniMax-M3 / Step-3.5-Flash / LongCat  同上，目录挂了但无 provider
LLM_CHOICES: tuple[LLMOption, ...] = (
    LLMOption(
        "Qwen/Qwen3.5-35B-A3B",
        "Qwen3.5-35B-A3B（默认）",
        "MoE 架构，质量与速度均衡，日常问答的稳妥选择",
    ),
    LLMOption(
        "Qwen/Qwen3.8-Flash-Next",
        "Qwen3.8-Flash-Next（最快）",
        "实测约 1.3 秒出答案，适合连续追问",
    ),
    LLMOption(
        "Qwen/Qwen3.8-27B",
        "Qwen3.8-27B",
        "新一代 dense 模型，长片段理解更稳",
    ),
    LLMOption(
        "Qwen/Qwen3.5-122B-A10B",
        "Qwen3.5-122B-A10B（最强）",
        "参数量最大，复杂跨章节问题首选；高峰期可能限流",
    ),
)

_LLM_IDS = frozenset(option.id for option in LLM_CHOICES)


def resolve_llm_model(value: str, fallback: str = "") -> str:
    """把任意输入收敛到白名单内。空串或未知名字一律回落 fallback。

    不做「未知就报错」：模型名来自浏览器，前端旧版本可能还存着已下线的名字，
    为此让整个请求失败不值得。静默回落 + `verify` 里能看出实际用了哪个模型即可。
    """
    candidate = (value or "").strip()[:_MAX_MODEL]
    return candidate if candidate in _LLM_IDS else fallback


@dataclass(frozen=True)
class UserCredentials:
    """一次请求携带的凭据。所有 Key 都可为空，空表示沿用服务端公共配置。"""

    username: str
    embedding_key: str = ""
    llm_key: str = ""
    llm_model: str = ""
    # 扫描件识别是第三个独立的付费环节（视觉模型），Key 与上面两把都不通用。
    # 只传 Key，不传"用哪个模型"——模型由服务端锁死，理由见 `ocr_setup` 模块头：
    # 换模型会让同一份文件重新切块、换一套 chunk_id，代价远大于收益。
    ocr_key: str = ""

    @property
    def has_keys(self) -> bool:
        return bool(self.embedding_key or self.llm_key or self.ocr_key)

    @property
    def fingerprint(self) -> str:
        """缓存键用哈希：内存里的字典键也不该是明文 Key。

        **必须把 llm_model 算进去**：同名 Key 换模型时要拿到新的 LLM 实例，
        否则池子里那份还指着旧模型，界面上显示已切换、实际却在跑旧模型。
        """
        raw = f"{self.embedding_key}\x00{self.llm_key}\x00{self.llm_model}".encode()
        return hashlib.sha256(raw).hexdigest()[:32]


# 每个花钱的环节各需要哪一把 Key。
#
# 为什么要按角色分开判（2026-10-27 复审 P1-1）：`from_headers()` 只要带**任意**一个
# 头就返回非空凭据——只传 `X-R4P-LLM-Model` 也算。于是"凭据非空就放行"会把
# 「只填了 embedding Key、却要出答案」的请求放过去，而 `CredentialPool.settings_for()`
# 会把缺的那把静默补成服务端 Key：配置写着"不代付"，账单上却照付。
# 判据必须落到"这个接口真正要用的那把 Key 是用户自己的吗"。
KEY_ROLES: dict[str, tuple[str, str]] = {
    "embedding": (HEADER_EMBEDDING_KEY, "检索与入库要把文本变成向量"),
    "llm": (HEADER_LLM_KEY, "生成答案要调用对话模型"),
    "ocr": (HEADER_OCR_KEY, "扫描件识别要调用视觉模型"),
}

# 角色 -> 凭据字段。用表而不是三元表达式：加第三个角色时，`else` 分支会安静地
# 把「OCR 有没有自己的 Key」读成「LLM 有没有」，于是 strict 模式对 OCR 永远放行。
_ROLE_FIELDS = {"embedding": "embedding_key", "llm": "llm_key", "ocr": "ocr_key"}


def user_key(creds: UserCredentials | None, role: str) -> str:
    """取这次请求里用户自己提供的那把 Key，没提供就是空串。"""
    if creds is None:
        return ""
    field_name = _ROLE_FIELDS.get(role)
    return getattr(creds, field_name, "") if field_name else ""


def missing_roles(creds: UserCredentials | None, need: Sequence[str]) -> list[str]:
    """在"服务端不代付"的前提下，`need` 里哪些角色的 Key 用户还没提供。"""
    return [role for role in need if not user_key(creds, role)]


def from_headers(headers: Mapping[str, str]) -> UserCredentials | None:
    """从请求头解析凭据。一个识别字段都没有时返回 None——表示用服务端公共配置。

    只给其中一个也合法：例如只带自己的 embedding Key（入库花自己的额度），
    问答仍走服务端的 LLM Key。缺哪个就沿用哪个，不做「必须成对」的强校验。

    只带模型头 / 只带 OCR Key 同样合法：用服务端公共额度跑自选配置。所以判空要把
    **每个**识别字段都看一遍，少看一个就会把「只填了 OCR Key」的请求当成未登录——
    那样入库时服务端拿不到那把 Key，用户以为在用自己的额度，其实在花公共额度。

    用户名按 `encodeURIComponent` 的约定解码：HTTP 头只能是 ASCII，中文用户名
    不编码前端会直接抛 TypeError。纯英文名解码后不变，不影响 curl 用法。
    """
    embedding_key = (headers.get(HEADER_EMBEDDING_KEY) or "").strip()[:_MAX_KEY]
    llm_key = (headers.get(HEADER_LLM_KEY) or "").strip()[:_MAX_KEY]
    llm_model = (headers.get(HEADER_LLM_MODEL) or "").strip()[:_MAX_MODEL]
    ocr_key = (headers.get(HEADER_OCR_KEY) or "").strip()[:_MAX_KEY]
    if not any((embedding_key, llm_key, llm_model, ocr_key)):
        return None
    raw_name = (headers.get(HEADER_USER) or "").strip()
    name = unquote(raw_name)[:_MAX_NAME] or "anonymous"
    return UserCredentials(name, embedding_key, llm_key, llm_model, ocr_key)


# Key 的形态能认出平台。只收能确定的：认不出来就交给厂商判断，不要瞎猜。
_SIGNATURES = (
    ("ms-", "魔搭（ModelScope）"),
    ("ds-", "魔搭（ModelScope）"),
    ("sk-", "硅基流动 / 阿里百炼 / OpenAI"),
)

# 端点接受哪种前缀。硅基流动、百炼、OpenAI 都发 sk- 开头的 Key，所以这层只能
# 判断"明显不是这家的"（例如把 ms- 填给硅基流动），判断不了 sk- 之间的串用。
#
# 硅基流动分国内站与国际站，是两套独立账号：Key 不通用，向量模型目录也不同
# （国内站 BAAI/bge-m3，国际站 Qwen3-Embedding 系列）。两个域名互不为子串，
# 所以要分别列出，否则国际站的地址会认不出来、连平台名都提示不了。
_ENDPOINTS = (
    ("modelscope.cn", ("ms-", "ds-"), "魔搭（ModelScope）"),
    ("siliconflow.cn", ("sk-",), "硅基流动（国内站）"),
    ("siliconflow.com", ("sk-",), "硅基流动（国际站）"),
    ("dashscope.aliyuncs.com", ("sk-",), "阿里百炼"),
    ("api.openai.com", ("sk-",), "OpenAI"),
)


def platform_of(base_url: str) -> str:
    """从接口地址认出是哪家平台，认不出就返回空串。"""
    url = (base_url or "").lower()
    for domain, _, name in _ENDPOINTS:
        if domain in url:
            return name
    return ""


def platform_mismatch(key: str, base_url: str, *, label: str = "embedding") -> str | None:
    """Key 与接口地址明显不属于同一家时，返回一句能照着改的提示；否则 None。

    场景很具体：设置面板里只有两个输入框、形状完全一样，把魔搭令牌填进 embedding
    一栏是很容易发生的事。用户随后收到硅基流动的 `401 Token is invalid`，自然会去
    反复核对令牌有没有复制错——方向从一开始就偏了。这层判断不发请求、零成本。
    """
    if not key:
        return None
    url = (base_url or "").lower()
    for domain, accepted, name in _ENDPOINTS:
        if domain not in url:
            continue
        if key.startswith(accepted):
            return None
        owner = next((label_ for prefix, label_ in _SIGNATURES if key.startswith(prefix)), "")
        lead = f"这个 Key 看起来是{owner}签发的，" if owner else ""
        return (
            f"{lead}但 {label} 请求发往{name}（{domain}）。"
            f"每一家的 Key 只能配它自己的接口地址，请换成{name}签发的 Key。"
        )
    return None


def auth_failure_hint(base_url: str, key: str = "") -> str:
    """鉴权失败时补一句能照做的提醒：Key 有效但填错平台，表现同样是 401。"""
    name = platform_of(base_url)
    if not name:
        return ""
    hint = f"（该 Key 未被{name}接受：确认它由{name}签发，不同平台的 Key 不能混用）"
    if name.startswith("硅基流动"):
        # 这家的国内站与国际站是两套独立账号，两边都发 sk-、都叫硅基流动，
        # 唯一的区别是 Application 域名。填串了会收到 30014 "Token is invalid"，
        # 与「Key 被删」是同一个错误码——不说这一条，用户只会反复重建 Key。
        hint += (
            "硅基流动的国内站（siliconflow.cn）与国际站（siliconflow.com）是两套独立账号，"
            "Key 与向量模型目录都不通用；在其中一个站点申请的 Key 填给另一个站点的地址，"
            "收到的正是这个错误。请核对申请 Key 时的域名。"
        )
    elif key.startswith("sk-"):
        # 前缀能对上也不能证明是这家的：sk- 被硅基流动 / 百炼 / DeepSeek / 智谱共用，
        # 所以 `platform_mismatch()` 对 sk- 一律放行。真报错时这里再补一句，
        # 否则用户会以为是"复制漏了几位"。
        hint += (
            f"sk- 前缀被多家共用，请确认这把 Key 确实在{name}控制台生成"
            f"（别家签发的 sk- 在{name}这里同样报 Token 无效）。"
        )
    return hint

# 「Key 没错，是账号状态不让调」的厂商特征词。这类失败常被误读成 Key 失效，
# 用户于是反复重建 Key，而真正要做的是去控制台完成验证或充值。
_ACCOUNT_MARKERS: tuple[tuple[str, str], ...] = (
    ("complete your profile", "账号还没完成邮箱验证"),
    ("email verification", "账号还没完成邮箱验证"),
    ("verified account", "账号未通过验证"),
    ("only available to verified", "账号未通过验证"),
    ("arrearage", "账号欠费"),
    ("insufficient balance", "账户余额不足"),
    ("insufficient_quota", "额度已用尽"),
    ("quota exceeded", "额度已用尽"),
    ("exceeded your current quota", "额度已用尽"),
    ("余额不足", "账户余额不足"),
    ("欠费", "账号欠费"),
)

# 厂商没在错误里给自助链接时，按域名兜一个准确性已知的入口。
# 硅基流动两站的域名互不为子串，各配各的，不会串。
_ACCOUNT_PAGES = {
    "modelscope.cn": "https://modelscope.cn/my/settings/account",
    "siliconflow.cn": "https://cloud.siliconflow.cn/account/overview",
    "siliconflow.com": "https://cloud.siliconflow.com/account/overview",
}


def account_blocked_hint(message: str, base_url: str) -> str:
    """识别「鉴权已通过、但账号状态不允许调用」，返回一句能照着做的提示；否则空串。

    与 `auth_failure_hint()` 的区别是根因不同：那边是 Key 不被接受，这边 Key 完全有效，
    卡在邮箱验证 / 实名 / 欠费 / 额度。两种情况报错都长得像"调不通"，但用户该做的事
    完全不同——一个去重新生成 Key，一个去验证账号。
    """
    lowered = (message or "").lower()
    reason = next((text for mark, text in _ACCOUNT_MARKERS if mark in lowered), "")
    if not reason:
        return ""
    host = (base_url or "").split("//")[-1].split("/")[0].lower()
    # 厂商在错误里给的自助链接最准（还带上下文），优先用它。
    # 链接通常夹在 JSON 里，尾部会粘上引号或标点，不剥掉用户点不开。
    link = next(
        (
            word.strip(".,;:)]}\"'）】」")
            for word in (message or "").split()
            if word.startswith("http")
        ),
        "",
    )
    if not link:
        link = next((page for domain, page in _ACCOUNT_PAGES.items() if domain in host), "")
    tail = f"处理入口：{link}" if link else "请到该平台控制台检查邮箱验证 / 实名 / 余额状态。"
    return f"（这不是 Key 的问题：{reason}。该 Key 已通过鉴权，只是账号状态不允许调用。{tail}）"


def settings_for_ocr(settings: Settings, key: str) -> Settings:
    """把用户填的 OCR Key 落到 `Settings` 上。留空就沿用服务端那把。

    `CredentialPool.settings_for()` 与 `/api/credentials/verify` 共用这一处，
    免得两边的回落规则长歪：一处漏了回落，就会出现"验证用服务端 Key 通过、
    入库却说没配 Key"。
    """
    if not key:
        return settings
    return settings.model_copy(update={"ocr_api_key": key})


class CredentialPool:
    """按 Key 指纹复用 Embedder / LLM 客户端。

    为什么需要池：每个请求都 new 一个 OpenAI client，底层 httpx 的连接会不断堆积，
    不关就是泄漏。这里按指纹复用，超出上限淘汰最久未用的，并关闭它的连接。

    上限取 32：这是「几个人各自带 Key」的场景，不是多租户；真要成百上千用户，
    应该换成服务端托管的密钥 + 真正的连接池，而不是把上限调大。
    """

    def __init__(
        self, settings: Settings, conn: sqlite3.Connection, *, limit: int = 32
    ) -> None:
        self._settings = settings
        self._conn = conn
        self._limit = max(1, limit)
        self._items: OrderedDict[str, tuple[Embedder | None, LLM | None]] = OrderedDict()

    def __len__(self) -> int:
        return len(self._items)

    def settings_for(self, creds: UserCredentials) -> Settings:
        """替换 Key 与 LLM 模型，**embedding 模型和维度一律沿用服务端配置**。

        边界由用途决定：LLM 只读已检索到的片段、不参与建索引，换它不影响库里
        任何向量；embedding 模型换一个就是换一套语义坐标，必须锁死（见模块头）。
        OCR 只影响"这次进来的文本长什么样"，也不产生向量，所以它能换 Key，
        但**不能换模型**——换模型会让同一份文件重新切块、换一套 chunk_id，
        而 `ocr_cache` 的键里带着模型名，等于整份重新付费识别一遍。
        """
        base = self._settings.model_copy(
            update={
                "embedding_api_key": creds.embedding_key or self._settings.embedding_api_key,
                "llm_api_key": creds.llm_key or self._settings.llm_api_key,
                "llm_model": resolve_llm_model(creds.llm_model, self._settings.llm_model),
            }
        )
        return settings_for_ocr(base, creds.ocr_key)

    def acquire(self, creds: UserCredentials) -> tuple[Embedder | None, LLM | None]:
        """拿到（必要时创建）该凭据对应的客户端。

        构造阶段不会泄露 Key 是否有效：OpenAI 客户端创建时不发请求，
        Key 错了要等第一次调用才暴露。想提前发现请走 `/api/credentials/verify`。
        """
        cached = self._items.get(creds.fingerprint)
        if cached is not None:
            self._items.move_to_end(creds.fingerprint)
            return cached

        local = self.settings_for(creds)
        try:
            embedder: Embedder | None = Embedder(local, self._conn)
        except EmbeddingError:
            embedder = None
        try:
            llm: LLM | None = LLM(local)
        except LLMError:
            llm = None

        item = (embedder, llm)
        self._items[creds.fingerprint] = item
        self._items.move_to_end(creds.fingerprint)
        self._evict()
        return item

    def close(self) -> None:
        while self._items:
            _, item = self._items.popitem(last=False)
            _close_item(item)

    def _evict(self) -> None:
        while len(self._items) > self._limit:
            _, item = self._items.popitem(last=False)
            _close_item(item)


def _close_item(item: tuple[Embedder | None, LLM | None]) -> None:
    for client in item:
        if client is None:
            continue
        with suppress(Exception):  # pragma: no cover - 关闭失败不该影响请求
            client.close()
