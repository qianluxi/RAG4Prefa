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
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from urllib.parse import unquote

from .adapters.embed import Embedder, EmbeddingError
from .adapters.llm import LLM, LLMError
from .config import Settings

HEADER_USER = "X-R4P-User"
HEADER_EMBEDDING_KEY = "X-R4P-Embedding-Key"
HEADER_LLM_KEY = "X-R4P-LLM-Key"

_MAX_NAME = 64
_MAX_KEY = 4096


@dataclass(frozen=True)
class UserCredentials:
    """一次请求携带的凭据。两个 Key 都可为空，空表示沿用服务端公共配置。"""

    username: str
    embedding_key: str = ""
    llm_key: str = ""

    @property
    def has_keys(self) -> bool:
        return bool(self.embedding_key or self.llm_key)

    @property
    def fingerprint(self) -> str:
        """缓存键用哈希：内存里的字典键也不该是明文 Key。"""
        raw = f"{self.embedding_key}\x00{self.llm_key}".encode()
        return hashlib.sha256(raw).hexdigest()[:32]


def from_headers(headers: Mapping[str, str]) -> UserCredentials | None:
    """从请求头解析凭据。两个 Key 头都不存在时返回 None——表示用服务端公共 Key。

    只给其中一个也合法：例如只带自己的 embedding Key（入库花自己的额度），
    问答仍走服务端的 LLM Key。缺哪个就沿用哪个，不做「必须成对」的强校验。

    用户名按 `encodeURIComponent` 的约定解码：HTTP 头只能是 ASCII，中文用户名
    不编码前端会直接抛 TypeError。纯英文名解码后不变，不影响 curl 用法。
    """
    embedding_key = (headers.get(HEADER_EMBEDDING_KEY) or "").strip()[:_MAX_KEY]
    llm_key = (headers.get(HEADER_LLM_KEY) or "").strip()[:_MAX_KEY]
    if not embedding_key and not llm_key:
        return None
    raw_name = (headers.get(HEADER_USER) or "").strip()
    name = unquote(raw_name)[:_MAX_NAME] or "anonymous"
    return UserCredentials(name, embedding_key, llm_key)


# Key 的形态能认出平台。只收能确定的：认不出来就交给厂商判断，不要瞎猜。
_SIGNATURES = (
    ("ms-", "魔搭（ModelScope）"),
    ("ds-", "魔搭（ModelScope）"),
    ("sk-", "硅基流动 / 阿里百炼 / OpenAI"),
)

# 端点接受哪种前缀。硅基流动、百炼、OpenAI 都发 sk- 开头的 Key，所以这层只能
# 判断"明显不是这家的"（例如把 ms- 填给硅基流动），判断不了 sk- 之间的串用。
_ENDPOINTS = (
    ("modelscope.cn", ("ms-", "ds-"), "魔搭（ModelScope）"),
    ("siliconflow.cn", ("sk-",), "硅基流动"),
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
    """鉴权失败时补一句通用提醒：Key 有效但填错平台，表现同样是 401。"""
    name = platform_of(base_url)
    if not name:
        return ""
    hint = f"（该 Key 未被{name}接受：确认它由{name}签发，不同平台的 Key 不能混用）"
    # 前缀能对上也不能证明是这家的：sk- 被硅基流动 / 百炼 / DeepSeek / 智谱共用，
    # 所以 `platform_mismatch()` 对 sk- 一律放行。真报错时这里再补一句，
    # 否则用户会以为是"复制漏了几位"。
    if key.startswith("sk-") and name in _SK_SHARED:
        hint += (
            f"sk- 前缀被多家共用，请确认这把 Key 确实在{name}控制台生成"
            f"（别家签发的 sk- 在{name}这里同样报 Token 无效）。"
        )
    return hint


# 这些平台都发 sk- 开头的 Key，光看形态分不出来
_SK_SHARED = frozenset({"硅基流动", "阿里百炼", "OpenAI"})

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

# 厂商没在错误里给自助链接时，按域名兜一个准确性已知的入口
_ACCOUNT_PAGES = {
    "modelscope.cn": "https://modelscope.cn/my/settings/account",
    "siliconflow.cn": "https://cloud.siliconflow.cn/account/overview",
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
        """只替换 Key，模型、维度、后端地址一律沿用服务端配置。"""
        return self._settings.model_copy(
            update={
                "embedding_api_key": creds.embedding_key or self._settings.embedding_api_key,
                "llm_api_key": creds.llm_key or self._settings.llm_api_key,
            }
        )

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
