"""大模型适配器：OpenAI 兼容 + 真流式（token 级直通，不做切词伪装）。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

from openai import AsyncOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import Settings


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise LLMError("未配置 R4P_LLM_API_KEY")
        self._settings = settings
        self._async = AsyncOpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0
        )
        self._sync = OpenAI(
            base_url=settings.llm_base_url, api_key=settings.llm_api_key, timeout=120.0
        )

    def _thinking(self) -> dict:
        """思考开关：只在显式配置时才把字段发给服务端。

        不设默认值是有原因的——同一个模型名在两个平台上默认行为不同：魔搭的
        Qwen3.5-35B-A3B 默认不思考，硅基流动的同名模型默认思考。后者在判官这种
        要求严格两行输出的场景里会把 200 token 全花在思维链上（正文为空、单次
        105 秒），表现成"模型不听话"，实际是没关思考。反过来，给不认识这个字段
        的服务端乱发 body 字段又可能 400，所以 `None` 时一个字节都不多发。
        """
        enabled = self._settings.llm_enable_thinking
        if enabled is None:
            return {}
        return {"extra_body": {"enable_thinking": enabled}}

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        stream = await self._async.chat.completions.create(
            model=self._settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            stream=True,
            **self._thinking(),
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=15), reraise=True)
    def complete(self, system: str, user: str, *, max_tokens: int | None = None) -> str:
        resp = self._sync.chat.completions.create(
            model=self._settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self._settings.llm_temperature,
            max_tokens=max_tokens or self._settings.llm_max_tokens,
            **self._thinking(),
        )
        return resp.choices[0].message.content or ""

    def probe(self) -> str:
        """最小代价试一次 Key 是否可用。

        不走 `complete` 的重试：错 Key 要等三次退避才报错，校验按钮会像卡死。
        `max_tokens` 给得很小——这里只验「通不通」，不验模型好不好。
        """
        resp = self._sync.chat.completions.create(
            model=self._settings.llm_model,
            messages=[{"role": "user", "content": "只回复 OK"}],
            temperature=0,
            max_tokens=16,
            **self._thinking(),
        )
        return (resp.choices[0].message.content or "").strip()

    def close(self) -> None:
        """释放同步与异步客户端。凭据池淘汰条目时调用。"""
        try:
            self._sync.close()
        finally:
            self._close_async()

    def _close_async(self) -> None:
        # AsyncOpenAI.close() 是协程：在事件循环里只能排程，不在循环里就直接跑完。
        # 两种路径都要兜住，否则删掉一个用户的凭据时会漏掉连接。
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            with suppress(Exception):  # pragma: no cover - 关闭失败不该影响请求
                asyncio.run(self._async.close())
        else:
            loop.create_task(self._async.close())
