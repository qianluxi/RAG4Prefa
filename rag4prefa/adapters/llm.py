"""大模型适配器：OpenAI 兼容 + 真流式（token 级直通，不做切词伪装）。"""

from __future__ import annotations

from collections.abc import AsyncIterator

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

    async def stream(self, system: str, user: str) -> AsyncIterator[str]:
        stream = await self._async.chat.completions.create(
            model=self._settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self._settings.llm_temperature,
            max_tokens=self._settings.llm_max_tokens,
            stream=True,
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
        )
        return resp.choices[0].message.content or ""
