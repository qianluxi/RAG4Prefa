"""重排：可选、可关闭、可替换。默认 Noop，保证"没有重排服务也能跑"。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from ..config import Settings
from ..domain import Hit


class Reranker(Protocol):
    async def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]: ...


class NoopReranker:
    async def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        return list(hits[:top_k])


class RemoteReranker:
    """兼容 SiliconFlow / Jina 形态的 /rerank 接口：
    POST {model, query, documents, top_n} → {results: [{index, relevance_score}]}
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.rerank_url
        self._key = settings.rerank_api_key
        self._model = settings.rerank_model

    async def rerank(self, query: str, hits: Sequence[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": [h.text for h in hits],
            "top_n": min(top_k, len(hits)),
        }
        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(self._url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            # 重排是增强项，失败就退回原顺序——但不能静默丢失信息，调用方会记录警告
            raise

        ranked: list[Hit] = []
        for item in data.get("results", [])[:top_k]:
            idx = int(item["index"])
            if 0 <= idx < len(hits):
                ranked.append(
                    hits[idx].model_copy(update={"rerank_score": item["relevance_score"]})
                )
        return ranked or list(hits[:top_k])


def build_reranker(settings: Settings) -> Reranker:
    return RemoteReranker(settings) if settings.rerank_url else NoopReranker()
