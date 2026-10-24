"""向量化适配器：只面向 OpenAI 兼容的 /embeddings 接口，不绑定任何厂商。

- 批量 + 指数退避重试（远程服务限流是常态，不是异常）；
- 结果按 (模型, 文本sha) 缓存进 SQLite，重复入库不重复计费；
- 维度由首次调用探测，写库校验，换模型后 reindex 时能被明确拒绝而不是静默错配。
"""

from __future__ import annotations

import hashlib
import sqlite3
from array import array

from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import Settings


class EmbeddingError(RuntimeError):
    pass


class Embedder:
    def __init__(self, settings: Settings, conn: sqlite3.Connection | None = None) -> None:
        if not settings.embedding_api_key:
            raise EmbeddingError("未配置 R4P_EMBEDDING_API_KEY")
        self._settings = settings
        self._client = OpenAI(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key,
            timeout=settings.embedding_timeout,
        )
        self._conn = conn
        self._dim: int = settings.embedding_dim
        self._model = settings.embedding_model

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    # ── 内部 ────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, max=20),
        reraise=True,
    )
    def _call(self, batch: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=batch)
        return [item.embedding for item in resp.data]

    def _cache_get(self, sha: str) -> list[float] | None:
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT dim, vec FROM embed_cache WHERE model = ? AND sha = ?", (self._model, sha)
        ).fetchone()
        if not row:
            return None
        vec = array("f")
        vec.frombytes(row["vec"])
        if self._dim and row["dim"] != self._dim:
            return None  # 模型维度变了，缓存作废
        self._dim = row["dim"]
        return vec.tolist()

    def _cache_put(self, sha: str, vec: list[float]) -> None:
        if self._conn is None:
            return
        arr = array("f", vec)
        self._conn.execute(
            "INSERT OR REPLACE INTO embed_cache (model, sha, dim, vec) VALUES (?, ?, ?, ?)",
            (self._model, sha, len(vec), arr.tobytes()),
        )

    # ── 对外 ────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        shas = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
        out: list[list[float] | None] = [self._cache_get(s) for s in shas]

        missing = [i for i, v in enumerate(out) if v is None]
        batch_size = max(1, self._settings.embedding_batch)
        for start in range(0, len(missing), batch_size):
            idx = missing[start : start + batch_size]
            vectors = self._call([texts[i] for i in idx])
            if len(vectors) != len(idx):
                raise EmbeddingError(
                    f"embedding 服务返回 {len(vectors)} 条向量，期望 {len(idx)} 条"
                )
            for i, vec in zip(idx, vectors, strict=True):
                out[i] = vec
                self._cache_put(shas[i], vec)
        if self._conn is not None:
            # 整批只提交一次：缓存是加速层，没必要每条向量都 fsync 一遍
            self._conn.commit()

        result = [v for v in out if v is not None]
        if result and not self._dim:
            self._dim = len(result[0])
        for vec in result:
            if len(vec) != self._dim:
                raise EmbeddingError(
                    f"向量维度不一致：期望 {self._dim}，实际 {len(vec)}。"
                    "换过 embedding 模型？请执行 `python -m rag4prefa reindex`。"
                )
        return result

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def probe(self, text: str = "凭据校验") -> int:
        """最小代价试一次 Key 是否可用，返回维度。

        刻意不走 `_call` 的退避重试：校验 Key 时错一次要等 1+2+4 秒才报错，
        用户会以为按钮卡住了。也不写 `embed_cache`——校验不该污染缓存。
        """
        resp = self._client.embeddings.create(model=self._model, input=[text])
        vectors = [item.embedding for item in resp.data]
        if not vectors:
            raise EmbeddingError("embedding 服务返回了空结果")
        dim = len(vectors[0])
        if self._dim and dim != self._dim:
            raise EmbeddingError(
                f"向量维度不一致：服务端按 {self._dim} 维建库，这把 Key 对应模型返回 {dim} 维。"
                "换模型必须全量 reindex，不能靠换 Key 绕过。"
            )
        self._dim = dim
        return dim

    def close(self) -> None:
        """释放底层连接。凭据池淘汰条目时调用。"""
        self._client.close()
