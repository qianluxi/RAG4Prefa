"""集中配置：唯一的配置入口，全部来自环境变量 / .env，代码里不再出现密钥与魔法值。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", env_prefix="R4P_", extra="ignore"
    )

    # 数据
    data_dir: Path = Path("data")

    # 向量化
    embedding_base_url: str = "https://api-inference.modelscope.cn/v1"
    embedding_api_key: str = ""
    embedding_model: str = "iic/nlp_gte_sentence-embedding_chinese-base"
    embedding_dim: int = 0  # 0 = 探测后写库
    embedding_batch: int = 32
    embedding_timeout: float = 60.0

    # 生成
    llm_base_url: str = "https://api-inference.modelscope.cn/v1"
    llm_api_key: str = ""
    llm_model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # 检索
    recall_k: int = 50
    top_k: int = 8
    rrf_k: int = 60
    chunk_max_tokens: int = 480
    chunk_min_tokens: int = 80

    # 重排（可选）
    rerank_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # 向量后端：auto 先试 Milvus，不可用则退到 numpy（会在 /health 与启动日志中说明）
    vector_backend: Literal["auto", "milvus", "numpy"] = "auto"
    milvus_uri: str = "data/vectors.db"
    milvus_collection: str = "chunks_vec"

    log_level: str = Field(default="INFO")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def missing_credentials(self) -> list[str]:
        """返回缺失的关键配置，供 /health 直接暴露，而不是启动时静默降级。"""
        missing: list[str] = []
        if not self.embedding_api_key:
            missing.append("R4P_EMBEDDING_API_KEY")
        if not self.llm_api_key:
            missing.append("R4P_LLM_API_KEY")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
