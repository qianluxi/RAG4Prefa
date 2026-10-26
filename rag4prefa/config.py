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

    # 向量化。默认硅基流动国际站 + Qwen3-Embedding-0.6B：1024 维，与下面的 1024 对得上。
    #
    # 硅基流动的国内站（siliconflow.cn）与国际站（siliconflow.com）是**两套独立账号**，
    # Key 不通用，模型目录也不同：国内站有 BAAI/bge-m3，国际站只有 Qwen3-Embedding 系列。
    # 在 .com 申请的 Key 填给 .cn 的地址只会在第一次调用时收到 "Token is invalid"，
    # 看起来像 Key 被删——这是本项目最容易搞错的一处，两站对应关系写在 .env.example。
    embedding_base_url: str = "https://api.siliconflow.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 0  # 0 = 探测后写库
    embedding_batch: int = 32  # 百炼 text-embedding-v3/v4 只能 10
    embedding_timeout: float = 60.0

    # 生成
    llm_base_url: str = "https://api-inference.modelscope.cn/v1"
    llm_api_key: str = ""
    # 魔搭的模型目录会下线旧 ID，写死一个名字迟早会收到
    # "Model id : ... has no provider supported"。拿不准时先列一遍当前可调用的：
    #   curl -s https://api-inference.modelscope.cn/v1/models
    # 这个列表不需要鉴权，直接能看到该平台当前供哪些模型。
    llm_model: str = "Qwen/Qwen3.5-35B-A3B"
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

    # 未携带用户 Key 的请求，是否允许回落到 `.env` 里那份**服务端公共 Key**。
    #
    # 单机演示必须开着：CLI、curl、旧客户端都不带 `X-R4P-*` 头，关掉就什么都调不通。
    # 但公网部署必须关掉——那份 Key 的额度是"任何能访问端口的人"共用的，匿名请求
    # 可以无限消耗它，而账单只有一个人付。关掉之后，未带 Key 的请求会收到 401，
    # 并被告知去哪里填自己的 Key，而不是先烧掉几百次调用再报额度不足。
    allow_server_key_fallback: bool = True

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
