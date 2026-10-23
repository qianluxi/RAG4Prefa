"""领域模型：不含 IO、不含框架，被所有层共享。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SectionKind(StrEnum):
    chapter = "chapter"
    section = "section"
    subsection = "subsection"
    body = "body"
    table = "table"


class Section(BaseModel):
    """解析器输出的最小单元：一个标题 + 它名下累积的正文。"""

    title: str
    kind: SectionKind
    level: int = Field(ge=0, le=9)
    path: list[str] = Field(default_factory=list)  # 祖先标题，不含自身
    text: str = ""
    number: str | None = None

    @property
    def full_path(self) -> list[str]:
        return [*self.path, self.title]


class Chunk(BaseModel):
    id: str
    doc_id: str
    ordinal: int
    chapter_path: list[str]
    title: str
    text: str
    token_count: int
    content_sha: str


class Filter(BaseModel):
    """检索约束。随请求传入，不存在任何进程级状态。"""

    projects: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    keyword: str | None = None
    feature: str | None = None

    def merge(self, other: Filter) -> Filter:
        """请求显式传入的条件优先，未给出的用自动解析出的条件补齐。"""
        return Filter(
            projects=self.projects or other.projects,
            year_from=self.year_from if self.year_from is not None else other.year_from,
            year_to=self.year_to if self.year_to is not None else other.year_to,
            keyword=self.keyword or other.keyword,
            feature=self.feature or other.feature,
        )

    @property
    def is_empty(self) -> bool:
        return not any((self.projects, self.year_from, self.year_to, self.keyword, self.feature))


class Hit(BaseModel):
    chunk_id: str
    score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    project_name: str = ""
    project_code: str = ""
    filename: str = ""
    doc_year: int | None = None
    chapter_path: list[str] = Field(default_factory=list)
    title: str = ""
    text: str = ""


class ProjectInfo(BaseModel):
    name: str
    code: str = ""
    started_on: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    last_updated: str | None = None


class DocumentInfo(BaseModel):
    id: str
    project_name: str
    filename: str
    content_sha: str
    section_count: int
    chunk_count: int
    status: str
    uploaded_at: str
    error: str | None = None


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class Job(BaseModel):
    id: str
    kind: str
    state: JobState
    progress: float = 0.0
    message: str = ""
    payload: dict = Field(default_factory=dict)
    created_at: str = ""
    finished_at: str | None = None
