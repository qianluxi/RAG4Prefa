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
    # 起始物理页码（1-based）。只有 PDF 有稳定的物理页码；docx 的分页是渲染期
    # 动态决定的，文件里根本不存在页码，所以 docx 恒为 None，定位靠 chapter_path。
    # 这是**展示字段**，不参与 chunk_id（内容寻址），排版微调导致的页码变化
    # 不会让同一段正文换 id。
    page: int | None = None
    # 正文**每一行**各自的物理页码，与 `text.split("\n")` 一一对应；docx 恒为 None。
    #
    # 为什么不能只用 `page`（2026-10-27 复审 P1-2）：那是**标题**所在页。一节常常跨
    # 好几页（"4.2 悬挑构造"从第 4 页开始、正文延续到第 8 页），拿它给这一节的所有
    # 片段做页码，用户在引用卡片上看到"第 4 页"，翻过去却找不到那段话——工程文档
    # 知识库里这等于引用不可核。分块时按行的页码定位，片段就能报自己真正的起始页。
    line_pages: list[int | None] | None = None

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
    # 见 Section.page / line_pages：**片段自己**的起始物理页码（1-based），
    # 不是所在章节标题页。展示用，不参与寻址。docx 为 None。
    page: int | None = None


class Filter(BaseModel):
    """检索约束。随请求传入，不存在任何进程级状态。

    字段只保留**真的参与召回**的维度：projects / year_from / year_to 进标量过滤，
    keyword 并入 BM25 查询式。曾经有一个 feature 字段能从问句里解析出来却从不参与过滤，
    这种"看着生效其实无效"的字段比没有更糟，已删除。
    """

    projects: list[str] | None = None
    year_from: int | None = None
    year_to: int | None = None
    keyword: str | None = None

    def merge(self, other: Filter) -> Filter:
        """请求显式传入的条件优先，未给出的用自动解析出的条件补齐。"""
        return Filter(
            projects=self.projects or other.projects,
            year_from=self.year_from if self.year_from is not None else other.year_from,
            year_to=self.year_to if self.year_to is not None else other.year_to,
            keyword=self.keyword or other.keyword,
        )

    @property
    def is_empty(self) -> bool:
        return not any((self.projects, self.year_from, self.year_to, self.keyword))


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
    # 起始物理页码（1-based），docx 为 None。展示用，不参与召回与寻址。
    page: int | None = None


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
    # 年份是**过滤条件**（documents / vectors 归档 / 后端标量列三处都要一致），
    # 入库时要拿它和已有记录比对，所以它必须出现在这个视图里。
    doc_year: int | None = None
    section_count: int
    chunk_count: int
    status: str
    uploaded_at: str
    error: str | None = None
    # 视觉模型认出过的页码（1-based）。`None` = 未记录（加这一列之前入库的文档），
    # `[]` = 明确没有经过识别。**不能合并成"空"**：前者是"不知道"，后者是"没有"，
    # 而正文漏字这件事只有后者才敢说"这句是原文"。
    ocr_pages: list[int] | None = None


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
