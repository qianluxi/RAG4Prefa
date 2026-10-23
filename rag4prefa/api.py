"""HTTP 层：只做协议转换与校验，业务全部在用例层。"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import admin, ask, consistency, ingest, repo
from .adapters.embed import EmbeddingError
from .adapters.llm import LLMError
from .consistency import IndexMismatch
from .domain import Filter, JobState
from .services import Services, build_services

WEB_DIR = Path(__file__).parent / "web"


# ── 请求模型 ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filter: Filter | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    rerank: bool = False


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filter: Filter | None = None
    top_k: int = Field(default=8, ge=1, le=50)


# ── SSE ────────────────────────────────────────────────────


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _chat_stream(services: Services, req: ChatRequest) -> AsyncIterator[str]:
    known = repo.project_names(services.conn)
    effective = ask.build_filter(req.question, req.filter, known)
    try:
        async for event, data in ask.stream_answer(
            services.conn,
            services.store,
            services.require_embedder(),
            services.require_llm(),
            services.reranker,
            services.settings,
            question=req.question,
            filt=effective,
            top_k=req.top_k,
            use_rerank=req.rerank,
        ):
            yield _sse(event, data)
    except (EmbeddingError, LLMError, IndexMismatch) as exc:
        yield _sse("error", {"message": str(exc)})
    except Exception as exc:  # 流已开始，不能再改 HTTP 状态码，用 error 事件收尾
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})


# ── 应用 ───────────────────────────────────────────────────


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        yield
        app.state.services.close()

    app = FastAPI(title="RAG4Prefa", version="0.1.0", lifespan=lifespan)

    def svc(request: Request) -> Services:
        return request.app.state.services

    @app.get("/api/health")
    def health(request: Request) -> dict:
        return svc(request).health()

    # ── 入库 ────────────────────────────────────────────

    @app.post("/api/projects/{project_name}/documents")
    async def upload_documents(
        request: Request,
        project_name: str,
        files: list[UploadFile],
        code: str = Form(default=""),
        started_on: str = Form(default=""),
        year: int | None = Form(default=None),
    ) -> JSONResponse:
        services = svc(request)
        target = services.settings.upload_dir / _safe_name(project_name)
        target.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for upload in files:
            if not upload.filename or not upload.filename.lower().endswith(".docx"):
                continue
            path = target / _safe_name(upload.filename)
            path.write_bytes(await upload.read())
            saved.append(path)
        if not saved:
            raise HTTPException(400, "没有可处理的 .docx 文件")

        job_id = uuid.uuid4().hex[:12]
        repo.upsert_project(services.conn, project_name, code, started_on or None)
        repo.create_job(
            services.conn,
            job_id,
            "ingest",
            {"project": project_name, "files": [p.name for p in saved]},
        )
        doc_year = year or (int(started_on[:4]) if started_on[:4].isdigit() else None)

        async def run() -> None:
            try:
                await asyncio.to_thread(
                    ingest.ingest_batch,
                    services.conn,
                    services.store,
                    services.require_embedder(),
                    services.settings,
                    job_id=job_id,
                    files=saved,
                    project_name=project_name,
                    project_code=code,
                    doc_year=doc_year,
                )
                services.maybe_promote()
            except Exception as exc:
                repo.update_job(
                    services.conn,
                    job_id,
                    state=JobState.failed,
                    message=f"{type(exc).__name__}: {exc}",
                )

        asyncio.create_task(run())
        return JSONResponse({"job_id": job_id, "files": [p.name for p in saved]}, status_code=202)

    @app.get("/api/jobs/{job_id}")
    def get_job(request: Request, job_id: str) -> dict:
        job = repo.get_job(svc(request).conn, job_id)
        if not job:
            raise HTTPException(404, "任务不存在")
        return job.model_dump(mode="json")

    # ── 项目 ────────────────────────────────────────────

    @app.get("/api/projects")
    def list_projects(request: Request) -> list[dict]:
        return [p.model_dump() for p in admin.list_projects(svc(request).conn)]

    @app.get("/api/projects/{project_name}")
    def project_detail(request: Request, project_name: str) -> dict:
        services = svc(request)
        info = repo.get_project(services.conn, project_name)
        if not info:
            raise HTTPException(404, "项目不存在")
        summary = ""
        if services.llm:
            try:
                summary = admin.summarize_project(services.conn, services.llm, project_name)
            except Exception:
                summary = ""
        return {
            **info.model_dump(),
            "summary": summary,
            "documents": [
                d.model_dump() for d in admin.list_documents(services.conn, project_name)
            ],
            "chapters": admin.chapter_distribution(services.conn, project_name),
        }

    @app.delete("/api/projects/{project_name}")
    def delete_project(request: Request, project_name: str) -> dict:
        result = admin.delete_project(svc(request).conn, svc(request).store, project_name)
        if not result["deleted"]:
            raise HTTPException(404, result["reason"])
        return result

    # ── 检索与问答 ───────────────────────────────────────

    @app.post("/api/search")
    async def search(request: Request, req: SearchRequest) -> dict:
        services = svc(request)
        effective = ask.build_filter(req.question, req.filter, repo.project_names(services.conn))
        try:
            hits, applied, timings = await ask.search(
                services.conn,
                services.store,
                services.require_embedder(),
                services.settings,
                question=req.question,
                filt=effective,
                top_k=req.top_k,
            )
        except (EmbeddingError, LLMError, IndexMismatch) as exc:
            raise HTTPException(503, str(exc)) from exc
        return {
            "filter": applied.model_dump(exclude_none=True),
            "timings": timings,
            "results": [h.model_dump() for h in hits],
        }

    @app.post("/api/chat")
    async def chat(request: Request, req: ChatRequest) -> StreamingResponse:
        services = svc(request)
        try:
            services.require_embedder()
            services.require_llm()
        except (EmbeddingError, LLMError) as exc:
            raise HTTPException(503, str(exc)) from exc
        # 模型不一致/索引与事实不符的检查放在开始流之前：能改 HTTP 状态码的时候说清楚，
        # 别用 SSE error 收尾（客户端已经开始读 token 流了）
        try:
            consistency.ensure_queryable(
                services.conn,
                services.require_embedder().model,
                services.store.dim,
                store_count=services.store.count(),
            )
        except IndexMismatch as exc:
            raise HTTPException(503, str(exc)) from exc
        return StreamingResponse(
            _chat_stream(services, req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── 前端 ────────────────────────────────────────────

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    return app


def _safe_name(name: str) -> str:
    """防目录穿越：只取最后一段，并剔除非法字符。"""
    base = Path(name.replace("\\", "/")).name
    base = "".join(ch for ch in base if ch not in '<>:"|?*\x00')
    if not base or base in {".", ".."}:
        raise HTTPException(400, f"非法名称：{name!r}")
    return base


app = create_app()
