"""HTTP 层：只做协议转换与校验，业务全部在用例层。"""

from __future__ import annotations

import asyncio
import hmac
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import admin, ask, concurrency, consistency, ingest, repo
from .adapters.embed import Embedder, EmbeddingError
from .adapters.llm import LLM, LLMError
from .adapters.ocr import VlmOcr
from .adapters.parsers import SUPPORTED_SUFFIXES, explain_unsupported
from .adapters.vectors import BackendUnavailable, StoreRef
from .concurrency import MaintenanceBusy
from .consistency import IndexMismatch
from .credentials import (
    HEADER_ADMIN_TOKEN,
    KEY_ROLES,
    LLM_CHOICES,
    account_blocked_hint,
    auth_failure_hint,
    from_headers,
    missing_roles,
    platform_mismatch,
    resolve_llm_model,
    settings_for_ocr,
)
from .domain import Filter, JobState
from .ocr_setup import effective
from .services import Services, build_services

WEB_DIR = Path(__file__).parent / "web"


# ── 请求模型 ────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filter: Filter | None = None
    top_k: int = Field(default=8, ge=1, le=50)
    # None = 跟随服务端配置（配了 R4P_RERANK_URL 就重排）；显式 false 仍然能关掉。
    rerank: bool | None = None


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    filter: Filter | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class CredentialCheck(BaseModel):
    """校验用户填的 Key。空字段表示"用服务端公共 Key"，便于一键自检。"""

    username: str = Field(default="", max_length=64)
    embedding_key: str = Field(default="", max_length=4096)
    llm_key: str = Field(default="", max_length=4096)
    # 只影响这一项的探测目标；embedding 模型不接受外部指定（见 credentials 模块头）
    llm_model: str = Field(default="", max_length=128)
    # 扫描件识别是第三个独立环节：Key 与上面两把不通用，档位决定探测哪个端点
    ocr_key: str = Field(default="", max_length=4096)


CLEAR_CONFIRM = "清空"


class ClearRequest(BaseModel):
    """清空数据的确认词。

    前端弹窗保护不了 curl，也保护不了误点的脚本：真正的闸门在服务端，要求调用方打出
    与 `CLEAR_CONFIRM` 完全一致的词。这个词刻意选成中文——它会出现在日志与命令历史里，
    一眼能认出"这是那次删除"。
    """

    confirm: str = Field(default="", max_length=16)



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
    except (EmbeddingError, LLMError, IndexMismatch, BackendUnavailable) as exc:
        yield _sse("error", {"message": str(exc)})
    except Exception as exc:  # 流已开始，不能再改 HTTP 状态码，用 error 事件收尾
        yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"})


# ── 应用 ───────────────────────────────────────────────────


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        # 占住"服务进程租约"：它只是一把活着的证明，不保护数据。别处的 CLI 靠它区分
        # "上一个进程死了、该清理遗留任务"和"这个进程正在写、别动它的任务"。
        # 抢不到说明同一个库上已经有另一个服务进程——那种部署本身就不对（整进程一条
        # 写连接），如实说出来，但不要因此起不来。
        root_services = app.state.services
        if not concurrency.WORKER_LEASE.acquire(root_services.conn, timeout=0.0):
            root_services.notes.append(
                "同一个数据库上已经有另一个服务进程在运行；本进程不会清理未完成任务，"
                "两个进程同时写同一个库也不被支持"
            )
        yield
        concurrency.WORKER_LEASE.release()
        app.state.services.close()

    app = FastAPI(title="RAG4Prefa", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(MaintenanceBusy)
    def _maintenance_busy(_: Request, exc: MaintenanceBusy) -> JSONResponse:
        """维护动作互斥：抢不到锁不是服务器坏了，是"现在不行"。

        用 409 而不是 500：前端要能区分"稍后重试"和"真的出错了"，否则用户会去
        翻日志找一个根本不存在的 bug。
        """
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(IndexMismatch)
    def _index_mismatch(_: Request, exc: IndexMismatch) -> JSONResponse:
        """索引不能自证一致时也一样：这是"现在不能动"，不是服务器故障。

        用 409 而不是 500 的理由和上一条相同——前端据此提示"先处理索引"，
        而不是让用户去翻日志找一个不存在的 bug。删除项目这类没有单独 try 的端点，
        靠这条兜住（例如"活动后端已被别的进程换过"）。
        """
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(BackendUnavailable)
    def _backend_unavailable(_: Request, exc: BackendUnavailable) -> JSONResponse:
        """后端还没就绪（没打开、被别的维护动作占着）：503，不是 500。

        为什么单独一条（2026-10-24 第十七轮复审）：这类失败以前会从 `store.count()`
        或 `store.search()` 直接冒出来，变成一个 500——运维看到"服务器内部错误"，
        会去查代码而不是去查那个占着维护锁的进程。它和"现在不能动"是同一类事：
        等一等、重启服务就好。
        """
        return JSONResponse({"detail": str(exc)}, status_code=503)

    def svc(request: Request) -> Services:
        """每个请求都按请求头换一次凭据。

        没带 `X-R4P-*` 头时 `for_credentials` 返回同一个实例，等于沿用 `.env` 里的
        公共 Key——所以旧客户端、curl 和 CLI 都不受影响。
        """
        return request.app.state.services.for_credentials(from_headers(request.headers))

    def svc_billable(request: Request, *, need: tuple[str, ...]) -> Services:
        """会花掉 Key 额度的入口（入库 / 检索 / 问答）用它，而不是 `svc()`。

        `R4P_ALLOW_SERVER_KEY_FALLBACK=false` 时，**用户没提供这个接口真正要用的
        那把 Key 就直接拒绝**。这一条是给公网部署用的：`.env` 里那份公共 Key 的额度
        对"任何能访问端口的人"开放，匿名请求可以无限消耗它，而账单只有一个人付。
        单机演示保持默认 true，CLI 与 curl 的行为完全不变。

        为什么按 `need` 逐个判、而不是判"凭据为空"（2026-10-27 复审 P1-1）：
        `from_headers()` 只要带任意一个头就返回非空凭据（只传模型选择头也算），
        而 `settings_for()` 会把没提供的 Key 静默补成服务端那两把。于是
        「只填 embedding Key 调 /api/chat」「只填 LLM Key 调 /api/search」
        「只传模型头」全都能绕过配置——配置写着不代付，账单照付。

        为什么要在这里拦而不是让厂商报 401：那时候额度已经烧掉了一部分，而且用户看到
        的是厂商的 "Token is invalid"，会去反复核对那把**他根本没填过**的 Key。
        """
        creds = from_headers(request.headers)
        root = request.app.state.services
        if not root.settings.allow_server_key_fallback:
            missing = missing_roles(creds, need)
            if missing:
                raise HTTPException(401, _missing_key_message(missing))
        return root.for_credentials(creds)

    def require_admin(request: Request) -> None:
        """清库 / 删项目 / 删文档这三条不可撤销的入口，必须先过这一关。

        在这之前它们**没有任何身份校验**：`confirm: "清空"` 只防手滑，不证明"这个人
        有权删"。额度那一层早就收口了（`R4P_ALLOW_SERVER_KEY_FALLBACK=false` 之后匿名
        请求用不了服务端 Key），"谁能删库"这一层一直是空的——谁能访问端口谁就能删。

        三条规则见 `config.admin_token`：配了口令就校验；没配、又是各付各的部署，
        **直接 503**（宁可删不了，也不能让访客删掉全库）；没配、且单机演示则放行，
        那时打开页面的人就是主人。
        """
        settings = request.app.state.services.settings
        expected = settings.admin_token
        if not expected:
            if settings.allow_server_key_fallback:
                return
            raise HTTPException(
                503,
                "这个部署没有配置管理口令，清空库 / 删项目 / 删文件已停用。"
                "在 .env 里设 R4P_ADMIN_TOKEN=<口令> 并重启，"
                "然后到页面「API 设置」里填同一个口令。"
                "（各付各的部署不接受匿名删除：这几条不可撤销。）",
            )
        got = request.headers.get(HEADER_ADMIN_TOKEN, "")
        # 定长比较：口令是共享秘密，逐字符比较会给"猜口令"留下时间侧信道
        if not got or not hmac.compare_digest(got, expected):
            raise HTTPException(
                403,
                "管理口令不对：在「API 设置」里填与 R4P_ADMIN_TOKEN 相同的口令再试。",
            )

    @app.get("/api/health")
    def health(request: Request) -> dict:
        return svc(request).health()

    # ── 凭据 ────────────────────────────────────────────

    @app.get("/api/llm-models")
    def llm_models(request: Request) -> dict:
        """可选的对话模型清单。前端用它渲染下拉。

        只返回白名单里、且逐个真调通过的模型。别拿 `/v1/models` 直接渲染：
        那个目录里 DeepSeek V4 全系、MiniMax-M3 等一大半调不通，列出来就是
        一排点不动的按钮，用户点一次失败一次。
        """
        current = svc(request).settings.llm_model
        options = [
            {
                "id": option.id,
                "label": option.label,
                "note": option.note,
                "selected": option.id == current,
            }
            for option in LLM_CHOICES
        ]
        if current and resolve_llm_model(current) != current:
            # .env 里配了清单外的模型（自建网关、测试假服务）：它必须出现在下拉里，
            # 否则前端一项都选不中，用户会以为配置坏了，而实际是「服务端默认」。
            options.insert(
                0,
                {
                    "id": current,
                    "label": f"{current}（服务端配置）",
                    "note": "来自 .env 的 R4P_LLM_MODEL，不在内置清单内；选它等于用服务端默认。",
                    "selected": True,
                },
            )
        return {
            "current": current,
            "options": options,
            # 把"为什么只有它不能选"一并给前端，免得用户找不到切换入口以为漏了
            "embedding_fixed": True,
        }

    @app.post("/api/credentials/verify")
    def verify_credentials(request: Request, payload: CredentialCheck) -> dict:
        """用给定的 Key 各试调一次，把"哪一项不通、为什么"说清楚。

        返回 200 而不是 4xx：部分成功是常态（embedding 通了、LLM 没通），
        用状态码表达会把另一项的结果一起丢掉。真正的失败写进各自字段。

        这里刻意不写缓存、不建索引，只做一次最小调用：
        `probe()` 不走退避重试，否则错 Key 要等十几秒，前端会以为按钮卡死。
        """
        # 注意这里取的是**根服务**的配置，不是 `svc(request)` 那个按用户视图：
        # 留空字段要回落到服务端公共 Key，若回落到"当前登录用户"的 Key，就会验出
        # 一次假的通过——A 已登录、给 B 只填 embedding、LLM 栏留空，验的其实是 A 的
        # Key，却被标成 source=server；B 保存后真正发请求走的是服务端 Key，直接失败。
        services = request.app.state.services
        base = services.settings
        # 关掉回落时，留空必须解释成"这一项我没配"，而不是"替我试一下服务端那把"。
        # 否则这个接口本身就是一条免费探测通道：不填任何 Key 反复点"验证"，
        # 每一次都真的打到厂商、花服务端的钱（2026-10-27 复审 P1-1）。
        strict_keys = not base.allow_server_key_fallback
        embedding_key = payload.embedding_key.strip() or (
            "" if strict_keys else base.embedding_api_key
        )
        llm_key = payload.llm_key.strip() or ("" if strict_keys else base.llm_api_key)
        # 下拉里选了什么就验什么；不在白名单里就回落服务端默认，并如实回显用了哪个
        llm_model = resolve_llm_model(payload.llm_model, base.llm_model)
        local = base.model_copy(
            update={
                "embedding_api_key": embedding_key,
                "llm_api_key": llm_key,
                "llm_model": llm_model,
            }
        )
        secrets = (
            payload.embedding_key,
            payload.llm_key,
            payload.ocr_key,
            base.embedding_api_key,
            base.llm_api_key,
        )

        no_key_error = (
            "这个部署不接受服务端代付额度，请填入你自己的 Key"
            if strict_keys
            else "未配置 Key"
        )
        # 留空的那一项来源必须如实写：`user` 是用户自己的 Key，`server` 是服务端
        # 公共 Key 代付，`missing` 是"这个部署不代付、你也没填"。前端据 source
        # 决定要不要把这一项标成"未配置"，混成一个值会让用户以为验过自己的 Key。
        emb_source = (
            "user"
            if payload.embedding_key.strip()
            else ("missing" if strict_keys else "server")
        )
        llm_source = (
            "user" if payload.llm_key.strip() else ("missing" if strict_keys else "server")
        )
        ocr_key = payload.ocr_key.strip() or (
            "" if strict_keys else (base.ocr_api_key or "")
        )
        ocr_source = (
            "user"
            if payload.ocr_key.strip()
            else ("missing" if (strict_keys or not ocr_key) else "server")
        )
        result: dict = {
            # 前端要拿它提示"你的 Key 必须属于同一家服务商"——embedding 模型由服务端统一决定
            "server": {
                "embedding_base_url": local.embedding_base_url,
                "embedding_model": local.embedding_model,
                "embedding_dim": local.embedding_dim,
                "llm_base_url": local.llm_base_url,
                "llm_model": local.llm_model,
            },
            "user": payload.username.strip() or "anonymous",
            "embedding": {"ok": False, "source": emb_source, "error": no_key_error},
            "llm": {"ok": False, "source": llm_source, "error": no_key_error},
            # OCR 是第三项，默认「未配置」而不是「通过」：没验过的项不能报 ok
            "ocr": {
                "ok": False,
                "source": ocr_source,
                "model": effective(base).model or None,
                "error": no_key_error,
            },
        }

        # 先做本地预检：明显填错平台的 Key 不必真发请求，也免得把厂商那句
        # "Token is invalid" 原样转给用户——那句话会把人引去检查令牌有没有复制错。
        emb_hint = platform_mismatch(
            payload.embedding_key.strip(), local.embedding_base_url, label="embedding"
        )
        if emb_hint:
            result["embedding"] = {"ok": False, "source": emb_source, "error": emb_hint}
        elif embedding_key:
            embedder: Embedder | None = None
            try:
                embedder = Embedder(local, services.conn)
                dim = embedder.probe()
                result["embedding"] = {
                    "ok": True,
                    "model": local.embedding_model,
                    "dim": dim,
                    # 留空会回落到服务端 Key，必须标明来源，否则用户会以为验的是自己的
                    "source": emb_source,
                }
            except Exception as exc:
                message = _redact(f"{type(exc).__name__}: {exc}", *secrets)
                result["embedding"] = {
                    "ok": False,
                    "source": emb_source,
                    "error": _explain_error(
                        message, local.embedding_base_url, payload.embedding_key.strip()
                    ),
                }
            finally:
                if embedder is not None:
                    embedder.close()

        llm_hint = platform_mismatch(
            payload.llm_key.strip(), local.llm_base_url, label="LLM"
        )
        if llm_hint:
            result["llm"] = {"ok": False, "source": llm_source, "error": llm_hint}
        elif llm_key:
            llm: LLM | None = None
            try:
                llm = LLM(local)
                reply = llm.probe()
                result["llm"] = {
                    "ok": True,
                    "model": local.llm_model,
                    "source": llm_source,
                    # 有些模型会把 16 个 token 全花在思考上、正文为空——连接是通的，
                    # 所以只记长度，不把空回复当失败
                    "reply_chars": len(reply),
                }
            except Exception as exc:
                message = _redact(f"{type(exc).__name__}: {exc}", *secrets)
                result["llm"] = {
                    "ok": False,
                    "source": llm_source,
                    # 模型名不在该平台的当前目录里时，鉴权是通的、报的是 400，
                    # 用户的直觉却是"我的 token 有问题"，所以把排查方向写清楚
                    "error": _explain_error(
                        message, local.llm_base_url, payload.llm_key.strip()
                    ),
                }
            finally:
                if llm is not None:
                    llm.close()

        # ── OCR ──
        # 这一项验的是「这把 Key 能不能真发出去一次请求」。与上面两项不同的是：
        # 只有真送一张图才知道通不通，所以用一张 64×64 白图探一次。
        if ocr_key:
            probe_settings = settings_for_ocr(base, ocr_key)
            ocr_eff = effective(probe_settings)
            # Key 的形态与端点对不上时先拦一次：零成本，还能把两个域名直接摆出来。
            ocr_hint = platform_mismatch(ocr_key, ocr_eff.base_url, label="OCR")
            if ocr_hint:
                result["ocr"] = {
                    "ok": False,
                    "source": ocr_source,
                    "model": ocr_eff.model,
                    "error": ocr_hint,
                }
            else:
                provider: VlmOcr | None = None
                try:
                    provider = VlmOcr(probe_settings)
                    reply = provider.probe()
                    result["ocr"] = {
                        "ok": True,
                        "source": ocr_source,
                        "model": ocr_eff.model,
                        "base_url": ocr_eff.base_url,
                        # 探测图是纯白的，识别结果本应为空；能拿到响应就说明链路通
                        "probe_chars": len(reply),
                    }
                except Exception as exc:  # noqa: BLE001 - 单项失败不影响另外两项
                    message = _redact(f"{type(exc).__name__}: {exc}", *secrets)
                    # 400 是**正常**的：视觉模型有权拒绝一张没有内容的探测图。
                    # 这与 401 完全不同——Key 已经过了鉴权。不说这一条，用户会把
                    # 「探测图被拒」读成「我的 Key 不对」，然后反复重建一把没错的 Key。
                    if "400" in message and "401" not in message:
                        message += (
                            "\n（Key 已通过鉴权：模型拒绝的是这张 64×64 的探测图，不是你的 Key。"
                            "真正的识别要送整页图片，建议直接拿一份扫描件试。）"
                        )
                    result["ocr"] = {
                        "ok": False,
                        "source": ocr_source,
                        "model": ocr_eff.model,
                        "error": _explain_error(message, ocr_eff.base_url, ocr_key),
                    }
                finally:
                    if provider is not None:
                        provider.close()

        return result

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
        services = svc_billable(request, need=("embedding",))
        # 没有可用 key 时在"创建项目/落盘"之前就拒绝，别留一个空项目让人以为导过了
        try:
            services.require_embedder()
        except EmbeddingError as exc:
            raise HTTPException(503, str(exc)) from exc
        # 后端没确认时不许收：普通入库会同时改归档与向量索引，而"库里从没登记过活动
        # 后端"意味着没有任何一步证明过索引装的是这一版。落盘（upload_dir 是持久归档区）
        # 与建任务都在这一道之后——被拒绝时应当一个字节都没动（2026-10-24 第十七轮复审）。
        try:
            services.assert_write_ready()
        except (IndexMismatch, BackendUnavailable) as exc:
            raise HTTPException(503, str(exc)) from exc
        # 重建/另一个入库正在跑时立刻拒绝，而不是收下文件后让任务在后台失败：
        # 用户至少知道"该等一会儿"，而不是看到一条迟到的 failed。
        busy = concurrency.is_busy(services.conn)
        if busy:
            raise HTTPException(
                503, f"正在执行其他维护动作（{busy}），暂时不能入库，请稍后重试。"
            )
        target = services.settings.upload_dir / _safe_name(project_name)
        target.mkdir(parents=True, exist_ok=True)

        # 文件先落盘、再抢锁。落盘即**持久**：`upload_dir` 是原始文档归档区，
        # 不是临时暂存区——重试、重导入、事后审计都要用到这份原文。所以被维护动作
        # 挡下时文件留在原地是设计意图，不做"失败即删"；同名文件下次上传直接覆盖，
        # 不会越积越多。代价是这里比锁早了一步：`is_busy()` 只是预检查（上面），
        # 真正的准入仍是 `ingest_batch` 里的维护锁，被拒时任务消息会说明文件已保存。
        saved: list[Path] = []
        skipped: list[dict] = []
        for upload in files:
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower()
            # 不支持的格式不能静默丢掉：用户选了 9 个文件、页面上只多出 6 份，
            # 剩下 3 份在任何地方都不出现，看起来就是"系统吃掉了"。
            if suffix not in SUPPORTED_SUFFIXES:
                skipped.append(
                    {"filename": upload.filename, "reason": explain_unsupported(suffix)}
                )
                continue
            path = target / _safe_name(upload.filename)
            path.write_bytes(await upload.read())
            saved.append(path)
        if not saved:
            detail = "；".join(f"{s['filename']}：{s['reason']}" for s in skipped)
            raise HTTPException(
                400,
                f"没有可处理的文件（只接受 {' / '.join(SUPPORTED_SUFFIXES)}）。"
                + (f"你选的这些都不行——{detail}" if detail else ""),
            )

        job_id = uuid.uuid4().hex[:12]
        # 项目刻意不在这里建：它一旦落库，用户在列表里就会看到"已经导入过"。
        # 真正的准入是 `ingest_batch` 里的维护锁，而它在项目的写入之前——
        # 见下面 `run()`。
        repo.create_job(
            services.conn,
            job_id,
            "ingest",
            {
                "project": project_name,
                "files": [p.name for p in saved],
                "skipped": skipped,
            },
        )
        doc_year = year or (int(started_on[:4]) if started_on[:4].isdigit() else None)

        async def run() -> None:
            # 后端指针取根服务**此刻**的值：本视图是请求进来到的瞬间拍的快照，
            # 期间根服务可能已经迁到 Milvus。继续往视图里的旧 numpy 写，就成了
            # "归档在 A、根服务查 B"。Key 仍然按这个请求的凭据走。
            root = services.owner
            # 用 StoreRef 而不是把 `root.store` 传下去：批次可能要跑几分钟，
            # 期间后端被迁移换掉时，那个"取参数时定死"的对象就过期了。
            sink = StoreRef(lambda: root.store)
            try:
                await asyncio.to_thread(
                    ingest.ingest_batch,
                    root.conn,
                    sink,
                    services.require_embedder(),
                    services.settings,
                    job_id=job_id,
                    files=saved,
                    project_name=project_name,
                    project_code=code,
                    started_on=started_on or None,
                    doc_year=doc_year,
                )
            except MaintenanceBusy as exc:
                # 批次级准入被拒时，项目、文档、任务状态都还没被动过（见 ingest_batch）。
                # 这里只把"没执行"如实写进任务，不让它显示成"入库失败"。
                # 原文已经落盘（upload_dir 是归档区），所以要告诉用户"不用重新选文件"，
                # 否则他会在维护结束后再选一次，把同一份文件重新上传一遍。
                repo.update_job(
                    root.conn,
                    job_id,
                    state=JobState.failed,
                    message=f"没有执行：{exc}（原始文件已保存在服务端，稍后重新提交即可，"
                    "无需重新选择文件）",
                )
                return
            except Exception as exc:
                repo.update_job(
                    root.conn,
                    job_id,
                    state=JobState.failed,
                    message=f"{type(exc).__name__}: {exc}",
                )
                return
            # 迁移只能由根服务执行（`maybe_promote` 内部会转交），
            # 在视图上迁移等于"迁完了根服务还不知道"。
            root.maybe_promote()

        asyncio.create_task(run())
        return JSONResponse(
            {
                "job_id": job_id,
                "files": [p.name for p in saved],
                # 逐个点名"为什么不收"，而不是让它们从结果里安静消失
                "skipped": skipped,
            },
            status_code=202,
        )

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
        # 摘要会真调一次大模型。这一页的主体是项目信息，不值得为"没填 Key"整页 401；
        # 但也不能拿服务端公共额度去替匿名请求生成摘要——那是同一个漏洞的侧门。
        # 判据用 `llm_billable` 而不是 `user_label`：用户名对任何请求都非空（只传模型头
        # 也会解析成 anonymous），拿它当"已登录"用等于这道侧门根本没关（2026-10-27 复审）。
        may_generate = services.llm_billable
        if services.llm and may_generate:
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
        require_admin(request)
        services = svc(request)
        # 删除同时改事实与索引，准入与入库同源：后端没确认时不动任何数据。
        # （用例层还有一道"库里是否登记过活动后端"，两者都要在。）
        try:
            services.assert_write_ready()
        except (IndexMismatch, BackendUnavailable) as exc:
            raise HTTPException(503, str(exc)) from exc
        result = admin.delete_project(services.conn, services.store, project_name)
        if not result["deleted"]:
            raise HTTPException(404, result["reason"])
        return result

    @app.delete("/api/projects/{project_name}/documents/{doc_id}")
    def delete_document(request: Request, project_name: str, doc_id: str) -> dict:
        """删单个文件，连同它的片段、全文索引与向量。

        路径里带项目名不是为了定位（`doc_id` 已经唯一），而是为了让"从我看到的这个项目里
        删掉这一份"变成一次可校验的意图：`doc_id` 属于别的项目时直接 404，而不是照着 id
        把另一边的数据删掉——那是最难在事后解释的一类事故。
        """
        require_admin(request)
        services = svc(request)
        # 与删除项目同源：后端没确认、索引不自证一致时一个字节都不动。
        try:
            services.assert_write_ready()
        except (IndexMismatch, BackendUnavailable) as exc:
            raise HTTPException(503, str(exc)) from exc
        doc = repo.get_document_by_id(services.conn, doc_id)
        if not doc or doc.project_name != project_name:
            raise HTTPException(404, "这个项目下没有该文档")
        result = admin.delete_document(services.conn, services.store, doc_id)
        if not result["deleted"]:
            raise HTTPException(404, result["reason"])
        return {**result, "filename": doc.filename}

    @app.post("/api/admin/clear")
    def clear_all(request: Request, req: ClearRequest) -> dict:
        """清空整个库。不可撤销，所以要求 `confirm` 与 `CLEAR_CONFIRM` 完全一致。

        这条路由**不做** `assert_write_ready`：它是修复入口，索引卡在半途（`unsynced`、
        重建没走完）时恰恰最需要它。安全性由 `admin.clear_all` 里那三条保证——后端身份
        对得上、索引确实归零、业务表确实清空。

        但它**要做**口令校验：能修数据的入口，同样能被用来毁数据。走 CLI 的
        `python -m rag4prefa clear` 不受影响（本机身份已由 shell 建立）。
        """
        require_admin(request)
        if req.confirm.strip() != CLEAR_CONFIRM:
            raise HTTPException(
                400, f'要清空必须显式确认：{{"confirm": "{CLEAR_CONFIRM}"}}'
            )
        services = svc(request)
        return admin.clear_all(services.conn, services.store)

    # ── 检索与问答 ───────────────────────────────────────

    @app.post("/api/search")
    async def search(request: Request, req: SearchRequest) -> dict:
        services = svc_billable(request, need=("embedding",))
        effective = ask.build_filter(req.question, req.filter, repo.project_names(services.conn))
        try:
            services.assert_query_ready()
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
        services = svc_billable(request, need=("embedding", "llm"))
        try:
            services.require_embedder()
            services.require_llm()
        except (EmbeddingError, LLMError) as exc:
            raise HTTPException(503, str(exc)) from exc
        # 模型不一致/索引与事实不符的检查放在开始流之前：能改 HTTP 状态码的时候说清楚，
        # 别用 SSE error 收尾（客户端已经开始读 token 流了）
        try:
            # 先看"本进程有没有确认过活动后端"，再看索引与配置对不对得上：
            # 前者挡的是"手里这份索引可能压根不是这一版"，后者挡的是"模型/维度不匹配"。
            services.assert_query_ready()
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


def _missing_key_message(roles: list[str]) -> str:
    """把"缺哪把 Key、它用来做什么"一次说全。

    只写"请填写自己的 API Key"是不够的：用户常常填了其中一把，看到 401 会以为
    是填的那把无效，于是反复核对一把没问题的 Key。缺项逐个点名，指向才唯一。
    """
    parts = "；".join(f"{KEY_ROLES[role][0]}（{KEY_ROLES[role][1]}）" for role in roles)
    return (
        "这个部署不接受匿名请求：服务端不代付模型额度。"
        f"缺少你自己的 Key：{parts}。"
        "请在设置里填入自己的 API Key，只要它属于服务端指定的那家服务商即可。"
    )


def _redact(message: str, *secrets: str) -> str:
    """把错误信息里可能出现的 Key 抹掉。

    客户端异常有时会把请求体或 header 带进文本里，直接回显等于把 Key 写进前端日志。
    太短的串（如 "test"）替换会误伤正常文字，所以只处理有实际长度的。
    """
    for secret in secrets:
        if secret and len(secret) >= 6:
            message = message.replace(secret, "***")
    return message[:600]


def _explain_error(message: str, base_url: str, key: str = "") -> str:
    """把厂商的英文报错补成「下一步该做什么」。

    三类根因互斥，只补最先命中的那条——它们的结论互相矛盾：
    账号阻断说明「Key 已通过鉴权」，此时再说「Key 未被接受」会把人带去重建 Key；
    模型不在目录同理，那是配置问题，不是凭据问题。越具体的根因越先判。
    """
    for step in (_with_account_hint, _with_model_hint):
        explained = step(message, base_url, key)
        if explained != message:
            return explained
    return _with_auth_hint(message, base_url, key)


def _with_account_hint(message: str, base_url: str, key: str = "") -> str:
    """账号未验证 / 欠费 / 额度用尽时，指出这是账号状态而不是凭据问题。"""
    hint = account_blocked_hint(message, base_url)
    return f"{message}\n{hint}" if hint else message


def _with_auth_hint(message: str, base_url: str, key: str = "") -> str:
    """401 类错误补一句"Key 可能不属于这个平台"。

    同一个 401 有两种成因：Key 复制错了，或者 Key 是别家的。后者更常见于
    "两个输入框长得一样"的设置面板，而报错文本本身永远只说 Token 无效。
    """
    lowered = message.lower()
    if "401" in message or "authentication" in lowered or "token is invalid" in lowered:
        return message + auth_failure_hint(base_url, key)
    return message


def _with_model_hint(message: str, base_url: str, key: str = "") -> str:
    """模型名不在该平台目录里时，指出正确的排查方向与查法。

    这种失败很像鉴权问题（都是调不通），但 token 其实是好的。不写清楚的话，
    用户会反复重建 Key，而真正该改的是模型名。
    """
    lowered = message.lower()
    missing_model = (
        "no provider supported" in lowered
        or "model not found" in lowered
        or "invalid model" in lowered
    )
    if missing_model:
        host = base_url.split("//")[-1].split("/")[0]
        return (
            f"{message}\n该平台的模型目录会下线旧 ID。列出当前可调用的模型（无需 Key）："
            f"curl -s https://{host}/v1/models"
        )
    return message


def _safe_name(name: str) -> str:
    """防目录穿越：只取最后一段，并剔除非法字符。"""
    base = Path(name.replace("\\", "/")).name
    base = "".join(ch for ch in base if ch not in '<>:"|?*\x00')
    if not base or base in {".", ".."}:
        raise HTTPException(400, f"非法名称：{name!r}")
    return base


app = create_app()
