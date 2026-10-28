"""命令行入口：`python -m rag4prefa <command>`。

保留 CLI 的理由：入库、重索引、评估都是批处理，不该被迫起 HTTP 服务。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import admin, ask, concurrency, consistency, ingest, repo
from .adapters.parsers import SUPPORTED_SUFFIXES, explain_unsupported
from .adapters.vectors import BackendUnavailable, RebuildError, StoreRef
from .concurrency import MaintenanceBusy
from .config import get_settings
from .consistency import IndexMismatch
from .db import init_db
from .domain import Filter, JobState
from .processcontrol import stop_port
from .services import build_services


def _print(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_init(_: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    init_db(settings.db_path)
    print(f"已初始化：{settings.db_path}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("rag4prefa.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """停掉监听指定端口的本程序。

    先问 `/api/health` 认身份，认不出来就不动手——这个命令最容易写错的地方
    是"谁占着端口就杀谁"，那等于给了自己一次误杀别人服务的权限。
    """
    result = stop_port(args.port, force=args.force)
    print(result.message)
    return result.exit_code


def cmd_doctor(_: argparse.Namespace) -> int:
    services = build_services()
    try:
        _print(services.health())
        return 0 if services.health()["ok"] else 1
    finally:
        services.close()


def _walk_corpus(root: Path) -> list[Path]:
    """目录里的候选文件，**递归**收集。

    以前这里用 `iterdir()`，只拿第一层。语料按 `文献资料/0X_分类/子类/` 组织之后，
    子目录里的文件既不在 `files` 里、也不在 `skipped` 里——连"被跳过"都算不上，是
    彻底消失。2026-10-28 实测：45 份的目录丢掉了 30 份，而导入报告一切正常。

    以 `.` 开头的目录整个跳过：工程目录下还有 `.workbuddy/`（工作记忆），rglob 会
    把里面的 md 当成语料收走。
    """
    picked: list[Path] = []
    for entry in sorted(root.rglob("*")):
        rel = entry.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if not entry.is_file() or entry.name.startswith("~$"):
            continue  # Office 锁文件不是"被跳过的语料"
        picked.append(entry)
    return picked


def _collect_files(targets: list[str]) -> tuple[list[Path], list[tuple[Path, str]]]:
    """分离「能处理」与「被跳过」，并给每个跳过项一个可执行的理由。

    以前目录导入只按扩展名 glob 一遍，`.doc` / `.xlsx` 这些文件**不会出现在任何输出里**：
    用户看到"入库成功"就以为都进去了，实际少了几份，而且没有任何提示。语料不全是个
    安静的错误——检索效果变差，但没人知道是缺文件导致的。

    目录递归收集；同一个文件被重复指定（例如既给了目录又给了目录里的某个文件）只算一次，
    否则它会入库两次、白花一遍 embedding。
    """
    files: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    seen: set[Path] = set()
    for target in targets:
        path = Path(target)
        if path.is_dir():
            for entry in _walk_corpus(path):
                if entry in seen:
                    continue
                seen.add(entry)
                if entry.suffix.lower() in SUPPORTED_SUFFIXES:
                    files.append(entry)
                else:
                    skipped.append((entry, explain_unsupported(entry.suffix)))
        elif path.suffix.lower() in SUPPORTED_SUFFIXES:
            if path not in seen:
                seen.add(path)
                files.append(path)
        elif path.exists():
            skipped.append((path, explain_unsupported(path.suffix)))
        else:
            skipped.append((path, "文件不存在"))
    return files, skipped


def _report_skipped(skipped: list[tuple[Path, str]]) -> None:
    if not skipped:
        return
    print(f"\n跳过 {len(skipped)} 个文件（未入库）：", file=sys.stderr)
    for path, reason in skipped:
        print(f"  - {path.name}：{reason}", file=sys.stderr)


def cmd_ingest(args: argparse.Namespace) -> int:
    files, skipped = _collect_files(args.paths)
    _report_skipped(skipped)
    if not files:
        print(f"没有找到可处理的文件（{' / '.join(SUPPORTED_SUFFIXES)}）", file=sys.stderr)
        return 2

    services = build_services()
    # `--ocr` 只是一次性的运行期覆盖，不回写配置：operator 试一次整份重识别，
    # 不该顺手把下次的默认行为也改掉。放在 assert_write_ready 之前——
    # 缺 Key 要在建任务之前就报出来，别留下一条"看起来导过了"的失败记录。
    if getattr(args, "ocr", None):
        services.settings.ocr_mode = args.ocr
    # 与 API 上传同一条准入：后端没确认时不许动事实与索引。放在建任务之前，
    # 被拒绝时库里连一条"看起来导过了"的记录都不会留下。
    try:
        services.assert_write_ready()
    except IndexMismatch as exc:
        services.close()
        print(f"入库未执行：{exc}", file=sys.stderr)
        return 1
    # 自己也是"正在工作的进程"：占住租约，另一个终端里的 doctor 就不会把这次入库的
    # 任务标成"进程中断"。抢不到说明有服务进程在跑——那时租约在它手上，清理本来也
    # 不会发生，继续入库即可（并发写由维护锁挡住）。
    leased = concurrency.WORKER_LEASE.acquire(services.conn, timeout=0.0)
    try:
        import uuid

        job_id = uuid.uuid4().hex[:12]
        # 项目与任务都不在这里预建：`ingest_batch` 会把它们放进维护锁里面写。
        # 先在锁外建好，一旦被别的维护动作挡住，就会留下一个"看起来导过了"的空项目。
        repo.create_job(
            services.conn,
            job_id,
            "ingest",
            {"project": args.project, "files": [f.name for f in files]},
        )
        try:
            # 传 StoreRef 而不是 `services.store`：装配发生在拿维护锁**之前**，
            # 等锁的这段时间里后端指针可能被别的进程换掉（例如它刚把 numpy 迁到
            # Milvus）。传对象就是"用取参数那一刻的后端写完整批"，而那一份可能
            # 再也没人查。真正的最后一道防线是锁内的 `ensure_backend_current`：
            # 本进程的指针跟上没跟上，以库里记着的活动后端为准。
            outcomes = ingest.ingest_batch(
                services.conn,
                StoreRef(lambda: services.store),
                services.require_embedder(),
                services.settings,
                job_id=job_id,
                files=files,
                project_name=args.project,
                project_code=args.code,
                started_on=args.started_on or None,
                doc_year=args.year,
                # 操作员按下去就知道要等：排队而不是立刻拒绝
                timeout=None,
                force=args.force,
            )
        except MaintenanceBusy as exc:
            # 排队也会被推到超时（另一进程持锁不放手），此时一个文件都没处理
            repo.update_job(
                services.conn, job_id, state=JobState.failed, message=f"没有执行：{exc}"
            )
            print(f"入库未执行：{exc}", file=sys.stderr)
            return 2
        services.maybe_promote()
        _print([o.as_dict() for o in outcomes])
        return 0 if any(o.status == "indexed" for o in outcomes) else 1
    finally:
        if leased:
            concurrency.WORKER_LEASE.release()
        services.close()


def cmd_reindex(args: argparse.Namespace) -> int:
    services = build_services()
    try:
        result = ingest.reindex(
            services.conn,
            StoreRef(lambda: services.store),
            services.require_embedder(),
            services.settings,
            project_name=args.project,
        )
        _print(result)
        return 0
    except ingest.ReindexError as exc:
        # 这类错误需要人做决定（比如维度变了只能全量重建），直接给出可执行的下一步
        print(f"reindex 未执行：{exc}", file=sys.stderr)
        return 1
    finally:
        services.close()


def cmd_rebuild(_: argparse.Namespace) -> int:
    """按 SQLite 归档重建向量索引，**不重新调用 embedding**。

    为什么单独有这个命令：`reindex` 会把每段重新算一遍向量，而重建失败之后归档里
    本来就存着完整的向量副本——再花钱算一遍纯属浪费。两者都是"清除索引切换标记"
    的合法出口，区别只在要不要重新付 embedding 的钱。

    恢复是否成功不看"函数有没有抛异常"，而看三条一起到位：索引条数 = 归档条数、
    元数据 = 归档里的模型与维度、标记回到 ready。任何一条不满足都返回非零，
    并把标记留在原处。
    """
    services = build_services()
    try:
        total = ingest.rebuild_store(services.conn, StoreRef(lambda: services.store))
        _print(
            {
                "rebuilt": total,
                "model": repo.get_meta(services.conn, "embedding_model"),
                "dim": repo.get_meta(services.conn, "embedding_dim"),
                "index_state": repo.get_index_state(services.conn),
                # 换后端也靠这条命令，所以要把"现在登记的是谁"打出来：
                # 只报 rebuilt 条数的话，运维看不出这次把活动后端改掉了
                "vector_backend": repo.get_vector_backend(services.conn),
                # 恢复用的是归档里的模型，它可能和当前配置不同——指出来，否则用户
                # 会在下一次检索被拒时一头雾水
                "configured_model": (
                    services.embedder.model if services.embedder else None
                ),
            }
        )
        return 0
    except (ingest.ReindexError, RebuildError) as exc:
        print(f"rebuild 未完成：{exc}", file=sys.stderr)
        return 1
    finally:
        services.close()


def cmd_ask(args: argparse.Namespace) -> int:
    services = build_services()
    try:
        filt = Filter(projects=[args.project] if args.project else None)
        # 与 API 走同一个准入函数，不靠"路由那边已经拦过"。CLI 是这个库的另一个
        # 使用入口：后端没确认时它同样可能拿一份内容过期的索引回答，而那份答案看起来
        # 完全正常（2026-10-24 第十七轮复审）。
        services.assert_query_ready()
        # ask.search 是 async：必须跑在事件循环里，否则拿到的是协程而不是结果
        asked_rerank = getattr(args, "rerank", None)
        hits, applied, timings = asyncio.run(
            ask.search(
                services.conn,
                services.store,
                services.require_embedder(),
                services.settings,
                question=args.question,
                filt=filt,
                top_k=args.top_k,
                # 重排交给 ask.search 做：那里保证"先重排、后截断"。
                # 命令行这边自己截一次再去重排，就只剩 top_k 条候选可挑。
                reranker=getattr(services, "reranker", None),
                use_rerank=asked_rerank,
            )
        )
        do_rerank = ask.rerank_available(
            getattr(services, "reranker", None)
        ) if asked_rerank is None else asked_rerank
        _print(
            {
                "filter": applied.model_dump(exclude_none=True),
                "timings": timings,
                "rerank": do_rerank,
                "results": [
                    {
                        "chunk_id": h.chunk_id,
                        "score": round(h.rerank_score or h.score, 4),
                        "project": h.project_name,
                        "file": h.filename,
                        "chapter": " > ".join(h.chapter_path),
                        "text": h.text[:160],
                    }
                    for h in hits
                ],
            }
        )
        if args.answer and hits:
            from .prompts import SYSTEM_ANSWER, build_answer_prompt

            reply = services.require_llm().complete(
                SYSTEM_ANSWER, build_answer_prompt(args.question, hits, "")
            )
            print("\n--- 回答 ---\n" + reply)
        return 0
    except (IndexMismatch, BackendUnavailable) as exc:
        # 异常消息本身已经写清了"检索已拒绝/已停用"与修法，不再套一层前缀，
        # 否则终端里会出现「检索未执行：检索已拒绝：……」。
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        services.close()


def cmd_verify(_: argparse.Namespace) -> int:
    """逐 id 核对「片段 / 归档 / 实际索引」三方是否一致。

    health 里只做计数比对（廉价，每次请求都能跑）；条数相同但内容不同的情况
    只能靠这个命令发现，代价是要把索引主键全拉一遍。
    """
    services = build_services()
    try:
        report = consistency.verify_ids(services.conn, services.store)
        _print(report)
        if not report["ok"]:
            # 两种"不一致"要说清是哪一种：id 对得上、只是上次切换没走完时，
            # 让人去查"哪里少了几条"只会白费功夫。
            if report.get("pending_rebuild") and not any(report.get("counts", {}).values()):
                print(
                    f"\n上次索引重建没有走完（{report['pending_rebuild']}）：三方 id 全都对得上，"
                    "但索引里的向量可能还是旧的。执行 `python -m rag4prefa reindex`"
                    "（不带 --project）重建，或 `rebuild` 直接从归档恢复。",
                    file=sys.stderr,
                )
            else:
                print(
                    "\n索引与事实不一致。先用 `python -m rag4prefa reindex` 全量重建，"
                    "重建后重跑本命令确认干净。",
                    file=sys.stderr,
                )
            return 1
        return 0
    finally:
        services.close()


def cmd_projects(_: argparse.Namespace) -> int:
    services = build_services()
    try:
        _print([p.model_dump() for p in admin.list_projects(services.conn)])
        return 0
    finally:
        services.close()


def cmd_delete(args: argparse.Namespace) -> int:
    """删项目。默认只预览，`--yes` 才真删。

    这个命令以前只存在于 HTTP 接口（`DELETE /api/projects/{name}`）里，命令行用户删掉
    磁盘上的文件之后没有办法把库里的记录一起删掉——只能看着一个"文件已经不在了"的项目
    继续参与检索。而删项目是**事与索引一起删**，和入库同一类动作，必须走维护锁、
    必须能核对影响面，所以它该有一个正经的命令行入口。

    默认不删，是因为这个动作没有撤销：片段、向量、项目摘要一起没。
    """
    services = build_services()
    try:
        docs = admin.list_documents(services.conn, args.project)
        if not docs:
            print(f"项目「{args.project}」不存在", file=sys.stderr)
            return 1
        total = sum(d.chunk_count for d in docs)
        print(f"项目「{args.project}」：{len(docs)} 份文档、{total} 个片段")
        for d in docs:
            print(f"  - {d.filename}（{d.chunk_count} 块）")
        if not args.yes:
            print("\n未删除。确认要删就加 --yes（片段、向量、项目摘要会一起没，没有撤销）")
            return 0
        result = admin.delete_project(services.conn, services.store, args.project)
        _print(result)
        return 0 if result.get("deleted") else 1
    except MaintenanceBusy as exc:
        print(f"删除未执行：{exc}", file=sys.stderr)
        return 2
    finally:
        services.close()


def cmd_delete_doc(args: argparse.Namespace) -> int:
    """删单个文件（连同片段、全文索引与向量）。默认只预览。

    定位方式二选一：`--doc-id`（接口/前端给的是这个），或 `--project` + `--file`
    （命令行用户手里有的是文件名）。后者要求文件名与入库时**完全一致**，所以没找到时
    会明确说"文件名要对上"，而不是含糊地说"不存在"。
    """
    services = build_services()
    try:
        if args.doc_id:
            doc = repo.get_document_by_id(services.conn, args.doc_id)
        elif args.project and args.file:
            doc = repo.get_document(services.conn, args.project, args.file)
        else:
            print("要删哪一份？给 --doc-id，或者 --project 加 --file", file=sys.stderr)
            return 1
        if not doc:
            print(
                "没找到这份文档。--file 要与入库时的文件名完全一致（含扩展名），"
                "用 `python -m rag4prefa delete-doc --help` 看用法；"
                "项目里有哪些文件可以看 `python -m rag4prefa projects` 之后的项目详情。",
                file=sys.stderr,
            )
            return 1
        print(f"{doc.project_name} / {doc.filename}：{doc.chunk_count} 个片段")
        if not args.yes:
            print("\n未删除。确认要删就加 --yes（片段与向量会一起没，没有撤销）")
            return 0
        result = admin.delete_document(services.conn, services.store, doc.id)
        _print(result)
        return 0 if result.get("deleted") else 1
    except MaintenanceBusy as exc:
        print(f"删除未执行：{exc}", file=sys.stderr)
        return 2
    finally:
        services.close()


def cmd_clear(args: argparse.Namespace) -> int:
    """清空整个库：项目、文档、片段、向量、任务记录。默认只预览。

    embedding 缓存与配置（模型、后端身份）刻意保留：前者只是"这段文字算过向量"的
    缓存，删了要多花一次钱；后者描述"这个库怎么跑"，清数据不该顺手把它一起清掉。
    """
    services = build_services()
    try:
        counts = repo.data_counts(services.conn)
        print("将删除：")
        for key, label in (
            ("projects", "项目"),
            ("documents", "文档"),
            ("chunks", "片段"),
            ("vectors", "向量（索引与归档）"),
            ("jobs", "任务记录"),
        ):
            print(f"  {label}　{counts[key]}")
        if not args.yes:
            print("\n未清空。确认要清空就加 --yes（不可撤销；embedding 缓存与配置会保留）")
            return 0
        result = admin.clear_all(services.conn, services.store)
        _print(result)
        return 0
    except MaintenanceBusy as exc:
        print(f"清空未执行：{exc}", file=sys.stderr)
        return 2
    finally:
        services.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag4prefa", description="RAG4Prefa 课题文档知识库")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="初始化数据目录与数据库")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("serve", help="启动 HTTP 服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="检查依赖与配置")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("stop", help="停止监听端口上的服务（先认身份，再杀整棵进程树）")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--force",
        action="store_true",
        help="跳过身份确认直接 taskkill。只在确认端口上就是本程序、但健康检查没回应（卡死）时用；"
        "端口上是别的软件时会误杀",
    )
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("ingest", help="导入文档（docx / pdf，文件或目录）")
    p.add_argument("paths", nargs="+")
    p.add_argument("--project", required=True)
    p.add_argument("--code", default="")
    p.add_argument("--started-on", default="")
    p.add_argument("--year", type=int, default=None)
    p.add_argument(
        "--force",
        action="store_true",
        help="强制重新解析并覆盖（重新调 embedding）。解析逻辑升级后补新字段（如 PDF 页码）时用；"
        "不带 --year 时沿用该文档已有的年份，不会清空",
    )
    p.add_argument(
        "--ocr",
        choices=("off", "auto", "all"),
        default=None,
        help="临时覆盖 R4P_OCR_MODE：auto 只识别没有文字层的页；all 整份重做；"
        "off 关掉。给扫描件补内容时用，改完的结论不会写回 .env",
    )
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("reindex", help="按 SQLite 重建向量索引")
    p.add_argument("--project", default=None)
    p.set_defaults(func=cmd_reindex)

    p = sub.add_parser("rebuild", help="用归档里的现成向量重建索引（不重新 embedding）")
    p.set_defaults(func=cmd_rebuild)

    p = sub.add_parser("ask", help="命令行检索/问答")
    p.add_argument("question")
    p.add_argument("--project", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--answer", action="store_true", help="额外调用大模型给出回答")
    p.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否重排；不写则跟随配置（配了 R4P_RERANK_URL 就开）",
    )
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("projects", help="列出项目")
    p.set_defaults(func=cmd_projects)

    p = sub.add_parser("delete", help="删除项目（默认只预览，--yes 才真删）")
    p.add_argument("--project", required=True, help="项目名")
    p.add_argument("--yes", action="store_true", help="确认删除；不加只打印影响面")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser(
        "delete-doc", help="删除单个文件及其片段/向量（默认只预览，--yes 才真删）"
    )
    p.add_argument("--doc-id", default="", help="文档 id（接口与前端用的是这个）")
    p.add_argument("--project", default="", help="项目名（与 --file 一起用）")
    p.add_argument("--file", default="", help="文件名，要与入库时完全一致")
    p.add_argument("--yes", action="store_true", help="确认删除；不加只打印影响面")
    p.set_defaults(func=cmd_delete_doc)

    p = sub.add_parser("clear", help="清空整个库（默认只预览，--yes 才真清）")
    p.add_argument("--yes", action="store_true", help="确认清空；不加只打印影响面")
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser("verify", help="逐 id 核对片段、归档与向量索引是否一致")
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except IndexMismatch as exc:
        # 从命令体里漏出来的"索引/后端对不上"：这不是崩溃，是"该修或该重来"。
        # 让 traceback 出现在终端里，操作员只会以为程序有 bug。
        print(f"{args.command} 未执行：{exc}", file=sys.stderr)
        return 1
    except MaintenanceBusy as exc:
        print(f"{args.command} 未执行（另一个维护动作正在跑）：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
