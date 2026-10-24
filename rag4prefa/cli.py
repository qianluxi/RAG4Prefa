"""命令行入口：`python -m rag4prefa <command>`。

保留 CLI 的理由：入库、重索引、评估都是批处理，不该被迫起 HTTP 服务。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from . import admin, ask, consistency, ingest, repo
from .adapters.vectors import RebuildError
from .config import get_settings
from .consistency import IndexMismatch
from .db import init_db
from .domain import Filter
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


def cmd_doctor(_: argparse.Namespace) -> int:
    services = build_services()
    try:
        _print(services.health())
        return 0 if services.health()["ok"] else 1
    finally:
        services.close()


def cmd_ingest(args: argparse.Namespace) -> int:
    files = []
    for target in args.paths:
        path = Path(target)
        if path.is_dir():
            files.extend(sorted(p for p in path.glob("*.docx")))
        elif path.suffix.lower() == ".docx":
            files.append(path)
    if not files:
        print("没有找到 .docx 文件", file=sys.stderr)
        return 2

    services = build_services()
    try:
        import uuid

        job_id = uuid.uuid4().hex[:12]
        repo.upsert_project(services.conn, args.project, args.code, args.started_on)
        repo.create_job(
            services.conn,
            job_id,
            "ingest",
            {"project": args.project, "files": [f.name for f in files]},
        )
        outcomes = ingest.ingest_batch(
            services.conn,
            services.store,
            services.require_embedder(),
            services.settings,
            job_id=job_id,
            files=files,
            project_name=args.project,
            project_code=args.code,
            doc_year=args.year,
        )
        services.maybe_promote()
        _print([o.as_dict() for o in outcomes])
        return 0 if any(o.status == "indexed" for o in outcomes) else 1
    finally:
        services.close()


def cmd_reindex(args: argparse.Namespace) -> int:
    services = build_services()
    try:
        result = ingest.reindex(
            services.conn,
            services.store,
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
        total = ingest.rebuild_store(services.conn, services.store)
        _print(
            {
                "rebuilt": total,
                "model": repo.get_meta(services.conn, "embedding_model"),
                "dim": repo.get_meta(services.conn, "embedding_dim"),
                "index_state": repo.get_index_state(services.conn),
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
        # ask.search 是 async：必须跑在事件循环里，否则拿到的是协程而不是结果
        hits, applied, timings = asyncio.run(
            ask.search(
                services.conn,
                services.store,
                services.require_embedder(),
                services.settings,
                question=args.question,
                filt=filt,
                top_k=args.top_k,
            )
        )
        _print(
            {
                "filter": applied.model_dump(exclude_none=True),
                "timings": timings,
                "results": [
                    {
                        "chunk_id": h.chunk_id,
                        "score": round(h.score, 4),
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
    except IndexMismatch as exc:
        print(f"检索已拒绝：{exc}", file=sys.stderr)
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

    p = sub.add_parser("ingest", help="导入 docx（文件或目录）")
    p.add_argument("paths", nargs="+")
    p.add_argument("--project", required=True)
    p.add_argument("--code", default="")
    p.add_argument("--started-on", default="")
    p.add_argument("--year", type=int, default=None)
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
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("projects", help="列出项目")
    p.set_defaults(func=cmd_projects)

    p = sub.add_parser("verify", help="逐 id 核对片段、归档与向量索引是否一致")
    p.set_defaults(func=cmd_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
