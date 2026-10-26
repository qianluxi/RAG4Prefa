"""任务状态：进程重启后，遗留的 running/queued 必须有个确定的结局。

入库任务跑在进程内（`asyncio.create_task`），进程一死它们永远不会前进。
不处理的话，前端会永远显示"运行中"，运维无法判断是排队还是已经死了。
"""

from __future__ import annotations

import sqlite3

from rag4prefa import repo
from rag4prefa.config import Settings
from rag4prefa.domain import JobState


def test_interrupted_jobs_are_marked_failed(conn: sqlite3.Connection) -> None:
    repo.create_job(conn, "job-running", "ingest", {"project": "甲"})
    repo.update_job(conn, "job-running", state=JobState.running, progress=0.4, message="解析中")
    repo.create_job(conn, "job-queued", "ingest", {"project": "乙"})
    repo.create_job(conn, "job-done", "ingest", {"project": "丙"})
    repo.update_job(conn, "job-done", state=JobState.done, progress=1.0)

    changed = repo.mark_interrupted_jobs(conn)

    assert changed == 2
    running = repo.get_job(conn, "job-running")
    assert running.state is JobState.failed
    assert running.finished_at
    assert "进程重启" in running.message
    assert "解析中" in running.message, "原有进度说明应保留下来便于排查"
    assert repo.get_job(conn, "job-queued").state is JobState.failed
    assert repo.get_job(conn, "job-done").state is JobState.done, "已完成的任务不该被动"


def test_no_interrupted_job_is_a_noop(conn: sqlite3.Connection) -> None:
    assert repo.mark_interrupted_jobs(conn) == 0


# ── "谁还活着"：别把别人正在跑的任务标成中断 ──────────────────


def test_a_running_job_survives_a_second_process_starting_up(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """服务在跑的时候，另开一个 CLI 不该把它的入库任务标成失败。

    `build_services()` 原来无条件清理 running/queued——那是为崩溃恢复准备的。
    可同一个库现在同时被 uvicorn 和 CLI 使用，跑一次 `doctor` 就能把在线任务标成
    failed，而数据其实还在写。库里只有 `state='running'` 这一个事实，分不出
    "旧进程死了"和"另一个进程正在干活"，所以要看进程租约。
    """
    from rag4prefa.concurrency import WorkerLease
    from rag4prefa.services import build_services

    repo.create_job(conn, "job-live", "ingest", {"project": "甲"})
    repo.update_job(conn, "job-live", state=JobState.running, progress=0.5, message="解析中")

    lease = WorkerLease()
    assert lease.acquire(conn, timeout=0.0), "测试自己要能占住租约"
    try:
        # 另一个"进程"（这里只是另一条连接）装配服务
        services = build_services(settings)
        try:
            assert repo.get_job(conn, "job-live").state is JobState.running, (
                "有活着的服务进程时不许动它的任务"
            )
            assert any("另一个服务进程" in n for n in services.notes), (
                "要说清为什么这次没清理，否则看起来像清理逻辑失效了"
            )
        finally:
            services.close()
    finally:
        lease.release()


def test_stale_jobs_are_still_cleaned_when_no_worker_is_alive(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """没有活着的服务进程时，清理照旧——否则崩溃残留会永远显示"运行中"。"""
    from rag4prefa import concurrency
    from rag4prefa.services import build_services

    concurrency.WORKER_LEASE.release()  # 确保本进程也不是"活着的服务"（通常是空操作）
    repo.create_job(conn, "job-dead", "ingest", {"project": "乙"})
    repo.update_job(conn, "job-dead", state=JobState.running, progress=0.2, message="解析中")

    services = build_services(settings)
    try:
        assert repo.get_job(conn, "job-dead").state is JobState.failed
        assert any("标记为失败" in n for n in services.notes)
    finally:
        services.close()
