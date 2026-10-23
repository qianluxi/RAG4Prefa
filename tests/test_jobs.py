"""任务状态：进程重启后，遗留的 running/queued 必须有个确定的结局。

入库任务跑在进程内（`asyncio.create_task`），进程一死它们永远不会前进。
不处理的话，前端会永远显示"运行中"，运维无法判断是排队还是已经死了。
"""

from __future__ import annotations

import sqlite3

from rag4prefa import repo
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
