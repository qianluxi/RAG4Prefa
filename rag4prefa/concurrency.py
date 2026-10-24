"""写入协调：同一时刻只允许一个会改事实或改索引的动作。

为什么需要两把锁，而不是一把：

1. **写锁（进程内，可重入）**——所有写事务共用一条 SQLite 连接。
   `check_same_thread=False` 只解决"能不能用"，没解决"能不能交错"：两个线程同时开
   事务时，一个 `COMMIT` 会把另一个还没写完的事务一起提交，回滚也会互相踩。这把锁
   包住每一个写事务，粒度是"一条 SQL 批次"，只要求别交叉，不要求别并发。
2. **维护锁（进程内 + 跨进程）**——重建、入库、后端切换、删项目这类动作需要**独占**，
   而且必须跨进程：`uvicorn` 和 `python -m rag4prefa reindex` 是两个进程，进程内锁
   管不到对方。跨进程用操作系统文件锁：进程被杀时内核自动释放，不会留下"永远上锁"
   的假故障（用 `O_EXCL` 建锁文件就得自己判活、自己清残留，那是另一类 bug 的来源）。

维护锁的粒度是"一次动作"，不是"一条语句"。理由是重建的执行顺序：读快照 →
算 embedding（可能几分钟）→ 写归档 → 换索引。如果只在写的时候加锁，算完才发现事实
表已经被改过——那时条数可能仍然一样，只有 id 不一样（`rebuild_store` 的条数校验
发现不了，`verify_ids` 才能，而它不在主路径上）。所以从读快照到切换完成必须独占。

抢不到锁的语义分两种，调用方自己选：
- `timeout=0`：立刻抛 `MaintenanceBusy`。API 场景要的就是这个——用户宁可看到
  "正在重建索引，请稍后"，也不要一个挂在"运行中"、十分钟后才失败的任务。
- `timeout=None`：排队等。CLI 维护命令用这个，操作员按下去就知道要等。
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

MAINTENANCE_LOCK_NAME = "maintenance.lock"
_HOLDER_SUFFIX = ".holder"


class MaintenanceBusy(RuntimeError):
    """另一个维护动作正在进行，本次操作没有执行（不是失败，是被挡住）。"""


# 写事务串行化。用 RLock 而不是 Lock：`transaction()` 允许被同线程嵌套调用
# （例如归档写入里再调一次别的 repo 函数），嵌套时不该自己把自己锁死。
_SQLITE_WRITE_LOCK = threading.RLock()


def sqlite_write() -> threading.RLock:
    """包住写事务的锁。读不加锁——见模块说明与 HANDOFF 的边界说明。"""
    return _SQLITE_WRITE_LOCK


# 维护锁的进程内部分。单独一把 Lock，和写锁分开：维护动作要独占好几分钟，
# 不能让它把这段时间里的普通写事务也一起锁住（那些写本来就会被维护锁挡在门外）。
_MAINTENANCE_GUARD = threading.Lock()
_depth = threading.local()


def lock_path_for(conn) -> Path | None:
    """维护锁文件的位置：跟数据库同一个目录。

    从连接上读而不是从配置读，是为了让所有拿得到 `conn` 的调用方（ingest、reindex、
    rebuild、api、cli）用同一把锁——传配置容易漏一处，漏掉的那处就是并发窗口。
    内存库没有文件可放锁，返回 None，退化成进程内互斥。
    """
    try:
        rows = conn.execute("PRAGMA database_list").fetchall()
    except Exception:  # pragma: no cover - 连接已经关掉时才会走到
        return None
    for row in rows:
        # 列顺序是 (seq, name, file)：按序号取值，别指望 row_factory
        if row[1] == "main" and row[2]:
            return Path(row[2]).parent / MAINTENANCE_LOCK_NAME
    return None


class _FileLock:
    """操作系统级排他锁，锁住文件第 0 个字节。

    Windows 用 `msvcrt.locking`，POSIX 用 `flock`。两者都在**进程退出时由内核释放**，
    所以崩溃不会留下需要人工清理的锁文件。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: IO[bytes] | None = None

    def try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 句柄要一直拿着（锁跟着句柄走），不能用 with
        handle = open(self.path, "a+b")  # noqa: SIM115
        try:
            if os.name == "nt":
                import msvcrt

                if handle.seek(0, os.SEEK_END) == 0:
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        self._handle = handle
        return True

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _holder_path(path: Path) -> Path:
    return path.with_name(path.name + _HOLDER_SUFFIX)


def _write_holder(path: Path, purpose: str) -> None:
    """把"谁占着锁"写到旁边的说明文件。

    只用于报错时能说清是谁——"另一个维护动作正在运行"这种话等于没说，操作员要知道
    是等还是去停掉哪个进程。写失败不影响加锁本身。
    """
    info = {
        "pid": os.getpid(),
        "purpose": purpose,
        "since": datetime.now(UTC).replace(microsecond=0).isoformat(),
    }
    with contextlib.suppress(OSError):
        _holder_path(path).write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


def _read_holder(path: Path) -> str:
    try:
        info = json.loads(_holder_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    purpose = info.get("purpose") or "未知动作"
    pid = info.get("pid")
    since = info.get("since") or ""
    tail = f"，pid {pid}" if pid else ""
    return f"{purpose}{tail}" + (f"，自 {since} 起" if since else "")


def busy_message(path: Path | None, purpose: str) -> str:
    holder = _read_holder(path) if path else ""
    if holder:
        return (
            f"另一个维护动作正在运行（{holder}），本次「{purpose}」没有执行。"
            "等它结束后重试；如果确认那个进程已经不存在，直接重试即可——"
            "锁随进程退出由操作系统释放。"
        )
    return (
        f"另一个维护动作正在运行，本次「{purpose}」没有执行，稍后重试。"
        "（同一时刻只允许一个动作改事实或改索引：重建期间算出的向量一旦碰上并发的"
        "入库，条数可能仍然一样，只有 id 不一样。）"
    )


class _Guard:
    """维护锁。**同一线程可重入**：`maybe_promote` 里再调 `rebuild_store` 是嵌套的，
    `ingest_batch` 里再调 `ingest_file` 也是嵌套的。

    重入深度记在**线程**上而不是实例上——每次 `maintenance_lock()` 都是一个新对象，
    按实例记深度等于没有可重入，嵌套的第二层会去抢自己线程已经占着的锁。
    """

    def __init__(self, path: Path | None, purpose: str, timeout: float | None) -> None:
        self.path = path
        self.purpose = purpose
        self.timeout = timeout
        self._file: _FileLock | None = None
        self._owner = False  # 是否由本对象真正拿到锁（决定退出时要不要释放）

    def __enter__(self) -> _Guard:
        if getattr(_depth, "value", 0):
            _depth.value += 1
            return self
        deadline = None if self.timeout is None else time.monotonic() + max(0.0, self.timeout)
        while True:
            if self._try_take():
                _depth.value = 1
                self._owner = True
                if self.path is not None:
                    _write_holder(self.path, self.purpose)
                return self
            if deadline is not None and time.monotonic() >= deadline:
                raise MaintenanceBusy(busy_message(self.path, self.purpose))
            # 排队时不要空转：持锁方可能是几分钟的重建
            remaining = float("inf") if deadline is None else deadline - time.monotonic()
            time.sleep(0.02 if remaining < 0.5 else 0.2)

    def _try_take(self) -> bool:
        if not _MAINTENANCE_GUARD.acquire(blocking=False):
            return False
        if self.path is None:
            return True
        file_lock = _FileLock(self.path)
        if file_lock.try_acquire():
            self._file = file_lock
            return True
        _MAINTENANCE_GUARD.release()
        return False

    def __exit__(self, *exc_info) -> None:
        _depth.value = max(0, getattr(_depth, "value", 1) - 1)
        if not self._owner:
            return
        try:
            if self._file is not None:
                with contextlib.suppress(OSError):
                    _holder_path(self.path).unlink(missing_ok=True)
                self._file.release()
                self._file = None
        finally:
            _MAINTENANCE_GUARD.release()


def maintenance_lock(
    conn, *, purpose: str, timeout: float | None = 0.0
) -> _Guard:
    """拿维护锁。`conn` 只用来定位锁文件，不用它做任何查询。

    用法：
        with maintenance_lock(conn, purpose="全量重建", timeout=None):
            ...
    """
    return _Guard(lock_path_for(conn), purpose, timeout)


def is_busy(conn) -> str | None:
    """不改变任何状态地问一句"现在有维护动作在跑吗"，有就返回它的说明。

    只用于给用户一句更好的话（例如上传前先看一眼），不能当准入依据——问完到真正
    动手之间还有窗口，真正的准入只能是把维护锁拿在手里。
    """
    if getattr(_depth, "value", 0):
        # 本线程正占着锁（例如入库流程内部问一句），不能把自己报成"别人在跑"
        return None
    path = lock_path_for(conn)
    if path is None:
        if not _MAINTENANCE_GUARD.acquire(blocking=False):
            return "另一个维护动作"
        _MAINTENANCE_GUARD.release()
        return None
    file_lock = _FileLock(path)
    if file_lock.try_acquire():
        file_lock.release()
        return None
    return _read_holder(path) or "另一个维护动作"
