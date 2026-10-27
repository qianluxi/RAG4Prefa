"""停服判据：谁在听、是不是本程序。

`rag4prefa stop` 会 `taskkill /T /F`，判错一次就误杀别人家的服务。所以这里钉的
只有两件事，而且每条都带反例：

1. **谁在听** —— 只认 `LISTENING`、且端口精确相等。`TIME_WAIT`（停服后立刻重看会
   有一堆）、`ESTABLISHED`（浏览器正连着）都不占监听端口；`88990` 不能被当成 8899。
2. **是不是本程序** —— `/api/health` 自报的 `app` 字段。问不到（None）与答"不是"
   （False）必须分开：前者可能是卡死的本程序，后者是别人家的服务，提示与动作都不同。

身份这层用真起一个本地 HTTP 服务来验，不 mock `urllib`——mock 掉就等于不验
"这一段字节到底能不能解析成 JSON"。
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from rag4prefa.processcontrol import (
    APP_MARKER,
    StopResult,
    parse_listening_pids,
    probe_app,
    stop_port,
)

NETSTAT = """活动连接

  协议  本地地址          外部地址        状态           PID
  TCP    0.0.0.0:8899           0.0.0.0:0              LISTENING       4211
  TCP    127.0.0.1:8899         127.0.0.1:57000        ESTABLISHED     4211
  TCP    127.0.0.1:88990        0.0.0.0:0              LISTENING       9999
  TCP    127.0.0.1:8899         127.0.0.1:57111        CLOSE_WAIT      7777
  TCP    127.0.0.1:57000        127.0.0.1:8899         TIME_WAIT       0
  TCP    [::1]:8899             [::]:0                 LISTENING       4211
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       880
"""


def _no_kill_calls() -> tuple[list[int], object]:
    calls: list[int] = []

    def kill(pid: int) -> bool:
        calls.append(pid)
        return True

    return calls, kill


def _netstat_seq(*texts: str):
    """按顺序吐出 netstat 输出；用完之后一直重复最后一份。"""
    remaining = list(texts)

    def fn() -> str:
        return remaining.pop(0) if len(remaining) > 1 else remaining[0]

    return fn


# ── 判据一：谁在听 ────────────────────────────────────────────


def test_only_listening_rows_on_the_exact_port() -> None:
    """停在 CLOSE_WAIT / ESTABLISHED 上的同端口连接不算占用。

    这里必须让**非 LISTENING 行换一个 PID**（7777）：如果那条行和监听者是同一个
    PID，去掉 LISTENING 判据后去重会把差别吃掉，测试就变成恒真。
    """
    text = "\n".join(
        [
            "  TCP    0.0.0.0:8899           0.0.0.0:0              LISTENING       4211",
            "  TCP    127.0.0.1:8899         127.0.0.1:57111        CLOSE_WAIT      7777",
            "  TCP    127.0.0.1:8899         127.0.0.1:57000        ESTABLISHED     8888",
        ]
    )
    assert parse_listening_pids(text, 8899) == [4211]


def test_longer_port_is_not_a_prefix_match() -> None:
    """88990 那一行不能被判成 8899：按端口号做 `in` 判断会在这里翻车。"""
    text = "\n".join(
        [
            "  TCP    0.0.0.0:8899           0.0.0.0:0              LISTENING       4211",
            "  TCP    127.0.0.1:88990        0.0.0.0:0              LISTENING       9999",
        ]
    )
    assert parse_listening_pids(text, 8899) == [4211]
    assert parse_listening_pids(text, 88990) == [9999]


def test_ipv6_listener_is_deduped_with_ipv4() -> None:
    """同一进程同时监听 v4/v6 时只报一次，否则会对着同一个 PID 杀两次。"""
    pids = parse_listening_pids(NETSTAT, 8899)
    assert pids.count(4211) == 1


def test_garbage_and_udp_lines_are_skipped() -> None:
    text = "\n".join(
        [
            "活动连接",
            "",
            "  UDP    0.0.0.0:8899           *:*                                    1234",
            "  TCP    127.0.0.1:8899         0.0.0.0:0              LISTENING       abc",
            "  TCP    ::1",
        ]
    )
    assert parse_listening_pids(text, 8899) == []


# ── 判据二：是不是本程序 ──────────────────────────────────────


class _Handler(BaseHTTPRequestHandler):
    body = b"{}"
    status = 200

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 的接口
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args) -> None:
        pass


def _serve(body: bytes, status: int = 200):
    handler = type("H", (_Handler,), {"body": body, "status": status})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def test_probe_recognises_its_own_health_payload() -> None:
    server, port = _serve(b'{"app": "rag4prefa", "ok": true}')
    try:
        assert probe_app(port) is True
    finally:
        server.shutdown()
        server.server_close()


def test_probe_reports_false_when_someone_else_answers() -> None:
    """答了但不是我们——这是"别杀它"的信号，不能退化成 None。"""
    server, port = _serve(b'{"app": "someone-else"}')
    try:
        assert probe_app(port) is False
    finally:
        server.shutdown()
        server.server_close()


def test_probe_reports_unknown_for_non_json_and_dead_port() -> None:
    # 返回 404 的 HTML、端口压根没开，都必须落到 None（无法确认），
    # 不能当成 False——那会让"服务还在启动中"永远停不掉。
    server, port = _serve(b"<html>not found</html>", status=404)
    try:
        assert probe_app(port) is None
    finally:
        server.shutdown()
        server.server_close()

    assert probe_app(1, timeout=0.3) is None


def test_probe_treats_missing_app_field_as_unknown_not_foreign() -> None:
    """旧版本的 /api/health 没有 app 字段：没有身份声明 ≠ 自称是别人。

    判成 False 的话，用户把新代码装好、旧进程还占着端口时会被 stop 拒绝，
    只能去手动 taskkill——正好是这条命令想避免的事。
    """
    server, port = _serve(b'{"ok": true, "index": {"chunks": 66}}')
    try:
        assert probe_app(port) is None
    finally:
        server.shutdown()
        server.server_close()


# ── 组合：敢不敢动手 ──────────────────────────────────────────


def _stop(**kwargs) -> StopResult:
    kwargs.setdefault("netstat", lambda: NETSTAT)
    kwargs.setdefault("name", lambda pid: "python.exe")
    kwargs.setdefault("wait", 0.0)
    return stop_port(8899, **kwargs)


def test_nothing_listening_is_not_an_error() -> None:
    calls, kill = _no_kill_calls()
    result = _stop(netstat=lambda: "", kill=kill)
    assert result.status == "unused"
    assert result.exit_code == 0
    assert calls == []


def test_refuses_to_kill_a_foreign_service() -> None:
    calls, kill = _no_kill_calls()
    result = _stop(probe=lambda port: False, kill=kill)
    assert result.status == "refused"
    assert result.exit_code == 1
    assert calls == [], "认不出身份还去杀，等于按端口号胡乱清理别人的进程"
    assert "PID 4211" in result.message
    assert "start.bat 8900" in result.message
    # 人家已经答"我不是"，就不该再递一把 /force 去劝用户误杀
    assert "/force" not in result.message


def test_unknown_identity_refuses_but_points_at_force() -> None:
    calls, kill = _no_kill_calls()
    result = _stop(probe=lambda port: None, kill=kill)
    assert result.status == "refused"
    assert calls == []
    assert "8899 /force" in result.message


def test_kills_the_whole_tree_when_identity_is_proven() -> None:
    calls, kill = _no_kill_calls()
    # 第一次探到监听，杀完再探已经空
    result = _stop(probe=lambda port: True, kill=kill, netstat=_netstat_seq(NETSTAT, ""))
    assert result.status == "stopped"
    assert calls == [4211]
    assert "已确认是本程序" in result.message
    assert result.exit_code == 0


def test_force_skips_identity_but_says_so() -> None:
    calls, kill = _no_kill_calls()
    result = _stop(
        probe=lambda port: None,
        kill=kill,
        force=True,
        netstat=_netstat_seq(NETSTAT, ""),
    )
    assert calls == [4211]
    assert "未做身份确认" in result.message, "强制停必须把'没验证'写在结果里"


def test_port_still_listening_after_kill_is_a_failure() -> None:
    calls, kill = _no_kill_calls()
    result = _stop(probe=lambda port: True, kill=kill)  # netstat 一直报同一个 PID
    assert result.status == "failed"
    assert result.exit_code == 1
    assert "仍在监听" in result.message
    assert "taskkill /PID 4211 /T /F" in result.message, "失败时要给出能直接粘的命令"


def test_marker_matches_the_health_field() -> None:
    """`probe_app` 比的是 `app` 字段的字面值，改了 health 就必须同步改这里。"""
    assert APP_MARKER == "rag4prefa"
