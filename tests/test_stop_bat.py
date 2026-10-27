"""`stop.bat` 的契约：能停、会拒绝、退出码传得出去。

和 `start.bat` 同一类风险——脚本语法正确、能跑完、退出码还是 0，所以"文件里出现
taskkill"这种断言恒真。真正有鉴别力的是三件事：

1. `/force` 必须透传到 `rag4prefa stop`：健康检查没回应（服务卡死）时，这是唯一
   出路；bat 吞掉参数等于把这个出口关掉了。
2. 退出码要传出去，而且必须在 `endlocal` **同一行**展开——`endlocal` 之后再取
   `%RC%` 已经是空的了。
3. 每条结束路径都 `pause`：双击运行时窗口一闪而过等于没有输出。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOP = ROOT / "stop.bat"
START = ROOT / "start.bat"


def _text(path: Path) -> str:
    return path.read_bytes().decode("gbk")


def _section(text: str, label: str) -> str:
    out: list[str] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r":[a-z_]+", stripped):
            if inside:
                break
            inside = stripped == f":{label}"
            continue
        if inside:
            out.append(line)
    return "\n".join(out)


def test_stop_bat_is_gbk_and_crlf() -> None:
    raw = STOP.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "bat 不能带 UTF-8 BOM"
    assert raw.decode("gbk"), "bat 必须能被 GBK 解码（cmd.exe 按系统代码页解析）"
    assert raw.count(b"\n") == raw.count(b"\r\n"), "bat 必须全 CRLF"


def test_force_is_forwarded() -> None:
    text = _text(STOP)
    # 两种写法都认，别指望用户记得住是 /force 还是 --force
    assert '"/force"' in text and '"--force"' in text
    assert "set \"FORCE=--force\"" in text
    # 参数要真的拼进命令行里
    assert "rag4prefa stop --port %PORT% %FORCE%" in text


def test_exit_code_survives_endlocal() -> None:
    bail = _section(_text(STOP), "bail")
    assert 'set "RC=%ERRORLEVEL%"' in _text(STOP), "没有接住 stop 子命令的返回码"
    assert "endlocal & exit /b %RC%" in bail, "endlocal 之后再取 %RC% 会是空的"
    assert "pause" in bail, "结束路径必须停在窗口里"


def test_missing_venv_falls_back_to_system_commands() -> None:
    """虚拟环境都没了还得能停服：给出 netstat + taskkill 两条现成命令。"""
    text = _text(STOP)
    assert 'if not exist ".venv\\Scripts\\python.exe"' in text
    assert "taskkill /PID" in text


def test_start_bat_points_at_stop_bat_with_a_pid() -> None:
    """占位提示必须能回答"谁占着、怎么停"，否则用户只能猜。"""
    taken = _section(_text(START), "port_taken")
    assert "call :listening_pid" in taken, "没有取占用者 PID"
    assert "%OWNER%" in taken
    assert "tasklist /FI" in taken, "没把进程名打出来，用户认不出是不是本程序"
    assert "stop.bat" in taken, "没有指向真正的停止入口"
    assert "Ctrl+C" in taken
