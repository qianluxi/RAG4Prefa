"""`start.bat` 的启动契约。

2026-10-26 现场反馈"双击 bat 没启动成功、localhost 拒绝连接"，实际原因是脚本
**先 echo 地址、再 serve**：端口被占时 uvicorn 报 `[Errno 10048]`，脚本走到
`:end` 没有 `pause`，黑窗口立刻关掉。用户只看到闪过一行地址，浏览器打开当然
拒绝连接。

这类缺陷的特点是脚本语法正确、能跑完、退出码还是 0，所以"文件里含 http://"
这种断言恒真、抓不住。真正有鉴别力的是两条**顺序 / 覆盖**断言：

1. 打印地址只在端口预检通过之后（否则地址是"打算用"，不是"已就绪"）；
2. 每条失败路径都必须 `pause`（否则窗口一闪而过，等于没有报错）。

另外编码是硬约束：`cmd.exe` 按系统代码页读 `.bat`，UTF-8 的中文会把命令字符
吃掉（`:fail` 变成 `ail`），所以这里也钉住 GBK + CRLF。
"""

from __future__ import annotations

import re
from pathlib import Path

BAT = Path(__file__).resolve().parent.parent / "start.bat"


def _text() -> str:
    return BAT.read_bytes().decode("gbk")


def _section(text: str, label: str) -> str:
    """取出 `:label` 到下一个顶格标签之间的脚本行。"""
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
    assert inside or out, f"脚本里找不到标签 :{label}"
    return "\n".join(out)


def test_bat_is_gbk_and_crlf() -> None:
    raw = BAT.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "bat 不能带 UTF-8 BOM"
    assert raw.decode("gbk"), "bat 必须能被 GBK 解码（cmd.exe 按系统代码页解析）"
    assert raw.count(b"\n") == raw.count(b"\r\n"), "bat 必须全 CRLF，混用 LF 会让 cmd 解析错位"


def test_address_printed_only_after_port_probe() -> None:
    """地址只能在端口预检通过后打印——这是本文件存在的首要理由。"""
    text = _text()
    # 预检必须在主流程里、且在跳进 :ready 之前
    probe = text.index("call :port_listening")
    ready = text.index(":ready")
    assert probe < ready, "端口预检必须出现在 :ready 之前"

    # 成功分支里的地址位于 :ready 段，因此天然在预检之后
    ready_body = _section(text, "ready")
    assert "http://%HOST%:%PORT%" in ready_body

    # 端口被占用时也要给出地址：那是"可能已经开着"，不是"本次启动成功"
    taken = _section(text, "port_taken")
    assert "http://%HOST%:%PORT%" in taken
    assert "已经在监听" in taken


def test_port_probe_only_counts_listening() -> None:
    """只看 LISTENING：TIME_WAIT / 已建立连接都不算端口被占用。"""
    body = _section(_text(), "port_listening")
    assert "netstat" in body
    assert "LISTENING" in body
    assert "-p TCP" in body
    # 本地地址后带空格，避免把 88990 误判成 8899
    assert ':%%PORT% ' in body or ':%PORT% ' in body


def test_every_failure_path_pauses() -> None:
    """失败必须停窗。窗口一闪而过等于没有报错。"""
    text = _text()
    assert "pause" in _section(text, "bail"), ":bail 分支没有 pause"
    assert "pause" in _section(text, "port_taken"), ":port_taken 分支没有 pause"

    # serve 之后的异常出口：先取返回码，再判 Ctrl+C，最后 pause
    # 从 serve 那一行切到 :port_taken 标签本体（用 rindex：前面还有一条 goto 引用它）
    tail = text[text.index("-m rag4prefa serve") : text.rindex("\n:port_taken")]
    assert 'set "RC=%ERRORLEVEL%"' in tail, "没有接住 serve 的返回码"
    assert "-1073741510" in tail, "没有把 Ctrl+C 中断和真失败区分开"
    assert "pause" in tail, "serve 异常退出后没有 pause"


def test_no_unreachable_url_in_header() -> None:
    """横幅只报名字，不报地址——避免"看到地址就以为起来了"。"""
    head = _text().split(":ready")[0]
    banner = head.split("cd /d")[0]
    assert "http://" not in banner
