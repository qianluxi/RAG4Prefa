"""停服：从 netstat 认 PID、从 `/api/health` 认身份，再决定杀不杀。

为什么需要它：点黑窗口右上角的 X 只保证 cmd 没了，Python 进程树不一定跟着退
（venv 里的 `python.exe` 是启动器，真正监听端口的是它拉起的子进程）。端口没释放，
再双击 `start.bat` 就只剩一句"已经在监听"——用户看到的是"我明明关了"，实际是
"关窗口"在 Windows 上本来就杀不干净进程树，得显式 `taskkill /T`。

两条判据，缺一不可：

1. **谁在听** —— `netstat -ano` 里 state 为 `LISTENING`、本地地址端口等于目标端口的
   PID。`TIME_WAIT` / `ESTABLISHED` 都不算，它们不占监听端口；停服后立刻再启会看到
   一堆 TIME_WAIT，那是客户端残留，不是"没关干净"。
2. **是不是本程序** —— 连 `/api/health` 看 `app` 字段。只按端口杀等于"谁占着 8899
   就杀谁"，误杀别人家的服务只需要一次巧合，而误杀的代价是人家丢了状态。

身份问不出来（服务卡死、启动到一半）时**不动手**，把 PID 和命令打给用户，让他用
`--force` 显式确认。判据可以放宽，但不能悄悄放宽。
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

# `/api/health` 里的自报身份。比对它是为了回答"端口上这个进程是不是我们家的"，
# 不依赖进程名（用户可能把 python.exe 改名、也可能同时开着别的 Python 服务）。
APP_MARKER = "rag4prefa"

NetstatFn = Callable[[], str]
ProbeFn = Callable[[int], "bool | None"]
KillFn = Callable[[int], bool]
NameFn = Callable[[int], "str | None"]


@dataclass(frozen=True)
class StopResult:
    """`status` 只有四种：unused（没在跑）、refused（认不出身份，没动）、
    stopped（已停且端口已释放）、failed（杀了但端口还在听）。"""

    status: str
    pids: tuple[int, ...] = ()
    message: str = ""

    @property
    def exit_code(self) -> int:
        # 只有"没在跑"和"停干净了"算成功。refused 是拒绝执行，脚本据此能区分
        # "已经没什么可做"与"你想让我做的事我没敢做"。
        return 0 if self.status in {"unused", "stopped"} else 1


def parse_listening_pids(netstat_text: str, port: int) -> list[int]:
    """从 `netstat -ano` 的输出里挑出监听 `port` 的 PID（去重，保持出现顺序）。

    用 `endswith(":8899")` 而不是 `in`：`127.0.0.1:88899` 的尾部是 `:88899`？不是——
    `endswith(":8899")` 要求倒数第五个字符是冒号，所以 88899 不会命中 8899。
    IPv6 的 `[::1]:8899` 同样以 `:8899` 结尾，一并匹配。
    """
    suffix = f":{port}"
    pids: list[int] = []
    for line in netstat_text.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[3] != "LISTENING":
            continue
        if not parts[1].endswith(suffix):
            continue
        try:
            pid = int(parts[4])
        except ValueError:
            continue
        if pid > 0 and pid not in pids:
            pids.append(pid)
    return pids


def probe_app(port: int, timeout: float = 2.0) -> bool | None:
    """问一句"你是谁"。

    True/False 是明确答复；None 表示问不到（没起、超时、回的不是 JSON）。
    三种情况必须分开：把 None 当 False 会让"卡死的本程序"永远停不掉，
    把 None 当 True 又会去杀一个根本不知道自己占着端口的进程。

    "JSON 里没有 app 字段"（旧版本的服务）算 None 而不是 False：没有身份声明
    不等于自称是别人，硬判成"不是本程序"会让用户升级到一半时停不掉旧进程。
    """
    url = f"http://127.0.0.1:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    app = payload.get("app")
    if not isinstance(app, str) or not app:
        return None
    return app == APP_MARKER


def _netstat_tcp() -> str:
    # 中文 Windows 下 netstat/taskkill 的输出是 GBK，用 replace 兜住乱码，
    # 反正我们只取 ASCII 的地址、状态与 PID 列。
    proc = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True)
    return proc.stdout.decode("gbk", errors="replace")


def _taskkill(pid: int) -> bool:
    # /T 连子进程一起：启动器与真正的服务是两级，只杀一级会留下另一半。
    proc = subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True
    )
    return proc.returncode == 0


def _process_name(pid: int) -> str | None:
    """尽力而为地取进程名，只用于提示里让人一眼认出"这不是本程序"。"""
    proc = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True
    )
    text = proc.stdout.decode("gbk", errors="replace").strip()
    if not text:
        return None
    first = text.splitlines()[0]
    if not first.startswith('"'):
        return None  # "信息: 没有运行的任务..." 之类
    return first.split('","')[0].strip('"')


def stop_port(
    port: int,
    *,
    force: bool = False,
    wait: float = 5.0,
    netstat: NetstatFn | None = None,
    probe: ProbeFn | None = None,
    kill: KillFn | None = None,
    name: NameFn | None = None,
) -> StopResult:
    """停掉监听 `port` 的本程序。依赖全部可注入，测试不需要真起服务。"""
    netstat = netstat or _netstat_tcp
    probe = probe or probe_app
    kill = kill or _taskkill
    name = name or _process_name

    pids = parse_listening_pids(netstat(), port)
    if not pids:
        return StopResult("unused", (), f"端口 {port} 没有程序在监听，不用停。")

    proven = probe(port)
    if proven is not True and not force:
        detail = "、".join(f"PID {pid}（{name(pid) or '进程名取不到'}）" for pid in pids)
        if proven is None:
            why = "/api/health 没有回应，无法确认它是不是本程序（可能还在启动，或已经卡死）"
            # 问不到才可能是"卡死的本程序"，这种情况才引导用户去 --force；
            # 人家答了"我不是"，还递一把 force 就等于劝人误杀。
            hint = f"  确实就是本程序、只是健康检查没回应 → stop.bat {port} /force\n"
        else:
            why = "它回应了 /api/health，但自报的身份不是 RAG4Prefa"
            hint = ""
        return StopResult(
            "refused",
            tuple(pids),
            f"端口 {port} 被 {detail}占着，没有停。\n"
            f"原因：{why}。\n"
            f"{hint}"
            f"  其实是别的软件 → 别杀它，换个端口：start.bat {port + 1}\n"
            f"  想自己看：netstat -ano | findstr \":{port}\"",
        )

    killed = [pid for pid in pids if kill(pid)]

    # 杀完再验端口：taskkill 返回 0 只说明"发了终止信号"，子进程可能还在退出中。
    attempts = max(1, round(wait / 0.2))
    left = parse_listening_pids(netstat(), port)
    for _ in range(attempts - 1):
        if not left:
            break
        time.sleep(0.2)
        left = parse_listening_pids(netstat(), port)

    if left:
        return StopResult(
            "failed",
            tuple(pids),
            f"发出终止后端口 {port} 仍在监听：PID {'、'.join(map(str, left))}。\n"
            "  可能是权限不够（用管理员身份再试一次），或进程卡在不可中断状态：\n"
            f"      taskkill /PID {left[0]} /T /F",
        )

    how = "已确认是本程序" if proven is True else "按 --force 未做身份确认"
    note = "" if killed == list(pids) else f"（其中 {len(pids) - len(killed)} 个杀失败了）"
    return StopResult(
        "stopped",
        tuple(pids),
        f"已停止{note}：{how}，PID {'、'.join(map(str, pids))}。\n"
        f"  端口 {port} 已释放，可以重新双击 start.bat。\n"
        "  正在进行的入库会被打断，重启后任务标为「已中断」，重跑一次即可。",
    )
