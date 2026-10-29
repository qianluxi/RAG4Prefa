"""报告身份：工作区脏的时候，`commit` 是猜的，源码指纹才是事实。

背景（2026-10-29 专家复审 P0-2）：20 份报告都写着"代码 `xxx`（有未提交改动）"——
等于这 20 份的代码身份是空的。`rev-parse` 只说明"基于哪个提交"，脏工作区里它
描述的不是跑出这组数字的那份代码。

补的是**内容哈希**，不是 `git diff`。diff 只说"跟某个提交比变过"，不说"变成什么"；
两份报告都标着"有改动"时，diff 救不了你——它们的改动可能压根不是同一处。
内容哈希可判等：同 `src_sha` ⇒ 影响数字的源码逐字节相同。

这个文件钉住的正是"可判等"这件事：稳定、对内容敏感、对无关改动不敏感。
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_script_identity", ROOT / "scripts" / "eval.py")
assert _spec and _spec.loader
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_same_content_gives_same_sha(tmp_path: Path) -> None:
    """内容相同 ⇒ 指纹相同。这是"两份报告能不能比"唯一的判据。"""
    a, b = tmp_path / "a", tmp_path / "b"
    _write(a, "rag4prefa/ask.py", "def f(): return 1\n")
    _write(b, "rag4prefa/ask.py", "def f(): return 1\n")
    # b 里的文件是后来写的（mtime 不同）——指纹不该因此变化，否则换台机器就对不上了。
    sha_a, n_a = ev._fingerprint_files(a, ["rag4prefa/ask.py"])
    sha_b, n_b = ev._fingerprint_files(b, ["rag4prefa/ask.py"])
    assert (sha_a, n_a) == (sha_b, n_b)
    assert n_a == 1


def test_one_byte_change_changes_sha(tmp_path: Path) -> None:
    """改一个字符就要变——否则"同一份代码"这句话是自我安慰。"""
    _write(tmp_path, "rag4prefa/ask.py", "def f(): return 1\n")
    before, _ = ev._fingerprint_files(tmp_path, ["rag4prefa/ask.py"])
    _write(tmp_path, "rag4prefa/ask.py", "def f(): return 2\n")
    after, _ = ev._fingerprint_files(tmp_path, ["rag4prefa/ask.py"])
    assert before != after


def test_irrelevant_files_are_ignored(tmp_path: Path) -> None:
    """文档与前端不算数：改 README 不该让上一份报告显得不可比。

    指纹一旦对无关改动敏感，人就会开始无视它——那它还不如不记。
    """
    _write(tmp_path, "rag4prefa/ask.py", "x = 1\n")
    _write(tmp_path, "README.md", "第一版\n")
    _write(tmp_path, "rag4prefa/web/index.html", "<html></html>")
    before, n_before = ev._fingerprint_files(
        tmp_path, ["rag4prefa/ask.py", "README.md", "rag4prefa/web/index.html"]
    )
    _write(tmp_path, "README.md", "第二版，改了很多字\n")
    _write(tmp_path, "rag4prefa/web/index.html", "<html><body>改了</body></html>")
    after, n_after = ev._fingerprint_files(
        tmp_path, ["rag4prefa/ask.py", "README.md", "rag4prefa/web/index.html"]
    )
    assert before == after
    assert (n_before, n_after) == (1, 1), "只有 .py/.json 该被计入"


def test_term_json_counts(tmp_path: Path) -> None:
    """词表 JSON 必须计入。

    报告名里只有词表的**文件名**（`-termmap-public_to_standard.v2`）：原地改几条映射，
    文件名一个字都不变——不把内容算进指纹，这种改动就是隐形的。
    """
    _write(tmp_path, "rag4prefa/terms/map.json", '{"entries": []}\n')
    before, n = ev._fingerprint_files(tmp_path, ["rag4prefa/terms/map.json"])
    _write(tmp_path, "rag4prefa/terms/map.json", '{"entries": [{"id": "x"}]}\n')
    after, _ = ev._fingerprint_files(tmp_path, ["rag4prefa/terms/map.json"])
    assert n == 1
    assert before != after


def test_path_is_part_of_the_identity(tmp_path: Path) -> None:
    """同样的一串内容换个文件名，指纹必须不同。

    只哈希内容不哈希路径，`ask.py` 与 `repo.py` 对调就变成"同一份代码"——而 SQL 全在
    repo.py 里，那种对调恰恰最要命。用"改名"这个最小形态来钉：**喂给哈希的内容序列
    逐字节相同**，只有路径不同，所以这条用例能区分"路径进没进哈希"，而"互换两个文件"
    不能（互换后内容顺序也跟着变了，去不去的都会得到不同结果）。
    """
    _write(tmp_path, "rag4prefa/a.py", "A\n")
    _write(tmp_path, "rag4prefa/b.py", "B\n")
    before, _ = ev._fingerprint_files(tmp_path, ["rag4prefa/a.py", "rag4prefa/b.py"])
    _write(tmp_path, "rag4prefa/c.py", "B\n")
    after, _ = ev._fingerprint_files(tmp_path, ["rag4prefa/a.py", "rag4prefa/c.py"])
    assert before != after


def test_missing_file_does_not_raise(tmp_path: Path) -> None:
    """`git ls-files` 列了但盘上没有：跳过，不给评估添一个新失败点。"""
    _write(tmp_path, "rag4prefa/ask.py", "x = 1\n")
    sha, count = ev._fingerprint_files(tmp_path, ["rag4prefa/ask.py", "rag4prefa/gone.py"])
    assert count == 1 and len(sha) == 12


def test_dirty_state_keeps_full_file_name(tmp_path: Path) -> None:
    """脏文件名不能被截成 `cripts/eval.py`。

    `git status --porcelain` 的首行是 ` M scripts/eval.py`（状态位前有个空格），
    对整段输出 `.strip()` 会把它吃掉，之后按固定列切路径就少一个字。这个 bug
    以前没人看见，是因为没人用过这些文件名。
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True,
            env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
        )

    git("init", "-q")
    (repo / "scripts").mkdir()
    (repo / "rag4prefa").mkdir()
    (repo / "scripts" / "eval.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "rag4prefa" / "ask.py").write_text("y = 1\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    (repo / "scripts" / "eval.py").write_text("x = 2\n", encoding="utf-8")

    original = ev.ROOT
    ev.ROOT = repo
    try:
        state = ev._code_state()
    finally:
        ev.ROOT = original

    assert state["dirty"] is True
    assert state["dirty_files"] == ["scripts/eval.py"]
    assert state["dirty_count"] == 1


def test_line_endings_do_not_change_sha(tmp_path: Path) -> None:
    """LF 与 CRLF 必须同指纹。

    本仓库 `.gitattributes` 写的是 `eol=lf`，但 Windows 上 `core.autocrlf=true` 会把
    工作区落成 CRLF（实测 `scripts/eval.py` 1445 行全是 CRLF）。不归一化，同一份代码
    在 Linux 上算出另一个哈希——"报告里的 sha 可被复算"这句话当场失效，而它正是加
    指纹的全部理由。
    """
    _write(tmp_path, "rag4prefa/ask.py", "a = 1\nb = 2\n")
    (tmp_path / "rag4prefa" / "ask.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    crlf, _ = ev._fingerprint_files(tmp_path, ["rag4prefa/ask.py"])
    (tmp_path / "rag4prefa" / "ask.py").write_bytes(b"a = 1\nb = 2\n")
    lf, _ = ev._fingerprint_files(tmp_path, ["rag4prefa/ask.py"])
    assert crlf == lf


def test_real_repo_records_source_sha() -> None:
    """本仓库里必须真的给出指纹（12 位十六进制），否则报告上那行永远是空的。"""
    state = ev._code_state()
    assert state.get("commit"), "拿不到提交号"
    assert len(state.get("src_sha", "")) == 12
    assert isinstance(state.get("src_files"), int) and state["src_files"] > 10


def test_report_renders_source_sha_and_names() -> None:
    """渲染层：赋值了不等于印出来了（这个坑本项目踩过一次）。"""
    result = {
        "samples": 3,
        "modes": {"dense": {"recall": [1.0] * 3, "mrr": [1.0] * 3, "ndcg": [1.0] * 3}},
        "evaluated": {"dense": 3},
        "skipped": [],
        "warnings": [],
        "context": {},
        "generated_at": "2026-10-29T12:00:00+08:00",
        "code": {
            "commit": "7311c41",
            "dirty": True,
            "dirty_files": ["rag4prefa/ask.py", "rag4prefa/config.py"],
            "dirty_count": 2,
            "src_sha": "601f6c2411f3",
            "src_files": 39,
        },
    }
    text = ev.report(result, 5)
    assert "源码 sha `601f6c2411f3`" in text
    assert "39 个文件" in text
    assert "rag4prefa/ask.py" in text and "rag4prefa/config.py" in text, "脏在哪要能看见"


def test_report_truncates_long_dirty_list() -> None:
    """脏文件多的时候只列前几个 + 总数，否则报告头会长到没人读。"""
    files = [f"rag4prefa/m{i}.py" for i in range(9)]
    result = {
        "samples": 1,
        "modes": {"dense": {"recall": [1.0], "mrr": [1.0], "ndcg": [1.0]}},
        "evaluated": {"dense": 1},
        "skipped": [],
        "warnings": [],
        "context": {},
        "generated_at": "2026-10-29T12:00:00+08:00",
        "code": {"commit": "abc1234", "dirty": True, "dirty_files": files, "dirty_count": 9},
    }
    text = ev.report(result, 5)
    assert "等 9 个文件" in text
    assert "m8.py" not in text, "第九个不该被列出来"
