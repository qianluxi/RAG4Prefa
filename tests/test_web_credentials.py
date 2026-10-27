"""前端「保存并验证」的判据必须覆盖**三个**凭据角色。

这条用例之所以存在，是因为一个只在浏览器里才看得见的失败：`saveCreds()` 原来判的是
`!result.embedding.ok && !result.llm.ok`，漏掉了 `ocr`。于是"只填了视觉模型 Key、
另两栏留空"的人会看到三项验证结果、其中一项通过，却被判定为"未保存"——什么都没存下来。

它不属于后端契约，但同样会让人以为系统坏了，而且**没有别的用例能看见它**：
单测跑的是 Python，`node --check` 只查语法，语法完全正确。

`tests/test_web_markdown.py` 用标记块抽取要测的函数；这里按函数名抽取，避免为了
一条断言往页面上加标记。
"""

from __future__ import annotations

from pathlib import Path

HTML = Path(__file__).resolve().parent.parent / "rag4prefa" / "web" / "index.html"


def _save_body() -> str:
    html = HTML.read_text(encoding="utf-8")
    start = html.index("async function saveCreds()")
    end = html.index("function clearCreds()", start)
    return html[start:end]


def test_save_judges_all_three_credential_roles() -> None:
    """保存判据必须把 ocr 算进去：有一项通过就该存下来。

    这不是"宽松"，是凭据本来就是**分角色**的：扫描件只要视觉模型 Key 加 embedding，
    问答只要 embedding 加 LLM。任何一项通过都说明"这个用户名下至少能用一件事"，
    挡着不存只会让用户重复填同一把 Key。
    """
    body = _save_body()
    guards = [line for line in body.splitlines() if "const checks" in line]
    assert guards, "保存判据被挪走或改名了，这条用例需要跟着改"
    guard = guards[0]
    for role in ("result.embedding", "result.llm", "result.ocr"):
        assert role in guard, f"判据漏了 {role}：只填这一项的人会验证通过却保存不了"


def test_save_still_refuses_when_nothing_works() -> None:
    """三项全不通时仍然不保存——存下来只会让之后每次请求都 401。"""
    body = _save_body()
    assert "some(" in body, "判据必须仍是「至少一项通过」，不能退化成「全通过」"
    assert "验证未通过，未保存" in body


def test_saved_message_names_the_missing_roles() -> None:
    """保存成功时要把"还没通过的那几项"点出来。

    只说"已保存"，用户会以为配全了，然后在提问撞上 401 的那一刻才发现另一栏是空的。
    报告缺项是这一步的职责，因为只有这里同时知道三项的结果。

    断言要落在"它真的进了提示语"上：只声明一个 `lackNote` 变量、却没人拼进
    `textContent`，用户一个字也看不见，而"变量存在"这种断言照样通过——
    第一版就是这么写错的，变异一测就露了。
    """
    body = _save_body()
    lines = body.splitlines()
    assert any("lacks.length" in line for line in lines), "缺项汇总退化成常量或没算"
    assert any("lacks.join" in line for line in lines), "缺项名字没被拼进说明"
    assert any(
        "已保存并切换为" in line and "${lackNote}" in line for line in lines
    ), "算出来的缺项没拼进保存成功的提示语"
    for role in ("Embedding", "LLM", "视觉模型"):
        assert role in body, f"缺项提示里没提 {role}"
