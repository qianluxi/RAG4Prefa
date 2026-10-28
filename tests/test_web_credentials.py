"""前端凭据逻辑的**行为**测试：在 Node 里真跑一遍，而不是拿正则匹配源码。

为什么不用纯 Python 断言"代码里有这个词"：真正要守的是**用户会得到什么**——填错的 Key
到底会不会被发出去、清空之后那句提示是不是假话。第一版就是源码结构断言，复审指出这类
断言会在重构后失真（2026-10-28 复审 §4）。

做法：`index.html` 里那几条判据包在 `CRED-LOGIC-START/END` 之间，抽出来拼成一个函数体，
用 Node 执行、结果打成 JSON，再由 pytest 断言。没装 Node 就整段 skip——其余用例仍离线可跑。

覆盖五件事：
1. 哪些角色算"启用"（`verified[role] !== false`；旧数据没有 `verified` 时按启用处理）；
2. 未启用的 Key **不进请求头**（P2-2：验证失败后仍无差别发送，会把问题拖进异步任务）；
3. 清空后的提示跟着部署走（P2-1：各付各的部署里"使用服务端公共 Key"是假话）；
4. 保存判据"三项至少一项通过"（既不能漏角色，也不能退化成"全通过"）；
5. `saveCreds()` 确实把 `verified` 写进了 localStorage 并走了这条判据——接线断了的话，
   上面几条全退化成"永远启用 / 永远通过"，测试仍然是绿的。
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parent.parent / "rag4prefa" / "web" / "index.html"
START = "/* CRED-LOGIC-START */"
END = "/* CRED-LOGIC-END */"

node = shutil.which("node")

_HARNESS = """
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const names = [
  "credentialHeaders", "roleEnabled", "pendingRoles", "shouldSave",
  "clearedMessage", "noUserMessage", "adminHeader", "CRED_ROLE_LABELS", "CRED_ROLES",
];
const api = new Function(src + "\\nreturn {" + names.join(", ") + "};")();
const ok = (value) => ({ ok: value });

// 一把 Key 通过、另一把填错的典型情形：错的不能跟着发出去
const mixed = {
  embeddingKey: "sk-bad", llmKey: "sk-ok", llmModel: "m1", ocrKey: "sk-ocr",
  verified: { embedding: false, llm: true, ocr: true },
};
// 视觉模型这一项没通过：它同样不该发
const ocrBad = {
  embeddingKey: "sk-emb", llmKey: "sk-llm", ocrKey: "sk-bad-ocr", llmModel: "",
  verified: { embedding: true, llm: true, ocr: false },
};
// 老版本存下来的用户没有 verified 字段
const legacy = { embeddingKey: "sk-a", llmKey: "sk-b", ocrKey: "" };
const allBad = {
  embeddingKey: "a", llmKey: "b", ocrKey: "c",
  verified: { embedding: false, llm: false, ocr: false },
};

console.log(JSON.stringify({
  roles: api.CRED_ROLES,
  mixed_headers: api.credentialHeaders(mixed),
  mixed_pending: api.pendingRoles(mixed),
  ocr_bad_headers: api.credentialHeaders(ocrBad),
  legacy_headers: api.credentialHeaders(legacy),
  legacy_pending: api.pendingRoles(legacy),
  all_bad_headers: api.credentialHeaders(allBad),
  no_creds_headers: api.credentialHeaders(null),
  save_only_ocr_ok: api.shouldSave({ embedding: ok(false), llm: ok(false), ocr: ok(true) }),
  save_none_ok: api.shouldSave({ embedding: ok(false), llm: ok(false), ocr: ok(false) }),
  save_two_ok: api.shouldSave({ embedding: ok(true), llm: ok(false), ocr: ok(true) }),
  save_without_ocr_field: api.shouldSave({ embedding: ok(true), llm: ok(true) }),
  admin_header_empty: api.adminHeader(""),
  admin_header_set: api.adminHeader("s3cret"),
  cleared_strict: api.clearedMessage("张三", false),
  cleared_demo: api.clearedMessage("张三", true),
  nouser_strict: api.noUserMessage(false),
  nouser_demo: api.noUserMessage(true),
}));
"""


@pytest.fixture(scope="session")
def behavior(tmp_path_factory: pytest.TempPathFactory) -> dict:
    if node is None:
        pytest.skip("没有 node，跳过前端行为用例（node --check 同样需要它）")
    html = HTML.read_text(encoding="utf-8")
    start = html.index(START)
    end = html.index(END, start)
    block = html[start + len(START) : end]
    tmp = tmp_path_factory.mktemp("cred")
    (tmp / "block.js").write_text(block, encoding="utf-8")
    (tmp / "harness.js").write_text(_HARNESS, encoding="utf-8")
    proc = subprocess.run(
        [str(node), str(tmp / "harness.js"), str(tmp / "block.js")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"Node 执行失败：{proc.stderr}"
    return json.loads(proc.stdout)


def test_unverified_key_never_leaves_the_browser(behavior: dict) -> None:
    """验证失败的 Key 不能随请求发出去（P2-2）。

    让一把已知错误的 Key 出门，结果是入库这种**异步**任务跑到一半失败，报错还来自
    服务商；用户刚刚才在设置里看到"这一项没通过"，这里却又把它送出去了。
    """
    headers = behavior["mixed_headers"]
    assert "X-R4P-Embedding-Key" not in headers, "没通过的 embedding Key 被发出去了"
    assert headers["X-R4P-LLM-Key"] == "sk-ok", "通过的那把必须照常发"
    assert headers["X-R4P-OCR-Key"] == "sk-ocr"
    assert headers["X-R4P-LLM-Model"] == "m1", "模型选择与凭据无关，照常带"
    assert behavior["mixed_pending"] == ["embedding"]


def test_unverified_ocr_key_is_not_sent_either(behavior: dict) -> None:
    """三个角色一视同仁：OCR 也会花钱，没通过就更不该发。"""
    assert "X-R4P-OCR-Key" not in behavior["ocr_bad_headers"]
    assert behavior["ocr_bad_headers"]["X-R4P-Embedding-Key"] == "sk-emb"


def test_legacy_record_without_verified_stays_usable(behavior: dict) -> None:
    """旧版本存的用户没有 `verified` 字段，必须按"启用"处理。

    升级一次就把别人一直在用的 Key 停掉，是比 P2-2 更严重的回归——他会看到
    "配置都填着，请求却 401"。
    """
    assert behavior["legacy_pending"] == []
    assert behavior["legacy_headers"]["X-R4P-Embedding-Key"] == "sk-a"
    assert behavior["legacy_headers"]["X-R4P-LLM-Key"] == "sk-b"


def test_all_pending_sends_no_credential_at_all(behavior: dict) -> None:
    """三项全没通过时只留用户名——等同于"没有凭据"，而不是"带着三把错的"。"""
    assert behavior["all_bad_headers"] == {}
    assert behavior["no_creds_headers"] == {}


def test_cleared_message_follows_the_billing_mode(behavior: dict) -> None:
    """清空后的提示必须跟着部署走（P2-1）。

    各付各的部署里，清掉 Key 之后什么都调不通；再说"后续请求使用服务端公共 Key"，
    用户会以为还能用，然后撞上 401。
    """
    strict = behavior["cleared_strict"]
    assert "服务端公共 Key" not in strict, "严格模式下这句是假话"
    assert "不代付" in strict and "你自己的 Key" in strict
    assert "服务端公共 Key" in behavior["cleared_demo"], "单机演示下这句是对的，别删掉"


def test_no_current_user_message_also_follows_the_mode(behavior: dict) -> None:
    """没有当前用户时的提示同样要分支——这是复审点名的同一处矛盾。"""
    assert "已经是服务端公共配置" not in behavior["nouser_strict"]
    assert "不代付" in behavior["nouser_strict"]
    assert "已经是服务端公共配置" in behavior["nouser_demo"]


# ── 管理口令：不属于任何用户，也不该因为"没选用户"就不发 ──


def test_admin_header_is_independent_of_the_current_user(behavior: dict) -> None:
    """口令留空就不带头（后端据此判断该不该拦），填了就带头。

    关键在于它和"当前选了哪个用户"无关：删库、删项目、删文件都要过这一关，
    而这些动作不该要求先选一个普通用户。
    """
    assert behavior["admin_header_empty"] == {}
    assert behavior["admin_header_set"] == {"X-R4P-Admin-Token": "s3cret"}


def test_admin_header_is_attached_even_without_credentials() -> None:
    """接线：`authHeaders()` 必须在"没有当前用户"的分支之前就把口令放进去。

    写成"有用户才回头填口令"，就成了"清掉自己的 Key 就再也删不掉文件"。
    """
    html = HTML.read_text(encoding="utf-8")
    start = html.index("function authHeaders()")
    body = html[start : html.index("\n}", start)]
    admin_at = body.index("adminHeader(credStore().admin)")
    user_at = body.index("currentCreds()")
    assert admin_at < user_at, "口令必须在取当前用户之前就加进去"


# ── 接线：判据生效的前提是保存时记下了验证结论 ──────────


def _save_body() -> str:
    html = HTML.read_text(encoding="utf-8")
    start = html.index("async function saveCreds()")
    end = html.index("function clearCreds()", start)
    return html[start:end]


def test_save_records_which_roles_verified() -> None:
    """`saveCreds()` 必须把三项的验证结论写进存储。

    少了这一步，`roleEnabled()` 只能一直返回真（没有 `verified` 就按启用处理），
    上面那几条行为用例全部变成"永远发"——测试是绿的，功能是坏的。
    """
    body = _save_body()
    assert "verified:" in body, "没记验证结论，未通过的 Key 会照样发出去"
    for role in ("embedding", "llm", "ocr"):
        assert f"{role}:" in body.split("verified:")[1], f"verified 里缺 {role}"


def test_save_allows_any_single_verified_role(behavior: dict) -> None:
    """三项里只要有一项通过就允许保存（2026-10-27 复审 P2 的那条）。

    凭据是分角色的：只填视觉模型 Key 的人，扫描件用得上，就该存下来。
    退化成"全通过才存"等于要求用户一次配齐，而那三把 Key 来自两家服务商。
    """
    assert behavior["save_only_ocr_ok"] is True, "只通过视觉模型就被判未保存"
    assert behavior["save_two_ok"] is True
    assert behavior["save_without_ocr_field"] is True, "老服务端不返回 ocr 字段时不该挡人"


def test_save_refuses_when_nothing_verified(behavior: dict) -> None:
    """三项全不通仍然不保存：存下来只会让之后每次请求都 401。"""
    assert behavior["save_none_ok"] is False


def test_save_goes_through_the_shared_judgement(behavior: dict) -> None:
    """`saveCreds()` 必须调用同一个判据，而不是自己再写一份。

    两份判据早晚会走偏（历史教训：一份判两个角色、另一份判三个）。
    """
    assert "shouldSave(result)" in _save_body()


def test_saved_message_names_the_missing_roles() -> None:
    """保存成功时要把"还没通过、因此不会发出去"的项点出来。

    只说"已保存"，用户会以为配全了，然后在提问撞上 401 的那一刻才发现另一栏是空的。
    断言要落在"它真的进了提示语"上：只声明一个变量却没拼进 `textContent`，
    用户一个字也看不见，而"变量存在"这种断言照样通过。
    """
    lines = _save_body().splitlines()
    assert any("lacks.length" in line for line in lines), "缺项汇总退化成常量或没算"
    assert any("lacks.join" in line for line in lines), "缺项名字没被拼进说明"
    assert any(
        "已保存并切换为" in line and "${lackNote}" in line for line in lines
    ), "算出来的缺项没拼进保存成功的提示语"
    for role in ("Embedding", "LLM", "视觉模型"):
        assert role in "\n".join(lines), f"缺项提示里没提 {role}"
