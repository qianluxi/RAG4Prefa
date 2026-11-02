"""前端引用来源面板（`index.html` 里的 `renderCitations`）的验收测试。

背景（2026-11-02 用户反馈）：问完之后「查看来源」按钮凭空消失，看起来像功能被删了。
真实原因是那两条**没有候选**的分支（检索为空、证据闸门拒答）后端发的是空数组，
而 `renderCitations` 直接 `return` —— 按钮不显示、也没有任何一句话说明为什么。

判据：**没有来源时不能安静地什么都不做**。要区分"这次没有"与"这个功能不存在"，
就得在按钮的位置上留下一句能读的说明，而且说明要指出是哪条分支。

做法沿用 `test_web_markdown.py`：抽出 `SRC-PANEL-START/END` 之间的真实实现，交给
node 执行。这里比那边多一层：`renderCitations` 会碰 DOM，所以喂一个最小假 DOM
（createElement / appendChild / querySelectorAll），断言的是**产出结构**，不是
"文件里有没有 src-empty 这个字符串"。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

HTML = Path(__file__).resolve().parent.parent / "rag4prefa" / "web" / "index.html"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="需要 node 才能执行抽出的前端函数")

_MARKERS = re.compile(r"/\* SRC-PANEL-START.*?/\* SRC-PANEL-END \*/", re.S)

# 假 DOM 只实现 renderCitations 真正用到的那几个方法。刻意不引 jsdom：
# 这个项目的页面是单文件离线可用的，测试也不该为了一个 appendChild 拉一整棵 DOM 树。
_HARNESS = """
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

function node(tag) {
  return {
    tag, id: "", className: "", textContent: "", innerHTML: "",
    style: {}, dataset: {}, children: [],
    classList: { add() {}, remove() {} },
    appendChild(child) { this.children.push(child); return child; },
    querySelectorAll() { return []; },
    scrollIntoView() {},
  };
}
const el = (tag, cls, text) => {
  const n = node(tag);
  n.className = cls || "";
  if (text !== undefined) n.textContent = text;
  return n;
};
let drawerBody = node("div");
const $ = (id) => (id === "sources-body" ? drawerBody : node("div"));
const openSources = () => {};

const api = new Function(
  "el", "$", "openSources",
  src + "\\nreturn {renderCitations, noSourceReason};",
)(el, $, openSources);

const out = cases.map((c) => {
  drawerBody = node("div");
  const botMsg = node("div");
  const toggle = node("button");
  toggle.style.display = "none";
  api.renderCitations(c.items, botMsg, toggle, c.reason || "");
  const empty = botMsg.children.filter((n) => n.className === "src-empty");
  return {
    emptyNotes: empty.map((n) => n.textContent),
    cards: drawerBody.children.filter((n) => n.className === "card").length,
    toggleDisplay: toggle.style.display,
    toggleText: toggle.textContent,
    reason: api.noSourceReason(c.refusal || null, c.candidates),
  };
});
process.stdout.write(JSON.stringify(out));
"""


def run(*cases: dict) -> list[dict]:
    """按同一份实现跑多个场景，返回每个场景渲染出的结构。"""
    html = HTML.read_text(encoding="utf-8")
    block = _MARKERS.search(html)
    assert block, "index.html 里找不到 SRC-PANEL-START/END 标记，来源面板被挪走或删了"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "panel.js").write_text(block.group(0), encoding="utf-8")
        (tmp_path / "cases.json").write_text(
            json.dumps(list(cases), ensure_ascii=False), encoding="utf-8"
        )
        (tmp_path / "run.js").write_text(_HARNESS, encoding="utf-8")
        proc = subprocess.run(
            [
                NODE,
                str(tmp_path / "run.js"),
                str(tmp_path / "panel.js"),
                str(tmp_path / "cases.json"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


ITEM = {
    "n": 1,
    "project": "装配式内装",
    "project_code": "LS2024",
    "filename": "技术要点.pdf",
    "chapter": "墙面 > 基层",
    "page": 12,
    "snippet": "基层应清理干净",
}


def test_empty_citations_explains_instead_of_vanishing() -> None:
    """核心判据：没有片段时必须留下说明，且按钮不能假装可用。"""
    (out,) = run({"items": [], "reason": "资料不足已拒答：候选为空"})
    assert out["emptyNotes"] == ["未引用片段：资料不足已拒答：候选为空"]
    assert out["toggleDisplay"] == "none", "没有来源时不该出现可点的「查看来源」"
    assert out["cards"] == 0


def test_empty_without_reason_still_says_something() -> None:
    """原因是可选的：拿不到原因也不能退回静默。"""
    (out,) = run({"items": []})
    assert out["emptyNotes"] == ["未引用片段"]
    assert out["toggleDisplay"] == "none"


def test_items_render_cards_and_show_toggle() -> None:
    (out,) = run({"items": [ITEM, {**ITEM, "n": 2}]})
    assert out["cards"] == 2
    assert out["toggleDisplay"] == "", "有来源时按钮必须露出来"
    assert out["toggleText"] == "查看来源（2 条）"
    assert out["emptyNotes"] == [], "有来源时不该再补一句「未引用片段」"


def test_reason_prefers_refusal_then_empty_candidates() -> None:
    """三条分支要能分开：拒答说拒答，筛空说筛空，其余不硬编理由。"""
    refusal, empty, other = run(
        {"items": [], "refusal": {"refuse": True, "reasons": ["候选为空", "地域未覆盖"]}},
        {"items": [], "candidates": 0},
        {"items": [], "candidates": 8},
    )
    assert refusal["reason"] == "资料不足已拒答：候选为空；地域未覆盖"
    assert empty["reason"] == "当前筛选范围内没有检索到片段"
    assert other["reason"] == "这次回答没有标注引用", "候选非空又没引用，别编成拒答"
