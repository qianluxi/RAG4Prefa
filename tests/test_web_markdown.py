"""前端答案渲染（`index.html` 里的 Markdown 子集）的验收测试。

背景：模型本来就按 Markdown 组织答案，但前端早期只把 `[n]` 换成角标、其余原样
输出，于是 `**粗体**`、`* 列表`、`1. 编号` 全以源码形态糊成一段，用户反馈"没有
层次感"。为此在 `index.html` 里实现了一个够用的子集（不引 CDN，页面单文件、离线
可用），这里对**真实实现**做验收，而不是对"文件里有没有 <ul> 这个字符串"。

做法：抽出 `/* MD-RENDER-START */ … /* MD-RENDER-END */` 之间的源码，交给 node
执行（`new Function` 起一个干净作用域，不碰 DOM），再断言输出 HTML。

断言的对象都是"判据"而不是"现象"：
- 结构：段落 / 有序 / 无序 / 两级嵌套各成标签，不是一堆字面量；
- 安全：模型输出里的 `<script>` 必须变成实体，插入的标签只能来自渲染器；
- 边界：未闭合的 `**` 在流式过程中保持源码、`3*4*5` 不被当成斜体、行内代码里的
  `*` 与 `[2]` 不被二次解释、`[2](url)` 是链接而不是"角标 + 裸地址"。
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

pytestmark = pytest.mark.skipif(NODE is None, reason="需要 node 才能执行抽出的前端渲染函数")

_MARKERS = re.compile(r"/\* MD-RENDER-START.*?/\* MD-RENDER-END \*/", re.S)

_HARNESS = """
const fs = require("fs");
const src = fs.readFileSync(process.argv[2], "utf8");
const cases = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const api = new Function(src + "\\nreturn {renderAnswer, mdInline};")();
process.stdout.write(JSON.stringify(cases.map((c) => api.renderAnswer(c))));
"""


def render(*cases: str) -> list[str]:
    """按同一份实现渲染多个输入，返回输出 HTML 列表。"""
    html = HTML.read_text(encoding="utf-8")
    block = _MARKERS.search(html)
    assert block, "index.html 里找不到 MD-RENDER-START/END 标记，渲染函数被挪走或删了"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "md.js").write_text(block.group(0), encoding="utf-8")
        (tmp_path / "cases.json").write_text(
            json.dumps(list(cases), ensure_ascii=False), encoding="utf-8"
        )
        (tmp_path / "run.js").write_text(_HARNESS, encoding="utf-8")
        proc = subprocess.run(
            [NODE, str(tmp_path / "run.js"), str(tmp_path / "md.js"), str(tmp_path / "cases.json")],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# 现场那条答案的骨架：粗体小标题 + 有序列表 + 无序列表 + 引用角标
SAMPLE = """根据提供的资料片段，涉及的规范及管理办法如下：
1. **国家标准**：《装配式建筑评价标准》（GB/T51129-2017）[2]
2. **浙江省地方文件**：《浙江省装配式建筑评价认定管理办法》[1][2]
**依据：**
*   资料片段 [2] 中提到“《装配式建筑评价标准》”。"""


def test_structure_is_real_markup_not_literal_source() -> None:
    (out,) = render(SAMPLE)
    assert "<strong>国家标准</strong>" in out
    assert "<strong>浙江省地方文件</strong>" in out
    ordered = re.search(r"<ol>(.*?)</ol>", out, re.S)
    assert ordered, "有序列表没有渲染成 <ol>"
    assert ordered.group(1).count("<li>") == 2
    assert "<ul>" in out and "<li>资料片段" in out
    assert "<strong>依据：</strong>" in out
    assert "**" not in out, "字面量 ** 说明加粗没被解释"
    assert not re.search(r"^\s*\d\.\s", out, re.M), "列表编号应当变成 <li>，而不是留在正文里"


def test_citation_markers_become_chips() -> None:
    (out,) = render("- 《标准》[2]\n- 《办法》[1][2]")
    assert out.count('<span class="cite" data-n="2">[2]</span>') == 2
    assert '<span class="cite" data-n="1">[1]</span>' in out


def test_model_output_is_escaped_before_parsing() -> None:
    payload = '看这个 <img src=x onerror=alert(1)> 和 <script>alert(2)</script>'
    (out,) = render(payload)
    assert "<img" not in out and "<script" not in out
    assert "&lt;img src=x onerror=alert(1)&gt;" in out
    assert "&lt;script&gt;" in out


def test_unclosed_emphasis_survives_streaming() -> None:
    """流式到一半只有前半截，不能把后面的内容整段吞掉或吐成标签。"""
    half, full = render("**国家标准", "**国家标准**")
    assert "**国家标准" in half
    assert "<strong>" not in half
    assert "<strong>国家标准</strong>" in full


def test_code_span_is_not_reinterpreted() -> None:
    (out,) = render("写成 `a**b**c` 与 `[2]` 时按原文显示")
    assert "<code>a**b**c</code>" in out
    assert "<code>[2]</code>" in out
    assert "<strong>" not in out
    assert "cite" not in out


def test_nested_list_nests_inside_parent_item() -> None:
    (out,) = render("1. 一层\n    - 二层\n2. 回到一层")
    assert out == "<ol><li>一层<ul><li>二层</li></ul></li><li>回到一层</li></ol>"


def test_arithmetic_is_not_italic() -> None:
    (out,) = render("体积按 3*4*5 计算")
    assert "<em>" not in out
    assert "3*4*5" in out


def test_link_is_not_split_into_citation() -> None:
    (out,) = render("见[2](https://example.com/a)")
    assert '<a href="https://example.com/a"' in out
    assert "cite" not in out, "[2](url) 被拆成了角标 + 裸地址"


def test_heading_starts_below_page_heading_levels() -> None:
    """页面自己的标题是 h1/h2，答案里的 # 只能从 h3 起，否则会打乱文档大纲。"""
    (out,) = render("# 结论\n正文")
    assert "<h3>结论</h3>" in out
    assert "<h1>" not in out and "<h2>" not in out


def test_plain_text_keeps_line_breaks() -> None:
    """中文答案里单个换行通常就是"换行"，不能被折成空格粘成一行。"""
    (out,) = render("第一行\n第二行")
    assert out == "<p>第一行<br>第二行</p>"


def test_pipe_table_renders_as_table() -> None:
    (out,) = render("| 项目 | 数值 |\n| --- | --- |\n| 层高 | 3.0m |")
    assert "<table><thead><tr><th>项目</th><th>数值</th></tr></thead>" in out
    assert "<td>层高</td><td>3.0m</td>" in out
    assert "|" not in out, "竖线留在正文里说明表格没被识别"


def test_pipe_line_without_separator_stays_text() -> None:
    """没有 |---| 分隔行的竖线文本不是表格，别把普通句子拆成单元格。"""
    (out,) = render("| 只有一个竖线行 |")
    assert "<table>" not in out
    assert "| 只有一个竖线行 |" in out
