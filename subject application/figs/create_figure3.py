"""生成图 3：典型公众咨询的多轮交互与可信处置流程（SVG）。

视觉规范与图 1、图 2 保持一致：白底、灰蓝为主、绿色标“本课题拟研究”、橙色标“判断与证据处置”。
输出：figs/figure3_public_interaction_flow.svg（PNG 由 render_png.py 渲染）
"""

from pathlib import Path

OUT = Path(__file__).parent / "figure3_public_interaction_flow.svg"

NAVY = "#173A5E"
BLUE = "#315A7C"
TEAL = "#176B5B"
ORANGE = "#8A5A12"
GRAY = "#607286"
LINE = "#D3E0EA"

FONT = "'Noto Sans CJK SC','Microsoft YaHei','SimHei',sans-serif"
svg: list[str] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rect(x, y, w, h, fill="#FFFFFF", stroke=LINE, rx=14, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
    )


def text(x, y, s, size=15, fill=GRAY, weight=400, anchor="start", bold=False):
    svg.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{700 if bold else weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{esc(s)}</text>'
    )


def lines(x, y, seq, size=14.5, fill=GRAY, lh=26, anchor="start", weight=400):
    svg.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">'
    )
    for i, s in enumerate(seq):
        svg.append(f'<tspan x="{x}" dy="{0 if i == 0 else lh}">{esc(s)}</tspan>')
    svg.append("</text>")


def arrow(d, color="#718096", width=3, dash=None, marker="arrow"):
    dd = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dd} marker-end="url(#{marker})"/>'
    )


def chip(x, y, w, h, label, fill="#FFFFFF", stroke="#D1DDE7", color=BLUE, size=14.5, bold=True):
    rect(x, y, w, h, fill, stroke, h / 2, 1.2)
    text(x + w / 2, y + h / 2 + 5, label, size, color, anchor="middle", bold=bold)


# ---------------------------------------------------------------- 画布与标题
svg.append(
    '<svg xmlns="http://www.w3.org/2000/svg" width="170mm" height="105.4mm" '
    'viewBox="0 0 2000 1240">'
)
svg.append(
    "<defs>"
    '<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" '
    'orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L11,6 L1,11 z" fill="#718096"/></marker>'
    '<marker id="arrowTeal" markerWidth="12" markerHeight="12" refX="10" refY="6" '
    'orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L11,6 L1,11 z" fill="#176B5B"/></marker>'
    '<marker id="arrowWarm" markerWidth="12" markerHeight="12" refX="10" refY="6" '
    'orient="auto" markerUnits="userSpaceOnUse"><path d="M1,1 L11,6 L1,11 z" fill="#C08A2E"/></marker>'
    "</defs>"
)
svg.append('<rect width="2000" height="1240" fill="#FFFFFF"/>')

text(62, 58, "典型公众咨询的多轮交互与可信处置流程", 34, NAVY, bold=True)
text(64, 97, "以“我在宁波，2026 年准备进行旧房厨卫改造，有哪些政策支持？”为例", 18, GRAY)

# 图例
chip(1372, 42, 156, 34, "前期已具备能力", "#F4F8FB", "#D3E0EA", BLUE, 14)
chip(1544, 42, 160, 34, "本课题拟研究", "#E9F6F2", "#C4DED6", TEAL, 14)
chip(1720, 42, 220, 34, "判断与证据处置节点", "#FFF6E6", "#EBD9B6", ORANGE, 14)
svg.append('<line x1="60" y1="122" x2="1940" y2="122" stroke="#E6EDF1" stroke-width="2"/>')

# ---------------------------------------------------------------- ① 用户提问
rect(55, 150, 360, 320, "#F7FAFC", LINE, 16, 1.8)
text(78, 186, "① 用户提问", 19, BLUE, bold=True)
svg.append('<line x1="78" y1="200" x2="392" y2="200" stroke="#E4EBF0" stroke-width="1.4"/>')
rect(75, 214, 320, 100, "#FFFFFF", "#DFE8ED", 10, 1.2)
lines(92, 246, ["“我在宁波，2026 年准备进行", "旧房厨卫改造，有哪些", "政策支持？”"], 14.5, "#2F4B60", 26)
text(75, 342, "自然语言中显式给出的条件", 13, "#6D7E90")
chip(75, 352, 68, 32, "宁波", "#EAF1F8", "#CFDDE9", BLUE, 13.5)
chip(151, 352, 92, 32, "2026 年", "#EAF1F8", "#CFDDE9", BLUE, 13.5)
chip(251, 352, 68, 32, "旧房", "#EAF1F8", "#CFDDE9", BLUE, 13.5)
chip(75, 392, 108, 32, "厨卫改造", "#EAF1F8", "#CFDDE9", BLUE, 13.5)
chip(191, 392, 108, 32, "政策支持", "#EAF1F8", "#CFDDE9", BLUE, 13.5)
text(75, 448, "意图清晰，但“政策支持”未指向具体类别。", 12.5, "#7A8B99")

# ---------------------------------------------------------------- ② 意图与条件抽取
rect(470, 150, 380, 320, "#E9F6F2", "#C4DED6", 16, 1.8)
text(492, 186, "② 意图识别与条件抽取", 19, TEAL, bold=True)
chip(712, 166, 118, 28, "本课题拟研究", "#FFFFFF", "#BFD9D1", TEAL, 12.5)
svg.append('<line x1="492" y1="200" x2="828" y2="200" stroke="#D3E6DF" stroke-width="1.4"/>')
items = [
    ("意图判定", "政策咨询（补贴申领 / 提取）"),
    ("槽位抽取", "地域、年份、房屋类型、部位"),
    ("口语归一", "厨卫改造→厨房卫生间装修"),
    ("类别识别", "“政策支持”→拆为三个类别"),
]
iy = 236
for k, v in items:
    svg.append(f'<circle cx="497" cy="{iy - 4}" r="4.5" fill="{TEAL}"/>')
    text(509, iy, k, 13.5, TEAL, bold=True)
    text(509, iy + 24, v, 13, "#3B5566")
    iy += 62

# ---------------------------------------------------------------- ③ 条件缺失判断
rect(900, 150, 280, 320, "#FFF6E6", "#EBD9B6", 16, 1.8)
text(922, 186, "③ 条件是否缺失", 19, ORANGE, bold=True)
svg.append('<line x1="922" y1="200" x2="1158" y2="200" stroke="#EFE0C4" stroke-width="1.4"/>')
svg.append(
    '<path d="M1040 208 L1150 260 L1040 312 L930 260 z" fill="#FFFFFF" stroke="#D8B87C" stroke-width="2"/>'
)
text(1040, 267, "条件缺失？", 17, ORANGE, anchor="middle", bold=True)
text(922, 340, "是：仅追问缺失槽位，", 14, ORANGE)
text(922, 364, "保留已确认条件", 14, ORANGE)
text(922, 400, "否：进入政策类型查询", 14, "#3B5566")
text(922, 438, "追问不重问已知条件", 12, "#96703A")

# 追问回环：③ → ②
arrow("M898 356 C880 356, 872 356, 858 356", "#C08A2E", 2.2, "7 6", "arrowWarm")

# ---------------------------------------------------------------- ④ 政策类型与知识查询
rect(1230, 150, 715, 320, "#F7FAFC", LINE, 16, 1.8)
text(1253, 186, "④ 政策类型识别与属地知识查询", 19, BLUE, bold=True)
svg.append('<line x1="1253" y1="200" x2="1922" y2="200" stroke="#E4EBF0" stroke-width="1.4"/>')
pol = [
    (1250, "材料购置补贴", ["装修材料", "按比例补贴"]),
    (1486, "智能家居购新补贴", ["按品类与额度", "限额补贴"]),
    (1722, "公积金装修提取", ["自住住房装修", "提取公积金"]),
]
for px, name, desc in pol:
    rect(px, 210, 222, 82, "#FFFFFF", "#DCE6EC", 10, 1.2)
    text(px + 111, 240, name, 14.5, NAVY, anchor="middle", bold=True)
    lines(px + 111, 264, desc, 12.5, "#64798A", 21, anchor="middle")

text(1253, 328, "检索范围：浙江省与宁波市现行文件，按地域、年份与类型过滤", 13.5, "#3B5566")
text(1253, 362, "逐条校验：", 13.5, "#3B5566")
pills = [(1352, 112, "有效时间"), (1474, 112, "适用对象"), (1596, 140, "来源权威等级"), (1746, 132, "文件状态")]
for px, pw, lab in pills:
    chip(px, 342, pw, 32, lab, "#FFFFFF", "#D8E4EC", BLUE, 12.5)
text(1253, 412, "同一句“政策支持”对应三类政策，适用条件与办理渠道各异，须分类型检索与作答。", 12.5, "#96703A")
text(1253, 440, "文件状态取“实施 / 截止 / 废止”，无失效日期者不默认有效。", 12.5, "#96703A")

# ---------------------------------------------------------------- ②→③、③→④ 连接
arrow("M852 260 H894", "#718096", 3)
arrow("M1182 260 H1224", "#718096", 3)
text(1204, 248, "否", 12.5, "#7A8B99", anchor="middle")

# ---------------------------------------------------------------- ④ → ⑤ 连接
arrow("M1588 470 V546 H1010 V564", "#718096", 3)

# ---------------------------------------------------------------- ⑤ 适用性与证据判断
svg.append(
    '<path d="M1000 566 L1230 616 L1000 666 L770 616 z" fill="#FFF6E6" stroke="#D8B87C" stroke-width="2.2"/>'
)
text(1000, 606, "⑤ 适用性与证据充分性判断", 16.5, ORANGE, anchor="middle", bold=True)
text(1000, 634, "依据是否存在 → 是否适用 → 是否充分", 12, "#96703A", anchor="middle")

# ⑤ → 三出口
arrow("M930 640 C760 662, 520 664, 355 694", "#C08A2E", 2.6, None, "arrowWarm")
arrow("M1000 668 V694", "#C08A2E", 2.6, None, "arrowWarm")
arrow("M1070 640 C1240 662, 1460 664, 1595 694", "#C08A2E", 2.6, None, "arrowWarm")

# ---------------------------------------------------------------- 三类处置出口
cards = [
    (55, "证据充分", "正常回答", TEAL, "#F3FAF7", "#C4DED6", [
        "解释适用条件、办理流程、所需材料与时限",
        "附正式文件出处：文件名称、条款、生效状态",
        "提示“最终资格以主管部门核验为准”",
    ]),
    (675, "相关但证据不足", "提示与追问", ORANGE, "#FFFBF2", "#EBD9B6", [
        "说明已能确认的内容及其依据",
        "指出缺少哪些条件（面积、产权、施工主体）",
        "继续追问，或建议补充材料后再判断",
    ]),
    (1295, "无依据或不适用", "安全处置", "#4B6478", "#F7F9FB", "#D8E2E8", [
        "不猜测政策结论，不生成未收录条款",
        "说明不适用原因：地域、时间或对象不符",
        "提供主管部门与官方查询渠道",
    ]),
]
for x, head, sub, color, fill, stroke, bullets in cards:
    rect(x, 700, 600, 250, fill, stroke, 14, 1.6)
    rect(x, 700, 600, 54, "#FFFFFF", stroke, 14, 1.4)
    text(x + 24, 735, head, 17, color, bold=True)
    text(x + 600 - 24, 735, sub, 13.5, "#7A8B99", anchor="end")
    by = 780
    for b in bullets:
        svg.append(f'<circle cx="{x + 28}" cy="{by - 5}" r="4" fill="{color}"/>')
        text(x + 42, by, b, 14, "#3B5566")
        by += 30

# ---------------------------------------------------------------- 底部：上下文与评测
rect(55, 985, 1130, 148, "#F5F8FA", "#DCE6EB", 14, 1.6)
text(80, 1022, "多轮上下文持续保留", 17, BLUE, bold=True)
lines(80, 1054, ["地域、年份、房屋类型、改造部位一次确认后不在后续轮次重复询问；",
                 "用户补充或更正时即时更新槽位，澄清轮数纳入交互效率指标。"], 13.5, "#4A6273", 26)

rect(1210, 985, 735, 148, "#F5F8FA", "#DCE6EB", 14, 1.6)
text(1235, 1022, "每次处置进入评测", 17, BLUE, bold=True)
lines(1235, 1054, ["按处置类型分别统计：拦截率、误拒率、",
                   "任务完成率、澄清轮数与引用可核验率。"], 13.5, "#4A6273", 26)

svg.append('<line x1="60" y1="1168" x2="1940" y2="1168" stroke="#E6EDF1" stroke-width="1.5"/>')
text(62, 1200, "注：蓝色为前期已具备的检索与生成能力，绿色为本课题拟研究内容，橙色为判断与证据处置节点。", 12.5, "#7A8B99")

svg.append("</svg>")

OUT.write_text("\n".join(svg), encoding="utf-8")
print(OUT)
