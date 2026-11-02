"""图 3 典型公众咨询的多轮交互与可信处置流程。

画布 860x570，五步主干 + 三个出口，只留步骤名与短标签；条件取值用点分隔，
不再写解释句（解释留给正文第二节）。
"""

from pathlib import Path

from figstyle import (FS_BOX, FS_TAG, LINE, MUTED, NAVY, ORANGE, SLATE, TEAL,
                      Fig)

W, H = 860, 586
CX = 430


def build() -> Fig:
    f = Fig(W, H)
    f.legend(844, 20, [(SLATE, "前期基础"), (TEAL, "拟研究"), (ORANGE, "判断节点")])

    # ① 用户提问
    f.box(150, 30, 560, 62, "#FFFFFF", LINE)
    f.box(150, 30, 6, 62, SLATE, SLATE, rx=3, sw=0)
    f.text(172, 58, "①  用户提问", FS_BOX, "#41637A", weight="700")
    f.text(172, 82, "我在宁波，2026 年准备进行旧房厨卫改造，有哪些政策支持？", FS_TAG, MUTED)

    # ② 意图识别与条件抽取
    f.box(150, 120, 560, 62, "#FFFFFF", LINE)
    f.box(150, 120, 6, 62, TEAL, TEAL, rx=3, sw=0)
    f.text(172, 148, "②  意图识别与条件抽取", FS_BOX, TEAL, weight="700")
    f.text(172, 172, "意图判定 · 槽位抽取 · 口语归一 · 政策分类", FS_TAG, MUTED)

    f.arrow(CX, 94, CX, 116, SLATE, 2)

    # ③ 条件是否缺失
    f.diamond(CX, 230, 300, 74, "③  条件是否缺失？")
    f.arrow(CX, 122, CX, 192, SLATE, 2)
    # 是：只追问缺失项，折回②
    f.path("M280 230 H96 V151 H146", "#B4802E", 2, "6 5", "arrowOrange")
    f.text(116, 206, "是：只追问缺失项", FS_TAG, "#96703A")

    # 否：进入政策类型查询
    f.arrow(CX, 268, CX, 300, SLATE, 2)
    f.text(CX + 10, 289, "否", FS_TAG, MUTED)

    # ④ 政策类型识别与属地查询
    f.text(16, 292, "④  政策类型识别与属地查询", FS_TAG, NAVY, weight="700")
    for i, name in enumerate(["材料购置补贴", "智能家居购新补贴", "公积金装修提取"]):
        x = 16 + i * 284
        f.box(x, 302, 260, 50, "#FBFDFE", "#D6E3E9", rx=6, sw=1.2)
        f.text(x + 130, 333, name, 16, "#2A5F7A", "middle", weight="700")

    # 三类政策汇合到⑤
    for cx in (146, CX, 714):
        f.path(f"M{cx} 352 V362", SLATE, 2, None, None)
    f.path("M146 362 H714", SLATE, 2, None, None)
    f.arrow(CX, 362, CX, 372, SLATE, 2)

    # ⑤ 证据是否充分且适用
    f.diamond(CX, 408, 380, 78, "⑤  证据是否充分且适用？")

    # ⑤ 三出口
    f.path(f"M{CX} 446 V456 H146", SLATE, 2, None, None)
    outs = [
        (16, 146, "正常回答", "附文件出处", "#EAF5F2", TEAL),
        (300, CX, "提示与追问", "说明缺哪些条件", "#FDF4E6", "#B4802E"),
        (584, 714, "安全拒答", "给官方查询渠道", "#F1F4F6", "#5F7482"),
    ]
    for x, cx, title, tag, fill, color in outs:
        f.arrow(cx, 456, cx, 468, SLATE, 2)
        f.box(x, 470, 260, 58, fill, LINE, rx=6, sw=1.2)
        f.text(cx, 497, title, 16, color, "middle", weight="700")
        f.text(cx, 518, tag, FS_TAG, MUTED, "middle")

    f.text(CX, 558, "多轮上下文持续保留：地域 · 年份 · 房屋 · 部位", FS_TAG, MUTED,
           "middle")

    return f


if __name__ == "__main__":
    here = Path(__file__).parent
    print(build().save(here / "figure3_public_interaction_flow.svg"))
