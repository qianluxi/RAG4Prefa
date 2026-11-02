"""图 2 系统总体架构与技术路线：A 知识构建链路 + B 公众问答服务链路 + 评测闭环。

画布 860x456，每个节点只留名称与 4~6 字标签，具体做法交给正文第三、四节。
"""

from pathlib import Path

from figstyle import (FS_PANEL, FS_TAG, LINE, MUTED, NAVY, PANEL, SLATE, TEAL,
                      Fig)

W, H = 860, 456

X0, GAP, NW = 16, 18, 193  # 四节点行：起点、间距、节点宽


def cols():
    return [X0 + i * (NW + GAP) for i in range(4)]


def build() -> Fig:
    f = Fig(W, H)
    c = cols()

    f.legend(844, 20, [(SLATE, "前期基础"), (TEAL, "拟研究")])

    # ===== A 知识构建链路 =====
    f.box(16, 34, 5, 18, NAVY, NAVY, rx=2, sw=0)
    f.text(30, 49, "A  知识构建链路", FS_PANEL, NAVY, weight="700")

    a_nodes = [
        ("多源知识", "标准·政策·案例"),
        ("解析与结构化", "章节·表格·OCR"),
        ("多维元数据治理", "地域·时效·等级"),
        ("知识库与索引", "事实库+双索引"),
    ]
    for i, (title, tag) in enumerate(a_nodes):
        f.node(c[i], 62, NW, 64, title, tag, TEAL if i == 2 else SLATE)
        if i:
            f.arrow(c[i] - 16, 94, c[i] - 2, 94, SLATE, 2)

    # ===== B 公众问答服务链路 =====
    f.box(16, 156, 5, 18, NAVY, NAVY, rx=2, sw=0)
    f.text(30, 171, "B  公众问答服务链路", FS_PANEL, NAVY, weight="700")

    b_nodes = [
        ("公众提问", "自然语言"),
        ("意图识别与澄清", "槽位·追问"),
        ("术语映射+混合检索", "双通道召回"),
        ("适用性与证据判断", "有据才回答"),
    ]
    for i, (title, tag) in enumerate(b_nodes):
        f.node(c[i], 184, NW, 64, title, tag, SLATE if i == 0 else TEAL)
        if i:
            f.arrow(c[i] - 16, 216, c[i] - 2, 216, SLATE, 2)

    # 知识库供检索调用：从 A 末节点折到 B 的检索节点上方
    f.path(f"M{c[3] + NW / 2} 126 V146 H{c[2] + NW / 2} V180", SLATE, 2, "6 5",
           "arrowGray")

    # ===== 证据状态三分支 =====
    f.path(f"M{c[3] + NW / 2} 248 V270 H146", SLATE, 2, None, None)
    outs = [
        (16, 146, "正常回答", "证据充分且适用", "#EAF5F2", TEAL),
        (300, 430, "提示与追问", "相关但证据不足", "#FDF4E6", "#B4802E"),
        (584, 714, "安全拒答", "无依据或不适用", "#F1F4F6", "#5F7482"),
    ]
    for x, cx, title, tag, fill, color in outs:
        f.arrow(cx, 270, cx, 290, SLATE, 2)
        f.box(x, 292, 260, 62, fill, LINE, rx=6, sw=1.2)
        f.text(cx, 319, title, 16, color, "middle", weight="700")
        f.text(cx, 341, tag, FS_TAG, MUTED, "middle")

    # ===== 评测与反馈闭环 =====
    f.box(16, 384, 828, 54, PANEL, "#C9D7E0", rx=6, sw=1.2)
    f.text(32, 417, "评测与反馈闭环", 16, NAVY, weight="700")
    f.text(192, 417, "检索质量 · 引用准确 · 拒答效果 · 任务完成 · 公众满意度",
           FS_TAG, MUTED)
    # 横条本身即闭环；不再画单点箭头，避免被读成只反馈给某一张出口卡

    return f


if __name__ == "__main__":
    here = Path(__file__).parent
    print(build().save(here / "figure2_system_architecture.svg"))
