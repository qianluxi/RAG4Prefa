"""图 1 项目总体研究框架：现实问题 → 四项核心研究 → 预期成果，底部前期基础。

画布 860x420，插入 15.5cm 宽后高约 7.6cm，正文 16 号字打印约 8pt。
"""

from pathlib import Path

from figstyle import (FS_BOX, FS_PANEL, FS_TAG, LINE, MUTED, NAVY, PANEL, SLATE,
                      TEAL, WHITE, Fig)

W, H = 860, 420


def build() -> Fig:
    f = Fig(W, H)

    # 图例（右上）
    f.legend(844, 20, [(SLATE, "前期基础"), (TEAL, "拟研究"), (NAVY, "预期成果")])

    # 列标题
    f.text(91, 46, "现实问题", FS_PANEL, NAVY, "middle", weight="700")
    f.text(386, 46, "四项核心研究", FS_PANEL, NAVY, "middle", weight="700")
    f.text(725, 46, "预期成果", FS_PANEL, NAVY, "middle", weight="700")

    # 左：三个落差
    gaps = [("语言落差", "术语难懂"), ("适用性落差", "地域时效"), ("可信度落差", "有据难判")]
    for i, (title, tag) in enumerate(gaps):
        y = 60 + i * 96
        f.box(16, y, 150, 80, PANEL, LINE)
        f.box(16, y, 6, 80, SLATE, SLATE, rx=3, sw=0)
        f.text(91, y + 34, title, FS_BOX, "#41637A", "middle", weight="700")
        f.text(91, y + 60, tag, FS_TAG, MUTED, "middle")

    # 中：总目标 + 四卡
    f.box(186, 60, 400, 46, NAVY, NAVY, rx=6, sw=0)
    f.text(386, 89, "可理解 · 可交互 · 可信赖的公众知识服务", 16, WHITE, "middle",
           weight="700")

    cards = [
        ("多源知识治理", "元数据与适用性"),
        ("双向语义转换", "术语与口语互译"),
        ("证据约束判断", "有据才回答"),
        ("多轮交互智能体", "澄清与任务引导"),
    ]
    for i, (title, tag) in enumerate(cards):
        x = 186 + (i % 2) * 209
        y = 124 + (i // 2) * 96
        f.box(x, y, 191, 80, WHITE, LINE)
        f.box(x, y, 191, 4, TEAL, TEAL, rx=2, sw=0)
        f.text(x + 16, y + 38, title, FS_BOX, TEAL, weight="700")
        f.text(x + 16, y + 62, tag, FS_TAG, MUTED)

    # 右：预期成果
    outs = ["公众知识库", "交互智能体", "评测方法", "试点应用"]
    for i, title in enumerate(outs):
        y = 60 + i * 70
        f.box(606, y, 238, 58, PANEL, LINE)
        f.box(606, y, 6, 58, NAVY, NAVY, rx=3, sw=0)
        f.text(628, y + 36, title, FS_BOX, NAVY, weight="700")

    # 横向箭头：两条同高，表示「问题 → 研究 → 成果」的整体流向，不暗示某两张卡相连
    f.arrow(172, 200, 182, 200)
    f.arrow(590, 200, 600, 200)

    # 底部：前期工程基础
    f.box(16, 344, 828, 54, PANEL, "#C9D7E0", rx=6, sw=1.2)
    f.text(32, 377, "前期工程基础（已完成）", 16, NAVY, weight="700")
    for i, label in enumerate(["文档解析", "混合检索", "引用生成", "可复现评测"]):
        f.chip(288 + i * 140, 357, 128, 28, label)
    # 基础向上支撑核心研究
    f.arrow(388, 344, 388, 308, SLATE, 2, "6 5")
    f.arrow(492, 344, 492, 308, SLATE, 2, "6 5")

    return f


if __name__ == "__main__":
    here = Path(__file__).parent
    svg = build().save(here / "figure1_research_framework.svg")
    print(svg)
