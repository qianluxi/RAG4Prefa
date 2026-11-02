"""申报书三张配图的共用绘图工具。

**尺寸约定（关键）**：画布宽度统一 860 单位，插入 Word 时按 15.5cm 宽排版，
即 1 单位 ≈ 0.018cm。正文取 16 号字，打印后约 8.2pt，A4 纸面清晰可读；
若把画布放大到 1800 宽而字号仍是 14，打印后只有 3pt，必然看不清。
加内容时优先扩高度，不要改画布宽度。

**文字约定**：图上只写模块名称与极短标签（≤ 9 字），说明性长句一律不写，
由正文承担；单张图主要节点控制在 15~22 个。

配色沿用三张图统一的图例：灰蓝=前期已有，蓝绿=本课题拟研究，深蓝=成果，
橙色=判断与证据处置节点。
"""

from pathlib import Path

W = 860  # 画布标准宽度

NAVY = "#254E68"      # 成果
TEAL = "#2A8C82"      # 本课题拟研究
SLATE = "#5B7E96"     # 前期工程基础
ORANGE = "#C08A2E"    # 判断 / 证据处置
INK = "#20394B"
MUTED = "#5F7482"
LINE = "#D3DFE6"
PANEL = "#F7FAFC"
WHITE = "#FFFFFF"

# 字号阶梯：正文 17，标签 14.5，不再出现更小的字（打印约 8.7pt / 7.4pt）
FS_TITLE = 19
FS_PANEL = 16
FS_BOX = 17
FS_TAG = 14.5

FONT = "'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Fig:
    def __init__(self, w: int = W, h: int = 600):
        self.w, self.h = w, h
        self.body: list[str] = []

    # ---------- 基本图元 ----------
    def box(self, x, y, w, h, fill=WHITE, stroke=LINE, rx=8, sw=1.4, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def text(self, x, y, s, size=FS_BOX, color=INK, anchor="start", weight=None,
             opacity=None):
        a = f' text-anchor="{anchor}"' if anchor != "start" else ""
        wgt = f' font-weight="{weight}"' if weight else ""
        op = f' opacity="{opacity}"' if opacity else ""
        self.body.append(
            f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}"{a}{wgt}{op}>'
            f"{esc(s)}</text>"
        )

    def lines(self, x, y, seq, size=FS_BOX, color=INK, lh=None, anchor="start",
              weight=None):
        """多行文字，逐行给出（不自动换行，宁可改写短句）。"""
        lh = lh or size + 8
        for i, s in enumerate(seq):
            self.text(x, y + i * lh, s, size, color, anchor, weight)

    def arrow(self, x1, y1, x2, y2, color=SLATE, sw=2.2, dash=None, head="arrow"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{head})"' if head else ""
        self.body.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round"{d}{m}/>'
        )

    def path(self, d, color=TEAL, sw=2.2, dash=None, head="arrowTeal", fill="none"):
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{head})"' if head else ""
        self.body.append(
            f'<path d="{d}" fill="{fill}" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round"{ds}{m}/>'
        )

    def diamond(self, cx, cy, w, h, label, fill="#FFF6E6", stroke=ORANGE,
                color="#8A6420", size=FS_BOX):
        hw, hh = w / 2, h / 2
        self.body.append(
            f'<path d="M{cx} {cy - hh} L{cx + hw} {cy} L{cx} {cy + hh} '
            f'L{cx - hw} {cy} z" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        self.text(cx, cy + size * 0.36, label, size, color, "middle", weight="700")

    def band(self, x, y, w, h, label, fill="#EEF4F7", stroke="#C9D7E0"):
        """链路带标题条（左侧色块 + 文字），用于 A/B 两条链路的标识。"""
        self.box(x, y, w, h, fill, stroke, rx=6, sw=1.2)
        self.box(x, y, 6, h, NAVY, NAVY, rx=3, sw=0)
        self.text(x + 16, y + h / 2 + 6, label, FS_PANEL, NAVY, weight="700")

    # ---------- 常用组合 ----------
    def node(self, x, y, w, h, title, tag=None, accent=SLATE, tag_color=None):
        """一个流程节点：标题 + 可选短标签；左侧 6 单位色条标出归属图例。"""
        self.box(x, y, w, h, WHITE, LINE)
        self.box(x, y, 6, h, accent, accent, rx=3, sw=0)
        if tag:
            self.text(x + 18, y + h / 2 - 4, title, FS_BOX, INK, weight="700")
            self.text(x + 18, y + h / 2 + 18, tag, FS_TAG, tag_color or MUTED)
        else:
            self.text(x + 18, y + h / 2 + 6, title, FS_BOX, INK, weight="700")

    def card(self, x, y, w, h, title, tag=None, accent=TEAL, fill=WHITE):
        """研究内容卡：编号圆点 + 标题 + 短标签。"""
        self.box(x, y, w, h, fill, LINE)
        self.text(x + 16, y + 26, title, FS_BOX, accent, weight="700")
        if tag:
            self.text(x + 16, y + 50, tag, FS_TAG, MUTED)

    def chip(self, x, y, w, h, label, fill=PANEL, color=MUTED, size=FS_TAG):
        self.box(x, y, w, h, fill, LINE, rx=6, sw=1)
        self.text(x + w / 2, y + h / 2 + size * 0.36, label, size, color, "middle")

    def legend(self, x_right, y, items):
        """图例右对齐；items = [(颜色, 文字), ...]，自右向左排布。"""
        x = x_right
        for color, label in reversed(items):
            tw = len(label) * FS_TAG + 6
            x -= tw
            self.text(x, y, label, FS_TAG, MUTED)
            x -= 20
            self.box(x, y - 11, 12, 12, color, color, rx=2, sw=0)
            x -= 16

    # ---------- 输出 ----------
    def svg(self) -> str:
        markers = "\n".join(
            f'  <marker id="{mid}" markerWidth="9" markerHeight="9" refX="7.5" '
            f'refY="4.5" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L9,4.5 L0,9 z" fill="{col}"/></marker>'
            for mid, col in (
                ("arrow", SLATE), ("arrowTeal", TEAL), ("arrowNavy", NAVY),
                ("arrowOrange", ORANGE), ("arrowGray", "#8A9AA5"),
            )
        )
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
            f"<defs>\n{markers}\n"
            f'  <style>text{{font-family:{FONT};}}</style>\n</defs>\n'
            f'<rect width="{self.w}" height="{self.h}" fill="#FFFFFF"/>\n'
        )
        return head + "\n".join(self.body) + "\n</svg>\n"

    def save(self, path) -> Path:
        path = Path(path)
        path.write_text(self.svg(), encoding="utf-8")
        return path
