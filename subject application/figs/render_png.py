"""把 figs 下的 SVG 用本机浏览器渲染成高分辨率 PNG，供 Word 内嵌（Word 对 SVG 支持不稳）。

用法：python render_png.py                     # 渲染目录内全部 svg
      python render_png.py a.svg b.svg         # 只渲染指定文件
"""

import subprocess
import sys
import tempfile
from pathlib import Path

CHROME = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

# 目标像素宽度，按原图 viewBox 宽度等比放大，保证 300dpi 左右的印刷清晰度
TARGET_W = 3000

# Chrome 的截图区域是视口，比 --window-size 小：宽少 16px、高少 94px（窗口边框与标题栏）。
# 不补偿会把图右侧 0.5%、底部 6% 裁掉——底部注释行会被切一半。实测校准值见 README。
PAD_W, PAD_H = 16, 94


def browser() -> str:
    for p in CHROME:
        if Path(p).exists():
            return p
    raise SystemExit("找不到 Chrome / Edge，无法渲染")


def render(svg: Path) -> Path:
    raw = svg.read_text(encoding="utf-8")
    # 从 width/height 或 viewBox 取尺寸
    import re

    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', raw)
    if m:
        vw, vh = float(m.group(1)), float(m.group(2))
    else:
        vw, vh = 1800.0, 1120.0
    out_w = TARGET_W
    out_h = round(out_w * vh / vw)

    # 用 HTML 包一层，固定画布尺寸，避免浏览器滚动条/默认边距影响截图
    raw = re.sub(r'\swidth="[^"]*"', "", raw, count=1)
    raw = re.sub(r'\sheight="[^"]*"', "", raw, count=1)
    html = (
        "<html><head><meta charset='utf-8'><style>"
        "html,body{margin:0;padding:0;background:#fff;overflow:hidden;}"
        f"svg{{display:block;width:{out_w}px;height:{out_h}px;}}"
        "</style></head><body>"
        + raw.replace("<svg", f'<svg width="{out_w}" height="{out_h}"', 1)
        + "</body></html>"
    )
    tmp_dir = Path(tempfile.mkdtemp())
    html_path = tmp_dir / "fig.html"
    html_path.write_text(html, encoding="utf-8")

    png = svg.with_suffix(".png")
    # 截图路径必须是绝对路径，否则 Chrome 以自身工作目录解析会报"拒绝访问"
    profile = Path(tempfile.mkdtemp())
    cmd = [
        browser(),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        f"--user-data-dir={profile}",
        f"--window-size={out_w + PAD_W},{out_h + PAD_H}",
        "--default-background-color=FFFFFFFF",
        f"--screenshot={png.resolve()}",
        html_path.as_uri(),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"{svg.name} -> {png.name}  {out_w}x{out_h}")
    return png


if __name__ == "__main__":
    here = Path(__file__).parent
    targets = [Path(a) for a in sys.argv[1:]] or sorted(here.glob("*.svg"))
    for svg in targets:
        render(Path(svg) if Path(svg).is_absolute() else here / svg)
