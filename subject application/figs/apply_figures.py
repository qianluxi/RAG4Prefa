"""把 figs 下最新的三张 PNG 换进申请书 docx，并按新图比例校正显示尺寸。

python-docx 不支持替换图片二进制，这里分两步：先改 XML 里的 extent（宽固定
FIG_W_CM，高按 PNG 实际比例算），再直接重写 docx 包内 word/media 下的三个文件。
图号与媒体的对应关系从 rels 现读，不依赖插入顺序。

用法：python apply_figures.py ../<申请书>.docx
"""

import os
import struct
import sys
import zipfile
from pathlib import Path

import docx
from docx.oxml.ns import qn

FIG_W_CM = 15.5
EMU_PER_CM = 360000

# Word 内 media 文件名 → 本地图（按插入顺序编号，与图 1/2/3 的对应关系不随段落顺序变化）
MEDIA_MAP = {
    "media/image1.png": "figure1_research_framework.png",
    "media/image2.png": "figure2_system_architecture.png",
    "media/image3.png": "figure3_public_interaction_flow.png",
}


def png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{path} 不是 PNG")
    return struct.unpack(">II", head[16:24])


def main(doc: Path) -> None:
    here = Path(__file__).parent
    d = docx.Document(doc)
    rels = d.part.rels
    cx = int(FIG_W_CM * EMU_PER_CM)

    for tbl in d.tables:
        for row in tbl.rows:
            for cell in row.cells[:1]:
                for p in cell.paragraphs:
                    blip = p._p.find(".//" + qn("a:blip"))
                    if blip is None:
                        continue
                    target = rels[blip.get(qn("r:embed"))].target_ref
                    name = MEDIA_MAP.get(target)
                    if not name:
                        raise SystemExit(f"未登记的媒体文件：{target}")
                    w, h = png_size(here / name)
                    cy = round(cx * h / w)
                    for el in p._p.iter():
                        if el.tag in (qn("wp:extent"), qn("a:ext")) and el.get("cx"):
                            el.set("cx", str(cx))
                            el.set("cy", str(cy))
                    print(f"{target} <- {name}  {FIG_W_CM:.2f} x {cy / EMU_PER_CM:.2f} cm")
    d.save(doc)

    # 重写包内 media（writestr 保留原条目压缩方式，其余条目原样拷贝）
    tmp = doc.with_suffix(".repack.docx")
    with zipfile.ZipFile(doc) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            key = item.filename[len("word/"):] if item.filename.startswith("word/") else None
            data = (here / MEDIA_MAP[key]).read_bytes() if key in MEDIA_MAP else zin.read(item.filename)
            zout.writestr(item, data)
    os.replace(tmp, doc)
    print("已替换图片：", doc.name)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
