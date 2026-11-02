# 申请书配图

三张图为正文唯一配图，分工互不重叠：

| 图 | 文件 | 回答的问题 | 正文位置 |
| --- | --- | --- | --- |
| 图 1 | `figure1_research_framework` | 为什么研究、研究什么、形成什么 | 一、1.1 研究目的末尾 |
| 图 2 | `figure2_system_architecture` | 技术上如何实现、增量在哪一层 | 三、技术路线主链路段之后 |
| 图 3 | `figure3_public_interaction_flow` | 系统如何服务公众、如何避免错答 | 二、(一) 4. 智能体与效果评价之后 |

## 文件说明

- `*.svg`：编辑以它为准，矢量、可直接在浏览器/Illustrator 里改字。
- `*.png`：由 `render_png.py` 用本机 Chrome 无头渲染（3000px 宽，约 300dpi），Word 内嵌用这一个。
- `create_figure1.py`、`create_figure2.py`、`create_figure3.py`：生成脚本。图 3 的脚本与本目录直接配套可运行；图 1、图 2 的脚本是早期在 Linux 沙箱里写的（输出路径 `/mnt/data`、依赖 cairosvg），改图后请直接改 SVG 再渲染，或改脚本里的输出路径。
- `../insert_figures.py`：把三张 PNG 插回 docx 并加居中图题，可重复执行（会断言锚点段落）。

## 重出图流程

```bash
python create_figure3.py                 # 改图后先出 SVG
python render_png.py                     # SVG -> PNG（全量）
python ../insert_figures.py              # 插入 docx（注意先在 WPS 里关闭文档）
```

配色约定：灰蓝=前期已具备能力，蓝绿=本课题拟研究，橙色=判断与证据处置节点；三张图共用这一套图例。
