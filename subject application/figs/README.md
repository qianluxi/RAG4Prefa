# 申请书配图

三张图为正文唯一配图，分工互不重叠：

| 图 | 文件 | 回答的问题 | 正文位置 |
| --- | --- | --- | --- |
| 图 1 | `figure1_research_framework` | 为什么研究、研究什么、形成什么 | 一、1.1 研究目的末尾 |
| 图 2 | `figure2_system_architecture` | 技术上如何实现、增量在哪一层 | 三、技术路线主链路段之后 |
| 图 3 | `figure3_public_interaction_flow` | 系统如何服务公众、如何避免错答 | 二、(一) 4. 智能体与效果评价之后 |

## 尺寸与字号（先看这条再改图）

图按 **15.5cm 宽** 插进 A4 正文，画布统一 **860 单位宽**，因此 1 单位 ≈ 0.018cm。
正文取 16 号字，打印后约 **8.2pt**；标签 13.5 号约 **6.9pt**。

早期版本画布 1800~2000 宽、字号 14，打印后只有 3pt 左右，在 A4 上根本看不清——
所以**改图时加内容请往高度方向长，不要加大画布宽度，也不要减小字号**。
目前文字量：图 1 约 130 字、图 2 约 130 字、图 3 约 130 字，节点均 ≤ 20 个。

## 文件说明

- `figstyle.py`：三张图共用的绘图层（配色、字号阶梯、`node` / `card` / `diamond` / `legend` 等元件）。改配色和字号只改这里。
- `create_figure1.py`、`create_figure2.py`、`create_figure3.py`：生成 SVG，均已可直接运行（不再依赖 cairosvg）。
- `*.svg`：矢量源，改字可读；`*.png` 由 `render_png.py` 渲染，Word 内嵌用 PNG。
- `render_png.py`：Chrome 无头渲染，3000px 宽。**注意**：截图区域比 `--window-size` 小 16×94px（窗口边框与标题栏），已用 `PAD_W/PAD_H` 补偿；不补偿会把图右侧和底部裁掉。
- `apply_figures.py`：把最新 PNG 换进 docx 并校正显示尺寸（宽固定 15.5cm，高按图片比例）。
- `../insert_figures.py`：首次插图用（插图片段落 + 图题），已插过就不必再跑。

## 重出图流程

```bash
python create_figure1.py                 # 改哪张跑哪个，产出 SVG
python render_png.py                     # SVG -> PNG（不带参数则全量）
python apply_figures.py ../<申请书>.docx  # 换进 docx（先在 WPS 里关闭文档）
```

配色约定：灰蓝 `#5B7E96` = 前期已具备能力，蓝绿 `#2A8C82` = 本课题拟研究，
深蓝 `#254E68` = 成果，橙色 `#C08A2E` = 判断与证据处置节点；三张图共用这一套图例。
