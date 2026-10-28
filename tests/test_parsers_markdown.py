"""Markdown 解析：结构从 `#` 来，没有 `#` 时从编号来。

这份用例的重点不是"能读出字"，而是三件在真实语料上真出过问题的事：

1. `#` 必须从标题文本里剥掉。`Section.title` 进章节路径，章节路径进 `chunk_id`——
   留着 `#` 会让同一份文档在"改了标题级别"后整份换 id。
2. 抓下来的政策页面，`索引号` / `发布时间` 恰好排在正文之前，它们的**值**
   （`002482242/2023-79897`、`2023-04-23 04:48`）长得像编号，会被读成一级标题，
   于是整篇文章的第一个"章"就是流水号，章节树从这里歪到底。
3. 网页模板行（发布机构 / 浏览次数 / 打印关闭）不是正文，混进去只会让每个片段
   都带着同一串模板字，BM25 上表现为"搜什么都能命中"。
"""

from __future__ import annotations

from pathlib import Path

from rag4prefa.adapters.parsers import SUPPORTED_SUFFIXES, parse_document_report
from rag4prefa.adapters.parsers.markdown import parse_markdown
from rag4prefa.domain import SectionKind


def _write(tmp_path: Path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_atx_headings_become_levels_and_lose_the_hashes(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "# 总则\n\n正文一。\n\n## 材料\n\n正文二。\n\n### 板材\n\n正文三。\n",
    )
    sections = parse_markdown(path)
    assert [s.title for s in sections] == ["总则", "材料", "板材"]
    assert [s.level for s in sections] == [1, 2, 3]
    assert all("#" not in s.title for s in sections), "标题进了章节路径，也就进了 chunk_id"
    assert sections[1].path == ["总则"]


def test_numbered_policy_text_without_hashes_still_builds_a_tree(tmp_path: Path) -> None:
    """网页正文常常一条 `#` 都没有，只有「一、」「（一）」。"""
    path = _write(
        tmp_path,
        "浙江省开展装配化装修试点工作方案\n"
        "一、总体要求\n"
        "（一）指导思想。以提升装配式建筑品质为目标。\n"
        "二、试点范围\n"
        "确定杭州、宁波、绍兴、台州等4个设区市为试点市。\n",
    )
    sections = parse_markdown(path)
    titles = [s.title for s in sections]
    assert "一、总体要求" in titles
    assert "二、试点范围" in titles
    chapter = next(s for s in sections if s.title == "一、总体要求")
    assert chapter.level == 1
    assert chapter.kind == SectionKind.chapter
    assert "指导思想" in chapter.text


def test_date_and_serial_lines_are_not_headings(tmp_path: Path) -> None:
    """索引号的值与发布时间不能当章——它们是模板的**值**，不是结构。

    断言的是"第一个 section 是前言"，不是"文件里有日期行"：后者在解析器把日期
    读成一级标题时同样成立，等于什么都没测。
    """
    path = _write(
        tmp_path,
        "2024-09-10\n"
        "002489639/2024-00622\n"
        "2023-04-23 04:48\n"
        "2023年4月23日\n"
        "厨卫换新样板房顺利完工，补助比例为合同价的36%。\n",
    )
    sections = parse_markdown(path)
    assert sections[0].title == "前言"
    assert sections[0].level == 0
    assert "36%" in sections[0].text
    assert all(s.title not in {"2024-09-10", "002489639/2024-00622"} for s in sections)
    assert all("2023" not in s.title for s in sections)
    # 光"不是标题"不够：这些行的值是流水号与时间戳，当正文留着只会让前言变成一串编号
    body = "\n".join(s.text for s in sections)
    assert "2024-09-10" not in body
    assert "002489639/2024-00622" not in body


def test_serial_lines_behind_a_list_marker_are_not_headings(tmp_path: Path) -> None:
    """`- 2024-09-10` 这种写法在抓取整理的文件里很常见。

    裸行由 `_is_noise` 在剥列表标记之前挡掉；列表项要等标记剥掉、露出数字开头，
    才会被当成编号标题——判据在 `_split_heading` 里要再来一次，否则章节树第一层
    又是流水号。这条用例盯着这个"剥完才露出来"的路径。
    """
    path = _write(
        tmp_path,
        "- 2024-09-10\n- 002482242/2023-79897\n- 1.8 万元封顶的补助标准。\n",
    )
    sections = parse_markdown(path)
    assert sections[0].title == "前言", sections[0].title
    assert all("2024-09-10" not in s.title for s in sections)
    assert all("002482242" not in s.title for s in sections)
    body = "\n".join(s.text for s in sections)
    assert "2024-09-10" not in body
    assert "002482242" not in body


def test_web_template_lines_are_dropped_but_body_survives(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "首页\n"
        "当前位置：首页\n"
        "索引号: 002482242/2023-79897\n"
        "发布机构：浙江省住房和城乡建设厅\n"
        "文　　号：浙建管发〔2023〕50号\n"
        "浏览次数:\n"
        "装配化装修是采用干式工法的一种装修方式。\n"
        "打印关闭\n",
    )
    sections = parse_markdown(path)
    text = "\n".join(s.text for s in sections)
    assert "干式工法" in text
    for noise in ("首页", "当前位置", "索引号", "发布机构", "浏览次数", "打印关闭"):
        assert noise not in text, f"{noise} 是网页模板，不该进正文"


def test_front_matter_and_code_fences_do_not_become_headings(tmp_path: Path) -> None:
    """front matter 的 `---` 与代码块里的 `#` 都不是章节。"""
    path = _write(
        tmp_path,
        "---\n"
        "title: 试点方案\n"
        "date: 2024-09-10\n"
        "---\n"
        "\n"
        "## 安装\n"
        "\n"
        "```bash\n"
        "# 依赖先装好\n"
        "pip install rag4prefa\n"
        "```\n",
    )
    sections = parse_markdown(path)
    titles = [s.title for s in sections]
    body = "\n".join(s.text for s in sections)
    assert titles == ["安装"], titles
    assert "title: 试点方案" not in body, "front matter 是元信息，不是正文"
    # 代码块里的 `#` 是注释不是章节：它留在正文里（连 `#` 一起），不出现在标题里
    assert "# 依赖先装好" in body
    assert "pip install" in body, "代码块内容当正文保留"


def test_markdown_is_a_supported_format(tmp_path: Path) -> None:
    """注册到分派层：上传与目录导入都靠 `SUPPORTED_SUFFIXES` 判断收不收。"""
    assert ".md" in SUPPORTED_SUFFIXES
    path = _write(tmp_path, "# 标题\n\n正文。\n")
    sections, report = parse_document_report(path)
    assert [s.title for s in sections] == ["标题"]
    assert report.pages == 0, "md 没有物理页"
    assert report.note() is None


def test_gbk_encoded_file_is_read_not_rejected(tmp_path: Path) -> None:
    """抓取工具留下的 GBK 文件很常见，整份拒收会扔掉 90% 可用的正文。"""
    path = tmp_path / "gbk.md"
    path.write_bytes("# 总则\n\n装配化装修采用干式工法。\n".encode("gbk"))
    sections = parse_markdown(path)
    assert sections[0].title == "总则"
    assert "干式工法" in sections[0].text
