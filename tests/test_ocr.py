"""OCR 路径的离线测试。

全部用假识别器，不联网：这条路的**接线**（哪一页送去识别、结果怎么折成章节、
缓存怎么保证可重入）与"模型认字准不准"是两件事，后者只能靠真实文档人工抽检，
前者必须每次提交都跑。

样本用 pymupdf 现造：`doc.new_page()` 之后一个字都不写，就是一页没有文字层的
"扫描页"——与真实扫描件在 `get_text()` 这一层没有任何区别。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf", reason="pdf 解析需要 pymupdf")

from rag4prefa import ingest, repo  # noqa: E402
from rag4prefa.adapters.ocr import OcrError, OcrOptions, VlmOcr, clean_markdown  # noqa: E402
from rag4prefa.adapters.parsers import ParseReport, parse_document_report  # noqa: E402
from rag4prefa.adapters.parsers.pdf import (  # noqa: E402
    _markdown_events,
    _page_needs_ocr,
)
from rag4prefa.adapters.vectors import NumpyStore  # noqa: E402
from rag4prefa.config import Settings  # noqa: E402
from tests.conftest import FakeEmbedder  # noqa: E402

_CN = "china-s"
_TITLE = "第1章 总则"
_SECTION = "1.1 适用范围"
_BODY = "本工程为装配式框架结构，地上十二层，标准层层高四点二米，抗震设防烈度六度。"


def _make_pdf(path: Path, pages: list[list[tuple[float, str, float]]]) -> Path:
    """`[]` 表示这一页没有文字——即"扫描页"。

    扫描页上还要画一道独一无二的线：真实的两页扫描图不会逐字节相同，而两张完全
    一样的页图会被"同图只识别一次"合并，测出来的调用次数就跟真实情况对不上了。
    """
    doc = pymupdf.open()
    for index, lines in enumerate(pages):
        page = doc.new_page()
        if not lines:
            y = 100 + index * 40
            page.draw_line((60, y), (400, y), color=(0.2, 0.2, 0.2))
            continue
        for y, text, size in lines:
            page.insert_text((72, y), text, fontname=_CN, fontsize=size)
    doc.save(str(path))
    doc.close()
    return path


def _text_page() -> list[tuple[float, str, float]]:
    return [(72, _TITLE, 18.0), (110, _BODY, 10.0)]


def _section_page() -> list[tuple[float, str, float]]:
    """第二页的标题与正文都要和第一页不同。

    页图逐字节相同的两页会被"同图只识别一次"合并，然后用例就只能看到一次调用，
    测出来的东西和真实文档对不上。
    """
    return [(72, _SECTION, 18.0), (110, _BODY.replace("十二层", "十八层"), 10.0)]


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    """两页有文字层 + 一页扫描。这是真实公文的常见形态：封面扫描、正文可复制。"""
    return _make_pdf(tmp_path / "混合.pdf", [_text_page(), _section_page(), []])


@pytest.fixture
def scanned_pdf(tmp_path: Path) -> Path:
    return _make_pdf(tmp_path / "扫描.pdf", [[], [], []])


class _RecordingOcr:
    """记录被问过的页码，返回可配置的 Markdown。"""

    def __init__(self, markdown: str = "# 第2章 识别要求\n\n识别出来的正文。") -> None:
        self.markdown = markdown
        self.pages: list[int] = []

    def __call__(self, png: bytes, page: int) -> str:
        assert png[:8] == b"\x89PNG\r\n\x1a\n", "送进来的必须是 PNG"
        self.pages.append(page)
        return self.markdown


def _options(fn, **kwargs) -> OcrOptions:
    return OcrOptions(fn=fn, **kwargs)


# ── 逐页判据 ───────────────────────────────────────────────


def test_page_needs_ocr_flags_blank_pages_but_not_short_ones() -> None:
    assert _page_needs_ocr("") is True
    assert _page_needs_ocr("   \n  ") is True
    # 一页只有"图 4-1 悬挑节点大样"也是**有**文字层：图注是正文，送去识别只会得到
    # 一张图里本来就没有的字
    assert _page_needs_ocr("图 4-1 悬挑节点大样") is False
    assert _page_needs_ocr(_BODY) is False
    # 但整页只剩一个页码（页眉页脚剔除后的残留），等同于没有文字层
    assert _page_needs_ocr("3") is True


def test_page_needs_ocr_flags_garbled_font_pages() -> None:
    """字体缺 ToUnicode 的页：字符数很健康，取出来的却不是汉字。

    只查字符数的判据对这一类完全失效——那正是它危险的地方，一段乱码能一路入库，
    把 BM25 的文档频率和向量空间一起拖坏。
    """
    garbled = "!" * 40 + "#$%&'()*+," * 40
    assert len(garbled.strip()) > 300
    assert _page_needs_ocr(garbled) is True


# ── 混合型文档：现在是"报出来"，不是"静默丢" ─────────────────


def test_mixed_pdf_reports_missing_pages_instead_of_dropping_them(mixed_pdf: Path) -> None:
    """OCR 关着时，缺文字层的页不能一句提示都没有。

    以前的行为是：整份闸门看全文（两页有字就过关），扫描那页在 `_collect` 里
    取到空列表，于是**一个字都不进来，也没有任何记录**。用户以为入了三页，
    实际只有两页参与检索——问到了就答不上，还会被当成"检索没做好"。
    """
    sections, report = parse_document_report(mixed_pdf)

    assert report.pages == 3
    assert report.missing_pages == (3,)
    assert report.ocr_pages == ()
    note = report.note()
    assert note is not None and "第 3 页" in note

    # 有文字的两页照常解析，不能因为缺一页就整份拒绝
    assert any(_TITLE in s.title for s in sections)
    assert {s.page for s in sections if s.page} == {1, 2}


def test_all_blank_pdf_without_ocr_names_the_page_count(scanned_pdf: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_document_report(scanned_pdf)
    message = str(excinfo.value)
    assert "共 3 页" in message
    assert "R4P_OCR_MODE" in message  # 告诉用户怎么补，而不是只报"不行"


# ── OCR 开启后 ─────────────────────────────────────────────


def test_ocr_only_asks_for_pages_without_text(mixed_pdf: Path) -> None:
    ocr = _RecordingOcr()
    sections, report = parse_document_report(mixed_pdf, ocr=_options(ocr))

    assert ocr.pages == [3], "有文字层的页也送去识别，等于每页多花一次钱"
    assert report.ocr_pages == (3,)
    assert report.missing_pages == ()
    assert any(s.title == "第2章 识别要求" for s in sections)


def test_ocr_page_keeps_its_real_physical_page(scanned_pdf: Path) -> None:
    """识别出来的章节必须落在**它自己那一页**上，不是第 1 页。

    引用卡片上写"第 2 页"、翻过去却是别的内容，这个库就没有一处可以核对了。
    """
    asked: list[int] = []

    def only_page_two(png: bytes, page: int) -> str:
        asked.append(page)
        return "# 第1章 总则\n\n只有第二页有这一句。" if page == 2 else "<EMPTY>"

    sections, report = parse_document_report(scanned_pdf, ocr=_options(only_page_two))

    chapter = next(s for s in sections if s.title == "第1章 总则")
    assert chapter.page == 2, "页码取错了，引用就指向别的内容"
    assert asked == [1, 2, 3]
    assert report.empty_pages == (1, 3), "识别过但没内容的页要单独报，别说成'已识别'"


def test_ocr_reports_pages_the_model_says_are_blank(scanned_pdf: Path) -> None:
    """模型回 <EMPTY> 要记成"这页没内容"，不能变成一节空章节。

    空白页、整页插图在工程文档里很常见。造出一节空章节的代价是检索时多一条
    永远匹配不上的片段，还会在目录里显示一个没有正文的标题。
    """
    sections, report = parse_document_report(
        scanned_pdf, ocr=_options(lambda png, page: "<EMPTY>")
    )

    assert sections == []
    assert report.empty_pages == (1, 2, 3)
    assert report.ocr_pages == ()
    note = report.note() or ""
    assert "识别后仍无文字" in note


def test_identical_page_images_are_sent_once(tmp_path: Path) -> None:
    """同一批里逐字节相同的页图只识别一次。

    空白页、重复的模板页渲染出来完全一样。不合并的话几个线程会同时查缓存、
    同时未命中，同一张图付几次钱——缓存救不了并发下的这一刻。
    """
    blank = tmp_path / "全空白.pdf"
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page().draw_line((60, 100), (400, 100))
    doc.save(str(blank))
    doc.close()

    calls: list[int] = []

    def record(png: bytes, page: int) -> str:
        calls.append(page)
        return "<EMPTY>"

    _, report = parse_document_report(blank, ocr=_options(record))
    assert calls == [1], "两张一模一样的页图应该只问一次"
    assert report.empty_pages == (1, 2), "但两页都要记成'识别后仍无文字'"


def test_unmarked_numbered_line_is_still_a_heading() -> None:
    """模型漏标 `#` 时，编号是唯一还能证明"这是标题"的文字证据。

    实测 v1 提示词下 Qwen3-VL-8B 在一份 24 页公文的第 2、8、9 页一条 `#` 都不给，
    17 636 个字因此全落进同一节、章节路径全空。编号是文档自带的，不该跟着模型一起丢。
    """
    events = _markdown_events("（二）公共建筑类项目\n地上总建筑面积超过 8000 平方米。", page=2)
    assert events == [
        (0, "（二）公共建筑类项目", 2),  # 0 = 相对级别，由 build_sections 挂到下一层
        (None, "地上总建筑面积超过 8000 平方米。", 2),
    ]


def test_unmarked_sentence_starting_with_a_number_stays_body() -> None:
    """没有 `#`、又只是句子的行，绝不能因为"以数字开头"变成标题。

    这类行在公文里遍地都是："1. 新建居住建筑类项目应落实装配式建筑；" 只要被当成
    标题，一节正文就会被切成几十个空章节。
    """
    sentence = "1. 新建居住建筑类项目应落实装配式建筑；"
    assert _markdown_events(sentence, page=3) == [(None, sentence, 3)]


def test_unmarked_long_line_is_not_promoted_even_with_a_number() -> None:
    """跨页断在半句的长行，不能因为"开头是 4. 又没标点"就当标题。

    实测：一条 55 字的正文因为标题名被截在两页之间，成了章节标题，后面整段正文
    都挂到了它名下。中文标题极少超过 40 字，这是唯一能挡住它的判据。
    """
    truncated = (
        "4. 施工图审查机构应按照《计算细则》核对项目装配式建筑指标，"
        "并将核对后的指标填写在施工图设计文件审查合"
    )
    assert len(truncated) > 40
    assert _markdown_events(truncated, page=5) == [(None, truncated, 5)]


def test_ocr_mode_all_reprocesses_pages_that_already_have_text(mixed_pdf: Path) -> None:
    ocr = _RecordingOcr()
    _, report = parse_document_report(mixed_pdf, ocr=_options(ocr, mode="all"))
    assert ocr.pages == [1, 2, 3]
    # 全部交给 OCR 后，字号那条路一页都没解析到——两条路不能同时产出正文，
    # 否则同一段话会进库两遍，引用出现两个一模一样的角标
    assert report.ocr_pages == (1, 2, 3)


def test_ocr_max_pages_stops_a_mis_dropped_file(scanned_pdf: Path) -> None:
    with pytest.raises(ValueError) as excinfo:
        parse_document_report(
            scanned_pdf, ocr=_options(lambda png, page: "", max_pages=2)
        )
    assert "超过上限 2" in str(excinfo.value)


# ── Markdown → 事件流 ────────────────────────────────────


def test_heading_that_is_actually_a_sentence_is_demoted_to_body() -> None:
    """实测 32B 把「独立建筑单体可不实施装配式建筑；」标成了一级标题。

    那是上一段的续行。只看 `#` 就会凭空长出一棵和正文无关的树，而且每个假章节
    都只有一行。标题过 `looks_like_title()` 之后这类句子一律降格为正文。
    """
    events = _markdown_events("# 独立建筑单体可不实施装配式建筑；", page=3)
    assert events == [(None, "独立建筑单体可不实施装配式建筑；", 3)]


def test_numbering_decides_the_level_not_the_hash_count() -> None:
    """`#` 说"是标题"，编号说"第几级"。两者冲突时以编号为准。"""
    events = _markdown_events("### 4.2.1 悬挑构造", page=7)
    # 编号 4.2.1 -> 三级；`###` 若被当真也是三级，但换成两级编号就能看出区别
    assert events == [(3, "4.2.1 悬挑构造", 7)]

    events = _markdown_events("# 1.1 一般规定", page=7)
    assert events == [(2, "1.1 一般规定", 7)], "如果按 # 的个数判，这里会变成一级"


def test_markdown_table_rows_become_plain_body_lines() -> None:
    events = _markdown_events("| 项目 | 数值 |\n|---|---|\n| 预制率 | 40% |", page=2)
    assert events == [
        (None, "项目 | 数值", 2),
        (None, "预制率 | 40%", 2),
    ], "分隔行 |---|---| 必须丢掉，否则它自己会变成一条正文与一条向量"


def test_footer_page_numbers_from_the_model_are_dropped() -> None:
    events = _markdown_events("# 第1章 总则\n\n- 3 -\n\n正文。", page=2)
    assert events == [(1, "第1章 总则", 2), (None, "正文。", 2)]


def test_clean_markdown_strips_code_fences_and_empty_mark() -> None:
    assert clean_markdown("```markdown\n# 标题\n```") == "# 标题"
    assert clean_markdown("<EMPTY>") == ""
    assert clean_markdown("  ") == ""


# ── 缓存：可重入的前提（不是提速） ────────────────────────────


class _CountingVlm:
    """每次调用输出都不同——视觉模型的真实性质。缓存没兜住，id 就会全变。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def model(self) -> str:
        return "fake-vlm#v1"

    calls = 0

    def __call__(self, png: bytes, page: int) -> str:
        type(self).calls += 1
        return f"# 第1章 总则\n\n第 {page} 页，第 {type(self).calls} 次识别。"


def test_reingest_of_a_scanned_doc_keeps_the_same_chunk_ids(
    conn: sqlite3.Connection,
    settings: Settings,
    scanned_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一份扫描件重跑，chunk_id 必须一模一样。

    `chunk_id` 是内容寻址的：正文差一个字就是另一批 id。视觉模型每次输出都不同，
    没有缓存时 `--force` 重入库会得到"内容相同、id 全变"的一份新文档，旧向量原地
    成孤儿——而 `verify` 只比 id 集合是否自洽，查不出"这份已经不是你上次那份"。
    """
    settings.ocr_mode = "auto"
    monkeypatch.setattr(ingest, "VlmOcr", _CountingVlm)
    _CountingVlm.calls = 0

    store = NumpyStore(conn, FakeEmbedder.dim)
    embedder = FakeEmbedder()
    doc_id = ingest._doc_id("测试项目", scanned_pdf.name)

    first = ingest.ingest_file(
        conn, store, embedder, settings, path=scanned_pdf, project_name="测试项目"
    )
    assert first.status == "indexed", first.error
    ids_first = set(repo.chunk_ids_of_doc(conn, doc_id))
    calls_first = _CountingVlm.calls
    assert calls_first == 3, "三页都该识别一次"

    second = ingest.ingest_file(
        conn, store, embedder, settings, path=scanned_pdf, project_name="测试项目", force=True
    )

    assert second.status == "indexed", second.error
    assert set(repo.chunk_ids_of_doc(conn, doc_id)) == ids_first
    assert _CountingVlm.calls == calls_first, "第二次应该全部命中缓存，一次调用都不该发"


def test_empty_ocr_result_is_cached_not_re_requested(
    conn: sqlite3.Connection, settings: Settings
) -> None:
    """空白页识别出空串也要存：不存的话每次重跑都为同一张空白图片再付一次费。

    `get_ocr_cache` 返回 None 与 "" 必须分开——把空当"没识别过"，这件事就发生了；
    反过来把"没识别过"当空，整份扫描件会变成零正文，而任务是"成功"的。
    """
    calls: list[int] = []

    class Blank:
        model = "m#v1"

        def __call__(self, png: bytes, page: int) -> str:
            calls.append(page)
            return ""

    ocr, close = ingest._cached_ocr(conn, "m#v1", Blank)

    assert ocr(b"same-bytes", 5) == ""
    assert ocr(b"same-bytes", 5) == ""
    assert calls == [5]
    close()


def test_cache_key_includes_the_model(conn: sqlite3.Connection) -> None:
    repo.put_ocr_cache(conn, "sha", "model-a#v1", "甲")
    repo.put_ocr_cache(conn, "sha", "model-b#v1", "乙")
    assert repo.get_ocr_cache(conn, "sha", "model-a#v1") == "甲"
    assert repo.get_ocr_cache(conn, "sha", "model-b#v1") == "乙"
    assert repo.get_ocr_cache(conn, "sha", "model-c#v1") is None


# ── 配置闸门 ─────────────────────────────────────────────


def test_ocr_off_builds_nothing(settings: Settings, conn: sqlite3.Connection) -> None:
    settings.ocr_mode = "off"
    assert ingest._build_ocr(settings, conn) is None


def test_ocr_is_off_by_default() -> None:
    """默认必须关着。这是账单闸门：开着意味着"任何人上传一份 300 页扫描件"
    都会变成几百次三方调用，而默认配置不该替用户做这个决定。"""
    assert Settings(_env_file=None).ocr_mode == "off"


def test_missing_credentials_fail_before_any_page_is_sent(settings: Settings) -> None:
    """缺 Key 要在装配阶段就报，别等到第 1 页要识别时才发现。

    那时用户已经等了半分钟，而且报错长得像"这个文件有问题"，会去反复重传。
    """
    settings.ocr_mode = "auto"
    settings.ocr_base_url = ""
    settings.ocr_api_key = ""
    with pytest.raises(OcrError) as excinfo:
        VlmOcr(settings)
    assert "R4P_OCR_BASE_URL" in str(excinfo.value)
    assert "R4P_OCR_API_KEY" in str(excinfo.value)


def test_docx_report_is_empty(conn: sqlite3.Connection, docx_file: Path) -> None:
    """docx 里不存在"某页没文字层"这件事——正文一定在文件里。"""
    _, report = parse_document_report(docx_file)
    assert report == ParseReport()


def test_ocr_toc_page_is_dropped_instead_of_growing_a_fake_tree(scanned_pdf: Path) -> None:
    """识别出来的目录页必须整页丢掉，否则会凭空长出一棵树。

    `_toc_pages()` 的判据跑在文字层上，而这类文件（整页图 + 一行水印、或字体缺
    ToUnicode）根本没有可用文字层——判据看不到目录，目录页就进了 OCR。
    **点线引导符活不过识别**：实测 `1 General provisions ......... (1)` 会被还原成
    `1 General provisions (1)`，所以判目录得用"行尾是页码 + 行都很短"这条证据。
    """
    toc = (
        "# 目录\n\n1 总则 (1)\n2 术语 (2)\n3 基本规定 (3)\n"
        "4 装配率计算 (4)\n5 评价 (11)\n术语说明 (12)\n条文说明 (13)"
    )

    def fake(png: bytes, page: int) -> str:
        return toc if page == 1 else f"# 第{page}章 正文\n\n第 {page} 页的正文内容。"

    sections, report = parse_document_report(scanned_pdf, ocr=_options(fake))

    assert report.toc_pages == (1,), "目录页要单独报出来，别混进'识别后无文字'"
    assert report.ocr_pages == (2, 3), "目录页识别出来了，但它的内容没进库"
    assert [s.title for s in sections] == ["第2章 正文", "第3章 正文"]
    # 这句话要说得具体：只报"有页被跳过"用户不知道该信哪一段
    assert "第 1 页是目录，已跳过" in (report.note() or "")
