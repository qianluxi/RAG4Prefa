"""`scripts/ocr_audit.py` 的判据。

这个脚本不进线上链路，但它是识别稿唯一的自动质量线索——判据错了比没有更糟：
人会照着它去核对，核对完发现是误报，下次就不信它了。所以这里测的不是"能跑"，
是**它报的每一处是不是真该报**。

两个反面用例是实测撞出来的（GB 55038 的表 7.4.1→7.4.5、JGJ/T 467 的图 5.2.6），
第一版把表号按条文号那套"连续性"查，两条全误报。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ocr_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("ocr_audit", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # 必须先登记再 exec：模块里有 @dataclass，dataclasses 会回头去
    # `sys.modules[cls.__module__]` 取注解，没登记就抛 AttributeError。
    sys.modules["ocr_audit"] = module
    spec.loader.exec_module(module)
    return module


audit = _load()


def test_table_that_is_only_referenced_is_reported():
    """正文引用了表 4.3.10，全文却没有这张表的表头——就是它漏了。"""
    text = "4.3.10 骨架隔墙的构造应符合表 4.3.10 的规定。\n4.3.11 面板接缝应做防裂处理。"
    assert audit._tables_without_body(text) == ["表 4.3.10 被引用，但全文找不到它的表头"]


def test_table_with_header_row_is_not_reported():
    text = "表 4.3.10 骨架隔墙构造\n构造层 | 材料 | 厚度\n龙骨 | 轻钢 | 75mm"
    assert audit._tables_without_body(text) == []


def test_table_numbers_need_not_be_consecutive():
    """GB 55038 的实际形状：表 7.4.1 之后直接是表 7.4.5。

    中间那三个编号（7.4.2~7.4.4）是**条文号**，规范里本来就没有表 7.4.2。
    按连续性查会报"缺 3 张表"，三条正文其实都在库里——第一版就是这么错的。
    """
    text = (
        "7.4.1 住宅建筑应设供配电系统，主要用电负荷等级不应低于表 7.4.1 的规定。\n"
        "表 7.4.1 住宅建筑主要用电负荷等级\n"
        "住宅建筑高度 | 主要用电负荷名称 | 用电负荷等级\n"
        "7.4.2 每套住宅应设电能表。\n"
        "7.4.3 每套住宅应设家居配电箱。\n"
        "7.4.4 住宅照明回路应分别设置。\n"
        "7.4.5 电源插座的设置应符合表 7.4.5 的规定。\n"
        "表 7.4.5 每套住宅电源插座的设置要求及数量\n"
        "名称 | 设置要求 | 数量\n"
    )
    assert audit._tables_without_body(text) == []


def test_figure_numbers_are_not_audited_as_tables():
    """图的编号天然不连续，没有表头可对照——不查。

    JGJ/T 467 有图 5.2.4、图 5.2.5、图 5.2.7，中间没有图 5.2.6，因为 5.2.6 是
    条文。第一版把它算进"表/图"池，报了"缺图 5.2.6"。
    """
    text = (
        "5.2.4 应预留安装尺寸（图 5.2.4-1）。\n图 5.2.4-1 整体卫生间壁板预留安装尺寸\n"
        "5.2.5 设置外窗时应协同设计（图 5.2.5-1）。\n图 5.2.5-1 整体卫生间外窗开设尺寸\n"
        "5.2.6 设备管线穿越主体结构时应协调。\n"
        "5.2.7 门的设计选型应与内装设计协调（图 5.2.7）。\n图 5.2.7 门洞口中心线\n"
    )
    assert audit._tables_without_body(text) == []


def test_separator_row_alone_does_not_count_as_a_header():
    """只有 `|---|---|` 说明识别出了表格骨架、表头丢了——仍要报。"""
    text = "表 4.3.10 骨架隔墙构造\n|---|---|\n4.3.11 面板接缝应做防裂处理。"
    assert audit._tables_without_body(text) == ["表 4.3.10 被引用，但全文找不到它的表头"]


def test_one_qualified_occurrence_is_enough():
    """同一张表被引用两处，只要定义那处有表头，就不报。

    引用**必须排在定义前面**，否则这条测试没有鉴别力：如果第一处引用后面紧跟
    表头，那么"只查第一处"的实现也能过（实测变异存活）。规范里"先引用、后给表"
    是常态，识别出来的片段又常把表切到下一个片段里，所以这不是造出来的场景。
    """
    text = (
        "3.0.5 主要用电负荷等级应符合表 3.0.5 的规定。\n"
        "3.0.6 其它应符合表 3.0.5 的规定。\n"
        "表 3.0.5 用电负荷等级\n"
        "负荷名称 | 等级\n"
    )
    assert audit._tables_without_body(text) == []


def test_clause_gaps_still_catch_what_was_really_lost():
    """条文号连续性这条是真有用的：JGJ/T 477 的 5.3.2 整行消失就是这么发现的。"""
    assert audit._gaps(
        [(5, 3, 1), (5, 3, 3), (5, 3, 4)], "条文"
    ) == ["条文 5.3 缺 5.3.2"]


@pytest.mark.parametrize("triples", [[(5, 3, 1)], [(5, 3, 1), (5, 4, 1)]])
def test_single_number_in_a_section_is_not_a_gap(triples):
    """一条编号无从判断连续性，不该报。"""
    assert audit._gaps([tuple(t) for t in triples], "条文") == []
