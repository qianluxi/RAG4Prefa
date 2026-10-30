from __future__ import annotations

from rag4prefa.domain import Filter
from rag4prefa.filters import parse_filter

PROJECTS = ["广州项目", "海洋学院", "苏州华贸超限报告"]


def test_project_name_detected_as_substring() -> None:
    filt = parse_filter("海洋学院的结构做法有哪些？", PROJECTS)
    assert filt.projects == ["海洋学院"]


def test_year_range() -> None:
    filt = parse_filter("2020-2023年的超限报告", PROJECTS)
    assert (filt.year_from, filt.year_to) == (2020, 2023)


def test_year_after_and_before() -> None:
    assert parse_filter("2022年以后的项目", PROJECTS).year_from == 2022
    assert parse_filter("2019年之前的项目", PROJECTS).year_to == 2019


def test_recent_years_uses_current_year() -> None:
    filt = parse_filter("近三年的做法", PROJECTS, current_year=2026)
    assert (filt.year_from, filt.year_to) == (2023, 2026)


def test_single_year() -> None:
    filt = parse_filter("2021年的方案", PROJECTS)
    assert (filt.year_from, filt.year_to) == (2021, 2021)


def test_target_year_is_not_a_document_year() -> None:
    """政策目标年份不是资料年份。

    「到2027年，老旧住宅电梯要怎么处理？」的正文在 2024 年的文件里。按年份筛会把
    gold 判成「不在本题筛选范围」——这是检索前的静默否决，召回、重排、词表都救不回。
    """
    for question in (
        "到2027年，老旧住宅电梯要怎么处理？",
        "2027年底前要完成多少户改造？",
        "比2020年增长了百分之多少？",
        "到2027年标准体系要基本建成吗？",  # 紧跟「标准」也不当资料年份
        "到2027年的文件覆盖率要达到多少？",  # 紧跟「的文件」也不当资料年份
    ):
        filt = parse_filter(question, PROJECTS)
        assert (filt.year_from, filt.year_to) == (None, None), question


def test_document_year_still_filters() -> None:
    """明确指向资料年份的写法不能被上面那条规则误伤。"""
    assert parse_filter("2021年发布的文件里怎么规定的？", PROJECTS).year_from == 2021
    assert parse_filter("2020年版规范怎么写的？", PROJECTS).year_from == 2020
    assert parse_filter("2020年文件里有没有提？", PROJECTS).year_from == 2020
    assert parse_filter("2021年的方案", PROJECTS).year_from == 2021


def test_target_year_next_to_document_year() -> None:
    """同一句里既有目标年又有资料年时，只认资料年。"""
    filt = parse_filter("到2027年，相比2021年发布的文件有什么变化？", PROJECTS)
    assert (filt.year_from, filt.year_to) == (2021, 2021)

    filt = parse_filter("2021年和2023年的文件有什么不同？", PROJECTS)
    assert (filt.year_from, filt.year_to) == (2021, 2023)


def test_since_year() -> None:
    assert parse_filter("2019年以来有哪些文件？", PROJECTS).year_from == 2019


def test_quoted_keyword() -> None:
    filt = parse_filter("关于“缓粘结预应力”的条文", PROJECTS)
    assert filt.keyword == "缓粘结预应力"


def test_no_conditions_means_empty_filter() -> None:
    assert parse_filter("这个项目怎么做", []).is_empty


def test_request_filter_wins_over_parsed() -> None:
    parsed = parse_filter("2020年的方案", PROJECTS)
    request = Filter(projects=["广州项目"])
    merged = request.merge(parsed)
    assert merged.projects == ["广州项目"]  # 请求显式指定优先
    assert merged.year_from == 2020  # 未指定的部分由问题解析补齐


# ── 年份区间：资料年 vs 任务期 ──────────────────────────────


def test_year_range_as_task_period_is_not_a_document_year() -> None:
    """专家复审给的原句：2021—2023 是改造任务的时间区间，不是文件年份。

    区间此前是无条件命中的，于是这句话会把检索范围缩成 2021-2023 发的文件——
    而问的是"这三年干得怎么样"，检索前的静默否决，后面怎么排都救不回来。
    """
    filt = parse_filter("到2027年，2021—2023年间老旧小区改造任务进展如何？", PROJECTS)
    assert (filt.year_from, filt.year_to) == (None, None)


def test_year_range_followed_by_period_words_is_ignored() -> None:
    for question in (
        "2021—2023年间完成的改造有多少？",
        "2021年至2023年期间的政策有什么变化？",
        "2021-2023年内开工的项目进展如何？",
        "2021—2023年期间完成的改造有哪些？",
    ):
        filt = parse_filter(question, PROJECTS)
        assert (filt.year_from, filt.year_to) == (None, None), question


def test_year_range_with_document_marker_still_filters() -> None:
    """放宽不等于放弃：明确在说"哪几年的文件"时照旧筛选。

    只写"任务期不筛"的话，把 `_range_span` 改成恒返回 None 也能全绿——那等于把一个
    能用的过滤维度删掉了。
    """
    assert parse_filter("2021—2023年的文件怎么规定？", PROJECTS).year_from == 2021
    assert parse_filter("2021—2023年的文件怎么规定？", PROJECTS).year_to == 2023
    assert parse_filter("2020-2023年的超限报告", PROJECTS).year_to == 2023
    # 区间后面直接接资料名词、没有"的"，也按资料年处理（历史行为，未变）。
    assert parse_filter("2021-2023年装配式装修政策有哪些？", PROJECTS).year_from == 2021
