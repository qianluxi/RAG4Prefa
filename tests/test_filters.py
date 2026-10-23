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
