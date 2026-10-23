from __future__ import annotations

import pytest

from rag4prefa.adapters.vectors import BackendUnavailable, MilvusStore, VectorItem
from rag4prefa.domain import Filter

pytest.importorskip("pymilvus")


@pytest.fixture
def store(tmp_path):
    try:
        return MilvusStore(str(tmp_path / "vectors.db"), "chunks_vec", dim=4)
    except BackendUnavailable as exc:  # 平台不支持时跳过，而不是让整个测试套红
        pytest.skip(f"Milvus 不可用：{exc}")


def test_upsert_search_and_filter(store: MilvusStore) -> None:
    store.upsert(
        [
            VectorItem("a", [1, 0, 0, 0], "甲项目", "kc01", 2024),
            VectorItem("b", [0, 1, 0, 0], "乙项目", "kc02", 2020),
            VectorItem("c", [1, 0, 0, 0], "甲项目", "kc01", 2020),
        ]
    )
    assert store.count() == 3

    hits = store.search([1, 0, 0, 0], limit=3)
    assert {h[0] for h in hits} == {"a", "b", "c"}
    assert hits[0][0] in {"a", "c"}
    assert hits[0][1] > 0.9

    only_jia = store.search([1, 0, 0, 0], limit=3, filt=Filter(projects=["甲项目"]))
    assert {h[0] for h in only_jia} == {"a", "c"}

    recent = store.search([1, 0, 0, 0], limit=3, filt=Filter(year_from=2021))
    assert {h[0] for h in recent} == {"a"}


def test_upsert_is_idempotent(store: MilvusStore) -> None:
    item = VectorItem("a", [1, 0, 0, 0], "甲项目", "kc01", 2024)
    store.upsert([item])
    store.upsert([item])
    assert store.count() == 1


def test_delete(store: MilvusStore) -> None:
    store.upsert([VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024)])
    store.delete(["a"])
    assert store.count() == 0


def test_reopened_collection_is_searchable(tmp_path) -> None:
    """新进程打开已有集合时必须先 load，否则状态是 released，search 直接报错。

    CLI 与 HTTP 服务每次都是新进程，这条路径曾经完全不可用。
    """
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        first = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    first.upsert([VectorItem("a", [1, 0, 0, 0], "甲项目", "kc01", 2024)])

    reopened = MilvusStore(uri, name, dim=4)  # 模拟进程重启
    hits = reopened.search([1, 0, 0, 0], limit=3)
    assert [h[0] for h in hits] == ["a"]


def test_rebuild_changes_dimension(tmp_path) -> None:
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        store = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    store.upsert([VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024)])

    store.rebuild(
        [
            VectorItem("a", [1, 0, 0, 0, 0, 0], "甲项目", "", 2024),
            VectorItem("b", [0, 1, 0, 0, 0, 0], "甲项目", "", 2024),
        ]
    )

    assert store.dim == 6
    assert store.count() == 2
    assert {h[0] for h in store.search([1, 0, 0, 0, 0, 0], limit=2)} == {"a", "b"}
