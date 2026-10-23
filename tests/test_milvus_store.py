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


def test_rebuild_same_dim_is_available_throughout_and_prunes_orphans(tmp_path) -> None:
    """同维度整体重建不删集合，并且要把不在新集合里的旧 id 清掉。"""
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        store = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    store.upsert(
        [
            VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024),
            VectorItem("旧孤儿", [0, 1, 0, 0], "甲项目", "", 2024),
        ]
    )

    store.rebuild([VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024)])

    assert store.count() == 1, "不在本次重建范围内的旧向量必须被清掉"
    assert [h[0] for h in store.search([1, 0, 0, 0], limit=5)] == ["a"]


def test_rebuild_leaves_no_staging_or_retired_collections(tmp_path) -> None:
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        store = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    store.rebuild([VectorItem("a", [1, 0, 0, 0, 0, 0], "甲项目", "", 2024)])

    names = set(store._client.list_collections())
    assert names == {name}, f"重建后不该留下中间集合：{names}"


def test_failed_rebuild_keeps_previous_index(tmp_path, monkeypatch) -> None:
    """暂存阶段就失败时，正式索引必须原样可用。"""
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        store = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    store.upsert([VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024)])

    def boom(*args, **kwargs):
        raise RuntimeError("模拟写入失败")

    monkeypatch.setattr(store, "_upsert_into", boom)
    with pytest.raises(RuntimeError, match="模拟写入失败"):
        store.rebuild([VectorItem("a", [1, 0, 0, 0, 0, 0], "甲项目", "", 2024)])

    assert store.count() == 1, "旧索引必须完好"
    assert [h[0] for h in store.search([1, 0, 0, 0], limit=3)] == ["a"]
    assert store.dim == 4, "维度不能被改成失败后的新维度"
    assert set(store._client.list_collections()) == {name}


def test_reopen_restores_retired_after_interrupted_swap(tmp_path) -> None:
    """切换过程被进程杀死（正式集合已让位、新集合还没顶上）时，重启要能自动还原。"""
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        first = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    first.upsert([VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024)])

    # 手工制造"切换做了一半"的状态：正式集合改名成 __retired，暂存集合已建好
    first._client.rename_collection(old_name=name, new_name=f"{name}__retired")

    reopened = MilvusStore(uri, name, dim=4)

    assert reopened.notes, "恢复动作必须留下说明"
    assert [h[0] for h in reopened.search([1, 0, 0, 0], limit=3)] == ["a"]
    assert set(reopened._client.list_collections()) == {name}


def test_reopen_rejects_dim_mismatch(tmp_path) -> None:
    """集合实际维度与配置不一致时必须明确失败，而不是写入后才报错。"""
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")

    with pytest.raises(BackendUnavailable, match="维度"):
        MilvusStore(uri, name, dim=8)

    # 不给维度时可以沿用集合里的实际维度（首启不知道维度也能打开）
    adopted = MilvusStore(uri, name, dim=0)
    assert adopted.dim == 4
