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


def test_ids_and_count_are_exact(store: MilvusStore) -> None:
    """verify 命令靠 ids() 做逐 id 比对，count() 必须是精确值而不是统计估算。"""
    store.upsert(
        [
            VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024),
            VectorItem("b", [0, 1, 0, 0], "甲项目", "", 2024),
            VectorItem("c", [0, 0, 1, 0], "甲项目", "", 2024),
        ]
    )
    assert store.ids() == {"a", "b", "c"}
    assert len(store.ids()) == store.count() == 3

    store.delete(["b"])
    assert store.ids() == {"a", "c"}
    assert store.count() == 2, "删除后条数必须立刻准确，否则一致性检查会误判"


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


def test_rebuild_same_dim_prunes_orphans(tmp_path) -> None:
    """同维度整体重建也要把不在新集合里的旧 id 清掉。"""
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


def test_rebuild_same_dim_also_goes_through_staging(tmp_path, monkeypatch) -> None:
    """同维度重建也不再原地覆盖正式集合。

    原地覆盖到一半失败，正式集合就是新旧混杂的一堆向量，而检索看不出任何异常。
    这里让"写暂存"这一步炸掉，断言正式索引**一条都没被动过**。
    """
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        store = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    store.upsert([VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024)])

    def boom(*args, **kwargs):
        raise RuntimeError("模拟写暂存失败")

    monkeypatch.setattr(store, "_upsert_into", boom)
    with pytest.raises(RuntimeError, match="模拟写暂存失败"):
        store.rebuild(
            [
                VectorItem("a", [0, 1, 0, 0], "甲项目", "", 2024),
                VectorItem("b", [1, 0, 0, 0], "甲项目", "", 2024),
            ]
        )

    assert store.count() == 1, "正式集合必须保持原样，不能混进 b"
    assert set(store._client.list_collections()) == {name}, "失败的暂存集合必须被清掉"
    # 旧向量还能被查出来，且分数仍是与 a 的原始向量算出来的
    hits = store.search([1, 0, 0, 0], limit=5)
    assert [h[0] for h in hits] == ["a"]


def test_reopen_discards_staging_when_final_is_intact(tmp_path) -> None:
    """进程在切换前被杀死：正式集合完好，暂存是半成品，重启后必须丢掉暂存。"""
    uri, name = str(tmp_path / "vectors.db"), "chunks_vec"
    try:
        first = MilvusStore(uri, name, dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")
    first.upsert([VectorItem("a", [1, 0, 0, 0], "甲项目", "", 2024)])
    # 手工制造"暂存写了一半、切换还没开始"的状态
    first._create_collection(f"{name}__staging", 4)

    reopened = MilvusStore(uri, name, dim=4)

    assert set(reopened._client.list_collections()) == {name}
    assert reopened.notes, "丢弃暂存要留下说明"
    assert [h[0] for h in reopened.search([1, 0, 0, 0], limit=3)] == ["a"]


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


def test_non_ascii_path_is_rejected_before_touching_disk(tmp_path) -> None:
    """Windows + 中文路径：faiss 建索引必失败，必须在开集合前就拒绝并给出出路。

    实测结论：同一份数据放 `C:/tmp/...` 正常，放 `.../课题/...` 下只要触发
    release+load 就报 "faiss::FileIOWriter ... No such file or directory"，
    之后集合直接处于不可用状态。与其让人在重建那一刻收到看不懂的报错，不如早拒绝。
    """
    import os

    if os.name != "nt":
        pytest.skip("该限制只在 Windows 上出现")

    with pytest.raises(BackendUnavailable, match="非 ASCII"):
        MilvusStore(str(tmp_path / "课题" / "vectors.db"), "chunks_vec", dim=4)


def test_relative_uri_is_checked_against_absolute_path(tmp_path, monkeypatch) -> None:
    """相对路径的字面量干净，不代表落盘位置干净。

    默认配置就是 `data/vectors.db`。工程放在中文目录下时，这个字符串里没有半个
    非 ASCII 字符，旧检查直接放行，直到 MilvusClient(uri) 才报
    "Open local milvus failed"——那条报错既没提 faiss 也没提中文，
    看的人根本不知道要改什么。所以必须按 abspath 判断。
    """
    import os

    if os.name != "nt":
        pytest.skip("该限制只在 Windows 上出现")

    workdir = tmp_path / "课题"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    with pytest.raises(BackendUnavailable, match="非 ASCII"):
        MilvusStore("data/vectors.db", "chunks_vec", dim=4)


def test_relative_uri_in_ascii_dir_is_allowed(tmp_path, monkeypatch) -> None:
    """反方向也要钉住：ASCII 工作目录下的相对路径不能因为这次收紧被误杀。"""
    import os

    if os.name != "nt":
        pytest.skip("该限制只在 Windows 上出现")

    monkeypatch.chdir(tmp_path)
    try:
        MilvusStore("vectors.db", "chunks_vec", dim=4)
    except BackendUnavailable as exc:
        pytest.skip(f"Milvus 不可用：{exc}")


def test_non_ascii_collection_name_is_rejected(tmp_path) -> None:
    """uri 干净但集合名带中文，索引文件路径照样是中文，同样要拒绝。"""
    import os

    if os.name != "nt":
        pytest.skip("该限制只在 Windows 上出现")

    with pytest.raises(BackendUnavailable, match="非 ASCII"):
        MilvusStore(str(tmp_path / "vectors.db"), "课题向量", dim=4)


def test_open_store_falls_back_to_numpy_for_non_ascii_path(tmp_path) -> None:
    """默认 backend=auto 时，这类环境问题应自动回退到 numpy，而不是启动即崩。"""
    import os

    if os.name != "nt":
        pytest.skip("该限制只在 Windows 上出现")

    from rag4prefa.config import Settings
    from rag4prefa.db import connect, init_db
    from rag4prefa.services import open_store

    data = tmp_path / "课题"
    settings = Settings(
        data_dir=data, vector_backend="auto", milvus_uri=str(data / "vectors.db")
    )
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)
    try:
        store, notes, fell_back = open_store(settings, conn)
        assert store.name == "numpy"
        assert fell_back is True
        assert any("非 ASCII" in n for n in notes), "回退原因必须写清楚"
    finally:
        conn.close()


def test_explicit_milvus_backend_does_not_silently_fall_back(tmp_path) -> None:
    import os

    if os.name != "nt":
        pytest.skip("该限制只在 Windows 上出现")

    from rag4prefa.config import Settings
    from rag4prefa.db import connect, init_db
    from rag4prefa.services import open_store

    data = tmp_path / "课题"
    settings = Settings(
        data_dir=data, vector_backend="milvus", milvus_uri=str(data / "vectors.db")
    )
    settings.ensure_dirs()
    init_db(settings.db_path)
    conn = connect(settings.db_path)
    try:
        with pytest.raises(BackendUnavailable, match="不会自动回退"):
            open_store(settings, conn)
    finally:
        conn.close()
