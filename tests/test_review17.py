"""第十七轮复审：把"后端未确认"变成**所有**普通入口共用的准入条件。

上一轮只保护了 API 的两个查询路由，于是同一份库里出现了两种结论：
`/api/search` 说"不能查"，`python -m rag4prefa ask` 照样回答，上传入口照样往
一个没被证明过的索引里写。原因不是漏了一个 `if`，而是准入只挂在调用路径上，
而"库里从没登记过活动后端"这条判据本身在 `ensure_backend_current` 里被当成放行。

这一轮把判据下沉到 `consistency`（谁能写、谁能查），并补上两件"看起来对"的事：

- 同名后端不等于同一个索引：`milvus` 只说明类型，地址与集合名也是身份；
- 名称相同不等于对象已就绪：抢不到维护锁的 Milvus 会 `_defer()`，名字照样是 milvus。

所以这里的断言一律落在"入口有没有被拒绝"和"数据有没有被改"上，不落在条数上。
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag4prefa import admin, ask, cli, consistency, ingest, repo
from rag4prefa.adapters.rerank import NoopReranker
from rag4prefa.adapters.vectors import BackendUnavailable, NumpyStore
from rag4prefa.api import create_app
from rag4prefa.config import Settings
from rag4prefa.consistency import IndexMismatch
from rag4prefa.domain import Filter, JobState
from rag4prefa.ingest import ingest_batch, ingest_file
from rag4prefa.services import Services
from tests.conftest import FakeEmbedder

# ── 夹具 ───────────────────────────────────────────────────


def _forget_backend(conn: sqlite3.Connection) -> None:
    """把"活动后端已登记"这件事抹掉，模拟 `settle_backend` 失败后留下的库。

    真实路径长这样：库里有内容，`settle_backend` 逐 id 校验没通过（遗留集合、
    重建失败、抢不到锁），于是**什么都不写**——登记表里那条记录就一直是空的。
    这不是"还没配置"，而是"有人试过确认，并且失败了"。
    """
    conn.execute("DELETE FROM meta WHERE key = ?", (repo.VECTOR_BACKEND_KEY,))
    conn.execute("DELETE FROM meta WHERE key = ?", (repo.VECTOR_BACKEND_ID_KEY,))
    conn.commit()
    assert repo.get_vector_backend(conn) is None


def _services(
    settings: Settings, conn: sqlite3.Connection, store, **kwargs
) -> Services:
    options = {
        "settings": settings,
        "conn": conn,
        "store": store,
        "embedder": FakeEmbedder(),
        "llm": None,
        "reranker": NoopReranker(),
        # 连接归测试夹具管：lifespan 结束时不能把它关掉
        "owns_conn": False,
    }
    options.update(kwargs)
    return Services(**options)


def _seed(
    conn: sqlite3.Connection, store, settings: Settings, docx: Path, project: str = "甲项目"
) -> int:
    outcome = ingest_file(
        conn, store, FakeEmbedder(), settings, path=docx, project_name=project, doc_year=2024
    )
    assert outcome.status == "indexed", outcome.error
    return store.count()


def _job_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) AS c FROM jobs").fetchone()["c"])


class _NamedStore:
    """只有名字和位置的假后端：用来单独验"身份"这一层，不牵扯真实存储。"""

    dim = FakeEmbedder.dim

    def __init__(self, name: str, identity: str | None) -> None:
        self.name = name
        self.rows: set[str] = set()
        if identity:
            self.identity = identity
        self.notes: list[str] = []

    def probe(self) -> None:  # 就绪，能回答
        return None

    def rebuild(self, items) -> None:  # type: ignore[override]
        self.rows = {i.chunk_id for i in items}

    def upsert(self, items) -> None:  # type: ignore[override]
        self.rows |= {i.chunk_id for i in items}

    def delete(self, chunk_ids) -> None:  # type: ignore[override]
        self.rows -= set(chunk_ids)

    def clear(self) -> None:  # type: ignore[override]
        self.rows = set()

    def count(self) -> int:
        return len(self.rows)

    def ids(self) -> set[str]:
        return set(self.rows)


class _DeferredStore(_NamedStore):
    """抢不到维护锁的 Milvus：名字是对的，对象没打开。"""

    def probe(self) -> None:
        raise BackendUnavailable("另一个维护动作正在进行（全量重建），本进程没有打开向量索引")

    def count(self) -> int:
        raise BackendUnavailable("Milvus 索引尚未打开")


class _CountFails(_NamedStore):
    """打开了、但读不出条数（SDK 超时、连接被占）。"""

    def count(self) -> int:
        raise BackendUnavailable("milvus count failed: deadline exceeded")


# ── P1：库里有内容却没有活动后端 → 所有普通入口都拒绝 ─────────


def test_query_is_refused_when_the_library_has_content_but_no_registered_backend(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """条数、模型、维度全都能对上，也不能拿这份索引回答。

    这就是"证明不了"最难查的形态：旧集合与归档条数相同、模型维度也相同，只有向量值
    和 `doc_year` 停在上一版。`ensure_queryable` 一条也挑不出来，唯一能拦住它的是
    "没有任何一步证明过这个后端"这条记录。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    _forget_backend(conn)

    with pytest.raises(IndexMismatch, match="从来没有登记过活动后端"):
        asyncio.run(
            ask.retrieve(conn, store, FakeEmbedder(), settings, question="悬挑", filt=Filter())
        )


def test_cli_ask_is_refused_by_the_same_gate_as_the_api(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m rag4prefa ask` 是**另一个入口**，不能只靠 API 路由的保护。

    上一轮修完只在 `/api/chat`、`/api/search` 上加了准入，CLI 查询仍然走
    `ask.search()`，于是同一个库在两种入口下给出两种结论：网页说"不能查"，
    命令行给出一份看起来正常的引用。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    _forget_backend(conn)
    monkeypatch.setattr(cli, "build_services", lambda: _services(settings, conn, store))

    args = argparse.Namespace(question="悬挑", project=None, top_k=5, answer=False)

    assert cli.cmd_ask(args) == 1
    captured = capsys.readouterr()
    assert "从来没有登记过活动后端" in captured.err
    assert "rebuild" in captured.err, "拒绝时必须给出可执行的下一步"
    assert captured.out == "", "被拒绝时不能输出一份看起来正常的检索结果"


def test_cli_ask_is_refused_when_the_process_could_not_confirm_the_backend(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """库里登记得好好的，但**本进程**没证明过它——同样不能回答。

    这是 `backend_confirmed` 存在的理由：登记值是"上一次有人证明过"，不是"我这次
    打开的就是它"。启动时抢不到维护锁、重建失败，都会落到这个状态。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    services = _services(settings, conn, store)
    services.backend_confirmed = False
    monkeypatch.setattr(cli, "build_services", lambda: services)

    args = argparse.Namespace(question="悬挑", project=None, top_k=5, answer=False)

    assert cli.cmd_ask(args) == 1
    assert "没能确认活动后端" in capsys.readouterr().err


def test_ordinary_ingest_is_refused_before_touching_anything(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """普通入库没有修复能力，在没有证据的库上继续写只会让不一致更深。

    准入排在项目/任务/归档/索引**任何一处变更之前**：任务必须还是 queued，
    项目不存在，片段与归档一条不多。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    _forget_backend(conn)
    before = repo.vector_stats(conn)

    job_id = "job-unconfirmed"
    repo.create_job(conn, job_id, "ingest", {"project": "乙项目"})

    with pytest.raises(IndexMismatch, match="从来没有登记过活动后端"):
        ingest_batch(
            conn,
            store,
            FakeEmbedder(),
            settings,
            job_id=job_id,
            files=[docx_file],
            project_name="乙项目",
            doc_year=2024,
        )

    job = repo.get_job(conn, job_id)
    assert job is not None and job.state is JobState.queued, "被拒绝时任务不该变成运行中/失败"
    assert repo.get_project(conn, "乙项目") is None, "被拒绝时不许留下空项目"
    assert repo.vector_stats(conn) == before, "归档一条都不许动"


def test_deleting_a_project_is_refused_when_the_backend_was_never_confirmed(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """删除会同时改事实与索引，准入与入库同源。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    total = _seed(conn, store, settings, docx_file)
    _forget_backend(conn)

    with pytest.raises(IndexMismatch, match="从来没有登记过活动后端"):
        admin.delete_project(conn, store, "甲项目")

    assert repo.get_project(conn, "甲项目") is not None
    assert repo.chunk_count(conn) > 0
    assert repo.archived_chunk_ids(conn)
    assert store.count() == total


def test_api_upload_is_refused_before_saving_files_or_creating_a_job(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """上传入口也要过这一道：文件落盘、建任务都排在准入之后。

    `upload_dir` 是持久归档区（不是临时暂存），所以"先收下再失败"会留下真正需要人
    清理的痕迹；任务记录还会让界面显示"正在入库"。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    services = _services(settings, conn, store)
    services.backend_confirmed = False

    with TestClient(create_app(services)) as client:
        resp = client.post(
            "/api/projects/新项目/documents",
            files={"files": (docx_file.name, docx_file.read_bytes(), "application/octet-stream")},
        )

    assert resp.status_code == 503, resp.text
    assert "没能确认活动后端" in resp.json()["detail"]
    assert _job_count(conn) == 0, "被拒绝时不该建任务"
    assert repo.get_project(conn, "新项目") is None
    saved = list(settings.upload_dir.rglob("*.docx")) if settings.upload_dir.exists() else []
    assert saved == [], f"被拒绝时不该落盘：{saved}"


def test_api_delete_is_refused_when_the_backend_was_never_confirmed(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """删除走用例层那道判据，被拒绝时项目、片段、归档都还在。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    _forget_backend(conn)
    services = _services(settings, conn, store)

    with TestClient(create_app(services)) as client:
        resp = client.delete("/api/projects/甲项目")

    assert resp.status_code == 503, resp.text
    assert "从来没有登记过活动后端" in resp.json()["detail"]
    assert repo.get_project(conn, "甲项目") is not None
    assert repo.chunk_count(conn) > 0


def test_full_reindex_repairs_an_unregistered_library_and_registers_it(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """修复入口必须留着，而且修完要真的能查——不能"命令成功、检索继续被拒"。

    全量 reindex 按事实重算整库向量、逐 id 校验，证据强度与 `rebuild_store` 相同，
    所以它成功之后自己登记活动后端（类型 + 位置）。旧行为是只写模型/维度/状态：
    命令报 0，下一次查询仍然被"从没登记过活动后端"挡住，用户只能猜还差哪一步。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    _forget_backend(conn)
    with pytest.raises(IndexMismatch):
        consistency.ensure_active_backend(conn, store)

    result = ingest.reindex(conn, store, FakeEmbedder(), settings)

    assert result["rebuilt"] is True
    assert repo.get_vector_backend(conn) == "numpy"
    assert repo.get_vector_backend_id(conn) == store.identity
    consistency.ensure_active_backend(conn, store)  # 不抛即通过
    hits, _, _ = asyncio.run(
        ask.search(conn, store, FakeEmbedder(), settings, question="悬挑", filt=Filter())
    )
    assert hits, "修完之后检索必须真的能用"


def test_partial_reindex_is_refused_while_the_library_is_unregistered(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """局部重建没有资格代替"整库确认"，拒绝时要说清该换成全量。

    放过它只会得到另一种"成功但查不了"：这个项目的向量被重写了一遍，库里仍然没有
    任何一条"活动后端已确认"的记录，检索继续被拒。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    _forget_backend(conn)

    with pytest.raises(ingest.ReindexError, match="去掉 --project"):
        ingest.reindex(conn, store, FakeEmbedder(), settings, project_name="甲项目")

    assert repo.get_vector_backend(conn) is None, "被拒绝时不该顺手登记"


# ── P2：同名不等于同一个索引，也不等于已经就绪 ───────────────


def test_a_deferred_backend_is_not_reported_as_confirmed(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """库里登记 milvus，本进程的 MilvusStore 因抢不到锁而 deferred。

    旧代码的快速路径只看"登记的 `name` 与手里的 `name` 相同"，于是直接
    `backend_confirmed = True`：`/api/health` 报正常，查询要等到 `search()` 里才炸，
    而那里通常只处理 `IndexMismatch`——用户看到的是 500，不是一个能读懂的 503。
    """
    archive = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, archive, settings, docx_file)
    deferred = _DeferredStore("milvus", "milvus|C:/data/v.db|docs")
    repo.set_vector_backend(conn, "milvus", identity=deferred.identity)

    services = _services(settings, conn, deferred)
    assert services.settle_backend() is False, "没打开就不许报已确认"
    assert services.backend_confirmed is False
    assert any("没能就绪" in note for note in services.notes), services.notes

    health = services.health()
    assert health["backend"]["confirmed"] is False
    assert health["ok"] is False
    assert any("没能确认活动后端" in issue for issue in health["index"]["issues"])

    with TestClient(create_app(services)) as client:
        resp = client.post("/api/search", json={"question": "悬挑"})

    assert resp.status_code == 503, resp.text
    assert "没能确认活动后端" in resp.json()["detail"]


def test_the_fast_path_still_skips_a_rebuild_when_the_backend_is_ready(
    conn: sqlite3.Connection,
    settings: Settings,
    docx_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """加了"就绪检查"不等于每次启动都重建一遍：启动时间不该随库变大。"""
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)

    def explode(*_args, **_kwargs):
        raise AssertionError("已登记且能就绪的后端不该被重建一遍")

    monkeypatch.setattr("rag4prefa.ingest.rebuild_store", explode)
    services = _services(settings, conn, store)

    assert services.settle_backend() is True
    assert services.backend_confirmed is True
    assert repo.get_vector_backend_id(conn) == store.identity, "历史库没记位置时要就地补记"


def test_the_same_backend_name_at_a_different_location_is_refused(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """两个进程共用一份 SQLite、各配一个 Milvus 集合：名字都是 milvus，索引不是同一份。

    `vector_backend` 只记类型，所以"名字相同"曾经是全部的证据。身份（类型 + 地址 +
    集合名）一起登记之后，这种部署会在第一次比对时就被指出来，而不是让两边各自
    写一半、谁也查不到谁。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    mine = _NamedStore("milvus", "milvus|C:/data/a.db|docs")
    other = _NamedStore("milvus", "milvus|C:/data/b.db|docs")
    repo.set_vector_backend(conn, "milvus", identity=mine.identity)

    consistency.ensure_backend_current(conn, mine)  # 位置一致就放行

    with pytest.raises(IndexMismatch, match="不是同一份索引"):
        consistency.ensure_backend_current(conn, other)
    with pytest.raises(IndexMismatch, match="不是同一份索引"):
        consistency.ensure_active_backend(conn, other)

    services = _services(settings, conn, other)
    assert services.settle_backend() is False, "位置不对就不许确认"
    assert any("不是同一份索引" in note for note in services.notes)


def test_rebuild_records_the_location_so_a_moved_collection_is_detectable(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """换后端/换集合是许可的动作，但它必须把新位置登记下来。

    不记位置的话，换完之后下一次启动仍然只比名字——"同一个 milvus"这个结论会一直
    成立，而实际查的已经是另一个集合。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    repo.set_vector_backend(conn, "milvus", identity="milvus|C:/data/a.db|docs")
    target = _NamedStore("milvus", "milvus|C:/data/b.db|docs")

    ingest.rebuild_store(conn, target)

    assert repo.get_vector_backend(conn) == "milvus"
    assert repo.get_vector_backend_id(conn) == target.identity
    consistency.ensure_backend_current(conn, target)  # 不抛即通过


def test_a_backend_that_is_not_ready_yields_503_instead_of_500(
    conn: sqlite3.Connection, settings: Settings, docx_file: Path
) -> None:
    """后端读不出条数时，接口要给 503，而不是一个看不懂的 500。

    这类失败以前会从 `store.count()` 直接冒出来：运维看到"服务器内部错误"，
    会去查代码，而真正该做的是去看那个占着维护锁的进程或重建那次索引。
    """
    store = NumpyStore(conn, FakeEmbedder.dim)
    _seed(conn, store, settings, docx_file)
    broken = _CountFails("numpy", "numpy")

    with TestClient(create_app(_services(settings, conn, broken))) as client:
        resp = client.post("/api/search", json={"question": "悬挑"})

    assert resp.status_code == 503, resp.text
    assert "deadline exceeded" in resp.json()["detail"]
