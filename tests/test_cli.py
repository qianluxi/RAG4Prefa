"""CLI 冒烟测试：命令必须真的跑得通，而不只是"能导入"。

背景：`cmd_ask` 曾直接解包 `ask.search()` 的返回值，而它是 async 函数，
拿到的是协程，README 里的查询命令一执行就 TypeError。
"""

from __future__ import annotations

import argparse

from rag4prefa import cli
from rag4prefa.domain import Filter, Hit


class _FakeServices:
    """只实现 CLI 用到的几个成员，避免真的去连数据库和模型。"""

    conn = object()
    store = object()
    settings = object()
    closed = False

    def require_embedder(self):
        return object()

    def require_llm(self):
        raise AssertionError("--answer 未开启时不该调用大模型")

    def assert_query_ready(self) -> None:
        """准入必须被调用：CLI 是独立入口，不能只靠 API 路由那边拦。"""
        type(self).checked = True

    def close(self) -> None:
        type(self).closed = True


def test_cmd_ask_awaits_async_search(monkeypatch, capsys) -> None:
    async def fake_search(*args, **kwargs):
        hit = Hit(chunk_id="c1", score=0.9, text="张拉时机与摩阻损失", project_name="甲项目")
        return [hit], Filter(projects=["甲项目"]), {"total_ms": 1.0}

    _FakeServices.closed = False
    _FakeServices.checked = False
    monkeypatch.setattr(cli, "build_services", lambda: _FakeServices())
    monkeypatch.setattr(cli.ask, "search", fake_search)

    args = argparse.Namespace(question="张拉控制要点", project="甲项目", top_k=5, answer=False)

    assert cli.cmd_ask(args) == 0
    out = capsys.readouterr().out
    assert "c1" in out
    assert "甲项目" in out
    assert _FakeServices.checked, "查询前必须过一遍准入，不能只靠 API 那边的保护"
    assert _FakeServices.closed, "命令结束必须关闭连接"


def test_cmd_reindex_reports_reindex_error(monkeypatch, capsys) -> None:
    def boom(*args, **kwargs):
        raise cli.ingest.ReindexError("embedding 维度由 16 变为 8，请去掉 --project")

    monkeypatch.setattr(cli, "build_services", lambda: _FakeServices())
    monkeypatch.setattr(cli.ingest, "reindex", boom)

    assert cli.cmd_reindex(argparse.Namespace(project="甲项目")) == 1
    err = capsys.readouterr().err
    assert "维度" in err and "reindex 未执行" in err


def test_cmd_verify_reports_drift(monkeypatch, capsys) -> None:
    """verify 发现差异时必须返回非零，并指向可执行的修复命令。"""
    monkeypatch.setattr(cli, "build_services", lambda: _FakeServices())
    monkeypatch.setattr(
        cli.consistency,
        "verify_ids",
        lambda conn, store: {
            "ok": False,
            "chunks": 10,
            "archived": 9,
            "indexed": 9,
            "counts": {"片段没有归档向量": 1},
            "samples": {"片段没有归档向量": ["c1"]},
        },
    )

    assert cli.cmd_verify(argparse.Namespace()) == 1
    captured = capsys.readouterr()
    assert "c1" in captured.out
    assert "reindex" in captured.err


def test_cmd_verify_returns_zero_when_clean(monkeypatch) -> None:
    monkeypatch.setattr(cli, "build_services", lambda: _FakeServices())
    monkeypatch.setattr(
        cli.consistency,
        "verify_ids",
        lambda conn, store: {"ok": True, "chunks": 10, "archived": 10, "indexed": 10},
    )
    assert cli.cmd_verify(argparse.Namespace()) == 0


# ---------------------------------------------------------------- 目录导入的"被跳过"


def test_collect_files_separates_supported_and_skipped(tmp_path) -> None:
    """目录导入必须把不能处理的文件**逐个报出来**。

    以前的实现只按扩展名 glob 一遍，`.doc` 这类文件既不在 `files` 里、也不会出现在
    任何输出里——用户看到"入库成功"就以为全都进去了。语料缺文件是个安静的错误：
    检索变差，但没人能归因到"少了一份文档"。
    """
    (tmp_path / "a.docx").write_bytes(b"x")
    (tmp_path / "b.pdf").write_bytes(b"x")
    (tmp_path / "c.doc").write_bytes(b"x")
    (tmp_path / "d.xlsx").write_bytes(b"x")
    (tmp_path / "~$a.docx").write_bytes(b"x")  # Office 锁文件不是"被跳过的语料"
    (tmp_path / "子目录").mkdir()

    files, skipped = cli._collect_files([str(tmp_path)])

    assert sorted(p.name for p in files) == ["a.docx", "b.pdf"]
    reasons = {p.name: why for p, why in skipped}
    assert set(reasons) == {"c.doc", "d.xlsx"}
    # 只说"不支持 .doc"会诱导用户去改后缀名——那是另一套二进制格式，改了就打不开
    assert ".docx" in reasons["c.doc"] and "另存为" in reasons["c.doc"]


def test_cmd_ingest_stops_when_everything_was_skipped(tmp_path, monkeypatch, capsys) -> None:
    (tmp_path / "只有老文档.doc").write_bytes(b"x")
    monkeypatch.setattr(
        cli, "build_services", lambda: (_ for _ in ()).throw(AssertionError("不该装配组件"))
    )
    assert cli.cmd_ingest(argparse.Namespace(paths=[str(tmp_path)])) == 2
    err = capsys.readouterr().err
    assert "只有老文档.doc" in err and "跳过 1 个文件" in err


# ---------------------------------------------------------------- 删除项目


def _docs(*names: str):
    from rag4prefa.domain import DocumentInfo

    return [
        DocumentInfo(
            id=f"d{i}",
            project_name="软件手册",
            filename=name,
            content_sha="0" * 40,
            section_count=1,
            chunk_count=301,
            status="indexed",
            uploaded_at="2026-10-26T00:00:00",
        )
        for i, name in enumerate(names)
    ]


def test_cmd_delete_is_a_dry_run_without_yes(monkeypatch, capsys) -> None:
    """删项目没有撤销，默认必须只预览——不能因为少打一个参数就把语料删了。"""
    calls: list[str] = []
    monkeypatch.setattr(cli, "build_services", lambda: _FakeServices())
    monkeypatch.setattr(cli.admin, "list_documents", lambda conn, name: _docs("手册.pdf"))
    monkeypatch.setattr(
        cli.admin,
        "delete_project",
        lambda conn, store, name: calls.append(name) or {"deleted": True},
    )

    assert cli.cmd_delete(argparse.Namespace(project="软件手册", yes=False)) == 0
    assert calls == [], "没加 --yes 就不许真删"
    out = capsys.readouterr().out
    assert "301" in out and "--yes" in out


def test_cmd_delete_runs_with_yes(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "build_services", lambda: _FakeServices())
    monkeypatch.setattr(cli.admin, "list_documents", lambda conn, name: _docs("手册.pdf"))
    monkeypatch.setattr(
        cli.admin,
        "delete_project",
        lambda conn, store, name: calls.append(name) or {"deleted": True},
    )
    assert cli.cmd_delete(argparse.Namespace(project="软件手册", yes=True)) == 0
    assert calls == ["软件手册"]


def test_cmd_delete_unknown_project_is_not_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "build_services", lambda: _FakeServices())
    monkeypatch.setattr(cli.admin, "list_documents", lambda conn, name: [])
    assert cli.cmd_delete(argparse.Namespace(project="不存在的项目", yes=True)) == 1
    assert "不存在" in capsys.readouterr().err
