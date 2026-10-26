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
