"""语料体检：入库之前先看清"这批文件能变成多少片段、有哪些进不来"。

为什么在入库**之前**单独跑一遍：

- 入库要花钱（embedding 按 token 计费）。而"扫描件没文字层""字体缺 ToUnicode 映射
  读出来全是乱码"这两类问题，等到入库时才暴露，钱已经花过一轮，更麻烦的是它们会以
  "入库成功但内容缺失"的形式留在库里，静默拖坏 BM25 与向量空间。
- 课题要论证的是"语料对题"。片段总数里有多少真的讲到装配式内装 / 装配化装修，
  这个比例决定了评测题能不能在库里找到答案，必须在建索引之前就量出来。

只读：不调 embedding、不写库、不动文件。

用法：
    python scripts/corpus_audit.py ../工程文档
    python scripts/corpus_audit.py ../工程文档 --md ../工程文档/_体检报告.md
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag4prefa.adapters.parsers import (  # noqa: E402
    SUPPORTED_SUFFIXES,
    parse_document_report,
)
from rag4prefa.chunking import build_chunks  # noqa: E402
from rag4prefa.ingest import chunk_tokens  # noqa: E402

# 课题主题词。命中任意一个就算"对题片段"，分开计数是为了看哪一类讲得少。
# 这份清单跟着课题主题走，不是通用词表：改方向就改这里。
THEME_TERMS: dict[str, tuple[str, ...]] = {
    "装配式内装": ("装配式内装", "装配式装修", "装配化装修", "装配式内装修"),
    "架空与管线": ("架空", "管线分离", "管线与结构分离"),
    "干式工法": ("干式工法", "干法施工", "干式施工"),
    "模块化部品": ("模块化", "整体厨房", "整体卫生间", "集成吊顶", "集成墙面"),
    "政策与补贴": ("补贴", "以旧换新", "厨卫换新", "试点"),
    "标准与评价": ("评价标准", "技术标准", "验收标准", "规范"),
}

# 不是语料的目录：隐藏目录装的是工具状态（.workbuddy），_下载日志 是抓取记录。
SKIP_DIRS = {".workbuddy", ".git", "_下载日志"}


@dataclass
class FileAudit:
    path: Path
    rel: str
    status: str = "ok"  # ok / error / unsupported / empty
    pages: int = 0
    sections: int = 0
    chunks: int = 0
    tokens: int = 0
    note: str = ""
    theme_hits: Counter = field(default_factory=Counter)
    theme_any: int = 0  # 命中任一主题的片段数（并集，不是各主题相加）


def _iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts):
            continue
        out.append(path)
    return out


def audit_one(path: Path, root: Path) -> FileAudit:
    rel = path.relative_to(root).as_posix()
    item = FileAudit(path=path, rel=rel)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        item.status = "unsupported"
        supported = " / ".join(s.lstrip(".") for s in SUPPORTED_SUFFIXES)
        item.note = f"{path.suffix or '无后缀'} 还不支持（当前只解析 {supported}）"
        return item

    try:
        sections, report = parse_document_report(path, ocr=None)
    except Exception as exc:  # 解析器的错误类型不止一种，体检要的是"能不能进"
        item.status = "error"
        item.note = f"{type(exc).__name__}: {exc}"
        return item

    item.pages = report.pages
    item.sections = len(sections)
    # doc_id 只影响 chunk_id，体检不需要与入库一致；用相对路径保证同一份文件稳定
    chunks = build_chunks(sections, rel, max_tokens=480, min_tokens=80)
    item.chunks = len(chunks)
    item.tokens = chunk_tokens(chunks)
    item.note = report.note() or ""

    for chunk in chunks:
        body = chunk.text
        matched = False
        for name, terms in THEME_TERMS.items():
            if any(term in body for term in terms):
                item.theme_hits[name] += 1
                matched = True
        if matched:
            item.theme_any += 1
    if not chunks:
        item.status = "empty"
        item.note = item.note or "没有可用正文"
    return item


def _fmt_table(items: list[FileAudit]) -> list[str]:
    lines = [
        "| # | 文件 | 页 | 章节 | 片段 | token | 状态 |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for idx, item in enumerate(items, 1):
        status = item.status
        if status == "ok" and item.note:
            status = "ok（有缺页）"
        lines.append(
            f"| {idx} | {item.rel} | {item.pages or '-'} | {item.sections or '-'} "
            f"| {item.chunks or '-'} | {item.tokens or '-'} | {status} |"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="入库前的语料体检（只读，不调 embedding）")
    parser.add_argument("root", type=Path, help="语料目录")
    parser.add_argument("--md", type=Path, default=None, help="把报告写成 markdown")
    parser.add_argument(
        "--quiet", action="store_true", help="只打印汇总，不逐份列（目录很大时用）"
    )
    args = parser.parse_args(argv)

    root: Path = args.root
    if not root.is_dir():
        print(f"目录不存在：{root}", file=sys.stderr)
        return 2

    files = _iter_files(root)
    if not files:
        print(f"{root} 下没有文件", file=sys.stderr)
        return 2

    items: list[FileAudit] = []
    for path in files:
        items.append(audit_one(path, root))

    by_status = Counter(item.status for item in items)
    total_chunks = sum(item.chunks for item in items)
    total_tokens = sum(item.tokens for item in items)
    theme = Counter()
    for item in items:
        for name, count in item.theme_hits.items():
            theme[name] += count

    out: list[str] = []
    out.append(f"# 语料体检：{root}")
    out.append("")
    # 报告头带时间与命令：这份文件会随语料变动重生成，没有这两项的话，两份"同名不同语料"
    # 的报告在事后无法区分——第三十三轮踩过一模一样的坑（评测报告缺语料身份）。
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out.append(f"生成时间：{stamp}　命令：`python scripts/corpus_audit.py {root}`")
    out.append("")
    out.append(
        f"文件 {len(items)} 份 → 可入库 {by_status['ok']} 份、"
        f"解析失败 {by_status['error']} 份、无正文 {by_status['empty']} 份、"
        f"格式不支持 {by_status['unsupported']} 份"
    )
    out.append(f"合计 {total_chunks} 个片段、约 {total_tokens} token（按入库同一套切块参数）")
    out.append("")
    if total_chunks:
        out.append("## 主题覆盖（命中关键词的片段数 / 总片段）")
        out.append("")
        out.append("| 主题 | 片段 | 占比 |")
        out.append("|---|---:|---:|")
        for name in THEME_TERMS:
            hits = theme[name]
            out.append(f"| {name} | {hits} | {hits / total_chunks:.1%} |")
        any_hit = sum(item.theme_any for item in items)
        out.append(
            f"| 任一主题（并集） | {any_hit} | {any_hit / total_chunks:.1%} |"
        )
        out.append("")

    bad = [item for item in items if item.status != "ok" or item.note]
    if bad:
        out.append("## 需要注意的文件")
        out.append("")
        for item in bad:
            out.append(f"- `{item.rel}`：{item.status} —— {item.note}")
        out.append("")

    if not args.quiet:
        out.append("## 逐份明细")
        out.append("")
        out.extend(_fmt_table(items))

    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text + "\n", encoding="utf-8")
        print(f"\n已写入 {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
