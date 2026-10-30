"""评测缓存的身份：**这份缓存能不能被这次实验复用**，只能由身份字段回答。

起因是 2026-10-30 专家复审第三节：`judge_sufficiency.py` 的候选缓存只按问题文本做
key，元数据虽然写了 `mode`/`k` 却**加载时不校验**；判官结论的 key 加了模型名，却没加
服务商/接口地址/思考设置/提示词指纹/候选身份。这与本轮刚发现的"同名模型在两个服务商
上行为相反"直接冲突——同一份缓存文件接着用，会静默拿到上一家的结论，而报告上看不出
任何异常。

两种处理方式都能写，区别很大：

- **警告后继续**：日志里多一行，跑的人九成不会看见；结论已经混进报告。
- **拒绝复用**：抛 `CacheMismatch`，把"哪个字段不一样"打出来，让人显式选择
  `--rebuild`（重检索）或 `--rejudge`（重判）。

这里选后者。缓存的全部价值是"省一次调用"，而它唯一的风险是"用错一次却不自知"；
两者不对称，所以宁可停下来。

一句话概括每个缓存该带什么身份：

| 缓存 | 身份 |
|---|---|
| 检索候选 | 问题、Filter、召回方式、深度、**语料 id 集合摘要**、检索源码指纹 |
| 判官结论 | 判官版本、模型、**服务商地址**、思考设置、top-k、**提示词指纹**、**候选 id 摘要** |

候选 id 摘要这一条容易漏：提示词没变、模型没变，但检索结果变了（语料更新、改了融合），
同一句话的候选就不是同一批——拿旧结论当新结论，正是"A/B 两行看起来只差一个变量、
其实差了两个"的老问题。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path

from rag4prefa import repo

ROOT = Path(__file__).resolve().parent.parent

# 影响"检索出来什么"的源码范围。与 `eval.py` 的 `_FINGERPRINT_ROOTS` 口径一致，
# 再加上实验脚本自己：`judge_sufficiency.py` 里改了 `top_n`/截断长度，一样会改结论。
SRC_ROOTS: tuple[str, ...] = ("rag4prefa",)
SRC_EXTRA: tuple[str, ...] = (
    "scripts/eval.py",
    "scripts/judge_sufficiency.py",
    "scripts/cache_identity.py",
)
SOURCE_SUFFIXES = frozenset({".py", ".json"})

# 缓存文件格式版本。加字段不必换版本，改结构才换：旧文件没有 `schema` 也能读出来，
# 但会被当成"身份不完整"拒绝（见 `check`），这正是我们要的。
SCHEMA = 2


class CacheMismatch(RuntimeError):
    """缓存身份与本次实验不符。消息里带字段差异，调用方可以原样打给用户。"""


def _digest(pieces: Iterable[str]) -> str:
    h = hashlib.sha256()
    for piece in pieces:
        h.update(piece.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _normalized(path: Path) -> bytes:
    """读文件并把 CRLF 归一成 LF。

    不归一的话，同一份代码在 Windows 工作区与 Linux CI 上算出两个指纹——"同指纹即同
    代码"这个承诺就此失效，而这种差异谁都不会想到去查（`eval.py` 的 `src_sha` 同理）。
    """
    return path.read_bytes().replace(b"\r\n", b"\n")


def fingerprint_files(root: Path, rels: Iterable[str]) -> tuple[str, int]:
    """把一批相对路径的**内容**揉成一个短哈希，返回 (sha, 计入的文件数)。"""
    h = hashlib.sha256()
    count = 0
    for rel in sorted(rels):
        path = root / rel
        if not path.is_file():
            continue
        h.update(rel.replace("\\", "/").encode("utf-8"))
        h.update(b"\0")
        h.update(_normalized(path))
        h.update(b"\0")
        count += 1
    return h.hexdigest()[:16], count


def source_files(root: Path = ROOT) -> list[str]:
    """影响检索与判官的源码清单（相对路径）。

    走文件系统而不是 `git ls-files`：未 `git add` 的新文件、不在仓库里的部署目录都
    该被算进去。`__pycache__` 里的 `.pyc` 靠后缀自然排除。
    """
    rels: list[str] = []
    for name in SRC_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in SOURCE_SUFFIXES:
                rels.append(path.relative_to(root).as_posix())
    rels.extend(name for name in SRC_EXTRA if (root / name).is_file())
    return sorted(rels)


def source_fingerprint(root: Path = ROOT) -> dict:
    """检索与判官逻辑的内容指纹。工作区脏的时候，这是唯一可信的身份。"""
    sha, count = fingerprint_files(root, source_files(root))
    return {"src_sha": sha, "src_files": count}


def corpus_identity(conn: sqlite3.Connection) -> dict:
    """语料身份：文档数 + 片段数 + 全量 chunk id 的摘要。

    只哈希 id 不哈希正文，理由是 `chunk_id` 本身内容寻址（文档、章节路径、正文哈希、
    同段序号），id 集合变了就是语料变了；正文改了而 id 不变不可能发生。
    """
    ids = sorted(repo.all_chunk_ids(conn))
    return {
        "docs": repo.document_count(conn),
        "chunks": len(ids),
        "ids_sha": _digest(ids),
    }


def prompt_fingerprint(*texts: str) -> str:
    """判官提示词的指纹。改一个字都要让旧结论失效——它判的是同一批片段，理由却换了。

    换行同样归一：从网上复制进来的提示词常带 CRLF。
    """
    return _digest(text.replace("\r\n", "\n") for text in texts)


def filter_key(filt) -> str:  # noqa: ANN001 - 传 domain.Filter，这里只关心它的可序列化形态
    """Filter 的规范写法：字段顺序固定，空条件与「没传」一致。"""
    data = filt.model_dump()
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def candidate_key(hits: Iterable) -> str:  # noqa: ANN001 - 元素是 domain.Hit
    """候选片段的身份：顺序敏感的 id 串。

    顺序要进哈希：同一批片段换个名次，重排分与"最相关那条第几"都不一样，结论可能翻。
    """
    return _digest(hit.chunk_id for hit in hits)


def cache_key(*parts: object) -> str:
    """通用 key：参与比较的每一段都要给，缺一段就是少一道校验。"""
    return _digest(str(part) for part in parts)


def check(path: Path, meta: Mapping | None, expect: Mapping[str, object], *, kind: str,
          bypass: str) -> None:
    """加载前校验。身份不符就抛，**不返回空字典**。

    返回空字典等于"当作没有缓存、悄悄重跑"——对实验是安全的，对成本是不安全的，
    而且它会让人以为"缓存命中率低"而不是"身份变了"。抛错把选择权交回给人。
    """
    if not meta:
        raise CacheMismatch(
            f"{path} 没有身份信息（旧格式或不是本脚本写的），拒绝复用。\n"
            f"确认要重来就加 {bypass}。"
        )
    diffs = [
        f"  · {name}: 缓存 {meta.get(name, '<无>')!r} ≠ 本次 {value!r}"
        for name, value in expect.items()
        if meta.get(name) != value
    ]
    if not diffs:
        return
    raise CacheMismatch(
        f"{kind}缓存 `{path}` 与本次实验不符，拒绝复用：\n" + "\n".join(diffs)
        + f"\n确认要重来就加 {bypass}；或者换一个缓存路径，把两轮结果分开存。"
    )


def load(path: Path, *, expect: Mapping[str, object], kind: str, bypass: str) -> dict:
    """读缓存并校验身份，返回条目表。"""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CacheMismatch(f"{path} 不是一个对象，拒绝复用。确认要重来就加 {bypass}。")
    check(path, payload.get("meta"), expect, kind=kind, bypass=bypass)
    entries = payload.get("hits")
    return entries if isinstance(entries, dict) else {}


def save(path: Path, entries: Mapping[str, object], meta: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "meta": dict(meta), "hits": dict(entries)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
