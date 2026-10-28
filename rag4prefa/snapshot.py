"""语料快照：给每一次评测一个可追溯的"当时库里到底有什么"。

**为什么非要有这个东西。** 仓库里曾经同时存在 `baseline-30.md`（30 题）和
`baseline-new.md`（25 题），两份报告的"数据集"一栏写的都是 `eval/dataset.jsonl`，
而该路径当时已经换过版本；更麻烦的是同一批 10 道口语题，dense 的 Recall@5 先记 0.500、
后记 0.700。于是"两种问法差多少"这个本来很硬的结论，变成了一个不知道该引用哪份的问题。
根因不在数字算错，在于**报告里没有记录"跑的时候库里有什么"**。

**为什么从数据库导出，而不是重新解析一遍语料。** 补语料的时候旧文件会被删掉
（2026-10-28 实测：`eval/corpus.json` 里 6 份文件，本地一份都不剩）。而"当时的语料指纹"
只能由当时那批片段决定，重新解析既拿不回旧文件，也可能因为解析器改进而算出不同的 id。
数据库里的 `chunks` 才是那批检索真正用过的片段，只要库还在，身份就还在。

**为什么把全量 id 也存下来。** 只存汇总哈希能回答"变没变"，回答不了"哪道题的标注失效了"。
存全量 id 之后，`dataset*.jsonl` 里的 gold 是否还在库里可以纯离线判定——
这在源文件已经删干净的情况下，是唯一还能守住的判据。

**为什么"按名字找源文件"也在这里。** `eval/corpus.json` 用裸文件名当键（数据库里
`documents.filename` 就是裸名），而语料是按分类目录存的（2026-10-28 起 `文献资料/`
下八大类）。`corpus_dir / filename` 这种拼法在扁平目录里成立，一分类就全部找不到——
而且找不到的表现是"用例整体跳过"，也就是校验以跳过的姿态悄悄消失。定位函数只有一处，
两份调用方（重标注脚本、标注校验用例）才不会各自写一套并各自走偏。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from . import repo

# 相对仓库根。评测脚本、测试、CLI 都按这个位置读写，避免三处各写一份路径。
DEFAULT_PATH = Path("eval") / "corpus-snapshot.json"


def ids_digest(ids: list[str] | set[str]) -> str:
    """一组片段 id 的指纹，取 sha256 前 16 位。

    算法必须与 `eval/corpus.json` 里 `ids_sha` 的算法完全一致（排序后换行连接），
    否则"清单与快照是不是同一批"这件事又要靠人肉判断。
    """
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def build_snapshot(conn: sqlite3.Connection, *, chunk_min_tokens: int = 0,
                   chunk_max_tokens: int = 0) -> dict:
    """从运行时数据库导出一份语料快照。

    只读 SQLite，不碰向量后端、不联网、不需要 Key：这是"记录当时有什么"的动作，
    不是"确认索引可用"的动作。索引是否一致由调用方自己先确认（`assert_query_ready`），
    两件事混在一起会让快照在索引异常时也导不出来——而恰恰是那时候最需要它。
    """
    grouped = repo.chunk_ids_by_doc(conn)
    documents: dict[str, dict] = {}
    every_id: list[str] = []

    for project in repo.list_projects(conn):
        for doc in repo.list_documents(conn, project.name):
            ids = grouped.get(doc.id, [])
            every_id.extend(ids)
            key = doc.filename
            # 同名文件（不同项目下的"计算书.pdf"）不能互相覆盖：快照里少了任何一份，
            # 指纹就和实际库对不上，而"对不上"正是这份文件唯一要回答的问题。
            if key in documents:
                key = f"{doc.filename} [{project.name}]"
            documents[key] = {
                "project": project.name,
                "doc_id": doc.id,
                "chunks": len(ids),
                "ids_sha": ids_digest(ids),
                "doc_year": doc.doc_year,
                "section_count": doc.section_count,
                "uploaded_at": doc.uploaded_at,
            }

    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        # 模型与维度以 meta 为准：它登记的是"这批向量由谁产生"，
        # 配置只是"打算用什么"，两者不一致时检索本来就会被拒。
        "embedding_model": repo.get_meta(conn, "embedding_model") or "",
        "embedding_dim": repo.get_meta(conn, "embedding_dim") or "",
        "chunk_min_tokens": chunk_min_tokens,
        "chunk_max_tokens": chunk_max_tokens,
        "document_count": len(documents),
        "chunk_count": len(every_id),
        "ids_sha": ids_digest(every_id),
        "documents": documents,
        "chunk_ids": sorted(every_id),
    }


def write_snapshot(snapshot: dict, path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return path


def load_snapshot(path: Path = DEFAULT_PATH) -> dict | None:
    """读快照。文件不在或坏了都返回 None——它是**佐证**，不是运行前提。

    让一份记录性文件变成硬依赖，会在"第一次跑评估"和"快照丢了"这两种情况下
    把整条评测拦住，而这两种情况恰恰都不该停。
    """
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def locate_corpus_file(corpus_dir: Path, filename: str) -> Path | None:
    """在语料目录里按**文件名**找源文件，找不到返回 None。

    语料是按分类目录存的，`corpus_dir / filename` 只在扁平目录下成立：一旦分了子目录，
    这份清单里每一条都找不到，而调用方通常把"找不到"当成"本机没有这套语料"→ 整体跳过。
    于是一个月的校验就这么静默消失了（2026-10-28 实测踩到）。

    同名多个时抛 `LookupError` 而不是挑一个：清单的键是裸文件名，两条同名记录本来
    就分不清谁是谁，随便挑一份等于拿别的文档去验指纹。
    """
    if not corpus_dir.is_dir():
        return None
    direct = corpus_dir / filename
    if direct.is_file():
        return direct
    # 不用 rglob(filename)：文件名里的 `[` `]` 会被当成通配符，
    # 真实语料里带方括号的文件名不少（标准号、修订版标注）。
    hits = sorted(p for p in corpus_dir.rglob("*") if p.is_file() and p.name == filename)
    if len(hits) > 1:
        raise LookupError(
            f"语料里有 {len(hits)} 个同名文件 `{filename}`："
            + "、".join(str(p.relative_to(corpus_dir)) for p in hits)
        )
    return hits[0] if hits else None


def dataset_digest(path: Path) -> tuple[str, int]:
    """数据集文件的指纹与题数。

    哈希按**原始字节**算：注释行也是内容的一部分，"改了注释但题目没动"同样应当
    反映成不同的数据集版本——否则改一个说法就能悄悄替换评测集。
    """
    raw = path.read_bytes()
    rows = sum(
        1
        for line in raw.decode("utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    )
    return hashlib.sha256(raw).hexdigest()[:16], rows


def compare(saved: dict | None, current: dict) -> list[str]:
    """比较"快照文件记录的"与"此刻库里的"，返回人话差异。

    只报会导致数字不可比的三件事：片段集合、文档数、embedding 模型。
    其余字段（时间戳、年份）变了不影响 Recall，报出来只会训练读者忽略警告。
    """
    if saved is None:
        return []
    notes: list[str] = []
    if saved.get("ids_sha") != current.get("ids_sha"):
        notes.append(
            f"语料片段集合变了（记录 `{saved.get('ids_sha')}` → 当前 "
            f"`{current.get('ids_sha')}`）"
        )
    if saved.get("document_count") != current.get("document_count"):
        notes.append(
            f"文档数变了（{saved.get('document_count')} → {current.get('document_count')}）"
        )
    if saved.get("embedding_model") and saved.get("embedding_model") != current.get(
        "embedding_model"
    ):
        notes.append(
            f"embedding 模型变了（`{saved.get('embedding_model')}` → "
            f"`{current.get('embedding_model')}`）"
        )
    return notes


def gold_missing(snapshot: dict, dataset: Path) -> list[str]:
    """数据集里有哪些 gold 已经不在快照的语料里。

    这是"标注过期"的离线判据。没有它，`dataset*.jsonl` 会在解析器或语料变动后
    继续跑出数字，只是 Recall 悄悄低一截，看不出是标注死了还是检索退化。
    """
    known = set(snapshot.get("chunk_ids") or [])
    if not known:
        return []
    missing: list[str] = []
    for line in dataset.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        row = json.loads(line)
        for chunk_id in row.get("gold_chunk_ids") or []:
            if chunk_id not in known:
                missing.append(f"{str(row.get('question'))[:24]} → {chunk_id[:12]}")
    return missing
