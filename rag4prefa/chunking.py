"""分块：按标题层级切，而不是按固定字数硬切。

规则：
1. 每个标题节点自成一块的"种子"，正文超出预算时按句子边界再切；
2. 同一父节点下的相邻小块合并（不跨父级，避免原实现"跨章节拼标题"的问题）；
3. 空标题（无正文）不单独成块——它的标题已经出现在子节点的 chapter_path 里；
4. **chunk_id 由内容决定，不由位置决定**——见 `_chunk_id` 的说明。
"""

from __future__ import annotations

import hashlib
from bisect import bisect_right
from collections import Counter

from .domain import Chunk, Section
from .tokens import estimate_tokens, split_by_tokens


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(doc_id: str, chapter_path: list[str], content_sha: str, seq: int) -> str:
    """内容寻址：同一文档、同一章节路径、同一段正文 → 同一个 id。

    为什么不用 `doc_id|ordinal`：位置寻址会让"同一 id 可能是不同正文"。文档在新版本中间
    插一段，后面所有片段都换号；反过来，不改号的那些位置可能已经换成别的内容。于是
    "先写向量、再切事实"就会让索引里的新向量配上事实表里的旧正文——检索能跑，但找到的
    是另一段话。这类错位不报错，只会让引用看起来莫名对不上。

    改成内容寻址后，"同 id 必同正文"是结构性质：先写索引不会污染旧版本，改一段也只影响
    那一段。`seq` 只用于同一章节里出现完全相同的段落（合并单元格解析后可能出现），
    同位置同内容的重复仍然稳定。
    """
    key = f"{doc_id}|{' > '.join(chapter_path)}|{content_sha}"
    if seq:
        key += f"|{seq}"
    return hashlib.sha1(key.encode()).hexdigest()[:32]


def contextual_text(chunk: Chunk) -> str:
    """送入 embedding 的文本：带上完整层级路径，让"3.2 悬挑构造"具备可检索的上下文。"""
    path = " > ".join(chunk.chapter_path)
    return f"{path}\n{chunk.text}" if path else chunk.text


def _flat_pages(section: Section, text: str) -> tuple[list[int], list[int | None]] | None:
    """在**去掉换行的扁平文本**上，返回每一行的起始偏移与该行页码。

    为什么不用原文偏移（2026-10-27 实测踩到）：`split_by_tokens` 按
    `(?<=[。！？；\\n])` 切句，单独的换行会被切出来、再被 `if s.strip()` 丢掉，
    于是拼回的片段里**没有换行符**——拿片段去原文里 `find()` 一律落空，
    页码全部退回节起始页，等于这个功能没做。去掉换行后，片段就是原文的连续子串：
    两者的唯一差别就是这些被丢弃的分隔符，字符本身与顺序都不变。

    行数对不上（`line_pages` 与 `text.split("\\n")` 不同长）时返回 None：
    外部构造的 Section、或将来某个解析器改了换行处理都可能撞上，那时宁可退回
    章节起始页——错位的页码比没有页码更容易误导人。
    """
    pages = section.line_pages
    if not pages:
        return None
    lines = text.split("\n")
    if len(pages) != len(lines):
        return None
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line)  # 扁平文本里没有换行符
    return starts, list(pages)


def _page_at(offset: int, starts: list[int], pages: list[int | None]) -> int | None:
    """扁平文本里某个字符偏移落在哪一行上 → 那一行的页码。"""
    if not starts:
        return None
    return pages[bisect_right(starts, offset) - 1]


def build_chunks(
    sections: list[Section],
    doc_id: str,
    *,
    max_tokens: int = 480,
    min_tokens: int = 80,
) -> list[Chunk]:
    seeds: list[Chunk] = []
    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        # 片段报**自己**的起始页，而不是整节标题所在的页（2026-10-27 复审 P1-2）：
        # "4.2 悬挑构造"从第 4 页起、正文排到第 8 页，第 7 页上的那一段如果标成
        # "第 4 页"，用户翻过去找不到原文，引用就不可核。页码不参与 chunk_id，
        # 所以按行定位只改展示，不动寻址。
        flat = text.replace("\n", "")
        located = _flat_pages(section, text)
        starts, line_pages = located if located else ([], [])
        cursor = 0
        for piece in split_by_tokens(text, max_tokens):
            page = section.page
            if starts:
                # 片段是扁平文本里的连续子串，顺序向后找即可（整节只有几 KB）。
                # 游标必须按**扁平后**的长度推进：片段里可能带着换行（切句时保留在
                # 句尾），按带换行的长度推进会多走几个字符，后面几块就再也找不到，
                # 页码全部退回节起始页——2026-10-27 实测踩到，表现为页码交错跳回。
                needle = piece.replace("\n", "")
                offset = flat.find(needle, cursor)
                if offset >= 0:
                    cursor = offset + len(needle)
                    found = _page_at(offset, starts, line_pages)
                    if found is not None:
                        page = found
            seeds.append(
                Chunk(
                    id="",
                    doc_id=doc_id,
                    ordinal=0,
                    chapter_path=section.full_path,
                    title=section.title,
                    text=piece,
                    token_count=estimate_tokens(piece),
                    content_sha=_sha(piece),
                    page=page,
                )
            )

    merged: list[Chunk] = []
    for chunk in seeds:
        prev = merged[-1] if merged else None
        same_parent = prev is not None and prev.chapter_path[:-1] == chunk.chapter_path[:-1]
        small = prev is not None and (
            prev.token_count < min_tokens or chunk.token_count < min_tokens
        )
        fits = prev is not None and prev.token_count + chunk.token_count <= max_tokens
        if same_parent and small and fits:
            title = prev.title if prev.title == chunk.title else f"{prev.title} / {chunk.title}"
            text = f"{prev.text}\n{chunk.text}"
            merged[-1] = prev.model_copy(
                update={
                    "title": title,
                    "text": text,
                    "token_count": estimate_tokens(text),
                    "content_sha": _sha(text),
                }
            )
        else:
            merged.append(chunk)

    numbered: list[Chunk] = []
    seen: Counter[tuple[str, str]] = Counter()
    for ordinal, chunk in enumerate(merged):
        key = (" > ".join(chunk.chapter_path), chunk.content_sha)
        seq = seen[key]
        seen[key] += 1
        numbered.append(
            chunk.model_copy(
                update={
                    "ordinal": ordinal,
                    "id": _chunk_id(doc_id, chunk.chapter_path, chunk.content_sha, seq),
                }
            )
        )
    return numbered
