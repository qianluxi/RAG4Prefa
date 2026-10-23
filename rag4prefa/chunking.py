"""分块：按标题层级切，而不是按固定字数硬切。

规则：
1. 每个标题节点自成一块的"种子"，正文超出预算时按句子边界再切；
2. 同一父节点下的相邻小块合并（不跨父级，避免原实现"跨章节拼标题"的问题）；
3. 空标题（无正文）不单独成块——它的标题已经出现在子节点的 chapter_path 里；
4. **chunk_id 由内容决定，不由位置决定**——见 `_chunk_id` 的说明。
"""

from __future__ import annotations

import hashlib
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
        for piece in split_by_tokens(text, max_tokens):
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
