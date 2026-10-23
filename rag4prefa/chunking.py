"""分块：按标题层级切，而不是按固定字数硬切。

规则：
1. 每个标题节点自成一块的"种子"，正文超出预算时按句子边界再切；
2. 同一父节点下的相邻小块合并（不跨父级，避免原实现"跨章节拼标题"的问题）；
3. 空标题（无正文）不单独成块——它的标题已经出现在子节点的 chapter_path 里。
"""

from __future__ import annotations

import hashlib

from .domain import Chunk, Section
from .tokens import estimate_tokens, split_by_tokens


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(doc_id: str, ordinal: int) -> str:
    return hashlib.sha1(f"{doc_id}|{ordinal}".encode()).hexdigest()[:32]


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

    return [
        c.model_copy(update={"ordinal": i, "id": _chunk_id(doc_id, i)})
        for i, c in enumerate(merged)
    ]
