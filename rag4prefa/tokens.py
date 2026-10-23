"""token 估算：中文按字计、英文/符号按 3.5 字符计。

工程上不需要精确 tokenizer（会引入额外模型文件），但必须**统一**：
分块阈值、检索裁剪、prompt 预算都用同一个函数，避免"这里按字符那里按 token"。
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return int(cjk + other / 3.5) or 1


_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")


def split_by_tokens(text: str, max_tokens: int) -> list[str]:
    """按句子边界切分，尽量不把一句话劈开；单句超长时退化为硬切。"""
    text = text.strip()
    if not text:
        return []
    if estimate_tokens(text) <= max_tokens:
        return [text]

    sentences = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0
    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if buf and buf_tokens + sent_tokens > max_tokens:
            pieces.append("".join(buf).strip())
            buf, buf_tokens = [], 0
        if sent_tokens > max_tokens:
            # 极端长句（表格、无标点段落）：硬切
            for i in range(0, len(sent), max_tokens):
                pieces.append(sent[i : i + max_tokens])
            continue
        buf.append(sent)
        buf_tokens += sent_tokens
    if buf:
        pieces.append("".join(buf).strip())
    return [p for p in pieces if p]
