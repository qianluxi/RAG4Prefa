"""从语料抽取「规范用语」候选——词表的输入侧必须来自条文，不来自测试题。

判据（都不看评测集）：

1. **只从"规范句"里抽**：含 应/宜/不应/不得/应按/应符合 的句子。这些是条文在说话，
   用词是标准用语；而导言的叙述句、新闻稿的口语词会被排除。
2. **跨文档复现**：至少出现在 3 份不同文档里。只在一份文件里出现一次的词多半是
   专有名词（项目名、地名），换不成"公众说法"。
3. **长度 2~6 字**，且不含数字/标点。
4. 用「在规范句里的频率 / 在全部文本里的频率」做提纯比：一个词越是"只在条文里
   出现"，越可能是规范用语；越是在普通叙述里也常见（"装修""房子"），越可能是
   日常词，不需要映射。

输出按提纯比排序，人工从上往下挑。
"""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "app.db"

NORMATIVE = re.compile(
    r"应(?:按|符合|采|设|满|符|为|大|小|能|有|无|留|予|当|采用|设置)?|宜|不得|不应|严禁"
)
CJK = re.compile(r"[\u4e00-\u9fff]+")
SENT_SPLIT = re.compile(r"[。；;\n]")

# 这些是"人人都说"的词，即使高频出现也不值得做映射
STOP = {
    "装修", "装配", "建筑", "工程", "设计", "施工", "住宅", "项目", "标准", "规定",
    "要求", "技术", "进行", "采用", "应当", "不应", "不得", "可以", "需要", "符合",
    "浙江省", "建设", "质量", "安全", "管理", "工作", "实施", "相关", "有关", "以及",
    "或者", "并且", "按照", "根据", "通过", "方式", "情况", "内容", "问题", "方面",
    "部位", "位置", "空间", "尺寸", "高度", "宽度", "长度", "面积", "数量", "材料",
    "产品", "部件", "构件", "系统", "设备", "设施", "功能", "性能", "指标", "方法",
}


MODAL = set("应宜不严禁当须需")
EDGE_BAD = set("的和与或时在为于按等上下中的了及等其该此各每")

def _bad(g: str) -> bool:
    """含情态动词的 n-gram 是句式碎片（"应符合设计要"），不是术语。"""
    if any(ch in MODAL for ch in g):
        return True
    return g[0] in EDGE_BAD or g[-1] in EDGE_BAD


def _maximal(grams: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """去掉同频次的子串：`装配式` 与 `装配式装修` 计数相同 → 只留长的。"""
    grams = sorted(set(grams), key=lambda kv: (-len(kv[1]), -kv[0]))
    kept: list[tuple[int, str]] = []
    for cnt, g in grams:
        if any(g in k and cnt <= c for c, k in kept):
            continue
        kept.append((cnt, g))
    return kept


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    docs = {r["id"]: r["filename"] for r in conn.execute("select id, filename from documents")}

    norm_count: dict[str, int] = defaultdict(int)
    all_count: dict[str, int] = defaultdict(int)
    norm_docs: dict[str, set[str]] = defaultdict(set)
    all_docs: dict[str, set[str]] = defaultdict(set)

    for row in conn.execute("select doc_id, text, chapter_path from chunks"):
        doc = docs[row["doc_id"]]
        text = re.sub(r"\s+", "", row["text"])
        all_docs_here: set[str] = set()
        for s in SENT_SPLIT.split(text):
            if len(s) < 4:
                continue
            for run in CJK.findall(s):
                for n in range(2, 7):
                    for i in range(len(run) - n + 1):
                        g = run[i : i + n]
                        if g in STOP:
                            continue
                        all_count[g] += 1
                        all_docs_here.add(g)
                        if NORMATIVE.search(s):
                            norm_count[g] += 1
                            norm_docs[g].add(doc)
        for g in all_docs_here:
            all_docs[g].add(doc)

    pool = [(nc, g) for g, nc in norm_count.items() if nc >= 4 and len(norm_docs[g]) >= 3]
    pool = [(nc, g) for nc, g in pool if not _bad(g)]
    rows = []
    for nc, g in _maximal(pool):
        ac = all_count[g]
        purity = nc / ac
        rows.append((purity, nc, len(norm_docs[g]), ac, g))
    rows.sort(reverse=True)

    print(f"候选 {len(rows)} 个（按提纯比排序，前 120）\n")
    for purity, nc, nd, ac, g in rows[:120]:
        print(f"  {purity:5.2f}  规范句{nc:3d}  文档{nd:2d}  总{ac:4d}  {g}")


if __name__ == "__main__":
    main()
