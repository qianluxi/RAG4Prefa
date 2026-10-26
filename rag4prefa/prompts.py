"""提示词集中管理：与代码分离，便于调参和版本对比（评估时可以直接换模板做 A/B）。"""

from __future__ import annotations

from .domain import Hit

SYSTEM_ANSWER = """你是一名建筑工程技术资料助手，服务对象是设计、施工与科研人员。
回答规则：
1. 只依据给定的资料片段作答，不要引入片段之外的事实。
2. 每个结论后用 [编号] 标注来源，编号必须来自给定片段。
3. 如果片段不足以回答，直接说明"资料中未涉及"，并指出还需要什么资料，不要猜测。
4. 涉及规范条文、数值、构造做法时，原文照录关键表述，不要改写数值。
5. 用简体中文，结论先行，再用依据支撑。
6. 用 Markdown 组织层次：小标题用 **加粗** 单独成行；并列条目用「1. 2. 3.」或「- 」；
   多方案对比用竖线表格。不要输出 HTML 标签，不要用代码块包裹正文。"""

SYSTEM_SUMMARY = """你是建筑工程技术文档分析助手。请用 2 句话概括给定文档片段的主题与技术要点，
不超过 120 字，不要编造片段中没有的内容，不要加前缀。"""


def build_answer_prompt(question: str, hits: list[Hit], filters_desc: str) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        path = " > ".join(hit.chapter_path) or hit.title
        blocks.append(
            f"[{n}] 项目：{hit.project_name}｜文件：{hit.filename}｜章节：{path}\n{hit.text}"
        )
    context = "\n\n".join(blocks)
    scope = f"当前检索范围：{filters_desc}\n\n" if filters_desc else ""
    return f"""{scope}资料片段：
{context}

用户问题：{question}

请基于以上片段作答。"""


def build_summary_prompt(project_name: str, hits: list[Hit]) -> str:
    snippets = "\n".join(f"- {h.text[:400]}" for h in hits[:5])
    return f"项目：{project_name}\n\n文档片段：\n{snippets}"
