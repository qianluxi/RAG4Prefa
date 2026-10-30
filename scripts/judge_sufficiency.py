"""判官实测：规则闸门 vs 让模型判"这批片段够不够"。

回答的问题只有一个：**在 `near_topic`（主题相近但答不了）这一类上，判官能不能拦住，
同时不误伤库内题**。它默认不进生产判定（见 `sufficiency.py` 第二层注释），
数字过关才谈接入。

为什么要先冻结候选：规则闸门与判官必须看**同一批片段**。两次检索各跑一遍，重排服务的
末位近分候选会翻（实测同池同进程重排 4 次名次一致，跨运行浮动 1~2 位），
"判官比闸门多拦了一题"就可能只是它拿到了不同的候选。所以候选先落盘，
之后所有重跑（换提示词、换模型、调 top_n）都从同一份候选出发。

**缓存的复用条件写在 `scripts/cache_identity.py` 里**（2026-10-30 专家复审第三节）：
候选缓存带语料 id 摘要与检索源码指纹，判官结论缓存带服务商、思考设置、提示词指纹与
候选 id 摘要；加载时校验，对不上就抛错而不是静默复用。同名模型在两个服务商上行为相反
这件事已经实测过，只按模型名做 key 等于给 A/B 埋一颗看不见的雷。

用法：
    python scripts/judge_sufficiency.py --limit 5          # 先冒烟
    python scripts/judge_sufficiency.py                    # 全量，候选走缓存
    python scripts/judge_sufficiency.py --rebuild          # 重新检索
    python scripts/judge_sufficiency.py --judge v3         # 换判官（结论缓存按版本分）
    python scripts/judge_sufficiency.py --save eval/refusal-judge-20261030.md
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa import ask, prompts, repo, sufficiency  # noqa: E402
from rag4prefa.domain import Filter, Hit  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402
from scripts import cache_identity  # noqa: E402

# 库内题集：判官在这里唯一的任务是**别乱拒**。四份合起来 122 题，
# 与 `eval/refusal-20261029.md` 的误拒口径一致，两次报告的"误拒 0"才是同一件事。
# 判官定温。不定温做过一轮，代价是**同一份输入两次跑出不同结论**：155 题里 11 题翻转，
# 库内误拒 6 vs 8。这个量级与"v2 改成 v3"的差别同阶，于是版本之间的边际差异无法解释。
# 定温 0 是这一层能被当作测量而不是抽样的前提。
JUDGE_TEMPERATURE = 0.0

POSITIVE_SETS = (
    "eval/dataset.jsonl",
    "eval/dataset.paraphrase.jsonl",
    "eval/dataset.independent.jsonl",
    "eval/dataset.independent-b.jsonl",
)


def _eval_module():
    """借用 `eval.py` 的召回链路，避免两处口径漂移（同 `term_ablation.py`）。"""
    spec = importlib.util.spec_from_file_location("ev_judge", ROOT / "scripts" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows(path: Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {lineno} 行不是合法 JSON：{exc}") from exc
    return rows


def build_worklist(refuse_path: Path, positive_sets: list[Path]) -> list[dict]:
    """统一成一张表：`kind` 只有两种，`expect` 决定该不该拒。"""
    work: list[dict] = []
    for row in load_rows(refuse_path):
        question = str(row.get("question") or "").strip()
        if question:
            work.append(
                {
                    "kind": "refuse",
                    "question": question,
                    "category": str(row.get("category") or "未分类"),
                    "filter": row.get("filter") or {},
                }
            )
    seen = {w["question"] for w in work}
    for path in positive_sets:
        if not path.is_file():
            raise SystemExit(f"库内题集不存在：{path}")
        for row in load_rows(path):
            question = str(row.get("question") or "").strip()
            # 同一句话出现在两份题集里只算一次：误拒率的分母不该被重复计。
            if not question or question in seen:
                continue
            seen.add(question)
            work.append(
                {
                    "kind": "positive",
                    "question": question,
                    "category": path.stem.replace("dataset.", "") or "base",
                    "filter": row.get("filter") or {},
                }
            )
    return work


def _thinking_label(value: bool | None) -> str:
    """思考开关的三态写成可比较的字符串。`None`（不发字段）与 `False`（显式关）不能合并：
    它们对模型的行为在魔搭与硅基流动上恰好相反，合并了就看不出换了哪个设置。"""
    return {None: "不发送", True: "开", False: "关"}[value]


def _prompt_signature(judge: str) -> str:
    """判官提示词指纹：拿空片段构造一次，得到的是与具体题目无关的模板文本。

    改一个字都要让旧结论失效——旧结论是"用另一套判据看同一批片段"得出的，留着它等于
    把 A/B 的两行混成一行。不把 `src_sha` 放进判官缓存的身份里，是因为那太粗：改一句
    注释就会让几百次调用白花，而这一次改动对判官结论毫无影响。

    这里**不接收 k**：`top_n` 只影响截取几条片段，不改变模板文本（空片段时它是空操作）。
    k 由 `verdict_meta` 与 `verdict_key` 各自带着，写成参数只会造成"它被算进指纹"的错觉。
    """
    return cache_identity.prompt_fingerprint(
        prompts.JUDGE_PROMPTS[judge],
        prompts.build_judge_prompt("问题", []),
        prompts.build_judge_retry_prompt("问题", []),
    )


async def _retrieve(ev, services, mode: str, question: str, filt: Filter, k: int,
                    attempts: int = 3) -> list[Hit]:
    """一次检索，失败重试到第 3 次仍失败则抛——不吞掉失败（同 `term_ablation._rerank_full`）。"""
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return await ev.retrieve_for(services, mode, question, filt, k, filt.keyword)
        except Exception as exc:  # noqa: BLE001 - 远端服务偶发 5xx，重试后仍失败才报
            last = exc
            await asyncio.sleep(2)
    raise RuntimeError(f"{question} 检索失败：{last}")


def _complete(services, system: str, user: str, *, delays: tuple[int, ...] = (5, 15, 30)) -> str:
    """一次判官调用，遇限流/5xx 退避重试。

    `LLM.complete` 自己带三次指数退避，但实测撞到 429 时它三次都失败——远端按模型限流，
    靠"多试几次"解决不了，只能拉开间隔。这里再套一层，且**接的是 Exception 而不是
    LLMError**：429 是 `openai.RateLimitError`，适配器没有包成 LLMError，只捕后者会让
    整轮任务在第一个 429 上崩掉（2026-10-30 实际发生：118/156 处全丢）。
    """
    last: Exception | None = None
    for delay in (*delays, None):
        try:
            return services.llm.complete(system, user, max_tokens=200,
                                         temperature=JUDGE_TEMPERATURE)
        except Exception as exc:  # noqa: BLE001 - 限流、超时、5xx 同一处理
            last = exc
            if delay is None:
                break
            print(f"    判官调用失败（{type(exc).__name__}），{delay}s 后重试", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"判官调用失败：{last}")


def hits_identity(*, mode: str, k: int, corpus: dict, source: dict) -> dict:
    """候选缓存的身份。

    四个来源缺一不可：`mode`/`k` 决定怎么取，语料 id 摘要决定"库里还是那批东西"，
    源码指纹决定"取法有没有变"。少了语料那条，重跑入库之后旧候选会被当成新的用——
    问题文本没变、片段已经不是同一批。
    """
    return {"kind": "hits", "mode": mode, "k": k, **corpus, **source}


def verdict_identity(*, judge: str, model: str, provider: str, thinking: str, k: int,
                     prompt_sha: str, corpus: dict, temperature: float) -> dict:
    """判官结论缓存的身份（不含源码指纹，理由见 `_prompt_signature`）。

    `temperature` 必须在里面：同一套提示词、同一个模型，只把温度从 0.2 改成 0，判定
    就会变（实测 155 题里 11 题翻转）。它和"换服务商"是同一类事实——结论绑在设置上，
    不绑在提示词上。
    """
    return {
        "kind": "verdicts", "judge": judge, "model": model, "provider": provider,
        "thinking": thinking, "k": k, "prompt_sha": prompt_sha,
        "temperature": temperature,
        "corpus_ids_sha": corpus.get("ids_sha", ""),
    }


def verdict_key(*, judge: str, model: str, provider: str, thinking: str, k: int,
                prompt_sha: str, temperature: float, hits: list[Hit], question: str) -> str:
    """单题结论的 key：身份 + **候选 id 摘要** + 问句。

    候选摘要这一项是 2026-10-30 复审点出来的漏项：提示词没变、模型没变，但检索结果
    变了（语料更新、融合改动），同一句话的片段就不是同一批，旧结论不能当新结论用。
    """
    return cache_identity.cache_key(
        judge, model, provider, thinking, k, prompt_sha, temperature,
        cache_identity.candidate_key(hits), question,
    )


async def measure(work: list[dict], *, mode: str, k: int, cache_path: Path,
                  rebuild: bool, progress: bool = True, concurrency: int = 4,
                  judge: str = "v1", verdict_path: Path | None = None,
                  rejudge: bool = False) -> dict:
    """先串行把这个题集的候选全部落实，再并发问判官。

    分两段不是为了快一点，是因为两段的约束完全不同：
      - 检索与规则闸门都碰 SQLite（`repo.documents_mentioning`、融合查询），
        同一条连接不能被并发使用（会抛 `InterfaceError` 且只丢一路，报告上只表现为
        "少了一题"）——所以第一段必须串行；
      - 判官只是一次 HTTP 调用、每题独立，串行跑 156 题要四十分钟，并发四路够快
        且不至于触发远端限流。
    """
    services = build_services()
    services.assert_query_ready()
    if not services.llm:
        raise SystemExit("未配置大模型服务（R4P_LLM_API_KEY），判官无法实测")
    settings = services.settings
    system = prompts.JUDGE_PROMPTS[judge]
    model = settings.llm_model
    # 服务商与思考设置也进身份：同名模型在两个服务商上默认行为相反（一个思考一个不思考），
    # 上一轮实测只换服务商就让误拒数从 41 变成 32。只按模型名做 key，换个 base_url 就会
    # 静默读到上一家的结论。
    provider = urlparse(settings.llm_base_url).netloc or settings.llm_base_url
    thinking = _thinking_label(settings.llm_enable_thinking)
    corpus = cache_identity.corpus_identity(services.conn)
    source = cache_identity.source_fingerprint()
    hits_meta = hits_identity(mode=mode, k=k, corpus=corpus, source=source)
    # 每个判官版本一个结论缓存文件：身份里含 `judge`，混在一个文件里第二次跑就会被
    # 自己的校验挡住。分文件的副作用正好是想要的——换版本做 A/B 时两边的结论互不可见。
    verdict_path = verdict_path or Path(f"eval/reports/judge-verdicts-{judge}.json")
    ev = _eval_module()
    # `--rebuild` / `--rejudge` 是唯一的绕行口：身份不符时拒绝复用，由人决定重来哪一层。
    cache = {} if rebuild else cache_identity.load(
        cache_path, expect=hits_meta, kind="候选", bypass="--rebuild"
    )
    known_projects = repo.project_names(services.conn)

    prepared: list[dict] = []
    fetched = 0
    for i, item in enumerate(work, start=1):
        question = item["question"]
        request_filter = Filter(**item["filter"]) if item["filter"] else Filter()
        effective = ask.build_filter(question, request_filter, known_projects)
        # 问题文本进 key 不够：同一个问句配上不同的 Filter（项目、年份、关键词）会检索出
        # 不同的东西，只按问句复用等于把上一次的范围带进这一次。
        fkey = cache_identity.cache_key(question, cache_identity.filter_key(effective))
        entry = cache.get(fkey)
        if entry and "hits" in entry:
            hits = [Hit(**d) for d in entry["hits"]]
        else:
            hits = await _retrieve(ev, services, mode, question, effective, k)
            cache[fkey] = {
                "question": question,
                "filter": effective.model_dump(),
                "hits": [h.model_dump() for h in hits],
            }
            fetched += 1
            # 每检索一条就落一次盘：一轮十分钟，中途断掉不该全丢。
            cache_identity.save(cache_path, cache, {**hits_meta, "partial": True})
        top = hits[:k]
        gate = sufficiency.judge(
            services.conn, question, top,
            min_rerank=services.settings.refuse_min_rerank,
            min_dense=services.settings.refuse_min_dense,
        )
        prepared.append({"item": item, "top": top, "gate": gate, "index": i})
        if progress:
            print(f"[取候选 {i}/{len(work)}] {question[:30]}", file=sys.stderr)
    cache_identity.save(cache_path, cache, {**hits_meta, "partial": False})

    # 判官结论单独按题缓存：**候选缓存只解决一半问题**。2026-10-30 就是候选全在缓存里、
    # 判官结论没落盘，跑到 118/156 被 429 打断后，那 118 次调用白花。
    #
    # key 里除了版本与模型，还要放**候选 id 摘要**：提示词没变、模型没变，但检索结果变了
    # （语料更新、融合改动），同一句话的片段就不是同一批，旧结论不能当新结论用。
    prompt_sha = _prompt_signature(judge)
    verdict_meta = verdict_identity(
        judge=judge, model=model, provider=provider, thinking=thinking, k=k,
        prompt_sha=prompt_sha, corpus=corpus, temperature=JUDGE_TEMPERATURE,
    )
    verdicts = {} if rejudge else cache_identity.load(
        verdict_path, expect=verdict_meta, kind="判官结论", bypass="--rejudge"
    )
    sem = asyncio.Semaphore(max(1, concurrency))
    done = 0

    def remember(key: str, reply: sufficiency.JudgeReply) -> None:
        # 只缓存**解析成功**的结论：格式故障下次要重判，缓存下来等于把它固化。
        if reply.can_answer is None:
            return
        verdicts[key] = {
            "can_answer": reply.can_answer,
            "reason": reply.reason,
            "attempts": reply.attempts,
        }
        cache_identity.save(verdict_path, verdicts, {**verdict_meta, "partial": True})

    async def one(entry: dict) -> dict:
        nonlocal done
        item, top, gate = entry["item"], entry["top"], entry["gate"]
        key = verdict_key(
            judge=judge, model=model, provider=provider, thinking=thinking, k=k,
            prompt_sha=prompt_sha, temperature=JUDGE_TEMPERATURE, hits=top,
            question=item["question"],
        )
        error = ""
        if key in verdicts:
            d = verdicts[key]
            reply = sufficiency.JudgeReply(d["can_answer"], d["reason"], "", d.get("attempts", 1))
        else:
            async with sem:
                try:
                    reply = await asyncio.to_thread(
                        sufficiency.judge_by_llm,
                        item["question"], top,
                        lambda s, u: _complete(services, s, u),
                        top_n=k, system=system,
                    )
                except RuntimeError as exc:
                    # 单题失败不该带走整轮：记为未表态，下一轮从缓存缺口处补。
                    reply = sufficiency.JudgeReply(None, "", "")
                    error = str(exc)[:120]
            if not error:
                remember(key, reply)
        done += 1
        if progress:
            # 进度按完成顺序打，题号会乱跳——并发下这是正常的，不按序号重排。
            mark = "拒" if gate.refuse else ("拒" if reply.can_answer is False else "答")
            print(
                f"[判 {done}/{len(prepared)}] {mark} 规则={int(gate.refuse)} "
                f"判官={reply.can_answer} {item['question'][:26]}",
                file=sys.stderr,
            )
        return {
            **item,
            "rule_refuse": gate.refuse,
            "rule_reasons": gate.reasons,
            "judge_can_answer": reply.can_answer,
            "judge_refuse": reply.can_answer is False,
            "judge_reason": reply.reason,
            "judge_attempts": reply.attempts,
            "judge_raw": reply.raw if reply.can_answer is None else "",
            "judge_error": error,
            "top": {
                "filename": top[0].filename if top else "",
                "score": top[0].score if top else None,
            },
            "n_hits": len(top),
            "_index": entry["index"],
        }

    tasks = [asyncio.create_task(one(entry)) for entry in prepared]
    results = []
    for fut in asyncio.as_completed(tasks):
        results.append(await fut)
    results.sort(key=lambda r: r.pop("_index"))
    cache_identity.save(verdict_path, verdicts, {**verdict_meta, "partial": False})
    return {
        "results": results, "fetched": fetched, "mode": mode, "k": k,
        "cache": str(cache_path), "judge": judge, "model": model,
        "errors": sum(1 for r in results if r["judge_error"]),
        # 报告里要能自证：这批数字是在哪份语料、哪份源码、哪个服务商、什么思考设置下跑的。
        "identity": {
            "provider": provider, "thinking": thinking, "prompt_sha": prompt_sha,
            "temperature": JUDGE_TEMPERATURE,
            "corpus": corpus, "source": source,
        },
    }


def summarize(out: dict) -> dict:
    """按类别分栏。库内题也按题集分——"判官在口语题上误拒多少"与"在标准题集上误拒多少"
    是两个不同的结论，合成一个总数会把口语题那一栏藏起来。"""
    rows = out["results"]
    by_cat: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "rule": 0, "judge": 0, "undecided": 0, "either": 0}
    )
    pos: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "rule": 0, "judge": 0, "undecided": 0, "either": 0}
    )
    for r in rows:
        bucket = by_cat if r["kind"] == "refuse" else pos
        slot = bucket[r["category"]]
        slot["n"] += 1
        rule = bool(r["rule_refuse"])
        judge = bool(r["judge_refuse"])
        undecided = r["judge_can_answer"] is None
        slot["rule"] += int(rule)
        slot["judge"] += int(judge)
        slot["undecided"] += int(undecided)
        slot["either"] += int(rule or judge)
    return {"refuse": {k: dict(v) for k, v in by_cat.items()},
            "positive": {k: dict(v) for k, v in pos.items()}}


def render(out: dict, summary: dict) -> str:
    rows = out["results"]
    refuse = [r for r in rows if r["kind"] == "refuse"]
    pos = [r for r in rows if r["kind"] == "positive"]
    total_n = sum(s["n"] for s in summary["refuse"].values())
    total_rule = sum(s["rule"] for s in summary["refuse"].values())
    total_judge = sum(s["judge"] for s in summary["refuse"].values())

    lines = [
        "# 判官实测：规则闸门 vs 模型判断（2026-10-30）",
        "",
        "## 一、这一轮要回答什么",
        "",
        "规则闸门（`sufficiency.judge`）只能命名确定性理由，`near_topic` 这一类它一条也拦不住。",
        "判官的假设是：给模型看同一批片段，问它「这些够不够回答问题」，它判得出「主题相同但缺事实」。",
        "要验证的就两件事——**该拒的拦住多少**，**库内题误伤多少**。",
        "",
        f"口径：判官 `{out.get('judge', 'v1')}`、模型 `{out.get('model', '未记录')}`、"
        f"召回方式 `{out['mode']}`、`k={out['k']}`（= 生产 `settings.top_k`）"
        "（与 `eval.py` 误拒统计同口径），"
        "规则闸门与判官看的是同一批片段（候选已冻结，见 §五）。",
        "",
        "## 二、库外题：按类别",
        "",
        "| 类别 | 题数 | 规则闸门 | 判官 | 判官未表态 | 任一（规则∨判官） |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in sorted(summary["refuse"].items()):
        lines.append(
            f"| `{name}` | {s['n']} | {s['rule']}/{s['n']} | {s['judge']}/{s['n']} | "
            f"{s['undecided']} | {s['either']}/{s['n']} |"
        )
    lines.append(f"| **合计** | **{total_n}** | **{total_rule}** | **{total_judge}** | "
                 f"**{sum(s['undecided'] for s in summary['refuse'].values())}** | "
                 f"**{sum(s['either'] for s in summary['refuse'].values())}** |")

    pos_sets = summary["positive"]
    tot = {k: sum(v[k] for v in pos_sets.values()) for k in ("n", "rule", "judge",
                                                             "undecided", "either")}
    lines += [
        "",
        "## 三、库内题：误拒",
        "",
        "误拒比漏拒更贵——用户问的东西明明在库里，系统却说没有。按题集分开列，",
        "因为口语题与标准题集在判官眼里的难度不是一回事。",
        "",
        "| 题集 | 题数 | 规则闸门误拒 | 判官误拒 | 判官未表态 | 任一误拒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, v in sorted(pos_sets.items()):
        lines.append(
            f"| `{name}` | {v['n']} | {v['rule']} | {v['judge']} | {v['undecided']} | "
            f"{v['either']} |"
        )
    lines.append(
        f"| **合计** | **{tot['n']}** | **{tot['rule']}** | **{tot['judge']}** | "
        f"**{tot['undecided']}** | **{tot['either']}** |"
    )
    bad = [r for r in pos if r["judge_refuse"] or r["judge_can_answer"] is None]
    if bad:
        lines += ["", "逐题（判官判「不可」或未表态的库内题）：", ""]
        for r in bad:
            verdict = "不可" if r["judge_refuse"] else "未表态"
            reason = (r["judge_reason"] or r["judge_raw"] or "").replace("\n", " ")[:60]
            lines.append(f"- **{r['question']}** — {verdict}：{reason}")

    missed = [r for r in refuse if not r["rule_refuse"] and not r["judge_refuse"]]
    lines += [
        "",
        "## 四、漏拒逐题",
        "",
        f"规则与判官都没拦住的 {len(missed)} 题：",
        "",
    ]
    for r in missed:
        verdict = {True: "可", False: "不可", None: "未表态"}[r["judge_can_answer"]]
        reason = (r["judge_reason"] or r["judge_raw"] or "").replace("\n", " ")[:60]
        lines.append(f"- `{r['category']}` **{r['question']}** — 判官：{verdict}，{reason}")

    ident = out.get("identity") or {}
    corpus = ident.get("corpus") or {}
    source = ident.get("source") or {}
    lines += [
        "",
        "## 五、可复现性",
        "",
        f"- 候选缓存：`{out['cache']}`（本轮新检索 {out['fetched']} 题，其余复用）。",
        "- 判官输出原样留在 `--dump` 的 JSON 里（含 `judge_raw`），判错的题可以逐条回看原话。",
        f"- 本轮判官调用失败 {out.get('errors', 0)} 题（记为未表态，不计入拒答）；"
        "结论缓存按「版本+题」存，续跑只补缺口。",
        "- **判官结论依赖当时的 LLM 版本**，与重排服务同理：换模型必须重跑，不能沿用这份数字。",
    ]
    if ident:
        lines += [
            "",
            "缓存身份（`scripts/cache_identity.py`，加载时校验，不符即拒绝复用）：",
            "",
            f"- 语料：{corpus.get('docs', '?')} 份文档 / {corpus.get('chunks', '?')} 个片段，"
            f"id 摘要 `{corpus.get('ids_sha', '?')}`",
            f"- 源码：`{source.get('src_sha', '?')}`（{source.get('src_files', '?')} 个文件）；"
            f"召回 `{out['mode']}`、深度 `{out['k']}`",
            f"- 判官：`{out.get('judge', 'v1')}`、模型 `{out.get('model', '?')}`、"
            f"服务商 `{ident.get('provider', '?')}`、思考 `{ident.get('thinking', '?')}`、"
            f"温度 `{ident.get('temperature', '?')}`、提示词 `{ident.get('prompt_sha', '?')}`",
            "- 判官结论的 key 另含**候选 id 摘要**：同一问句的片段换了一批，旧结论自动失效。",
        ]
    lines += ["", ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refuse", default="eval/dataset.refuse.jsonl")
    ap.add_argument("--positive", nargs="*", default=list(POSITIVE_SETS))
    ap.add_argument("--mode", default="rerank", help="召回方式（默认生产用的 rerank）")
    ap.add_argument("--k", type=int, default=8,
                    help="返回与送判官的片段数（默认 8 = 生产的 settings.top_k）")
    ap.add_argument("--cache", default="eval/reports/judge-hits.json")
    ap.add_argument("--rebuild", action="store_true", help="忽略候选缓存重新检索")
    ap.add_argument("--verdict-cache", default="",
                    help="判官结论缓存，默认 eval/reports/judge-verdicts-<版本>.json"
                         "（key 含模型+服务商+思考设置+提示词+候选摘要，续跑只补缺口）")
    ap.add_argument("--rejudge", action="store_true", help="忽略判官结论缓存重新判")
    ap.add_argument("--dump", default="eval/reports/judge-20261030.json")
    ap.add_argument("--save", default="", help="把报告写到这个路径")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--concurrency", type=int, default=4, help="判官并发路数")
    ap.add_argument("--from-dump", default="", help="从已落盘的结果重排版，不重跑")
    ap.add_argument("--judge", choices=sorted(prompts.JUDGE_PROMPTS), default="v1",
                    help="判官版本（v1 单一口径 / v2 按取值型与做法型分开）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if args.from_dump:
        # 重排版式不该再花一次 156 次模型调用：从落盘的结果重算表格即可。
        saved = json.loads(Path(args.from_dump).read_text(encoding="utf-8"))
        out = {k: saved[k] for k in ("results", "mode", "k", "cache") if k in saved}
        out["judge"] = saved.get("judge", "v1")
        out["model"] = saved.get("model", "")
        out["identity"] = saved.get("identity", {})
        out["fetched"] = 0
    else:
        work = build_worklist(Path(args.refuse), [Path(p) for p in args.positive])
        if args.limit:
            work = work[: args.limit]
        try:
            out = asyncio.run(
                measure(
                    work, mode=args.mode, k=args.k, cache_path=Path(args.cache),
                    rebuild=args.rebuild, progress=not args.quiet,
                    concurrency=args.concurrency, judge=args.judge,
                    verdict_path=Path(args.verdict_cache) if args.verdict_cache else None,
                rejudge=args.rejudge,
                )
            )
        except cache_identity.CacheMismatch as exc:
            # 缓存身份不符是**要人决定**的事，不是程序该猜的：重检索还是重判，代价差别很大。
            print(f"缓存不可复用：\n{exc}", file=sys.stderr)
            return 2
    summary = summarize(out)
    dump = Path(args.dump)
    dump.parent.mkdir(parents=True, exist_ok=True)
    # 先落盘再渲染：上一轮标定脚本因为渲染里抛错，四分钟的测量结果一起没了。
    dump.write_text(
        json.dumps({"summary": summary, **out}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    report = render(out, summary)
    print(report)
    if args.save:
        Path(args.save).write_text(report + "\n", encoding="utf-8")
        print(f"\n已写入 {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
