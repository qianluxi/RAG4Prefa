"""判官实测：规则闸门 vs 让模型判"这批片段够不够"。

回答的问题只有一个：**在 `near_topic`（主题相近但答不了）这一类上，判官能不能拦住，
同时不误伤库内题**。它默认不进生产判定（见 `sufficiency.py` 第二层注释），
数字过关才谈接入。

为什么要先冻结候选：规则闸门与判官必须看**同一批片段**。两次检索各跑一遍，重排服务的
末位近分候选会翻（实测同池同进程重排 4 次名次一致，跨运行浮动 1~2 位），
"判官比闸门多拦了一题"就可能只是它拿到了不同的候选。所以候选先落盘，
之后所有重跑（换提示词、换模型、调 top_n）都从同一份候选出发。

用法：
    python scripts/judge_sufficiency.py --limit 5          # 先冒烟
    python scripts/judge_sufficiency.py                    # 全量，候选走缓存
    python scripts/judge_sufficiency.py --rebuild          # 重新检索
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag4prefa import ask, prompts, repo, sufficiency  # noqa: E402
from rag4prefa.domain import Filter, Hit  # noqa: E402
from rag4prefa.services import build_services  # noqa: E402

# 库内题集：判官在这里唯一的任务是**别乱拒**。四份合起来 122 题，
# 与 `eval/refusal-20261029.md` 的误拒口径一致，两次报告的"误拒 0"才是同一件事。
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


def load_cache(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("hits", {})


def save_cache(path: Path, hits: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "hits": hits}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


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
            return services.llm.complete(system, user, max_tokens=200)
        except Exception as exc:  # noqa: BLE001 - 限流、超时、5xx 同一处理
            last = exc
            if delay is None:
                break
            print(f"    判官调用失败（{type(exc).__name__}），{delay}s 后重试", file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError(f"判官调用失败：{last}")


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
    system = prompts.JUDGE_PROMPTS[judge]
    # 缓存 key 必须带模型：魔搭与硅基流动上有同名模型，行为并不相同（一个默认不思考、
    # 一个默认思考）。只按「判官版本 + k」做 key，换服务商后会静默读到上一家的结论。
    model = services.settings.llm_model
    verdict_path = verdict_path or Path("eval/reports/judge-verdicts.json")
    ev = _eval_module()
    cache = {} if rebuild else load_cache(cache_path)
    known_projects = repo.project_names(services.conn)

    prepared: list[dict] = []
    fetched = 0
    for i, item in enumerate(work, start=1):
        question = item["question"]
        request_filter = Filter(**item["filter"]) if item["filter"] else Filter()
        effective = ask.build_filter(question, request_filter, known_projects)
        if question in cache:
            hits = [Hit(**d) for d in cache[question]]
        else:
            hits = await _retrieve(ev, services, mode, question, effective, k)
            cache[question] = [h.model_dump() for h in hits]
            fetched += 1
            # 每检索一条就落一次盘：一轮十分钟，中途断掉不该全丢。
            save_cache(cache_path, cache, {"mode": mode, "k": k, "partial": True})
        top = hits[:k]
        gate = sufficiency.judge(
            services.conn, question, top,
            min_rerank=services.settings.refuse_min_rerank,
            min_dense=services.settings.refuse_min_dense,
        )
        prepared.append({"item": item, "top": top, "gate": gate, "index": i})
        if progress:
            print(f"[取候选 {i}/{len(work)}] {question[:30]}", file=sys.stderr)
    save_cache(cache_path, cache, {"mode": mode, "k": k, "partial": False})

    # 判官结论单独按题缓存：**候选缓存只解决一半问题**。2026-10-30 就是候选全在缓存里、
    # 判官结论没落盘，跑到 118/156 被 429 打断后，那 118 次调用白花。换提示词做 A/B 时
    # 这个缓存也必须能区分版本（key 里带 judge 版本与 k），否则 v2 会读到 v1 的结论。
    verdicts = {} if rejudge else load_cache(verdict_path)
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
        save_cache(verdict_path, verdicts,
                       {"judge": judge, "model": model, "k": k, "partial": True})

    async def one(entry: dict) -> dict:
        nonlocal done
        item, top, gate = entry["item"], entry["top"], entry["gate"]
        key = f"{judge}|{model}|{k}|{item['question']}"
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
    save_cache(verdict_path, verdicts,
                   {"judge": judge, "model": model, "k": k, "partial": False})
    return {"results": results, "fetched": fetched, "mode": mode, "k": k,
            "cache": str(cache_path), "judge": judge, "model": model,
            "errors": sum(1 for r in results if r["judge_error"])}


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
        f"召回方式 `{out['mode']}`、`k={out['k']}`"
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

    lines += [
        "",
        "## 五、可复现性",
        "",
        f"- 候选缓存：`{out['cache']}`（本轮新检索 {out['fetched']} 题，其余复用）。",
        "- 判官输出原样留在 `--dump` 的 JSON 里（含 `judge_raw`），判错的题可以逐条回看原话。",
        f"- 本轮判官调用失败 {out.get('errors', 0)} 题（记为未表态，不计入拒答）；"
        "结论缓存按「版本+题」存，续跑只补缺口。",
        "- **判官结论依赖当时的 LLM 版本**，与重排服务同理：换模型必须重跑，不能沿用这份数字。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refuse", default="eval/dataset.refuse.jsonl")
    ap.add_argument("--positive", nargs="*", default=list(POSITIVE_SETS))
    ap.add_argument("--mode", default="rerank", help="召回方式（默认生产用的 rerank）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cache", default="eval/reports/judge-hits.json")
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存重新检索")
    ap.add_argument("--verdict-cache", default="eval/reports/judge-verdicts.json",
                    help="判官结论缓存（按版本+每题存，续跑只补缺口）")
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
        out = {k: saved[k] for k in ("results", "mode", "k", "cache")}
        out.setdefault("judge", saved.get("judge", "v1"))
        out["fetched"] = 0
    else:
        work = build_worklist(Path(args.refuse), [Path(p) for p in args.positive])
        if args.limit:
            work = work[: args.limit]
        out = asyncio.run(
            measure(
                work, mode=args.mode, k=args.k, cache_path=Path(args.cache),
                rebuild=args.rebuild, progress=not args.quiet,
                concurrency=args.concurrency, judge=args.judge,
                verdict_path=Path(args.verdict_cache), rejudge=args.rejudge,
            )
        )
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
