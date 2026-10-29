#!/usr/bin/env bash
# 从干净提交重跑 P0-2 点名的关键实验。参数逐份照原报告头部抄，不凭记忆。
set -u
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python
V2=rag4prefa/terms/public_to_standard.v2.json
V1=rag4prefa/terms/public_to_standard.json
BASE=eval/dataset.jsonl
PARA=eval/dataset.paraphrase.jsonl

run() {
  echo "===== $(date +%H:%M:%S) $*"
  $PY scripts/eval.py "$@" 2>&1 | tail -22
  echo "===== exit=$? ----"
}

run --dataset "$PARA" --top-k 5 --modes dense,dense_rerank,sparse,hybrid,rerank --fusion rrf --save
run --dataset "$PARA" --top-k 5 --modes dense,dense_rerank,sparse,hybrid,rerank --fusion zscore --save
run --dataset "$BASE" --top-k 5 --modes dense,dense_rerank,sparse,hybrid,rerank --fusion rrf --save
run --dataset "$BASE" --top-k 5 --modes dense,dense_rerank,sparse,hybrid,rerank --fusion zscore --save
run --dataset "$BASE" --top-k 5 --modes hybrid,hybrid_parallel,rerank,rerank_parallel \
    --term-map "$V2" --term-form terms --term-rerank terms --save
run --dataset "$BASE" --top-k 5 --modes hybrid,hybrid_parallel,rerank,rerank_parallel \
    --term-map "$V2" --term-form terms --term-rerank append --save
run --dataset "$PARA" --top-k 5 --modes hybrid,hybrid_parallel,rerank,rerank_parallel \
    --term-map "$V2" --term-form terms --term-rerank append --save
run --dataset "$PARA" --top-k 5 --modes hybrid,hybrid_parallel,rerank,rerank_parallel \
    --term-map "$V2" --term-form joined --term-rerank append --save
run --dataset "$PARA" --top-k 5 --modes rerank,rerank_parallel \
    --term-map "$V1" --term-form terms --term-rerank append --save

echo "===== ALL DONE $(date +%H:%M:%S)"
