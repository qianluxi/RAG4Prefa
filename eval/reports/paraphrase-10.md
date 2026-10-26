# 检索评估报告

- 标注有效的题目数：10（跳过 0 条无标注/无问题的记录）
- 指标：Recall@5、MRR、nDCG@5，相关性按 chunk 二值判定
- Recall@5 = |TopK ∩ Gold| / |Gold|，Gold 为**本题筛选范围内**的相关 chunk 集合
- 片段标注已解析成范围内包含它的全部 chunk，三个指标共用同一个 Gold
- 坐标系：embedding `Qwen/Qwen3-Embedding-0.6B`（1024 维）· chunk 80~480 token · recall_k=50 · rrf_k=60 · 重排 `Qwen/Qwen3-Reranker-0.6B`
- 数据集：`eval\dataset.paraphrase.jsonl`

| 召回方式 | 有效样本 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| dense | 10 | 0.500 | 0.333 | 0.376 |
| sparse | 10 | 0.500 | 0.275 | 0.332 |
| hybrid | 10 | 0.700 | 0.520 | 0.565 |
| rerank | 10 | 0.800 | 0.542 | 0.606 |

混合召回相对纯向量：Recall@5 +0.200
混合召回相对纯稀疏：Recall@5 +0.200
开重排相对 hybrid：Recall@5 +0.100