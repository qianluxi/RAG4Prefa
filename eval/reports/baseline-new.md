# 检索评估报告

- 标注有效的题目数：25（跳过 0 条无标注/无问题的记录）
- 指标：Recall@5、MRR、nDCG@5，相关性按 chunk 二值判定
- Recall@5 = |TopK ∩ Gold| / |Gold|，Gold 为**本题筛选范围内**的相关 chunk 集合
- 片段标注已解析成范围内包含它的全部 chunk，三个指标共用同一个 Gold
- 坐标系：embedding `Qwen/Qwen3-Embedding-0.6B`（1024 维）· chunk 80~480 token · recall_k=50 · rrf_k=60 · 重排 `Qwen/Qwen3-Reranker-0.6B`
- 数据集：`eval\dataset.jsonl`

| 召回方式 | 有效样本 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| dense | 25 | 0.960 | 0.777 | 0.822 |
| sparse | 25 | 0.920 | 0.687 | 0.746 |
| hybrid | 25 | 0.960 | 0.797 | 0.837 |
| rerank | 25 | 0.920 | 0.770 | 0.808 |

混合召回相对纯向量：Recall@5 +0.000
混合召回相对纯稀疏：Recall@5 +0.040
开重排相对 hybrid：Recall@5 -0.040