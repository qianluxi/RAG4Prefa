# 检索评估报告

- 标注有效的题目数：10（跳过 0 条无标注/无问题的记录）
- 指标：Recall@5、MRR、nDCG@5，相关性按 chunk 二值判定
- Recall@5 = |TopK ∩ Gold| / |Gold|，Gold 为**本题筛选范围内**的相关 chunk 集合
- 片段标注已解析成范围内包含它的全部 chunk，三个指标共用同一个 Gold
- 坐标系：embedding `Qwen/Qwen3-Embedding-0.6B`（1024 维）· chunk 80~480 token · recall_k=50 · rrf_k=60 · 重排 `Qwen/Qwen3-Reranker-0.6B`
- 语料快照：9 份文档 / 152 个片段 · ids_sha `4a096f6bd0a23d13`
- 数据集：`eval\dataset.paraphrase.jsonl` · 10 题 · sha `f2361b35d1ca09f7`
- 生成于 2026-10-28T10:42:19+08:00 · 代码 `399d0cb`（有未提交改动）

| 召回方式 | 有效样本 | Recall@5 | MRR | nDCG@5 |
|---|---|---|---|---|
| dense | 10 | 0.400 | 0.400 | 0.400 |
| sparse | 10 | 0.700 | 0.492 | 0.543 |
| hybrid | 10 | 0.600 | 0.550 | 0.563 |
| rerank | 10 | 0.800 | 0.650 | 0.686 |

混合召回相对纯向量：Recall@5 +0.200
混合召回相对纯稀疏：Recall@5 -0.100
开重排相对 hybrid：Recall@5 +0.200