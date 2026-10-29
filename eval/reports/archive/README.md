# 历史评测报告（2026-10-28 归档）

这些报告生成时，评测输出还没有记录**语料快照**与**数据集哈希**。于是"两份报告是不是跑在同一批语料上"
只能靠回忆回答——而这个问题恰恰决定了它们能不能并列引用。

## 为什么整体归档

核查发现两处硬伤：

| 报告 | 题量 | 问题 |
|---|---|---|
| `baseline-30.md` | 30 | 当前 `eval/dataset.jsonl` 只有 25 题，两份报告却把数据集写成同一个路径 |
| `baseline-new.md` | 25 | 题量对得上，但当时库里只有 66 个片段（现在是 152） |
| `paraphrase-10.md` / `paraphrase-new.md` | 各 10 | 同一批题，dense 分别记 0.500 与 0.700，中间没有模型或参数改动记录 |

更直观的是**结论方向相反**：`baseline-30` 显示"纯稀疏 0.950 优于混合 0.900"，`baseline-new` 显示
"混合 0.960 优于纯稀疏 0.920"。两份并排放进开题报告，翻两页就能自相矛盾。

## 现行基线在哪

上一级目录的四份 `*-20261028-208ec157*.md`（语料 75 份 / 2560 片段 /
`ids_sha = 208ec1576dfc11cc`）：无后缀是 A 基线，`-expand` 是 B（扩展只改召回），
`-expand-rerank` 是 C（扩展连重排一起用），同名 `.jsonl` 是逐题证据。

## 为什么还有 `*-pool25-*`

2026-10-28 修了「声明 50 深、实得 25」的候选池 bug 之后，同名报告会被重新生成覆盖。
旧数字不能丢——它们是"bug 存在时世界长什么样"的唯一记录，所以按原样存成
`dataset-pool25-208ec157.md` 与 `dataset.paraphrase-pool25-208ec157.md`。
两组数字**不可直接比较**：口语集 `rerank 0.267 → 0.333` 就是池深从 25 修到 50 的效果，
不是重排变好了。

## 怎么保证不再重演

`scripts/eval.py` 现在把「语料快照 + 数据集哈希与题数 + 代码版本 + 工作区是否干净 + 生成时间」写进报告头部，
文件名带日期与语料指纹（`<数据集>-<日期>-<ids_sha 前 8 位>.md`）。语料一变，`--save` 生成的就是另一个文件，
不会覆盖上一份，也不会让两份不同语料的数字共用一个名字。

## 2026-10-29 第二批：`*.as-of-<提交>-dirty`

专家复审 P0-2 指出：报告头写着「代码 `3c77c36`（有未提交改动）」，等于这批报告
的代码身份是空的——`rev-parse` 在脏工作区里描述的不是跑出这组数字的那份代码。
按要求从干净提交重跑关键实验后，旧文件按原样归档到这里，后缀 `-dirty` 标明它
当时的身份缺口。

**为什么仍要留**：它们是"zscore 融合还没提交时是什么样"的原始记录；且新旧两份若
数字不同，差额本身就是重排服务漂移的度量（见 HANDOFF 关于 `dense_rerank` 0.839→0.806
的记录）。归档不改一个字节。

对应关系：

| 归档文件 | 旧报告里的代码行 | 现在去哪了 |
|---|---|---|
| `dataset-5d9cf3ff-…-fusion{rrf,zscore}.as-of-3c77c36-dirty.*` | `3c77c36`（有未提交改动） | `eval/reports/` 同名文件（干净提交重跑） |
| `dataset-5d9cf3ff-…-termmap-public_to_standard.v2-terms-w05-rerank{,append}-fusionzscore.as-of-18a00d3-dirty.*` | `18a00d3` | 同上 |
| `dataset.paraphrase-3da85a91-…-fusion{rrf,zscore}.as-of-3c77c36-dirty.*` | `3c77c36` | 同上 |
| `dataset.paraphrase-3da85a91-…-termmap-public_to_standard.v2-joined-w05-rerankappend-fusionzscore.as-of-7ddaf11-dirty.*` | `7ddaf11` | 同上 |
| `dataset.paraphrase-3da85a91-…-termmap-public_to_standard{,.v2}-terms-w05-rerankappend-fusionzscore.as-of-18a00d3-dirty.*` | `18a00d3` | 同上 |

没重跑的 11 份（`expand-parallel-*`、`termmap-terms-w0{3,5}`、`termmap-joined-w1-rerank`
等）都是已被证伪的旧实验线，结论早已写进 HANDOFF，不占用评测额度。
