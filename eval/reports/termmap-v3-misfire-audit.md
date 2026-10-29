# 词表误触发审计

- 词表：`28` 条 · 意图分布 acceptance 1、boundary 3、performance 7、policy 7、practice 9、prohibition 1
- 负例闸门：7 条，违反 0 条

## 各数据集触发率

| 数据集 | 角色 | 题数 | 触发 | 触发率 | 按意图（题） |
|---|---|---|---|---|---|
| `dataset.paraphrase.jsonl` | 口语集（该碰） | 50 | 14 | 28% | policy 4、practice 4、performance 3、boundary 2、prohibition 1 |
| `dataset.jsonl` | 专业问法集（基本不该碰） | 31 | 3 | 10% | policy 2、boundary 1、acceptance 1 |
| `dataset.refuse.jsonl` | 库外集（只管有没有加重误导） | 6 | 1 | 17% | policy 1 |

## dataset.paraphrase.jsonl 逐题

- 装修完想拿补贴，得先准备哪些材料？
  → subsidy-apply(policy)←补贴 + 材料
- 厨房没窗户，还能装燃气表吗？
  → gas-ventilation(practice)←燃气 + 窗户
- 卫生间想装成轮椅能进去的，门口那个坎怎么做？
  → accessibility(practice)←轮椅 + 坎 + 进去
- 墙里的电线以后坏了能换吗？
  → pipeline-hidden(practice)←电线 + 墙里
- 不是新盖的老房子，也能用这套做法吗？
  → existing-building(boundary)←老房子 + 能用
- 集成吊顶里头包不包含灯具？
  → ceiling-scope(boundary)←吊顶 + 包含 + 灯具
- 装修好的东西以后坏了，保修多久？
  → warranty(policy)←保修 + 多久
- 吊顶里走了管子，以后怎么检修？
  → access-panel(practice)←检修 + 吊顶
- 装修材料防火有没有硬性要求？
  → fire-rating(performance)←防火 + 材料
- 卫生间里洗澡老是怕触电，有什么保护措施？
  → rcd(performance)←触电 + 保护
- 装修能不能把承重墙砸掉？
  → load-bearing(prohibition)←承重墙 + 能不能
- 家里的配电箱要不要装漏电保护？
  → rcd(performance)←漏电 + 保护
- 补贴批下来多久能到账？
  → subsidy-payout(policy)←补贴 + 到账
- 合同、发票、收款账号必须是同一个人吗？
  → invoice(policy)←发票 + 必须

## dataset.jsonl 逐题

- 申领旧房装修厨卫改造补贴需要提交哪些材料？
  → subsidy-apply(policy)←补贴 + 材料
- 2025 年补贴物品材料品类中的集成吊顶是否包含电器设备？
  → ceiling-scope(boundary)←吊顶 + 包含 + 电器 + subsidy-apply(policy)←补贴 + 材料
- 隐蔽工程验收记录应包含哪些内容？
  → acceptance-hidden(acceptance)←验收 + 隐蔽

### 多意图冲突：1 题

- 2025 年补贴物品材料品类中的集成吊顶是否包含电器设备？ → ceiling-scope/boundary + subsidy-apply/policy

## dataset.refuse.jsonl 逐题

- 北京市装配化装修的补贴标准是多少？
  → subsidy-amount(policy)←补贴 + 多少

## 单字规则挡掉了什么（strict=False 的反事实）

**dataset.paraphrase.jsonl**
- 两块板拼的缝，以后会不会裂开？ → 挡掉 ['panel-crack']

**dataset.jsonl**
- 架空楼地面内敷设管线时，检修口宜设置在什么位置？ → 挡掉 ['pipeline-hidden']
- 装配式内装修的设备和管线与主体结构的关系应遵循什么原则？ → 挡掉 ['pipeline-hidden']

