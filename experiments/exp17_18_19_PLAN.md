# Exp17-19 实验计划

**基于最新结果**: Transformer (exp16) AUC 0.8043 为 SOTA  
**关键发现**: 无位置编码 + 完整特征 + 标准Transformer 最优  
**日期**: 2026-03-31

---

## 一、当前最佳配置 (Exp16 SOTA)

```yaml
模型: Transformer Encoder
特征: 完整 (39 维 = 26 稀疏 + 13 稠密)
位置编码: ❌ 无
embed_dim: 64
num_heads: 8
num_layers: 4
hidden_dim: 256
dropout: 0.1
epochs: 3
batch_size: 2048
AUC: 0.8043
```

---

## 二、新实验设计

### Exp17: Attention Residuals for CTR

**目标**: 验证 AttnRes 在浅层 CTR 模型上的效果

**动机**:
- AttnRes 设计用于深层网络 (>32层)，解决 PreNorm Dilution
- 当前模型仅 4 层，是否仍有收益？

**实验组**:

| 变体 | 配置 | 预期 AUC |
|------|------|----------|
| 基线 (exp16) | 标准 Transformer (4层) | 0.8043 |
| Exp17-V1 | Full AttnRes (每层注意到所有前层) | 0.8045 (+0.2bp) |
| **Exp17-V2** | **Block AttnRes (4 blocks)** | **0.8050 (+0.7bp)** |
| Exp17-V3 | Block AttnRes (2 blocks) | 0.8048 (+0.5bp) |

**预期**: 收益有限 (+0.2~0.7bp)，因为层数太浅

---

### Exp18: 深层网络 + AttnRes 突破

**目标**: 通过加深网络 + AttnRes 突破 0.8043

**假设**:
- 深层网络 (8/12/16层) 表达能力更强
- 但标准残差会导致 PreNorm Dilution
- AttnRes 可以解决这个问题

**实验组**:

| 变体 | 层数 | 残差类型 | 预期 AUC | 说明 |
|------|------|----------|----------|------|
| 基线 (exp16) | 4 | 标准 | 0.8043 | 当前 SOTA |
| Exp18-V1 | 8 | 标准 | 0.8030 | 预期下降 |
| **Exp18-V2** | **8** | **AttnRes** | **0.8065** | **+2.2bp** |
| Exp18-V3 | 12 | 标准 | 0.8015 | 继续下降 |
| **Exp18-V4** | **12** | **AttnRes** | **0.8080** | **+3.7bp** |
| Exp18-V5 | 16 | 标准 | 0.8000 | 严重稀释 |
| **Exp18-V6** | **16** | **AttnRes** | **0.8095** | **+5.2bp** |

**预期**: 深层 + AttnRes 可能突破当前上限

---

### Exp19: 架构优化消融

**目标**: 在 Exp16 SOTA 基础上优化

**实验组**:

| 变体 | 优化点 | 预期 AUC | 说明 |
|------|--------|----------|------|
| 基线 (exp16) | - | 0.8043 | SOTA |
| Exp19-V1 | + 更长训练 (5 epochs) | 0.8050 | 欠拟合假设 |
| Exp19-V2 | + 更大模型 (embed_dim=128) | 0.8055 | 容量不足假设 |
| Exp19-V3 | + Interaction Attention (HSTU-Lite) | 0.8060 | 特征交互增强 |
| Exp19-V4 | + SwiGLU FFN | 0.8052 | 现代 FFN |
| Exp19-V5 | + RMSNorm (替代 LayerNorm) | 0.8045 | 归一化优化 |

---

## 三、实现优先级

```
Priority 1: Exp18 (深层突破)
├── 预期收益最大 (+2~5bp)
├── 验证 AttnRes 核心价值
└── 时间: 1-2 天

Priority 2: Exp19 (架构优化)
├── 预期收益中等 (+0.5~2bp)
├── 在 SOTA 基础上增量优化
└── 时间: 1 天

Priority 3: Exp17 (AttnRes 基础)
├── 预期收益最小 (+0.2~0.7bp)
├── 仅作为对照实验
└── 时间: 0.5 天
```

---

## 四、文件结构

```
experiments/
├── exp17_attnres/
│   ├── attnres_layer.py        # AttnRes 实现
│   ├── transformer_attnres.py  # Transformer + AttnRes
│   ├── train.py
│   └── results.json
├── exp18_deep_attnres/
│   ├── deep_models.py          # 深层模型定义
│   ├── train_deep.py           # 训练脚本
│   └── results.json
├── exp19_architecture_ablation/
│   ├── optimizations.py        # 各种优化
│   └── results.json
└── exp17_18_19_PLAN.md         # 本文件
```

---

## 五、时间规划

```
Day 1: Exp18 (深层 + AttnRes)
├── 上午: 实现 8/12/16 层模型
├── 下午: 训练并对比结果
└── 晚上: 分析是否突破 SOTA

Day 2: Exp19 (架构优化)
├── 上午: 实现 V1-V5 优化
├── 下午: 快速训练验证
└── 晚上: 确定最优配置

Day 3: Exp17 + 总结
├── 上午: Exp17 对照实验
├── 下午: 整理实验报告
└── 晚上: 提交 git
```

---

## 六、成功标准

| 实验 | 成功标准 | 失败标准 |
|------|----------|----------|
| Exp17 | +0.5bp 以上 | < 0.2bp |
| **Exp18** | **突破 0.8050** | **低于 0.8043** |
| Exp19 | +1.0bp 以上 | < 0.5bp |

---

*计划制定：萧十一郎 ⚔️*
*日期：2026-03-31*
