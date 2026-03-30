# Exp03: Full Transformer 生成式 CTR 模型

## 实验目标

构建完整版 Transformer 架构的生成式 CTR 模型，将 CTR 预测建模为序列生成问题。

## 模型架构

```
Input: 26个类别特征
  ↓
Embedding Layer: 独立嵌入表，每个特征 embed_dim=32
  ↓
Position Encoding: 可学习的位置编码 (已验证有害)
  ↓
CLS Token: 用于聚合序列信息
  ↓
Transformer Encoder: 2层自注意力，num_heads=4
  ↓
MLP Head: [128 → 64 → 1]
  ↓
Output: CTR 预测概率
```

## 实验配置

- **embed_dim**: 32
- **num_heads**: 4
- **num_layers**: 2
- **dropout**: 0.1
- **optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **scheduler**: CosineAnnealingLR
- **epochs**: 3
- **batch_size**: 2048

## 实验结果

| 指标 | 值 |
|------|-----|
| AUC | **0.7678** |
| PCOC | 1.02 |
| 参数量 | ~2.3M |

## 与基线对比

| 模型 | AUC | 相对提升 |
|------|-----|----------|
| DeepFM 基线 | 0.7472 | - |
| **FullTransformer V2** | **0.7678** | **+2.06%** |

## 关键发现

1. **生成式建模有效**: 将特征视为序列，通过自注意力学习特征交互，超越传统 CTR 模型
2. **独立嵌入表优于共享嵌入表**: 每个特征独立嵌入表比共享嵌入表效果更好
3. **位置编码在 CTR 场景有害**: 消融实验证明去掉位置编码可提升 1.6bp (详见 exp10)

## 后续优化方向

1. 移除位置编码，预期 AUC 可达 0.77+
2. 尝试 Field Embedding 替代位置编码
3. 增加特征交叉显式建模
