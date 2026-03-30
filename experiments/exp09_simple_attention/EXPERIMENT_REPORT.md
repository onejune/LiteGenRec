# Exp09: Simple Attention 实验

## 实验目标

使用最简单的注意力机制作为 baseline，验证复杂架构的必要性。

## 模型架构

```
Input: 26个类别特征
  ↓
Embedding Layer
  ↓
Simple Attention: single head, no FFN
  ↓
Mean Pooling
  ↓
MLP Head
  ↓
CTR 预测
```

## 实验配置

- **embed_dim**: 32
- **attention**: 单头注意力，无 FFN
- **pooling**: mean pooling
- **epochs**: 3

## 实验结果

| 指标 | 值 |
|------|-----|
| AUC | 0.7391 |
| 参数量 | ~1.2M |

## 与其他模型对比

| 模型 | AUC | 备注 |
|------|-----|------|
| DeepFM 基线 | 0.7472 | -0.81% ❌ |
| FullTransformer V2 | 0.7678 | -2.87% ❌ |
| **Simple Attention** | 0.7391 | 本实验 |

## 分析

1. **低于所有基线**: 简化过度导致性能下降
2. **缺少 FFN**: 没有 FFN 层，模型表达能力不足
3. **Mean Pooling 信息损失**: 简单平均丢失重要信息

## 关键发现

❌ **过度简化不可取**
- Transformer 的 FFN 层对性能至关重要
- CLS Token 池化优于 Mean Pooling
- 多头注意力比单头效果更好

## 最小有效架构

从消融实验反推，最小有效架构应该包含:
1. ✅ 独立嵌入表
2. ✅ 多头注意力 (至少 2 heads)
3. ✅ FFN 层
4. ✅ CLS Token 池化
5. ❌ 位置编码 (可以去掉)

## 结论

⚠️ 架构简化有下限，Simple Attention 过于简化导致性能显著下降
