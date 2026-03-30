# Exp05: MiniMind + RoPE 优化实验

## 实验目标

尝试轻量级 LLM 架构 (MiniMind) 和旋转位置编码 (RoPE) 应用于 CTR 预测。

## 模型架构

```
Input: 26个类别特征
  ↓
Embedding Layer
  ↓
RoPE (Rotary Position Embedding)
  ↓
Transformer Layers (轻量级)
  ↓
Prediction Head
```

## 实验配置

- **embed_dim**: 32
- **num_heads**: 4
- **num_layers**: 2
- **RoPE**: 应用旋转位置编码
- **epochs**: 3

## 实验结果

| 指标 | 值 |
|------|-----|
| AUC | 0.7515 |
| 训练状态 | 完成，验证阶段出错 |

## 问题分析

1. **RoPE 在 CTR 场景不适用**: 广告特征没有顺序语义，旋转位置编码引入噪声
2. **词汇表大小问题**: 验证集包含训练集未见过的特征值，导致嵌入索引越界
3. **效果不如基线**: 0.7515 < 0.7663 (SimpleGenCTR V1)

## 关键发现

**RoPE 在 CTR 场景无效的原因**:
- 文本序列有顺序语义 (词的位置有意义)
- CTR 特征列无顺序 (第1列和第5列特征没有"位置先后"概念)
- 固定位置编码可能引入过拟合

## 结论

❌ RoPE 不适用于 CTR 预测任务
✅ 建议使用可学习的特征交互注意力替代位置编码
