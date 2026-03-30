# Exp16: 完整特征 + 多架构对比

## 实验目标

1. 验证完整特征 (39 维) + 无位置编码的效果
2. 对比 Transformer / HSTU-Lite / Mamba4CTR 三种架构

## 实验配置

| 参数 | 值 |
|------|-----|
| 稠密特征 | 13 个 (I1-I13)，离散化 50 bins |
| 稀疏特征 | 26 个 (C1-C26) |
| 总特征数 | 39 |
| embed_dim | 64 |
| num_heads | 8 |
| num_layers | 4 |
| 位置编码 | ❌ 无 |
| epochs | 3 |
| batch_size | 2048 |

## 对比架构

### 1. Transformer (无位置编码)
- 标准 Transformer Encoder
- 移除位置编码
- Pre-LN 结构

### 2. HSTU-Lite
- Pointwise Attention（点积注意力）
- 比 Multi-Head Attention 更轻量
- 层次化结构

### 3. Mamba4CTR
- 状态空间模型 (SSM)
- 序列建模能力
- 线性复杂度

## 预期结果

| 模型 | 预期 AUC | 理由 |
|------|----------|------|
| Transformer | 0.77+ | 完整特征 + 无位置编码 |
| HSTU-Lite | 0.76-0.77 | 轻量注意力 |
| Mamba4CTR | 0.75-0.76 | SSM 探索 |

## 运行方式

```bash
cd /mnt/workspace/git_project/LiteGenRec/experiments/exp16_full_features_no_pos
python train.py
```
