# Exp03: Full Transformer 生成式 CTR 模型

## 实验目标

构建完整版 Transformer 架构的生成式 CTR 模型，将 CTR 预测建模为序列生成问题。

## 数据集说明

### Criteo Display Advertising Challenge

- **来源**: Kaggle Competition (2014)
- **任务**: 预测用户是否会点击广告
- **规模**: 7天点击日志，约 4500 万样本

### 本实验使用的数据

| 项目 | 值 |
|------|-----|
| **数据集名称** | Criteo Standard |
| **训练集样本数** | 1,345,295 |
| **验证集样本数** | 75,071 |
| **类别特征数** | 26 个 (C1-C26) |
| **数值特征数** | 13 个 (I1-I13)，本实验未使用 |
| **标签分布** | 点击率约 25% |

### 数据路径

```
/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet/
```

### 数据预处理

1. **类别特征编码**: LabelEncoder，缺失值填充后编码为正整数
2. **词汇表大小**: 每个特征的词汇表大小不同，最大约 500K
3. **嵌入表**: 每个特征独立嵌入表，padding_idx=0

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

| 参数 | 值 |
|------|-----|
| embed_dim | 32 |
| num_heads | 4 |
| num_layers | 2 |
| dropout | 0.1 |
| optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| scheduler | CosineAnnealingLR |
| epochs | 3 |
| batch_size | 2048 |

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
