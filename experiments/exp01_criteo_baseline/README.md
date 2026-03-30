# Exp01: Criteo 基线实验

## 实验目标

在 Criteo 数据集上建立基线模型，为后续生成式 CTR 模型提供对比基准。

## 数据集

- **数据集**: Criteo Display Advertising Challenge
- **训练集**: ~1.35M 样本
- **验证集**: ~75K 样本
- **特征**: 26 个类别特征 + 13 个数值特征（本实验仅使用类别特征）

## 模型配置

| 模型 | 参数量 | 配置 |
|------|--------|------|
| DeepFM | ~2.1M | embed_dim=32, dnn_hidden=[128, 64] |
| DCN | ~1.8M | embed_dim=32, cross_layers=3 |

## 实验结果

| 模型 | AUC | 备注 |
|------|-----|------|
| DeepFM | 0.7472 | 基线 |
| DCN | 0.7410 | - |

## 关键发现

1. DeepFM 作为经典 CTR 模型，在 Criteo 数据集上表现稳定
2. 后续生成式模型需要超越 0.7472 AUC 才能证明有效性

## 文件说明

- `results.json`: 实验结果数据
