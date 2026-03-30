# Exp01: Criteo 基线实验

## 实验目标

在 Criteo 数据集上建立基线模型，为后续生成式 CTR 模型提供对比基准。

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
| **数值特征数** | 13 个 (I1-I13) |
| **标签分布** | 点击率约 25% |

### 数据路径

```
/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet/
├── train_train.parquet/   # 训练集 (多个 parquet 文件)
└── train_valid.parquet/   # 验证集
```

### 数据预处理

1. **类别特征**: 填充缺失值为 `__MISSING__`，使用 LabelEncoder 编码
2. **数值特征**: 本实验未使用
3. **词汇表**: 在全量数据上 fit，保证训练集和验证集编码一致

## 模型配置

| 模型 | 参数量 | 配置 |
|------|--------|------|
| DeepFM | ~2.1M | embed_dim=32, dnn_hidden=[128, 64] |
| DCN | ~1.8M | embed_dim=32, cross_layers=3 |

## 实验结果

| 模型 | AUC | PCOC | 备注 |
|------|-----|------|------|
| DeepFM | 0.7472 | 1.01 | DL基线 |
| DCN | 0.7410 | - | - |

## 关键发现

1. DeepFM 作为经典 CTR 模型，在 Criteo 数据集上表现稳定
2. 后续生成式模型需要超越 0.7472 AUC 才能证明有效性

## 文件说明

- `results.json`: 实验结果数据
- `README.md`: 本文件
