# Exp04: SOTA 模型对比实验

## 实验目标

对比多种经典 CTR 模型，建立完整的基准对比。

## 模型列表

| 模型 | 类型 | 特点 |
|------|------|------|
| Logistic Regression | 传统 ML | 线性模型基线 |
| Random Forest | 集成学习 | 树模型 |
| XGBoost | 集成学习 | GBDT |
| Wide&Deep | 深度学习 | 宽深结合 |
| DeepFM | 深度学习 | FM+DNN |
| DCN | 深度学习 | 交叉网络 |

## 实验结果

| 模型 | AUC | 备注 |
|------|-----|------|
| Logistic Regression | 0.6412 | 线性基线 |
| Random Forest | 0.7091 | - |
| XGBoost | 0.7420 | 树模型最佳 |
| Wide&Deep | 运行中 | 嵌入维度问题 |
| DeepFM | 0.7472 | DL基线 |
| DCN | 0.7410 | - |

## 关键发现

1. **XGBoost 表现优异**: 在小规模数据上，树模型仍有竞争力
2. **DeepFM 适合作为基线**: 深度模型中表现最稳定
3. **Wide&Deep 需要调整**: 嵌入维度计算需要修正

## 遇到的问题

- 嵌入表索引越界: 需要确保 vocab_size 包含所有可能的特征值
- 线性层维度不匹配: 需要正确计算输入维度
