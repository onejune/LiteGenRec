# Exp04: SOTA 模型对比实验

## 目标
对比多种主流 CTR 模型在 Criteo 数据集上的性能，包括传统机器学习模型和深度学习模型。

## 模型列表
1. **Logistic Regression** (传统线性模型)
2. **Random Forest** (传统集成模型)
3. **XGBoost** (梯度提升树)
4. **DeepFM** (经典深度学习 CTR 模型)
5. **Wide & Deep** (Google 经典模型)
6. **LiteGenRec V1** (生成式 CTR 模型 - 我们的模型)

## 数据集
- **来源**: Criteo (标准分割)
- **特征**: 13个稠密特征 + 26个稀疏特征
- **训练集**: ~675K 样本
- **验证集**: ~75K 样本

## 评估指标
- AUC (Area Under Curve)
- LogLoss
- PCOC (Predicted Click Over Click)

## 实验设计
- 所有模型使用相同的训练/验证集划分
- 统一的数据预处理流程
- 相同的评估指标计算方式

## 进度
- [x] 模型实现 (LR, RF, XGB, DeepFM, Wide&Deep)
- [x] 数据预处理流程
- [ ] 模型训练
- [ ] 结果评估
- [ ] 性能对比分析