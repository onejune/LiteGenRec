# Exp02: 生成式 CTR 模型 (LiteGenRec V1)

## 目标
基于 MiniOneRec 思路，使用序列生成建模 CTR 预估，探索大模型在程序化广告推荐中的应用。

## 模型架构
- **模型**: GenCTR (Generation-based CTR)
- **核心**: Transformer 编码器 + 序列建模
- **特征处理**: 
  - 稠密特征 (I1-I13): 归一化 + 离散化 + Embedding
  - 稀疏特征 (C1-C26): Label Encoding + Embedding
- **输出**: 点击概率预测

## 模型配置
- Embedding 维度: 64
- 隐藏层维度: 256
- 注意力头数: 8
- Transformer 层数: 3
- 最大序列长度: 50
- Dropout: 0.1

## 数据集
- **来源**: Criteo (标准分割)
- **特征**: 13个稠密特征 + 26个稀疏特征
- **训练集**: ~1.35M 样本
- **验证集**: ~75K 样本

## 评估指标
- AUC (Area Under Curve)
- LogLoss
- PCOC (Predicted Click Over Click)

## 进度
- [x] 模型架构设计
- [x] 数据预处理流程
- [ ] 模型训练
- [ ] 结果评估
- [ ] 与 DeepFM 基线对比