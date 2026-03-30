# Exp03: 完整版 Transformer 架构 (LiteGenRec V2)

## 目标
实现完整的 Transformer 架构生成式 CTR 模型，进一步提升性能。

## 模型架构
- **模型**: FullTransformerGenCTR (完整版 Transformer-based CTR)
- **核心**: 完整 Transformer 编码器 + 多头注意力特征交互
- **特征处理**: 
  - 稠密特征 (I1-I13): 归一化 + 离散化 + Embedding
  - 稀疏特征 (C1-C26): Label Encoding + Embedding
- **序列长度**: 39 (13+26 个特征)
- **输出**: 点击概率预测

## 模型配置
- Embedding 维度: 64
- 隐藏层维度: 256
- 注意力头数: 8
- Transformer 层数: 4
- Dropout: 0.1
- 优化器: Adam (lr=1e-3)

## 数据集
- **来源**: Criteo (标准分割)
- **特征**: 13个稠密特征 + 26个稀疏特征
- **训练集**: ~675K 样本
- **验证集**: ~75K 样本

## 评估指标
- AUC (Area Under Curve)
- LogLoss
- PCOC (Predicted Click Over Click)

## 进度
- [x] 模型架构设计
- [ ] 模型训练
- [ ] 结果评估
- [ ] 与 V1 版本对比