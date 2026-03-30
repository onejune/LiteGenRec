# Exp05: 基于 MiniMind 的轻量级 CTR 模型

## 目标
探索大模型在程序化广告推荐中的轻量化落地方案，基于 MiniMind 思路构建高效 CTR 模型。

## 模型架构
- **模型**: MiniMindCTR_RoPE (带旋转位置编码的轻量级 CTR)
- **核心**: 轻量级 Transformer + RoPE + 参数高效设计
- **特征处理**: 
  - 稠密特征 (I1-I13): 归一化 + 离散化 + Embedding
  - 稀疏特征 (C1-C26): Label Encoding + Embedding
- **关键技术**: 
  - 旋转位置编码 (RoPE)
  - 轻量级多头注意力
  - GELU 激活函数

## 模型配置
- Embedding 维度: 32
- 隐藏层维度: 128
- 注意力头数: 4
- 层数: 2
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
- [x] 模型架构设计 (MiniMind 风格)
- [x] RoPE 旋转位置编码实现
- [ ] 模型训练
- [ ] 结果评估
- [ ] 与其它模型对比