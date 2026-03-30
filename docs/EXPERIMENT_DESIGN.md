# LiteGenRec 后续实验设计方案

> 调研时间: 2026-03-30
> 最后更新: 2026-03-30 19:14
> 基于: 消融实验结果 + 最新研究进展

## 一、当前最佳结果

| 模型 | AUC | 架构特点 |
|------|-----|----------|
| DeepFM 基线 | 0.7472 | 传统 CTR 模型 |
| SimpleGenCTR V1 | 0.7663 | FFN 序列建模 |
| FullTransformer V2 | 0.7678 | Transformer + Interaction Attention |
| **Mamba4CTR V2** | **0.7846** | State Space Model + Dense Features 🥇 |
| **HSTU-Lite V2** | **0.7845** | Pointwise Attention + Dense Features 🥈 |
| **Hierarchical Semantic V2** | **0.7841** | VQ-VAE + Attention + Dense Features 🥉 |

**关键发现**:
- 稠密特征贡献约 3bp AUC
- 三种新架构都超越了 Transformer 基线

## 二、新架构调研总结

### 1. HSTU (Meta, ICML 2024) ⭐⭐⭐⭐⭐

**论文**: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations

**核心创新**:
- **Pointwise Aggregated Attention**: 用 pointwise normalization 替代 softmax，适合非平稳词汇表
- **高效稀疏注意力**: 将注意力计算转换为 grouped GEMMs，GPU 友好
- **Stochastic Length (SL)**: 随机长度采样，减少计算成本
- **内存效率**: 减少 linear layers，支持 2x 更深网络

**性能**:
- 比 FlashAttention2 Transformer 快 5.3x - 15.2x
- NDCG 提升 65.8%
- 1.5 万亿参数，线上 A/B 提升 12.4%

**适用于 CTR 的原因**:
- ✅ 高基数特征处理
- ✅ 流式数据非平稳特性
- ✅ 效率优势明显

### 2. TIGER (NeurIPS 2023) ⭐⭐⭐⭐

**论文**: Recommender Systems with Generative Retrieval

**核心创新**:
- **Semantic ID**: 使用 RQ-VAE 将物品编码为语义 ID 元组
- **层次化表示**: 多层 ID 捕获不同粒度语义
- **生成式检索**: 预测下一个物品的 Semantic ID

**局限性**:
- ❌ 我们 exp06 实验证明 VQ-VAE 在 CTR 场景效果不佳
- ⚠️ 更适合检索场景，非 CTR 预测

### 3. Mamba / State Space Models ⭐⭐⭐⭐

**论文**: Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models

**核心创新**:
- **线性时间复杂度**: O(n) vs Transformer O(n²)
- **选择性状态空间**: 数据依赖的状态转换
- **长序列建模**: 更好地捕获长期依赖

**性能**:
- 推理速度比 Transformer 快
- 参数效率更高
- 适合长序列场景

**适用于 CTR 的原因**:
- ✅ 效率优势明显
- ✅ 26 个特征不算长序列，但可扩展性好
- ⚠️ 需要验证在 CTR 特征交互上的效果

### 4. Diffusion Models ⭐⭐⭐

**论文**: A Survey on Diffusion Models for Recommender Systems

**核心创新**:
- **去噪生成**: 学习数据分布，生成推荐
- **多模态融合**: 结合内容特征
- **鲁棒性**: 处理噪声数据

**局限性**:
- ❌ 推理速度慢（多步采样）
- ⚠️ 更适合生成场景，非判别式 CTR 预测

## 三、实验设计方案

### Exp11: LiteGenRec V3 (已创建)

**目标**: 移除位置编码，验证消融结果

**预期**: AUC 0.769+

---

### Exp12: HSTU-Lite for CTR ✅ 已完成

**结果**: AUC **0.7845** (+1.67bp vs V2 Transformer)

**架构**:
```
Input: 39个特征 (13稠密 + 26稀疏)
  ↓
Embedding Layer (独立嵌入表)
  ↓
HSTU Block (4层):
  - Pointwise Aggregated Attention
  - MLP with GELU
  ↓
CLS Token Pooling
  ↓
Prediction
```

**V3 版本**: 添加 Interaction Attention 层，运行中...

---

### Exp13: Mamba4CTR ✅ 已完成

**结果**: AUC **0.7846** (+1.68bp vs V2 Transformer) 🥇 最佳

**架构**:
```
Input: 39个特征
  ↓
Embedding Layer
  ↓
Mamba Block (4层):
  - Bidirectional State Space Model
  - Linear Time Complexity
  ↓
CLS Token Pooling
  ↓
Prediction
```

**V3 版本**: 添加 Interaction Attention 层，运行中...

---

### Exp14: Hierarchical Semantic CTR ✅ 已完成

**结果**: AUC **0.7841** (+1.63bp vs V2 Transformer)

**架构**:
```
Input: 39个特征
  ↓
Embedding Layer + Hierarchical VQ-VAE Encoder
  ↓
Multi-head Attention
  ↓
CLS Token Pooling
  ↓
Prediction
```

**V3 版本**: 添加第二层 Interaction Attention，运行中...

---

## 四、实验进度

| 状态 | 实验 | AUC | 备注 |
|------|------|-----|------|
| ✅ 完成 | V3 无位置编码 (exp12) | 0.7511 | 小配置验证 |
| ✅ 完成 | HSTU-Lite V2 (exp13) | **0.7845** | +1.67bp |
| ✅ 完成 | Mamba4CTR V2 (exp14) | **0.7846** | +1.68bp 🥇 |
| ✅ 完成 | Hierarchical Semantic V2 (exp15) | **0.7841** | +1.63bp |
| 🔄 进行中 | HSTU-Lite V3 | - | +Interaction Attention |
| 🔄 进行中 | Mamba4CTR V3 | - | +Interaction Attention |
| 🔄 进行中 | Hierarchical Semantic V3 | - | +Interaction Attention |

## 五、后续工作流

1. **先跑 V3** → 验证消融结果
2. **实现 HSTU-Lite** → 核心创新，SOTA 架构
3. **实现 Mamba4CTR** → 效率优化方向
4. **根据结果调整** → Hybrid 或其他变体

## 六、参考文献

1. Zhai et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations" ICML 2024
2. Rajput et al. "Recommender Systems with Generative Retrieval" NeurIPS 2023
3. Liu & Lin "Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models" arXiv 2024
4. Gu & Dao "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" 2024
