# LiteGenRec 后续实验设计方案

> 调研时间: 2026-03-30
> 基于: 消融实验结果 + 最新研究进展

## 一、当前最佳结果

| 模型 | AUC | 架构特点 |
|------|-----|----------|
| DeepFM 基线 | 0.7472 | 传统 CTR 模型 |
| SimpleGenCTR V1 | 0.7663 | FFN 序列建模 |
| **FullTransformer V2** | **0.7678** | 自注意力 (最佳) |
| V3 预期 | 0.769+ | 移除位置编码 |

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

### Exp12: HSTU-Lite for CTR

**目标**: 实现 HSTU 的轻量化版本，应用于 CTR 预测

**架构**:
```
Input: 26个类别特征
  ↓
Embedding Layer (独立嵌入表)
  ↓
HSTU Block:
  - Pointwise Aggregated Attention
  - Simplified MLP (2层)
  ↓
CLS Token Pooling
  ↓
Prediction
```

**关键组件**:
1. Pointwise Aggregated Attention (替代 softmax)
2. 高效稀疏实现 (可选)
3. 无位置编码

**预期收益**:
- 训练速度提升 2-3x
- AUC 可能持平或略优

---

### Exp13: Mamba4CTR

**目标**: 将 Mamba 应用于 CTR 预测

**架构**:
```
Input: 26个类别特征
  ↓
Embedding Layer
  ↓
Mamba Block:
  - Selective State Space Model
  - Linear Time Complexity
  ↓
Pool + MLP
  ↓
Prediction
```

**关键优势**:
- 线性复杂度
- 参数效率高
- 长序列扩展性好

**预期收益**:
- 训练/推理速度提升
- AUC 需验证

---

### Exp14: Hybrid Architecture

**目标**: 结合 Transformer 和传统 CTR 模型的优势

**架构**:
```
Input: 26个类别特征
  ↓
┌─────────────────┬─────────────────┐
│  Transformer    │   Cross Network │
│  (隐式交互)      │   (显式交互)     │
└────────┬────────┴────────┬────────┘
         ↓                 ↓
         └───── Concat ─────┘
                 ↓
            MLP Head
```

**关键思想**:
- Transformer 捕获高阶隐式交互
- Cross Network 捕获低阶显式交互
- 互补融合

**预期收益**:
- AUC 可能 +1-2bp
- 参数量增加 50%

---

### Exp15: Attention Variants

**目标**: 对比不同注意力机制

**对比项**:
| 注意力类型 | 特点 | 复杂度 |
|-----------|------|--------|
| Standard Self-Attention | 原始 | O(n²) |
| Linear Attention | 线性近似 | O(n) |
| Pointwise Aggregated (HSTU) | 无 softmax | O(n) |
| FlashAttention | IO 优化 | O(n²) |

**实验设计**:
- 同样的模型架构
- 只替换注意力模块
- 对比 AUC 和训练速度

---

## 四、实验优先级

| 优先级 | 实验 | 理由 |
|--------|------|------|
| **P0** | Exp11: V3 无位置编码 | 消融已验证，预期明确 |
| **P1** | Exp12: HSTU-Lite | SOTA 架构，效率提升明显 |
| **P1** | Exp13: Mamba4CTR | 线性复杂度，扩展性好 |
| **P2** | Exp14: Hybrid | 架构创新，风险较高 |
| **P3** | Exp15: Attention Variants | 消融研究，完善理解 |

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
