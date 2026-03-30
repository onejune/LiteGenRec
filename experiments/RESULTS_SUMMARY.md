# LiteGenRec 实验结果总览

**最后更新**: 2026-03-30 23:03

---

## 一、最终结果排名

| 排名 | 模型 | AUC | vs DeepFM | 特征 | 架构特点 |
|------|------|-----|-----------|------|----------|
| 🥇 | **HSTU-Lite V3** | **0.7853** | **+3.81bp** | 39维 | Pointwise Attention + Interaction Attn |
| 🥈 | Mamba4CTR V2 | **0.7846** | **+3.74bp** | 39维 | Bidirectional SSM |
| 🥉 | HSTU-Lite V2 | 0.7845 | +3.73bp | 39维 | Pointwise Attention |
| 4 | Hierarchical Semantic V2 | 0.7841 | +3.69bp | 39维 | VQ-VAE + Attention |
| 5 | Hierarchical Semantic V3 | 0.7826 | +3.54bp | 39维 | VQ-VAE + 2层 Attention |
| 6 | Mamba4CTR V3 | 0.7803 | +3.31bp | 39维 | SSM + Interaction Attn (过拟合) |
| 7 | V2 Transformer | 0.7678 | +2.06bp | 39维 | Transformer + Interaction Attn |
| 8 | SimpleGenCTR V1 | 0.7663 | +1.91bp | 39维 | FFN 序列建模 |
| 9 | DCNv2+AutoInt | 0.7532 | +0.60bp | 26维 | 混合架构 |
| 10 | DeepFM 基线 | 0.7472 | - | 26维 | 传统 CTR 模型 |

---

## 二、V2 vs V3 对比

| 模型 | V2 AUC | V3 AUC | 变化 | Interaction Attention 效果 |
|------|--------|--------|------|---------------------------|
| HSTU-Lite | 0.7845 | **0.7853** | **+0.8bp** | ✅ 略有提升 |
| Mamba4CTR | **0.7846** | 0.7803 | **-4.3bp** | ❌ 显著下降 |
| Hierarchical Semantic | **0.7841** | 0.7826 | **-1.5bp** | ❌ 略有下降 |

**结论**: Interaction Attention 只对 HSTU-Lite 有效，对其他模型反而有害。

---

## 三、稠密特征消融

| 配置 | AUC | 特征数 |
|------|-----|--------|
| 仅稀疏特征 | ~0.75 | 26维 |
| 稠密+稀疏特征 | ~0.78 | 39维 |
| **稠密特征贡献** | **+3bp** | +13维 |

**结论**: 13维稠密特征贡献约 3bp AUC 提升。

---

## 四、位置编码消融

| 配置 | 小配置 AUC | 大配置 AUC |
|------|-----------|-----------|
| 有位置编码 | 0.7438 | 0.7491 |
| 无位置编码 | 0.7454 | 0.7511 |
| **变化** | **+1.6bp** | **+2.0bp** |

**结论**: CTR 特征无序列语义，位置编码是噪声，移除后性能提升。

---

## 五、失败实验

| 实验 | AUC | 问题 |
|------|-----|------|
| VQ-VAE (exp06) | 0.5000 | 层次化量化不适合 CTR |
| Ali-CCP 数据集 | ~0.64 | 时序分割导致分布不一致，过拟合 |

---

## 六、数据集信息

| 项目 | 内容 |
|------|------|
| 名称 | Criteo Display Advertising Challenge |
| 训练样本 | 1,345,295 |
| 验证样本 | 75,071 |
| 特征 | 13 稠密 (数值) + 26 稀疏 (类别) |
| 数据路径 | `/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet` |

---

## 七、统一训练配置

```python
embed_dim = 64
num_heads = 8
num_layers = 4
dropout = 0.1
batch_size = 256
epochs = 2
learning_rate = 1e-3
optimizer = AdamW (weight_decay=0.01)
scheduler = CosineAnnealingLR
```

---

## 八、关键发现

### 1. 新架构优于 Transformer
- HSTU-Lite、Mamba4CTR、Hierarchical Semantic 都超越了 Transformer 基线
- **最佳**: HSTU-Lite V3 (0.7853)，比 Transformer +2.06bp

### 2. Interaction Attention 不是银弹
- 只对 HSTU-Lite 有效 (+0.8bp)
- 对 Mamba4CTR (-4.3bp) 和 Hierarchical Semantic (-1.5bp) 有害

### 3. 稠密特征贡献巨大
- 13维稠密特征贡献 +3bp AUC
- 忽略稠密特征会导致严重性能损失

### 4. 位置编码在 CTR 场景有害
- CTR 特征无序列语义
- 移除位置编码后 AUC +1.6~2.0bp

### 5. VQ-VAE 方案失败
- 层次化语义量化不适合 CTR 预测任务
- AUC 0.5000 (随机水平)

---

## 九、推荐方案

**生产推荐**: **HSTU-Lite V3** (AUC 0.7853)

**理由**:
1. 最高 AUC
2. Pointwise Attention 效率高 (无 softmax)
3. 架构简洁，易于部署

**备选**: Mamba4CTR V2 (AUC 0.7846)
- 线性时间复杂度
- 适合超长序列场景
- 但添加 Interaction Attention 后性能下降

---

## 十、实验文件

```
experiments/
├── data_loader.py                    # 统一数据加载
├── exp03_full_transformer/           # V2 Transformer 基线
├── exp13_hstu_lite/
│   ├── hstu_lite_v2.py              # V2: 无 Interaction Attn
│   └── hstu_lite_v3.py              # V3: 有 Interaction Attn
├── exp14_mamba4ctr/
│   ├── mamba4ctr_v2.py              # V2
│   └── mamba4ctr_v3.py              # V3
└── exp15_tiger_semantic/
    ├── hierarchical_semantic_v2.py  # V2
    └── hierarchical_semantic_v3.py  # V3
```
