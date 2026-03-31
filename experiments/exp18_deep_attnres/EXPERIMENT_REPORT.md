# Exp18 实验报告

**实验名称**: 深层网络 + Attention Residuals  
**实验时间**: 2026-03-31  
**实验目标**: 验证 AttnRes 在深层 CTR 模型上的效果

---

## 一、实验设计

### 1.1 数据配置

```yaml
数据集: Criteo
训练集: 500,000 样本 (采样)
验证集: 50,000 样本 (采样)
特征: 39 维 (13 稠密 + 26 稀疏)
处理方式: StandardScaler + LabelEncoder (与之前实验一致)
```

### 1.2 模型配置

```yaml
模型架构: Transformer Encoder
embed_dim: 64
num_heads: 8
hidden_dim: 256
dropout: 0.1
位置编码: ❌ 无
```

### 1.3 实验组

| 变体 | 层数 | AttnRes | 说明 |
|------|------|---------|------|
| V1 | 4 | No | 基线 |
| V2 | 8 | No | 验证深层是否稀释 |
| V3 | 8 | Yes | 验证 AttnRes 效果 |
| V4 | 12 | Yes | 更深 + AttnRes |
| V5 | 16 | Yes | 最深 + AttnRes |

---

## 二、实验结果

### 2.1 最终结果

| 实验 | 层数 | AttnRes | Val AUC | vs 基线 | 训练时间/epoch |
|------|------|---------|---------|---------|----------------|
| V1 | 4 | No | 0.7799 | - | 89s |
| V2 | 8 | No | 0.7800 | +0.1bp | 128s |
| **V3** | **8** | **Yes** | **0.7805** | **+0.6bp** | 147s |
| **V4** | **12** | **Yes** | **0.7809** | **+1.0bp** | ⭐ 191s |
| V5 | 16 | Yes | 0.7770 | -2.9bp | 233s |

### 2.2 训练曲线

**V1 (4层, 基线)**:
```
Epoch 1: Train AUC 0.7455, Val AUC 0.7689
Epoch 2: Train AUC 0.7763, Val AUC 0.7773
Epoch 3: Train AUC 0.7968, Val AUC 0.7799
```

**V3 (8层 + AttnRes)**:
```
Epoch 1: Train AUC 0.7450, Val AUC 0.7691
Epoch 2: Train AUC 0.7773, Val AUC 0.7759
Epoch 3: Train AUC 0.7976, Val AUC 0.7805
```

**V4 (12层 + AttnRes, 最优)**:
```
Epoch 1: Train AUC 0.7408, Val AUC 0.7674
Epoch 2: Train AUC 0.7745, Val AUC 0.7775
Epoch 3: Train AUC 0.7939, Val AUC 0.7809
```

**V5 (16层 + AttnRes)**:
```
Epoch 1: Train AUC 0.7390, Val AUC 0.7669
Epoch 2: Train AUC 0.7729, Val AUC 0.7745
Epoch 3: Train AUC 0.7925, Val AUC 0.7770
```

---

## 三、关键发现

### 3.1 AttnRes 确实有效

| 对比 | 层数 | AttnRes | AUC | 差异 |
|------|------|---------|-----|------|
| V2 vs V3 | 8 | No vs Yes | 0.7800 vs 0.7805 | **+0.5bp** |

**结论**: AttnRes 在 8 层网络上有 +0.5bp 提升

### 3.2 最优深度是 12 层

| 层数 | AttnRes | AUC |
|------|---------|-----|
| 4 | No | 0.7799 |
| 8 | Yes | 0.7805 |
| **12** | **Yes** | **0.7809** ⭐ |
| 16 | Yes | 0.7770 |

**结论**: 12 层 + AttnRes 达到最优，16 层过深

### 3.3 PreNorm Dilution 在 CTR 场景不明显

- V2 (8层无AttnRes) 与 V1 (4层) 持平
- 说明 CTR 模型深度不足以产生严重稀释
- AttnRes 收益来自更好的特征融合，非解决稀释

### 3.4 16 层对 CTR 过深

- V5 性能下降 -2.9bp
- 即使有 AttnRes 也无法挽救
- CTR 任务相对简单，不需要太深网络

---

## 四、与之前实验对比

| 模型 | 层数 | AttnRes | AUC | 说明 |
|------|------|---------|-----|------|
| HSTU-Lite V3 | 4 | No | 0.7853 | 之前最优 (Pointwise Attn) |
| Transformer V2 | 4 | No | 0.7678 | 标准 Transformer |
| **Transformer + AttnRes (V4)** | **12** | **Yes** | **0.7809** | **本实验最优** |

**对比**:
- V4 (0.7809) 仍低于 HSTU-Lite V3 (0.7853)
- 但高于标准 Transformer (0.7678)
- Pointwise Attention 优于 Softmax Attention

---

## 五、结论

1. **AttnRes 有效**: 在 8 层网络提升 +0.5bp
2. **最优配置**: 12 层 + AttnRes, AUC 0.7809
3. **CTR 场景深度上限**: 12-16 层，再深会下降
4. **仍不如 HSTU-Lite**: Pointwise Attention 更适合 CTR

---

## 六、后续建议

1. **结合 HSTU-Lite**: 在 Pointwise Attention 基础上加 AttnRes
2. **更长训练**: 当前仅 3 epochs，可尝试 5-10 epochs
3. **全量数据**: 当前仅 500k 样本，可用全量验证

---

**实验完成时间**: 2026-03-31 11:45  
**实验人员**: 萧十一郎 ⚔️
