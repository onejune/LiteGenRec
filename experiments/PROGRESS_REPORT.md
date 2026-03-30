# LiteGenRec 实验进展报告

**最后更新**: 2026-03-30 19:14

---

## 实验结果汇总

### V2 实验 (39维特征: 13稠密 + 26稀疏)

| 实验 | 架构 | AUC | vs V2 Transformer |
|------|------|-----|-------------------|
| **Mamba4CTR V2** | State Space Model | **0.7846** | **+1.68bp** 🥇 |
| **HSTU-Lite V2** | Pointwise Attention | **0.7845** | **+1.67bp** 🥈 |
| **Hierarchical Semantic V2** | VQ-VAE + Attention | **0.7841** | **+1.63bp** 🥉 |
| V2 Transformer (exp03) | Transformer + Interaction Attn | 0.7678 | 基线 |
| V1 SimpleGenCTR | FFN | 0.7663 | - |
| DeepFM Baseline | DeepFM | 0.7472 | - |

### V3 实验 (添加 Interaction Attention) - 进行中

| 实验 | 状态 | 预期 |
|------|------|------|
| HSTU-Lite V3 | 运行中 | 验证 Interaction Attention 效果 |
| Mamba4CTR V3 | 运行中 | 验证 Interaction Attention 效果 |
| Hierarchical Semantic V3 | 运行中 | 验证 Interaction Attention 效果 |

---

## 关键发现

### 1. 稠密特征贡献巨大
- V1 实验只用了 26 维稀疏特征，AUC ~0.75
- V2 添加 13 维稠密特征后，AUC 提升到 ~0.78
- **稠密特征贡献约 3bp AUC**

### 2. V2 Transformer 的关键优化
V2 (exp03_full_transformer) 有两层特征交互：
```
输入 → 嵌入层 → Transformer 编码器 (4层) → Interaction Attention → MLP
                              ↑                    ↑
                         第一次交互            第二次交互 ⭐
```

其他版本只有一次特征交互，V2 多了一层 **Interaction Attention**。

### 3. 位置编码在 CTR 场景可能有害
- exp10 消融实验：移除位置编码后 AUC +1.6bp (小配置)
- exp12 消融实验：大配置同样结论
- **CTR 特征无序列语义，位置编码是噪声**

### 4. VQ-VAE 方案失败
- exp06 VQ-VAE: AUC 0.5000 (随机水平)
- 层次化量化不适合 CTR 预测

---

## 数据集

**Criteo** (标准 CTR 数据集)
- 训练集: 1,345,295 样本
- 验证集: 75,071 样本
- 特征: 13 稠密 (数值) + 26 稀疏 (类别)
- 数据路径: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet`

---

## 统一配置

V2/V3 实验统一配置：
```python
embed_dim = 64
num_heads = 8
num_layers = 4
dropout = 0.1
batch_size = 256
epochs = 2
learning_rate = 1e-3
```

---

## 文件结构

```
experiments/
├── data_loader.py              # 统一数据加载器
├── PROGRESS_REPORT.md          # 本文件
├── exp03_full_transformer/     # V2 Transformer 基线
├── exp13_hstu_lite/
│   ├── hstu_lite_v2.py         # V2: 无 Interaction Attention
│   └── hstu_lite_v3.py         # V3: 有 Interaction Attention
├── exp14_mamba4ctr/
│   ├── mamba4ctr_v2.py         # V2
│   └── mamba4ctr_v3.py         # V3
└── exp15_tiger_semantic/
    ├── hierarchical_semantic_v2.py  # V2
    └── hierarchical_semantic_v3.py  # V3
```

---

## 下一步

1. 等待 V3 实验完成
2. 对比 V2/V3 结果，验证 Interaction Attention 效果
3. 如果有效，确定最佳架构组合

---

## 已废弃的实验

- **exp06 VQ-VAE**: 层次化语义 ID，AUC 0.5000 (失败)
- **Ali-CCP 数据集**: 时序分割导致过拟合，AUC ~0.64 (已切换到 Criteo)
