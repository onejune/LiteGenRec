# LiteGenRec 实验结果总览

**最后更新**: 2026-03-31 00:30

---

## 一、最终结果排名

| 排名 | 模型 | AUC | vs DeepFM | 特征 | 架构 |
|------|------|-----|-----------|------|------|
| 🥇 | **HSTU-Lite V3** | **0.7853** | **+3.81bp** | 39维 | 判别式 + Pointwise Attn |
| 🥈 | Mamba4CTR V2 | **0.7846** | **+3.74bp** | 39维 | 判别式 + SSM |
| 🥉 | HSTU-Lite V2 | 0.7845 | +3.73bp | 39维 | 判别式 + Pointwise Attn |
| 4 | Hierarchical Semantic V2 | 0.7841 | +3.69bp | 39维 | 判别式 + VQ-VAE |
| 5 | Hierarchical Semantic V3 | 0.7826 | +3.54bp | 39维 | 判别式 |
| 6 | Mamba4CTR V3 | 0.7803 | +3.31bp | 39维 | 判别式 |
| 7 | V2 Transformer | 0.7678 | +2.06bp | 39维 | 判别式 |
| 8 | SimpleGenCTR V1 | 0.7663 | +1.91bp | 39维 | 判别式 |
| 9 | **掩码特征预测** | 0.7571 | +0.99bp | 39维 | 生成式预训练 |
| 10 | DCNv2+AutoInt | 0.7532 | +0.60bp | 26维 | 判别式 |
| 11 | DeepFM 基线 | 0.7472 | - | 26维 | 判别式 |
| 12 | **自回归生成** | 0.7235 | -2.37bp | 39维 | 生成式 ❌ |

---

## 二、生成式实验结论

### 自回归生成 (AUC 0.7235)
- **问题**: Criteo 无真实用户序列，伪序列无法学习有效模式
- **结论**: CTR 预测本质是判别式任务，自回归生成不适合

### 掩码特征预测 (AUC 0.7571)
- **优势**: 优于 DeepFM +1bp，特征间依赖建模有效
- **劣势**: 不如直接特征交互 (HSTU-Lite V3: 0.7853)
- **结论**: 生成式预训练有正收益，但判别式架构更优

### 核心发现
**CTR 预测不适合生成式建模**，原因：
1. 无真实序列 (用户行为)
2. 特征重建不如特征交互
3. 判别式任务用判别式模型更直接

---

## 三、判别式优化方向

### 已验证有效
- ✅ Pointwise Attention (HSTU-Lite)
- ✅ State Space Model (Mamba4CTR)
- ✅ 稠密特征 (+3bp)
- ✅ 移除位置编码 (+1.6~2bp)

### 待探索
- 🔄 显式特征交叉 (FM/DCN)
- 🔄 多任务学习 (CTR+CVR)
- 🔄 序列行为建模 (需真实序列数据)

---

## 四、推荐方案

**生产推荐**: **HSTU-Lite V3** (AUC 0.7853)

**理由**:
1. 最高 AUC
2. Pointwise Attention 效率高 (无 softmax)
3. 架构简洁，易于部署

**放弃方向**: 生成式建模 (自回归/掩码预训练)

---

## 五、文件结构

```
experiments/
├── exp03_full_transformer/      # V2 Transformer 基线
├── exp13_hstu_lite/             # HSTU-Lite V2/V3
├── exp14_mamba4ctr/             # Mamba4CTR V2/V3
├── exp15_tiger_semantic/        # Hierarchical Semantic V2/V3
├── exp16_autoregressive/        # 自回归生成实验
│   ├── gpt_generator.py         # V1: 自回归
│   └── masked_model.py          # V2: 掩码预测
├── data_loader.py               # 统一数据加载
├── RESULTS_SUMMARY.md           # 结果汇总
└── PROGRESS_REPORT.md           # 进展报告
```

---

## 六、后续方向

1. **特征工程**: 显式特征交叉
2. **多任务学习**: CTR + CVR 联合训练
3. **模型压缩**: 边缘端部署优化
4. **真实序列数据**: 切换到有序列的数据集 (如 KuaiRec)
