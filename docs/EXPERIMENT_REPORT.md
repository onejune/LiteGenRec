# LiteGenRec 实验报告

**项目**: 轻量级生成式推荐技术研究
**时间**: 2026-03-28 ~ 2026-03-31
**作者**: 萧十一郎

---

## 一、项目背景

### 1.1 研究目标

探索轻量级生成式推荐技术在程序化广告 DSP 场景的落地可行性。

### 1.2 核心问题

- **判别式 vs 生成式**: CTR 预测能否用生成式方法建模？
- **架构选择**: Transformer / HSTU / Mamba / VQ-VAE 哪个更适合？
- **轻量化**: 如何在 < 100M 参数下达到 SOTA？

### 1.3 数据集

**Criteo Display Advertising Challenge**
- 训练样本: 1,345,295
- 验证样本: 75,071
- 特征: 39 维 (13 稠密 + 26 稀疏)
- 数据路径: `/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet`

---

## 二、实验设计

### 2.1 特征处理

```python
# 稠密特征
- 标准化 (StandardScaler)
- 线性投影到 13×64 维

# 稀疏特征
- LabelEncoder + 1 (padding_idx=0)
- max_vocab_size = 50000
- 独立嵌入表 (每字段一个)
```

### 2.2 模型架构对比

| 架构 | 核心思想 | 参数量 |
|------|----------|--------|
| Transformer | 多头自注意力 | ~47M |
| HSTU-Lite | Pointwise Attention (无 softmax) | ~47M |
| Mamba4CTR | State Space Model | ~47M |
| VQ-VAE | 向量量化语义 ID | ~50M |
| GPT-Style | 自回归生成 | ~47M |

### 2.3 训练配置

```yaml
embed_dim: 64
num_heads: 8
num_layers: 4
batch_size: 256
epochs: 2-3
learning_rate: 1e-3
optimizer: AdamW
scheduler: CosineAnnealingLR
```

---

## 三、实验结果

### 3.1 最终排名

| 排名 | 模型 | AUC | vs DeepFM | 特征数 | 架构类型 |
|:----:|------|:---:|:---------:|:------:|:--------:|
| 🥇 | **HSTU-Lite V3** | **0.7853** | **+3.81bp** | 39 | 判别式 |
| 🥈 | Mamba4CTR V2 | 0.7846 | +3.74bp | 39 | 判别式 |
| 🥉 | HSTU-Lite V2 | 0.7845 | +3.73bp | 39 | 判别式 |
| 4 | Hierarchical Semantic V2 | 0.7841 | +3.69bp | 39 | 判别式 |
| 5 | Hierarchical Semantic V3 | 0.7826 | +3.54bp | 39 | 判别式 |
| 6 | Mamba4CTR V3 | 0.7803 | +3.31bp | 39 | 判别式 |
| 7 | Transformer V2 | 0.7678 | +2.06bp | 39 | 判别式 |
| 8 | SimpleGenCTR V1 | 0.7663 | +1.91bp | 39 | 判别式 |
| 9 | 掩码特征预测 | 0.7571 | +0.99bp | 39 | 生成式 |
| 10 | DeepFM 基线 | 0.7472 | - | 26 | 判别式 |
| 11 | 自回归生成 | 0.7235 | -2.37bp | 39 | 生成式 ❌ |

### 3.2 关键发现

#### 发现 1: 稠密特征贡献显著

| 配置 | 特征 | AUC | 差异 |
|------|------|-----|------|
| 仅稀疏 | 26 | 0.7492 | 基线 |
| 稠密+稀疏 | 39 | 0.7678 | **+1.86bp** |

**结论**: 稠密特征贡献 +1.86bp，必须包含。

#### 发现 2: 位置编码有害

| 配置 | AUC | 差异 |
|------|-----|------|
| 有位置编码 | 0.7491 | 基线 |
| 无位置编码 | 0.7511 | **+2.0bp** |

**结论**: CTR 特征无序列语义，位置编码是噪声。

#### 发现 3: Interaction Attention 效果不一

| 模型 | V2 (无) | V3 (有) | 差异 |
|------|---------|---------|------|
| HSTU-Lite | 0.7845 | **0.7853** | +0.8bp ✅ |
| Mamba4CTR | **0.7846** | 0.7803 | -4.3bp ❌ |
| Hierarchical | **0.7841** | 0.7826 | -1.5bp ❌ |

**结论**: Interaction Attention 只对 HSTU-Lite 有效。

#### 发现 4: 生成式建模不适合 CTR

| 方法 | AUC | vs DeepFM |
|------|-----|-----------|
| 自回归生成 | 0.7235 | -2.37bp ❌ |
| 掩码特征预测 | 0.7571 | +0.99bp ⚠️ |
| 判别式 (HSTU-Lite V3) | 0.7853 | +3.81bp ✅ |

**原因**:
1. Criteo 无真实用户序列
2. 特征重建不如特征交互
3. CTR 预测本质是判别式任务

---

## 四、架构分析

### 4.1 HSTU-Lite V3 (最佳模型)

```
输入: [batch, 39] (13 dense + 26 sparse)
  ↓
嵌入层
  ├── Dense: Linear(13, 13×64)
  └── Sparse: 26 × Embedding(vocab, 64)
  ↓
拼接: [batch, 39, 64]
  ↓
Position Encoding: ❌ 不使用
  ↓
Transformer Encoder (4层)
  ├── Multi-Head Attention (8 heads)
  ├── Add & Norm
  ├── FFN (256)
  └── Add & Norm
  ↓
Interaction Attention (第二层注意力)
  ↓
MLP Head
  ├── Linear(64, 128)
  ├── ReLU
  ├── Dropout(0.1)
  └── Linear(128, 1)
  ↓
输出: CTR 概率
```

**参数量**: ~47M
**推理延迟**: < 5ms (CPU)

### 4.2 Mamba4CTR V2

```
输入 → 嵌入 → Mamba Blocks (SSM) → MLP → 输出
```

**优势**: O(n) 复杂度，适合长序列
**劣势**: Interaction Attention 反而降低效果

### 4.3 自回归生成 (失败)

```
输入: 历史 N 个样本的特征序列
  ↓
GPT-style Decoder (Causal Attention)
  ↓
预测: 下一个样本的特征值
  ↓
推理: 生成特征 → 检索物品 → CTR
```

**问题**:
- 无真实序列，随机采样作为历史
- Causal Attention 无法学习有效模式
- 特征预测目标与 CTR 目标不一致

---

## 五、消融实验

### 5.1 位置编码消融

| 配置 | 小模型 (32/4/2) | 大模型 (64/8/4) |
|------|-----------------|-----------------|
| 有位置编码 | 0.7438 | 0.7491 |
| 无位置编码 | **0.7454** | **0.7511** |
| 差异 | +1.6bp | +2.0bp |

**结论**: 大小模型一致，位置编码有害。

### 5.2 Interaction Attention 消融

| 模型 | 无 IA (V2) | 有 IA (V3) | 差异 |
|------|------------|------------|------|
| HSTU-Lite | 0.7845 | **0.7853** | +0.8bp |
| Mamba4CTR | **0.7846** | 0.7803 | -4.3bp |
| Hierarchical | **0.7841** | 0.7826 | -1.5bp |

**结论**: 只有 HSTU-Lite 受益于 Interaction Attention。

### 5.3 特征数消融

| 特征数 | AUC | 差异 |
|--------|-----|------|
| 26 (仅稀疏) | 0.7492 | 基线 |
| 39 (稠密+稀疏) | **0.7678** | +1.86bp |

**结论**: 稠密特征贡献 +1.86bp。

---

## 六、关键结论

### 6.1 架构选择

| 场景 | 推荐架构 | AUC |
|------|----------|-----|
| **生产部署** | HSTU-Lite V3 | 0.7853 |
| 长序列场景 | Mamba4CTR V2 | 0.7846 |
| 快速迭代 | Transformer V2 | 0.7678 |

### 6.2 设计原则

1. **必须包含稠密特征** (+1.86bp)
2. **不使用位置编码** (+2.0bp)
3. **优先判别式架构** (vs 生成式 +2.82bp)
4. **Interaction Attention 需验证** (效果不一)

### 6.3 放弃方向

- ❌ VQ-VAE 语义 ID (AUC 0.5000)
- ❌ 自回归生成 (AUC 0.7235)
- ❌ RoPE 位置编码 (CTR 无序列语义)

---

## 七、后续方向

### 7.1 短期优化 (预期 +0.5~1bp)

| 方向 | 预期收益 | 难度 |
|------|----------|------|
| 更长训练 (5 epochs) | +0.5bp | 低 |
| 全量数据 (45M 样本) | +1~2bp | 低 |
| 学习率调优 | +0.3bp | 低 |

### 7.2 中期优化 (预期 +1~2bp)

| 方向 | 预期收益 | 难度 |
|------|----------|------|
| 显式特征交叉 (FM/DCN) | +1~2bp | 中 |
| 稠密特征离散化 | +0.5bp | 低 |
| 多任务学习 (CTR+CVR) | +1~3bp | 高 |

### 7.3 长期方向

| 方向 | 描述 |
|------|------|
| 真实序列数据 | 切换到 KuaiRec 等有序列的数据集 |
| 模型压缩 | 边缘端部署 (< 10MB, < 50ms) |
| 少样本学习 | 新物品/新用户冷启动 |

---

## 八、代码与文件

### 8.1 目录结构

```
LiteGenRec/
├── experiments/
│   ├── exp03_full_transformer/      # Transformer V2
│   ├── exp13_hstu_lite/             # HSTU-Lite V2/V3
│   ├── exp14_mamba4ctr/             # Mamba4CTR V2/V3
│   ├── exp15_tiger_semantic/        # Hierarchical Semantic V2/V3
│   ├── exp16_autoregressive/        # 生成式实验
│   │   ├── gpt_generator.py         # 自回归
│   │   └── masked_model.py          # 掩码预测
│   ├── data_loader.py               # 统一数据加载
│   ├── RESULTS_SUMMARY.md           # 结果汇总
│   └── PROGRESS_REPORT.md           # 进展报告
├── docs/
│   ├── EXPERIMENT_DESIGN.md         # 实验设计
│   ├── AUTOREGRESSIVE_SUMMARY.md    # 生成式总结
│   └── EXPERIMENT_REPORT.md         # 本报告
└── README.md
```

### 8.2 Git 提交

```
a154f3e - 📝 自回归生成实验总结文档
0b9229a - 📊 Exp16 实验结果
2bdfb84 - 📊 更新实验结果汇总
```

---

## 九、致谢

感谢老板的信任和支持，本实验验证了判别式架构在 CTR 预测任务上的优势，为后续轻量级推荐系统的落地提供了技术基础。

---

**报告完成时间**: 2026-03-31 06:45
**报告人**: 萧十一郎 ⚔️
