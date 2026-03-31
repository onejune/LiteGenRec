# Transformer 位置编码消融实验

> **实验目的**: 验证位置编码对 CTR 预估的影响  
> **实验日期**: 2026-03-31  
> **状态**: 🔄 进行中

---

## 📋 实验背景

### 研究问题

CTR 特征是无序集合，位置编码是否有必要？

### 实验假设

**位置编码有害**: 强加位置语义会引入噪声

---

## 🎯 实验设计

### 对比模型

| 模型 | 位置编码 | 注意力机制 |
|------|----------|------------|
| Transformer+PE (exp03) | ✅ 有 | Multi-Head Attn |
| Transformer-NoPE (exp16) | ❌ 无 | Multi-Head Attn |
| HSTU-Lite V3 (exp13) | ❌ 无 | Pointwise Attn |

### 统一配置

```yaml
特征: 完整 (39 维 = 26 稀疏 + 13 稠密)
embed_dim: 64
num_heads: 8
num_layers: 4
hidden_dim: 256
dropout: 0.1
```

---

## 📊 数据集

### Criteo Standard

| 数据集 | 样本数 | 正样本率 |
|--------|--------|----------|
| 训练集 | 41,253,427 | 25.62% |
| 验证集 | 3,438,383 | 25.68% |
| **总计** | **44,691,810** | 25.62% |

### 数据路径

```
/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet/
```

### 特征处理

- **稀疏特征 (C1-C26)**: LabelEncoder + 频率过滤 (<5 归入 -1)
- **稠密特征 (I1-I13)**: KBinsDiscretizer (quantile, 50 bins)

---

## 📁 目录结构

```
transformer_position_encoding_ablation/
├── README.md              # 本文件 - 实验说明
├── gpu_router.py          # GPU 智能路由模块
├── models.py              # 模型定义 + 训练脚本
├── run_experiment.sh      # 实验启动脚本
├── RESULTS.md             # 实验结果汇总
├── analysis.ipynb         # 结果分析 notebook (可选)
└── logs/                  # 训练日志
    ├── quick_run.log      # 快速验证 (采样)
    └── full_run.log       # 全量实验
```

---

## 🚀 快速开始

### 1. 快速验证 (采样数据)

```bash
# 前 5 个分片 + 1 epoch，约 15 分钟
bash run_experiment.sh --max-files 5 --epochs 1
```

### 2. 全量实验

```bash
# 全量数据 + 3 epochs，约 2 小时
bash run_experiment.sh --epochs 3
```

### 3. 指定 GPU

```bash
# 使用 GPU 0
CUDA_VISIBLE_DEVICES=0 python models.py --epochs 3

# 自动选择（推荐）
python models.py --epochs 3 --complexity high
```

---

## 📈 实验结果

### 快速验证 (5 分片 × 1 epoch)

| 模型 | Val AUC | vs DeepFM | 排名 |
|------|---------|-----------|------|
| **Transformer-NoPE (exp16)** | **0.7877** | **+404.6bp** | 🥇 |
| Transformer+PE (exp03) | 0.7863 | +391.2bp | 🥈 |
| HSTU-Lite V3 (exp13) | 0.7767 | +295.3bp | 🥉 |
| DeepFM baseline | 0.7472 | - | - |

**结论**: 无位置编码优于有位置编码 (+14bp)

### 全量实验 (进行中)

待更新...

---

## 🔍 核心发现

### 1. 位置编码有害 (-14bp)

```
Transformer+PE:   0.7863
Transformer-NoPE: 0.7877
差异: +14bp (无 PE 更好)
```

**原因**: CTR 特征是无序集合，位置编码引入虚假的顺序语义

### 2. Transformer > HSTU-Lite (+110bp)

```
Transformer-NoPE: 0.7877
HSTU-Lite V3:     0.7767
差异: +110bp
```

**原因**: Multi-Head Attention 表达能力更强

---

## 📝 待办事项

- [ ] 完成全量数据实验
- [ ] 更新 RESULTS.md
- [ ] 分析训练曲线
- [ ] 提交 git

---

## 📚 参考文献

1. Vaswani et al. "Attention Is All You Need" (2017) - Transformer 原论文
2. Criteo Kaggle Competition 2014 - 数据集来源

---

*实验维护：牛顿 🍎*
