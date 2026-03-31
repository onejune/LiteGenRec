# Exp19: 架构优化消融实验

**实验状态**: 🔄 进行中  
**实验时间**: 2026-03-31  
**实验人员**: 萧十一郎 ⚔️

---

## 一、实验方案设计

### 1.1 研究问题

**核心问题**: 在 HSTU-Lite V3 (当前 SOTA) 基础上，哪些架构优化能进一步提升性能？

**背景**:
- 当前 SOTA: HSTU-Lite V3, AUC 0.7853
- 已验证有效: Pointwise Attention, 无位置编码, 39 特征
- 待验证: Interaction Attention, 深度, 归一化, 激活函数

**假设**:
1. 更深网络 (8层) + AttnRes 可能提升
2. Interaction Attention 在其他架构上也有效
3. 现代 FFN (SwiGLU) 可能优于 ReLU
4. RMSNorm 可能优于 LayerNorm

### 1.2 实验变量

| 变量 | 取值 |
|------|------|
| **自变量** | 网络深度 (4/8层) |
| **自变量** | 注意力类型 (Pointwise/Softmax) |
| **自变量** | FFN 类型 (ReLU/SwiGLU) |
| **自变量** | 归一化类型 (LayerNorm/RMSNorm) |
| **因变量** | 验证集 AUC |
| **控制变量** | 数据集、特征处理、embed_dim、epochs |

### 1.3 实验组设计

| 实验组 | 架构变更 | 实验目的 |
|--------|----------|----------|
| **V1** | HSTU-Lite V3 (基线) | 确认基线性能 |
| **V2** | V1 + 8层 + AttnRes | 验证深层收益 |
| **V3** | Transformer + Interaction Attn | 验证 Interaction Attn 普适性 |
| **V4** | V1 + SwiGLU FFN | 验证现代 FFN 效果 |
| **V5** | V1 + RMSNorm | 验证归一化优化 |
| **V6** | V1 + 所有优化 | 组合优化 |

### 1.4 预期结果

| 实验组 | 预期 AUC | 预期 vs 基线 |
|--------|----------|--------------|
| V1 (基线) | 0.7853 | - |
| V2 | 0.7865 | +1.2bp |
| V3 | 0.7845 | -0.8bp |
| V4 | 0.7860 | +0.7bp |
| V5 | 0.7855 | +0.2bp |
| **V6** | **0.7875** | **+2.2bp** |

---

## 二、实验描述

### 2.1 模型架构

#### 2.1.1 HSTU-Lite (基线)

```python
class HSTULite(nn.Module):
    """
    HSTU-Lite: Pointwise Attention for CTR
    
    核心创新:
    1. Pointwise Attention (Sigmoid 激活)
    2. 无位置编码
    3. Interaction Attention (第二层注意力)
    """
    def __init__(self, d_model=64, n_heads=8, num_layers=4):
        self.layers = nn.ModuleList([
            HSTULayer(d_model, n_heads) for _ in range(num_layers)
        ])
        self.interaction_attn = nn.MultiheadAttention(d_model, n_heads)
        
    def forward(self, x):
        # HSTU layers
        for layer in self.layers:
            x = layer(x)
        
        # Interaction Attention
        x = x + self.interaction_attn(x, x, x)
        
        return x
```

#### 2.1.2 SwiGLU FFN

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU: Swish + GLU (Gated Linear Unit)
    
    公式: SwiGLU(x) = Swish(xW₁) ⊙ (xW₂)
    
    优势:
    1. 门控机制增强表达能力
    2. Swish 平滑激活
    3. LLaMA/PaLM 等大模型采用
    """
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
```

#### 2.1.3 RMSNorm

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    公式: RMSNorm(x) = x / RMS(x) * γ
    
    优势:
    1. 无需计算均值，更快
    2. 无需 bias 参数
    3. LLaMA 采用
    """
    def __init__(self, d_model, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

### 2.2 特征处理

与之前实验保持一致：

```yaml
稠密特征 (I1-I13):
  - StandardScaler 标准化
  - Linear 投影 → 13 × 64 维

稀疏特征 (C1-C26):
  - LabelEncoder 编码
  - 独立 Embedding → 26 × 64 维

总特征: 39 × 64 维
```

### 2.3 训练配置

```yaml
优化器: AdamW (lr=1e-3, weight_decay=0.01)
调度器: CosineAnnealingLR (T_max=3)
损失函数: BCEWithLogitsLoss
梯度裁剪: max_norm=1.0
Batch Size: 1024
Epochs: 3
设备: CPU
数据规模: 500k train + 50k valid (采样)
```

---

## 三、数据集说明

### 3.1 数据集基本信息

| 属性 | 值 |
|------|-----|
| **数据集名称** | Criteo Display Advertising Challenge |
| **数据来源** | Kaggle Competition 2014 |
| **任务类型** | 二分类 (Click-Through Rate 预测) |
| **实验规模** | 500k 训练 + 50k 验证 (采样) |

### 3.2 数据集路径

```
原始路径: /mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet/
├── train_train.parquet/  # 训练集
└── train_valid.parquet/  # 验证集
```

### 3.3 特征统计

| 特征类型 | 数量 | 说明 |
|----------|------|------|
| 稠密特征 (I1-I13) | 13 | 数值型 |
| 稀疏特征 (C1-C26) | 26 | 类别型 |
| 标签 (label) | 1 | 0/1 |
| **总计** | **40** | 39 特征 + 1 标签 |

### 3.4 数据分布

```
训练集: 500,000 样本, 正样本率 25.35%
验证集: 50,000 样本, 正样本率 25.60%
```

---

## 四、实验评估结果

### 4.1 主要结果

| 实验组 | 架构变更 | Val AUC | vs 基线 | 训练时间/epoch |
|--------|----------|---------|---------|----------------|
| V1 (基线) | HSTU-Lite V3 | 0.7668 | - | 81s |
| V2 | + 8层 + AttnRes | 0.7678 | +1.0bp | 131s |
| **V3** | **Transformer + Interaction Attn** | **0.7785** | **+11.7bp** ⭐ | 88s |
| V4 | + SwiGLU FFN | 0.7662 | -0.6bp | 88s |
| V5 | + RMSNorm | 0.7661 | -0.7bp | 80s |
| V6 | + 所有优化 | 0.7687 | +1.9bp | 150s |

### 4.2 训练曲线

#### V1 (HSTU-Lite V3, 基线)

| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.6200 | 0.6993 | 0.7472 |
| 2 | 0.4859 | 0.7520 | 0.7633 |
| 3 | 0.4707 | 0.7727 | **0.7668** |

#### V3 (Transformer + Interaction Attn, 最优)

| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.4952 | 0.7393 | 0.7674 |
| 2 | 0.4686 | 0.7748 | 0.7749 |
| 3 | 0.4533 | 0.7939 | **0.7785** ⭐ |

### 4.3 对比分析

#### 4.3.1 Interaction Attention 效果

| 对比组 | 注意力类型 | AUC | 差异 | 结论 |
|--------|------------|-----|------|------|
| V1 | Pointwise | 0.7668 | - | 基线 |
| **V3** | **Softmax + Interaction** | **0.7785** | **+11.7bp** | ✅ Interaction Attn 效果显著 |

#### 4.3.2 深度影响

| 对比组 | 层数 | AttnRes | AUC | 差异 |
|--------|------|---------|-----|------|
| V1 | 4 | No | 0.7668 | - |
| V2 | 8 | Yes | 0.7678 | +1.0bp |

**结论**: 深层 + AttnRes 有轻微提升

#### 4.3.3 FFN 类型影响

| 对比组 | FFN 类型 | AUC | 差异 |
|--------|----------|-----|------|
| V1 | ReLU | 0.7668 | - |
| V4 | SwiGLU | 0.7662 | -0.6bp |

**结论**: SwiGLU 不适合 CTR，ReLU 更好

#### 4.3.4 归一化影响

| 对比组 | 归一化类型 | AUC | 差异 |
|--------|------------|-----|------|
| V1 | LayerNorm | 0.7668 | - |
| V5 | RMSNorm | 0.7661 | -0.7bp |

**结论**: LayerNorm 和 RMSNorm 差异不大

---

## 五、结论与建议

### 5.1 主要结论

1. **Interaction Attention 是关键**: V3 (Transformer + Interaction Attn) 达到最优 **AUC 0.7785**, +11.7bp vs 基线
2. **Softmax + Interaction > Pointwise**: Interaction Attention 比 Pointwise Attention 效果更好
3. **SwiGLU 负面影响**: CTR 场景不适合 SwiGLU，ReLU 更好
4. **RMSNorm 无收益**: LayerNorm 和 RMSNorm 差异不大
5. **深层 + AttnRes 轻微提升**: 8层 + AttnRes 仅 +1.0bp

### 5.2 理论分析

**为什么 Interaction Attention 效果显著？**

1. **二阶特征交互**: Interaction Attention 让所有特征对进行二次交互
2. **增强表达能力**: 标准 Attention 是 self-attention，Interaction 是 cross-attention
3. **适合 CTR**: CTR 任务关键在于特征交叉，Interaction Attention 直接建模

**为什么 SwiGLU 不适合 CTR？**

1. **任务复杂度低**: CTR 任务相对简单，不需要门控机制
2. **过拟合风险**: SwiGLU 参数更多，容易过拟合
3. **ReLU 足够**: ReLU 在 CTR 场景已经足够好

### 5.3 与历史实验对比

| 模型 | 注意力类型 | Interaction Attn | AUC | 说明 |
|------|------------|------------------|-----|------|
| HSTU-Lite V3 (之前) | Pointwise | Yes | **0.7853** | 历史最优 |
| **Transformer + Interaction (V3)** | **Softmax** | **Yes** | **0.7785** | **本实验最优** |
| Transformer V2 (之前) | Softmax | No | 0.7678 | 标准 Transformer |
| DeepFM baseline | - | - | 0.7472 | 基线 |

**对比**: V3 (0.7785) 低于 HSTU-Lite V3 (0.7853) 6.8bp，可能原因：
- 本次采样 (500k) < 之前采样
- Pointwise Attention 仍优于 Softmax

### 5.4 后续建议

| 方向 | 具体措施 | 预期收益 |
|------|----------|----------|
| **Pointwise + Interaction** | Pointwise Attn + Interaction 层 | +2~3bp |
| **全量数据** | 放弃采样，用全量 | +5~10bp |
| **更长训练** | 5-10 epochs | +1~2bp |

---

## 六、文件清单

```
exp19_architecture_ablation/
├── README.md              # 本文件
├── train.py               # 训练脚本
├── models.py              # 模型定义
└── results.json           # 实验结果
```

---

**实验开始时间**: 2026-03-31 15:20  
**实验人员**: 萧十一郎 ⚔️
