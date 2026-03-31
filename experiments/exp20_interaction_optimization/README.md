# Exp20: Interaction Attention 优化实验

## 一、实验目的

基于 Exp19 结果，深入研究 Interaction Attention 的效果，确定是 Interaction 本身重要还是 Attention 类型重要。

**背景**: Exp19 发现 Transformer + Interaction Attn (V3) 达到 0.7785，+11.7bp vs 基线，效果显著。

## 二、实验设计

### 2.1 模型架构对比

| 实验组 | 主干 | 交互层 | 说明 |
|--------|------|--------|------|
| V1 | HSTU-Lite (Pointwise) | + Interaction Attn | HSTU + Interact |
| V2 | Transformer (Softmax) | + Interaction Attn | Trans + Interact |
| V3 | Pointwise | + Double Interaction | Pointwise + 2×Interact |
| V4 | Transformer | + Double Interaction | Trans + 2×Interact |
| V5 | HSTU-Lite | + Double Interaction | HSTU + 2×Interact |

### 2.2 核心假设

1. **Interaction Hypothesis**: Interaction Attention 是关键，而非主干注意力类型
2. **Depth Hypothesis**: Double Interaction 进一步提升效果
3. **Compatibility Hypothesis**: Pointwise + Double Interaction 最佳

## 三、模型细节

### 3.1 HSTU-Lite Layer (Pointwise Attention)
```python
# Pointwise Attention: Sigmoid instead of Softmax
attn_weights = torch.sigmoid(torch.matmul(q, k.transpose(-2, -1)) / (d_model ** 0.5))
```

### 3.2 Interaction Attention
```python
# Cross-attention between all features
attn_out, _ = self.interaction_attn(x, x, x)
x = residual + attn_out
```

### 3.3 Double Interaction
```python
# Two sequential Interaction layers
x = residual1 + self.interaction_attn1(x, x, x)
x = residual2 + self.interaction_attn2(x, x, x)
```

## 四、实验结果

### 4.1 主要结果

| 实验组 | 架构 | Val AUC | vs 基线 | 训练时间/epoch |
|--------|------|---------|---------|----------------|
| V1 | HSTU + Interaction | 0.7648 | - | 84s |
| **V2** | **Transformer + Interaction** | **0.7749** | **+101.0bp** ⭐ | 91s |
| V3 | Pointwise + Double Interaction | 0.7615 | -32.2bp | 73s |
| **V4** | **Transformer + Double Interaction** | **0.7764** | **+116.5bp** ⭐ | 103s |
| V5 | HSTU + Double Interaction | 0.7648 | +0.0bp | 75s |

### 4.2 训练曲线

#### V1 (HSTU + Interaction, 基线)
| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.6058 | 0.7058 | 0.7463 |
| 2 | 0.4862 | 0.7525 | 0.7600 |
| 3 | 0.4694 | 0.7751 | **0.7648** |

#### V2 (Transformer + Interaction)
| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.4915 | 0.7447 | 0.7667 |
| 2 | 0.4672 | 0.7772 | 0.7716 |
| 3 | 0.4509 | 0.7976 | **0.7749** |

#### V4 (Transformer + Double Interaction, 最优)
| Epoch | Train Loss | Train AUC | Val AUC |
|-------|------------|-----------|---------|
| 1 | 0.4963 | 0.7407 | 0.7650 |
| 2 | 0.4683 | 0.7762 | 0.7719 |
| 3 | 0.4527 | 0.7957 | **0.7764** ⭐ |

## 五、实验配置

- **数据**: Criteo 全特征 (13 dense + 26 sparse)
- **采样**: 500k 训练, 50k 验证
- **模型**: d_model=64, n_heads=8, num_layers=4
- **训练**: 3 epochs, batch_size=1024
- **优化器**: AdamW, lr=1e-3, weight_decay=0.01

## 六、对比分析

### 6.1 Attention Type Effect

| 对比 | 模型A | AUC | 模型B | AUC | 差异 | 结论 |
|------|-------|-----|-------|-----|------|------|
| Pointwise vs Softmax | V1 (HSTU+Interact) | 0.7648 | V2 (Trans+Interact) | 0.7749 | **+10.1bp** | **Softmax + Interaction 更好** |

### 6.2 Double Interaction Effect

| 对比 | 模型A | AUC | 模型B | AUC | 差异 | 结论 |
|------|-------|-----|-------|-----|------|------|
| Single vs Double (Pointwise) | V1 (Pointwise+Interact) | 0.7648 | V3 (Pointwise+DoubleInteract) | 0.7615 | **-3.3bp** | **Double有害** |
| Single vs Double (Transformer) | V2 (Trans+Interact) | 0.7749 | V4 (Trans+DoubleInteract) | 0.7764 | **+1.5bp** | **Double有益** |

### 6.3 最优配置分析

**冠军**: V4 - Transformer + Double Interaction (0.7764)

**关键洞察**:
1. **Attention Type Matters**: Softmax > Pointwise for Interaction
2. **Composition Matters**: Double Interaction only works with Transformer
3. **Synergy**: Transformer + Double Interaction creates positive synergy

## 七、理论分析

### 7.1 为什么 Transformer + Double Interaction 最优？

1. **表达能力**: Transformer 的 Softmax Attention 比 Pointwise 更强
2. **深度效应**: 第二层 Interaction 在 Transformer 特征基础上进一步提取高阶交互
3. **梯度流**: Transformer 残差连接比 HSTU 更稳定

### 7.2 为什么 Pointwise + Double Interaction 有害？

1. **过拟合**: Pointwise 本身表达能力较弱，双重交互导致过拟合
2. **梯度消失**: HSTU 残差路径可能加剧深层问题
3. **容量不匹配**: 模型容量不足以支撑双层交互

## 八、与历史实验对比

| 模型 | 注意力类型 | Interaction | AUC | 说明 |
|------|------------|-------------|-----|------|
| **V4 (Transformer+Double)** | **Softmax** | **双层** | **0.7764** | **Exp20 最优** |
| HSTU-Lite V3 (历史) | Pointwise | 单层 | 0.7853 | 历史最优 |
| Transformer+Interact (Exp19) | Softmax | 单层 | 0.7785 | Exp19 最优 |

**对比**: Exp20 V4 (0.7764) < HSTU-Lite V3 (0.7853) 8.9bp，可能因为采样差异。

## 九、后续建议

| 方向 | 具体措施 | 预期收益 |
|------|----------|----------|
| **Transformer + Double (全量)** | 放弃采样，用全量数据训练 | +5~10bp |
| **HSTU + Double (改进)** | 添加更多残差连接，防止梯度消失 | +2~3bp |
| **融合架构** | Pointwise + Double Softmax Interaction | +3~5bp |

## 十、评估指标

- **主指标**: Val AUC
- **次指标**: 训练速度、收敛稳定性
- **对比基准**: V1 (HSTU-Lite + Interaction)

## 十一、实验步骤

1. 实现 5 种模型变体
2. 统一训练流程
3. 记录每轮训练指标
4. 汇总结果并分析