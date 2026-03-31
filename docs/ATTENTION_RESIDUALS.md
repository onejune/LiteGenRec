# Attention Residuals (AttnRes) - Kimi 2026年3月

**论文**: [Attention Residuals](https://arxiv.org/abs/2603.15031)
**团队**: 月之暗面 (Kimi)
**时间**: 2026年3月
**核心贡献**: 用深度方向的 Softmax 注意力替代固定等权残差连接

---

## 一、问题背景：PreNorm Dilution

### 1.1 传统残差连接

```python
# 标准残差
h_l = h_{l-1} + f_{l-1}(h_{l-1})

# 展开
h_l = h_0 + Σ f_i(h_i)  # 每项权重固定为 1
```

**问题**: 所有层等权累加，深层隐状态量级无限增长 O(L)

### 1.2 PreNorm Dilution (预归一化稀释)

```
深度 l    残差流量级    深层贡献权重    结果
10层      ~3×          尚可           训练稳定
50层      ~7×          明显稀释       深层梯度偏小
100层     ~10×         严重稀释       训练不稳定
200层     ~14×         几乎失效       极深网络性能受损
```

**核心矛盾**: 残差流量级随深度增长，但每层归一化输入尺度固定，深层贡献需要越来越大的输出幅度。

---

## 二、Attention Residuals 核心原理

### 2.1 核心思想

> **深度方向的 Softmax 注意力替代固定等权累加**

```python
# 标准残差
h_l = Σ v_i                    # 固定权重 1

# Attention Residuals
h_l = Σ α_{i→l} · v_i          # 动态注意力权重
```

### 2.2 数学形式

```python
# 注意力权重计算
α_{i→l} = softmax(w_l^T · RMSNorm(v_i))

# 其中:
# - w_l: 伪查询向量 (每层可学习，固定参数)
# - v_i: 第 i 层的隐状态输出 (值向量)
# - RMSNorm(v_i): 归一化后的键向量
```

### 2.3 关键设计

| 设计元素 | 具体做法 | 设计意图 |
|---------|---------|---------|
| 伪查询向量 w_l | 每层固定参数，不依赖输入 | 降低计算开销 |
| 值向量 v_i | 第 i 层实际隐状态 | 覆盖全部历史层 |
| 键向量 RMSNorm(v_i) | 归一化后作键 | 内容依赖检索 |
| 零初始化 | w_l 初始化为全零 | 训练初期等价标准残差 |
| Softmax 归一化 | 权重和 = 1 | 控制隐状态量级 |

---

## 三、Block AttnRes (工程化方案)

### 3.1 Full AttnRes 的瓶颈

- 内存开销 O(L·d): 需要存储所有历史层隐状态
- 对于 48B 参数、54 层模型，几乎不可行

### 3.2 Block AttnRes 解法

```python
# 分块策略
L 层 → N 个 Block

# 块内: 标准残差累加
h_{block_i} = h_{block_i-1} + f(h_{block_i-1})

# 块间: Softmax 注意力聚合
h_{final} = Σ α_j · block_summary_j
```

**内存优化**: O(L·d) → O(N·d)

### 3.3 最优块数

| 块数 N | 性能 | 显存开销 |
|--------|------|---------|
| 1 | 标准残差 | 最低 |
| 8 (推荐) | 接近 Full AttnRes | 可接受 |
| 16 | 开始退化 | 较高 |
| L | Full AttnRes | 最高 |

---

## 四、实验结果

### 4.1 扩展律实验 (5 个规模)

| 变体 | 扩展律拟合 | 含义 |
|------|-----------|------|
| 标准 PreNorm | L = 1.891 × C^{-0.057} | 基准 |
| Block AttnRes | L = 1.870 × C^{-0.058} | 等效 1.25× 算力 |
| Full AttnRes | L = 1.865 × C^{-0.057} | 略优 |

**核心结论**: Block AttnRes 等效于基线模型使用 **1.25× 计算量**

### 4.2 Kimi Linear 48B 下游任务

| 基准 | 基线 | Block AttnRes | 提升 |
|------|------|---------------|------|
| GPQA-Diamond | 36.9 | 44.4 | **+7.5** ⭐ |
| Math | 53.5 | 57.1 | +3.6 |
| HumanEval | 59.1 | 62.2 | +3.1 |
| MMLU | 73.5 | 74.6 | +1.1 |

**关键发现**: 多步推理任务提升最显著 (GPQA +7.5)

---

## 五、与 mHC (DeepSeek) 对比

| 维度 | mHC | AttnRes |
|------|-----|---------|
| 稳定性 | Birkhoff 流形投影 | Softmax 归一化 |
| 内存 I/O/层 | 34d | 5.5d (6× 优势) |
| 推理延迟 | 较高 | < 2% |
| 工程难度 | ⭐⭐⭐⭐ 高 | ⭐⭐ 低 |
| 验证规模 | 27B / 1T | 48B / 1.4T |

**选择建议**:
- 需要**多流并行表征** → mHC
- 需要**轻量插件式方案** → AttnRes

---

## 六、应用到 CTR 预测

### 6.1 适用性分析

| 条件 | CTR 场景 | 是否满足 |
|------|---------|---------|
| 深度网络 (>32层) | 通常 4-8 层 | ❌ 较浅 |
| PreNorm 架构 | 是 | ✅ |
| 深层梯度问题 | 不明显 | ❌ |
| 显存敏感 | 是 | ✅ |

**结论**: CTR 模型层数较浅，AttnRes 收益可能有限，但仍可尝试。

### 6.2 实现方案

```python
class CTRWithAttnRes(nn.Module):
    """CTR 模型 + Block AttnRes"""
    def __init__(self, d_model, num_layers, num_blocks=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.layers_per_block = num_layers // num_blocks
        
        # 每个块一个伪查询向量
        self.query_vectors = nn.Parameter(torch.zeros(num_blocks, d_model))
        
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerLayer(d_model)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        block_outputs = [x]  # 初始嵌入
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 块边界: 存储块摘要
            if (i + 1) % self.layers_per_block == 0:
                block_outputs.append(x)
        
        # 跨块注意力聚合
        query = self.query_vectors[-1]  # 最后一层的查询
        keys = torch.stack([RMSNorm(b) for b in block_outputs])
        attn_weights = F.softmax(query @ keys.T, dim=-1)
        
        output = sum(w * b for w, b in zip(attn_weights, block_outputs))
        return output
```

### 6.3 预期收益

- **LLM**: +1.1~7.5 分 (取决于任务)
- **CTR**: +0.3~0.8bp (估计，需实验验证)

---

## 七、关键启示

1. **深度维度 = 序列维度**: 深度方向的残差传递可以用 Attention 思想优化
2. **权重和为 1**: Softmax 归一化根治 PreNorm Dilution
3. **插件式设计**: Block AttnRes 可无缝替换标准残差
4. **工程友好**: < 2% 推理延迟，O(N·d) 内存开销

---

**参考资源**:
- [论文](https://arxiv.org/abs/2603.15031)
- [官方 GitHub](https://github.com/MoonshotAI/Attention-Residuals)
