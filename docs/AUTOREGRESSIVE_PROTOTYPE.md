# LiteGenRec Phase 2: 自回归生成原型

**目标**: 实现生成式推荐原型，将 CTR 预测建模为序列生成任务

**时间**: 2026-03-30 开始

---

## 一、核心思路

### 判别式 vs 生成式

```python
# 当前: 判别式 (二分类)
score = model(features)
loss = BCE(score, label)

# 目标: 生成式 (自回归)
logits = model.generate(history_features)
loss = CrossEntropy(logits, target_item)
```

### 架构选择

```
输入: 用户历史交互序列 (特征序列)
  ↓
Embedding Layer
  ↓
GPT-style Decoder (Causal Attention)
  ↓
预测下一个物品的 Semantic ID
```

---

## 二、技术方案

### Phase 2.1: 特征序列建模

**简化方案**: 不生成物品 ID，而是生成特征组合

```python
# 输入: 历史特征序列
history = [f1, f2, f3, ...]  # 每个交互的特征向量

# 模型: GPT-style
class GenerativeCTR(nn.Module):
    def __init__(self):
        self.embedding = FeatureEmbedding()
        self.decoder = TransformerDecoder()  # Causal Mask
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, history_features):
        # history: [batch, seq_len, feature_dim]
        x = self.embedding(history_features)
        x = self.decoder(x)  # Causal attention
        logits = self.head(x[:, -1, :])  # 最后一个位置
        return logits
```

**问题**: 直接预测什么？
- 物品 ID? 词汇表太大 (百万级)
- 特征组合? 不够语义化
- Semantic ID? 需要 VQ-VAE

### Phase 2.2: 简化方案 - 特征级生成

**思路**: 每个特征独立生成

```python
# 预测下一个交互的各特征值
# C1, C2, ..., C26 各自生成

class FeatureLevelGenerator(nn.Module):
    def forward(self, history):
        # 历史序列编码
        context = self.encoder(history)

        # 多头输出 (每个特征一个)
        outputs = {}
        for i, field in enumerate(sparse_fields):
            logits = self.heads[i](context)
            outputs[field] = logits

        return outputs  # Dict[field, logits]
```

**训练目标**: 预测下一个交互的特征值

**推理**: 生成特征组合 → 检索匹配物品

---

## 三、实验计划

### Exp16: 特征级自回归原型

**数据**: Criteo (单交互，无序列)

**模拟序列**:
- 随机打乱样本，模拟历史
- 或用用户 ID 分组 (Criteo 无用户 ID)

**简化**: 无真实序列，用随机 N 个样本作为历史

### Exp17: 添加 Semantic ID

**先训练 RQ-VAE**:
1. 用物品特征训练量化器
2. 生成 Semantic ID
3. 用 GPT 预测 Semantic ID

---

## 四、代码实现

### 第一步: 简单 GPT-style 模型

```python
class SimpleGPT4CTR(nn.Module):
    def __init__(self, num_features, embed_dim, num_heads, num_layers):
        self.embedding = nn.Linear(num_features, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # 每个特征独立输出头
        self.heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size)
            for vocab_size in vocab_sizes
        ])

    def forward(self, x):
        # x: [batch, seq_len, num_features]
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1))
        x = self.decoder(x, x, tgt_mask=mask)

        # 多头输出
        outputs = [head(x[:, -1, :]) for head in self.heads]
        return outputs
```

---

## 五、评估方式

### 训练目标
- 预测下一个样本的特征值
- 多任务 Loss: sum(CrossEntropy for each field)

### 评估指标
- Accuracy@1 / @5 / @10 for each field
- 整体特征组合匹配率

---

## 六、下一步

1. 实现 Exp16 特征级生成原型
2. 验证自回归建模是否有效
3. 如果有效，继续 Exp17 (Semantic ID)

开始编码...
