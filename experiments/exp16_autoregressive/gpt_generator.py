#!/usr/bin/env python3
"""
Exp16: 特征级自回归生成原型

将 CTR 预测建模为特征序列生成任务
- 输入: 历史 N 个样本的特征
- 输出: 预测下一个样本的特征值
- 推理: 生成的特征组合 → 检索匹配物品 → 计算 CTR
"""

import os
import sys
sys.path.append('/mnt/workspace/walter.wan/git_project/github_onejune/LiteGenRec/experiments')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pyarrow.parquet as pq
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# 数据集: 构建伪序列
# ============================================================

class CriteoSequenceDataset(Dataset):
    """
    Criteo 数据集没有真实用户序列
    简化方案: 每个样本 + 前 N 个随机样本作为"历史"
    """
    def __init__(self, data_dir, split='train', seq_len=5, max_vocab_size=50000,
                 num_samples=None):
        self.seq_len = seq_len

        # 加载数据
        print(f"Loading Criteo {split} data...")
        if split == 'train':
            path = f"{data_dir}/train_train.parquet"
        else:
            path = f"{data_dir}/train_valid.parquet"

        files = sorted([f for f in os.listdir(path) if f.endswith('.parquet')])
        if split == 'train':
            files = files[:2]  # 只用前 2 个文件
        else:
            files = files[:1]

        dfs = [pq.read_table(os.path.join(path, f)).to_pandas() for f in files]
        df = pd.concat(dfs, ignore_index=True)

        if num_samples:
            df = df.head(num_samples)

        print(f"Loaded {len(df)} samples")

        # 特征列
        dense_cols = [f'I{i}' for i in range(1, 14)]
        sparse_cols = [f'C{i}' for i in range(1, 27)]

        # 处理稠密特征
        for col in dense_cols:
            df[col] = df[col].fillna(0)
        scaler = StandardScaler()
        df[dense_cols] = scaler.fit_transform(df[dense_cols])

        # 处理稀疏特征
        self.vocab_sizes = []
        for col in sparse_cols:
            df[col] = df[col].fillna('__MISSING__')
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col].astype(str)) + 1
            df[col] = np.clip(df[col], 0, max_vocab_size)
            self.vocab_sizes.append(min(int(df[col].max()) + 1, max_vocab_size + 1))

        # 保存特征
        self.dense_features = df[dense_cols].values.astype(np.float32)
        self.sparse_features = df[sparse_cols].values.astype(np.int64)
        self.labels = df['label'].values.astype(np.float32)

        print(f"Dense: {len(dense_cols)}, Sparse: {len(sparse_cols)}")
        print(f"Vocab sizes (first 5): {self.vocab_sizes[:5]}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 当前样本
        dense_current = self.dense_features[idx]
        sparse_current = self.sparse_features[idx]
        label = self.labels[idx]

        # 随机选择 seq_len-1 个"历史"样本
        history_indices = np.random.choice(
            len(self.labels), self.seq_len - 1, replace=True
        )

        dense_history = self.dense_features[history_indices]
        sparse_history = self.sparse_features[history_indices]

        # 拼接: [history..., current]
        dense_seq = np.vstack([dense_history, dense_current])  # [seq_len, 13]
        sparse_seq = np.vstack([sparse_history, sparse_current])  # [seq_len, 26]

        return {
            'dense_seq': torch.tensor(dense_seq),
            'sparse_seq': torch.tensor(sparse_seq),
            'label': torch.tensor(label),
            'target_sparse': torch.tensor(sparse_current),  # 预测目标
        }


# ============================================================
# 模型: GPT-style 自回归生成
# ============================================================

class GPTStyleGenerator(nn.Module):
    """
    GPT-style 自回归模型
    - 输入: 特征序列 [batch, seq_len, feature_dim]
    - 输出: 预测下一个样本的特征值
    """
    def __init__(
        self,
        num_dense_features: int = 13,
        num_sparse_features: int = 26,
        vocab_sizes: list = None,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_dense = num_dense_features
        self.num_sparse = num_sparse_features
        self.embed_dim = embed_dim

        # 稠密特征嵌入
        self.dense_embed = nn.Linear(num_dense_features, num_dense_features * embed_dim)

        # 稀疏特征嵌入
        self.sparse_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])

        # 序列总长度: 13稠密 + 26稀疏 = 39个token
        total_tokens = num_dense_features + num_sparse_features
        self.total_tokens = total_tokens

        # Token 位置编码
        self.token_pos = nn.Parameter(torch.randn(1, total_tokens, embed_dim))
        self.seq_pos = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Transformer Decoder (Causal)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # 输出头: 每个稀疏特征一个
        self.output_heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size)
            for vocab_size in vocab_sizes
        ])

        # CLS token for CTR prediction
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ctr_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.token_pos, std=0.02)
        nn.init.normal_(self.seq_pos, std=0.02)

    def forward(self, dense_seq, sparse_seq):
        """
        Args:
            dense_seq: [batch, seq_len, 13]
            sparse_seq: [batch, seq_len, 26]
        Returns:
            feature_logits: list of [batch, vocab_size] for each sparse field
            ctr_logits: [batch] for CTR prediction
        """
        batch_size, seq_len = dense_seq.size(0), dense_seq.size(1)

        # 嵌入稠密特征
        dense_emb = self.dense_embed(dense_seq)  # [batch, seq_len, 13*64]
        dense_emb = dense_emb.view(batch_size, seq_len, self.num_dense, self.embed_dim)

        # 嵌入稀疏特征
        sparse_embs = []
        for i in range(self.num_sparse):
            field_vals = sparse_seq[:, :, i].long().clamp(0, self.sparse_embeddings[i].num_embeddings - 1)
            emb = self.sparse_embeddings[i](field_vals)  # [batch, seq_len, 64]
            sparse_embs.append(emb)
        sparse_emb = torch.stack(sparse_embs, dim=2)  # [batch, seq_len, 26, 64]

        # 合并: [batch, seq_len, 39, 64]
        tokens = torch.cat([dense_emb, sparse_emb], dim=2)

        # 展平为序列: [batch, seq_len * 39, 64]
        tokens = tokens.view(batch_size, seq_len * self.total_tokens, self.embed_dim)

        # 添加位置编码 (广播到整个序列)
        total_seq_len = seq_len * self.total_tokens
        # token_pos: [1, 39, 64] -> repeat for each seq position
        token_pos = self.token_pos.repeat(1, seq_len, 1)  # [1, seq_len*39, 64]
        # seq_pos: [1, max_seq_len, 64] -> expand for each token
        seq_pos = self.seq_pos[:, :seq_len, :].unsqueeze(2).repeat(1, 1, self.total_tokens, 1)
        seq_pos = seq_pos.view(1, total_seq_len, self.embed_dim)

        tokens = tokens + token_pos + seq_pos

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Causal mask
        seq_total_len = tokens.size(1)
        mask = torch.triu(torch.ones(seq_total_len, seq_total_len, device=tokens.device) * float('-inf'), diagonal=1)

        # Transformer (自编码，用 tgt_mask 实现 causal)
        # 这里简化: 用 nn.TransformerEncoder 手动加 mask
        # 改用自定义实现
        encoded = self._causal_transformer(tokens, mask)

        # 提取最后一个序列位置的表示
        last_pos = encoded[:, -self.total_tokens:, :]  # [batch, 39, 64]

        # 预测各稀疏特征
        feature_logits = []
        for i, head in enumerate(self.output_heads):
            # 用第 i 个 token 预测第 i 个特征
            logits = head(last_pos[:, self.num_dense + i, :])  # [batch, vocab_size]
            feature_logits.append(logits)

        # CTR 预测 (用 CLS token)
        ctr_logits = self.ctr_head(encoded[:, 0, :]).squeeze(-1)

        return feature_logits, ctr_logits

    def _causal_transformer(self, x, mask):
        """简化的 causal transformer"""
        # 直接用 decoder 的 self-attention
        # 这里简化处理，不严格 causal
        for layer in self.decoder.layers:
            x = layer(x, x, tgt_mask=mask)
        return x


# ============================================================
# 训练
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, vocab_sizes):
    model.train()
    total_loss = 0
    total_feature_loss = 0
    total_ctr_loss = 0
    num_batches = 0

    criterion_feature = nn.CrossEntropyLoss(ignore_index=0)
    criterion_ctr = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        dense_seq = batch['dense_seq'].to(device)
        sparse_seq = batch['sparse_seq'].to(device)
        target_sparse = batch['target_sparse'].to(device)
        label = batch['label'].to(device).float()

        optimizer.zero_grad()

        feature_logits, ctr_logits = model(dense_seq, sparse_seq)

        # 特征预测 loss
        feature_loss = 0
        for i, logits in enumerate(feature_logits):
            target = target_sparse[:, i].long()
            feature_loss += criterion_feature(logits, target)
        feature_loss = feature_loss / len(feature_logits)

        # CTR 预测 loss
        ctr_loss = criterion_ctr(ctr_logits, label)

        # 总 loss
        loss = feature_loss + 0.5 * ctr_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_feature_loss += feature_loss.item()
        total_ctr_loss += ctr_loss.item()
        num_batches += 1

    scheduler.step()

    return {
        'loss': total_loss / num_batches,
        'feature_loss': total_feature_loss / num_batches,
        'ctr_loss': total_ctr_loss / num_batches,
    }


def evaluate(model, dataloader, vocab_sizes):
    model.eval()

    all_preds = []
    all_labels = []

    # 特征准确率
    feature_correct = [0] * len(vocab_sizes)
    feature_total = 0

    with torch.no_grad():
        for batch in dataloader:
            dense_seq = batch['dense_seq'].to(device)
            sparse_seq = batch['sparse_seq'].to(device)
            target_sparse = batch['target_sparse'].to(device)
            label = batch['label'].to(device).float()

            feature_logits, ctr_logits = model(dense_seq, sparse_seq)

            # CTR AUC
            preds = torch.sigmoid(ctr_logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy())

            # 特征准确率
            for i, logits in enumerate(feature_logits):
                pred = logits.argmax(dim=-1)
                correct = (pred == target_sparse[:, i]).sum().item()
                feature_correct[i] += correct
            feature_total += target_sparse.size(0)

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(all_labels, all_preds)

    # 平均特征准确率
    avg_feature_acc = sum(feature_correct) / (feature_total * len(vocab_sizes))

    return {
        'auc': auc,
        'feature_acc': avg_feature_acc,
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Exp16: 特征级自回归生成原型")
    print("=" * 70)

    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"

    # 数据集
    train_dataset = CriteoSequenceDataset(data_dir, split='train', seq_len=5, num_samples=500000)
    valid_dataset = CriteoSequenceDataset(data_dir, split='valid', seq_len=5, num_samples=50000)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=4)

    vocab_sizes = train_dataset.vocab_sizes

    # 模型
    model = GPTStyleGenerator(
        num_dense_features=13,
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        max_seq_len=10,
        dropout=0.1
    ).to(device)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)

    # 训练
    epochs = 3
    best_auc = 0

    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, vocab_sizes)
        valid_metrics = evaluate(model, valid_loader, vocab_sizes)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_metrics['loss']:.4f} (feature: {train_metrics['feature_loss']:.4f}, ctr: {train_metrics['ctr_loss']:.4f})")
        print(f"  Valid AUC: {valid_metrics['auc']:.4f}, Feature Acc: {valid_metrics['feature_acc']:.4f}")

        if valid_metrics['auc'] > best_auc:
            best_auc = valid_metrics['auc']
            torch.save(model.state_dict(), 'gpt_generator_best.pth')

    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)
    print(f"Best Valid AUC: {best_auc:.4f}")
    print(f"对比基线: HSTU-Lite V3 = 0.7853")
    print(f"对比基线: DeepFM = 0.7472")

    if best_auc > 0.7853:
        print(f"✅ 超越当前最佳!")
    elif best_auc > 0.7472:
        print(f"⚠️ 优于 DeepFM，但未超越 HSTU-Lite V3")
    else:
        print(f"❌ 未超越基线")

    # 保存结果
    import json
    results = {
        'experiment': 'exp16_autoregressive_generation',
        'model': 'GPTStyleGenerator',
        'best_auc': best_auc,
        'params': sum(p.numel() for p in model.parameters()),
        'epochs': epochs,
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return best_auc


if __name__ == "__main__":
    main()
