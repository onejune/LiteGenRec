#!/usr/bin/env python3
"""
Exp16 V2: 简化版自回归生成

改进点:
1. 不使用伪序列，直接用单样本
2. 每个特征独立预测 (类似多任务学习)
3. 简化架构，减少复杂度
"""

import os
import sys
sys.path.append('/mnt/workspace/walter.wan/git_project/github_onejune/LiteGenRec/experiments')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_data(data_dir, max_vocab_size=50000):
    """加载数据"""
    print("Loading Criteo dataset...")

    train_path = f"{data_dir}/train_train.parquet"
    valid_path = f"{data_dir}/train_valid.parquet"

    train_files = sorted([f for f in os.listdir(train_path) if f.endswith('.parquet')])[:2]
    train_dfs = [pq.read_table(os.path.join(train_path, f)).to_pandas() for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True).head(500000)

    valid_files = sorted([f for f in os.listdir(valid_path) if f.endswith('.parquet')])[:1]
    valid_dfs = [pq.read_table(os.path.join(valid_path, f)).to_pandas() for f in valid_files]
    valid_df = pd.concat(valid_dfs, ignore_index=True).head(50000)

    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}")

    dense_cols = [f'I{i}' for i in range(1, 14)]
    sparse_cols = [f'C{i}' for i in range(1, 27)]

    all_df = pd.concat([train_df, valid_df], ignore_index=True)

    for col in dense_cols:
        all_df[col] = all_df[col].fillna(0)
    scaler = StandardScaler()
    all_df[dense_cols] = scaler.fit_transform(all_df[dense_cols])

    vocab_sizes = []
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('__MISSING__')
        encoder = LabelEncoder()
        all_df[col] = encoder.fit_transform(all_df[col].astype(str)) + 1
        all_df[col] = np.clip(all_df[col], 0, max_vocab_size)
        vocab_sizes.append(min(int(all_df[col].max()) + 1, max_vocab_size + 1))

    train_df = all_df.iloc[:len(train_df)]
    valid_df = all_df.iloc[len(train_df):]

    return (
        train_df[dense_cols].values.astype(np.float32),
        train_df[sparse_cols].values.astype(np.int64),
        train_df['label'].values.astype(np.float32),
        valid_df[dense_cols].values.astype(np.float32),
        valid_df[sparse_cols].values.astype(np.int64),
        valid_df['label'].values.astype(np.float32),
        vocab_sizes
    )


class FeatureMaskedModel(nn.Module):
    """
    掩码特征预测模型
    随机 mask 部分特征，用其他特征预测
    类似 BERT 的 Masked Language Modeling
    """
    def __init__(
        self,
        num_dense: int,
        num_sparse: int,
        vocab_sizes: list,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        mask_prob: float = 0.15
    ):
        super().__init__()
        self.num_dense = num_dense
        self.num_sparse = num_sparse
        self.embed_dim = embed_dim
        self.mask_prob = mask_prob

        # 稠密特征嵌入
        self.dense_embed = nn.Linear(num_dense, num_dense * embed_dim)

        # 稀疏特征嵌入
        self.sparse_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])

        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)[MASK] token
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # 总 token 数
        total_tokens = num_dense + num_sparse

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 输出头
        # 稠密特征: 回归
        self.dense_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1)
            for _ in range(num_dense)
        ])

        # 稀疏特征: 分类
        self.sparse_heads = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size)
            for vocab_size in vocab_sizes
        ])

        # CTR 预测头
        self.ctr_head = nn.Sequential(
            nn.Linear(total_tokens * embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, dense_x, sparse_x, mask=None):
        """
        Args:
            dense_x: [batch, 13]
            sparse_x: [batch, 26]
            mask: [batch, 39] bool, True = masked
        """
        batch_size = dense_x.size(0)

        # 稠密特征嵌入
        dense_emb = self.dense_embed(dense_x).view(batch_size, self.num_dense, self.embed_dim)

        # 稀疏特征嵌入
        sparse_embs = []
        for i in range(self.num_sparse):
            field_vals = sparse_x[:, i].long().clamp(0, self.sparse_embeddings[i].num_embeddings - 1)
            emb = self.sparse_embeddings[i](field_vals)
            sparse_embs.append(emb)
        sparse_emb = torch.stack(sparse_embs, dim=1)

        # 合并: [batch, 39, 64]
        tokens = torch.cat([dense_emb, sparse_emb], dim=1)

        # 应用 mask
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(tokens)
            tokens = torch.where(mask_expanded, self.mask_token.expand_as(tokens), tokens)

        # Transformer
        encoded = self.encoder(tokens)

        # 预测各特征
        dense_preds = [head(encoded[:, i, :]) for i, head in enumerate(self.dense_heads)]
        sparse_logits = [head(encoded[:, self.num_dense + i, :]) for i, head in enumerate(self.sparse_heads)]

        # CTR 预测
        ctr_logits = self.ctr_head(encoded.view(batch_size, -1)).squeeze(-1)

        return dense_preds, sparse_logits, ctr_logits


def train_model(model, train_loader, valid_loader, vocab_sizes, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_sparse = nn.CrossEntropyLoss(ignore_index=0)
    criterion_dense = nn.MSELoss()
    criterion_ctr = nn.BCEWithLogitsLoss()

    best_auc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (dense_x, sparse_x, y) in enumerate(train_loader):
            dense_x = dense_x.to(device)
            sparse_x = sparse_x.to(device)
            y = y.to(device).float()

            # 随机 mask
            batch_size = dense_x.size(0)
            total_features = 13 + 26
            mask = torch.rand(batch_size, total_features, device=device) < model.mask_prob

            optimizer.zero_grad()

            dense_preds, sparse_logits, ctr_logits = model(dense_x, sparse_x, mask)

            # 计算 loss (只对 masked 位置)
            loss = 0

            # 稠密特征 loss
            for i, pred in enumerate(dense_preds):
                if mask[:, i].any():
                    target = dense_x[:, i].unsqueeze(-1)
                    loss += criterion_dense(pred[mask[:, i]], target[mask[:, i]])

            # 稀疏特征 loss
            for i, logits in enumerate(sparse_logits):
                if mask[:, 13 + i].any():
                    target = sparse_x[:, i]
                    loss += criterion_sparse(logits[mask[:, 13 + i]], target[mask[:, 13 + i]])

            # CTR loss
            loss += 0.5 * criterion_ctr(ctr_logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

        scheduler.step()

        # 验证
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for dense_x, sparse_x, y in valid_loader:
                dense_x = dense_x.to(device)
                sparse_x = sparse_x.to(device)

                _, _, ctr_logits = model(dense_x, sparse_x, mask=None)
                preds = torch.sigmoid(ctr_logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/num_batches:.4f}, Valid AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'masked_model_best.pth')

    return best_auc


def main():
    print("=" * 70)
    print("Exp16 V2: 掩码特征预测模型 (类似 BERT MLM)")
    print("=" * 70)

    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"

    (X_dense_train, X_sparse_train, y_train,
     X_dense_valid, X_sparse_valid, y_valid,
     vocab_sizes) = load_data(data_dir)

    train_dataset = TensorDataset(
        torch.tensor(X_dense_train),
        torch.tensor(X_sparse_train),
        torch.tensor(y_train)
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_dense_valid),
        torch.tensor(X_sparse_valid),
        torch.tensor(y_valid)
    )

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=4)

    model = FeatureMaskedModel(
        num_dense=13,
        num_sparse=26,
        vocab_sizes=vocab_sizes,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
        mask_prob=0.15
    ).to(device)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")

    best_auc = train_model(model, train_loader, valid_loader, vocab_sizes, epochs=3)

    print("\n" + "=" * 70)
    print("实验结果")
    print("=" * 70)
    print(f"Best Valid AUC: {best_auc:.4f}")
    print(f"对比: HSTU-Lite V3 = 0.7853, DeepFM = 0.7472")

    if best_auc > 0.7853:
        print("✅ 超越当前最佳!")
    elif best_auc > 0.7472:
        print("⚠️ 优于 DeepFM，但未超越 HSTU-Lite V3")
    else:
        print("❌ 未超越基线")

    return best_auc


if __name__ == "__main__":
    main()
