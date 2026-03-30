#!/usr/bin/env python3
"""
Exp06: Semantic ID GenCTR V3 - 简化版
去掉VQ-VAE复杂性，直接用离散化+Transformer
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pyarrow.parquet as pq
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class FeatureHasher(nn.Module):
    """特征哈希层 - 将高基数特征映射到固定大小的语义空间"""
    
    def __init__(self, num_buckets: int = 1000):
        super().__init__()
        self.num_buckets = num_buckets
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_features] -> [batch, num_features] 哈希后的索引"""
        return x % self.num_buckets


class SimpleGenCTR(nn.Module):
    """简化版生成式CTR - 直接用特征序列建模"""
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.embed_dim = embed_dim
        
        # 特征嵌入 - 使用统一的嵌入表
        max_vocab = max(vocab_sizes) + 1
        self.shared_embedding = nn.Embedding(max_vocab, embed_dim, padding_idx=0)
        
        # 特征类型嵌入 - 区分不同的特征字段
        self.field_embedding = nn.Embedding(num_sparse_features, embed_dim)
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, num_features] 稀疏特征索引
        返回: logits [batch]
        """
        batch_size = x.size(0)
        
        # 1. 特征值嵌入
        x_clamped = x.long().clamp(0, self.shared_embedding.num_embeddings - 1)
        value_embeds = self.shared_embedding(x_clamped)  # [batch, num_features, embed_dim]
        
        # 2. 特征字段嵌入
        field_ids = torch.arange(self.num_sparse_features, device=x.device)
        field_embeds = self.field_embedding(field_ids)  # [num_features, embed_dim]
        field_embeds = field_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 3. 组合嵌入
        combined = value_embeds + field_embeds  # [batch, num_features, embed_dim]
        combined = self.layer_norm(combined)
        
        # 4. 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        seq = torch.cat([cls_tokens, combined], dim=1)  # [batch, 1+num_features, embed_dim]
        
        # 5. Transformer编码
        encoded = self.transformer(seq)  # [batch, 1+num_features, embed_dim]
        
        # 6. 取CLS token的输出
        cls_output = encoded[:, 0, :]  # [batch, embed_dim]
        
        # 7. 预测
        logits = self.output(cls_output).squeeze(-1)  # [batch]
        
        return logits


def load_criteo_data(data_dir: str):
    """加载Criteo数据"""
    print("Loading Criteo dataset...")
    
    train_path = f"{data_dir}/train_train.parquet"
    valid_path = f"{data_dir}/train_valid.parquet"
    
    train_files = sorted([f for f in os.listdir(train_path) if f.endswith('.parquet')])[:2]
    train_dfs = [pq.read_table(os.path.join(train_path, f)).to_pandas() for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    valid_files = sorted([f for f in os.listdir(valid_path) if f.endswith('.parquet')])[:1]
    valid_dfs = [pq.read_table(os.path.join(valid_path, f)).to_pandas() for f in valid_files]
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    
    print(f"Loaded train: {len(train_df)}, valid: {len(valid_df)}")
    
    sparse_cols = [f'C{i}' for i in range(1, 27)]
    all_df = pd.concat([train_df, valid_df], ignore_index=True)
    
    vocab_sizes = []
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('__MISSING__')
        encoder = LabelEncoder()
        all_df[col] = encoder.fit_transform(all_df[col].astype(str)) + 1
        vocab_sizes.append(int(all_df[col].max()) + 1)
    
    train_df = all_df.iloc[:len(train_df)]
    valid_df = all_df.iloc[len(train_df):]
    
    X_train = train_df[sparse_cols].values
    X_valid = valid_df[sparse_cols].values
    y_train = train_df['label'].values
    y_valid = valid_df['label'].values
    
    print(f"Max vocab: {max(vocab_sizes)}")
    
    return X_train, X_valid, y_train, y_valid, vocab_sizes


def train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        # 验证
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Valid AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_simple_genctr.pt')
    
    return best_auc


def main():
    print("=" * 60)
    print("Exp06: Simple GenCTR V3 (Transformer-based)")
    print("=" * 60)
    
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    X_train, X_valid, y_train, y_valid, vocab_sizes = load_criteo_data(data_dir)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float)
    )
    valid_dataset = TensorDataset(
        torch.tensor(X_valid, dtype=torch.long),
        torch.tensor(y_valid, dtype=torch.float)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=False, num_workers=4)
    
    model = SimpleGenCTR(
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=64,
        hidden_dim=256,
        num_heads=8,
        num_layers=3,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3)
    
    print("\n" + "=" * 60)
    print(f"Best Valid AUC: {best_auc:.4f}")
    print("=" * 60)
    
    with open('results_v3.txt', 'w') as f:
        f.write(f"Simple GenCTR V3 Results\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")
    
    return best_auc


if __name__ == "__main__":
    main()
