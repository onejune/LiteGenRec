#!/usr/bin/env python3
"""
Exp09: Simple Attention GenCTR
核心思想: 回归简单，用最简洁的注意力机制 + 深度网络
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


class SimpleAttention(nn.Module):
    """简单自注意力 - 不用多头，直接算"""
    
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, embed_dim]"""
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.bmm(attn, V)
        return out


class SimpleAttentionGenCTR(nn.Module):
    """简单注意力CTR模型"""
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        num_attn_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        
        # 特征嵌入
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # 注意力层
        self.attn_layers = nn.ModuleList([
            nn.Sequential(
                SimpleAttention(embed_dim, dropout),
                nn.LayerNorm(embed_dim),
            )
            for _ in range(num_attn_layers)
        ])
        
        # 深度网络
        dnn_input = num_sparse_features * embed_dim
        self.dnn = nn.Sequential(
            nn.Linear(dnn_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 嵌入
        embeds = []
        for i in range(self.num_sparse_features):
            field_vals = x[:, i].long().clamp(0, self.embeddings[i].num_embeddings - 1)
            embeds.append(self.embeddings[i](field_vals))
        
        embed_stack = torch.stack(embeds, dim=1)  # [batch, num_fields, embed_dim]
        
        # 注意力
        attn_out = embed_stack
        for attn_layer in self.attn_layers:
            attn_out = attn_out + attn_layer(attn_out)
        
        # 展平
        flat = attn_out.reshape(batch_size, -1)
        
        # DNN
        logits = self.dnn(flat).squeeze(-1)
        
        return logits


def load_criteo_data(data_dir: str):
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
    
    return X_train, X_valid, y_train, y_valid, vocab_sizes


def train_model(model, train_loader, valid_loader, epochs=5, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
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
            torch.save(model.state_dict(), 'best_simple_attention.pt')
    
    return best_auc


def main():
    print("=" * 60)
    print("Exp09: Simple Attention GenCTR")
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
    
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=4)
    
    model = SimpleAttentionGenCTR(
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=16,
        hidden_dim=64,
        num_attn_layers=1,
        dropout=0.3
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=5, lr=1e-3)
    
    print("\n" + "=" * 60)
    print(f"Best Valid AUC: {best_auc:.4f}")
    print("=" * 60)
    
    print("\n对比:")
    print(f"  DeepFM baseline: 0.7472")
    print(f"  LiteGenRec V1:   0.7663")
    print(f"  LiteGenRec V2:   0.7678")
    print(f"  Exp09 (本实验):  {best_auc:.4f}")
    
    return best_auc


if __name__ == "__main__":
    main()
