#!/usr/bin/env python3
"""
Exp13: HSTU-Lite for CTR Prediction

基于 Meta HSTU (ICML 2024) 的轻量化实现
核心创新: Pointwise Aggregated Attention (无 softmax)

参考论文: Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers
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


class PointwiseAggregatedAttention(nn.Module):
    """
    HSTU 核心组件: Pointwise Aggregated Attention
    
    与标准 Transformer Attention 的区别:
    1. 无 softmax 归一化 (适合非平稳词汇表)
    2. 点-wise 聚合而非全局归一化
    3. 更适合高基数特征
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V 投影
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Pointwise Aggregated Attention (无 softmax)
        # 核心创新: 直接点积 + 激活，不做 softmax 归一化
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # HSTU 使用 sigmoid 或 relu 激活，而非 softmax
        attn = F.sigmoid(attn)  # pointwise activation
        attn = self.dropout(attn)
        
        # 加权聚合
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.o_proj(out)
        
        return out


class HSTUBlock(nn.Module):
    """HSTU Block: Pointwise Attention + MLP"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = PointwiseAggregatedAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # HSTU 使用简化的 MLP (2层而非6层)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-norm
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HSTULiteCTR(nn.Module):
    """HSTU-Lite for CTR Prediction"""
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        
        # 独立嵌入表
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # HSTU Blocks
        self.blocks = nn.ModuleList([
            HSTUBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # 输出层
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 特征嵌入
        embeds = []
        for i in range(self.num_sparse_features):
            field_vals = x[:, i].long().clamp(0, self.embeddings[i].num_embeddings - 1)
            embeds.append(self.embeddings[i](field_vals))
        
        # 序列化 (无位置编码，HSTU 设计)
        seq = torch.stack(embeds, dim=1)
        
        # 加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        
        # HSTU Blocks
        for block in self.blocks:
            seq = block(seq)
        
        # CLS token 预测
        logits = self.mlp(seq[:, 0, :]).squeeze(-1)
        
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


def train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3):
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
    
    return best_auc


def main():
    print("=" * 60)
    print("Exp13: HSTU-Lite for CTR Prediction")
    print("Core: Pointwise Aggregated Attention (no softmax)")
    print("=" * 60)
    
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    X_train, X_valid, y_train, y_valid, vocab_sizes = load_criteo_data(data_dir)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.long), torch.tensor(y_valid, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=4)
    
    model = HSTULiteCTR(
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=64,
        num_heads=8,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print("Architecture: Embedding -> HSTU Blocks (Pointwise Attn) -> MLP")
    print()
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3)
    
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    print(f"HSTU-Lite AUC: {best_auc:.4f}")
    print(f"V2 (标准 Transformer): 0.7678")
    
    if best_auc > 0.7678:
        print(f"✅ HSTU-Lite 优于 V2 by +{(best_auc - 0.7678) * 100:.2f}bp")
    else:
        print(f"❌ HSTU-Lite 低于 V2 by {(best_auc - 0.7678) * 100:.2f}bp")
    
    torch.save(model.state_dict(), 'hstu_lite_model.pth')
    print(f"\n模型已保存: hstu_lite_model.pth")
    
    return best_auc


if __name__ == "__main__":
    main()
