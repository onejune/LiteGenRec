#!/usr/bin/env python3
"""
Exp14: Mamba4CTR - State Space Model for CTR Prediction

基于 Mamba 架构，线性时间复杂度
适合长序列和高效推理

参考: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
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


class MambaBlock(nn.Module):
    """
    Simplified Mamba Block for CTR
    
    核心思想: Selective State Space Model
    - 线性复杂度 O(n)
    - 数据依赖的状态转换
    """
    
    def __init__(self, embed_dim, state_dim=16, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        
        # 输入投影
        self.x_proj = nn.Linear(embed_dim, embed_dim * 2)  # for delta, B
        
        # SSM 参数
        self.A = nn.Parameter(torch.randn(embed_dim, state_dim))
        self.D = nn.Parameter(torch.ones(embed_dim))
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化 A 为负值（稳定）
        nn.init.uniform_(self.A, -0.01, 0.01)
    
    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # 残差
        residual = x
        x = self.norm(x)
        
        # 投影得到 delta 和 B
        x_proj = self.x_proj(x)
        delta, B = x_proj.chunk(2, dim=-1)
        
        # delta: 数据依赖的步长 (选择性)
        delta = F.softplus(delta)  # 确保正数
        
        # 简化的 SSM: h_t = A * h_{t-1} + B * x_t
        # 使用累积和近似 (并行化)
        
        # 离散化 A
        A_discrete = torch.exp(delta.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(0))
        # [batch, seq_len, embed_dim, state_dim]
        
        # 状态更新 (简化版，实际 Mamba 使用更复杂的并行扫描)
        # 这里用简化版本: output = (A * x).cumsum() * B
        h = torch.cumsum(x.unsqueeze(-1) * B.unsqueeze(-1), dim=1)
        h = h * A_discrete
        
        # 输出
        out = h.sum(dim=-1) + x * self.D
        
        out = self.out_proj(out)
        out = self.dropout(out)
        
        return out + residual


class MambaLayer(nn.Module):
    """Mamba Layer: 包含前向和反向两个方向"""
    
    def __init__(self, embed_dim, state_dim=16, dropout=0.1):
        super().__init__()
        self.forward_mamba = MambaBlock(embed_dim, state_dim, dropout)
        self.backward_mamba = MambaBlock(embed_dim, state_dim, dropout)
        self.merge = nn.Linear(embed_dim * 2, embed_dim)
    
    def forward(self, x):
        # 双向 Mamba
        forward_out = self.forward_mamba(x)
        backward_out = self.backward_mamba(x.flip(dims=[1])).flip(dims=[1])
        
        # 合并
        out = torch.cat([forward_out, backward_out], dim=-1)
        out = self.merge(out)
        
        return out


class Mamba4CTR(nn.Module):
    """Mamba-based CTR Model"""
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 64,
        state_dim: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        
        # 嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # Mamba Layers
        self.mamba_layers = nn.ModuleList([
            MambaLayer(embed_dim, state_dim, dropout)
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
        
        # 序列化
        seq = torch.stack(embeds, dim=1)
        
        # 加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        
        # Mamba Layers
        for mamba_layer in self.mamba_layers:
            seq = mamba_layer(seq)
        
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
    print("Exp14: Mamba4CTR - State Space Model")
    print("Core: Linear Time Complexity O(n)")
    print("=" * 60)
    
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    X_train, X_valid, y_train, y_valid, vocab_sizes = load_criteo_data(data_dir)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.long), torch.tensor(y_valid, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=4)
    
    model = Mamba4CTR(
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=64,
        state_dim=16,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print("Architecture: Embedding -> Mamba Layers (Bi-directional) -> MLP")
    print()
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3)
    
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    print(f"Mamba4CTR AUC: {best_auc:.4f}")
    print(f"V2 (标准 Transformer): 0.7678")
    
    if best_auc > 0.7678:
        print(f"✅ Mamba4CTR 优于 V2 by +{(best_auc - 0.7678) * 100:.2f}bp")
    else:
        print(f"❌ Mamba4CTR 低于 V2 by {(best_auc - 0.7678) * 100:.2f}bp")
    
    torch.save(model.state_dict(), 'mamba4ctr_model.pth')
    print(f"\n模型已保存: mamba4ctr_model.pth")
    
    return best_auc


if __name__ == "__main__":
    main()
