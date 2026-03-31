#!/usr/bin/env python3
"""
Exp20: Interaction Attention 优化实验

实验组:
├── V1: HSTU-Lite + Interaction Attn (Pointwise + Interaction)
├── V2: Transformer + Interaction Attn (Softmax + Interaction)
├── V3: Pointwise + Double Interaction
├── V4: Transformer + Double Interaction
└── V5: HSTU-Lite + Double Interaction
"""

import os
import sys
sys.path.append('/mnt/workspace/walter.wan/git_project/github_onejune/LiteGenRec/experiments')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import json
from pathlib import Path

from data_loader import load_criteo_full_features

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================
# 基础组件
# ============================================================

class HSTULayer(nn.Module):
    """HSTU Layer with Pointwise Attention"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Pointwise Attention
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Pointwise Attention
        residual = x
        x = self.norm1(x)
        
        q = self.query(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, -1).transpose(1, 2)
        
        # Pointwise: Sigmoid activation instead of Softmax
        attn_weights = torch.sigmoid(torch.matmul(q, k.transpose(-2, -1)) / (self.d_model ** 0.5))
        attn_out = torch.matmul(attn_weights, v)
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        x = residual + self.dropout(attn_out)
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        
        return x


class TransformerLayer(nn.Module):
    """标准 Transformer Layer with Softmax Attention"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        
        return x


# ============================================================
# 模型定义
# ============================================================

class BaseModel(nn.Module):
    """基础模型类"""
    def __init__(
        self,
        num_dense=13,
        num_sparse=26,
        vocab_sizes=None,
        d_model=64,
        n_heads=8,
        num_layers=4,
        d_ff=256,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.dense_embed = nn.Linear(num_dense, num_dense * d_model)
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # Output
        total_features = num_dense + num_sparse
        self.output = nn.Sequential(
            nn.Linear(total_features * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )
        
    def embed_features(self, dense_x, sparse_x):
        batch_size = dense_x.size(0)
        dense_emb = self.dense_embed(dense_x).view(batch_size, 13, self.d_model)
        sparse_embs = [self.sparse_embeds[i](sparse_x[:, i]) for i in range(26)]
        sparse_emb = torch.stack(sparse_embs, dim=1)
        return torch.cat([dense_emb, sparse_emb], dim=1)


# ============================================================
# V1: HSTU-Lite + Interaction Attn
# ============================================================

class HSTULiteInteraction(BaseModel):
    """HSTU-Lite + Interaction Attention"""
    def __init__(self, vocab_sizes=None, d_model=64, n_heads=8, num_layers=4, d_ff=256, dropout=0.1):
        super().__init__(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        
        self.layers = nn.ModuleList([
            HSTULayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Interaction Attention
        self.interaction_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, dense_x, sparse_x):
        x = self.embed_features(dense_x, sparse_x)
        
        # HSTU layers
        for layer in self.layers:
            x = layer(x)
        
        # Interaction Attention
        residual = x
        x = self.norm(x)
        x = residual + self.interaction_attn(x, x, x)[0]
        
        x = x.reshape(x.size(0), -1)
        return self.output(x).squeeze(-1)


# ============================================================
# V2: Transformer + Interaction Attn
# ============================================================

class TransformerInteraction(BaseModel):
    """Transformer + Interaction Attention"""
    def __init__(self, vocab_sizes=None, d_model=64, n_heads=8, num_layers=4, d_ff=256, dropout=0.1):
        super().__init__(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Interaction Attention
        self.interaction_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, dense_x, sparse_x):
        x = self.embed_features(dense_x, sparse_x)
        
        for layer in self.layers:
            x = layer(x)
        
        # Interaction Attention
        residual = x
        x = self.norm(x)
        x = residual + self.interaction_attn(x, x, x)[0]
        
        x = x.reshape(x.size(0), -1)
        return self.output(x).squeeze(-1)


# ============================================================
# V3: Pointwise + Double Interaction
# ============================================================

class PointwiseDoubleInteraction(BaseModel):
    """Pointwise + Double Interaction Attention"""
    def __init__(self, vocab_sizes=None, d_model=64, n_heads=8, num_layers=4, d_ff=256, dropout=0.1):
        super().__init__(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        
        self.layers = nn.ModuleList([
            HSTULayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Double Interaction Attention
        self.interaction_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.interaction_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, dense_x, sparse_x):
        x = self.embed_features(dense_x, sparse_x)
        
        for layer in self.layers:
            x = layer(x)
        
        # First Interaction Attention
        residual = x
        x = self.norm1(x)
        x = residual + self.interaction_attn1(x, x, x)[0]
        
        # Second Interaction Attention
        residual = x
        x = self.norm2(x)
        x = residual + self.interaction_attn2(x, x, x)[0]
        
        x = x.reshape(x.size(0), -1)
        return self.output(x).squeeze(-1)


# ============================================================
# V4: Transformer + Double Interaction
# ============================================================

class TransformerDoubleInteraction(BaseModel):
    """Transformer + Double Interaction Attention"""
    def __init__(self, vocab_sizes=None, d_model=64, n_heads=8, num_layers=4, d_ff=256, dropout=0.1):
        super().__init__(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Double Interaction Attention
        self.interaction_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.interaction_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, dense_x, sparse_x):
        x = self.embed_features(dense_x, sparse_x)
        
        for layer in self.layers:
            x = layer(x)
        
        # First Interaction Attention
        residual = x
        x = self.norm1(x)
        x = residual + self.interaction_attn1(x, x, x)[0]
        
        # Second Interaction Attention
        residual = x
        x = self.norm2(x)
        x = residual + self.interaction_attn2(x, x, x)[0]
        
        x = x.reshape(x.size(0), -1)
        return self.output(x).squeeze(-1)


# ============================================================
# V5: HSTU-Lite + Double Interaction
# ============================================================

class HSTULiteDoubleInteraction(BaseModel):
    """HSTU-Lite + Double Interaction Attention"""
    def __init__(self, vocab_sizes=None, d_model=64, n_heads=8, num_layers=4, d_ff=256, dropout=0.1):
        super().__init__(vocab_sizes=vocab_sizes, d_model=d_model, n_heads=n_heads, num_layers=num_layers, d_ff=d_ff, dropout=dropout)
        
        self.layers = nn.ModuleList([
            HSTULayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Double Interaction Attention
        self.interaction_attn1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.interaction_attn2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, dense_x, sparse_x):
        x = self.embed_features(dense_x, sparse_x)
        
        for layer in self.layers:
            x = layer(x)
        
        # First Interaction Attention
        residual = x
        x = self.norm1(x)
        x = residual + self.interaction_attn1(x, x, x)[0]
        
        # Second Interaction Attention
        residual = x
        x = self.norm2(x)
        x = residual + self.interaction_attn2(x, x, x)[0]
        
        x = x.reshape(x.size(0), -1)
        return self.output(x).squeeze(-1)


# ============================================================
# 训练函数
# ============================================================

def train_model(model, train_loader, val_loader, epochs=3, lr=1e-3, name="model"):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        start_time = time.time()
        
        for batch_idx, (dense_x, sparse_x, y) in enumerate(train_loader):
            dense_x = dense_x.to(device)
            sparse_x = sparse_x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            outputs = model(dense_x, sparse_x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
            train_labels.extend(y.cpu().numpy())
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  [{name}] Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_auc = roc_auc_score(train_labels, train_preds)
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for dense_x, sparse_x, y in val_loader:
                dense_x = dense_x.to(device)
                sparse_x = sparse_x.to(device)
                outputs = model(dense_x, sparse_x)
                val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                val_labels.extend(y.numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        epoch_time = time.time() - start_time
        
        print(f"[{name}] Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, "
              f"Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Time={epoch_time:.1f}s")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_auc': train_auc,
            'val_auc': val_auc
        })
        
        if val_auc > best_auc:
            best_auc = val_auc
        
        scheduler.step()
    
    return history, best_auc


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("Exp20: Interaction Attention 优化实验")
    print("=" * 70)
    
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    # 加载数据
    (X_dense_train, X_sparse_train, y_train,
     X_dense_valid, X_sparse_valid, y_valid,
     vocab_sizes) = load_criteo_full_features(data_dir)
    
    # 采样
    print("\nSampling data for faster training...")
    train_idx = np.random.choice(len(y_train), 500000, replace=False)
    valid_idx = np.random.choice(len(y_valid), 50000, replace=False)
    
    X_dense_train = X_dense_train[train_idx]
    X_sparse_train = X_sparse_train[train_idx]
    y_train = y_train[train_idx]
    X_dense_valid = X_dense_valid[valid_idx]
    X_sparse_valid = X_sparse_valid[valid_idx]
    y_valid = y_valid[valid_idx]
    
    print(f"Train: {len(y_train)}, Valid: {len(y_valid)}")
    
    # DataLoader
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
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=4)
    
    # 实验配置
    experiments = [
        {"name": "V1_HSTULite_interaction", "model_class": HSTULiteInteraction, "kwargs": {}},
        {"name": "V2_Transformer_interaction", "model_class": TransformerInteraction, "kwargs": {}},
        {"name": "V3_Pointwise_double_interaction", "model_class": PointwiseDoubleInteraction, "kwargs": {}},
        {"name": "V4_Transformer_double_interaction", "model_class": TransformerDoubleInteraction, "kwargs": {}},
        {"name": "V5_HSTULite_double_interaction", "model_class": HSTULiteDoubleInteraction, "kwargs": {}},
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*70}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*70}")
        
        model = exp['model_class'](vocab_sizes=vocab_sizes, **exp['kwargs'])
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params/1e6:.2f}M")
        
        history, best_auc = train_model(
            model, train_loader, valid_loader, epochs=3, name=exp['name']
        )
        
        results[exp['name']] = {
            'model': exp['name'],
            'history': history,
            'best_auc': best_auc,
            'params': params
        }
        
        print(f"\nBest Val AUC: {best_auc:.4f}")
    
    # 结果汇总
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<35} {'Best AUC':<12} {'vs Baseline'}")
    print("-" * 70)
    
    baseline_auc = results['V1_HSTULite_interaction']['best_auc']
    
    for name, res in results.items():
        diff = (res['best_auc'] - baseline_auc) * 10000
        print(f"{name:<35} {res['best_auc']:.4f}     {diff:+.1f}bp")
    
    # 保存结果
    output_dir = Path(__file__).parent
    output_file = output_dir / 'results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
