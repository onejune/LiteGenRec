#!/usr/bin/env python3
"""
Exp18: 深层网络 + Attention Residuals

目标: 验证深层网络 + AttnRes 能否突破当前 SOTA

实验设计:
├── V1: 4层 + 标准残差 (基线)
├── V2: 8层 + 标准残差
├── V3: 8层 + AttnRes
├── V4: 12层 + AttnRes
└── V5: 16层 + AttnRes

数据: 统一使用 data_loader.py (与之前实验一致)
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
# Attention Residuals 实现
# ============================================================

class AttentionResidual(nn.Module):
    """
    Attention Residuals (AttnRes)
    
    核心思想: 用 Softmax 注意力替代固定残差连接
    h_l = Σ α_{i→l} · v_i
    """
    def __init__(self, d_model, num_blocks=4):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        
        # 伪查询向量 (每个块边界一个)
        self.query_vectors = nn.Parameter(torch.zeros(num_blocks, d_model))
        nn.init.normal_(self.query_vectors, std=0.02)
        
    def forward(self, block_outputs: list, current_hidden):
        """
        block_outputs: 已完成的块摘要 [[B, T, D], ...]
        current_hidden: 当前累积 [B, T, D]
        """
        if len(block_outputs) == 0:
            return current_hidden
        
        # 拼接所有值
        V = torch.stack(block_outputs + [current_hidden])  # [N+1, B, T, D]
        
        # 归一化作键
        K = F.normalize(V, dim=-1)
        
        # 当前块的查询向量
        query_idx = min(len(block_outputs), self.num_blocks - 1)
        query = F.normalize(self.query_vectors[query_idx], dim=0)
        
        # 计算注意力分数
        logits = torch.einsum('d, n b t d -> n b t', query, K)
        
        # Softmax 加权
        weights = F.softmax(logits, dim=0)
        
        # 聚合
        output = torch.einsum('n b t, n b t d -> b t d', weights, V)
        
        return output


# ============================================================
# Transformer Layer
# ============================================================

class TransformerLayer(nn.Module):
    """标准 Transformer 层 (无位置编码)"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x


# ============================================================
# Deep Transformer with AttnRes
# ============================================================

class DeepTransformerWithAttnRes(nn.Module):
    """深层 Transformer + Attention Residuals"""
    def __init__(
        self,
        num_dense: int = 13,
        num_sparse: int = 26,
        vocab_sizes: list = None,
        d_model: int = 64,
        n_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_blocks: int = 4,
        use_attn_res: bool = True
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_blocks = min(num_blocks, num_layers)
        self.use_attn_res = use_attn_res
        self.d_model = d_model
        
        # 稠密特征嵌入
        self.dense_embed = nn.Linear(num_dense, num_dense * d_model)
        
        # 稀疏特征嵌入
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # Transformer 层
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Attention Residuals (可选)
        if use_attn_res:
            self.attn_res = AttentionResidual(d_model, self.num_blocks)
        else:
            self.attn_res = None
        
        # 输出头
        total_features = num_dense + num_sparse
        self.output = nn.Sequential(
            nn.Linear(total_features * d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1)
        )
        
    def forward(self, dense_x, sparse_x):
        batch_size = dense_x.size(0)
        
        # 稠密特征嵌入
        dense_emb = self.dense_embed(dense_x).view(batch_size, 13, self.d_model)
        
        # 稀疏特征嵌入
        sparse_embs = []
        for i in range(26):
            emb = self.sparse_embeds[i](sparse_x[:, i])
            sparse_embs.append(emb)
        sparse_emb = torch.stack(sparse_embs, dim=1)
        
        # 拼接
        x = torch.cat([dense_emb, sparse_emb], dim=1)  # [B, 39, D]
        
        # 块大小
        block_size = self.num_layers // self.num_blocks if self.use_attn_res else self.num_layers
        
        # 存储块摘要
        block_outputs = [x]  # 初始嵌入
        
        # 逐层处理
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # 块边界: 应用 AttnRes + 存储摘要
            if self.use_attn_res and (i + 1) % block_size == 0 and i < self.num_layers - 1:
                x = self.attn_res(block_outputs, x)
                block_outputs.append(x)
        
        # 输出
        x = x.reshape(batch_size, -1)
        return self.output(x).squeeze(-1)


# ============================================================
# 训练函数
# ============================================================

def train_model(model, train_loader, val_loader, epochs=3, lr=1e-3):
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
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
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
        
        print(f"Epoch {epoch+1}/{epochs}: Loss={train_loss/len(train_loader):.4f}, "
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
    print("Exp18: 深层网络 + Attention Residuals")
    print("=" * 70)
    
    # 数据路径
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    # 加载数据
    (X_dense_train, X_sparse_train, y_train,
     X_dense_valid, X_sparse_valid, y_valid,
     vocab_sizes) = load_criteo_full_features(data_dir)
    
    print(f"\nTrain: {len(y_train)}, Valid: {len(y_valid)}")
    print(f"Click rate: Train={y_train.mean():.4f}, Valid={y_valid.mean():.4f}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=4)
    
    # 实验配置
    experiments = [
        {"name": "V1_4layers_baseline", "num_layers": 4, "use_attn_res": False},
        {"name": "V2_8layers_no_res", "num_layers": 8, "use_attn_res": False},
        {"name": "V3_8layers_attnres", "num_layers": 8, "use_attn_res": True, "num_blocks": 4},
        {"name": "V4_12layers_attnres", "num_layers": 12, "use_attn_res": True, "num_blocks": 4},
        {"name": "V5_16layers_attnres", "num_layers": 16, "use_attn_res": True, "num_blocks": 4},
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\n{'='*70}")
        print(f"Experiment: {exp['name']}")
        print(f"Layers: {exp['num_layers']}, AttnRes: {exp['use_attn_res']}")
        print(f"{'='*70}")
        
        model = DeepTransformerWithAttnRes(
            num_dense=13,
            num_sparse=26,
            vocab_sizes=vocab_sizes,
            d_model=64,
            n_heads=8,
            num_layers=exp['num_layers'],
            d_ff=256,
            dropout=0.1,
            num_blocks=exp.get('num_blocks', 4),
            use_attn_res=exp['use_attn_res']
        )
        
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params/1e6:.2f}M")
        
        history, best_auc = train_model(model, train_loader, valid_loader, epochs=3)
        
        results[exp['name']] = {
            'num_layers': exp['num_layers'],
            'use_attn_res': exp['use_attn_res'],
            'best_auc': best_auc,
            'history': history,
            'params': params
        }
        
        print(f"\nBest Val AUC: {best_auc:.4f}")
    
    # 结果汇总
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Layers':<8} {'AttnRes':<10} {'Best AUC':<12} {'vs Baseline'}")
    print("-" * 70)
    
    baseline_auc = results['V1_4layers_baseline']['best_auc']
    
    for name, res in results.items():
        diff = (res['best_auc'] - baseline_auc) * 10000
        attn_res = "Yes" if res['use_attn_res'] else "No"
        print(f"{name:<30} {res['num_layers']:<8} {attn_res:<10} {res['best_auc']:.4f}     {diff:+.1f}bp")
    
    # 保存结果
    output_dir = Path(__file__).parent
    output_file = output_dir / 'results.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
