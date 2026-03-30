#!/usr/bin/env python3
"""
Exp15: TIGER-inspired Semantic ID for CTR

基于 TIGER (NeurIPS 2023) 的思路，但针对 CTR 预测场景简化
不再生成 Semantic ID 序列，而是用层次化语义表示增强特征

核心思想: 
- 层次化特征表示 (多粒度语义)
- 保留 VQ-VAE 的层次化思想，但用于特征增强而非 ID 生成
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


class HierarchicalFeatureEncoder(nn.Module):
    """
    层次化特征编码器
    
    将特征编码为多个层次的表示:
    - Level 1: 粗粒度语义
    - Level 2: 中粒度语义
    - Level 3: 细粒度语义
    """
    
    def __init__(self, input_dim, embed_dim, num_levels=3, num_clusters=[256, 128, 64]):
        super().__init__()
        self.num_levels = num_levels
        
        # 每层的编码器
        self.encoders = nn.ModuleList()
        self.codebooks = nn.ModuleList()
        
        for i, num_c in enumerate(num_clusters[:num_levels]):
            # 编码器
            self.encoders.append(nn.Sequential(
                nn.Linear(input_dim if i == 0 else embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ))
            
            # 码本 (可学习的聚类中心)
            self.codebooks.append(nn.Embedding(num_c, embed_dim))
        
        # 层次化融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_levels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        """
        x: [batch, embed_dim] 单个特征的 embedding
        Returns: [batch, embed_dim] 层次化增强的表示
        """
        hierarchy_outputs = []
        
        for i in range(self.num_levels):
            # 编码
            h = self.encoders[i](x)
            
            # 量化 (软分配)
            # 计算与码本的距离
            codebook_weights = self.codebooks[i].weight  # [num_clusters, embed_dim]
            dists = torch.cdist(h, codebook_weights)  # [batch, num_clusters]
            weights = F.softmax(-dists, dim=-1)  # 概率分配
            
            # 加权聚合码本向量
            quantized = torch.matmul(weights, codebook_weights)  # [batch, embed_dim]
            
            hierarchy_outputs.append(quantized)
            
            # 残差连接到下一层
            x = h + quantized
        
        # 融合所有层次
        all_levels = torch.cat(hierarchy_outputs, dim=-1)
        output = self.fusion(all_levels)
        
        return output + x  # 残差


class HierarchicalCTRModel(nn.Module):
    """
    层次化语义增强的 CTR 模型
    
    流程:
    1. 特征嵌入
    2. 层次化语义增强
    3. 特征交互 (Transformer)
    4. 预测
    """
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_levels: int = 3,
        num_clusters: list = [256, 128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        
        # 基础嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # 层次化语义编码器 (共享)
        self.hierarchical_encoder = HierarchicalFeatureEncoder(
            embed_dim, embed_dim, num_levels, num_clusters
        )
        
        # 特征交互层 (简化的 Transformer)
        self.interaction = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
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
            emb = self.embeddings[i](field_vals)
            
            # 层次化语义增强
            enhanced_emb = self.hierarchical_encoder(emb)
            embeds.append(enhanced_emb)
        
        # 序列化
        seq = torch.stack(embeds, dim=1)  # [batch, num_features, embed_dim]
        
        # 加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        
        # 特征交互
        attn_out, _ = self.interaction(seq, seq, seq)
        seq = self.norm(seq + attn_out)
        
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
    print("Exp15: TIGER-inspired Hierarchical Semantic CTR")
    print("Core: Multi-level Semantic Enhancement")
    print("=" * 60)
    
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    X_train, X_valid, y_train, y_valid, vocab_sizes = load_criteo_data(data_dir)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.tensor(y_train, dtype=torch.float))
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.long), torch.tensor(y_valid, dtype=torch.float))
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=4)
    
    model = HierarchicalCTRModel(
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=64,
        num_heads=8,
        num_levels=3,
        num_clusters=[256, 128, 64],
        dropout=0.1
    ).to(device)
    
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print("Architecture: Embedding -> Hierarchical Semantic Enhancement -> Attention -> MLP")
    print()
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3)
    
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    print(f"Hierarchical Semantic CTR AUC: {best_auc:.4f}")
    print(f"V2 (标准 Transformer): 0.7678")
    
    if best_auc > 0.7678:
        print(f"✅ 层次化语义增强 优于 V2 by +{(best_auc - 0.7678) * 100:.2f}bp")
    else:
        print(f"❌ 层次化语义增强 低于 V2 by {(best_auc - 0.7678) * 100:.2f}bp")
    
    torch.save(model.state_dict(), 'hierarchical_semantic_model.pth')
    print(f"\n模型已保存: hierarchical_semantic_model.pth")
    
    return best_auc


if __name__ == "__main__":
    main()
