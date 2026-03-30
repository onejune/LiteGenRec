#!/usr/bin/env python3
"""
Exp06: Semantic ID GenCTR V2 - 修复版
核心思想: 用VQ-VAE将特征编码为离散语义ID，然后用Transformer建模
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


class VectorQuantizer(nn.Module):
    """向量量化模块"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, x: torch.Tensor):
        """
        x: [batch, dim] -> 返回 quantized, index (标量), loss
        """
        # 计算与所有码本向量的距离
        distances = (
            x.pow(2).sum(dim=-1, keepdim=True)
            - 2 * x @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=-1)
        )  # [batch, num_embeddings]
        
        # 选择最近的码本向量 - 返回标量索引
        indices = distances.argmin(dim=-1)  # [batch]
        quantized = self.embeddings(indices)  # [batch, dim]
        
        # VQ损失
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, indices, loss


class SemanticIDGenCTR(nn.Module):
    """基于语义ID的生成式CTR模型 - 简化版"""
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_codebooks: int = 4,
        codebook_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        
        # 特征嵌入
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # 特征压缩层 - 将所有特征嵌入压缩到hidden_dim
        total_embed_dim = num_sparse_features * embed_dim
        self.feature_compress = nn.Sequential(
            nn.Linear(total_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 多个VQ模块 - 每个产生一个语义ID
        self.vq_projectors = nn.ModuleList([
            nn.Linear(hidden_dim, embed_dim) for _ in range(num_codebooks)
        ])
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, embed_dim) for _ in range(num_codebooks)
        ])
        
        # 语义ID嵌入 - 用于Transformer输入
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim) for _ in range(num_codebooks)
        ])
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CTR预测头
        self.ctr_head = nn.Sequential(
            nn.Linear(embed_dim * num_codebooks, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x: torch.Tensor):
        """
        x: [batch, num_features] 稀疏特征索引
        返回: logits, vq_loss
        """
        batch_size = x.size(0)
        
        # 1. 特征嵌入
        embeds = []
        for i in range(self.num_sparse_features):
            field_vals = x[:, i].long().clamp(0, self.embeddings[i].num_embeddings - 1)
            embeds.append(self.embeddings[i](field_vals))  # [batch, embed_dim]
        
        # 拼接所有特征嵌入
        concat_embed = torch.cat(embeds, dim=-1)  # [batch, num_features * embed_dim]
        
        # 2. 压缩特征
        compressed = self.feature_compress(concat_embed)  # [batch, hidden_dim]
        
        # 3. 多个VQ编码 - 获取多个语义ID
        semantic_ids = []
        total_vq_loss = 0
        
        for i in range(self.num_codebooks):
            z = self.vq_projectors[i](compressed)  # [batch, embed_dim]
            _, indices, vq_loss = self.quantizers[i](z)  # indices: [batch]
            semantic_ids.append(indices)
            total_vq_loss = total_vq_loss + vq_loss
        
        # 4. 语义ID嵌入
        semantic_embeds = []
        for i, ids in enumerate(semantic_ids):
            emb = self.semantic_embeddings[i](ids.long())  # [batch, embed_dim]
            semantic_embeds.append(emb)
        
        # 5. 堆叠为序列 [batch, num_codebooks, embed_dim]
        semantic_seq = torch.stack(semantic_embeds, dim=1)
        
        # 6. Transformer编码
        encoded = self.transformer(semantic_seq)  # [batch, num_codebooks, embed_dim]
        
        # 7. 展平并预测CTR
        encoded_flat = encoded.reshape(batch_size, -1)  # [batch, num_codebooks * embed_dim]
        logits = self.ctr_head(encoded_flat).squeeze(-1)  # [batch]
        
        return logits, total_vq_loss


def load_criteo_data(data_dir: str):
    """加载Criteo数据"""
    print("Loading Criteo dataset from parquet...")
    
    train_path = f"{data_dir}/train_train.parquet"
    valid_path = f"{data_dir}/train_valid.parquet"
    
    # 加载训练数据
    train_files = sorted([f for f in os.listdir(train_path) if f.endswith('.parquet')])[:2]
    train_dfs = [pq.read_table(os.path.join(train_path, f)).to_pandas() for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # 加载验证数据
    valid_files = sorted([f for f in os.listdir(valid_path) if f.endswith('.parquet')])[:1]
    valid_dfs = [pq.read_table(os.path.join(valid_path, f)).to_pandas() for f in valid_files]
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    
    print(f"Loaded train: {len(train_df)}, valid: {len(valid_df)}")
    
    # 稀疏特征列
    sparse_cols = [f'C{i}' for i in range(1, 27)]
    
    # 合并编码
    all_df = pd.concat([train_df, valid_df], ignore_index=True)
    
    vocab_sizes = []
    encoders = {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('__MISSING__')
        encoders[col] = LabelEncoder()
        all_df[col] = encoders[col].fit_transform(all_df[col].astype(str)) + 1
        vocab_sizes.append(int(all_df[col].max()) + 1)
    
    # 分回
    train_df = all_df.iloc[:len(train_df)]
    valid_df = all_df.iloc[len(train_df):]
    
    X_train = train_df[sparse_cols].values
    X_valid = valid_df[sparse_cols].values
    y_train = train_df['label'].values
    y_valid = valid_df['label'].values
    
    print(f"Vocab sizes (first 5): {vocab_sizes[:5]}")
    
    return X_train, X_valid, y_train, y_valid, vocab_sizes


def train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3, vq_weight=0.1):
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            
            optimizer.zero_grad()
            logits, vq_loss = model(x)
            
            ctr_loss = criterion(logits, y)
            loss = ctr_loss + vq_weight * vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        
        # 验证
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                logits, _ = model(x)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Valid AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_semantic_id_model.pt')
    
    return best_auc


def main():
    print("=" * 60)
    print("Exp06: Semantic ID GenCTR V2")
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
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=2048, shuffle=False)
    
    model = SemanticIDGenCTR(
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=32,
        hidden_dim=128,
        num_codebooks=4,
        codebook_size=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=3, lr=1e-3, vq_weight=0.1)
    
    print("\n" + "=" * 60)
    print(f"Best Valid AUC: {best_auc:.4f}")
    print("=" * 60)
    
    with open('results.txt', 'w') as f:
        f.write(f"Semantic ID GenCTR V2 Results\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")
    
    return best_auc


if __name__ == "__main__":
    main()
