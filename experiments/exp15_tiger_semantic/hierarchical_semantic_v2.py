#!/usr/bin/env python3
"""
Exp15 V2: Hierarchical Semantic CTR with Dense+Sparse Features

修复版本: 使用完整的39维特征
"""

import sys
sys.path.append('/mnt/workspace/walter.wan/git_project/github_onejune/LiteGenRec/experiments')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from data_loader import load_criteo_full_features
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class HierarchicalFeatureEncoder(nn.Module):
    def __init__(self, embed_dim, num_levels=3, num_clusters=[256, 128, 64]):
        super().__init__()
        self.num_levels = num_levels
        
        self.encoders = nn.ModuleList()
        self.codebooks = nn.ModuleList()
        
        for i, num_c in enumerate(num_clusters[:num_levels]):
            self.encoders.append(nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            ))
            self.codebooks.append(nn.Embedding(num_c, embed_dim))
        
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_levels, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x):
        hierarchy_outputs = []
        
        for i in range(self.num_levels):
            h = self.encoders[i](x)
            codebook_weights = self.codebooks[i].weight
            dists = torch.cdist(h, codebook_weights)
            weights = F.softmax(-dists, dim=-1)
            quantized = torch.matmul(weights, codebook_weights)
            hierarchy_outputs.append(quantized)
            x = h + quantized
        
        all_levels = torch.cat(hierarchy_outputs, dim=-1)
        output = self.fusion(all_levels)
        
        return output + x


class HierarchicalCTRModel(nn.Module):
    def __init__(
        self,
        num_dense_features: int,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_levels: int = 3,
        num_clusters: list = [256, 128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_dense_features = num_dense_features
        self.num_sparse_features = num_sparse_features
        
        # 稠密特征嵌入
        self.dense_embed = nn.Linear(num_dense_features, num_dense_features * embed_dim)
        
        # 稀疏特征嵌入
        self.sparse_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # 层次化语义编码器
        self.hierarchical_encoder = HierarchicalFeatureEncoder(
            embed_dim, embed_dim, num_levels, num_clusters
        )
        
        # 特征交互层
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
    
    def forward(self, dense_x: torch.Tensor, sparse_x: torch.Tensor) -> torch.Tensor:
        batch_size = dense_x.size(0)
        
        # 稠密特征嵌入
        dense_emb = self.dense_embed(dense_x).view(batch_size, self.num_dense_features, -1)
        
        # 稀疏特征嵌入 + 层次化增强
        sparse_embs = []
        for i in range(self.num_sparse_features):
            field_vals = sparse_x[:, i].long().clamp(0, self.sparse_embeddings[i].num_embeddings - 1)
            emb = self.sparse_embeddings[i](field_vals)
            enhanced_emb = self.hierarchical_encoder(emb)
            sparse_embs.append(enhanced_emb)
        
        sparse_emb = torch.stack(sparse_embs, dim=1)
        
        # 合并
        seq = torch.cat([dense_emb, sparse_emb], dim=1)
        
        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)
        
        # 特征交互
        attn_out, _ = self.interaction(seq, seq, seq)
        seq = self.norm(seq + attn_out)
        
        # 预测
        logits = self.mlp(seq[:, 0, :]).squeeze(-1)
        
        return logits


def train_model(model, train_loader, valid_loader, epochs=2, lr=1e-3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_auc = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (dense_x, sparse_x, y) in enumerate(train_loader):
            dense_x = dense_x.to(device)
            sparse_x = sparse_x.to(device)
            y = y.to(device).float()
            
            optimizer.zero_grad()
            logits = model(dense_x, sparse_x)
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
            for dense_x, sparse_x, y in valid_loader:
                dense_x = dense_x.to(device)
                sparse_x = sparse_x.to(device)
                logits = model(dense_x, sparse_x)
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
    print("Exp15 V2: Hierarchical Semantic CTR with Dense+Sparse")
    print("修复版本: 39维特征")
    print("=" * 60)
    
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    (X_dense_train, X_sparse_train, y_train,
     X_dense_valid, X_sparse_valid, y_valid,
     vocab_sizes) = load_criteo_full_features(data_dir)
    
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
    
    model = HierarchicalCTRModel(
        num_dense_features=13,
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=64,
        num_heads=8,
        num_levels=3,
        num_clusters=[256, 128, 64],
        dropout=0.1
    ).to(device)
    
    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=2, lr=1e-3)
    
    print("\n" + "=" * 60)
    print("实验结果")
    print("=" * 60)
    print(f"Hierarchical Semantic V2 AUC: {best_auc:.4f}")
    print(f"V2 基线 (Transformer): 0.7678")
    
    if best_auc > 0.7678:
        print(f"✅ 层次化语义 优于 V2 by +{(best_auc - 0.7678) * 100:.2f}bp")
    else:
        print(f"❌ 层次化语义 低于 V2 by {(best_auc - 0.7678) * 100:.2f}bp")
    
    torch.save(model.state_dict(), 'hierarchical_semantic_v2_model.pth')
    
    return best_auc


if __name__ == "__main__":
    main()
