#!/usr/bin/env python3
"""
Exp08: Gated Fusion GenCTR
核心思想: 门控融合多种特征交互方式，参考 FiBiNET / GDCN
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


class SENetLayer(nn.Module):
    """Squeeze-Excitation 特征重要性学习"""
    
    def __init__(self, num_fields: int, reduction_ratio: int = 4):
        super().__init__()
        reduced_dim = max(1, num_fields // reduction_ratio)
        self.fc = nn.Sequential(
            nn.Linear(num_fields, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, num_fields),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_fields, embed_dim]"""
        # 全局平均池化
        pooled = x.mean(dim=-1)  # [batch, num_fields]
        weights = self.fc(pooled).unsqueeze(-1)  # [batch, num_fields, 1]
        return x * weights


class BilinearInteraction(nn.Module):
    """双线性特征交互"""
    
    def __init__(self, embed_dim: int, num_fields: int, bilinear_type: str = 'field_all'):
        super().__init__()
        self.bilinear_type = bilinear_type
        
        if bilinear_type == 'field_all':
            self.W = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        elif bilinear_type == 'field_each':
            self.W = nn.ParameterList([
                nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
                for _ in range(num_fields)
            ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, num_fields, embed_dim]"""
        batch_size, num_fields, embed_dim = x.shape
        
        interactions = []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                if self.bilinear_type == 'field_all':
                    vi_W = torch.matmul(x[:, i, :], self.W)
                else:
                    vi_W = torch.matmul(x[:, i, :], self.W[i])
                interaction = vi_W * x[:, j, :]  # [batch, embed_dim]
                interactions.append(interaction)
        
        # [batch, num_interactions, embed_dim]
        return torch.stack(interactions, dim=1)


class GatedFusionGenCTR(nn.Module):
    """门控融合生成式CTR模型"""
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: list,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.embed_dim = embed_dim
        
        # 特征嵌入
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # SE-Net 特征重要性
        self.senet = SENetLayer(num_sparse_features)
        
        # 双线性交互
        self.bilinear = BilinearInteraction(embed_dim, num_sparse_features, 'field_all')
        num_interactions = num_sparse_features * (num_sparse_features - 1) // 2
        
        # 深度网络
        dnn_input_dim = num_sparse_features * embed_dim + num_interactions * embed_dim
        self.dnn = nn.Sequential(
            nn.Linear(dnn_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 门控融合
        gate_input_dim = num_sparse_features * embed_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim // 2),
            nn.Sigmoid()
        )
        
        # 输出层
        self.output = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 1. 特征嵌入
        embeds = []
        for i in range(self.num_sparse_features):
            field_vals = x[:, i].long().clamp(0, self.embeddings[i].num_embeddings - 1)
            embeds.append(self.embeddings[i](field_vals))
        
        embed_stack = torch.stack(embeds, dim=1)  # [batch, num_fields, embed_dim]
        
        # 2. SE-Net 加权
        senet_out = self.senet(embed_stack)  # [batch, num_fields, embed_dim]
        senet_flat = senet_out.reshape(batch_size, -1)  # [batch, num_fields * embed_dim]
        
        # 3. 双线性交互
        bilinear_out = self.bilinear(embed_stack)  # [batch, num_interactions, embed_dim]
        bilinear_flat = bilinear_out.reshape(batch_size, -1)
        
        # 4. 拼接并通过DNN
        dnn_input = torch.cat([senet_flat, bilinear_flat], dim=-1)
        dnn_out = self.dnn(dnn_input)  # [batch, hidden_dim // 2]
        
        # 5. 门控融合
        gate_input = embed_stack.reshape(batch_size, -1)
        gate_weight = self.gate(gate_input)  # [batch, hidden_dim // 2]
        
        gated_out = dnn_out * gate_weight
        
        # 6. 输出
        logits = self.output(gated_out).squeeze(-1)
        
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


def train_model(model, train_loader, valid_loader, epochs=5, lr=5e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
    
    best_auc = 0
    patience_counter = 0
    
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
        scheduler.step(auc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Valid AUC: {auc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_gated_fusion.pt')
        else:
            patience_counter += 1
            if patience_counter >= 3:
                print("Early stopping!")
                break
    
    return best_auc


def main():
    print("=" * 60)
    print("Exp08: Gated Fusion GenCTR (SENet + Bilinear + Gate)")
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
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=4096, shuffle=False, num_workers=4)
    
    model = GatedFusionGenCTR(
        num_sparse_features=26,
        vocab_sizes=vocab_sizes,
        embed_dim=32,
        hidden_dim=128,
        dropout=0.2
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_auc = train_model(model, train_loader, valid_loader, epochs=5, lr=5e-4)
    
    print("\n" + "=" * 60)
    print(f"Best Valid AUC: {best_auc:.4f}")
    print("=" * 60)
    
    print("\n对比:")
    print(f"  DeepFM baseline: 0.7472")
    print(f"  LiteGenRec V1:   0.7663")
    print(f"  LiteGenRec V2:   0.7678")
    print(f"  Exp08 (本实验):  {best_auc:.4f}")
    
    with open('results.txt', 'w') as f:
        f.write(f"Gated Fusion GenCTR Results\n")
        f.write(f"Best AUC: {best_auc:.4f}\n")
    
    return best_auc


if __name__ == "__main__":
    main()
