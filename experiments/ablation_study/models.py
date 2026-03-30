#!/usr/bin/env python3
"""
消融实验：对比 exp16 (Transformer 无 PE)、exp03 (Transformer 有 PE)、exp13 (HSTU-Lite V3)

研究问题:
1. 位置编码的影响 (exp03 vs exp16)
2. 架构差异 (Transformer vs HSTU-Lite)
3. 特征交互方式 (Multi-Head Attn vs Pointwise Attn)

配置统一:
- 完整特征 (39 维 = 26 稀疏 + 13 稠密)
- embed_dim=64, num_heads=8, num_layers=4
- epochs=3, batch_size=2048, lr=1e-3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
import pandas as pd
import numpy as np
import time
import os
import glob
from pathlib import Path
import argparse

# 导入 GPU 路由器
from gpu_router import select_gpu_for_training


# ============================================================
# 配置
# ============================================================

class Config:
    # 特征配置
    num_dense_features = 13
    num_sparse_features = 26
    total_features = 39
    
    # 模型配置 (统一)
    embed_dim = 64
    num_heads = 8
    num_layers = 4
    hidden_dim = 256
    dropout = 0.1
    
    # 词表大小
    vocab_size_dense = 50
    vocab_size_sparse = 50000
    
    # 训练配置
    epochs = 3
    batch_size = 2048
    learning_rate = 1e-3
    weight_decay = 0.01
    
    # 数据路径
    train_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet/train_train.parquet/'
    val_path = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet/train_valid.parquet/'

config = Config()


# ============================================================
# 数据集
# ============================================================

class CriteoDataset(Dataset):
    def __init__(self, data_path, max_files=None, dense_encoder=None, sparse_encoders=None, fit=False):
        print(f"Loading {data_path}...")
        start = time.time()
        
        # 加载分片 parquet
        parquet_files = sorted(glob.glob(f"{data_path}/part.*.parquet"))
        if max_files:
            parquet_files = parquet_files[:max_files]
        
        dfs = []
        for pf in parquet_files:
            dfs.append(pd.read_parquet(pf))
        df = pd.concat(dfs, ignore_index=True)
        
        self.labels = df['label'].values.astype(np.float32)
        
        # 稠密特征
        dense_cols = [f"I{i}" for i in range(1, 14)]
        dense_data = df[dense_cols].values.copy()
        dense_data = np.nan_to_num(dense_data, nan=0.0)
        
        if fit:
            self.dense_encoder = KBinsDiscretizer(
                n_bins=config.vocab_size_dense,
                encode='ordinal',
                strategy='quantile'
            )
            dense_encoded = self.dense_encoder.fit_transform(dense_data)
            dense_encoded = np.clip(dense_encoded, 0, config.vocab_size_dense - 1)
            
            self.sparse_encoders = {}
            sparse_encoded = np.zeros((len(df), config.num_sparse_features), dtype=np.int64)
            
            for i, col in enumerate([f"C{i}" for i in range(1, 27)]):
                le = LabelEncoder()
                df[col] = df[col].fillna('-1')
                # 限制词表大小
                value_counts = df[col].value_counts()
                rare_values = value_counts[value_counts < 5].index
                df.loc[df[col].isin(rare_values), col] = '-1'
                sparse_encoded[:, i] = le.fit_transform(df[col].astype(str))
                self.sparse_encoders[col] = le
                sparse_encoded[:, i] = np.clip(sparse_encoded[:, i], 0, config.vocab_size_sparse - 1)
        else:
            self.dense_encoder = dense_encoder
            self.sparse_encoders = sparse_encoders
            dense_encoded = self.dense_encoder.transform(dense_data)
            dense_encoded = np.clip(dense_encoded, 0, config.vocab_size_dense - 1)
            
            sparse_encoded = np.zeros((len(df), config.num_sparse_features), dtype=np.int64)
            for i, col in enumerate([f"C{i}" for i in range(1, 27)]):
                df[col] = df[col].fillna('-1')
                known_classes = set(self.sparse_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if str(x) in known_classes else '-1')
                sparse_encoded[:, i] = self.sparse_encoders[col].transform(df[col].astype(str))
                sparse_encoded[:, i] = np.clip(sparse_encoded[:, i], 0, config.vocab_size_sparse - 1)
        
        self.dense_features = dense_encoded.astype(np.int64)
        self.sparse_features = sparse_encoded
        
        print(f"  Loaded {len(df)} samples in {time.time() - start:.2f}s")
        print(f"  Click rate: {self.labels.mean():.4f}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.dense_features[idx], dtype=torch.long),
            torch.tensor(self.sparse_features[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )


# ============================================================
# 模型定义
# ============================================================

# --- Model 1: Transformer with Position Encoding (exp03) ---

class TransformerWithPE(nn.Module):
    """exp03: Transformer + 位置编码"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dense_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_dense, config.embed_dim)
            for _ in range(config.num_dense_features)
        ])
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_sparse, config.embed_dim)
            for _ in range(config.num_sparse_features)
        ])
        
        # 位置编码 (exp03 的关键特征)
        self.pos_encoding = nn.Parameter(torch.randn(config.total_features, config.embed_dim))
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(config.embed_dim * config.total_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dense_feats, sparse_feats):
        batch_size = dense_feats.size(0)
        
        dense_embs = [self.dense_embeds[i](dense_feats[:, i]) for i in range(13)]
        sparse_embs = [self.sparse_embeds[i](sparse_feats[:, i]) for i in range(26)]
        all_embs = torch.stack(dense_embs + sparse_embs, dim=1)
        
        # 加位置编码
        all_embs = self.norm(all_embs) + self.pos_encoding
        
        out = self.transformer(all_embs)
        out = out.reshape(batch_size, -1)
        return self.output(out).squeeze(-1)


# --- Model 2: Transformer without Position Encoding (exp16) ---

class TransformerWithoutPE(nn.Module):
    """exp16: Transformer - 无位置编码 (SOTA)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dense_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_dense, config.embed_dim)
            for _ in range(config.num_dense_features)
        ])
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_sparse, config.embed_dim)
            for _ in range(config.num_sparse_features)
        ])
        
        # 无位置编码！
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(config.embed_dim * config.total_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dense_feats, sparse_feats):
        batch_size = dense_feats.size(0)
        
        dense_embs = [self.dense_embeds[i](dense_feats[:, i]) for i in range(13)]
        sparse_embs = [self.sparse_embeds[i](sparse_feats[:, i]) for i in range(26)]
        all_embs = torch.stack(dense_embs + sparse_embs, dim=1)
        
        # 不加位置编码
        all_embs = self.norm(all_embs)
        
        out = self.transformer(all_embs)
        out = out.reshape(batch_size, -1)
        return self.output(out).squeeze(-1)


# --- Model 3: HSTU-Lite V3 (Pointwise Attention) ---

class PointwiseAttention(nn.Module):
    """点积注意力 - 用 sigmoid 替代 softmax"""
    
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
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.sigmoid(attn)  # ← 关键：sigmoid 替代 softmax
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.o_proj(out)


class HSTUBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = PointwiseAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class HSTULiteV3(nn.Module):
    """exp13 V3: HSTU-Lite with Pointwise Attention"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dense_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_dense, config.embed_dim)
            for _ in range(config.num_dense_features)
        ])
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_sparse, config.embed_dim)
            for _ in range(config.num_sparse_features)
        ])
        
        self.hstu_layers = nn.ModuleList([
            HSTUBlock(config.embed_dim, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(config.embed_dim * config.total_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dense_feats, sparse_feats):
        batch_size = dense_feats.size(0)
        
        dense_embs = [self.dense_embeds[i](dense_feats[:, i]) for i in range(13)]
        sparse_embs = [self.sparse_embeds[i](sparse_feats[:, i]) for i in range(26)]
        x = torch.stack(dense_embs + sparse_embs, dim=1)
        
        for layer in self.hstu_layers:
            x = layer(x)
        
        x = x.reshape(batch_size, -1)
        return self.output(x).squeeze(-1)


# ============================================================
# 训练函数
# ============================================================

def train_model(model_name, model, train_loader, val_loader, device, epochs=3):
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCELoss()
    
    best_val_auc = 0.0
    history = []
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, (dense_feats, sparse_feats, labels) in enumerate(train_loader):
            dense_feats, sparse_feats, labels = dense_feats.to(device), sparse_feats.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(dense_feats, sparse_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_auc = roc_auc_score(train_labels, train_preds)
        train_logloss = log_loss(train_labels, train_preds)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for dense_feats, sparse_feats, labels in val_loader:
                dense_feats, sparse_feats, labels = dense_feats.to(device), sparse_feats.to(device), labels.to(device)
                outputs = model(dense_feats, sparse_feats)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        val_logloss = log_loss(val_labels, val_preds)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch:2d}: Loss={train_loss/len(train_loader):.4f}, Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Time={epoch_time:.1f}s")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss / len(train_loader),
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_logloss': val_logloss
        })
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        scheduler.step()
    
    return history, best_val_auc


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LiteGenRec Ablation Study")
    parser.add_argument('--max-files', type=int, default=None, help='最大加载数据文件数 (None=全部)')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2048, help='批次大小')
    parser.add_argument('--complexity', type=str, default='medium', choices=['low', 'medium', 'high'], help='任务复杂度')
    args = parser.parse_args()
    
    # 更新配置
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    
    print("="*60)
    print("LiteGenRec Ablation Study: exp03 vs exp16 vs exp13")
    print("="*60)
    
    # GPU 自动选择
    device = select_gpu_for_training(
        model_params_mb=100,  # 估计模型大小
        batch_size=config.batch_size,
        complexity=args.complexity
    )
    print(f"Using device: {device}\n")
    
    # 准备数据
    print("Preparing data...")
    train_dataset = CriteoDataset(config.train_path, max_files=args.max_files, fit=True)
    val_dataset = CriteoDataset(config.val_path, max_files=args.max_files, 
                                 dense_encoder=train_dataset.dense_encoder, 
                                 sparse_encoders=train_dataset.sparse_encoders)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    results = {}
    
    # 训练三个模型
    models = [
        ("Transformer+PE (exp03)", TransformerWithPE(config)),
        ("Transformer-NoPE (exp16)", TransformerWithoutPE(config)),
        ("HSTU-Lite V3 (exp13)", HSTULiteV3(config)),
    ]
    
    for model_name, model in models:
        params = sum(p.numel() for p in model.parameters())
        print(f"\n{model_name}: {params/1e6:.1f}M parameters")
        
        history, best_auc = train_model(model_name, model, train_loader, val_loader, device, config.epochs)
        results[model_name] = {'history': history, 'best_auc': best_auc}
    
    # 输出结果汇总
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<30} {'Best Val AUC':<15} {'vs Baseline'}")
    print("-"*60)
    
    baseline_auc = 0.7472  # DeepFM
    for model_name, res in results.items():
        auc = res['best_auc']
        diff = (auc - baseline_auc) * 10000  # 千分点
        print(f"{model_name:<30} {auc:.4f}          +{diff:.1f}bp")
    
    # 保存结果
    output_dir = Path(__file__).parent
    result_file = output_dir / 'RESULTS.md'
    
    with open(result_file, 'w') as f:
        f.write("# Ablation Study Results\n\n")
        f.write(f"> Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Embed Dim: {config.embed_dim}\n")
        f.write(f"- Num Heads: {config.num_heads}\n")
        f.write(f"- Num Layers: {config.num_layers}\n")
        f.write(f"- Epochs: {config.epochs}\n")
        f.write(f"- Batch Size: {config.batch_size}\n")
        f.write(f"- Train Samples: {len(train_dataset)}\n")
        f.write(f"- Val Samples: {len(val_dataset)}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Model | Best Val AUC | vs DeepFM |\n")
        f.write("|-------|--------------|-----------|\n")
        for model_name, res in results.items():
            auc = res['best_auc']
            diff = (auc - baseline_auc) * 10000
            f.write(f"| {model_name} | {auc:.4f} | +{diff:.1f}bp |\n")
        
        f.write("\n## Training History\n\n")
        for model_name, res in results.items():
            f.write(f"### {model_name}\n\n")
            f.write("| Epoch | Train Loss | Train AUC | Val AUC |\n")
            f.write("|-------|------------|-----------|---------|\n")
            for h in res['history']:
                f.write(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_auc']:.4f} | {h['val_auc']:.4f} |\n")
            f.write("\n")
    
    print(f"\nResults saved to: {result_file}")

if __name__ == '__main__':
    main()
