#!/usr/bin/env python3
"""
LiteGenRec V4: 完整特征 + 无位置编码 + 多架构对比

对比架构:
1. Transformer (无位置编码)
2. HSTU-Lite (Pointwise Attention)
3. Mamba4CTR (State Space Model)

目标: 验证完整特征下的最优架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import roc_auc_score, log_loss
import pyarrow.parquet as pq
import time
import json
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# ============================================================
# 配置
# ============================================================

class Config:
    # 特征配置
    num_dense_features = 13      # I1-I13
    num_sparse_features = 26     # C1-C26
    total_features = 39
    
    # 模型配置
    embed_dim = 64
    num_heads = 8
    num_layers = 4
    hidden_dim = 256
    dropout = 0.1
    
    # 稠密特征离散化
    dense_bins = 50
    vocab_size_dense = 50
    vocab_size_sparse = 50000
    
    # 训练配置
    epochs = 3
    batch_size = 2048
    learning_rate = 1e-3
    weight_decay = 0.01

config = Config()


# ============================================================
# 数据处理
# ============================================================

class CriteoDataset(Dataset):
    """Criteo 数据集 - 包含稠密和稀疏特征"""
    
    def __init__(self, data_path, dense_encoder=None, sparse_encoders=None, fit=False):
        print(f"Loading {data_path}...")
        start = time.time()
        
        # 加载数据
        df = pd.read_parquet(data_path)
        
        # 提取标签
        self.labels = df['label'].values.astype(np.float32)
        
        # 处理稠密特征 I1-I13
        dense_cols = [f"I{i}" for i in range(1, 14)]
        dense_data = df[dense_cols].values.copy()
        
        # 填充 NaN 为 0
        dense_data = np.nan_to_num(dense_data, nan=0.0)
        
        if fit:
            # 使用 KBinsDiscretizer 进行离散化
            self.dense_encoder = KBinsDiscretizer(
                n_bins=config.dense_bins, 
                encode='ordinal', 
                strategy='quantile'
            )
            dense_encoded = self.dense_encoder.fit_transform(dense_data)
            # 确保索引不越界 (KBinsDiscretizer 可能产生比 n_bins 多的 bins)
            actual_bins = dense_encoded.max() + 1
            print(f"  Dense features: {config.dense_bins} bins configured, {actual_bins} actual bins")
            # 裁剪到配置的 vocab_size
            dense_encoded = np.clip(dense_encoded, 0, config.vocab_size_dense - 1)
            self.sparse_encoders = {}
            sparse_encoded = np.zeros((len(df), config.num_sparse_features), dtype=np.int64)
            
            for i, col in enumerate([f"C{i}" for i in range(1, 27)]):
                le = LabelEncoder()
                # 填充缺失值
                df[col] = df[col].fillna('-1')
                # 限制词表大小
                value_counts = df[col].value_counts()
                rare_values = value_counts[value_counts < 5].index
                df.loc[df[col].isin(rare_values), col] = '-1'
                sparse_encoded[:, i] = le.fit_transform(df[col].astype(str))
                self.sparse_encoders[col] = le
                # 确保索引不越界
                max_idx = sparse_encoded[:, i].max()
                if max_idx >= config.vocab_size_sparse:
                    print(f"  Sparse feature {col}: {max_idx+1} unique values, clipping to {config.vocab_size_sparse}")
                    sparse_encoded[:, i] = np.clip(sparse_encoded[:, i], 0, config.vocab_size_sparse - 1)
        else:
            self.dense_encoder = dense_encoder
            self.sparse_encoders = sparse_encoders
            dense_data = np.nan_to_num(dense_data, nan=0.0)
            dense_encoded = self.dense_encoder.transform(dense_data)
            # 确保索引不越界
            dense_encoded = np.clip(dense_encoded, 0, config.vocab_size_dense - 1)
            sparse_encoded = np.zeros((len(df), config.num_sparse_features), dtype=np.int64)
            
            for i, col in enumerate([f"C{i}" for i in range(1, 27)]):
                df[col] = df[col].fillna('-1')
                # 处理未见过的值
                known_classes = set(self.sparse_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if str(x) in known_classes else '-1')
                sparse_encoded[:, i] = self.sparse_encoders[col].transform(df[col].astype(str))
                # 确保索引不越界
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

# -------------------- 1. Transformer (无位置编码) --------------------

class TransformerCTR(nn.Module):
    """标准 Transformer CTR 模型 - 无位置编码"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.dense_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_dense, config.embed_dim)
            for _ in range(config.num_dense_features)
        ])
        
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_sparse, config.embed_dim)
            for _ in range(config.num_sparse_features)
        ])
        
        # Layer Norm
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 输出层
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
        
        # 嵌入
        dense_embs = [self.dense_embeds[i](dense_feats[:, i]) for i in range(13)]
        sparse_embs = [self.sparse_embeds[i](sparse_feats[:, i]) for i in range(26)]
        all_embs = torch.stack(dense_embs + sparse_embs, dim=1)  # [B, 39, D]
        
        # Layer Norm
        all_embs = self.norm(all_embs)
        
        # Transformer (无位置编码)
        out = self.transformer(all_embs)
        
        # 展平并输出
        out = out.reshape(batch_size, -1)
        return self.output(out).squeeze(-1)


# -------------------- 2. HSTU-Lite (Pointwise Attention) --------------------

class PointwiseAttention(nn.Module):
    """点积注意力 - 比 Multi-Head Attention 更轻量"""
    
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.scale = embed_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: [B, L, D]
        q = k = v = x
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, L, L]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


class HSTULite(nn.Module):
    """HSTU-Lite: 轻量级层次化序列转导单元"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.dense_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_dense, config.embed_dim)
            for _ in range(config.num_dense_features)
        ])
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_sparse, config.embed_dim)
            for _ in range(config.num_sparse_features)
        ])
        
        # HSTU Layers: Pointwise Attn + FFN
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(config.embed_dim),
                'attn': PointwiseAttention(config.embed_dim, config.dropout),
                'norm2': nn.LayerNorm(config.embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(config.embed_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.embed_dim),
                    nn.Dropout(config.dropout)
                )
            })
            for _ in range(config.num_layers)
        ])
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(config.embed_dim * config.total_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dense_feats, sparse_feats):
        batch_size = dense_feats.size(0)
        
        # 嵌入
        dense_embs = [self.dense_embeds[i](dense_feats[:, i]) for i in range(13)]
        sparse_embs = [self.sparse_embeds[i](sparse_feats[:, i]) for i in range(26)]
        x = torch.stack(dense_embs + sparse_embs, dim=1)  # [B, 39, D]
        
        # HSTU Layers
        for layer in self.layers:
            # Pointwise Attention with residual
            x = x + layer['attn'](layer['norm1'](x))
            # FFN with residual
            x = x + layer['ffn'](layer['norm2'](x))
        
        # 输出
        x = x.reshape(batch_size, -1)
        return self.output(x).squeeze(-1)


# -------------------- 3. Mamba4CTR (Simplified SSM) --------------------

class MambaBlock(nn.Module):
    """简化的 Mamba 状态空间模型块"""
    
    def __init__(self, embed_dim, state_dim=16, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        
        # 投影
        self.proj_in = nn.Linear(embed_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)
        
        # SSM 参数
        self.A = nn.Parameter(torch.randn(embed_dim, state_dim) * 0.01)
        self.B = nn.Linear(embed_dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, embed_dim, bias=False)
        self.D = nn.Parameter(torch.ones(embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: [B, L, D]
        B, L, D = x.shape
        
        # 投影
        x_proj = self.proj_in(x)
        
        # 简化的 SSM: y = SSM(x)
        # 这里用线性变换模拟状态空间
        delta = F.softplus(self.B(x_proj))  # [B, L, S]
        h = torch.zeros(B, D, self.state_dim, device=x.device)
        outputs = []
        
        for t in range(L):
            # 状态更新
            h = h * torch.sigmoid(self.A.unsqueeze(0)) + delta[:, t:t+1, :].transpose(1, 2) * x_proj[:, t:t+1, :].unsqueeze(-1)
            # 输出
            y_t = self.C(h.sum(dim=-1)) + self.D * x_proj[:, t]
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, L, D]
        y = self.proj_out(y)
        
        return self.dropout(y) + x


class Mamba4CTR(nn.Module):
    """Mamba 风格的 CTR 模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.dense_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_dense, config.embed_dim)
            for _ in range(config.num_dense_features)
        ])
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_sparse, config.embed_dim)
            for _ in range(config.num_sparse_features)
        ])
        
        # Mamba Layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': nn.LayerNorm(config.embed_dim),
                'mamba': MambaBlock(config.embed_dim, state_dim=16, dropout=config.dropout),
                'ffn': nn.Sequential(
                    nn.Linear(config.embed_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.embed_dim)
                )
            })
            for _ in range(config.num_layers)
        ])
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(config.embed_dim * config.total_features, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, dense_feats, sparse_feats):
        batch_size = dense_feats.size(0)
        
        # 嵌入
        dense_embs = [self.dense_embeds[i](dense_feats[:, i]) for i in range(13)]
        sparse_embs = [self.sparse_embeds[i](sparse_feats[:, i]) for i in range(26)]
        x = torch.stack(dense_embs + sparse_embs, dim=1)
        
        # Mamba Layers
        for layer in self.layers:
            x = layer['mamba'](layer['norm'](x))
            x = x + layer['ffn'](x)
        
        # 输出
        x = x.reshape(batch_size, -1)
        return self.output(x).squeeze(-1)


# ============================================================
# 训练与评估
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for dense, sparse, label in dataloader:
        dense, sparse, label = dense.to(device), sparse.to(device), label.to(device)
        
        optimizer.zero_grad()
        pred = model(dense, sparse)
        loss = criterion(pred, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for dense, sparse, label in dataloader:
            dense, sparse = dense.to(device), sparse.to(device)
            pred = model(dense, sparse)
            preds.extend(pred.cpu().numpy())
            labels.extend(label.numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    
    auc = roc_auc_score(labels, preds)
    logloss = log_loss(labels, preds)
    pcoc = preds.sum() / labels.sum()
    
    return {'auc': auc, 'logloss': logloss, 'pcoc': pcoc}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 主函数
# ============================================================

def run_experiment(model_class, model_name, train_loader, valid_loader, config, device):
    print(f"\n{'='*70}")
    print(f"模型: {model_name}")
    print(f"{'='*70}")
    
    # 创建模型
    model = model_class(config).to(device)
    params = count_parameters(model)
    print(f"参数量: {params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    criterion = nn.BCELoss()
    
    # 训练
    best_auc = 0
    for epoch in range(config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, valid_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{config.epochs} - Loss: {train_loss:.4f} - AUC: {metrics['auc']:.4f} - LogLoss: {metrics['logloss']:.4f}")
        
        if metrics['auc'] > best_auc:
            best_auc = metrics['auc']
            best_metrics = metrics
    
    print(f"\n最佳 AUC: {best_auc:.4f}")
    return {
        'model': model_name,
        'params': params,
        'best_auc': best_auc,
        'logloss': best_metrics['logloss'],
        'pcoc': best_metrics['pcoc']
    }


def main():
    # 数据路径
    data_dir = Path("/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet")
    train_path = data_dir / "train_train.parquet"
    valid_path = data_dir / "train_valid.parquet"
    
    # 加载数据
    print("加载数据...")
    train_data = CriteoDataset(train_path, fit=True)
    valid_data = CriteoDataset(
        valid_path, 
        dense_encoder=train_data.dense_encoder,
        sparse_encoders=train_data.sparse_encoders
    )
    
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # 对比模型
    models = [
        (TransformerCTR, "Transformer (无位置编码)"),
        (HSTULite, "HSTU-Lite"),
        (Mamba4CTR, "Mamba4CTR"),
    ]
    
    results = []
    for model_class, model_name in models:
        result = run_experiment(model_class, model_name, train_loader, valid_loader, config, device)
        results.append(result)
    
    # 汇总结果
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    print(f"{'模型':<30} {'参数量':>12} {'AUC':>10} {'LogLoss':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['model']:<30} {r['params']:>12,} {r['best_auc']:>10.4f} {r['logloss']:>10.4f}")
    
    # 保存结果
    output = {
        'experiment': 'exp16_full_features_no_pos',
        'config': {
            'embed_dim': config.embed_dim,
            'num_heads': config.num_heads,
            'num_layers': config.num_layers,
            'total_features': config.total_features,
            'pos_encoding': False
        },
        'results': results
    }
    
    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n结果已保存到 results.json")


if __name__ == "__main__":
    main()
