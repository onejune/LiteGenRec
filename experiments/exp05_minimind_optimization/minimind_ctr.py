#!/usr/bin/env python3
"""
LiteGenRec Exp05: 基于 MiniMind 的轻量级 CTR 模型
探索大模型在程序化广告推荐中的轻量化落地方案
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pyarrow.parquet as pq
import os
import math
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 基于 MiniMind 思路的轻量级 Transformer 模型
class MiniMindCTR(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=32, num_heads=4, num_layers=2, hidden_dim=128):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # 特征嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(39, embed_dim))  # 39个特征
        
        # 轻量级多头注意力 + 前馈网络 (MiniMind 风格)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                    'norm1': nn.LayerNorm(embed_dim),
                    'ffn': nn.Sequential(
                        nn.Linear(embed_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, embed_dim),
                        nn.Dropout(0.1)
                    ),
                    'norm2': nn.LayerNorm(embed_dim)
                })
            )
        
        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * 39, hidden_dim),  # 39个特征 * embed_dim
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: [batch_size, 39] - 39个特征的索引
        """
        batch_size = x.size(0)
        
        # 嵌入所有特征
        embedded = []
        for i in range(39):  # 39个特征
            emb = self.embeddings[i](x[:, i].long())
            embedded.append(emb.unsqueeze(1))  # [batch_size, 1, embed_dim]
        
        # 合并所有特征嵌入 [batch_size, 39, embed_dim]
        x = torch.cat(embedded, dim=1)
        
        # 添加位置编码
        x = x + self.pos_encoding.unsqueeze(0)
        
        # 通过每一层
        for layer in self.layers:
            # 多头注意力
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # 前馈网络
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # 展平并输出
        x = x.view(batch_size, -1)  # [batch_size, 39 * embed_dim]
        output = self.output_proj(x).squeeze(-1)  # [batch_size]
        
        return output

# 旋转位置编码 (RoPE) - MiniMind 优化技术之一
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim == self.dim
        
        # 生成旋转角度
        inv_freq = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        position_ids = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1)  # [seq_len, 1]
        
        # 计算 sin/cos 旋转矩阵
        freqs = torch.einsum("ij,j->ij", position_ids, inv_freq)  # [seq_len, embed_dim//2]
        sin_freqs = torch.sin(freqs).repeat_interleave(2, dim=-1)  # [seq_len, embed_dim]
        cos_freqs = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [seq_len, embed_dim]
        
        # 扩展到批次大小
        sin_freqs = sin_freqs.unsqueeze(0)  # [1, seq_len, embed_dim]
        cos_freqs = cos_freqs.unsqueeze(0)  # [1, seq_len, embed_dim]
        
        # 应用旋转位置编码
        x_rotated = x * cos_freqs + self.rotate_half(x) * sin_freqs
        return x_rotated
    
    def rotate_half(self, x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.cat((-x2, x1), dim=-1)

# 带 RoPE 的改进版 MiniMindCTR
class MiniMindCTR_RoPE(nn.Module):
    def __init__(self, vocab_sizes, embed_dim=32, num_heads=4, num_layers=2, hidden_dim=128):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_features = len(vocab_sizes)  # 动态计算特征数量
        
        # 特征嵌入层
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        # 旋转位置编码
        self.rope = RotaryPositionEmbedding(embed_dim)
        
        # 轻量级多头注意力 + 前馈网络
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict({
                    'attention': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                    'norm1': nn.LayerNorm(embed_dim),
                    'ffn': nn.Sequential(
                        nn.Linear(embed_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, embed_dim),
                        nn.Dropout(0.1)
                    ),
                    'norm2': nn.LayerNorm(embed_dim)
                })
            )
        
        # 输出层 - 使用动态计算的特征数量
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * self.num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: [batch_size, num_features] - 特征的索引
        """
        batch_size = x.size(0)
        
        # 嵌入所有特征
        embedded = []
        num_features = min(x.size(1), len(self.embeddings))  # 防止索引超出范围
        
        for i in range(num_features):  # 遍历实际的特征数量
            # 确保输入值在有效范围内
            feat_vals = torch.clamp(x[:, i].long(), 0, self.embeddings[i].num_embeddings - 1)
            emb = self.embeddings[i](feat_vals)
            embedded.append(emb.unsqueeze(1))  # [batch_size, 1, embed_dim]
        
        # 合并所有特征嵌入 [batch_size, num_features, embed_dim]
        x = torch.cat(embedded, dim=1)
        
        # 应用旋转位置编码
        x = self.rope(x)
        
        # 通过每一层
        for layer in self.layers:
            # 多头注意力
            attn_out, _ = layer['attention'](x, x, x)
            x = layer['norm1'](x + attn_out)
            
            # 前馈网络
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # 展平并输出
        x = x.view(batch_size, -1)  # [batch_size, num_features * embed_dim]
        output = self.output_proj(x).squeeze(-1)  # [batch_size]
        
        return output

# 数据预处理器
class MiniMindFeatureProcessor:
    def __init__(self):
        self.dense_scalers = [MinMaxScaler() for _ in range(13)]
        self.sparse_encoders = [LabelEncoder() for _ in range(26)]
        self.vocab_sizes = []
        
    def fit_transform(self, df):
        """拟合并转换数据"""
        processed_df = df.copy()
        
        # 处理稠密特征 (I1-I13)
        dense_cols = [f"I{i}" for i in range(1, 14)]
        for i, col in enumerate(dense_cols):
            # 填充缺失值
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            # 归一化
            processed_df[col] = self.dense_scalers[i].fit_transform(
                processed_df[col].values.reshape(-1, 1)
            ).flatten()
            # 离散化为整数索引 (映射到 0-99)
            processed_df[col] = (processed_df[col] * 99).astype(int).clip(0, 99)
        
        # 处理稀疏特征 (C1-C26)
        sparse_cols = [f"C{i}" for i in range(1, 27)]
        for i, col in enumerate(sparse_cols):
            # 填充缺失值
            processed_df[col] = processed_df[col].fillna('__NULL__').astype(str)
            # 标签编码
            processed_df[col] = self.sparse_encoders[i].fit_transform(processed_df[col])
            # 限制词汇表大小
            processed_df[col] = np.clip(processed_df[col], 0, 9999)
        
        # 计算词汇表大小 - 确保不超过 embedding 层的索引范围
        self.vocab_sizes = []
        for i in range(13):
            # 稠密特征的词汇表大小（离散化后的范围）
            self.vocab_sizes.append(100)  # 因为我们离散化到了 0-99
        
        for i in range(26):
            # 稀疏特征的词汇表大小
            vocab_size = len(self.sparse_encoders[i].classes_)
            # 确保 embedding 层的大小与实际词汇表大小匹配
            self.vocab_sizes.append(vocab_size)
        
        return processed_df
    
    def transform(self, df):
        """转换新数据"""
        processed_df = df.copy()
        
        # 处理稠密特征 (I1-I13)
        dense_cols = [f"I{i}" for i in range(1, 14)]
        for i, col in enumerate(dense_cols):
            # 填充缺失值
            processed_df[col] = processed_df[col].fillna(processed_df[col].mean())
            # 归一化
            processed_df[col] = self.dense_scalers[i].transform(
                processed_df[col].values.reshape(-1, 1)
            ).flatten()
            # 离散化为整数索引
            processed_df[col] = (processed_df[col] * 99).astype(int).clip(0, 99)
        
        # 处理稀疏特征 (C1-C26)
        sparse_cols = [f"C{i}" for i in range(1, 27)]
        for i, col in enumerate(sparse_cols):
            # 填充缺失值
            processed_df[col] = processed_df[col].fillna('__NULL__').astype(str)
            
            # 处理未见过的标签 - 先替换为默认值，再进行编码
            unique_values = set(processed_df[col].unique())
            known_values = set(self.sparse_encoders[i].classes_)
            unknown_mask = ~processed_df[col].isin(known_values)
            processed_df.loc[unknown_mask, col] = self.sparse_encoders[i].classes_[0]  # 使用第一个类别作为默认值
            
            # 标签编码
            processed_df[col] = self.sparse_encoders[i].transform(processed_df[col])
            # 限制词汇表大小到训练时的最大值
            max_vocab_size = len(self.sparse_encoders[i].classes_)
            processed_df[col] = np.clip(processed_df[col], 0, max_vocab_size - 1)
        
        return processed_df

def train_minimind_ctr():
    print("开始训练 LiteGenRec V3 - 基于 MiniMind 的轻量级 CTR 模型")
    
    # 加载数据
    DATA_DIR = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet'
    train_path = f"{DATA_DIR}/train_train.parquet"
    valid_path = f"{DATA_DIR}/train_valid.parquet"
    
    # 读取训练集的前几个分片
    train_files = [f for f in os.listdir(train_path) if f.endswith('.parquet')][:1]
    train_dfs = []
    for f in train_files:
        df = pq.read_table(f"{train_path}/{f}").to_pandas()
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # 读取验证集的一个分片
    valid_files = [f for f in os.listdir(valid_path) if f.endswith('.parquet')][:1]
    valid_dfs = []
    for f in valid_files:
        df = pq.read_table(f"{valid_path}/{f}").to_pandas()
        valid_dfs.append(df)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    
    print(f"训练集: {len(train_df):,} 样本, 验证集: {len(valid_df):,} 样本")
    print(f"正样本率 - 训练: {train_df['label'].mean():.4f}, 验证: {valid_df['label'].mean():.4f}")
    
    # 预处理数据
    processor = MiniMindFeatureProcessor()
    train_df = processor.fit_transform(train_df)
    valid_df = processor.transform(valid_df)
    
    # 分离特征和标签
    all_cols = [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    
    train_features = train_df[all_cols]
    train_labels = train_df['label']
    
    valid_features = valid_df[all_cols]
    valid_labels = valid_df['label']
    
    # 创建 PyTorch 张量
    train_tensor = torch.LongTensor(train_features.values)
    train_label_tensor = torch.FloatTensor(train_labels.values)
    valid_tensor = torch.LongTensor(valid_features.values)
    valid_label_tensor = torch.FloatTensor(valid_labels.values)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # 初始化模型 (使用带 RoPE 的版本)
    model = MiniMindCTR_RoPE(
        vocab_sizes=processor.vocab_sizes,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        hidden_dim=128
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # 训练循环
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"开始第 {epoch+1} 轮训练...")
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 验证
        model.eval()
        valid_preds = []
        valid_true = []
        
        with torch.no_grad():
            valid_outputs = model(valid_tensor.to(device))
            valid_loss = criterion(valid_outputs, valid_label_tensor.to(device))
            
            valid_preds = valid_outputs.cpu().numpy()
            valid_true = valid_label_tensor.numpy()
        
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(valid_true, valid_preds)
        except:
            auc = 0.5
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/train_batches:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        print(f"  Valid AUC: {auc:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "/mnt/workspace/walter.wan/open_research/LiteGenRec/experiments/exp05_minimind_optimization/minimind_ctr_model.pth")
    print("MiniMind CTR 模型已保存")
    
    return model, processor

if __name__ == "__main__":
    model, processor = train_minimind_ctr()