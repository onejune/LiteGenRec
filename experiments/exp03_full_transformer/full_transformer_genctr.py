#!/usr/bin/env python3
"""
LiteGenRec V2: 完整版 Transformer 架构生成式 CTR 模型
基于 MiniOneRec 思路，使用完整 Transformer 进行序列建模
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
import sys
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 完整版模型配置
class FullTransformerConfig:
    def __init__(self):
        self.embed_dim = 64          # embedding 维度
        self.hidden_dim = 256        # 隐藏层维度
        self.num_heads = 8           # 注意力头数
        self.num_layers = 4          # Transformer 层数
        self.max_seq_len = 39        # 特征总数 (13+26)
        self.dropout = 0.1
        self.vocab_size_sparse = 50000  # 稀疏特征词表大小
        self.vocab_size_dense = 50      # 稠密特征离散化范围

config = FullTransformerConfig()

# 完整版 Transformer 架构的生成式 CTR 模型
class FullTransformerGenCTR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征嵌入层
        self.dense_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_dense, config.embed_dim) 
            for _ in range(13)  # I1-I13
        ])
        
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(config.vocab_size_sparse, config.embed_dim) 
            for _ in range(26)  # C1-C26
        ])
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(config.max_seq_len, config.embed_dim))
        
        # Layer Normalization
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # 完整的 Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 多头注意力层用于特征交互
        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 最终输出层
        self.output_proj = nn.Sequential(
            nn.Linear(config.embed_dim * config.max_seq_len, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, dense_features, sparse_features):
        """
        dense_features: [batch_size, 13] - 稠密特征
        sparse_features: [batch_size, 26] - 稀疏特征
        """
        batch_size = dense_features.size(0)
        
        # 嵌入稠密特征
        dense_embs = []
        for i in range(13):
            emb = self.dense_embeds[i](dense_features[:, i].long())
            dense_embs.append(emb.unsqueeze(1))  # [batch_size, 1, embed_dim]
        
        # 嵌入稀疏特征
        sparse_embs = []
        for i in range(26):
            emb = self.sparse_embeds[i](sparse_features[:, i].long())
            sparse_embs.append(emb.unsqueeze(1))  # [batch_size, 1, embed_dim]
        
        # 合并所有特征嵌入形成序列 [batch_size, 39, embed_dim]
        all_embs = torch.cat(dense_embs + sparse_embs, dim=1)
        
        # 应用 Layer Normalization
        all_embs = self.norm(all_embs)
        
        # 添加位置编码
        pos_enc = self.pos_encoding[:39, :].unsqueeze(0)  # [1, 39, embed_dim]
        all_embs = all_embs + pos_enc
        
        # 通过 Transformer 编码器
        transformer_out = self.transformer(all_embs)  # [batch_size, 39, embed_dim]
        
        # 通过多头注意力层进一步增强特征交互
        interaction_out, _ = self.interaction_attention(
            transformer_out, transformer_out, transformer_out
        )  # [batch_size, 39, embed_dim]
        
        # 展平所有特征
        flattened = interaction_out.reshape(batch_size, -1)  # [batch_size, 39 * embed_dim]
        
        # 输出点击概率
        probs = self.output_proj(flattened).squeeze(-1)  # [batch_size]
        
        return probs

# 简化的数据预处理器 (复用之前的逻辑)
class SimpleCriteoDataProcessor:
    def __init__(self):
        self.dense_scalers = [MinMaxScaler() for _ in range(13)]
        self.sparse_encoders = [LabelEncoder() for _ in range(26)]
    
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
            # 离散化为整数索引 (映射到 0-49)
            processed_df[col] = (processed_df[col] * 49).astype(int).clip(0, 49)
        
        # 处理稀疏特征 (C1-C26)
        sparse_cols = [f"C{i}" for i in range(1, 27)]
        for i, col in enumerate(sparse_cols):
            # 填充缺失值
            processed_df[col] = processed_df[col].fillna('__NULL__').astype(str)
            # 标签编码
            processed_df[col] = self.sparse_encoders[i].fit_transform(processed_df[col])
            # 限制词汇表大小
            processed_df[col] = np.clip(processed_df[col], 0, 49999)
        
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
            processed_df[col] = (processed_df[col] * 49).astype(int).clip(0, 49)
        
        # 处理稀疏特征 (C1-C26)
        sparse_cols = [f"C{i}" for i in range(1, 27)]
        for i, col in enumerate(sparse_cols):
            # 填充缺失值
            processed_df[col] = processed_df[col].fillna('__NULL__').astype(str)
            
            # 处理未见过的标签
            mask = ~processed_df[col].isin(self.sparse_encoders[i].classes_)
            processed_df.loc[mask, col] = '__NULL__'
            
            # 确保 '__NULL__' 在编码器中
            if '__NULL__' not in self.sparse_encoders[i].classes_:
                all_classes = list(self.sparse_encoders[i].classes_) + ['__NULL__']
                self.sparse_encoders[i].classes_ = np.array(all_classes)
            
            # 标签编码
            processed_df[col] = self.sparse_encoders[i].transform(processed_df[col])
            # 限制词汇表大小
            processed_df[col] = np.clip(processed_df[col], 0, 49999)
        
        return processed_df

# 数据集类
class CriteoDataset(Dataset):
    def __init__(self, dense_features, sparse_features, labels):
        self.dense_features = torch.FloatTensor(dense_features.values)
        self.sparse_features = torch.LongTensor(sparse_features.values)
        self.labels = torch.FloatTensor(labels.values)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'dense': self.dense_features[idx],
            'sparse': self.sparse_features[idx], 
            'label': self.labels[idx]
        }

def train_full_transformer_genctr():
    print("开始训练 LiteGenRec V2 - 完整版 Transformer 生成式 CTR 模型")
    
    # 加载 Criteo 数据
    DATA_DIR = '/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet'
    
    # 读取训练集的一部分用于训练
    train_path = f"{DATA_DIR}/train_train.parquet"
    valid_path = f"{DATA_DIR}/train_valid.parquet"
    
    # 读取训练集的前几个分片
    train_files = [f for f in os.listdir(train_path) if f.endswith('.parquet')][:1]  # 只用1个分片
    train_dfs = []
    for f in train_files:
        df = pq.read_table(f"{train_path}/{f}").to_pandas()
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # 读取验证集的一个分片
    valid_files = [f for f in os.listdir(valid_path) if f.endswith('.parquet')][:1]  # 只用1个分片
    valid_dfs = []
    for f in valid_files:
        df = pq.read_table(f"{valid_path}/{f}").to_pandas()
        valid_dfs.append(df)
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    
    print(f"训练集: {len(train_df):,} 样本, 验证集: {len(valid_df):,} 样本")
    print(f"正样本率 - 训练: {train_df['label'].mean():.4f}, 验证: {valid_df['label'].mean():.4f}")
    
    # 预处理数据
    processor = SimpleCriteoDataProcessor()
    train_df = processor.fit_transform(train_df)
    valid_df = processor.transform(valid_df)
    
    # 分离特征和标签
    dense_cols = [f"I{i}" for i in range(1, 14)]
    sparse_cols = [f"C{i}" for i in range(1, 27)]
    
    train_dense = train_df[dense_cols]
    train_sparse = train_df[sparse_cols]
    train_labels = train_df['label']
    
    valid_dense = valid_df[dense_cols]
    valid_sparse = valid_df[sparse_cols]
    valid_labels = valid_df['label']
    
    # 创建数据集和数据加载器
    train_dataset = CriteoDataset(train_dense, train_sparse, train_labels)
    valid_dataset = CriteoDataset(valid_dense, valid_sparse, valid_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)  # 减小批次大小以适应更复杂模型
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    
    # 初始化模型
    model = FullTransformerGenCTR(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # 训练循环
    num_epochs = 2  # 减少训练轮数以节省时间
    for epoch in range(num_epochs):
        print(f"开始第 {epoch+1} 轮训练...")
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            dense = batch['dense'].to(device)
            sparse = batch['sparse'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(dense, sparse)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 50 == 0:  # 每50个批次打印一次
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # 验证
        model.eval()
        valid_loss = 0.0
        valid_batches = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                dense = batch['dense'].to(device)
                sparse = batch['sparse'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(dense, sparse)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                valid_batches += 1
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算 AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5  # 防止计算错误
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss/train_batches:.4f}")
        print(f"  Valid Loss: {valid_loss/valid_batches:.4f}")
        print(f"  Valid AUC: {auc:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "/mnt/workspace/walter.wan/open_research/LiteGenRec/experiments/exp03_full_transformer/full_transformer_genctr_model.pth")
    print("完整版 Transformer 模型已保存")
    
    return model, processor

if __name__ == "__main__":
    model, processor = train_full_transformer_genctr()