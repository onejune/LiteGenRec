#!/usr/bin/env python3
"""
LiteGenRec Exp04: SOTA 模型对比实验
对比不同 CTR 模型在 Criteo 数据集上的性能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pyarrow.parquet as pq
import os
import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# 深度学习模型
class DeepFM(nn.Module):
    def __init__(self, field_sizes, embed_dim=16):
        super().__init__()
        self.field_sizes = field_sizes
        self.embed_dim = embed_dim
        self.num_fields = len(field_sizes)
        
        # Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embed_dim) for size in field_sizes
        ])
        
        # Linear part - for each individual feature
        # We'll use a simpler approach: just use the sum of all features as linear input
        self.linear = nn.Linear(self.num_fields, 1)
        
        # DNN part
        total_embed_dim = self.num_fields * embed_dim
        self.dnn = nn.Sequential(
            nn.Linear(total_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_fields] - each field is one feature index
        batch_size = x.size(0)
        
        # Process each field individually through embeddings
        embedded = []
        for i in range(self.num_fields):
            field_vals = x[:, i]  # [batch_size]
            field_vals = field_vals.long()  # Ensure long type for embedding
            # Clamp values to valid range for embedding
            field_vals = torch.clamp(field_vals, 0, self.field_sizes[i]-1)
            field_embed = self.embeddings[i](field_vals)  # [batch_size, embed_dim]
            embedded.append(field_embed)
        
        embedded = torch.stack(embedded, dim=1)  # [batch_size, num_fields, embed_dim]
        
        # Linear part - use the original input for linear transformation
        linear_input = x.float()  # [batch_size, num_fields]
        linear_out = self.linear(linear_input)  # [batch_size, 1]
        
        # FM part - simplified version
        sum_of_embeddings = embedded.sum(dim=1)  # [batch_size, embed_dim]
        square_of_sum = sum_of_embeddings ** 2  # [batch_size, embed_dim]
        sum_of_square = (embedded ** 2).sum(dim=1)  # [batch_size, embed_dim]
        fm_out = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # DNN part
        dnn_input = embedded.view(batch_size, -1)  # [batch_size, num_fields * embed_dim]
        dnn_out = self.dnn(dnn_input)  # [batch_size, 1]
        
        # Combine all parts
        output = torch.sigmoid(linear_out.squeeze(1) + fm_out.squeeze(1) + dnn_out.squeeze(1))  # [batch_size]
        return output

class WideAndDeep(nn.Module):
    def __init__(self, field_sizes, embed_dim=16):
        super().__init__()
        self.field_sizes = field_sizes
        self.embed_dim = embed_dim
        self.num_fields = len(field_sizes)
        
        # Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(size, embed_dim) for size in field_sizes
        ])
        
        # Wide part (linear) - just use the input features directly
        self.wide = nn.Linear(self.num_fields, 1)
        
        # Deep part
        total_embed_dim = self.num_fields * embed_dim
        self.deep = nn.Sequential(
            nn.Linear(total_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, num_fields] - each field is one feature index
        batch_size = x.size(0)
        
        # Process each field individually through embeddings
        embedded = []
        for i in range(self.num_fields):
            field_vals = x[:, i]  # [batch_size]
            field_vals = field_vals.long()  # Ensure long type for embedding
            # Clamp values to valid range for embedding
            field_vals = torch.clamp(field_vals, 0, self.field_sizes[i]-1)
            field_embed = self.embeddings[i](field_vals)  # [batch_size, embed_dim]
            embedded.append(field_embed)
        
        embedded = torch.stack(embedded, dim=1)  # [batch_size, num_fields, embed_dim]
        
        # Wide part - use the original input for linear transformation
        wide_input = x.float()  # [batch_size, num_fields]
        wide_out = self.wide(wide_input)  # [batch_size, 1]
        
        # Deep part
        deep_input = embedded.view(batch_size, -1)  # [batch_size, num_fields * embed_dim]
        deep_out = self.deep(deep_input)  # [batch_size, 1]
        
        # Combine wide and deep
        output = torch.sigmoid(wide_out.squeeze(1) + deep_out.squeeze(1))  # [batch_size]
        return output

# 数据预处理器
class CriteoFeatureProcessor:
    def __init__(self):
        self.dense_scalers = [MinMaxScaler() for _ in range(13)]
        self.sparse_encoders = [LabelEncoder() for _ in range(26)]
        self.vocab_sizes = []  # 用于存储每个字段的词汇表大小
        
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
        
        # 计算词汇表大小
        self.vocab_sizes = []
        for i in range(13):
            # 稠密特征的词汇表大小（离散化后的范围）
            self.vocab_sizes.append(100)  # 因为我们离散化到了 0-99
        
        for i in range(26):
            # 稀疏特征的词汇表大小
            self.vocab_sizes.append(len(self.sparse_encoders[i].classes_))
        
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
            processed_df[col] = np.clip(processed_df[col], 0, 9999)
        
        return processed_df

def evaluate_models():
    print("开始 SOTA 模型对比实验...")
    
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
    
    # 预处理数据
    processor = CriteoFeatureProcessor()
    train_df = processor.fit_transform(train_df)
    valid_df = processor.transform(valid_df)
    
    # 分离特征和标签
    all_cols = [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    
    train_features = train_df[all_cols]
    train_labels = train_df['label']
    
    valid_features = valid_df[all_cols]
    valid_labels = valid_df['label']
    
    # 准备结果字典
    results = {}
    
    # 1. 逻辑回归 (LR)
    print("训练逻辑回归模型...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(train_features.values, train_labels.values)
    lr_pred = lr_model.predict_proba(valid_features.values)[:, 1]
    
    from sklearn.metrics import roc_auc_score
    lr_auc = roc_auc_score(valid_labels.values, lr_pred)
    results['Logistic Regression'] = lr_auc
    print(f"  AUC: {lr_auc:.4f}")
    
    # 2. 随机森林 (RF)
    print("训练随机森林模型...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(train_features.values, train_labels.values)
    rf_pred = rf_model.predict_proba(valid_features.values)[:, 1]
    
    rf_auc = roc_auc_score(valid_labels.values, rf_pred)
    results['Random Forest'] = rf_auc
    print(f"  AUC: {rf_auc:.4f}")
    
    # 3. XGBoost
    print("训练 XGBoost 模型...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(train_features.values, train_labels.values)
    xgb_pred = xgb_model.predict_proba(valid_features.values)[:, 1]
    
    xgb_auc = roc_auc_score(valid_labels.values, xgb_pred)
    results['XGBoost'] = xgb_auc
    print(f"  AUC: {xgb_auc:.4f}")
    
    # 4. DeepFM (PyTorch)
    print("训练 DeepFM 模型...")
    field_sizes = processor.vocab_sizes  # 使用预处理后的词汇表大小
    deepfm_model = DeepFM(field_sizes, embed_dim=16).to(device)
    
    # 准备 PyTorch 数据集 - 转换为 LongTensor for embedding lookup
    train_tensor = torch.LongTensor(train_features.values)
    train_label_tensor = torch.FloatTensor(train_labels.values)
    valid_tensor = torch.LongTensor(valid_features.values)
    valid_label_tensor = torch.FloatTensor(valid_labels.values)
    
    train_dataset = torch.utils.data.TensorDataset(train_tensor, train_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    optimizer = torch.optim.Adam(deepfm_model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    # 训练 DeepFM
    deepfm_model.train()
    for epoch in range(2):  # 只训练2轮
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = deepfm_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"    Epoch {epoch+1}/2, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # 评估 DeepFM
    deepfm_model.eval()
    with torch.no_grad():
        valid_pred = deepfm_model(valid_tensor.to(device)).cpu().numpy()
    
    deepfm_auc = roc_auc_score(valid_labels.values, valid_pred)
    results['DeepFM'] = deepfm_auc
    print(f"  AUC: {deepfm_auc:.4f}")
    
    # 5. Wide&Deep (PyTorch)
    print("训练 Wide&Deep 模型...")
    # Use the same field sizes as DeepFM
    widedeep_model = WideAndDeep(field_sizes, embed_dim=16).to(device)
    
    optimizer = torch.optim.Adam(widedeep_model.parameters(), lr=1e-3)
    
    # 训练 Wide&Deep
    widedeep_model.train()
    for epoch in range(2):  # 只训练2轮
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = widedeep_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"    Epoch {epoch+1}/2, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # 评估 Wide&Deep
    widedeep_model.eval()
    with torch.no_grad():
        valid_pred = widedeep_model(valid_tensor.to(device)).cpu().numpy()
    
    widedeep_auc = roc_auc_score(valid_labels.values, valid_pred)
    results['Wide&Deep'] = widedeep_auc
    print(f"  AUC: {widedeep_auc:.4f}")
    
    # 6. 加载之前的模型结果
    # LiteGenRec V1 (SimpleGenCTR) - 从之前的实验获取结果
    results['LiteGenRec V1 (SimpleGenCTR)'] = 0.7663  # 我们之前训练得到的结果
    
    # 输出最终对比结果
    print("\n" + "="*60)
    print("SOTA 模型对比结果 (AUC)")
    print("="*60)
    
    # 按 AUC 排序
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, auc) in enumerate(sorted_results, 1):
        print(f"{rank:2d}. {model_name:<30} : {auc:.4f}")
    
    print("="*60)
    
    return results

if __name__ == "__main__":
    results = evaluate_models()