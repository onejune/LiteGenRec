#!/usr/bin/env python3
"""
统一数据加载器 - 支持 Criteo 稠密+稀疏特征

所有实验共用，确保公平对比
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pyarrow.parquet as pq


def load_criteo_full_features(data_dir: str, max_vocab_size: int = 50000):
    """
    加载 Criteo 数据集，包含稠密+稀疏特征
    
    Args:
        data_dir: 数据目录
        max_vocab_size: 稀疏特征词汇表大小上限
    
    Returns:
        X_dense_train, X_sparse_train, y_train
        X_dense_valid, X_sparse_valid, y_valid
        vocab_sizes (list): 稀疏特征词汇表大小
    """
    print("Loading Criteo dataset (dense + sparse features)...")
    
    train_path = f"{data_dir}/train_train.parquet"
    valid_path = f"{data_dir}/train_valid.parquet"
    
    # 读取训练集
    train_files = sorted([f for f in os.listdir(train_path) if f.endswith('.parquet')])[:2]
    train_dfs = [pq.read_table(os.path.join(train_path, f)).to_pandas() for f in train_files]
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # 读取验证集
    valid_files = sorted([f for f in os.listdir(valid_path) if f.endswith('.parquet')])[:1]
    valid_dfs = [pq.read_table(os.path.join(valid_path, f)).to_pandas() for f in valid_files]
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    
    print(f"Loaded train: {len(train_df)}, valid: {len(valid_df)}")
    
    # 稠密特征 (I1-I13)
    dense_cols = [f'I{i}' for i in range(1, 14)]
    
    # 稀疏特征 (C1-C26)
    sparse_cols = [f'C{i}' for i in range(1, 27)]
    
    # 处理稠密特征
    all_df = pd.concat([train_df, valid_df], ignore_index=True)
    
    # 填充缺失值并标准化
    for col in dense_cols:
        all_df[col] = all_df[col].fillna(0)
    
    scaler = StandardScaler()
    all_df[dense_cols] = scaler.fit_transform(all_df[dense_cols])
    
    # 处理稀疏特征
    vocab_sizes = []
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('__MISSING__')
        encoder = LabelEncoder()
        all_df[col] = encoder.fit_transform(all_df[col].astype(str)) + 1
        # 限制词汇表大小
        all_df[col] = np.clip(all_df[col], 0, max_vocab_size)
        vocab_sizes.append(min(int(all_df[col].max()) + 1, max_vocab_size + 1))
    
    # 分割训练集和验证集
    train_df = all_df.iloc[:len(train_df)]
    valid_df = all_df.iloc[len(train_df):]
    
    # 提取特征
    X_dense_train = train_df[dense_cols].values.astype(np.float32)
    X_sparse_train = train_df[sparse_cols].values.astype(np.int64)
    y_train = train_df['label'].values.astype(np.float32)
    
    X_dense_valid = valid_df[dense_cols].values.astype(np.float32)
    X_sparse_valid = valid_df[sparse_cols].values.astype(np.int64)
    y_valid = valid_df['label'].values.astype(np.float32)
    
    print(f"Dense features: {len(dense_cols)}, Sparse features: {len(sparse_cols)}")
    print(f"Vocab sizes (first 5): {vocab_sizes[:5]}")
    
    return (X_dense_train, X_sparse_train, y_train,
            X_dense_valid, X_sparse_valid, y_valid,
            vocab_sizes)


def load_criteo_sparse_only(data_dir: str, max_vocab_size: int = 50000):
    """
    加载 Criteo 数据集，仅稀疏特征 (用于消融对比)
    """
    print("Loading Criteo dataset (sparse features only)...")
    
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
        all_df[col] = np.clip(all_df[col], 0, max_vocab_size)
        vocab_sizes.append(min(int(all_df[col].max()) + 1, max_vocab_size + 1))
    
    train_df = all_df.iloc[:len(train_df)]
    valid_df = all_df.iloc[len(train_df):]
    
    X_train = train_df[sparse_cols].values
    X_valid = valid_df[sparse_cols].values
    y_train = train_df['label'].values
    y_valid = valid_df['label'].values
    
    return X_train, X_valid, y_train, y_valid, vocab_sizes
