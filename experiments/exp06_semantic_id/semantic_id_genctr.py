#!/usr/bin/env python3
"""
Exp06: Semantic ID GenCTR - 基于语义ID的生成式CTR模型

核心思想:
1. 使用 VQ-VAE 将特征编码为离散语义ID
2. 层次化ID捕获不同粒度的语义信息
3. 生成式建模预测点击概率

参考: TIGER (NeurIPS 2023), Meta HSTU (ICML 2024)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class VectorQuantizer(nn.Module):
    """向量量化模块 - VQ-VAE核心组件"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, dim] 输入向量
        Returns:
            quantized: 量化后的向量
            indices: 量化索引 (语义ID)
            loss: VQ损失
        """
        # 计算距离
        distances = (
            x.pow(2).sum(dim=-1, keepdim=True)
            - 2 * x @ self.embeddings.weight.t()
            + self.embeddings.weight.pow(2).sum(dim=-1)
        )
        
        # 找最近的码本向量
        indices = distances.argmin(dim=-1)
        quantized = self.embeddings(indices)
        
        # VQ损失
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        return quantized, indices, loss


class HierarchicalVQVAE(nn.Module):
    """层次化VQ-VAE - 生成多层语义ID"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_levels: int = 3,
        codebook_sizes: List[int] = [512, 256, 128],
        latent_dim: int = 32
    ):
        super().__init__()
        self.num_levels = num_levels
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 多层量化器
        self.quantizers = nn.ModuleList()
        self.pre_quant = nn.ModuleList()
        self.post_quant = nn.ModuleList()
        
        for i, codebook_size in enumerate(codebook_sizes[:num_levels]):
            self.pre_quant.append(nn.Linear(hidden_dim, latent_dim))
            self.quantizers.append(VectorQuantizer(codebook_size, latent_dim))
            self.post_quant.append(nn.Linear(latent_dim, hidden_dim))
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """编码并返回各层语义ID"""
        h = self.encoder(x)
        
        semantic_ids = []
        quantized_list = []
        total_vq_loss = 0
        
        residual = h
        for i in range(self.num_levels):
            z = self.pre_quant[i](residual)
            quantized, indices, vq_loss = self.quantizers[i](z)
            
            semantic_ids.append(indices)
            quantized_list.append(quantized)
            total_vq_loss = total_vq_loss + vq_loss
            
            # 残差连接
            residual = residual - self.post_quant[i](quantized)
        
        return semantic_ids, quantized_list, total_vq_loss
    
    def decode(self, quantized_list: List[torch.Tensor]) -> torch.Tensor:
        """从量化向量解码"""
        h = sum(self.post_quant[i](q) for i, q in enumerate(quantized_list))
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        semantic_ids, quantized_list, vq_loss = self.encode(x)
        recon = self.decode(quantized_list)
        recon_loss = F.mse_loss(recon, x)
        return recon, semantic_ids, vq_loss + recon_loss


class SemanticIDGenCTR(nn.Module):
    """基于语义ID的生成式CTR模型"""
    
    def __init__(
        self,
        num_sparse_features: int,
        vocab_sizes: List[int],
        embed_dim: int = 32,
        hidden_dim: int = 128,
        num_levels: int = 3,
        codebook_sizes: List[int] = [512, 256, 128],
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_sparse_features = num_sparse_features
        self.num_levels = num_levels
        
        # 特征嵌入
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
            for vocab_size in vocab_sizes
        ])
        
        # 层次化VQ-VAE (将特征编码为语义ID)
        total_embed_dim = num_sparse_features * embed_dim
        self.vqvae = HierarchicalVQVAE(
            input_dim=total_embed_dim,
            hidden_dim=hidden_dim,
            num_levels=num_levels,
            codebook_sizes=codebook_sizes,
            latent_dim=embed_dim
        )
        
        # 语义ID嵌入 (用于CTR预测)
        self.semantic_embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim)
            for codebook_size in codebook_sizes[:num_levels]
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
            nn.Linear(embed_dim * num_levels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_semantic_ids(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """获取语义ID"""
        # 特征嵌入
        embeds = []
        for i in range(self.num_sparse_features):
            field_vals = x[:, i].long().clamp(0, self.embeddings[i].num_embeddings - 1)
            embeds.append(self.embeddings[i](field_vals))
        
        # 拼接所有特征嵌入
        concat_embed = torch.cat(embeds, dim=-1)  # [batch, num_features * embed_dim]
        
        # 通过VQ-VAE获取语义ID
        _, semantic_ids, vq_loss = self.vqvae.encode(concat_embed)
        
        return semantic_ids, vq_loss
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, num_features] 稀疏特征
        Returns:
            logits: CTR预测
            vq_loss: VQ-VAE损失
        """
        # 获取语义ID
        semantic_ids, vq_loss = self.get_semantic_ids(x)
        
        # 语义ID嵌入 - ids是[batch]形状，embed后是[batch, embed_dim]
        semantic_embeds = []
        for i, (ids, embed_layer) in enumerate(zip(semantic_ids, self.semantic_embeddings)):
            emb = embed_layer(ids.long())  # [batch, embed_dim]
            semantic_embeds.append(emb)
        
        # 堆叠为序列 [batch, num_levels, embed_dim]
        semantic_seq = torch.stack(semantic_embeds, dim=1)  # [batch, num_levels, embed_dim]
        
        # Transformer编码
        encoded = self.transformer(semantic_seq)  # [batch, num_levels, embed_dim]
        
        # 展平并预测CTR
        encoded_flat = encoded.reshape(encoded.size(0), -1)  # [batch, num_levels * embed_dim]
        logits = self.ctr_head(encoded_flat)
        
        return logits.squeeze(-1), vq_loss


class CriteoDataProcessor:
    """Criteo数据处理器 - 使用parquet格式"""
    
    def __init__(self, data_dir: str, sample_size: int = 2000000):
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.sparse_encoders = {}
        self.vocab_sizes = []
        
    def load_and_process(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载并处理数据"""
        import pyarrow.parquet as pq
        
        print("Loading Criteo dataset from parquet...")
        
        train_path = f"{self.data_dir}/train_train.parquet"
        valid_path = f"{self.data_dir}/train_valid.parquet"
        
        # 加载训练数据 (只用部分分片)
        train_files = [f for f in os.listdir(train_path) if f.endswith('.parquet')][:2]
        train_dfs = []
        for f in train_files:
            df = pq.read_table(os.path.join(train_path, f)).to_pandas()
            train_dfs.append(df)
        train_df = pd.concat(train_dfs, ignore_index=True)
        
        # 加载验证数据
        valid_files = [f for f in os.listdir(valid_path) if f.endswith('.parquet')][:1]
        valid_dfs = []
        for f in valid_files:
            df = pq.read_table(os.path.join(valid_path, f)).to_pandas()
            valid_dfs.append(df)
        valid_df = pd.concat(valid_dfs, ignore_index=True)
        
        print(f"Loaded train: {len(train_df)}, valid: {len(valid_df)}")
        
        # 稀疏特征列
        sparse_cols = [f'C{i}' for i in range(1, 27)]
        
        # 合并数据集进行编码
        all_df = pd.concat([train_df, valid_df], ignore_index=True)
        
        # 处理稀疏特征
        for col in sparse_cols:
            all_df[col] = all_df[col].fillna('__MISSING__')
            self.sparse_encoders[col] = LabelEncoder()
            all_df[col] = self.sparse_encoders[col].fit_transform(all_df[col].astype(str)) + 1
            self.vocab_sizes.append(int(all_df[col].max()) + 1)
        
        # 分回训练和验证
        train_df = all_df.iloc[:len(train_df)]
        valid_df = all_df.iloc[len(train_df):]
        
        # 提取特征和标签
        X_train = train_df[sparse_cols].values
        X_valid = valid_df[sparse_cols].values
        y_train = train_df['label'].values
        y_valid = valid_df['label'].values
        
        print(f"Train: {len(X_train)}, Valid: {len(X_valid)}")
        print(f"Vocab sizes: {self.vocab_sizes[:5]}... (total {len(self.vocab_sizes)} features)")
        
        return X_train, X_valid, y_train, y_valid


def train_semantic_id_model(
    model: SemanticIDGenCTR,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = 3,
    lr: float = 1e-3,
    vq_weight: float = 0.1
) -> Dict:
    """训练模型"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0
    history = {'train_loss': [], 'valid_auc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()
            
            optimizer.zero_grad()
            logits, vq_loss = model(x)
            
            # 总损失 = CTR损失 + VQ损失
            ctr_loss = criterion(logits, y)
            loss = ctr_loss + vq_weight * vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f} (CTR: {ctr_loss.item():.4f}, VQ: {vq_loss.item():.4f})")
        
        avg_loss = total_loss / num_batches
        history['train_loss'].append(avg_loss)
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, y in valid_loader:
                x = x.to(device)
                logits, _ = model(x)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        
        auc = roc_auc_score(all_labels, all_preds)
        history['valid_auc'].append(auc)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Valid AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_semantic_id_model.pt')
    
    return {'best_auc': best_auc, 'history': history}


def main():
    """主函数"""
    print("=" * 60)
    print("Exp06: Semantic ID GenCTR")
    print("=" * 60)
    
    # 数据路径 - 使用parquet格式
    data_dir = "/mnt/data/oss_wanjun/pai_work/open_research/dataset/criteo_standard/criteo-parquet"
    
    # 加载数据
    processor = CriteoDataProcessor(data_dir, sample_size=2000000)
    X_train, X_valid, y_train, y_valid = processor.load_and_process()
    
    # 创建DataLoader
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
    
    # 创建模型
    model = SemanticIDGenCTR(
        num_sparse_features=26,
        vocab_sizes=processor.vocab_sizes,
        embed_dim=32,
        hidden_dim=128,
        num_levels=3,
        codebook_sizes=[512, 256, 128],
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    print("\n" + "=" * 60)
    print("Training Semantic ID GenCTR...")
    print("=" * 60)
    
    results = train_semantic_id_model(
        model,
        train_loader,
        valid_loader,
        epochs=3,
        lr=1e-3,
        vq_weight=0.1
    )
    
    print("\n" + "=" * 60)
    print(f"Best Valid AUC: {results['best_auc']:.4f}")
    print("=" * 60)
    
    # 保存结果
    with open('results.txt', 'w') as f:
        f.write(f"Semantic ID GenCTR Results\n")
        f.write(f"Best AUC: {results['best_auc']:.4f}\n")
        f.write(f"Training history: {results['history']}\n")
    
    return results


if __name__ == "__main__":
    main()
