#!/usr/bin/env python3
"""
Exp00: Baseline CTR 模型
在 Ali-CCP 数据集上训练传统 CTR 模型作为基线
"""

import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression

# DeepCTR
sys.path.insert(0, '/mnt/workspace/walter.wan/git_project/github_onejune/movas_hub/CTRAutoHyperopt/third_party/DeepCTR-Torch')
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM, DCN, AutoInt, WDL

import torch

# ============================================================
# 配置
# ============================================================
CONFIG = {
    'data_dir': '/mnt/data/oss_wanjun/pai_work/open_research/dataset/ali_ccp',
    'output_dir': '/mnt/workspace/walter.wan/open_research/LiteGenRec/outputs/exp00_baseline',
    'target': 'click',  # 'click' or 'purchase'
    'embed_dim': 16,
    'hidden_units': [256, 128, 64],
    'batch_size': 4096,
    'epochs': 3,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'models': ['LR', 'DeepFM', 'DCN', 'AutoInt', 'WDL'],
}

# ============================================================
# 数据加载
# ============================================================
def load_data():
    """加载 Ali-CCP 数据"""
    print("加载数据...")
    
    train_df = pd.read_csv(f"{CONFIG['data_dir']}/ali_ccp_train.csv")
    val_df = pd.read_csv(f"{CONFIG['data_dir']}/ali_ccp_val.csv")
    test_df = pd.read_csv(f"{CONFIG['data_dir']}/ali_ccp_test.csv")
    
    print(f"  训练集: {len(train_df):,}")
    print(f"  验证集: {len(val_df):,}")
    print(f"  测试集: {len(test_df):,}")
    
    return train_df, val_df, test_df


def preprocess_data(train_df, val_df, test_df):
    """预处理数据"""
    print("预处理数据...")
    
    # 识别特征列
    label_cols = ['click', 'purchase']
    sparse_cols = [c for c in train_df.columns if c not in label_cols and not c.startswith('D')]
    dense_cols = [c for c in train_df.columns if c.startswith('D')]
    
    print(f"  稀疏特征: {len(sparse_cols)}")
    print(f"  稠密特征: {len(dense_cols)}")
    
    # 合并数据进行编码
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    
    # Label Encoding 稀疏特征
    label_encoders = {}
    for col in sparse_cols:
        le = LabelEncoder()
        all_df[col] = le.fit_transform(all_df[col].astype(str))
        label_encoders[col] = le
    
    # MinMax Scaling 稠密特征
    if dense_cols:
        scaler = MinMaxScaler()
        all_df[dense_cols] = scaler.fit_transform(all_df[dense_cols])
    
    # 拆分回 train/val/test
    train_len = len(train_df)
    val_len = len(val_df)
    
    train_df = all_df.iloc[:train_len].copy()
    val_df = all_df.iloc[train_len:train_len+val_len].copy()
    test_df = all_df.iloc[train_len+val_len:].copy()
    
    # 构建特征列定义
    sparse_features = [SparseFeat(col, vocabulary_size=all_df[col].nunique(), embedding_dim=CONFIG['embed_dim']) 
                       for col in sparse_cols]
    dense_features = [DenseFeat(col, 1) for col in dense_cols]
    
    feature_columns = sparse_features + dense_features
    feature_names = get_feature_names(feature_columns)
    
    return train_df, val_df, test_df, feature_columns, feature_names, sparse_cols, dense_cols


def evaluate(y_true, y_pred, name=""):
    """计算评估指标"""
    auc = roc_auc_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred)
    pcoc = y_pred.mean() / y_true.mean() if y_true.mean() > 0 else 0
    
    print(f"  {name}: AUC={auc:.4f}, LogLoss={logloss:.4f}, PCOC={pcoc:.3f}")
    return {'auc': auc, 'logloss': logloss, 'pcoc': pcoc}


# ============================================================
# 模型训练
# ============================================================
def train_lr(train_df, test_df, sparse_cols, dense_cols, target):
    """训练 Logistic Regression"""
    print("\n训练 LR...")
    
    feature_cols = sparse_cols + dense_cols
    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target].values
    
    model = LogisticRegression(max_iter=100, solver='lbfgs', n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict_proba(X_test)[:, 1]
    return evaluate(y_test, y_pred, "LR")


def train_deepctr_model(model_name, train_df, val_df, test_df, 
                        feature_columns, feature_names, target):
    """训练 DeepCTR 模型"""
    print(f"\n训练 {model_name}...")
    
    # 准备数据
    train_input = {name: train_df[name].values for name in feature_names}
    val_input = {name: val_df[name].values for name in feature_names}
    test_input = {name: test_df[name].values for name in feature_names}
    
    y_train = train_df[target].values
    y_val = val_df[target].values
    y_test = test_df[target].values
    
    # 创建模型
    dnn_feature_columns = feature_columns
    linear_feature_columns = feature_columns
    
    if model_name == 'DeepFM':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, 
                       dnn_hidden_units=CONFIG['hidden_units'],
                       device=CONFIG['device'])
    elif model_name == 'DCN':
        model = DCN(linear_feature_columns, dnn_feature_columns,
                    dnn_hidden_units=CONFIG['hidden_units'],
                    device=CONFIG['device'])
    elif model_name == 'AutoInt':
        model = AutoInt(linear_feature_columns, dnn_feature_columns,
                        dnn_hidden_units=CONFIG['hidden_units'],
                        device=CONFIG['device'])
    elif model_name == 'WDL':
        model = WDL(linear_feature_columns, dnn_feature_columns,
                    dnn_hidden_units=CONFIG['hidden_units'],
                    device=CONFIG['device'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 编译
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    
    # 训练
    model.fit(train_input, y_train,
              batch_size=CONFIG['batch_size'],
              epochs=CONFIG['epochs'],
              validation_data=(val_input, y_val),
              verbose=1)
    
    # 预测
    y_pred = model.predict(test_input, batch_size=CONFIG['batch_size'])
    y_pred = y_pred.flatten()
    
    return evaluate(y_test, y_pred, model_name)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("Exp00: Baseline CTR 模型")
    print("=" * 60)
    
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 加载数据
    train_df, val_df, test_df = load_data()
    
    # 预处理
    train_df, val_df, test_df, feature_columns, feature_names, sparse_cols, dense_cols = \
        preprocess_data(train_df, val_df, test_df)
    
    target = CONFIG['target']
    print(f"\n目标: {target}")
    print(f"正样本率: {train_df[target].mean():.4f}")
    
    # 训练各模型
    results = {}
    
    for model_name in CONFIG['models']:
        try:
            if model_name == 'LR':
                result = train_lr(train_df, test_df, sparse_cols, dense_cols, target)
            else:
                result = train_deepctr_model(model_name, train_df, val_df, test_df,
                                             feature_columns, feature_names, target)
            results[model_name] = result
        except Exception as e:
            print(f"  {model_name} 失败: {e}")
            results[model_name] = {'auc': 0, 'logloss': 0, 'pcoc': 0, 'error': str(e)}
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("最终结果 (按 AUC 排序)")
    print("=" * 60)
    print(f"{'模型':<15} {'AUC':>8} {'LogLoss':>10} {'PCOC':>8}")
    print("-" * 45)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('auc', 0), reverse=True)
    for model_name, metrics in sorted_results:
        if 'error' not in metrics:
            print(f"{model_name:<15} {metrics['auc']:>8.4f} {metrics['logloss']:>10.4f} {metrics['pcoc']:>8.3f}")
    
    elapsed = time.time() - start_time
    print(f"\n总耗时: {elapsed/60:.1f} 分钟")
    
    # 保存结果
    import json
    result_file = f"{CONFIG['output_dir']}/baseline_results.json"
    with open(result_file, 'w') as f:
        json.dump({'config': CONFIG, 'results': results}, f, indent=2)
    print(f"结果保存: {result_file}")


if __name__ == '__main__':
    main()
