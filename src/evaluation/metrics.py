"""
评估指标
"""
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import roc_auc_score


def calc_auc(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算 AUC"""
    if len(np.unique(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


def calc_pcoc(labels: np.ndarray, preds: np.ndarray) -> float:
    """计算 PCOC (Predicted CTR / Observed CTR)"""
    observed_ctr = np.mean(labels)
    predicted_ctr = np.mean(preds)
    if observed_ctr == 0:
        return 0.0
    return predicted_ctr / observed_ctr


def calc_hr_at_k(targets: List[int], predictions: List[List[int]], k: int = 10) -> float:
    """
    计算 Hit Rate @ K
    targets: 真实 item id 列表
    predictions: 预测的 top-k item id 列表的列表
    """
    hits = 0
    for target, preds in zip(targets, predictions):
        if target in preds[:k]:
            hits += 1
    return hits / len(targets) if targets else 0.0


def calc_ndcg_at_k(targets: List[int], predictions: List[List[int]], k: int = 10) -> float:
    """
    计算 NDCG @ K
    """
    ndcg_sum = 0.0
    for target, preds in zip(targets, predictions):
        preds_k = preds[:k]
        if target in preds_k:
            rank = preds_k.index(target) + 1
            ndcg_sum += 1.0 / np.log2(rank + 1)
    return ndcg_sum / len(targets) if targets else 0.0


def calc_mrr(targets: List[int], predictions: List[List[int]]) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)
    """
    mrr_sum = 0.0
    for target, preds in zip(targets, predictions):
        if target in preds:
            rank = preds.index(target) + 1
            mrr_sum += 1.0 / rank
    return mrr_sum / len(targets) if targets else 0.0


def calc_grouped_auc(
    labels: np.ndarray,
    preds: np.ndarray,
    groups: np.ndarray
) -> Dict[str, float]:
    """
    分组计算 AUC (按 business_type 等)
    """
    results = {}
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask = groups == group
        group_labels = labels[mask]
        group_preds = preds[mask]
        
        if len(np.unique(group_labels)) >= 2:
            results[str(group)] = calc_auc(group_labels, group_preds)
        else:
            results[str(group)] = None
    
    # 总体 AUC
    results['overall'] = calc_auc(labels, preds)
    
    return results


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self, metrics: List[str] = None):
        self.metrics = metrics or ['auc', 'pcoc']
        self.metric_funcs = {
            'auc': calc_auc,
            'pcoc': calc_pcoc,
        }
    
    def calculate(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """计算所有指标"""
        results = {}
        
        for metric in self.metrics:
            if metric in self.metric_funcs:
                results[metric] = self.metric_funcs[metric](labels, preds)
        
        # 分组指标
        if groups is not None:
            results['grouped_auc'] = calc_grouped_auc(labels, preds, groups)
        
        return results
