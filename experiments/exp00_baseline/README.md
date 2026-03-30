# Exp00: Baseline CTR 模型

## 目标

建立传统 CTR 模型的基准性能，作为生成式推荐的对比基线。

## 数据集

- **Ali-CCP**: 阿里点击转化预测数据集
- 训练集: 2,859,201 样本
- 验证集: 1,454,317 样本
- 测试集: 1,454,317 样本
- 特征: 23 稀疏 + 8 稠密 = 31 特征
- 标签: click (33%), purchase (0.4%)

## 模型

| 模型 | 类型 | 说明 |
|------|------|------|
| LR | 线性 | Logistic Regression |
| DeepFM | DNN | 特征交互 |
| DCN | DNN | Cross Network |
| AutoInt | DNN | Self-Attention |

## 评估指标

- **AUC**: 排序能力
- **LogLoss**: 预测概率质量
- **PCOC**: 校准度 (pred_mean / label_mean)

## 实验配置

- 任务: CTR (click 预估)
- Embedding dim: 16
- Hidden units: [256, 128, 64]
- Batch size: 4096
- Epochs: 3

## 结果

| 模型 | AUC | LogLoss | PCOC |
|------|-----|---------|------|
| LR | - | - | - |
| DeepFM | - | - | - |
| DCN | - | - | - |
| AutoInt | - | - | - |

## 运行

```bash
cd /mnt/workspace/walter.wan/open_research/LiteGenRec
python repo/experiments/exp00_baseline/scripts/run_baseline.py
```
