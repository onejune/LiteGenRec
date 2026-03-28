# Exp02: Generative CTR Prediction

## 目标

验证生成式推荐范式在广告 CTR 预估任务上的有效性。

## 背景

- 传统 CTR: 特征工程 + DNN (DeepFM, DCN, etc.)
- 生成式 CTR: 用户行为序列 → LLM → 预测下一个点击广告的 SID

## 方法

| 版本 | 方案 | 描述 |
|------|------|------|
| v1 | SFT Only | 用户点击序列 → 预测下一个 SID |
| v2 | SFT + GRPO | 加入强化学习优化 |
| v3 | Multi-task | 同时预测 CTR + CVR |

## 数据

- 用户数: TBD
- 广告数: TBD
- 样本数: TBD
- 序列长度: TBD

## Baseline

| 模型 | 类型 | AUC | 备注 |
|------|------|-----|------|
| DeepFM | 传统 DNN | TBD | baseline |
| DCN | 传统 DNN | TBD | baseline |

## 评估指标

- **AUC**: 主指标
- **PCOC**: 校准度
- **HR@K**: 推荐准确率 (辅助)
- **NDCG@K**: 排序质量 (辅助)

## 结果

| 版本 | AUC | PCOC | HR@10 | 备注 |
|------|-----|------|-------|------|
| - | - | - | - | 待实验 |

## 结论

TBD

## TODO

- [ ] 依赖 Exp01 完成 SID 构建
- [ ] 构建用户点击序列数据
- [ ] 实现轻量 LLM (参考 MiniMind)
- [ ] 实验 v1: SFT 训练
- [ ] 对比 baseline
