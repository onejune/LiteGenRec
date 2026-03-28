# LiteGenRec 研究计划

## 1. 研究背景

传统推荐系统 (CTR 预估) 主要基于：
- 特征工程 + 浅层模型 (LR, GBDT)
- 深度学习模型 (DeepFM, DCN, DIN 等)

生成式推荐的新范式：
- 将推荐建模为序列生成任务
- 利用 LLM 的语义理解能力
- 统一多种推荐任务

## 2. 研究目标

探索在程序化 DSP 广告场景下，生成式推荐的轻量化落地方案：

1. **轻量化**: 模型参数 < 1B，可在单卡部署
2. **实用性**: 推理延迟满足广告实时竞价需求 (< 10ms)
3. **效果**: 在 CTR/CVR 指标上有竞争力

## 3. 技术路线

### Phase 1: 调研与复现
- [ ] 精读 MiniOneRec 论文与代码
- [ ] 精读 MiniMind 实现
- [ ] 调研其他生成式推荐工作 (P5, GPT4Rec, etc.)

### Phase 2: 数据准备
- [ ] 分析现有 DSP 广告数据特点
- [ ] 设计序列化表示方案
- [ ] 构建训练数据集

### Phase 3: 模型实验
- [ ] Baseline: 传统 CTR 模型
- [ ] 实验 1: 小规模 LLM + 推荐微调
- [ ] 实验 2: 生成式推荐范式

### Phase 4: 优化与部署
- [ ] 模型压缩 (量化、蒸馏)
- [ ] 推理优化
- [ ] A/B 测试方案

## 4. 关键问题

1. **序列建模**: 广告场景如何构建有效的用户行为序列？
2. **冷启动**: 生成式方法如何处理新广告/新用户？
3. **实时性**: 如何满足广告竞价的延迟要求？
4. **可解释性**: 生成式推荐的决策如何解释？

## 5. 基础开源项目

工作目录已 clone 以下 3 个开源项目作为研究基础：

### 5.1 MiniMind
- **路径**: `/mnt/workspace/walter.wan/open_research/LiteGenRec/minimind/`
- **简介**: 轻量级 LLM 从零实现，包含完整的训练、推理流程
- **核心文件**:
  - `model/` — 模型定义
  - `trainer/` — 训练逻辑
  - `eval_llm.py` — 评估脚本
- **GitHub**: https://github.com/jingyaogong/minimind

### 5.2 MiniOneRec
- **路径**: `/mnt/workspace/walter.wan/open_research/LiteGenRec/MiniOneRec/`
- **简介**: 生成式推荐系统，将推荐建模为序列生成任务
- **核心文件**:
  - `minionerec_trainer.py` — 主训练器
  - `sft.py` / `rl.py` — SFT 和 RL 训练
  - `sasrec.py` — SASRec 序列推荐模块
  - `data.py` — 数据处理
  - `evaluate.py` — 评估脚本
- **训练流程**: SFT → RL (GRPO)

### 5.3 SemanticID-Gen
- **路径**: `/mnt/workspace/walter.wan/open_research/LiteGenRec/SemanticID-Gen/`
- **简介**: 语义 ID 生成，将 item 编码为语义化的离散 ID
- **核心流程**:
  1. `step3_text2emb.sh` — 文本转 embedding
  2. `step4_train_rqvae.sh` — 训练 RQ-VAE
  3. `step5_generate_sid.sh` — 生成语义 ID
  4. `step6_text2sid.sh` — 文本直接生成 ID
- **关键技术**: RQ-VAE (Residual Quantization VAE)

## 6. 研究路线图

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  MiniMind   │     │ MiniOneRec  │     │SemanticID   │
│ (轻量 LLM)  │     │ (生成式推荐) │     │ (语义 ID)   │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                 ┌─────────────────┐
                 │   LiteGenRec    │
                 │ 轻量生成式推荐  │
                 └────────┬────────┘
                          ▼
                 ┌─────────────────┐
                 │  DSP 广告场景   │
                 │   落地实践      │
                 └─────────────────┘
```

## 7. 参考文献

待补充...
