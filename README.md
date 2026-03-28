# LiteGenRec

> 轻量级生成式推荐系统 — 探索大模型在程序化广告推荐中的落地实践

## 🎯 项目目标

探讨大语言模型 (LLM) 在推荐系统中的轻量化落地方案，结合：
- **MiniMind**: 轻量级 LLM 架构
- **MiniOneRec**: 生成式推荐范式
- **SemanticID-Gen**: 语义 ID 生成
- **程序化 DSP 广告**: 实际业务场景

## 📁 项目结构

```
LiteGenRec/
├── README.md                    # 项目说明
├── docs/                        # 文档
│   ├── research_plan.md         # 研究计划
│   └── papers/                  # 论文笔记
│
├── src/                         # 公共代码库
│   ├── models/                  # 模型定义
│   ├── data/                    # 数据处理
│   ├── trainers/                # 训练器
│   ├── evaluation/              # 评估指标
│   └── utils/                   # 工具函数
│
├── experiments/                 # 实验目录 (按子方向组织)
│   ├── README.md                # 实验索引
│   ├── exp01_sid_construction/  # SID 构建实验
│   └── exp02_generative_ctr/    # 生成式 CTR 实验
│
├── configs/                     # 全局配置模板
│   └── base.yaml
│
├── scripts/                     # 通用脚本
│   ├── train.py                 # 统一训练入口
│   └── evaluate.py              # 统一评估入口
│
└── notebooks/                   # 探索分析
```

## 🔧 开发环境

| 路径 | 用途 |
|------|------|
| `/mnt/workspace/walter.wan/git_project/github_onejune/LiteGenRec/` | Git 仓库 (代码/文档) |
| `/mnt/workspace/walter.wan/open_research/LiteGenRec/` | 工作目录 (运行/数据) |

### 工作目录结构

```
/mnt/workspace/walter.wan/open_research/LiteGenRec/
├── data/               # 数据集
├── logs/               # 训练日志
├── checkpoints/        # 模型检查点
├── outputs/            # 实验输出
├── repo/               # → Git 仓库软链接
└── reference/          # 参考开源项目
    ├── minimind/       # 轻量级 LLM
    ├── MiniOneRec/     # 生成式推荐
    └── SemanticID-Gen/ # 语义 ID 生成
```

## 📚 参考项目

| 项目 | 简介 | 核心技术 |
|------|------|----------|
| [MiniMind](https://github.com/jingyaogong/minimind) | 轻量级 LLM 从零实现 | Transformer, SFT, GRPO |
| [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) | 生成式推荐框架 | SID + SFT + RL |
| SemanticID-Gen | 语义 ID 生成 | RQ-VAE |

## 🧪 实验方向

| 序号 | 方向 | 目标 | 状态 | 优先级 |
|------|------|------|------|--------|
| 01 | SID Construction | 广告 Semantic ID 构建 | 🔲 | P0 |
| 02 | Generative CTR | 生成式 CTR 预估 | 🔲 | P0 |
| 03 | Lightweight LLM | 模型轻量化 (<100M) | 🔲 | P1 |
| 04 | Multi-Objective | CTR + CVR 多目标 | 🔲 | P2 |
| 05 | Cold Start | 新广告冷启动 | 🔲 | P2 |

## 📊 数据集

| 数据集 | 描述 | 状态 |
|--------|------|------|
| 公开数据集 | Amazon Reviews | 🔲 |
| DSP 广告数据 | 内部数据 | 🔲 |

## 📈 Baseline & 指标

### 评估指标
- **推荐指标**: HR@K, NDCG@K, MRR
- **广告指标**: AUC, PCOC

### Baseline
| 模型 | 类型 | AUC | 备注 |
|------|------|-----|------|
| DeepFM | DNN | TBD | baseline |
| DCN | DNN | TBD | baseline |

## 🚀 快速开始

```bash
# 进入工作目录
cd /mnt/workspace/walter.wan/open_research/LiteGenRec

# 训练
python repo/scripts/train.py --config repo/experiments/exp01_xxx/configs/v1.yaml

# 评估
python repo/scripts/evaluate.py --config repo/experiments/exp01_xxx/configs/v1.yaml --checkpoint checkpoints/xxx.pt
```

## 📝 实验规范

1. **配置驱动**: 用 YAML 管理超参，不硬编码
2. **README 先行**: 开实验前先写目标，跑完补结论
3. **结果可追溯**: 每次实验记录 config + metrics + git commit
4. **代码复用**: 公共逻辑放 `src/`，实验只写差异部分

## License

MIT
