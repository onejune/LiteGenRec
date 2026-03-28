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
├── README.md           # 项目说明
├── docs/               # 文档、论文笔记、技术调研
│   └── research_plan.md
├── experiments/        # 实验设计与报告
├── configs/            # 配置文件
├── src/                # 源代码
│   ├── models/         # 模型实现
│   ├── data/           # 数据处理
│   └── utils/          # 工具函数
├── scripts/            # 训练/评估脚本
└── notebooks/          # 探索性分析
```

## 🔧 开发环境

| 路径 | 用途 |
|------|------|
| `/mnt/workspace/walter.wan/git_project/github_onejune/LiteGenRec/` | Git 仓库 (代码/文档) |
| `/mnt/workspace/walter.wan/open_research/LiteGenRec/` | 工作目录 (运行/数据) |
| `工作目录/reference/` | 参考开源项目 |

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
| [MiniMind](https://github.com/jingyaogong/minimind) | 轻量级 LLM 从零实现 | Transformer, SFT, DPO |
| [MiniOneRec](https://github.com/xxx) | 生成式推荐 | SFT + RL (GRPO) |
| [SemanticID-Gen](https://github.com/xxx) | 语义 ID 生成 | RQ-VAE |

## 📊 数据集

| 数据集 | 描述 | 状态 |
|--------|------|------|
| 公开数据集 | 待定 | 🔲 |
| DSP 广告数据 | 内部数据 | 🔲 |

## 🧪 实验记录

| 日期 | 实验 | 描述 | 指标 | 状态 |
|------|------|------|------|------|
| 2026-03-28 | - | 项目初始化 | - | ✅ |

## 📈 Baseline & 指标

### 评估指标
- **推荐指标**: HR@K, NDCG@K, MRR
- **广告指标**: AUC, PCOC, CTR

### Baseline
| 模型 | 类型 | 指标 | 备注 |
|------|------|------|------|
| TBD | - | - | - |

## 🚀 快速开始

```bash
# 进入工作目录
cd /mnt/workspace/walter.wan/open_research/LiteGenRec

# 代码在 repo/ 软链接
cd repo/src
```

## 📝 TODO

- [ ] 调研 MiniMind 架构
- [ ] 调研 MiniOneRec 训练流程
- [ ] 调研 SemanticID-Gen 语义 ID 方案
- [ ] 设计 DSP 广告数据适配方案
- [ ] Baseline 实验

## License

MIT
