# 实验索引

本目录包含 LiteGenRec 项目的所有实验。

## 实验命名规范

```
exp{序号}_{方向简称}/
├── README.md          # 必须包含: 目标、方法、数据、结果、结论
├── configs/           # 配置文件 (版本化)
│   └── v1.yaml
├── scripts/           # 运行脚本
│   └── run_v1.sh
└── results/           # 结果 (可 gitignore 大文件)
    └── v1/
```

## 实验列表

| 序号 | 方向 | 目标 | 状态 | 优先级 |
|------|------|------|------|--------|
| 01 | SID Construction | 广告 Semantic ID 构建 | 🔲 | P0 |
| 02 | Generative CTR | 生成式 CTR 预估 | 🔲 | P0 |
| 03 | Lightweight LLM | 模型轻量化 (<100M) | 🔲 | P1 |
| 04 | Multi-Objective | CTR + CVR 多目标 | 🔲 | P2 |
| 05 | Cold Start | 新广告冷启动 | 🔲 | P2 |

## 状态说明

- 🔲 待开始
- 🔄 进行中
- ✅ 已完成
- ❌ 已放弃

## 实验记录模板

每个实验的 README.md 应包含：

```markdown
# Exp{序号}: {实验名称}

## 目标
{一句话描述实验目标}

## 方法
- v1: {方法描述}
- v2: {方法描述}

## 数据
- 数据集: xxx
- 样本数: xxx
- 时间范围: xxx

## 结果

| 版本 | 指标1 | 指标2 | 备注 |
|------|-------|-------|------|
| v1   | xxx   | xxx   | baseline |
| v2   | xxx   | xxx   | +改进点 |

## 结论
{实验结论和 insight}

## TODO
- [ ] 下一步计划
```
