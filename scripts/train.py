#!/usr/bin/env python3
"""
统一训练入口
Usage:
    python scripts/train.py --config experiments/exp01_xxx/configs/v1.yaml
"""
import argparse
import sys
from pathlib import Path

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_yaml, ExperimentConfig
from utils.logger import get_experiment_logger


def parse_args():
    parser = argparse.ArgumentParser(description="LiteGenRec Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = load_yaml(args.config)
    exp_name = config.get("name", "unknown")
    version = config.get("version", "v1")
    
    # 设置日志
    logger = get_experiment_logger(exp_name, version)
    logger.info(f"Starting experiment: {exp_name} {version}")
    logger.info(f"Config: {args.config}")
    
    # TODO: 根据 config 初始化模型、数据、训练器
    # model = build_model(config['model'])
    # dataloader = build_dataloader(config['data'])
    # trainer = build_trainer(config['training'])
    # trainer.train(model, dataloader)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
