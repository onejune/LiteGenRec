#!/usr/bin/env python3
"""
统一评估入口
Usage:
    python scripts/evaluate.py --config experiments/exp01_xxx/configs/v1.yaml --checkpoint path/to/model.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import load_yaml
from utils.logger import get_experiment_logger


def parse_args():
    parser = argparse.ArgumentParser(description="LiteGenRec Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_yaml(args.config)
    exp_name = config.get("name", "unknown")
    version = config.get("version", "v1")
    
    logger = get_experiment_logger(exp_name, version)
    logger.info(f"Evaluating: {exp_name} {version}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    # TODO: 加载模型、数据、评估
    # model = load_model(args.checkpoint)
    # dataloader = build_dataloader(config['data'], split='test')
    # metrics = evaluate(model, dataloader, config['evaluation'])
    # save_results(metrics, args.output)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
