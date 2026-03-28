"""
日志工具
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "LiteGenRec",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_str: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
) -> logging.Logger:
    """设置日志器"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(
    exp_name: str,
    version: str = "v1",
    log_dir: str = "logs"
) -> logging.Logger:
    """获取实验日志器"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{exp_name}/{version}_{timestamp}.log"
    return setup_logger(name=f"{exp_name}_{version}", log_file=log_file)
