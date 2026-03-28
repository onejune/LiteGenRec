"""
配置管理工具
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


def load_yaml(path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_yaml(config: Dict[str, Any], path: str):
    """保存配置到 YAML 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """合并多个配置，后面的覆盖前面的"""
    result = {}
    for config in configs:
        if config:
            _deep_merge(result, config)
    return result


def _deep_merge(base: Dict, update: Dict):
    """深度合并字典"""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str
    version: str = "v1"
    
    # 路径
    data_dir: str = ""
    output_dir: str = ""
    checkpoint_dir: str = ""
    
    # 模型
    model: Dict[str, Any] = field(default_factory=dict)
    
    # 数据
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 训练
    training: Dict[str, Any] = field(default_factory=dict)
    
    # 评估
    evaluation: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从 YAML 文件加载配置"""
        config = load_yaml(path)
        return cls(**config)
    
    def to_yaml(self, path: str):
        """保存配置到 YAML 文件"""
        save_yaml(self.__dict__, path)
