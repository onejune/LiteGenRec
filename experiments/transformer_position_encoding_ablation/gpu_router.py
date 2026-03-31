"""GPU 自动路由模块 - 根据任务复杂度智能选择 GPU"""

import torch
import subprocess
from typing import Optional, Dict, List
from enum import Enum
from dataclasses import dataclass


class TaskComplexity(Enum):
    """任务复杂度等级"""
    LOW = "low"           # 调试、小模型
    MEDIUM = "medium"     # 中等规模实验
    HIGH = "high"         # 大规模训练


@dataclass
class TaskConfig:
    """任务配置"""
    model_params_mb: float      # 模型参数量 (MB)
    batch_size: int             # 批次大小
    complexity: TaskComplexity  # 复杂度等级
    is_debug: bool = False      # 是否调试模式


class GPURouter:
    """GPU 自动路由器"""
    
    def __init__(self):
        self.gpus = self._detect_gpus()
        if not self.gpus:
            raise RuntimeError("未检测到可用的 GPU")
        print(f"🔍 检测到 {len(self.gpus)} 个 GPU:")
        for gpu in self.gpus:
            status = "✅ 空闲" if gpu['utilization'] < 10 else "⚡ 使用中"
            print(f"   GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']}GB) | 利用率：{gpu['utilization']:.0f}% | 显存：{gpu['memory_used_mb']:.0f}MB/{gpu['memory_gb']*1024:.0f}MB [{status}]")
    
    def _get_gpu_utilization(self, gpu_id: int) -> float:
        """获取 GPU 当前利用率 (%)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', 
                 '--format=csv,nounits', '-i', str(gpu_id)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                util_str = result.stdout.strip().split('\n')[-1].strip()
                return float(util_str)
        except Exception as e:
            print(f"⚠️  无法获取 GPU {gpu_id} 利用率：{e}")
        
        return 0.0  # 默认返回 0
    
    def _get_gpu_memory_used(self, gpu_id: int) -> float:
        """获取 GPU 已用显存 (MB)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', 
                 '--format=csv,nounits', '-i', str(gpu_id)],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                mem_str = result.stdout.strip().split('\n')[-1].strip()
                return float(mem_str)
        except Exception as e:
            print(f"⚠️  无法获取 GPU {gpu_id} 显存使用：{e}")
        
        return 0.0
    
    def _detect_gpus(self) -> List[Dict]:
        """检测所有可用 GPU 并获取详细信息（包括实时利用率）"""
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            utilization = self._get_gpu_utilization(i)
            memory_used = self._get_gpu_memory_used(i)
            
            gpus.append({
                'id': i,
                'name': props.name,
                'memory_gb': round(props.total_memory / (1024**3), 1),
                'memory_used_mb': memory_used,
                'utilization': utilization,
                'compute_capability': f"{props.major}.{props.minor}"
            })
        
        # 按利用率升序排列（优先选空闲的）
        return sorted(gpus, key=lambda x: x['utilization'])
    
    def estimate_memory_usage(self, config: TaskConfig) -> float:
        """
        估算显存需求 (GB)
        
        粗略估计公式：
        - 模型参数 + 梯度 + 优化器状态：params * 4 (fp32)
        - 激活值：batch_size * hidden_dim * layers (简化为 batch_size * 0.1)
        """
        params_mem = config.model_params_mb * 4 / 1024  # MB -> GB
        activation_mem = config.batch_size * 0.1
        total_mem = params_mem + activation_mem
        return round(total_mem, 2)
    
    def select_gpu(self, config: TaskConfig) -> int:
        """
        根据任务配置选择最优 GPU
        
        策略:
        - 调试模式：永远用显存最小的卡（节省资源）
        - LOW 复杂度：用小卡
        - MEDIUM 复杂度：选刚好够用的卡
        - HIGH 复杂度：直接上大卡
        """
        estimated_mem = self.estimate_memory_usage(config)
        print(f"\n📊 任务分析:")
        print(f"   模型大小：{config.model_params_mb}MB")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   预估显存需求：{estimated_mem}GB")
        print(f"   复杂度：{config.complexity.value}")
        
        # 核心策略：优先选择利用率最低的 GPU
        # 在利用率相同的情况下，再考虑显存大小
        
        # 过滤掉显存不够的 GPU
        available_gpus = [g for g in self.gpus if g['memory_gb'] >= estimated_mem * 1.2]
        
        if not available_gpus:
            # 都没够用，仍然优先选利用率最低的（而不是显存最大的）
            selected = min(self.gpus, key=lambda x: (x['utilization'], -x['memory_gb']))
            reason = f"所有 GPU 显存都紧张 ({estimated_mem}GB)，选择利用率最低的"
        else:
            # 从可用的 GPU 中选利用率最低的
            selected = min(available_gpus, key=lambda x: (x['utilization'], -x['memory_gb']))
            
            if config.is_debug or config.complexity == TaskComplexity.LOW:
                reason = f"调试/简单任务，选择最空闲的 GPU"
            elif config.complexity == TaskComplexity.HIGH:
                reason = f"复杂任务，选择最空闲的大显存 GPU"
            else:
                reason = f"中等任务，选择最空闲且显存足够的 GPU"
        
        print(f"\n✅ 选中 GPU {selected['id']}: {selected['name']} ({selected['memory_gb']}GB)")
        print(f"   原因：{reason}")
        print(f"   当前利用率：{selected['utilization']:.0f}%")
        print(f"   显存余量：{selected['memory_gb'] - estimated_mem:.1f}GB\n")
        
        return selected['id']
    
    def get_device(self, config: TaskConfig) -> torch.device:
        """便捷方法：直接返回 torch.device"""
        gpu_id = self.select_gpu(config)
        return torch.device(f"cuda:{gpu_id}")


def select_gpu_for_training(model_params_mb: float = 500, 
                            batch_size: int = 256,
                            complexity: str = "medium",
                            is_debug: bool = False) -> torch.device:
    """
    一键选择训练 GPU
    
    Args:
        model_params_mb: 模型参数量 (MB)
        batch_size: 批次大小
        complexity: 复杂度 ('low', 'medium', 'high')
        is_debug: 是否调试模式
    
    Returns:
        torch.device: 选中的设备
    """
    router = GPURouter()
    config = TaskConfig(
        model_params_mb=model_params_mb,
        batch_size=batch_size,
        complexity=TaskComplexity(complexity),
        is_debug=is_debug
    )
    return router.get_device(config)


if __name__ == "__main__":
    print("=== GPU 路由器测试 ===\n")
    device = select_gpu_for_training(
        model_params_mb=100,
        batch_size=2048,
        complexity="medium"
    )
    print(f"使用设备：{device}\n")
