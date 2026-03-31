#!/bin/bash
# 消融实验快速启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "🔬 LiteGenRec 消融实验"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

show_help() {
    echo -e "${BLUE}用法:${NC}"
    echo "  $0 [options]"
    echo ""
    echo -e "${BLUE}选项:${NC}"
    echo "  --max-files INT     最大加载数据文件数 (default: None, 全部加载)"
    echo "  --epochs INT        训练轮数 (default: 3)"
    echo "  --batch-size INT    批次大小 (default: 2048)"
    echo "  --complexity STR    任务复杂度：low/medium/high (default: medium)"
    echo "  --help              显示帮助信息"
    echo ""
    echo -e "${BLUE}示例:${NC}"
    echo "  # 快速验证 (前 5 个文件)"
    echo "  $0 --max-files 5 --epochs 1"
    echo ""
    echo "  # 完整实验"
    echo "  $0 --epochs 3"
    echo ""
}

# 默认参数
MAX_FILES=""
EPOCHS=3
BATCH_SIZE=2048
COMPLEXITY="medium"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-files) MAX_FILES="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --complexity) COMPLEXITY="$2"; shift 2 ;;
        --help) show_help; exit 0 ;;
        *) echo "未知参数：$1"; show_help; exit 1 ;;
    esac
done

echo -e "${GREEN}配置:${NC}"
echo "  Max Files: ${MAX_FILES:-'全部'}"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Complexity: $COMPLEXITY"
echo ""
echo "=========================================="
echo ""

# 构建命令
CMD="python models.py --epochs $EPOCHS --batch-size $BATCH_SIZE --complexity $COMPLEXITY"

if [ -n "$MAX_FILES" ]; then
    CMD="$CMD --max-files $MAX_FILES"
fi

# 运行实验
echo "运行命令: $CMD"
echo ""
eval $CMD

echo ""
echo "=========================================="
echo "✅ 实验完成！"
echo "=========================================="
