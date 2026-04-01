#!/usr/bin/env bash
# =====================================================================
# 脚本：run_multimodal_training.sh
# 功能：从指定 parquet 数据启动 Qwen2-VL 多模态模型微调训练
# 使用方法：bash run_multimodal_training.sh [GPUS]
# 示例：bash run_multimodal_training.sh 4
# =====================================================================

set -euo pipefail

# ------------------ 可配置参数 ------------------
GPUS=${1:-1}                                    # 默认使用 1 张 GPU
OUTPUT_DIR="./output/Qwen2-VL-2B"              # 保存检查点目录
MODEL_NAME_OR_PATH="Qwen/Qwen2-VL-2B-Instruct" # 模型标识符或本地缓存目录
BATCH_SIZE=4
ACCUM_STEPS=4
EPOCHS=2
LR=1e-4
SAVE_STEPS=100
LOGGING_STEPS=10

# ------------------ 环境准备 ------------------
echo "==> 激活 Python 环境并安装依赖（如有必要）"
# source ~/.bashrc
# conda activate your_env
# pip install -r requirements.txt

# ------------------ 分布式训练命令 ------------------
echo "==> 使用 ${GPUS} 张 GPU 开始训练"

torchrun --nproc_per_node=${GPUS} \
    --master_port=29500 \
    qwen_vl_train.py \
    --model-name-or-path ${MODEL_NAME_OR_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --per-device-train-batch-size ${BATCH_SIZE} \
    --gradient-accumulation-steps ${ACCUM_STEPS} \
    --num-train-epochs ${EPOCHS} \
    --learning-rate ${LR} \
    --save-steps ${SAVE_STEPS} \
    --logging-steps ${LOGGING_STEPS} \
    --gradient-checkpointing \
    --report-to none

echo "==> 训练完成，结果保存在 ${OUTPUT_DIR}"