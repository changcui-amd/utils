#!/bin/bash

rm -rf /root/.cache/vllm/

export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export MODEL_PATH=/mnt/models/models--deepseek-ai--DeepSeek-R1/snapshots/a157fa3d494497a54586a333a23df6c2143e7697

vllm serve $MODEL_PATH \
-tp 8 \
--block-size 1 \
--trust-remote-code \
--disable-log-requests \
--max-seq-len-to-capture 32768 \
--max-num-batched-tokens 32768 \
--no-enable-prefix-caching \
# --load-format dummy \
# --compilation_config '{"compile_sizes": [1, 2, 4, 8]}' \
# --compilation-config '{"full_cuda_graph": true}' \
