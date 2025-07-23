echo "deepseek-ai/DeepSeek-R1 aiter v1"

VLLM_RPC_TIMEOUT=1800000 \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
SAFETENSORS_FAST_GPU=1 \
lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1,tensor_parallel_size=8,enable_expert_parallel=True,max_model_len=32768,block_size=1 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto
