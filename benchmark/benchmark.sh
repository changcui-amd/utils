#!/bin/bash

PORT=8000
DATASETPATH=./datasets/ShareGPT_V3_unfiltered_cleaned_split.json
SEED=0

CONCURRENCY=8
NREQUESTS=100
ISL=6144
OSL=1024

python3 bench_serving.py --backend vllm \
--dataset-name random \
--dataset-path ${DATASETPATH} \
--num-prompts ${NREQUESTS} \
--random-input ${ISL} \
--random-output ${OSL} \
--random-range-ratio 1.0 \
--seed ${SEED} \
--max-concurrency ${CONCURRENCY} --warmup-requests ${CONCURRENCY} --port ${PORT}\
| tee sglang_benchmark_vllm_random_isl${ISL}_osl${OSL}_con${CONCURRENCY}.log
