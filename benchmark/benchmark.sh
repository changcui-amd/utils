#!/bin/bash

PORT=8000
SEED=0
DATASETPATH=./datasets/ShareGPT_V3_unfiltered_cleaned_split.json
if [ ! -f "${DATASETPATH}" ]; then
  echo "Dataset not found at ${DATASETPATH}, downloading..."
  bash -x download_dataset.sh
fi

CONCURRENCY=${1:-"1"}
NREQUESTS=$(($CONCURRENCY * 10))
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
