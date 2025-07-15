#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <jsonl_file>"
    exit 1
fi

file="$1"

if ! command -v jq >/dev/null; then
    echo "Error: jq is not installed." >&2
    exit 1
fi

echo -e "request_throughput (req/s)\tinput_throughput (tok/s)\toutput_throughput (tok/s)\ttotal_throughput (tok/s)\tmean_ttft (ms)\tmedian_ttft (ms)\tmean_tpot (ms)\tmedian_tpot (ms)"

jq -r '
  select(.request_throughput != null) |
  [
    .request_throughput,
    .input_throughput,
    .output_throughput,
    (.input_throughput + .output_throughput),
    .mean_ttft_ms,
    .median_ttft_ms,
    .mean_tpot_ms,
    .median_tpot_ms
  ] | @tsv
' "$file" | awk -F '\t' '{
  printf "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", $1, $2, $3, $4, $5, $6, $7, $8
}'
