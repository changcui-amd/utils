#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <jsonl_file>"
    exit 1
fi

file="$1"

jq -r '
  select(.request_throughput != null) |
  . as $obj |
  "request_throughput: \($obj.request_throughput)\n" +
  "input_throughput: \($obj.input_throughput)\n" +
  "output_throughput: \($obj.output_throughput)\n" +
  "total_throughput: \($obj.input_throughput + $obj.output_throughput)\n" +
  "mean_ttft_ms: \($obj.mean_ttft_ms)\n" +
  "median_ttft_ms: \($obj.median_ttft_ms)\n" +
  "mean_tpot_ms: \($obj.mean_tpot_ms)\n" +
  "median_tpot_ms: \($obj.median_tpot_ms)"
' "$file"
