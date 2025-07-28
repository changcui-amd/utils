#!/bin/bash

batch_sizes=(8 16 32 64 128)

for bs in "${batch_sizes[@]}"; do
    echo "Running benchmark.sh with batch size: $bs"
    bash benchmark.sh "$bs"

    if [ $? -ne 0 ]; then
        echo "benchmark.sh failed for batch size $bs, exiting."
        exit 1
    fi
done

echo "All benchmarks completed."
