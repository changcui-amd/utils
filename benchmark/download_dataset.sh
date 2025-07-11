#!/usr/bin/env bash

download_dataset() {
    local source="${1:-hf}"
    local output_dir="./datasets"
    local output_file="${output_dir}/ShareGPT_V3_unfiltered_cleaned_split.json"

    local url1="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    local url2="https://www.modelscope.cn/datasets/gliang1001/ShareGPT_V3_unfiltered_cleaned_split/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json"

    mkdir -p "$output_dir"

    if [[ "$source" == "hf" ]]; then
        echo "Downloading from Hugging Face..."
        if curl -fL --retry 3 -o "$output_file" "$url1"; then
            echo "Download succeeded from Hugging Face: $output_file"
        else
            echo "ERROR: Download from Hugging Face failed." >&2
            return 1
        fi
    elif [[ "$source" == "ms" ]]; then
        echo "Downloading from ModelScope..."
        if curl -fL --retry 3 -o "$output_file" "$url2"; then
            echo "Download succeeded from ModelScope: $output_file"
        else
            echo "ERROR: Download from ModelScope failed." >&2
            return 1
        fi
    else
        echo "ERROR: Unknown source '$source'. Use 'hf' or 'ms'." >&2
        return 1
    fi
}

download_dataset "$1"
