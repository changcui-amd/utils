download_dataset() {
    local output_dir="./datasets"
    local output_file="${output_dir}/ShareGPT_V3_unfiltered_cleaned_split.json"

    local url1="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    local url2="https://www.modelscope.cn/datasets/gliang1001/ShareGPT_V3_unfiltered_cleaned_split/resolve/master/ShareGPT_V3_unfiltered_cleaned_split.json"

    mkdir -p "$output_dir"

    echo "Trying to download from Hugging Face..."
    if curl -fL --retry 3 -o "$output_file" "$url1"; then
        echo "Download succeeded from Hugging Face: $output_file"
    else
        echo "Failed to download from Hugging Face, trying ModelScope..."
        if curl -fL --retry 3 -o "$output_file" "$url2"; then
            echo "Download succeeded from ModelScope: $output_file"
        else
            echo "ERROR: Failed to download dataset from both sources." >&2
            return 1
        fi
    fi
}
