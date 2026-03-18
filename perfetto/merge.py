import json
import glob
import gzip
import os
import re
import argparse
from tqdm import tqdm

def extract_events(file_path, rank):
    with gzip.open(file_path, 'rt') as f:
        trace = json.load(f)
    events = trace.get("traceEvents", trace.get("events", []))
    result = []
    for event in events:
        event["pid"] = rank
        result.append(event)
    result.append({
        "ph": "M",
        "pid": rank,
        "name": "process_name",
        "args": {
            "name": f"{os.path.basename(file_path)}"
        }
    })
    return result

def merge_traces(input_files):
    all_events = []
    print(f"🔍 找到 {len(input_files)} 个 trace 文件，开始合并...")
    rank = 0
    for f in tqdm(input_files, desc="合并进度"):
        match = re.search(r'TP-(\d+)', f)
        if not match:
           print(f"⚠️ 文件名 {f} 未匹配到 rank，跳过")
           continue
        rank = int(match.group(1))
        events = extract_events(f, rank)
        all_events.extend(events)
        rank += 1
    return {
        "traceEvents": all_events,
        "displayTimeUnit": "ns"
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON trace files into one")
    parser.add_argument("--dir", "-d", type=str, default=".", help="trace 文件所在目录")
    parser.add_argument("--output", "-o", type=str, default="merged_trace.json", help="输出文件路径")
    args = parser.parse_args()

    input_files = sorted(glob.glob(os.path.join(args.dir, "*.json.gz")))
    if not input_files:
        print(f"❌ 在目录 {args.dir} 未找到任何 .json.gz 文件！")
        exit(1)

    merged = merge_traces(input_files)

    print(f"💾 写入合并结果到 {args.output} ...")
    with open(args.output, "w") as out:
        json.dump(merged, out)

    print(f"✅ 合并完成，共处理 {len(input_files)} 个文件，输出文件大小约 {os.path.getsize(args.output) / 1024:.1f} KB")
