import gzip
import json
import glob
import os
from tqdm import tqdm

def extract_events(file_path, pid_offset):
    with gzip.open(file_path, 'rt') as f:
        trace = json.load(f)
    events = trace.get("traceEvents", trace.get("events", []))
    result = []
    for event in events:
        pid = event.get("pid")
        if isinstance(pid, int):
            event["pid"] = pid + pid_offset
        elif isinstance(pid, str) and pid.isdigit():
            event["pid"] = int(pid) + pid_offset
        result.append(event)
    # 添加进程标识，方便在 Perfetto 里识别
    result.append({
        "ph": "M",
        "pid": pid_offset,
        "name": "process_name",
        "args": {
            "name": f"{os.path.basename(file_path)}"
        }
    })
    return result

def merge_traces(input_files):
    all_events = []
    base_pid = 1000
    print(f"🔍 找到 {len(input_files)} 个 trace 文件，开始合并...")
    for i, f in enumerate(tqdm(input_files, desc="合并进度")):
        pid_offset = base_pid + i
        events = extract_events(f, pid_offset)
        all_events.extend(events)
    return {
        "traceEvents": all_events,
        "displayTimeUnit": "ns"
    }

if __name__ == "__main__":
    input_files = sorted(glob.glob("*.pt.trace.json.gz"))
    if not input_files:
        print("❌ 未找到任何 .pt.trace.json.gz 文件！")
        exit(1)

    merged = merge_traces(input_files)

    output_path = "merged_trace.json"
    print(f"💾 写入合并结果到 {output_path} ...")
    with open(output_path, "w") as out:
        json.dump(merged, out)

    print(f"✅ 合并完成，共处理 {len(input_files)} 个文件，输出文件大小约 {os.path.getsize(output_path) / 1024:.1f} KB")

