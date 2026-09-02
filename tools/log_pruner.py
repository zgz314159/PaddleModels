#!/usr/bin/env python3
import os
import sys

def prune_log_file(file_path, max_kb=500, keep_last_lines=1000):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    file_size_kb = os.path.getsize(file_path) / 1024
    if file_size_kb <= max_kb:
        print(f"File size {file_size_kb:.1f} KB is within limit {max_kb} KB. Skipping.")
        return

    print(f"Pruning {file_path} ({file_size_kb:.1f} KB)...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        if len(lines) > keep_last_lines:
            header = "# PowerAi - Pruned Log\n\n(Older entries removed to save memory)\n\n---\n\n"
            pruned_lines = lines[-keep_last_lines:]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(header)
                f.writelines(pruned_lines)
            print(f"Done. Retained last {keep_last_lines} lines. New size: {os.path.getsize(file_path)/1024:.1f} KB.")
        else:
            print("File has fewer lines than keep_last_lines. No pruning performed.")
    except Exception as e:
        print(f"Error pruning file: {e}")

if __name__ == "__main__":
    target = r"C:\Users\zgz31\AndroidStudioProjects\PowerAi\TASK_LOG.md"
    prune_log_file(target)
