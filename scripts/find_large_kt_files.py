import os

def find_large_files(root_dir, min_lines=300):
    large_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".kt"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if len(lines) >= min_lines:
                            large_files.append((path, len(lines)))
                except Exception:
                    pass
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    for path, count in large_files:
        print(f"{count:5} lines: {path}")

if __name__ == "__main__":
    find_large_files(r"C:\Users\zgz31\AndroidStudioProjects\PowerAi")
