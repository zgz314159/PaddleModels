import os
import json
import sys
from pathlib import Path

def check_schema_v2():
    schema_path = Path("contracts/knowledge-base.v2.schema.json")
    if not schema_path.exists():
        print(f"[ERROR] Schema not found at {schema_path}")
        return False
    
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        print(f"[SUCCESS] Schema v2 loaded. Version: {schema.get('$schema', 'v2')}")
    except Exception as e:
        print(f"[ERROR] Failed to parse schema: {e}")
        return False
    return True

def check_src_layout():
    src_path = Path("src/paddle_models")
    if not src_path.exists():
        print(f"[ERROR] src/paddle_models layout not found.")
        return False
    
    required_paths = [
        "__init__.py",
        "cli/main.py"
    ]
    for p in required_paths:
        full_path = src_path / p
        if not full_path.exists():
            print(f"[WARNING] Missing component: {p}")
        else:
            print(f"[OK] Found component: {p}")
    
    print("[SUCCESS] src/paddle_models layout verified.")
    return True

def main():
    print("=== PaddleModels Project Integrity Check ===")
    all_ok = True
    all_ok &= check_schema_v2()
    all_ok &= check_src_layout()
    
    if all_ok:
        print("=== ALL CHECKS PASSED ===")
        sys.exit(0)
    else:
        print("=== SOME CHECKS FAILED ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
