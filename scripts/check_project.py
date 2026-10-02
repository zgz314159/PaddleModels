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

def check_active_entrypoints():
    required_paths = [
        "src/paddle_models/cli/main.py",
        "pipeline/v2_runner.py"
    ]
    ok = True
    for p in required_paths:
        if not Path(p).exists():
            print(f"[ERROR] Missing active entrypoint: {p}")
            ok = False
        else:
            print(f"[OK] Found active entrypoint: {p}")
    return ok

def main():
    print("=== PaddleModels Project Integrity Check ===")
    all_ok = True
    all_ok &= check_schema_v2()
    all_ok &= check_active_entrypoints()
    
    if all_ok:
        print("=== ALL CHECKS PASSED ===")
        sys.exit(0)
    else:
        print("=== SOME CHECKS FAILED ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
