import json
import sys
from pathlib import Path

# Two distinct authoritative JSON contracts. Do not conflate them:
# - IR schema:  Canonical IR, written as knowledge_base.v2.json by pipeline/v2_runner.py.
# - KB schema:  Android knowledge base, written as knowledge_base.json (consumed by PowerAi).
IR_SCHEMA_PATH = Path("contracts/knowledge-base.v2.schema.json")
KB_SCHEMA_PATH = Path("contracts/knowledge_base_schema_v2.json")


def check_schema(schema_path, label):
    if not schema_path.exists():
        print(f"[ERROR] {label} schema not found at {schema_path}")
        return False

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        print(
            f"[SUCCESS] {label} schema loaded from {schema_path}. "
            f"Title: {schema.get('title', 'n/a')}"
        )
    except Exception as e:
        print(f"[ERROR] Failed to parse {label} schema at {schema_path}: {e}")
        return False
    return True


def check_ir_schema():
    """Validate the Canonical IR schema (knowledge_base.v2.json)."""
    return check_schema(IR_SCHEMA_PATH, "IR")


def check_kb_schema():
    """Validate the Android knowledge base schema (knowledge_base.json)."""
    return check_schema(KB_SCHEMA_PATH, "KB")


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
    all_ok &= check_ir_schema()
    all_ok &= check_kb_schema()
    all_ok &= check_active_entrypoints()
    
    if all_ok:
        print("=== ALL CHECKS PASSED ===")
        sys.exit(0)
    else:
        print("=== SOME CHECKS FAILED ===")
        sys.exit(1)

if __name__ == "__main__":
    main()
