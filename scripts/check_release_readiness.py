import os
import sys

def check_file(path, label):
    if os.path.exists(path):
        print(f"[OK] {label} found: {path}")
        return True
    else:
        print(f"[FAIL] {label} MISSING: {path}")
        return False

def check_readiness():
    ready = True
    
    # Backend checks
    ready &= check_file("pipeline/v2_runner.py", "V2 Runner")
    ready &= check_file("pyproject.toml", "Python project config")
    ready &= check_file("docs/ANDROID_CONTRACT_v2.md", "Android Contract Doc")
    
    # Android checks
    powerai_root = r"C:\Users\zgz31\AndroidStudioProjects\PowerAi"
    ready &= check_file(os.path.join(powerai_root, "core/model-contract/src/main/java/com/example/powerai/core/model/KnowledgeBlock.kt"), "KnowledgeBlock contract")
    ready &= check_file(os.path.join(powerai_root, "app/src/main/res/xml/data_extraction_rules.xml"), "Data extraction rules")
    
    # Check for sensitive files in git (heuristic)
    if os.path.exists("apikey.txt"):
        print("[WARNING] apikey.txt exists! Ensure it's not committed.")

    if ready:
        print("\nSUMMARY: Project is structuraly ready for release/handoff.")
    else:
        print("\nSUMMARY: Project has missing components.")
        sys.exit(1)

if __name__ == "__main__":
    check_readiness()
