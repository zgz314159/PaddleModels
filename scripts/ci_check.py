import subprocess
import sys
import os

def run_command(cmd, label):
    print(f"Running {label}: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[OK] {label} passed.")
        return True
    else:
        print(f"[FAIL] {label} failed.")
        print(result.stdout)
        print(result.stderr)
        return False

def main():
    success = True
    
    # 1. Syntax check all python files
    print("Checking syntax...")
    for root, dirs, files in os.walk("."):
        if any(d in root for d in [".venv", ".history", ".git"]): continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                if not run_command([sys.executable, "-m", "py_compile", path], f"Compile {path}"):
                    success = False

    # 2. Check JSON schemas
    if os.path.exists("contracts"):
        for schema in os.listdir("contracts"):
            if schema.endswith(".json"):
                print(f"[INFO] Schema found: {schema}")

    # 3. Readiness check
    if os.path.exists("scripts/check_release_readiness.py"):
        if not run_command([sys.executable, "scripts/check_release_readiness.py"], "Readiness Check"):
            success = False

    if not success:
        print("\nCI Checks FAILED.")
        sys.exit(1)
    else:
        print("\nCI Checks PASSED.")

if __name__ == "__main__":
    main()
