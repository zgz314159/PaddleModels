import argparse
import sys
from pathlib import Path

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from utils.file_id_sanitizer import sanitize_asset_relative_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve manifest.json path for a KB fileId under app/src/main/assets/kb.")
    ap.add_argument("--file-id", required=True, help="KB fileId (taxonomy path, uses / separators)")
    ap.add_argument("--assets-root", default="app/src/main/assets/kb", help="Assets KB root")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    safe_file_id = sanitize_asset_relative_path((args.file_id or "").strip())

    # NOTE: The on-disk folder name is '截图' (keep as-is).
    p = (repo_root / args.assets_root / Path(safe_file_id.replace("/", "\\")) / "截图" / "manifest.json").resolve()
    print(str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
