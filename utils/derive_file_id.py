import argparse
import sys
from pathlib import Path

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from utils.file_id_sanitizer import sanitize_asset_relative_path, sanitize_file_id_for_path_only


def derive_file_id(*, doc_path: str, prefix: str) -> str:
    p = Path((doc_path or "").strip().strip('"').strip("'"))
    stem = p.stem if p.name else "doc"
    stem = sanitize_file_id_for_path_only(stem, default="doc")

    pref = (prefix or "").strip().replace("\\", "/").strip("/")
    if pref:
        return sanitize_asset_relative_path(f"{pref}/{stem}")
    return sanitize_asset_relative_path(stem)


def main() -> int:
    ap = argparse.ArgumentParser(description="Derive a safe fileId from a doc path (and optional taxonomy prefix).")
    ap.add_argument("--doc", required=True, help="DOCX/PDF path used to derive fileId leaf slug")
    ap.add_argument("--prefix", default="", help="Optional taxonomy prefix, e.g. 铁路/规章制度/电力")
    args = ap.parse_args()

    file_id = derive_file_id(doc_path=args.doc, prefix=args.prefix)
    print(file_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
