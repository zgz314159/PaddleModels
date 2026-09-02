import argparse
import json
import sys
from pathlib import Path

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from utils.file_id_sanitizer import sanitize_asset_relative_path


def _has_visual_assets(kb: dict) -> bool:
    meta = (kb or {}).get("fileMetadata") or {}
    try:
        if int(meta.get("imagesCount") or 0) > 0:
            return True
    except Exception:
        pass

    entries = (kb or {}).get("entries") or []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if (e.get("kind") or "").strip().lower() == "table":
            return True
        img_uris = e.get("imageUris")
        if isinstance(img_uris, list) and any((str(x).strip() for x in img_uris if x is not None)):
            return True
        if (e.get("tableImageUri") or "").strip():
            return True

        # Future-proof: some KB formats may include blocks with image URIs.
        blocks = e.get("blocks")
        if isinstance(blocks, list):
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                uri = (b.get("uri") or b.get("imageUri") or "").strip()
                if uri:
                    return True

    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Detect whether knowledge_base.json contains visual assets (tables/images). "
            "Exit code: 0=no visuals (safe to skip PDF pipeline), 1=has visuals, 2=error."
        )
    )
    ap.add_argument("--file-id", required=True, help="KB fileId (taxonomy path, uses / separators)")
    ap.add_argument("--assets-root", default="app/src/main/assets/kb", help="Assets KB root")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    safe_file_id = sanitize_asset_relative_path((args.file_id or "").strip())
    kb_path = (repo_root / args.assets_root / Path(safe_file_id.replace("/", "\\")) / "knowledge_base.json").resolve()

    if not kb_path.exists():
        print(f"[Detect] knowledge_base.json not found: {kb_path}")
        return 2

    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception as ex:
        print(f"[Detect] Failed to read/parse JSON: {kb_path} err={ex}")
        return 2

    return 1 if _has_visual_assets(kb) else 0


if __name__ == "__main__":
    raise SystemExit(main())
