import argparse
import re
import sys
from pathlib import Path

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from utils.file_id_sanitizer import sanitize_asset_relative_path, sanitize_file_id_for_path_only

from PIL import Image

try:
    import pypdfium2 as pdfium
except Exception as ex:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: pypdfium2. Install with: pip install pypdfium2"
    ) from ex


def sanitize_folder_name(name: str) -> str:
    # PATH-ONLY: Use for directory/asset names.
    return sanitize_asset_relative_path(name)


def normalize_for_search_only(name: str) -> str:
    """Normalize a string for fuzzy *matching only*.

    IMPORTANT:
    - This function is ONLY for fuzzy matching document filenames.
    - It is NOT allowed for generating folders, writing files, or building Android asset paths.
      For paths, always use `sanitize_file_id_for_folder`.

    Behavior:
    - Drops whitespace and most punctuation so users don't need to type exact em-dash variants.
    - Keeps CJK + ASCII letters + digits.
    """

    s = (name or "").strip().lower()
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", s)
    return s


def _resolve_pdf_path(project_root: Path, pdf_arg: str) -> Path:
    p = Path(pdf_arg)
    if p.exists():
        return p

    # Try relative to repo root.
    p2 = project_root / pdf_arg
    if p2.exists():
        return p2

    # Fuzzy match inside assets/原文档
    assets_docs = project_root / "app" / "src" / "main" / "assets" / "原文档"
    if assets_docs.exists():
        want = normalize_for_search_only(p.name)
        candidates = [x for x in assets_docs.iterdir() if x.is_file() and x.suffix.lower() == ".pdf"]

        matches = []
        for c in candidates:
            if normalize_for_search_only(c.name) == want or normalize_for_search_only(c.stem) == want:
                matches.append(c)

        if len(matches) == 1:
            return matches[0]

    return p


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Render PDF pages into kb/<fileId>/pages for manual/semi-auto cropping. "
            "This is the first step toward automatic 原件截图 generation."
        )
    )
    ap.add_argument(
        "--pdf",
        required=True,
        help="Input PDF path",
    )
    ap.add_argument(
        "--file-id",
        default="auto",
        help=(
            "Relative folder path under app/src/main/assets/kb (建议与 build 脚本 --file-id 一致). "
            "Use 'auto' to infer from the PDF filename stem."
        ),
    )
    ap.add_argument(
        "--out-root",
        default="app/src/main/assets/kb",
        help="Output root folder",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Render DPI (180 is a good default for tables)",
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Limit pages for quick runs (0 = all)",
    )

    args = ap.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = _resolve_pdf_path(project_root, args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    out_root = (project_root / args.out_root).resolve()

    # PATH-ONLY: Use for directory/asset names.
    file_id_raw = (args.file_id or "").strip()
    if file_id_raw.lower() in ("", "auto"):
        file_id_raw = pdf_path.stem
    safe_file_id = sanitize_folder_name(file_id_raw)
    out_pages = out_root / safe_file_id / "pages"
    out_pages.mkdir(parents=True, exist_ok=True)

    scale = args.dpi / 72.0

    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    limit = args.max_pages if args.max_pages and args.max_pages > 0 else total
    limit = min(limit, total)

    written = 0
    for i in range(limit):
        page_no = i + 1
        page = pdf[i]

        # render() returns a PdfBitmap wrapper.
        bitmap = page.render(scale=scale)
        pil_image: Image.Image = bitmap.to_pil()

        out_path = out_pages / f"page_{page_no:03d}_p{page_no}.png"
        pil_image.save(out_path, format="PNG")
        written += 1

    print(f"Wrote {written} page images to: {out_pages}")
    print("Next: crop table/legend screenshots into 截图/ and name as table_p{page}_pos{position}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
