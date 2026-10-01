import argparse
import json
import math
import re
import time
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 自动注入项目根目录到 sys.path（兼容从任意工作目录执行） ──
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from utils.file_id_sanitizer import sanitize_asset_relative_path, sanitize_file_id_for_path_only


try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from skimage.segmentation import felzenszwalb
except Exception as ex:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: opencv-python + numpy + scikit-image\n\n"
        "Install with:\n"
        "  pip install opencv-python numpy scikit-image\n"
    ) from ex


def _maybe_configure_tesseract_cmd() -> None:
    """Optionally set pytesseract.tesseract_cmd from env.

    This helps on Windows where tesseract.exe may not be on PATH.
    """

    try:
        import os

        cmd = (os.environ.get("TESSERACT_CMD") or "").strip()
        if not cmd:
            return
        import pytesseract  # type: ignore

        pytesseract.pytesseract.tesseract_cmd = cmd
    except Exception:
        return


def _clean_ocr_text(text: str) -> str:
    if not text:
        return ""
    # collapse whitespace, keep Chinese/ASCII
    text = re.sub(r"[\t\r\n]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_non_blank_ocr_lines(text: str) -> List[str]:
    if not text:
        return []
    lines: List[str] = []
    for raw in re.split(r"[\r\n]+", text):
        compact = re.sub(r"\s+", "", raw or "").strip()
        if compact:
            lines.append(compact)
    return lines


def _ocr_region_text_lines(
    img_bgr: "np.ndarray",
    *,
    lang: str,
    debug: bool = False,
    debug_label: str = "",
) -> List[str]:
    """OCR helper that preserves line boundaries for layout classification."""

    try:
        from paddle_ocr_util import ocr_image_to_text

        text = ocr_image_to_text(img_bgr)
        lines = _extract_non_blank_ocr_lines(text)
        if lines:
            return lines
    except Exception:
        pass

    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return []

    _maybe_configure_tesseract_cmd()

    try:
        if img_bgr is None or img_bgr.size == 0:
            return []

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        if max(gray.shape[:2]) < 900:
            gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)

        _thr_val, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        candidates: List[Tuple[str, "np.ndarray"]] = [
            ("otsu", bw_otsu),
            ("adaptive", bw_adaptive),
            ("gray", gray),
        ]
        psm_values = [6, 11, 4]

        best_lines: List[str] = []
        best_score = -1
        for tag, candidate in candidates:
            prepared = cv2.copyMakeBorder(candidate, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            pil_img = Image.fromarray(prepared)
            for psm in psm_values:
                config = f"--psm {psm}"
                raw_text = pytesseract.image_to_string(pil_img, lang=lang, config=config)
                lines = _extract_non_blank_ocr_lines(raw_text)
                if not lines:
                    continue
                score = sum(len(line) for line in lines)
                if score > best_score:
                    best_lines = lines
                    best_score = score
                if len(lines) >= 4 or score >= 18:
                    return lines

        return best_lines
    except Exception as ex:
        if debug:
            try:
                msg = str(ex).strip().replace("\n", " ")
                print(f"[OCRDebug] OCR lines failed {debug_label}: {msg}")
            except Exception:
                pass
        return []


def _ocr_anchor_text(
    img_bgr: "np.ndarray",
    *,
    lang: str,
    debug: bool = False,
    debug_label: str = "",
) -> str:
    """OCR helper.

    Requires `pytesseract` + `Pillow` and a system tesseract installation.
    Returns cleaned text (may be empty).
    """

    # First try PaddleOCR (GPU) via helper wrapper.
    try:
        from paddle_ocr_util import ocr_image_to_text

        # paddle_ocr_util accepts numpy arrays or image paths
        text = ocr_image_to_text(img_bgr)
        if text:
            return _clean_ocr_text(text)
    except Exception:
        # PaddleOCR not available or failed; fallback to pytesseract below
        pass

    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""

    _maybe_configure_tesseract_cmd()

    try:
        if img_bgr is None or img_bgr.size == 0:
            return ""

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        _thr_val, bw_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bw_adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )

        candidates: List[Tuple[str, "np.ndarray"]] = [
            ("otsu", bw_otsu),
            ("adaptive", bw_adaptive),
            ("gray", gray),
        ]
        psm_values = [7, 6, 11] if gray.shape[0] <= 120 else [6, 11, 7]

        best_text = ""
        best_score = -1
        for tag, candidate in candidates:
            prepared = cv2.copyMakeBorder(candidate, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            pil_img = Image.fromarray(prepared)
            for psm in psm_values:
                config = f"--psm {psm}"
                text = _clean_ocr_text(pytesseract.image_to_string(pil_img, lang=lang, config=config))
                if not text:
                    continue
                score = len(text)
                if score > best_score:
                    best_text = text
                    best_score = score
                if score >= 4:
                    return text

        return best_text
    except Exception as ex:
        if debug:
            try:
                msg = str(ex).strip().replace("\n", " ")
                print(f"[OCRDebug] OCR failed {debug_label}: {msg}")
            except Exception:
                pass
        return ""


def _find_text_runs_in_band(
    band_bgr: "np.ndarray",
    *,
    min_row_density: float = 0.012,
    min_run_height: int = 6,
    join_gap: int = 3,
) -> List[Tuple[int, int]]:
    """Detect horizontal text-line runs using row projection.

    Returns a list of (y0, y1) runs in band-local coordinates.
    """

    if band_bgr is None or band_bgr.size == 0:
        return []

    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    # Invert so text strokes are white (non-zero)
    _t, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.medianBlur(bw, 3)

    h, w = bw.shape[:2]
    if h <= 0 or w <= 0:
        return []

    row_counts = np.count_nonzero(bw, axis=1)
    row_density = row_counts.astype(np.float32) / float(w)

    mask = row_density >= float(min_row_density)
    runs: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        if not bool(mask[y]):
            y += 1
            continue
        y0 = y
        while y < h and bool(mask[y]):
            y += 1
        y1 = y
        if (y1 - y0) >= int(min_run_height):
            runs.append((int(y0), int(y1)))

    if not runs:
        return []

    # Join close runs (handle broken strokes)
    merged: List[Tuple[int, int]] = []
    cur0, cur1 = runs[0]
    for r0, r1 in runs[1:]:
        if r0 - cur1 <= int(join_gap):
            cur1 = max(cur1, r1)
        else:
            merged.append((cur0, cur1))
            cur0, cur1 = r0, r1
    merged.append((cur0, cur1))
    return merged


def _refine_band_to_nearest_lines(
    band_bgr: "np.ndarray",
    *,
    keep_last_runs: int = 2,
) -> Optional[Tuple[int, int]]:
    """Keep the last 1-2 detected text runs closest to the band bottom."""

    runs = _find_text_runs_in_band(band_bgr)
    if not runs:
        return None
    k = max(1, int(keep_last_runs))
    sel = runs[-k:]
    y0 = min(r[0] for r in sel)
    y1 = max(r[1] for r in sel)
    # small padding
    y0 = max(0, y0 - 2)
    y1 = min(band_bgr.shape[0], y1 + 2)
    if y1 - y0 < 8:
        return None
    return (int(y0), int(y1))


def _dynamic_anchor_probe_box(
    img: "np.ndarray",
    *,
    x0: int,
    x1: int,
    y_top: int,
    y_bottom: int,
    debug_record_box: bool = True,
) -> Tuple[Optional[Dict[str, int]], List[Dict[str, int]]]:
    """Probe a vertical band, refine to nearest text lines, return box(es) only.

    OCR should be performed once later on the union box of all fragments.
    """

    band = img[y_top:y_bottom, x0:x1]
    if band.size == 0:
        return None, []

    refined = _refine_band_to_nearest_lines(band, keep_last_runs=2)
    if refined is None:
        # No detectable text runs; optionally record the searched band box for debugging.
        if not debug_record_box:
            return None, []
        band_box = {
            "left": int(x0),
            "top": int(y_top),
            "right": int(x1),
            "bottom": int(y_bottom),
            "width": int(x1 - x0),
            "height": int(y_bottom - y_top),
        }
        return None, [band_box]

    ry0, ry1 = refined
    fy0 = int(y_top + ry0)
    fy1 = int(y_top + ry1)
    box = {
        "left": int(x0),
        "top": int(fy0),
        "right": int(x1),
        "bottom": int(fy1),
        "width": int(x1 - x0),
        "height": int(fy1 - fy0),
    }
    return box, [box]


def _union_boxes(boxes: List[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if not boxes:
        return None
    try:
        left = min(int(b["left"]) for b in boxes)
        top = min(int(b["top"]) for b in boxes)
        right = max(int(b["right"]) for b in boxes)
        bottom = max(int(b["bottom"]) for b in boxes)
        if right - left < 2 or bottom - top < 2:
            return None
        return {
            "left": int(left),
            "top": int(top),
            "right": int(right),
            "bottom": int(bottom),
            "width": int(right - left),
            "height": int(bottom - top),
        }
    except Exception:
        return None


@dataclass(frozen=True)
class CropCandidate:
    kind: str  # 'table' | 'legend'
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    method: str


def sanitize_folder_name(name: str) -> str:
    # PATH-ONLY: Use for directory/asset names.
    return sanitize_asset_relative_path(name)


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _parse_page_number(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v if v > 0 else None
    if isinstance(v, str) and v.strip().isdigit():
        try:
            page = int(v.strip())
            return page if page > 0 else None
        except Exception:
            return None
    return None


def _collect_entry_pages(entry: Dict[str, Any]) -> List[int]:
    pages: List[int] = []
    direct_page = _parse_page_number(entry.get("pageNumber"))
    if direct_page is not None:
        pages.append(int(direct_page))

    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return pages
    for block in blocks:
        if not isinstance(block, dict):
            continue
        page = _parse_page_number(block.get("pageNumber"))
        if page is not None:
            pages.append(int(page))
    return pages


def _entry_page_hint(entry: Dict[str, Any]) -> Optional[int]:
    pages = _collect_entry_pages(entry)
    if not pages:
        return None
    counts: Dict[int, int] = {}
    for page in pages:
        counts[int(page)] = counts.get(int(page), 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _entry_kind_hint(entry: Dict[str, Any], title: str) -> str:
    kind = _safe_str(entry.get("kind")).strip().lower()
    if kind:
        return kind
    title_text = _safe_str(title).strip()
    if title_text.startswith("表") or title_text.startswith("表格#"):
        return "table"
    if "附件" in title_text or "附表" in title_text or "附图" in title_text:
        return "appendix"
    if "图" in title_text:
        return "image"
    return "unknown"


def _load_kb_heading_hints(root: Path, safe_file_id: str) -> Dict[int, List[Dict[str, Any]]]:
    kb_path = root / safe_file_id / "knowledge_base.json"
    if not kb_path.exists():
        return {}

    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}

    entries = kb.get("entries")
    if not isinstance(entries, list):
        return {}

    page_hints: Dict[int, List[Dict[str, Any]]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        title = _safe_str(entry.get("jobTitle") or entry.get("title")).strip()
        if not title:
            continue
        page = _entry_page_hint(entry)
        if page is None:
            continue
        hint = {
            "title": title,
            "kind": _entry_kind_hint(entry, title),
            "position": int(entry.get("position") if isinstance(entry.get("position"), int) else idx),
        }
        page_hints.setdefault(int(page), []).append(hint)

    for hints in page_hints.values():
        hints.sort(key=lambda hint: (int(hint.get("position") or 0), _safe_str(hint.get("title"))))
    return page_hints


def _kind_matches_heading_hint(crop_kind: str, hint: Dict[str, Any]) -> bool:
    hint_kind = _safe_str(hint.get("kind")).strip().lower()
    title = _safe_str(hint.get("title")).strip()
    if crop_kind == "table":
        return hint_kind == "table" or title.startswith("表")
    if crop_kind == "legend":
        if hint_kind in {"appendix", "image", "legend", "unknown"}:
            return True
        return ("附件" in title) or ("附表" in title) or ("图" in title)
    return True


def _collect_heading_hints_for_page(
    page_hints: Dict[int, List[Dict[str, Any]]],
    page_no: int,
    crop_kind: str,
    *,
    radius: int = 2,
) -> List[Dict[str, Any]]:
    all_hints: List[Dict[str, Any]] = []
    for delta in range(0, max(0, int(radius)) + 1):
        candidate_pages = [page_no] if delta == 0 else [page_no - delta, page_no + delta]
        for candidate_page in candidate_pages:
            if candidate_page <= 0:
                continue
            all_hints.extend(page_hints.get(int(candidate_page), []))

    if not all_hints:
        return []

    preferred = [hint for hint in all_hints if _kind_matches_heading_hint(crop_kind, hint)]
    return preferred or all_hints


def _pick_heading_text_for_crop(
    page_hints: Dict[int, List[Dict[str, Any]]],
    page_no: int,
    crop_kind: str,
    offsets: Dict[str, int],
) -> str:
    hints = _collect_heading_hints_for_page(page_hints, page_no, crop_kind)
    if not hints:
        return ""

    cursor_key = crop_kind if len(hints) > 1 else f"{crop_kind}:sticky"
    cursor = max(0, int(offsets.get(cursor_key, 0)))
    idx = min(cursor, len(hints) - 1)
    if len(hints) > 1:
        offsets[cursor_key] = idx + 1
    return _safe_str(hints[idx].get("title")).strip()


def _iter_page_images(pages_dir: Path) -> List[Tuple[int, Path]]:
    items: List[Tuple[int, Path]] = []
    if not pages_dir.exists():
        return items

    for p in pages_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue

        # Expected format: page_001_p1.png (export_pdf_pages_to_original_screenshots.py)
        m = re.search(r"(?:^|_)p(\d+)(?:\.|$)", p.name)
        if not m:
            # fallback: any number in filename
            m2 = re.search(r"(\d+)", p.stem)
            if not m2:
                continue
            page_no = int(m2.group(1))
        else:
            page_no = int(m.group(1))

        items.append((page_no, p))

    items.sort(key=lambda t: t[0])
    return items


def _clip_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    if x1 <= x0 + 1:
        x1 = min(w, x0 + 2)
    if y1 <= y0 + 1:
        y1 = min(h, y0 + 2)
    return x0, y0, x1, y1


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return float(inter) / float(denom)


def _nms(cands: List[CropCandidate], iou_threshold: float) -> List[CropCandidate]:
    if not cands:
        return []
    kept: List[CropCandidate] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        box_c = (c.x0, c.y0, c.x1, c.y1)
        if any(_iou(box_c, (k.x0, k.y0, k.x1, k.y1)) >= iou_threshold for k in kept):
            continue
        kept.append(c)
    return kept


def _bbox_overlap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(int(a1), int(b1)) - max(int(a0), int(b0)))


def _bbox_gap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, max(int(a0), int(b0)) - min(int(a1), int(b1)))


def _count_line_intersections(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    *,
    tol: int = 3,
) -> int:
    intersections = 0
    for hx0, hy0, hx1, hy1 in h_lines:
        y = int(round((hy0 + hy1) / 2.0))
        x_min = min(hx0, hx1) - int(tol)
        x_max = max(hx0, hx1) + int(tol)
        for vx0, vy0, vx1, vy1 in v_lines:
            x = int(round((vx0 + vx1) / 2.0))
            y_min = min(vy0, vy1) - int(tol)
            y_max = max(vy0, vy1) + int(tol)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                intersections += 1
    return intersections


def _count_long_line_components(mask: "np.ndarray", *, axis: str, min_span: int) -> int:
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if axis == "h":
            if w >= int(min_span) and h <= max(18, int(mask.shape[0] * 0.55)):
                count += 1
        else:
            if h >= int(min_span) and w <= max(18, int(mask.shape[1] * 0.55)):
                count += 1
    return count


def _collect_mask_row_runs(
    mask: "np.ndarray",
    *,
    min_row_density: float,
    min_run_height: int,
    join_gap: int,
) -> List[Tuple[int, int]]:
    if mask is None or mask.size == 0:
        return []

    h, w = mask.shape[:2]
    if h <= 0 or w <= 0:
        return []

    row_counts = np.count_nonzero(mask, axis=1)
    row_density = row_counts.astype(np.float32) / float(max(1, w))
    active = row_density >= float(min_row_density)

    runs: List[Tuple[int, int]] = []
    y = 0
    while y < h:
        if not bool(active[y]):
            y += 1
            continue
        y0 = y
        gap = 0
        y += 1
        while y < h:
            if bool(active[y]):
                gap = 0
                y += 1
                continue
            gap += 1
            if gap > int(join_gap):
                break
            y += 1
        y1 = max(y0 + 1, y - gap)
        if (y1 - y0) >= int(min_run_height):
            runs.append((int(y0), int(y1)))
    return runs


def _crop_text_signal_stats(gray: "np.ndarray", cand: CropCandidate) -> Dict[str, float]:
    roi = gray[cand.y0 : cand.y1, cand.x0 : cand.x1]
    if roi.size == 0:
        return {
            "text_runs": 0.0,
            "text_components": 0.0,
            "text_density": 0.0,
            "max_run_width_ratio": 0.0,
            "median_run_width_ratio": 0.0,
        }

    rh, rw = roi.shape[:2]
    if rw < 80 or rh < 60:
        return {
            "text_runs": 0.0,
            "text_components": 0.0,
            "text_density": 0.0,
            "max_run_width_ratio": 0.0,
            "median_run_width_ratio": 0.0,
        }

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )

    horiz_len = max(18, int(rw * 0.18))
    vert_len = max(18, int(rh * 0.18))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    horiz = cv2.erode(thr, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)
    vert = cv2.erode(thr, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)
    grid = cv2.bitwise_or(horiz, vert)

    text_mask = cv2.bitwise_and(thr, cv2.bitwise_not(grid))
    text_mask = cv2.medianBlur(text_mask, 3)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(text_mask, connectivity=8)
    filtered = np.zeros_like(text_mask)
    component_count = 0

    for label in range(1, int(num_labels)):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue
        if w <= 1 or h <= 1:
            continue
        if w > int(rw * 0.45) or h > int(rh * 0.16):
            continue
        if area > int(rw * rh * 0.015):
            continue
        fill_ratio = float(area) / float(max(1, w * h))
        if fill_ratio >= 0.95 and (w >= int(rw * 0.10) or h >= int(rh * 0.10)):
            continue
        filtered[labels == label] = 255
        component_count += 1

    if component_count <= 0:
        return {
            "text_runs": 0.0,
            "text_components": 0.0,
            "text_density": 0.0,
            "max_run_width_ratio": 0.0,
            "median_run_width_ratio": 0.0,
        }

    filtered = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)
    runs = _collect_mask_row_runs(
        filtered,
        min_row_density=max(0.007, min(0.020, 10.0 / float(max(1, rw)))),
        min_run_height=2,
        join_gap=2,
    )

    width_ratios: List[float] = []
    for y0, y1 in runs:
        band = filtered[y0:y1, :]
        col_counts = np.count_nonzero(band, axis=0)
        active_cols = np.flatnonzero(col_counts > 0)
        if active_cols.size <= 0:
            continue
        run_width = int(active_cols[-1] - active_cols[0] + 1)
        width_ratios.append(float(run_width) / float(max(1, rw)))

    width_ratios.sort()
    median_width_ratio = width_ratios[len(width_ratios) // 2] if width_ratios else 0.0
    return {
        "text_runs": float(len(runs)),
        "text_components": float(component_count),
        "text_density": float(cv2.countNonZero(filtered)) / float(max(1, rw * rh)),
        "max_run_width_ratio": float(max(width_ratios) if width_ratios else 0.0),
        "median_run_width_ratio": float(median_width_ratio),
    }


def _crop_ocr_text_signal_stats(gray: "np.ndarray", cand: CropCandidate) -> Dict[str, float]:
    roi = gray[cand.y0 : cand.y1, cand.x0 : cand.x1]
    if roi.size == 0:
        return {
            "line_count": 0.0,
            "avg_line_len": 0.0,
            "short_line_ratio": 0.0,
        }

    rh, rw = roi.shape[:2]
    if rw < 80 or rh < 60:
        return {
            "line_count": 0.0,
            "avg_line_len": 0.0,
            "short_line_ratio": 0.0,
        }

    if max(rh, rw) > 1600:
        scale = 1600.0 / float(max(rh, rw))
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    lines = _ocr_region_text_lines(roi_bgr, lang="chi_sim+eng")
    if not lines:
        return {
            "line_count": 0.0,
            "avg_line_len": 0.0,
            "short_line_ratio": 0.0,
        }

    lengths = [len(line) for line in lines if line]
    if not lengths:
        return {
            "line_count": 0.0,
            "avg_line_len": 0.0,
            "short_line_ratio": 0.0,
        }

    short_line_ratio = float(sum(1 for length in lengths if length <= 10)) / float(max(1, len(lengths)))
    return {
        "line_count": float(len(lengths)),
        "avg_line_len": float(sum(lengths)) / float(max(1, len(lengths))),
        "short_line_ratio": float(short_line_ratio),
    }


def _has_table_text_evidence(
    text_stats: Dict[str, float],
    ocr_stats: Dict[str, float],
) -> bool:
    text_runs = int(text_stats.get("text_runs", 0.0) or 0)
    text_density = float(text_stats.get("text_density", 0.0) or 0.0)
    max_run_width_ratio = float(text_stats.get("max_run_width_ratio", 0.0) or 0.0)
    median_run_width_ratio = float(text_stats.get("median_run_width_ratio", 0.0) or 0.0)

    line_count = int(ocr_stats.get("line_count", 0.0) or 0)
    avg_line_len = float(ocr_stats.get("avg_line_len", 0.0) or 0.0)
    short_line_ratio = float(ocr_stats.get("short_line_ratio", 0.0) or 0.0)

    if line_count >= 6 and text_runs >= 6 and median_run_width_ratio >= 0.60 and max_run_width_ratio >= 0.80:
        return True

    if line_count >= 8 and text_runs >= 8 and median_run_width_ratio >= 0.45 and max_run_width_ratio >= 0.60:
        return True

    if line_count >= 12 and short_line_ratio >= 0.45 and max_run_width_ratio >= 0.60:
        return True

    if line_count <= 0:
        return bool(text_runs >= 10 and max_run_width_ratio >= 0.70 and median_run_width_ratio >= 0.45 and text_density >= 0.015)

    return False


def _crop_table_signal_stats(gray: "np.ndarray", cand: CropCandidate) -> Dict[str, float]:
    roi = gray[cand.y0 : cand.y1, cand.x0 : cand.x1]
    if roi.size == 0:
        return {
            "grid_density": 0.0,
            "h_long": 0.0,
            "v_long": 0.0,
            "intersections": 0.0,
        }

    rh, rw = roi.shape[:2]
    if rw < 80 or rh < 60:
        return {
            "grid_density": 0.0,
            "h_long": 0.0,
            "v_long": 0.0,
            "intersections": 0.0,
        }

    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )

    horiz_len = max(18, int(rw * 0.18))
    vert_len = max(18, int(rh * 0.18))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    horiz = cv2.erode(thr, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)
    vert = cv2.erode(thr, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)
    grid = cv2.bitwise_or(horiz, vert)

    h_long = _count_long_line_components(horiz, axis="h", min_span=max(24, int(rw * 0.45)))
    v_long = _count_long_line_components(vert, axis="v", min_span=max(24, int(rh * 0.45)))

    h_lines: List[Tuple[int, int, int, int]] = []
    v_lines: List[Tuple[int, int, int, int]] = []
    try:
        lines = cv2.HoughLinesP(
            grid,
            1,
            np.pi / 180,
            threshold=25,
            minLineLength=max(18, int(min(rw, rh) * 0.20)),
            maxLineGap=6,
        )
    except Exception:
        lines = None

    if lines is not None:
        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy <= max(2, int(dx * 0.18)) and dx >= max(24, int(rw * 0.38)):
                h_lines.append((int(x1), int(y1), int(x2), int(y2)))
            elif dx <= max(2, int(dy * 0.18)) and dy >= max(24, int(rh * 0.30)):
                v_lines.append((int(x1), int(y1), int(x2), int(y2)))

    intersections = _count_line_intersections(h_lines, v_lines, tol=3)
    grid_density = float(cv2.countNonZero(grid)) / float(max(1, rw * rh))

    return {
        "grid_density": float(grid_density),
        "h_long": float(max(h_long, len(h_lines))),
        "v_long": float(max(v_long, len(v_lines))),
        "intersections": float(intersections),
    }


def _is_strong_table_candidate(gray: "np.ndarray", cand: CropCandidate) -> bool:
    stats = _crop_table_signal_stats(gray, cand)
    h_long = int(stats["h_long"])
    v_long = int(stats["v_long"])
    intersections = int(stats["intersections"])
    grid_density = float(stats["grid_density"])

    structure_table = bool(
        (intersections >= 5 and h_long >= 3 and v_long >= 2)
        or (intersections >= 7 and grid_density >= 0.028)
        or (h_long >= 4 and v_long >= 3 and grid_density >= 0.022)
    )
    if not structure_table:
        return False

    text_stats = _crop_text_signal_stats(gray, cand)
    text_runs = int(text_stats.get("text_runs", 0.0) or 0)
    max_run_width_ratio = float(text_stats.get("max_run_width_ratio", 0.0) or 0.0)

    if text_runs <= 1 and max_run_width_ratio <= 0.20 and intersections < 24:
        return False

    ocr_stats = _crop_ocr_text_signal_stats(gray, cand)
    return _has_table_text_evidence(text_stats, ocr_stats)


def _group_visual_fragment_candidates(
    cands: List[CropCandidate],
    *,
    page_w: int,
    page_h: int,
) -> List[CropCandidate]:
    if len(cands) <= 1:
        return list(cands)

    def _cand_area(cand: CropCandidate) -> int:
        return max(1, int(cand.x1 - cand.x0)) * max(1, int(cand.y1 - cand.y0))

    pruned = list(cands)
    suppressed: set[int] = set()
    by_area = sorted(range(len(pruned)), key=lambda idx: _cand_area(pruned[idx]), reverse=True)
    for pos, large_idx in enumerate(by_area):
        if large_idx in suppressed:
            continue
        large = pruned[large_idx]
        large_area = float(_cand_area(large))
        for small_idx in by_area[pos + 1 :]:
            if small_idx in suppressed:
                continue
            small = pruned[small_idx]
            small_area = float(_cand_area(small))
            inter_w = _bbox_overlap_1d(large.x0, large.x1, small.x0, small.x1)
            inter_h = _bbox_overlap_1d(large.y0, large.y1, small.y0, small.y1)
            inter_area = float(inter_w * inter_h)
            if inter_area <= 0.0:
                continue
            covered_ratio = inter_area / float(max(1.0, small_area))
            relative_area = small_area / float(max(1.0, large_area))
            if covered_ratio >= 0.78 and relative_area <= 0.38:
                suppressed.add(small_idx)

    cands = [cand for idx, cand in enumerate(pruned) if idx not in suppressed]
    if len(cands) <= 1:
        return list(cands)

    widths = sorted(max(1, c.x1 - c.x0) for c in cands)
    heights = sorted(max(1, c.y1 - c.y0) for c in cands)
    median_w = widths[len(widths) // 2]
    median_h = heights[len(heights) // 2]

    same_row_gap = max(18, min(int(page_w * 0.04), int(max(median_w * 0.18, median_h * 0.10))))
    same_col_gap = max(18, min(int(page_h * 0.03), int(max(median_h * 0.16, median_w * 0.08))))

    adjacency: Dict[int, List[int]] = {idx: [] for idx in range(len(cands))}

    for left_idx in range(len(cands)):
        left = cands[left_idx]
        left_w = max(1, int(left.x1 - left.x0))
        left_h = max(1, int(left.y1 - left.y0))
        left_area = max(1, left_w * left_h)
        for right_idx in range(left_idx + 1, len(cands)):
            right = cands[right_idx]
            right_w = max(1, int(right.x1 - right.x0))
            right_h = max(1, int(right.y1 - right.y0))
            right_area = max(1, right_w * right_h)

            inter_w = _bbox_overlap_1d(left.x0, left.x1, right.x0, right.x1)
            inter_h = _bbox_overlap_1d(left.y0, left.y1, right.y0, right.y1)
            inter_area = inter_w * inter_h

            min_area = float(min(left_area, right_area))
            vert_overlap_ratio = float(inter_h) / float(max(1, min(left_h, right_h)))
            horiz_overlap_ratio = float(inter_w) / float(max(1, min(left_w, right_w)))
            horiz_gap = _bbox_gap_1d(left.x0, left.x1, right.x0, right.x1)
            vert_gap = _bbox_gap_1d(left.y0, left.y1, right.y0, right.y1)

            should_group = False
            if inter_area > 0 and (float(inter_area) / float(max(1.0, min_area))) >= 0.30:
                should_group = True
            elif vert_overlap_ratio >= 0.28 and horiz_gap <= same_row_gap:
                should_group = True
            elif horiz_overlap_ratio >= 0.38 and vert_gap <= same_col_gap:
                should_group = True

            if should_group:
                adjacency[left_idx].append(right_idx)
                adjacency[right_idx].append(left_idx)

    groups: Dict[int, List[CropCandidate]] = {}
    visited: set[int] = set()
    next_label = 1
    for start_idx in range(len(cands)):
        if start_idx in visited:
            continue
        stack = [start_idx]
        members: List[CropCandidate] = []
        while stack:
            idx = stack.pop()
            if idx in visited:
                continue
            visited.add(idx)
            members.append(cands[idx])
            stack.extend(adjacency.get(idx, []))
        groups[next_label] = members
        next_label += 1

    merged: List[CropCandidate] = []
    for _label, members in groups.items():
        if not members:
            continue
        if len(members) == 1:
            merged.append(members[0])
            continue

        union_x0 = min(m.x0 for m in members)
        union_y0 = min(m.y0 for m in members)
        union_x1 = max(m.x1 for m in members)
        union_y1 = max(m.y1 for m in members)
        pad_x = max(8, min(same_row_gap // 2, 24))
        pad_y = max(8, min(same_col_gap // 2, 24))
        x0, y0, x1, y1 = _clip_box(
            union_x0 - pad_x,
            union_y0 - pad_y,
            union_x1 + pad_x,
            union_y1 + pad_y,
            page_w,
            page_h,
        )

        merged.append(
            CropCandidate(
                kind="legend",
                x0=int(x0),
                y0=int(y0),
                x1=int(x1),
                y1=int(y1),
                score=max(float(m.score) for m in members) + min(0.12, 0.03 * float(len(members) - 1)),
                method="grouped_visual_fragments",
            )
        )

    return merged


def _reclassify_and_group_candidates(
    gray: "np.ndarray",
    cands: List[CropCandidate],
    *,
    page_w: int,
    page_h: int,
) -> List[CropCandidate]:
    if not cands:
        return []

    strong_tables: List[CropCandidate] = []
    visual_fragments: List[CropCandidate] = []
    for cand in cands:
        if _is_strong_table_candidate(gray, cand):
            strong_tables.append(
                CropCandidate(
                    kind="table",
                    x0=int(cand.x0),
                    y0=int(cand.y0),
                    x1=int(cand.x1),
                    y1=int(cand.y1),
                    score=max(float(cand.score), 0.25),
                    method=(cand.method if cand.kind == "table" else "table_reclassified"),
                )
            )
        else:
            visual_fragments.append(
                CropCandidate(
                    kind="legend",
                    x0=int(cand.x0),
                    y0=int(cand.y0),
                    x1=int(cand.x1),
                    y1=int(cand.y1),
                    score=float(cand.score),
                    method=(cand.method if cand.kind == "legend" else "legend_fragment"),
                )
            )

    grouped_legends = _group_visual_fragment_candidates(
        visual_fragments,
        page_w=int(page_w),
        page_h=int(page_h),
    )

    return strong_tables + grouped_legends


def _detect_tables(gray: "np.ndarray", *, min_area_px: int) -> List[CropCandidate]:
    h, w = gray.shape[:2]

    # 1) binarize to make lines stand out
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        10,
    )

    # 2) morphology to isolate long horizontal/vertical strokes
    horiz_len = max(20, int(w * 0.03))
    vert_len = max(20, int(h * 0.03))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

    horiz = cv2.erode(thr, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)

    vert = cv2.erode(thr, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=2)

    grid = cv2.bitwise_or(horiz, vert)

    # 3) connect fragmented grids slightly
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)

    contours, _hier = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands: List[CropCandidate] = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = int(ww * hh)
        if area < int(min_area_px):
            continue

        # reject near-full-page masks
        if ww > int(w * 0.95) and hh > int(h * 0.95):
            continue

        # tables generally have at least moderate width & height
        if ww < int(w * 0.12) or hh < int(h * 0.06):
            continue

        # score: proportion of grid pixels inside bbox
        roi = grid[y : y + hh, x : x + ww]
        grid_density = float(cv2.countNonZero(roi)) / float(max(1, area))
        score = grid_density

        cands.append(CropCandidate(kind="table", x0=x, y0=y, x1=x + ww, y1=y + hh, score=score, method="grid_lines"))

    return cands


def _detect_legends(gray: "np.ndarray", *, min_area_px: int) -> List[CropCandidate]:
    h, w = gray.shape[:2]

    # Use Graph-Based Image Segmentation for more robust region detection
    segmenter = cv2.ximgproc.createGraphSegmenter(sigma=0.5, k=300, min_size=min_area_px // 2)
    segments = segmenter.processImage(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    cands: List[CropCandidate] = []
    for label in np.unique(segments):
        mask = np.uint8(segments == label) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue

        # Merge contours for a single segment label
        all_points = np.vstack([cnt for cnt in contours])
        x, y, ww, hh = cv2.boundingRect(all_points)
        
        area = int(ww * hh)
        if area < int(min_area_px):
            continue

        if ww > int(w * 0.98) and hh > int(h * 0.98):
            continue
        
        if ww < int(w * 0.05) or hh < int(h * 0.05):
            continue

        # Score based on density and rectangularity
        roi_mask = mask[y:y+hh, x:x+ww]
        density = cv2.countNonZero(roi_mask) / float(max(1, area))
        
        # Penalize very non-rectangular shapes
        cnt_area = float(sum(cv2.contourArea(cnt) for cnt in contours))
        rectangularity = cnt_area / float(max(1.0, area))
        
        score = 0.7 * density + 0.3 * rectangularity
        
        if score > 0.1: # Filter out very sparse or non-rectangular segments
            cands.append(CropCandidate(kind="legend", x0=x, y0=y, x1=x + ww, y1=y + hh, score=score, method="graph_segment"))

    return cands


def _pad_candidate(c: CropCandidate, *, pad_px: int, w: int, h: int) -> CropCandidate:
    x0, y0, x1, y1 = _clip_box(c.x0 - pad_px, c.y0 - pad_px, c.x1 + pad_px, c.y1 + pad_px, w, h)
    return CropCandidate(kind=c.kind, x0=x0, y0=y0, x1=x1, y1=y1, score=c.score, method=c.method)


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _fill_missing_anchor_text_from_same_page(items: List[Dict[str, object]]) -> None:
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        page_no = item.get("pageNumber")
        kind = _safe_str(item.get("kind")).strip().lower()
        if not isinstance(page_no, int) or page_no <= 0 or not kind:
            continue
        grouped.setdefault((int(page_no), kind), []).append(item)

    for grouped_items in grouped.values():
        donor_texts: List[str] = []
        for item in grouped_items:
            text = _safe_str(item.get("anchorText")).strip()
            if text:
                donor_texts.append(text)
        unique_donors = list(dict.fromkeys(donor_texts))
        if len(unique_donors) != 1:
            continue
        donor_text = unique_donors[0]
        for item in grouped_items:
            if _safe_str(item.get("anchorText")).strip():
                continue
            item["anchorText"] = donor_text
            item["anchorTextSource"] = "ocr_page_sibling"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Crop table/legend images from rendered PDF pages using OpenCV line/region detection. "
            "Input should be assets/kb/<fileId>/pages/*.png produced by export_pdf_pages_to_original_screenshots.py."
        )
    )
    ap.add_argument("--file-id", required=True, help="Relative folder path under app/src/main/assets/kb")
    ap.add_argument(
        "--root",
        default="app/src/main/assets/kb",
        help="Original screenshots root (contains <fileId>/pages and <fileId>/截图)",
    )
    ap.add_argument(
        "--min-area",
        type=int,
        default=20_000,
        help="Minimum crop bbox area in pixels (filters tiny noise)",
    )
    ap.add_argument(
        "--max-per-page",
        type=int,
        default=12,
        help="Max crops per page after NMS (keeps output manageable)",
    )
    ap.add_argument(
        "--pad",
        type=int,
        default=20,
        help="Padding (px) around detected bbox when cropping",
    )
    ap.add_argument(
        "--iou",
        type=float,
        default=0.35,
        help="IoU threshold for non-max suppression",
    )
    ap.add_argument(
        "--kinds",
        default="table,legend",
        help="Comma-separated kinds to detect: table,legend",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Write debug overlay images under <fileId>/debug/ for inspection",
    )

    # OCR anchor extraction (visual title strip above tables)
    ap.add_argument(
        "--ocr",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Enable OCR to extract anchorText for table crops by reading a title strip above the table bbox. "
            "Requires pytesseract + Pillow + system tesseract."
        ),
    )
    ap.add_argument(
        "--ocr-lang",
        default="chi_sim",
        help="pytesseract language(s), e.g. chi_sim (default) or chi_sim+eng",
    )
    ap.add_argument(
        "--anchor-offset-px",
        type=int,
        default=60,
        help="Base pixels above bbox to probe for title text (default: 60)",
    )
    ap.add_argument(
        "--anchor-offset-max-px",
        type=int,
        default=120,
        help="Max pixels above bbox to probe when near probe has no text (default: 120)",
    )
    ap.add_argument(
        "--anchor-pad-x",
        type=int,
        default=30,
        help="Extra horizontal padding for the title OCR region (default: 30)",
    )
    ap.add_argument(
        "--anchor-down-px",
        type=int,
        default=40,
        help="Downward probe height (px) if upward title OCR is too short (default: 40)",
    )
    ap.add_argument(
        "--anchor-min-chars",
        type=int,
        default=4,
        help="If OCR text length is below this, try extra probes (default: 4)",
    )
    ap.add_argument(
        "--structure",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable PP-Structure table recognition for table crops.",
    )

    args = ap.parse_args()

    if args.ocr:
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore

            _ = (pytesseract, Image)
        except Exception:
            print(
                "[WARN] OCR is enabled but pytesseract/Pillow are not available. "
                "anchorText will be empty. Install with: pip install pytesseract pillow\n"
                "       Also install Tesseract OCR and set TESSERACT_CMD if needed (Windows)."
            )

    project_root = Path(__file__).resolve().parent.parent
    root = Path(args.root)
    if not root.is_absolute():
        root = (project_root / root).resolve()

    # PATH-ONLY: Use for directory/asset names.
    safe_file_id = sanitize_folder_name(args.file_id)
    pages_dir = root / safe_file_id / "pages"
    out_dir = root / safe_file_id / "截图"
    debug_dir = root / safe_file_id / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    kinds = {k.strip().lower() for k in (args.kinds or "").split(",") if k.strip()}
    if not kinds:
        kinds = {"table", "legend"}

    pages = _iter_page_images(pages_dir)
    if not pages:
        raise SystemExit(f"No page images found under: {pages_dir}")

    manifest: Dict[str, object] = {
        "version": 2,
        "fileId": safe_file_id,
        "generatedAt": int(time.time() * 1000),
        "root": str(root.as_posix()),
        "pagesDir": str(pages_dir.as_posix()),
        "outDir": str(out_dir.as_posix()),
        "items": [],
    }

    page_heading_hints = _load_kb_heading_hints(root, safe_file_id)

    table_recognizer = None
    if args.structure:
        try:
            from pdf_table_structure import TableStructureRecognizer
            table_recognizer = TableStructureRecognizer()
        except Exception as e:
            print(f"[WARN] Failed to initialize TableStructureRecognizer: {e}")

    all_items: List[Dict[str, object]] = []
    total_written = 0

    for page_no, img_path in pages:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cands: List[CropCandidate] = []
        if "table" in kinds:
            cands.extend(_detect_tables(gray, min_area_px=int(args.min_area)))
        if "legend" in kinds:
            cands.extend(_detect_legends(gray, min_area_px=int(args.min_area)))

        # pad + NMS
        cands = [_pad_candidate(c, pad_px=int(args.pad), w=w, h=h) for c in cands]
        cands = _nms(cands, iou_threshold=float(args.iou))
        cands = _reclassify_and_group_candidates(gray, cands, page_w=w, page_h=h)
        cands = _nms(cands, iou_threshold=float(args.iou))

        # keep best N (then process in reading order for collision-aware anchoring)
        cands = sorted(cands, key=lambda c: c.score, reverse=True)[: int(args.max_per_page)]
        cands = sorted(cands, key=lambda c: (c.y0, c.x0, -c.score))

        overlay = img.copy() if args.debug else None

        prev_block_bottom: Optional[int] = None
        heading_offsets: Dict[str, int] = {}

        for idx, c in enumerate(cands, start=1):
            crop = img[c.y0 : c.y1, c.x0 : c.x1]
            if crop.size == 0:
                continue

            out_name = f"{c.kind}_p{page_no}_idx{idx}.png"
            out_path = out_dir / out_name
            cv2.imencode(".png", crop)[1].tofile(str(out_path))
            total_written += 1

            anchor_text = ""
            anchor_text_source = ""
            anchor_box: Optional[Dict[str, int]] = None
            anchor_boxes: List[Dict[str, int]] = []
            heading_text = _pick_heading_text_for_crop(page_heading_hints, page_no, c.kind, heading_offsets)
            if args.ocr:
                # Boundary-aware dynamic anchor locator:
                # - limit_top: never cross previous block bottom (avoid "误伤邻居")
                # - probe near-to-far upward: 30 -> base -> ... -> max
                # - within each probe: refine to nearest 1-2 text lines using projection
                # - if still no text, probe downward

                pad_x = int(args.anchor_pad_x)
                min_chars = max(1, int(args.anchor_min_chars))
                base_up = max(10, int(args.anchor_offset_px))
                max_up = max(base_up, int(args.anchor_offset_max_px))
                down_h = max(10, int(args.anchor_down_px))
                limit_top = int(prev_block_bottom) if isinstance(prev_block_bottom, int) else 0

                px0, _py0, px1, _py1 = _clip_box(c.x0 - pad_x, 0, c.x1 + pad_x, 1, w, h)

                probe_heights = [
                    min(30, max_up),
                    base_up,
                    min(90, max_up),
                    min(120, max_up),
                    max_up,
                    150,
                ]
                # Unique, sorted ascending, and keep within bounds
                seen_h = set()
                heights: List[int] = []
                for hh in probe_heights:
                    hh_i = int(hh)
                    if hh_i <= 0 or hh_i in seen_h:
                        continue
                    seen_h.add(hh_i)
                    heights.append(hh_i)
                heights.sort()

                chosen_box: Optional[Dict[str, int]] = None

                for hh in heights:
                    y_top = max(limit_top, int(c.y0) - int(hh))
                    y_bot = int(c.y0)
                    if y_bot - y_top < 10:
                        continue

                    box, boxes = _dynamic_anchor_probe_box(
                        img,
                        x0=int(px0),
                        x1=int(px1),
                        y_top=int(y_top),
                        y_bottom=int(y_bot),
                        debug_record_box=True,
                    )
                    anchor_boxes.extend(boxes)
                    if box is not None:
                        # Found near text lines; keep this box and stop expanding.
                        chosen_box = box
                        break

                # Downward fallback (e.g., bottom titles)
                if chosen_box is None:
                    y_top = int(c.y1)
                    y_bot = int(min(h, int(c.y1) + int(down_h)))
                    if y_bot - y_top >= 10:
                        box, boxes = _dynamic_anchor_probe_box(
                            img,
                            x0=int(px0),
                            x1=int(px1),
                            y_top=int(y_top),
                            y_bottom=int(y_bot),
                            debug_record_box=True,
                        )
                        anchor_boxes.extend(boxes)
                        if box is not None:
                            chosen_box = box

                # Union all recorded fragments (including refined lines) and OCR once.
                union_box = _union_boxes(anchor_boxes)
                if union_box is not None:
                    u0, v0, u1, v1 = _clip_box(
                        int(union_box["left"]),
                        int(union_box["top"]),
                        int(union_box["right"]),
                        int(union_box["bottom"]),
                        w,
                        h,
                    )
                    roi = img[v0:v1, u0:u1]
                    anchor_text = _ocr_anchor_text(
                        roi,
                        lang=str(args.ocr_lang),
                        debug=bool(args.debug),
                        debug_label=f"p{page_no} {c.kind} idx{idx}",
                    )
                    # Final anchorBox becomes the union
                    anchor_box = {
                        "left": int(u0),
                        "top": int(v0),
                        "right": int(u1),
                        "bottom": int(v1),
                        "width": int(u1 - u0),
                        "height": int(v1 - v0),
                    }
                    if anchor_text:
                        anchor_text_source = "ocr"
                else:
                    anchor_box = chosen_box

                if not anchor_text:
                    top_h = min(crop.shape[0], max(40, int(crop.shape[0] * 0.35)))
                    fallback_rois: List[Tuple[str, "np.ndarray", Optional[Dict[str, int]]]] = []
                    if top_h >= 16:
                        fallback_rois.append(
                            (
                                "crop_top",
                                crop[:top_h, :],
                                {
                                    "left": int(c.x0),
                                    "top": int(c.y0),
                                    "right": int(c.x1),
                                    "bottom": int(min(c.y1, c.y0 + top_h)),
                                    "width": int(c.x1 - c.x0),
                                    "height": int(top_h),
                                },
                            )
                        )
                    fallback_rois.append(
                        (
                            "crop_full",
                            crop,
                            {
                                "left": int(c.x0),
                                "top": int(c.y0),
                                "right": int(c.x1),
                                "bottom": int(c.y1),
                                "width": int(c.x1 - c.x0),
                                "height": int(c.y1 - c.y0),
                            },
                        )
                    )
                    for roi_label, roi_img, roi_box in fallback_rois:
                        fallback_text = _ocr_anchor_text(
                            roi_img,
                            lang=str(args.ocr_lang),
                            debug=bool(args.debug),
                            debug_label=f"p{page_no} {c.kind} idx{idx} {roi_label}",
                        )
                        if not fallback_text:
                            continue
                        anchor_text = fallback_text
                        anchor_text_source = "ocr"
                        if anchor_box is None and roi_box is not None:
                            anchor_box = roi_box
                        break

            if not anchor_text and heading_text:
                anchor_text = heading_text
                anchor_text_source = "kb_heading_fallback"

            if overlay is not None:
                # Draw detected crop bbox
                box_color = (0, 220, 0) if c.kind == "table" else (220, 120, 0)
                cv2.rectangle(overlay, (c.x0, c.y0), (c.x1, c.y1), box_color, 2)
                cv2.putText(
                    overlay,
                    f"{c.kind}:{c.score:.2f}",
                    (c.x0, max(20, c.y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

                # Draw OCR anchorBox(es) (title probes) in blue for any kind when OCR is enabled
                if anchor_boxes:
                    blue = (255, 0, 0)  # BGR
                    # Draw all fragment boxes thin
                    for box in anchor_boxes[:6]:
                        cv2.rectangle(
                            overlay,
                            (int(box["left"]), int(box["top"])),
                            (int(box["right"]), int(box["bottom"])),
                            blue,
                            1,
                        )

                    # Draw final union/anchorBox thicker with label
                    if anchor_box is not None:
                        cv2.rectangle(
                            overlay,
                            (int(anchor_box["left"]), int(anchor_box["top"])),
                            (int(anchor_box["right"]), int(anchor_box["bottom"])),
                            blue,
                            3,
                        )
                        label = "anchor" if not anchor_text else f"anchor:{anchor_text[:24]}"
                        cv2.putText(
                            overlay,
                            label,
                            (int(anchor_box["left"]), max(18, int(anchor_box["top"]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            blue,
                            2,
                            cv2.LINE_AA,
                        )

            item = {
                "kind": c.kind,
                "pageNumber": int(page_no),
                "sourcePageImage": str(img_path.as_posix()),
                "outFile": out_name,
                "assetUri": f"file:///android_asset/kb/{safe_file_id}/截图/{out_name}",
                "method": c.method,
                "score": float(c.score),
                "bbox": {
                    "left": int(c.x0),
                    "top": int(c.y0),
                    "right": int(c.x1),
                    "bottom": int(c.y1),
                    "width": int(c.x1 - c.x0),
                    "height": int(c.y1 - c.y0),
                },
                "bboxNormalized": {
                    "left": float(c.x0) / float(w),
                    "top": float(c.y0) / float(h),
                    "right": float(c.x1) / float(w),
                    "bottom": float(c.y1) / float(h),
                },
            }
            if anchor_text:
                item["anchorText"] = anchor_text
            if anchor_text_source:
                item["anchorTextSource"] = anchor_text_source
            if heading_text:
                item["headingText"] = heading_text
            if anchor_box is not None:
                item["anchorBox"] = anchor_box
            if anchor_boxes:
                item["anchorBoxes"] = anchor_boxes

            if c.kind == "table" and table_recognizer:
                try:
                    # Run PP-Structure on the cropped table image
                    structure_results = table_recognizer.process_image(crop)
                    if structure_results:
                        # Usually one crop contains one primary table
                        main_table = structure_results[0]
                        item["tableStructure"] = main_table.get("logic_structure", [])
                        item["tableHtml"] = main_table.get("html", "")

                        # Save structure sidecar files
                        struct_json_name = out_name.replace(".png", ".structure.json")
                        struct_html_name = out_name.replace(".png", ".structure.html")

                        with open(out_dir / struct_json_name, "w", encoding="utf-8") as f:
                            json.dump(main_table, f, ensure_ascii=False, indent=2)
                        with open(out_dir / struct_html_name, "w", encoding="utf-8") as f:
                            f.write(main_table.get("html", ""))

                        item["structureFile"] = struct_json_name
                except Exception as ex:
                    print(f"[WARN] Table structure recognition failed for {out_name}: {ex}")

            all_items.append(item)

            # Update collision boundary for the next block on this page.
            prev_block_bottom = max(int(prev_block_bottom or 0), int(c.y1))

        if overlay is not None:
            out_dbg = debug_dir / f"page_{page_no:03d}_overlay.png"
            cv2.imencode(".png", overlay)[1].tofile(str(out_dbg))

        print(f"[Page {page_no}] candidates={len(cands)} written={len(cands)}")

    _fill_missing_anchor_text_from_same_page(all_items)

    manifest["items"] = all_items
    manifest["written"] = int(total_written)

    manifest_path = out_dir / "manifest.json"
    _write_manifest(manifest_path, manifest)

    print(f"Wrote {total_written} crops to: {out_dir}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
