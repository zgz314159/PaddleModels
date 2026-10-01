from __future__ import annotations

import logging
from typing import Optional, List, Tuple

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

logger = logging.getLogger(__name__)

# Heuristic thresholds tuned to reduce text-page false positives.
FALLBACK_MIN_H_LONG = 4
FALLBACK_MIN_V_LONG = 2
FALLBACK_MIN_INTERSECTIONS = 10

def build_visual_signal_mask(block_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Return a binary mask where 'foreground' (text/lines) is 255."""
    if cv2 is None or block_bgr is None:
        return None
    try:
        gray = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2GRAY)
        inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9,
        )
        edges = cv2.Canny(gray, 40, 140)
        mask = cv2.bitwise_or(inv, edges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        return mask
    except Exception as ex:
        logger.error(f"Error building visual signal mask: {ex}")
        return None

def count_line_intersections(
    h_lines: List[Tuple[int, int, int, int]],
    v_lines: List[Tuple[int, int, int, int]],
    tol: int = 3,
) -> int:
    """Count approximate intersections between horizontal and vertical lines."""
    if not h_lines or not v_lines:
        return 0
    count = 0
    for x1h, y1h, x2h, y2h in h_lines:
        hy = int((y1h + y2h) / 2)
        hmin, hmax = (x1h, x2h) if x1h <= x2h else (x2h, x1h)
        for x1v, y1v, x2v, y2v in v_lines:
            vx = int((x1v + x2v) / 2)
            vmin, vmax = (y1v, y2v) if y1v <= y2v else (y2v, y1v)
            if (hmin - tol) <= vx <= (hmax + tol) and (vmin - tol) <= hy <= (vmax + tol):
                count += 1
    return count

def page_has_structural_visual_signal(image_bytes: bytes) -> bool:
    """Return True when a page has strong table/diagram-like line structure."""
    if cv2 is None or np is None:
        return False
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        page = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if page is None:
            return False

        ph, pw = page.shape[:2]
        gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(th, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(80, int(pw / 40)),
            minLineLength=max(30, int(pw / 8)),
            maxLineGap=8,
        )
        if lines is None:
            return False

        h_lines: List[Tuple[int, int, int, int]] = []
        v_lines: List[Tuple[int, int, int, int]] = []
        h_min = max(40, int(pw * 0.35))
        v_min = max(40, int(ph * 0.25))

        for ln in lines:
            x1, y1, x2, y2 = ln[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dy <= max(2, int(dx * 0.2)) and dx >= h_min:
                h_lines.append((x1, y1, x2, y2))
            elif dx <= max(2, int(dy * 0.2)) and dy >= v_min:
                v_lines.append((x1, y1, x2, y2))

        intersections = count_line_intersections(h_lines, v_lines, tol=4)
        return (
            len(h_lines) >= FALLBACK_MIN_H_LONG
            and len(v_lines) >= FALLBACK_MIN_V_LONG
            and intersections >= FALLBACK_MIN_INTERSECTIONS
        )
    except Exception as ex:
        logger.error(f"Error checking structural signal: {ex}")
        return False
