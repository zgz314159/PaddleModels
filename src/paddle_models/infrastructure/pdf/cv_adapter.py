from __future__ import annotations

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None
from typing import List, Optional
from dataclasses import dataclass
from paddle_models.domain.models import BBox, LayoutBlock, LayoutResult

@dataclass
class CvCropCandidate:
    kind: str
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    method: str

class CvLayoutAdapter:
    def __init__(self, min_area_px: int = 5000):
        self.min_area_px = min_area_px

    def detect(self, image: np.ndarray, page_number: int) -> LayoutResult:
        if cv2 is None:
            return LayoutResult(
                blocks=[],
                page_number=page_number,
                engine_name="opencv-heuristics",
                engine_version="1.0"
            )
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        tables = self._detect_tables(gray)
        # For now, just tables. Legends/figures can be added later.
        
        layout_blocks = []
        for t in tables:
            bbox = BBox(x=float(t.x0), y=float(t.y0), w=float(t.x1 - t.x0), h=float(t.y1 - t.y0))
            layout_blocks.append(LayoutBlock(
                type="table",
                bbox=bbox,
                confidence=t.score,
                source=f"cv_{t.method}"
            ))
            
        return LayoutResult(
            blocks=layout_blocks,
            page_number=page_number,
            engine_name="opencv-heuristics",
            engine_version="1.0"
        )

    def _detect_tables(self, gray: np.ndarray) -> List[CvCropCandidate]:
        if cv2 is None:
            return []
        h, w = gray.shape[:2]
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thr = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10
        )

        horiz_len = max(20, int(w * 0.03))
        vert_len = max(20, int(h * 0.03))
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))

        horiz = cv2.erode(thr, horiz_kernel, iterations=1)
        horiz = cv2.dilate(horiz, horiz_kernel, iterations=2)

        vert = cv2.erode(thr, vert_kernel, iterations=1)
        vert = cv2.dilate(vert, vert_kernel, iterations=2)

        grid = cv2.bitwise_or(horiz, vert)
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cands = []
        for cnt in contours:
            x, y, ww, hh = cv2.boundingRect(cnt)
            area = ww * hh
            if area < self.min_area_px:
                continue
            if ww > w * 0.95 and hh > h * 0.95:
                continue
            if ww < w * 0.12 or hh < h * 0.06:
                continue

            roi = grid[y : y + hh, x : x + ww]
            score = float(cv2.countNonZero(roi)) / float(max(1, area))
            cands.append(CvCropCandidate(kind="table", x0=x, y0=y, x1=x + ww, y1=y + hh, score=score, method="grid_lines"))

        return cands
