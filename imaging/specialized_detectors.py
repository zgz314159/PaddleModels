import io
import logging
from typing import List, Dict, Any, Optional, Set
import fitz
from PIL import Image

logger = logging.getLogger(__name__)

RED_ANNOTATION_MIN_PAGE_AREA_RATIO = 0.002
RED_ANNOTATION_STROKE_MIN_WIDTH = 0.5

def color_is_red(value: Any) -> bool:
    try:
        if value is None: return False
        comps = list(value)
        if len(comps) < 3: return False
        red, green, blue = float(comps[0]), float(comps[1]), float(comps[2])
        if red >= 0.75 and green <= 0.35 and blue <= 0.35: return True
        if red >= 0.70 and red >= max(green, blue) * 1.15 and green <= 0.55 and blue <= 0.55: return True
        return False
    except Exception: return False

def detect_red_annotation_layout_blocks(
    page: fitz.Page,
    image_bytes: bytes,
    *,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[Set] = None,
) -> List:
    # Re-implemented simplified version for specialized_detectors.py
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_w, img_h = img.size
    except Exception: return []

    # Logic to extract red boxes from PDF annotations
    boxes = []
    try:
        page_rect = page.rect
        for annot in page.annots():
            if annot.type[0] == fitz.PDF_ANNOT_SQUARE:
                stroke = (annot.colors or {}).get('stroke')
                if color_is_red(stroke):
                    rect = annot.rect
                    # Map PDF rect to render pixels
                    x0 = (rect.x0 - page_rect.x0) / page_rect.width * img_w
                    y0 = (rect.y0 - page_rect.y0) / page_rect.height * img_h
                    x1 = (rect.x1 - page_rect.x0) / page_rect.width * img_w
                    y1 = (rect.y1 - page_rect.y0) / page_rect.height * img_h
                    boxes.append((int(x0), int(y0), int(x1-x0), int(y1-y0)))
    except Exception: pass

    layout_blocks = []
    for box in boxes:
        # In real script there is a trim and type inference
        layout_blocks.append({
            'type': 'figure', # Default for red annotations
            'bbox': box,
            'source': 'red_annotation'
        })
    return layout_blocks
