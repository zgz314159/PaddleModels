import logging
import io
import os
import re
import traceback
from typing import List, Tuple, Dict, Any, Optional, Set
try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image
except Exception:
    Image = None

from imaging.bbox_utils import (
    clip_xywh_to_page,
    coerce_bbox_to_xywh,
    bbox_overlap_1d,
    bbox_iou_xywh,
    bbox_coverage_xywh,
    bbox_inside_xywh
)
from imaging.vision_utils import (
    build_visual_signal_mask,
    count_line_intersections,
    page_has_structural_visual_signal
)
from imaging.ocr_engine import ocr_image_bytes
from imaging.layout_detector import coerce_layout_items
from imaging.table_processor import detect_table_bboxes_from_image_bytes
from imaging.layout_refiner import (
    iteratively_refine_visual_bbox,
    shrink_visual_bbox_to_blank_edges,
    figure_box_looks_self_contained
)

logger = logging.getLogger(__name__)

# Heuristic thresholds
MIN_AREA_RATIO = 0.02
MAX_ASPECT_RATIO = 15.0
ALLOWED_LAYOUT_TYPES = ('table', 'figure', 'equation')
SKIP_TEXT_TYPES = ('text', 'title', 'header', 'footer')
MAX_AREA_RATIO = 0.98
CV_FALLBACK_MAX_AREA_RATIO = 0.68
CONFIDENCE_HIGH = 0.98

TABLE_MIN_H_LONG = 2
TABLE_MIN_V_LONG = 1
TABLE_MIN_INTERSECTIONS = 4
TABLE_CAPTION_MIN_LINES = 8
TABLE_CAPTION_LINE_LEN = 18.0
FIGURE_MAX_TEXT_LINES = 32
FIGURE_MAX_AVG_LINE_LEN = 16.0
FALLBACK_MIN_H_LONG = 4
FALLBACK_MIN_V_LONG = 2
FALLBACK_MIN_INTERSECTIONS = 10
TEXT_TRIM_MARGIN_PX = 6

def detect_layout_blocks(
    layout_res: Any, 
    image_bytes: bytes, 
    table_threshold: float = 0.2, 
    kb_image_dir: Optional[str] = None,
    page_text_boxes: Optional[List] = None,
    avoidance_constraints: Optional[List] = None
) -> List:
    """Detect layout blocks from pre-computed layout detector output."""
    try:
        items = coerce_layout_items(layout_res)
        
        ph, pw = 0, 0
        page_cv = None
        try:
            if cv2 is not None and np is not None:
                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                page_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if page_cv is not None:
                    ph, pw = page_cv.shape[:2]
        except Exception: pass
        
        page_area = float(pw * ph) if pw > 0 and ph > 0 else 1.0

        if not isinstance(items, list):
            fb_boxes = detect_table_bboxes_from_image_bytes(image_bytes)
            if fb_boxes:
                items = [
                    {
                        'type': 'table',
                        'bbox': [int(x), int(y), int(x + w), int(y + h)],
                        'score': 0.66,
                        'source': 'cv_fallback',
                    }
                    for x, y, w, h in fb_boxes
                    if int(w) > 0 and int(h) > 0
                ]

        if not isinstance(items, list): return []

        raw_blocks = []
        for it in items:
            try:
                typ = it.get('type') or it.get('label') or it.get('class') or 'table'
                bbox = it.get('bbox') or it.get('box') or it.get('rect')
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    w = int(x2 - x1) if x2 > x1 else int(bbox[2])
                    h = int(y2 - y1) if y2 > y1 else int(bbox[3])
                    src = str(it.get('source') or 'layout')
                    conf = it.get('score') or it.get('confidence') or it.get('prob')
                    
                    block_record = {
                        'type': str(typ).lower(),
                        'bbox': (int(x1), int(y1), int(w), int(h)),
                        'confidence': conf,
                        'source': str(src).lower()
                    }
                    raw_blocks.append(block_record)
            except Exception: continue

        page_cv = None
        try:
            if cv2 is not None and np is not None:
                arr = np.frombuffer(image_bytes, dtype=np.uint8)
                page_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception: pass
        
        page_area = None
        try:
            if Image is not None:
                pimg = Image.open(io.BytesIO(image_bytes))
                pw, ph = pimg.size
                page_area = float(pw) * float(ph)
        except Exception: pass

        filtered = []
        for b in raw_blocks:
            try:
                typ = str(b.get('type', '')).lower()
                x, y, w, h = b['bbox']
                source = b['source']
                
                if float(w) / float(max(1, h)) > MAX_ASPECT_RATIO: continue
                if page_area is not None and page_area > 1.0:
                    if (float(w) * float(h)) / page_area < MIN_AREA_RATIO: continue
                
                # Refinement
                refined_bbox = (x, y, w, h)
                if typ in ('figure', 'equation'):
                    refined_bbox = iteratively_refine_visual_bbox(
                        page_cv, (x, y, w, h), (x, y, w, h), 
                        page_text_boxes or [], avoidance_constraints, 
                        typ, pw, ph, page_area
                    )
                    refined_bbox = shrink_visual_bbox_to_blank_edges(
                        page_cv, refined_bbox, typ, page_text_boxes
                    )
                    if not figure_box_looks_self_contained(page_cv, refined_bbox):
                        continue

                b['bbox'] = refined_bbox
                filtered.append(b)
            except Exception: continue
            
        # Final overlap suppression
        if filtered:
            filtered_sorted = sorted(filtered, key=lambda it: (-(it['bbox'][2] * it['bbox'][3]), it['bbox'][1], it['bbox'][0]))
            deduped = []
            for cand in filtered_sorted:
                cb = cand['bbox']
                drop = False
                for kept in deduped:
                    kb = kept['bbox']
                    if bbox_iou_xywh((cb[0], cb[1], cb[2], cb[3]), (kb[0], kb[1], kb[2], kb[3])) >= 0.82:
                        drop = True
                        break
                if not drop:
                    deduped.append(cand)
            filtered = sorted(deduped, key=lambda it: (it['bbox'][1], it['bbox'][0]))

        return filtered
    except Exception as e:
        logger.error(f"Error in detect_layout_blocks: {e}")
        return []
