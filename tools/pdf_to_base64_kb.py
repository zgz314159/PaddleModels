#!/usr/bin/env python3
"""
Refactored PDF -> knowledge_base JSON generator.

Features added:
- Extract embedded images from PDF (fitz.get_images / extract_image)
- Table detection using OpenCV + automatic cropping

This script is defensive: OpenCV / PaddleOCR / pytesseract are optional and
will be used if available; missing dependencies will degrade behavior but
the script will still produce a KB with page images and extracted text.
"""
from __future__ import annotations

import sys
import os
import re
import json
import base64
import hashlib
import time
import argparse
import traceback
import subprocess
import shutil
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Set

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --- Semantics & Watermark extraction ---
from pipeline.semantics.role_inference import (
    infer_text_structure_semantic_role,
    is_pdf_page_marker_text,
    looks_like_pdf_attached_numbered_body_item,
    is_likely_pdf_callout_or_annotation
)
from imaging.watermark_utils import (
    looks_like_repeated_watermark_text as _looks_like_repeated_watermark_text,
    filter_repeated_watermark_lines as _filter_repeated_watermark_lines,
    filter_repeated_watermark_text as _filter_repeated_watermark_text,
    filter_repeated_watermark_dicts as _filter_repeated_watermark_dicts,
    collect_repeated_watermark_candidates as _collect_repeated_watermark_candidates,
)

# --- Bbox & Vision utilities ---
from imaging.bbox_utils import (
    clip_xywh_to_page as _clip_xywh_to_page,
    bbox_overlap_1d as _bbox_overlap_1d,
    bbox_gap_1d as _bbox_gap_1d,
    coerce_bbox_to_xywh as _coerce_bbox_to_xywh,
    xywh_to_bbox_dict as _xywh_to_bbox_dict,
    bbox_overlap_ratio_xywh as _bbox_overlap_ratio_xywh,
    bbox_center_inside_region as _bbox_center_inside_region,
    bbox_iou_xywh as _bbox_iou_xywh,
    bbox_inside_xywh as _bbox_inside_xywh,
    bbox_coverage_xywh as _bbox_coverage_xywh,
)
from imaging.vision_utils import (
    build_visual_signal_mask as _build_visual_signal_mask,
    count_line_intersections as _count_line_intersections,
    page_has_structural_visual_signal as _page_has_structural_visual_signal,
)
from imaging.layout_refiner import (
    collect_top_semantic_constraints as _collect_top_semantic_constraints,
    suppress_nested_figure_boxes as _suppress_nested_figure_boxes,
    suppress_compound_parent_figure_boxes as _suppress_compound_parent_figure_boxes,
    is_nested_figure_fragment as _is_nested_figure_fragment,
    prune_overlapping_boxes as _prune_overlapping_boxes,
    suppress_redundant_layout_figure_blocks as _suppress_redundant_layout_figure_blocks,
    merge_sidecar_figure_boxes as _merge_sidecar_figure_boxes,
    merge_stacked_table_boxes as _merge_stacked_table_boxes,
    pad_visual_bbox as _pad_visual_bbox,
    clamp_visual_bbox_growth as _clamp_visual_bbox_growth,
    trim_visual_bbox_with_text_boxes as _trim_visual_bbox_with_text_boxes,
    refine_visual_bbox_from_pixels as _refine_visual_bbox_from_pixels,
    iteratively_refine_visual_bbox as _iteratively_refine_visual_bbox,
    expand_refined_figure_bbox_with_support_boxes as _expand_refined_figure_bbox_with_support_boxes,
    clamp_padded_figure_bbox_to_support_band as _clamp_padded_figure_bbox_to_support_band,
    clamp_padded_figure_bbox_to_avoidance_constraints as _clamp_padded_figure_bbox_to_avoidance_constraints,
    trim_visual_bbox_with_image_rows as _trim_visual_bbox_with_image_rows,
    shrink_visual_bbox_to_blank_edges as _shrink_visual_bbox_to_blank_edges,
    figure_box_looks_self_contained as _figure_box_looks_self_contained,
)

# Utility: robust UTF logger that always writes to stderr to avoid stdout pollution
def _print_utf(msg: str) -> None:
    """Write a UTF-8-safe message to stderr without using built-in print.

    This deliberately avoids calling `print()` to prevent recursion when we
    override builtins.print below. All logs should go to stderr so stdout can
    be safely redirected to the JSON output file by callers.
    """
    try:
        import sys
        # prefer buffer when available to avoid encoding issues
        if hasattr(sys.stderr, 'buffer'):
            try:
                sys.stderr.buffer.write((str(msg) + "\n").encode('utf-8', errors='replace'))
                sys.stderr.buffer.flush()
                return
            except Exception:
                pass
        try:
            sys.stderr.write(str(msg) + "\n")
            try:
                sys.stderr.flush()
            except Exception:
                pass
            return
        except Exception:
            pass
    except Exception:
        pass

# Redirect built-in print to stderr via _print_utf to ensure callers that use
# print() don't accidentally write to stdout and corrupt JSON when `>` is used
# in .bat wrappers. We keep a fallback to the original print if everything
# else fails.
try:
    import builtins
    _original_print = builtins.print
    def _stderr_print(*args, sep=' ', end='\n', file=None, flush=False, **kwargs):
        try:
            # build string like print would
            s = sep.join(str(a) for a in args)
            # respect end explicitly
            if end:
                s = s + end.rstrip('\n')
            _print_utf(s)
        except Exception:
            try:
                # ensure fallback also writes to stderr to avoid stdout pollution
                import sys as _sys
                _original_print(*args, sep=sep, end=end, file=_sys.stderr, flush=flush, **kwargs)
            except Exception:
                pass
    builtins.print = _stderr_print
except Exception:
    # If we cannot override builtins, at least ensure _print_utf exists
    pass

# Environment health probe (early): try importing optional native libs
# We import into the canonical names used later (`np`, `cv2`) but do so
# defensively so the script can run even when some backends are missing.
try:
    import numpy as np
    _HAS_NUMPY = True
    try:
        _print_utf(f"[PROBE-ENV] NumPy Version: {np.__version__}")
        try:
            _print_utf(f"[PROBE-ENV] NumPy Path: {np.__file__}")
        except Exception:
            pass
    except Exception:
        pass
except Exception as _probe_e:
    _HAS_NUMPY = False
    _print_utf(f"[PROBE-ENV-CRITICAL] NumPy Import Failed: {_probe_e}")

try:
    import cv2
    _HAS_CV2 = True
    try:
        _print_utf(f"[PROBE-ENV] OpenCV Version: {cv2.__version__}")
    except Exception:
        pass
except Exception as _probe_e:
    _HAS_CV2 = False
    _print_utf(f"[PROBE-ENV] OpenCV import failed: {_probe_e}")

# Preload torch before Paddle on Windows. In this environment, importing
# Paddle first can make a later torch/ultralytics load fail with a DLL error.
try:
    import torch
    _HAS_TORCH = True
    _TORCH_IMPORT_ERR = ''
except Exception as _torch_import_e:
    torch = None
    _HAS_TORCH = False
    _TORCH_IMPORT_ERR = str(_torch_import_e)

# Optional OCR backends
try:
    from paddleocr import PaddleOCR
    _HAS_PADDLE = True
    _PADDLE_IMPORT_ERR = ''
except Exception as _paddle_import_e:
    _HAS_PADDLE = False
    _PADDLE_IMPORT_ERR = str(_paddle_import_e)

try:
    import pytesseract
    _HAS_TESSERACT = True
except Exception:
    _HAS_TESSERACT = False

_UltralyticsYOLO = None
_HAS_ULTRALYTICS = False
_ULTRALYTICS_IMPORT_ERR = ''

import argparse
import base64
import copy
import hashlib
import importlib
import io
import json
import os
import re
import sys
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise ImportError("PyMuPDF (fitz) is required: pip install PyMuPDF") from e

# helper utilities expected in project; provide safe fallbacks if missing
try:
    from utils.file_id_sanitizer import sanitize_file_id_for_path_only as sanitize_file_id, sanitize_asset_relative_path
except Exception:
    def sanitize_file_id(name: Optional[str], default: str = "doc") -> str:
        s = (name or "").strip()
        return s or default

def ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass

# Industrial detection thresholds
# Minimum fraction of the page area a detected block must occupy to be
# considered worth cropping/saving.
MIN_AREA_RATIO = 0.02
# Maximum allowed width/height ratio (very long horizontal stripes are noise)
MAX_ASPECT_RATIO = 15.0
# Allowed layout types for saving crops
ALLOWED_LAYOUT_TYPES = ('table', 'figure', 'equation')
# Explicitly skip these non-visual layout labels (they are handled via OCR text)
SKIP_TEXT_TYPES = ('text', 'title', 'header', 'footer')
FIGURE_SUPPORT_LAYOUT_TYPES = ('figure_caption', 'figure_reference', 'caption', 'reference')
# If a detected block occupies more than this fraction of the page, it's
# probably a full-page background/mis-detection unless PP-Structure reports
# a very high confidence for that block.
MAX_AREA_RATIO = 0.98
# CV fallback blocks are typically noisier than PP-Structure blocks, so keep
# a stricter upper bound to avoid full-page false positives.
CV_FALLBACK_MAX_AREA_RATIO = 0.68
# Confidence threshold to override MAX_AREA_RATIO (0.0-1.0)
CONFIDENCE_HIGH = 0.98

# Heuristic thresholds tuned to reduce text-page false positives.
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
VISUAL_BBOX_PAD_PX = 8
YOLO_LAYOUT_DEFAULT_CONF = 0.18
YOLO_LAYOUT_DEFAULT_IMGSZ = 1600
_PADDLE_OCR_INSTANCE = None
_PADDLE_OCR_INSTANCE_INIT_FAILED = False
_PADDLE_OCR_LOCK = threading.RLock()


def _get_shared_paddle_ocr() -> Optional[object]:
    global _PADDLE_OCR_INSTANCE, _PADDLE_OCR_INSTANCE_INIT_FAILED
    if not _HAS_PADDLE:
        return None
    if _PADDLE_OCR_INSTANCE is not None:
        return _PADDLE_OCR_INSTANCE
    if _PADDLE_OCR_INSTANCE_INIT_FAILED:
        return None
    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_INSTANCE is not None:
            return _PADDLE_OCR_INSTANCE
        if _PADDLE_OCR_INSTANCE_INIT_FAILED:
            return None
        try:
            _PADDLE_OCR_INSTANCE = PaddleOCR(use_angle_cls=True, lang='ch', use_mkldnn=False, enable_mkldnn=False)
        except Exception:
            _PADDLE_OCR_INSTANCE_INIT_FAILED = True
            _PADDLE_OCR_INSTANCE = None
            return None
        return _PADDLE_OCR_INSTANCE



def _collect_dynamic_page_artifact_signatures(doc: fitz.Document) -> Set[str]:
    """Detect page-level artifacts whose text changes per page (running page numbers/chapter headers).

    Unlike `_collect_repeated_watermark_candidates` (exact repeated text), this looks at text
    pinned to the extreme top/bottom of each page and normalizes digit runs to '#', so that
    incrementing page numbers like "- 1 -" / "- 2 -" or a running chapter header collapse to the
    same signature and can be recognized as recurring across many pages even though the literal
    text differs page to page.
    """
    page_hits: Dict[str, Set[int]] = {}
    try:
        page_count = int(getattr(doc, 'page_count', 0) or 0)
    except Exception:
        page_count = 0
    if page_count <= 2:
        return set()

    threshold = max(3, min(40, (page_count + 4) // 5))

    for page_index in range(page_count):
        try:
            page = doc.load_page(page_index)
            page_height = float(page.rect.height)
            data = page.get_text('dict') or {}
        except Exception:
            continue
        if page_height <= 0:
            continue
        top_zone_limit = page_height * 0.06
        bottom_zone_limit = page_height * 0.94
        blocks = data.get('blocks') if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            continue
        for blk in blocks:
            if not isinstance(blk, dict) or int(blk.get('type', 0)) != 0:
                continue
            bbox = blk.get('bbox')
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                y0, y1 = float(bbox[1]), float(bbox[3])
            except Exception:
                continue
            if y1 <= top_zone_limit:
                zone = 'top'
            elif y0 >= bottom_zone_limit:
                zone = 'bottom'
            else:
                continue
            lines = blk.get('lines') if isinstance(blk.get('lines'), list) else []
            span_texts: List[str] = []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                for span in (line.get('spans') or []):
                    if not isinstance(span, dict):
                        continue
                    txt = str(span.get('text') or '').strip()
                    if txt:
                        span_texts.append(txt)
            text = ' '.join(span_texts).strip()
            compact = re.sub(r'\s+', '', text)
            if not compact or len(compact) > 40:
                continue
            normalized = re.sub(r'\d+', '#', compact)
            has_digit = normalized != compact
            # Require either a digit run (page number) or short header-like text; long prose
            # accidentally touching the edge zone should not be swallowed as an artifact.
            if not has_digit and len(compact) > 24:
                continue
            signature = f'{zone}|{normalized}'
            page_hits.setdefault(signature, set()).add(page_index + 1)

    dynamic = {
        signature
        for signature, pages in page_hits.items()
        if len(pages) >= threshold
    }
    if dynamic:
        try:
            sample = ', '.join(sorted(dynamic)[:6])
            _print_utf(
                f"[ARTIFACT] dynamic_page_artifact_signatures={len(dynamic)} threshold={threshold} samples={sample}"
            )
        except Exception:
            pass
    return dynamic


def _looks_like_dynamic_page_artifact(
    text: str,
    *,
    zone: Optional[str],
    dynamic_artifact_signatures: Optional[set],
) -> bool:
    if not dynamic_artifact_signatures or not zone:
        return False
    compact = re.sub(r'\s+', '', str(text or ''))
    compact = compact.replace('**', '')
    if not compact:
        return False
    normalized = re.sub(r'\d+', '#', compact)
    return f'{zone}|{normalized}' in dynamic_artifact_signatures


def _extract_native_table_cells(table: Any) -> List[Dict]:
    """Convert PyMuPDF native table object to a list of TableCell-compatible dicts with Span detection."""
    try:
        # Mapping from (r, c) to text
        text_matrix = table.extract()
        if not text_matrix:
            return []

        cells = []
        # PyMuPDF table.extract() returns 'None' for cells that are part of a span.
        # We use a greedy expansion to determine rowSpan and colSpan.

        processed_matrix = [[False for _ in range(len(text_matrix[0]))] for _ in range(len(text_matrix))]

        for r_idx, row_data in enumerate(text_matrix):
            for c_idx, cell_text in enumerate(row_data):
                if processed_matrix[r_idx][c_idx]:
                    continue

                # Treat None as part of a previous span, but if it's the start, it's an empty cell
                # Actually, in PyMuPDF text_matrix, the root cell has the text, others are None.

                # Calculate colSpan
                cs = 1
                while c_idx + cs < len(row_data) and text_matrix[r_idx][c_idx + cs] is None:
                    cs += 1

                # Calculate rowSpan
                rs = 1
                while r_idx + rs < len(text_matrix):
                    # Check if the entire width of the span is None in the next row
                    all_none = True
                    for i in range(cs):
                        if text_matrix[r_idx + rs][c_idx + i] is not None:
                            all_none = False
                            break
                    if not all_none:
                        break
                    rs += 1

                # Mark all cells in this span as processed
                for i in range(rs):
                    for j in range(cs):
                        processed_matrix[r_idx + i][c_idx + j] = True

                # Infer alignment
                cell_alignment = 'left'
                if r_idx == 0: cell_alignment = 'center'
                elif cell_text and str(cell_text).strip().replace('.','',1).isdigit(): cell_alignment = 'center'

                cells.append({
                    'row': r_idx,
                    'col': c_idx,
                    'rowSpan': rs,
                    'colSpan': cs,
                    'text': str(cell_text or "").strip(),
                    'isHeader': r_idx == 0, # Heuristic
                    'alignment': cell_alignment,
                    'bbox': [float(v) for v in table.cells[r_idx * len(row_data) + c_idx][:4]] if hasattr(table, 'cells') and (r_idx * len(row_data) + c_idx) < len(table.cells) else None
                })
        return cells
    except Exception:
        return []


def _normalize_layout_block_type(label: Any) -> str:
    raw = str(label or '').strip().lower()
    if not raw:
        return ''

    compact = raw.replace('_', '').replace('-', '').replace(' ', '')
    table_aliases = {
        'table', 'tab', 'datatable', 'tablecell', 'tableblock', 'gridtable', 'tablearea'
    }
    figure_aliases = {
        'figure', 'fig', 'image', 'img', 'picture', 'photo', 'chart', 'diagram',
        'illustration', 'graphic', 'graphics', 'plot', 'drawing', 'legend'
    }
    equation_aliases = {
        'equation', 'formula', 'math', 'formulablock'
    }

    if compact in table_aliases or compact.startswith('table'):
        return 'table'
    if compact in equation_aliases or compact.startswith('equation') or compact.startswith('formula'):
        return 'equation'
    if compact in figure_aliases or compact.startswith('figure') or compact.startswith('image'):
        return 'figure'
    if compact in {'text', 'title', 'header', 'footer', 'caption', 'list', 'paragraph', 'reference'}:
        return ''
    return raw


def _normalize_layout_constraint_type(label: Any) -> str:
    raw = str(label or '').strip().lower()
    if not raw:
        return ''

    compact = raw.replace('_', '').replace('-', '').replace(' ', '')
    if compact in {'text', 'paragraph', 'bodytext', 'body'} or compact.startswith('text'):
        return 'text'
    if compact in {'title', 'heading', 'head'} or compact.startswith('title'):
        return 'title'
    if compact in {'header'} or compact.startswith('header'):
        return 'header'
    if compact in {'footer'} or compact.startswith('footer'):
        return 'footer'
    return ''


def _coerce_layout_items(layout_res: Any) -> Optional[list]:
    items = None
    try:
        if isinstance(layout_res, list):
            if len(layout_res) == 1 and isinstance(layout_res[0], dict):
                for k in ('structure', 'layout', 'blocks', 'data', 'result'):
                    if k in layout_res[0] and isinstance(layout_res[0][k], list):
                        items = layout_res[0][k]
                        break
            if items is None:
                items = layout_res
        elif isinstance(layout_res, dict):
            for k in ('structure', 'layout', 'blocks', 'data', 'result'):
                if k in layout_res and isinstance(layout_res[k], list):
                    items = layout_res[k]
                    break
            if items is None:
                for v in layout_res.values():
                    if isinstance(v, list):
                        items = v
                        break
    except Exception:
        items = None
    return items if isinstance(items, list) else None


def _instantiate_yolo_layout_detector(model_path: str):
    global _UltralyticsYOLO, _HAS_ULTRALYTICS, _ULTRALYTICS_IMPORT_ERR
    if not model_path:
        return None
    if _UltralyticsYOLO is None:
        try:
            ultralytics_module = importlib.import_module('ultralytics')
            _UltralyticsYOLO = getattr(ultralytics_module, 'YOLO', None)
            _HAS_ULTRALYTICS = _UltralyticsYOLO is not None
            _ULTRALYTICS_IMPORT_ERR = '' if _HAS_ULTRALYTICS else 'YOLO symbol not found'
        except Exception as ex:
            _UltralyticsYOLO = None
            _HAS_ULTRALYTICS = False
            _ULTRALYTICS_IMPORT_ERR = str(ex)
            return None
    if not _HAS_ULTRALYTICS or _UltralyticsYOLO is None:
        return None
    try:
        return _UltralyticsYOLO(model_path)
    except Exception:
        return None


def _detect_yolo_layout_blocks(
    detector: Any,
    image_bytes: bytes,
    *,
    conf: float,
    imgsz: int,
    device: str,
) -> List[Dict[str, Any]]:
    if detector is None or not image_bytes:
        return []
    if not (_HAS_NUMPY and _HAS_CV2):
        return []

    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return []

    try:
        predict_kwargs: Dict[str, Any] = {
            'source': img_rgb,
            'verbose': False,
            'conf': float(conf),
            'imgsz': int(max(320, imgsz)),
        }
        if str(device or '').strip():
            predict_kwargs['device'] = str(device).strip()
        results = detector.predict(**predict_kwargs)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    if not isinstance(results, list):
        return out

    for result in results:
        try:
            names = getattr(result, 'names', None) or getattr(detector, 'names', {}) or {}
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue

            xyxy_list = boxes.xyxy.tolist() if hasattr(boxes, 'xyxy') else []
            cls_list = boxes.cls.tolist() if hasattr(boxes, 'cls') else []
            conf_list = boxes.conf.tolist() if hasattr(boxes, 'conf') else []

            for index, xyxy in enumerate(xyxy_list):
                if not isinstance(xyxy, (list, tuple)) or len(xyxy) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[:4]]
                except Exception:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_id = None
                if index < len(cls_list):
                    try:
                        cls_id = int(round(float(cls_list[index])))
                    except Exception:
                        cls_id = None
                raw_label = names.get(cls_id, str(cls_id) if cls_id is not None else '') if isinstance(names, dict) else ''
                typ = _normalize_layout_block_type(raw_label)
                constraint_typ = _normalize_layout_constraint_type(raw_label)
                if typ not in ALLOWED_LAYOUT_TYPES and constraint_typ not in SKIP_TEXT_TYPES:
                    continue

                score = None
                if index < len(conf_list):
                    try:
                        score = float(conf_list[index])
                    except Exception:
                        score = None

                out.append({
                    'type': typ,
                    'bbox': [x1, y1, x2, y2],
                    'score': score,
                    'source': 'yolo_layout',
                    'label': str(raw_label or '').strip(),
                })
        except Exception:
            continue
    return out


def _layout_block_to_debug_record(block: Any) -> Dict[str, Any]:
    """Convert a layout block into a JSON-safe debug record."""
    record: Dict[str, Any] = {}
    if not isinstance(block, dict):
        return {'raw': str(block)}
    for key in ('type', 'source', 'label'):
        value = block.get(key)
        if value not in (None, ''):
            record[key] = str(value)
    if 'score' in block:
        try:
            record['score'] = float(block.get('score')) if block.get('score') is not None else None
        except Exception:
            record['score'] = None
    bbox = block.get('bbox')
    if isinstance(bbox, dict):
        try:
            left = int(round(float(bbox.get('left', 0))))
            top = int(round(float(bbox.get('top', 0))))
            right = int(round(float(bbox.get('right', left))))
            bottom = int(round(float(bbox.get('bottom', top))))
            record['bbox_xyxy'] = [left, top, right, bottom]
            record['bbox_xywh'] = [left, top, max(0, right - left), max(0, bottom - top)]
        except Exception:
            record['bbox'] = bbox
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            vals = [int(round(float(v))) for v in bbox[:4]]
            record['bbox_xywh'] = vals[:4]
            record['bbox_xyxy'] = [vals[0], vals[1], vals[0] + vals[2], vals[1] + vals[3]]
        except Exception:
            record['bbox'] = list(bbox[:4])
    return record


def _write_layout_debug_page(
    debug_dir: str,
    *,
    page_num: int,
    layout_engine: str,
    yolo_model_path: str,
    yolo_raw_items: List[Dict[str, Any]],
    pp_items: List[Any],
    accepted_layout_blocks: List[Dict[str, Any]],
) -> None:
    if not debug_dir:
        return
    try:
        ensure_dir(debug_dir)
        payload = {
            'pageNumber': int(page_num),
            'layoutEngine': str(layout_engine or ''),
            'yoloModelPath': str(yolo_model_path or ''),
            'counts': {
                'yoloRaw': len(yolo_raw_items or []),
                'ppstructureNormalized': len(pp_items or []),
                'accepted': len(accepted_layout_blocks or []),
            },
            'yoloRawBlocks': [_layout_block_to_debug_record(it) for it in (yolo_raw_items or [])],
            'ppstructureBlocks': [_layout_block_to_debug_record(it) for it in (pp_items or [])],
            'acceptedBlocks': [_layout_block_to_debug_record(it) for it in (accepted_layout_blocks or [])],
        }
        out_path = os.path.join(debug_dir, f'page_{int(page_num):04d}_layout_debug.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        _print_utf(f"[WARN] Failed to write layout debug page dump for p{page_num}: {ex}")



def _collect_layout_support_boxes(layout_items: Any) -> List[Dict[str, Any]]:
    items = _coerce_layout_items(layout_items)
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int, int, int, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_type = str(item.get('type') or item.get('label') or item.get('class') or '').strip().lower()
        support_type = raw_type.replace('-', '_').replace(' ', '_')
        if support_type not in FIGURE_SUPPORT_LAYOUT_TYPES:
            continue
        bbox = _coerce_bbox_to_xywh(item.get('bbox') or item.get('box') or item.get('rect'))
        if bbox is None:
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        key = (support_type, int(x), int(y), int(w), int(h))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'type': support_type,
            'bbox': (int(x), int(y), int(w), int(h)),
            'source': str(item.get('source') or 'layout').lower(),
        })
    return out


def _collect_layout_avoidance_constraints(layout_items: Any) -> List[Dict[str, Any]]:
    items = _coerce_layout_items(layout_items)
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, int, int, int, int]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        source = str(item.get('source') or 'layout').lower()
        if source != 'yolo_layout':
            continue
        raw_type = item.get('type') or item.get('label') or item.get('class') or ''
        constraint_type = _normalize_layout_constraint_type(raw_type)
        if constraint_type not in {'text', 'title'}:
            continue
        bbox = _coerce_bbox_to_xywh(item.get('bbox') or item.get('box') or item.get('rect'))
        if bbox is None:
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        key = (constraint_type, source, int(x), int(y), int(w), int(h))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'type': constraint_type,
            'bbox': (int(x), int(y), int(w), int(h)),
            'source': source,
        })
    return out


def _detect_top_text_band_cut(
    page_cv,
    bbox: Tuple[int, int, int, int],
) -> Optional[int]:
    if page_cv is None or not (_HAS_NUMPY and _HAS_CV2):
        return None
    try:
        ph, pw = page_cv.shape[:2]
        x, y, w, h = _clip_xywh_to_page(*bbox, pw, ph)
        if w < 120 or h < 120:
            return None

        band_h = max(34, min(120, int(float(h) * 0.22)))
        roi = page_cv[y:y + band_h, x:x + w]
        if roi is None or getattr(roi, 'size', 0) == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            9,
        )
        kernel_w = max(18, min(72, int(float(w) * 0.10)))
        text_lines = cv2.morphologyEx(
            inv,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3)),
            iterations=1,
        )
        text_lines = cv2.morphologyEx(
            text_lines,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)),
            iterations=1,
        )

        contours, _hier = cv2.findContours(text_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        text_bottom = -1
        for cnt in contours:
            lx, ly, lw, lh = cv2.boundingRect(cnt)
            if lw < max(28, int(float(w) * 0.18)):
                continue
            if lh > max(26, int(float(band_h) * 0.55)):
                continue
            if ly > max(24, int(float(band_h) * 0.42)):
                continue
            patch = text_lines[ly:ly + lh, lx:lx + lw]
            if patch is None or patch.size == 0:
                continue
            fill = float(cv2.countNonZero(patch)) / float(max(1, lw * lh))
            if fill < 0.045:
                continue
            text_bottom = max(text_bottom, ly + lh)

        if text_bottom <= 0:
            return None
        candidate_top = y + int(text_bottom) + TEXT_TRIM_MARGIN_PX
        if candidate_top >= y + max(80, int(float(h) * 0.36)):
            return None
        return candidate_top
    except Exception:
        return None


def _should_trace_avoidance_debug(debug_context: str = '') -> bool:
    try:
        trace_flag = str(os.environ.get('PDF_AVOIDANCE_TRACE') or '').strip().lower()
        if trace_flag not in {'1', 'true', 'yes', 'on'}:
            return False
        trace_filter = str(os.environ.get('PDF_AVOIDANCE_TRACE_FILTER') or '').strip()
        if trace_filter and trace_filter not in (debug_context or ''):
            return False
        return True
    except Exception:
        return False


def _trace_avoidance_debug(debug_context: str, message: str) -> None:
    if not _should_trace_avoidance_debug(debug_context):
        return
    try:
        prefix = f"[TRACE-AVOID] {debug_context}".strip()
        _print_utf(f"{prefix} {message}".strip())
    except Exception:
        pass


def _reconcile_figure_split_boxes(
    parent_bbox: Tuple[int, int, int, int],
    split_boxes: List[Tuple[int, int, int, int]],
    page_text_boxes: list,
    layout_support_boxes: Optional[list] = None,
    page_cv=None,
) -> List[Tuple[int, int, int, int]]:
    if len(split_boxes) <= 1:
        return split_boxes
    try:
        split_boxes = _suppress_nested_figure_boxes(split_boxes)
        if len(split_boxes) <= 1:
            return split_boxes
        if page_cv is not None:
            for child in split_boxes:
                if not _figure_box_looks_self_contained(page_cv, child):
                    return [parent_bbox]
        return split_boxes
    except Exception:
        return [parent_bbox]


def _is_duplicate_crop_candidate(
    candidate: Tuple[int, int, int, int],
    kept_boxes: List[Tuple[int, int, int, int]],
) -> bool:
    if not kept_boxes:
        return False
    try:
        cand_area = float(max(1, candidate[2] * candidate[3]))
    except Exception:
        return False
    for kept in kept_boxes:
        try:
            kept_area = float(max(1, kept[2] * kept[3]))
            iou = _bbox_iou_xywh(candidate, kept)
            if iou >= 0.84:
                return True
            area_ratio = min(cand_area, kept_area) / max(cand_area, kept_area)
            if area_ratio >= 0.78 and (
                _bbox_inside_xywh(candidate, kept, margin=10)
                or _bbox_inside_xywh(kept, candidate, margin=10)
            ):
                return True
        except Exception:
            continue
    return False



def _extract_pdf_text_boxes(
    page: fitz.Page,
    repeated_watermark_candidates: Optional[set] = None,
    dynamic_artifact_signatures: Optional[set] = None,
) -> List[Dict]:
    """Extract text block boxes from the native PDF layout for crop trimming."""
    out: List[Dict] = []
    try:
        data = page.get_text('dict') or {}
        blocks = data.get('blocks') if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            return out
        try:
            page_height = float(page.rect.height)
        except Exception:
            page_height = 0.0
        top_zone_limit = page_height * 0.06 if page_height > 0 else 0.0
        bottom_zone_limit = page_height * 0.94 if page_height > 0 else 0.0
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            if int(blk.get('type', 0)) != 0:
                continue
            bbox = blk.get('bbox')
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            x0, y0, x1, y1 = bbox[:4]
            try:
                x0 = int(round(float(x0)))
                y0 = int(round(float(y0)))
                x1 = int(round(float(x1)))
                y1 = int(round(float(y1)))
            except Exception:
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            lines = blk.get('lines') if isinstance(blk.get('lines'), list) else []
            spans_text: List[str] = []
            max_font_size = 0.0
            is_any_span_bold = False
            total_chars_in_blk = 0
            bold_chars_in_blk = 0

            for line in lines:
                if not isinstance(line, dict):
                    continue
                spans = line.get('spans') if isinstance(line.get('spans'), list) else []
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    txt = str(span.get('text') or '').strip()
                    if txt:
                        font_size = float(span.get('size', 0))
                        if font_size > max_font_size:
                            max_font_size = font_size

                        flags = int(span.get('flags', 0))
                        is_bold = bool(flags & 16)

                        char_count = len(re.sub(r'\s+', '', txt))
                        total_chars_in_blk += char_count
                        if is_bold:
                            bold_chars_in_blk += char_count
                            is_any_span_bold = True

            # Decide whether to use inline ** or block-level isBold
            use_inline_bold = False
            if is_any_span_bold:
                if total_chars_in_blk > 0 and (bold_chars_in_blk / total_chars_in_blk) < 0.85:
                    use_inline_bold = True

            for line in lines:
                if not isinstance(line, dict):
                    continue
                spans = line.get('spans') if isinstance(line.get('spans'), list) else []
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    txt = str(span.get('text') or '').strip()
                    if txt:
                        flags = int(span.get('flags', 0))
                        is_bold = bool(flags & 16)

                        if use_inline_bold and is_bold:
                            # Avoid double bolding if already has **
                            if not (txt.startswith('**') and txt.endswith('**')):
                                txt = f"**{txt}**"

                        spans_text.append(txt)

            text = ' '.join(spans_text).strip()
            if _looks_like_repeated_watermark_text(text, repeated_watermark_candidates):
                continue
            zone = None
            if page_height > 0:
                if y1 <= top_zone_limit:
                    zone = 'top'
                elif y0 >= bottom_zone_limit:
                    zone = 'bottom'
            if _looks_like_dynamic_page_artifact(
                text,
                zone=zone,
                dynamic_artifact_signatures=dynamic_artifact_signatures,
            ):
                continue
            line_count = max(1, len(lines))
            compact = re.sub(r'\s+', '', text)
            avg_line_len = (float(len(compact)) / float(line_count)) if line_count > 0 else 0.0
            out.append({
                'bbox': (x0, y0, x1 - x0, y1 - y0),
                'text': text,
                'line_count': line_count,
                'avg_line_len': avg_line_len,
                'font_size': max_font_size,
                'is_bold': is_any_span_bold,
            })
    except Exception:
        return out
    return out


def _extract_pdf_text_line_boxes(page: fitz.Page, repeated_watermark_candidates: Optional[set] = None) -> List[Dict]:
    out: List[Dict] = []
    try:
        data = page.get_text('dict') or {}
        blocks = data.get('blocks') if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            return out
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            if int(blk.get('type', 0)) != 0:
                continue
            lines = blk.get('lines') if isinstance(blk.get('lines'), list) else []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                bbox = line.get('bbox')
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                try:
                    x0 = int(round(float(bbox[0])))
                    y0 = int(round(float(bbox[1])))
                    x1 = int(round(float(bbox[2])))
                    y1 = int(round(float(bbox[3])))
                except Exception:
                    continue
                if x1 <= x0 or y1 <= y0:
                    continue
                spans = line.get('spans') if isinstance(line.get('spans'), list) else []
                span_texts: List[str] = []
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    txt = str(span.get('text') or '').strip()
                    if txt:
                        span_texts.append(txt)
                text = ' '.join(span_texts).strip()
                if _looks_like_repeated_watermark_text(text, repeated_watermark_candidates):
                    continue
                compact = re.sub(r'\s+', '', text)
                out.append({
                    'bbox': (x0, y0, x1 - x0, y1 - y0),
                    'text': text,
                    'line_count': 1,
                    'avg_line_len': float(len(compact)) if compact else 0.0,
                })
    except Exception:
        return out
    return out


def _clamp_figure_bbox_to_top_semantic_ceiling(
    bbox: Tuple[int, int, int, int],
    top_semantic_constraints: Optional[list],
    page_w: int,
    page_h: int,
    page_cv=None,
    *,
    debug_context: str = '',
    phase: str = '',
) -> Tuple[int, int, int, int]:
    if not top_semantic_constraints:
        return bbox
    try:
        cx, cy, cw, ch = bbox
        if cw <= 0 or ch <= 0:
            return bbox

        top_band_bottom = cy + max(42, min(140, int(float(ch) * 0.22)))
        min_overlap_px = max(24, int(float(cw) * 0.18))
        max_constraint_h = max(56, min(110, int(float(ch) * 0.20)))
        target_top: Optional[int] = None
        target_source = ''
        target_bottom = -1

        for constraint in top_semantic_constraints or []:
            try:
                tx, ty, tw, th = constraint.get('bbox', (0, 0, 0, 0))
            except Exception:
                continue
            if tw <= 0 or th <= 0:
                continue
            if th > max_constraint_h:
                continue

            constraint_bottom = int(ty) + int(th)
            if constraint_bottom <= cy:
                continue
            if int(ty) >= top_band_bottom:
                continue

            x_overlap = _bbox_overlap_1d(cx, cx + cw, tx, tx + tw)
            crop_overlap_ratio = float(x_overlap) / float(max(1, cw))
            text_overlap_ratio = float(x_overlap) / float(max(1, tw))
            if x_overlap < min_overlap_px:
                continue
            if crop_overlap_ratio < 0.22:
                continue
            if text_overlap_ratio < 0.12 and crop_overlap_ratio < 0.45:
                continue

            candidate_top = min(int(cy + ch - 1), constraint_bottom + TEXT_TRIM_MARGIN_PX)
            if candidate_top <= cy:
                continue
            if target_top is None or candidate_top > target_top:
                target_top = candidate_top
                target_source = str(constraint.get('source') or '')
                target_bottom = constraint_bottom

        if target_top is None:
            pixel_top = _detect_top_text_band_cut(page_cv, bbox)
            if pixel_top is not None and pixel_top > cy:
                target_top = int(pixel_top)
                target_source = 'pixel_text_band'
                target_bottom = int(pixel_top) - TEXT_TRIM_MARGIN_PX

        if target_top is None:
            return bbox

        new_h = int(ch) - int(target_top - cy)
        if new_h < max(80, int(float(ch) * 0.42)):
            return bbox

        clamped = _clip_xywh_to_page(int(cx), int(target_top), int(cw), int(new_h), page_w, page_h)
        _trace_avoidance_debug(
            debug_context,
            f"top_ceiling_hit phase={phase or 'unknown'} src={target_source or 'semantic'} bottom={target_bottom} old={cy} new={clamped[1]}",
        )
        return clamped
    except Exception:
        return bbox



def _make_text_structure_block(tb: Dict, *, page_number: int, order_index: int, page_render_w: int = 0) -> Optional[Dict]:
    try:
        x, y, w, h = tb.get('bbox', (0, 0, 0, 0))
        text = str(tb.get('text') or '').strip()
        if not text or int(w) <= 0 or int(h) <= 0:
            return None

        semantic_role = infer_text_structure_semantic_role(tb)

        # Map semantic role to a dedicated block type instead of generic 'code'
        block_type = 'paragraph'
        if semantic_role == 'heading':
            block_type = 'heading2'
        elif semantic_role == 'figure_callout':
            block_type = 'code'  # keep as code for distinct styling

        # Heuristic alignment detection
        alignment = None
        if page_render_w > 0 and int(w) > 20:
            block_center_x = int(x) + int(w) / 2
            page_center_x = page_render_w / 2
            # If block is relatively narrow and centered
            if int(w) < page_render_w * 0.72:
                if abs(block_center_x - page_center_x) < (page_render_w * 0.05):
                    alignment = 'center'

        block_id = hashlib.sha1(f"pdftext|{page_number}|{order_index}|{text[:200]}".encode('utf-8')).hexdigest()[:12]
        return {
            'id': f'b_text_{block_id}',
            'type': block_type,
            'language': 'markdown' if block_type == 'code' else None,
            'code': text if block_type == 'code' else None,
            'text': text if block_type != 'code' else None,
            'pageNumber': int(page_number),
            'semanticRole': semantic_role,
            'bbox': _xywh_to_bbox_dict(int(x), int(y), int(w), int(h)),
            'readingOrder': int(order_index),
            'structureSource': 'pdf_native_text_box',
            'confidence': 1.0,
            'fontSize': tb.get('font_size'),
            'isBold': tb.get('is_bold'),
            'alignment': alignment,
        }
    except Exception:
        return None


def _count_decimal_clause_text_blocks(blocks: List[Dict]) -> int:
    count = 0
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        text = str(block.get('text') or block.get('code') or block.get('contentMarkdown') or '').strip()
        if re.match(r'^\s*\d{1,2}\.\d{1,2}\.\d{1,3}(?!\d)', text):
            count += 1
    return count


def _extract_decimal_clause_labels_from_blocks(blocks: List[Dict]) -> set:
    labels = set()
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        text = str(block.get('text') or block.get('code') or block.get('contentMarkdown') or '').strip()
        for match in re.finditer(r'(?<!\d)(\d{1,2}\.\d{1,2}\.\d{1,3})(?!\d)', text):
            labels.add(match.group(1))
    return labels


def _assign_page_structure_order(blocks: List[Dict], *, page_number: int) -> List[Dict]:
    ordered: List[Tuple[Tuple[int, int, int], Dict]] = []
    fallback_rank = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if int(block.get('pageNumber') or page_number or 1) != int(page_number or 1):
            fallback_rank += 1
            ordered.append(((10**9, 10**9, fallback_rank), block))
            continue
        bbox = block.get('bbox') if isinstance(block.get('bbox'), dict) else None
        if isinstance(bbox, dict):
            top = int(bbox.get('top', 10**9))
            left = int(bbox.get('left', 10**9))
            fallback_rank += 1
            ordered.append(((top, left, fallback_rank), block))
        else:
            fallback_rank += 1
            ordered.append(((10**9, 10**9, fallback_rank), block))

    ordered.sort(key=lambda item: item[0])
    result: List[Dict] = []
    for idx, (_key, block) in enumerate(ordered, start=1):
        try:
            block['readingOrder'] = int(idx)
        except Exception:
            pass
        result.append(block)
    return result







def _render_page_to_jpeg_bytes(page: fitz.Page, max_width: Optional[int] = None, quality: int = 85, dpi: int = 300) -> bytes:
    """Render a PyMuPDF page to JPEG bytes.

    - Uses `dpi` to scale the page (matrix = dpi/72).
    - Resizes to `max_width` if provided.
    - Returns JPEG bytes; if JPEG creation fails or results are suspiciously small,
      falls back to PNG bytes.
    """
    try:
        scale = float(dpi) / 72.0 if dpi else 1.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        try:
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception:
            # fallback: use PIL.Image frombytes may fail for some pix formats
            png = pix.tobytes("png")
            img = Image.open(io.BytesIO(png)).convert("RGB")

        if max_width and img.width > max_width:
            ratio = float(max_width) / float(img.width)
            new_h = int(img.height * ratio)
            img = img.resize((int(max_width), new_h), Image.LANCZOS)

        buf = io.BytesIO()
        try:
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            jpg_bytes = buf.getvalue()
            if not jpg_bytes or len(jpg_bytes) < 2000:
                _print_utf('[PROBE-WARNING] JPEG render produced unexpectedly small output; using PNG fallback')
                buf = io.BytesIO()
                img.save(buf, format='PNG', optimize=True)
                return buf.getvalue()
            return jpg_bytes
        except Exception:
            buf = io.BytesIO()
            img.save(buf, format='PNG', optimize=True)
            return buf.getvalue()
    except Exception as e:
        _print_utf(f"[PROBE-WARNING] Render error: {e}")
        return b''


def _extract_embedded_images_from_page(doc: fitz.Document, page_index: int, out_dir: str) -> List[Dict]:
    """Extract embedded images on a page and save them to out_dir.

    Returns list of metadata dicts: {filename, path, page}
    """
    out: List[Dict] = []
    try:
        page = doc.load_page(page_index)
    except Exception:
        return out

    images = page.get_images(full=True)
    for idx, img in enumerate(images):
        xref = img[0]
        try:
            base = doc.extract_image(xref)
            img_bytes = base.get('image')
            ext = base.get('ext', 'png')
            filename = f'embedded_p{page_index+1}_{idx + 1}.{ext}'
            fpath = os.path.join(out_dir, filename)
            with open(fpath, 'wb') as f:
                f.write(img_bytes)
            out.append({'filename': filename, 'path': fpath, 'page': page_index + 1})
        except Exception:
            continue
    return out


def _detect_table_bboxes_from_image_bytes(image_bytes: bytes, debug: bool = False) -> List[Tuple[int, int, int, int]]:
    """Return list of bounding boxes (x,y,w,h) that likely contain tables.

    If OpenCV is not available, return empty list.
    """
    if not _HAS_CV2:
        return []
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # invert so lines become white
        gray = cv2.bitwise_not(gray)
        # adaptive threshold
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        horizontal = bw.copy()
        vertical = bw.copy()
        cols = horizontal.shape[1]
        rows = vertical.shape[0]
        horiz_size = max(10, cols // 30)
        vert_size = max(10, rows // 30)
        horiz_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
        horiz = cv2.erode(horizontal, horiz_structure)
        horiz = cv2.dilate(horiz, horiz_structure)

        vert_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
        vert = cv2.erode(vertical, vert_structure)
        vert = cv2.dilate(vert, vert_structure)

        mask = cv2.add(horiz, vert)
        # clean small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < 5000:
                continue
            # filter extremely narrow/tall
            if w < 50 or h < 30:
                continue
            boxes.append((x, y, w, h))
        # sort by top-left y then x
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes
    except Exception:
        return []


RED_ANNOTATION_STROKE_MIN_WIDTH = 0.9
RED_ANNOTATION_MIN_PAGE_AREA_RATIO = 0.015
RED_ANNOTATION_BORDER_TRIM_PX = 2


def _color_is_red(value: Any) -> bool:
    try:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return float(value) >= 0.85
        comps = list(value)
        if len(comps) < 3:
            return False
        red, green, blue = float(comps[0]), float(comps[1]), float(comps[2])
        return red >= 0.85 and green <= 0.25 and blue <= 0.25
    except Exception:
        return False


def _drawing_is_red_annotation_frame(drawing: Dict[str, Any]) -> bool:
    if not isinstance(drawing, dict):
        return False
    if not _color_is_red(drawing.get('color')):
        return False
    if drawing.get('fill') is not None:
        return False
    try:
        stroke_width = float(drawing.get('width') or 0.0)
    except Exception:
        stroke_width = 0.0
    return stroke_width >= RED_ANNOTATION_STROKE_MIN_WIDTH


def _map_pdf_rect_to_render_xywh(
    page: fitz.Page,
    rect: Any,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    page_rect = page.rect
    try:
        x0 = (float(rect.x0) - float(page_rect.x0)) / max(float(page_rect.width), 1.0) * float(img_w)
        y0 = (float(rect.y0) - float(page_rect.y0)) / max(float(page_rect.height), 1.0) * float(img_h)
        x1 = (float(rect.x1) - float(page_rect.x0)) / max(float(page_rect.width), 1.0) * float(img_w)
        y1 = (float(rect.y1) - float(page_rect.y0)) / max(float(page_rect.height), 1.0) * float(img_h)
    except Exception:
        return (0, 0, 1, 1)
    left = int(round(min(x0, x1)))
    top = int(round(min(y0, y1)))
    width = int(round(abs(x1 - x0)))
    height = int(round(abs(y1 - y0)))
    return _clip_xywh_to_page(left, top, max(1, width), max(1, height), int(img_w), int(img_h))


def _extract_red_annotation_boxes_from_page(
    page: fitz.Page,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    try:
        page_area = float(page.rect.width) * float(page.rect.height)
    except Exception:
        page_area = 0.0
    if page_area <= 0.0 or img_w <= 0 or img_h <= 0:
        return []

    boxes: List[Tuple[int, int, int, int]] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    for drawing in drawings:
        if not _drawing_is_red_annotation_frame(drawing):
            continue
        rect = drawing.get('rect')
        if rect is None:
            continue
        try:
            rect_area = float(rect.width) * float(rect.height)
        except Exception:
            continue
        if rect_area < page_area * RED_ANNOTATION_MIN_PAGE_AREA_RATIO:
            continue
        boxes.append(_map_pdf_rect_to_render_xywh(page, rect, img_w, img_h))

    try:
        annot_boxes = _extract_square_annotation_boxes_from_page(page, img_w, img_h, color='red')
        if annot_boxes:
            boxes.extend(annot_boxes)
    except Exception:
        pass

    try:
        boxes = _prune_overlapping_boxes(boxes)
    except Exception:
        pass
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _detect_red_annotation_boxes_from_image_bytes(image_bytes: bytes) -> List[Tuple[int, int, int, int]]:
    """Fallback: detect red closed rectangles directly on the rendered page image."""
    if not (_HAS_CV2 and _HAS_NUMPY) or not image_bytes:
        return []
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255])),
            cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255])),
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = float(w * h)
        boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 8:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            area = float(bw * bh)
            if area < page_area * RED_ANNOTATION_MIN_PAGE_AREA_RATIO:
                continue
            if bw < 40 or bh < 40:
                continue
            boxes.append((int(x), int(y), int(bw), int(bh)))

        try:
            boxes = _prune_overlapping_boxes(boxes)
        except Exception:
            pass
        return sorted(boxes, key=lambda b: (b[1], b[0]))
    except Exception:
        return []


def _infer_red_annotation_visual_type(
    crop_bytes: bytes,
    *,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[set] = None,
) -> str:
    if not crop_bytes:
        return 'figure'

    h_long, v_long, intersections = _line_evidence_from_crop_bytes(crop_bytes)
    strong_grid = (
        h_long >= FALLBACK_MIN_H_LONG
        and v_long >= FALLBACK_MIN_V_LONG
        and intersections >= FALLBACK_MIN_INTERSECTIONS
    )
    if strong_grid:
        return 'table'

    try:
        ocr_text = _ocr_image_bytes(
            crop_bytes,
            ocr_engine=ocr_engine,
            repeated_watermark_candidates=repeated_watermark_candidates,
        ) or ''
        table_rows = _text_to_table_rows(ocr_text)
        max_cols = max((len(row) for row in (table_rows or [])), default=0)
        if table_rows and len(table_rows) >= 2 and max_cols >= 2:
            return 'table'
    except Exception:
        pass
    return 'figure'


def _trim_red_annotation_crop_box(
    box: Tuple[int, int, int, int],
    page_w: int,
    page_h: int,
) -> Tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in box]
    inset = RED_ANNOTATION_BORDER_TRIM_PX
    return _clip_xywh_to_page(
        x + inset,
        y + inset,
        max(1, w - 2 * inset),
        max(1, h - 2 * inset),
        int(page_w),
        int(page_h),
    )


def _detect_red_annotation_layout_blocks(
    page: fitz.Page,
    image_bytes: bytes,
    *,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[set] = None,
) -> List[Dict[str, Any]]:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_w, img_h = img.size
    except Exception:
        return []

    boxes = _extract_red_annotation_boxes_from_page(page, img_w, img_h)
    if not boxes:
        boxes = _detect_red_annotation_boxes_from_image_bytes(image_bytes)

    if not boxes:
        return []

    layout_blocks: List[Dict[str, Any]] = []
    for box in boxes:
        try:
            x, y, w, h = _trim_red_annotation_crop_box(box, img_w, img_h)
            crop = img.crop((x, y, x + w, y + h))
            buf = io.BytesIO()
            crop.save(buf, format='PNG', optimize=True)
            crop_bytes = buf.getvalue()
            visual_type = _infer_red_annotation_visual_type(
                crop_bytes,
                ocr_engine=ocr_engine,
                repeated_watermark_candidates=repeated_watermark_candidates,
            )
            layout_blocks.append(
                {
                    'type': visual_type,
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'source': 'red_annotation',
                }
            )
        except Exception:
            continue

    try:
        print(f"[TRACE-RED] page={page.number + 1} accepted={len(layout_blocks)} red annotation boxes")
    except Exception:
        pass
    return layout_blocks


BLUE_ANNOTATION_MIN_PAGE_AREA_RATIO = 0.0015
BLUE_ANNOTATION_MIN_RENDER_HEIGHT_PX = 14
BLUE_ANNOTATION_BODY_EXCLUSION_OVERLAP_RATIO = 0.55
BLUE_ANNOTATION_BODY_PRESERVE_MIN_CHARS = 36


def _color_is_blue(value: Any) -> bool:
    try:
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return False
        comps = list(value)
        if len(comps) < 3:
            return False
        red, green, blue = float(comps[0]), float(comps[1]), float(comps[2])
        if blue >= 0.75 and red <= 0.35 and green <= 0.55:
            return True
        if blue >= 0.70 and blue >= max(red, green) * 1.15 and red <= 0.55 and green <= 0.65:
            return True
        return False
    except Exception:
        return False


def _drawing_is_blue_annotation_frame(drawing: Dict[str, Any]) -> bool:
    if not isinstance(drawing, dict):
        return False
    if not _color_is_blue(drawing.get('color')):
        return False
    if drawing.get('fill') is not None:
        return False
    try:
        stroke_width = float(drawing.get('width') or 0.0)
    except Exception:
        stroke_width = 0.0
    return stroke_width >= RED_ANNOTATION_STROKE_MIN_WIDTH


def _map_render_xywh_to_pdf_rect(
    page: fitz.Page,
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
) -> fitz.Rect:
    page_rect = page.rect
    try:
        x, y, w, h = [int(v) for v in box]
        x0 = float(page_rect.x0) + (float(x) / float(max(img_w, 1))) * float(page_rect.width)
        y0 = float(page_rect.y0) + (float(y) / float(max(img_h, 1))) * float(page_rect.height)
        x1 = float(page_rect.x0) + (float(x + w) / float(max(img_w, 1))) * float(page_rect.width)
        y1 = float(page_rect.y0) + (float(y + h) / float(max(img_h, 1))) * float(page_rect.height)
        return fitz.Rect(x0, y0, x1, y1)
    except Exception:
        return page.rect


def _extract_square_annotation_boxes_from_page(
    page: fitz.Page,
    img_w: int,
    img_h: int,
    *,
    color: str,
) -> List[Tuple[int, int, int, int]]:
    if img_w <= 0 or img_h <= 0:
        return []
    try:
        page_area = float(page.rect.width) * float(page.rect.height)
    except Exception:
        page_area = 0.0
    if page_area <= 0.0:
        return []

    min_area_ratio = (
        RED_ANNOTATION_MIN_PAGE_AREA_RATIO
        if str(color or '').strip().lower() == 'red'
        else BLUE_ANNOTATION_MIN_PAGE_AREA_RATIO
    )
    boxes: List[Tuple[int, int, int, int]] = []
    try:
        annots = page.annots()
    except Exception:
        annots = None
    if not annots:
        return boxes

    for annot in annots:
        try:
            if annot.type[0] != fitz.PDF_ANNOT_SQUARE:
                continue
            stroke = (annot.colors or {}).get('stroke')
            if not stroke:
                continue
            is_red = _color_is_red(stroke)
            is_blue = _color_is_blue(stroke)
            if color == 'red' and not is_red:
                continue
            if color == 'blue' and not is_blue:
                continue
            rect = annot.rect
            rect_area = float(rect.width) * float(rect.height)
            if rect_area < page_area * min_area_ratio:
                continue
            boxes.append(_map_pdf_rect_to_render_xywh(page, rect, img_w, img_h))
        except Exception:
            continue

    try:
        boxes = _prune_overlapping_boxes(boxes)
    except Exception:
        pass
    return sorted(boxes, key=lambda b: (b[1], b[0]))


def _assemble_blue_drawing_frame_boxes(
    page: fitz.Page,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    if img_w <= 0 or img_h <= 0:
        return []
    segments: List[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for drawing in drawings:
        if not _drawing_is_blue_annotation_frame(drawing):
            continue
        rect = drawing.get('rect')
        if rect is None:
            continue
        segments.append(rect)
    if len(segments) < 3:
        return []

    try:
        x0 = min(float(r.x0) for r in segments)
        y0 = min(float(r.y0) for r in segments)
        x1 = max(float(r.x1) for r in segments)
        y1 = max(float(r.y1) for r in segments)
        union = fitz.Rect(x0, y0, x1, y1)
        render_box = _map_pdf_rect_to_render_xywh(page, union, img_w, img_h)
        _x, _y, _w, _h = render_box
        if _h < BLUE_ANNOTATION_MIN_RENDER_HEIGHT_PX:
            return []
        page_area = float(img_w) * float(img_h)
        if page_area > 0 and float(_w * _h) / page_area < BLUE_ANNOTATION_MIN_PAGE_AREA_RATIO:
            return []
        return [render_box]
    except Exception:
        return []


def _detect_blue_annotation_boxes_from_image_bytes(image_bytes: bytes) -> List[Tuple[int, int, int, int]]:
    if not (_HAS_CV2 and _HAS_NUMPY) or not image_bytes:
        return []
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return []
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return []

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([90, 70, 50]), np.array([135, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = float(w * h)
        boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) < 4 or len(approx) > 10:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            area = float(bw * bh)
            if area < page_area * BLUE_ANNOTATION_MIN_PAGE_AREA_RATIO:
                continue
            if bw < 30 or bh < BLUE_ANNOTATION_MIN_RENDER_HEIGHT_PX:
                continue
            boxes.append((int(x), int(y), int(bw), int(bh)))

        try:
            boxes = _prune_overlapping_boxes(boxes)
        except Exception:
            pass
        return sorted(boxes, key=lambda b: (b[1], b[0]))
    except Exception:
        return []


def _extract_blue_annotation_boxes_from_page(
    page: fitz.Page,
    img_w: int,
    img_h: int,
    image_bytes: Optional[bytes] = None,
) -> List[Tuple[int, int, int, int]]:
    boxes = _extract_square_annotation_boxes_from_page(page, img_w, img_h, color='blue')
    if not boxes:
        boxes = _assemble_blue_drawing_frame_boxes(page, img_w, img_h)
    if not boxes and image_bytes:
        boxes = _detect_blue_annotation_boxes_from_image_bytes(image_bytes)
    return boxes



def _should_preserve_text_from_red_exclusion(text_box: Dict) -> bool:
    """Keep numbered headings and prose paragraphs that loosely overlap red crop frames."""
    text = str(text_box.get('text') or '').strip()
    if not text:
        return False
    compact = re.sub(r'\s+', '', text)
    if _detect_pdf_heading(text):
        return True
    if _PDF_SUBSECTION_RE.match(text) or _PDF_SUBSECTION_FUZZY_RE.match(text):
        return True
    if re.match(r'^[一二三四五六七八九十百千]+[、,，.．]', compact):
        return True
    if len(compact) >= BLUE_ANNOTATION_BODY_PRESERVE_MIN_CHARS and not is_likely_pdf_callout_or_annotation(text):
        return True
    return False


def _build_red_body_exclusion_regions(
    layout_blocks: List[Dict[str, Any]],
    page_w: int,
    page_h: int,
) -> List[Tuple[int, int, int, int]]:
    """Use accepted red visual crop boxes only, not loose raw annotation rectangles."""
    regions: List[Tuple[int, int, int, int]] = []
    if page_w <= 0 or page_h <= 0:
        return regions
    for item in layout_blocks or []:
        if not isinstance(item, dict):
            continue
        if str(item.get('source') or '').strip().lower() != 'red_annotation':
            continue
        bbox = item.get('bbox')
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            regions.append(
                _trim_red_annotation_crop_box(
                    (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    int(page_w),
                    int(page_h),
                )
            )
        except Exception:
            continue
    try:
        regions = _prune_overlapping_boxes(regions)
    except Exception:
        pass
    return regions


def _filter_text_boxes_excluding_regions(
    boxes: List[Dict],
    regions: List[Tuple[int, int, int, int]],
    *,
    min_overlap_ratio: float = BLUE_ANNOTATION_BODY_EXCLUSION_OVERLAP_RATIO,
) -> List[Dict]:
    if not boxes or not regions:
        return boxes
    kept: List[Dict] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        if _should_preserve_text_from_red_exclusion(box):
            kept.append(box)
            continue
        bbox = box.get('bbox')
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            kept.append(box)
            continue
        try:
            x, y, w, h = [int(v) for v in bbox[:4]]
        except Exception:
            kept.append(box)
            continue
        if w <= 0 or h <= 0:
            continue
        inner = (x, y, w, h)
        if any(_bbox_overlap_ratio_xywh(inner, region) >= float(min_overlap_ratio) for region in regions):
            continue
        kept.append(box)
    return kept



def _filter_ocr_callouts_inside_visual_crops(
    boxes: List[Dict],
    crop_regions: List[Tuple[int, int, int, int]],
) -> List[Dict]:
    """Drop only short OCR fragments clearly inside visual crops; never drop native PDF paragraphs."""
    if not boxes or not crop_regions:
        return boxes
    kept: List[Dict] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        text = str(box.get('text') or '').strip()
        if not text:
            continue
        if _should_preserve_text_from_red_exclusion(box):
            kept.append(box)
            continue
        bbox = box.get('bbox')
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            kept.append(box)
            continue
        try:
            inner = tuple(int(v) for v in bbox[:4])
        except Exception:
            kept.append(box)
            continue
        inside_crop = any(_bbox_center_inside_region(inner, region) for region in crop_regions)
        if inside_crop and is_likely_pdf_callout_or_annotation(text):
            continue
        kept.append(box)
    return kept


def _rebuild_plain_text_from_text_boxes(boxes: List[Dict]) -> str:
    lines: List[str] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        text = str(box.get('text') or '').strip()
        if text:
            lines.append(text)
    return '\n'.join(lines)


def _extract_blue_heading_hints_from_boxes(
    page: fitz.Page,
    boxes: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []
    for order_index, box in enumerate(boxes, start=1):
        try:
            pdf_rect = _map_render_xywh_to_pdf_rect(page, box, img_w, img_h)
            text = str(page.get_textbox(pdf_rect) or '').strip()
        except Exception:
            text = ''
        if not text:
            continue
        for line_index, line in enumerate([ln.strip() for ln in text.splitlines() if ln.strip()], start=1):
            detected = _detect_pdf_heading(line)
            if not detected:
                continue
            kind, _number, display = detected
            hints.append(
                {
                    'kind': kind,
                    'display': display,
                    'text': line,
                    'readingOrder': int(order_index * 100 + line_index),
                    'source': 'blue_annotation',
                }
            )
    return sorted(hints, key=lambda item: int(item.get('readingOrder') or 0))


def _apply_blue_page_heading_hints(
    page_hints: List[Dict[str, Any]],
    *,
    current_unit: str,
    current_section: str,
    current_subsection: str,
) -> Tuple[str, str, str, int]:
    """Apply blue hints in on-page reading order so later 节 resets earlier 小节 context."""
    applied = 0
    unit = current_unit
    section = current_section
    subsection = current_subsection
    ordered_hints = sorted(
        [hint for hint in page_hints if isinstance(hint, dict)],
        key=lambda item: int(item.get('readingOrder') or 0),
    )
    for hint in ordered_hints:
        kind = str(hint.get('kind') or '').strip().lower()
        if kind not in {'part', 'chapter', 'appendix', 'section', 'subsection'}:
            continue
        unit, section, subsection = _apply_heading_hint_to_context(
            hint,
            current_unit=unit,
            current_section=section,
            current_subsection=subsection,
        )
        applied += 1
    return unit, section, subsection, applied


def _apply_heading_hint_to_context(
    hint: Dict[str, Any],
    *,
    current_unit: str,
    current_section: str,
    current_subsection: str,
) -> Tuple[str, str, str]:
    kind = str(hint.get('kind') or '').strip().lower()
    display = str(hint.get('display') or '').strip()
    if not display:
        return current_unit, current_section, current_subsection
    if kind in {'part', 'chapter', 'appendix'}:
        return display, '', ''
    if kind == 'section':
        unit = current_unit or display
        return unit, display, ''
    if kind == 'subsection':
        unit = current_unit or current_section or display
        section = current_section or (current_unit if current_unit else '')
        return unit, section, display
    return current_unit, current_section, current_subsection


def _line_evidence_from_crop_bytes(
    crop_bytes: bytes,
    *,
    h_min_ratio: float = 0.35,
    v_min_ratio: float = 0.25,
) -> Tuple[int, int, int]:
    """Return (long_h_count, long_v_count, intersections) for a crop image."""
    if not (_HAS_NUMPY and _HAS_CV2) or not crop_bytes:
        return (0, 0, 0)
    try:
        arr = np.frombuffer(crop_bytes, dtype=np.uint8)
        blk = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if blk is None:
            return (0, 0, 0)

        h, w = blk.shape[:2]
        if h <= 0 or w <= 0:
            return (0, 0, 0)

        gray = cv2.cvtColor(blk, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        edges = cv2.Canny(th, 50, 150)
        min_line = max(10, int(w / 10))
        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=min_line, maxLineGap=5)
        except Exception:
            lines = None

        if lines is None:
            return (0, 0, 0)

        h_lines: List[Tuple[int, int, int, int]] = []
        v_lines: List[Tuple[int, int, int, int]] = []
        for ln in lines:
            try:
                x1, y1, x2, y2 = ln[0]
                dx = x2 - x1
                dy = y2 - y1
                if abs(dy) <= max(2, abs(dx) * 0.3):
                    h_lines.append((x1, y1, x2, y2))
                elif abs(dx) <= max(2, abs(dy) * 0.3):
                    v_lines.append((x1, y1, x2, y2))
            except Exception:
                continue

        h_min_span = max(24, int(w * h_min_ratio))
        v_min_span = max(24, int(h * v_min_ratio))
        h_long = [ln for ln in h_lines if abs(ln[2] - ln[0]) >= h_min_span]
        v_long = [ln for ln in v_lines if abs(ln[3] - ln[1]) >= v_min_span]
        intersections = _count_line_intersections(h_long, v_long, tol=3)
        return (len(h_long), len(v_long), int(intersections))
    except Exception:
        return (0, 0, 0)






def _split_stacked_figure_bbox(
    page_cv,
    x: int,
    y: int,
    w: int,
    h: int,
) -> List[Tuple[int, int, int, int]]:
    """Split tall figure blocks when two vertically stacked diagrams are merged."""
    if page_cv is None or not (_HAS_NUMPY and _HAS_CV2):
        return [(x, y, w, h)]
    if h < 260 or w < 320:
        return [(x, y, w, h)]

    try:
        blk = page_cv[y:y + h, x:x + w]
        if blk is None or getattr(blk, 'size', 0) == 0:
            return [(x, y, w, h)]

        mask = _build_visual_signal_mask(blk)
        if mask is None:
            gray = cv2.cvtColor(blk, cv2.COLOR_BGR2GRAY)
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        ink = np.sum(mask > 0, axis=1).astype(float)
        density = ink / max(1.0, float(w))
        smooth_win = max(5, int(h * 0.015))
        if smooth_win % 2 == 0:
            smooth_win += 1
        kernel = np.ones((smooth_win,), dtype=float) / float(smooth_win)
        density_s = np.convolve(density, kernel, mode='same')

        rs = int(h * 0.22)
        re = int(h * 0.82)
        if re <= rs:
            return [(x, y, w, h)]

        zone = density_s[rs:re]
        if zone.size <= 0:
            return [(x, y, w, h)]

        zone_med = float(np.median(zone))
        zone_min = float(np.min(zone))
        low_thr = min(zone_med * 0.62, (zone_med + zone_min) * 0.5)
        low_thr = max(0.003, low_thr)

        min_run = max(14, int(h * 0.04))
        best_score = -1.0
        best_mid = -1
        run_start = None
        for i in range(rs, re):
            if density_s[i] <= low_thr:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_end = i
                    run_len = run_end - run_start
                    if run_len >= min_run:
                        pre = density_s[max(rs, run_start - int(h * 0.12)):run_start]
                        post = density_s[run_end:min(re, run_end + int(h * 0.12))]
                        pre_peak = float(np.max(pre)) if pre.size > 0 else 0.0
                        post_peak = float(np.max(post)) if post.size > 0 else 0.0
                        valley = float(np.min(density_s[run_start:run_end]))
                        depth = max(0.0, ((pre_peak + post_peak) * 0.5) - valley)
                        score = float(run_len) * (1.0 + depth * 40.0)
                        if score > best_score:
                            best_score = score
                            best_mid = (run_start + run_end) // 2
                    run_start = None
        if run_start is not None:
            run_end = re
            run_len = run_end - run_start
            if run_len >= min_run:
                pre = density_s[max(rs, run_start - int(h * 0.12)):run_start]
                post = density_s[run_end:min(re, run_end + int(h * 0.12))]
                pre_peak = float(np.max(pre)) if pre.size > 0 else 0.0
                post_peak = float(np.max(post)) if post.size > 0 else 0.0
                valley = float(np.min(density_s[run_start:run_end]))
                depth = max(0.0, ((pre_peak + post_peak) * 0.5) - valley)
                score = float(run_len) * (1.0 + depth * 40.0)
                if score > best_score:
                    best_score = score
                    best_mid = (run_start + run_end) // 2

        # Component-gap fallback for cases where blank-band is not perfectly clean.
        if best_mid <= 0:
            try:
                num_labels, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
                min_cc_area = max(24, int(float(w * h) * 0.00012))
                comps: List[Tuple[int, int, int, int, int]] = []
                for label in range(1, int(num_labels)):
                    cx = int(stats[label, cv2.CC_STAT_LEFT])
                    cy = int(stats[label, cv2.CC_STAT_TOP])
                    cw = int(stats[label, cv2.CC_STAT_WIDTH])
                    ch = int(stats[label, cv2.CC_STAT_HEIGHT])
                    ca = int(stats[label, cv2.CC_STAT_AREA])
                    if ca < min_cc_area or cw <= 0 or ch <= 0:
                        continue
                    comps.append((cx, cy, cw, ch, ca))

                if len(comps) >= 4:
                    comps = sorted(comps, key=lambda c: c[1])
                    total_area = float(sum(c[4] for c in comps))
                    min_gap = max(12, int(h * 0.04))
                    best_gap = -1
                    best_gap_mid = -1
                    for i_comp in range(len(comps) - 1):
                        c0 = comps[i_comp]
                        c1 = comps[i_comp + 1]
                        c0_bottom = c0[1] + c0[3]
                        c1_top = c1[1]
                        gap = c1_top - c0_bottom
                        if gap < min_gap:
                            continue
                        cut_mid = c0_bottom + (gap // 2)
                        if cut_mid < int(h * 0.26) or cut_mid > int(h * 0.80):
                            continue
                        top_area = float(sum(c[4] for c in comps if (c[1] + c[3]) <= cut_mid))
                        bot_area = float(sum(c[4] for c in comps if c[1] >= cut_mid))
                        crossing_area = float(sum(c[4] for c in comps if c[1] < cut_mid < (c[1] + c[3])))
                        if total_area <= 0:
                            continue
                        if top_area < total_area * 0.20 or bot_area < total_area * 0.20:
                            continue
                        if crossing_area > total_area * 0.16:
                            continue
                        if gap > best_gap:
                            best_gap = gap
                            best_gap_mid = cut_mid

                    if best_gap_mid > 0:
                        best_mid = best_gap_mid
            except Exception:
                pass

        if best_mid <= 0:
            return [(x, y, w, h)]

        top_h = int(best_mid)
        bot_h = int(h - best_mid)
        if top_h < int(h * 0.30) or bot_h < int(h * 0.30):
            return [(x, y, w, h)]

        # Final guard: both halves should carry enough foreground signal, and
        # the split band itself should remain relatively sparse.
        try:
            total_signal = float(cv2.countNonZero(mask))
            if total_signal <= 0:
                return [(x, y, w, h)]
            top_signal = float(cv2.countNonZero(mask[:best_mid, :]))
            bot_signal = float(cv2.countNonZero(mask[best_mid:, :]))
            if top_signal < total_signal * 0.26 or bot_signal < total_signal * 0.26:
                return [(x, y, w, h)]

            band_half = max(4, int(h * 0.02))
            b0 = max(0, best_mid - band_half)
            b1 = min(h, best_mid + band_half)
            band = mask[b0:b1, :]
            band_density = float(cv2.countNonZero(band)) / float(max(1, band.shape[0] * band.shape[1]))
            local_med = float(np.median(zone)) if zone.size > 0 else 0.0
            if band_density > max(0.014, local_med * 0.82):
                return [(x, y, w, h)]
        except Exception:
            return [(x, y, w, h)]

        return [
            (x, y, w, top_h),
            (x, y + best_mid, w, bot_h),
        ]
    except Exception:
        return [(x, y, w, h)]


def _tight_bbox_to_mask_signal(
    mask,
    x: int,
    y: int,
    w: int,
    h: int,
    page_w: int,
    page_h: int,
    pad: int = 4,
) -> Tuple[int, int, int, int]:
    if mask is None or not (_HAS_NUMPY and _HAS_CV2):
        return (x, y, w, h)
    try:
        ys, xs = np.where(mask > 0)
        if xs.size <= 0 or ys.size <= 0:
            return (x, y, w, h)
        mx0 = max(0, int(xs.min()) - pad)
        my0 = max(0, int(ys.min()) - pad)
        mx1 = min(int(w), int(xs.max()) + 1 + pad)
        my1 = min(int(h), int(ys.max()) + 1 + pad)
        if mx1 <= mx0 or my1 <= my0:
            return (x, y, w, h)
        return _clip_xywh_to_page(x + mx0, y + my0, mx1 - mx0, my1 - my0, page_w, page_h)
    except Exception:
        return (x, y, w, h)


def _find_projection_gap(mask, axis: str) -> Tuple[int, float]:
    if mask is None or not (_HAS_NUMPY and _HAS_CV2):
        return (-1, 0.0)
    try:
        if axis == 'rows':
            major_len = int(mask.shape[0])
            minor_len = int(mask.shape[1])
            density = np.sum(mask > 0, axis=1).astype(float)
        else:
            major_len = int(mask.shape[1])
            minor_len = int(mask.shape[0])
            density = np.sum(mask > 0, axis=0).astype(float)
        if major_len < 120 or minor_len < 100:
            return (-1, 0.0)

        density = density / max(1.0, float(minor_len))
        smooth_win = max(5, int(major_len * 0.02))
        if smooth_win % 2 == 0:
            smooth_win += 1
        kernel = np.ones((smooth_win,), dtype=float) / float(smooth_win)
        density_s = np.convolve(density, kernel, mode='same')

        rs = int(major_len * 0.14)
        re = int(major_len * 0.86)
        if re <= rs:
            return (-1, 0.0)
        zone = density_s[rs:re]
        if zone.size <= 0:
            return (-1, 0.0)

        zone_med = float(np.median(zone))
        zone_min = float(np.min(zone))
        low_thr = min(zone_med * 0.58, (zone_med + zone_min) * 0.5)
        low_thr = max(0.0015, low_thr)
        min_run = max(10, int(major_len * 0.03))
        total_signal = float(cv2.countNonZero(mask))
        if total_signal <= 0.0:
            return (-1, 0.0)

        best_mid = -1
        best_score = 0.0

        def _score_run(run_start: int, run_end: int) -> Tuple[int, float]:
            run_len = int(run_end - run_start)
            if run_len < min_run:
                return (-1, 0.0)
            cut_mid = int((run_start + run_end) // 2)
            if cut_mid < int(major_len * 0.22) or cut_mid > int(major_len * 0.78):
                return (-1, 0.0)

            pre = density_s[max(rs, run_start - int(major_len * 0.12)):run_start]
            post = density_s[run_end:min(re, run_end + int(major_len * 0.12))]
            pre_peak = float(np.max(pre)) if pre.size > 0 else 0.0
            post_peak = float(np.max(post)) if post.size > 0 else 0.0
            if pre_peak <= 0.0 or post_peak <= 0.0:
                return (-1, 0.0)
            valley = float(np.min(density_s[run_start:run_end]))
            depth = max(0.0, ((pre_peak + post_peak) * 0.5) - valley)

            band_half = max(3, int(major_len * 0.015))
            if axis == 'rows':
                left_signal = float(cv2.countNonZero(mask[:cut_mid, :]))
                right_signal = float(cv2.countNonZero(mask[cut_mid:, :]))
                band = mask[max(0, cut_mid - band_half):min(major_len, cut_mid + band_half), :]
            else:
                left_signal = float(cv2.countNonZero(mask[:, :cut_mid]))
                right_signal = float(cv2.countNonZero(mask[:, cut_mid:]))
                band = mask[:, max(0, cut_mid - band_half):min(major_len, cut_mid + band_half)]

            if left_signal < total_signal * 0.18 or right_signal < total_signal * 0.18:
                return (-1, 0.0)

            band_area = float(max(1, band.shape[0] * band.shape[1]))
            band_density = float(cv2.countNonZero(band)) / band_area
            if band_density > max(0.018, zone_med * 0.78):
                return (-1, 0.0)

            return (cut_mid, float(run_len) * (1.0 + depth * 60.0))

        run_start = None
        for idx in range(rs, re):
            if density_s[idx] <= low_thr:
                if run_start is None:
                    run_start = idx
            else:
                if run_start is not None:
                    cand_mid, cand_score = _score_run(run_start, idx)
                    if cand_score > best_score:
                        best_mid = cand_mid
                        best_score = cand_score
                    run_start = None
        if run_start is not None:
            cand_mid, cand_score = _score_run(run_start, re)
            if cand_score > best_score:
                best_mid = cand_mid
                best_score = cand_score

        return (best_mid, best_score)
    except Exception:
        return (-1, 0.0)


def _should_split_figure_bbox(page_cv, bbox: Tuple[int, int, int, int]) -> bool:
    """Require strong compound-layout evidence before splitting a figure.

    Technical drawings often contain large internal blank bands, so size alone is
    not enough to justify decomposition.
    """
    if page_cv is None or not (_HAS_NUMPY and _HAS_CV2):
        return False
    try:
        page_h, page_w = page_cv.shape[:2]
        x, y, w, h = _clip_xywh_to_page(*bbox, page_w, page_h)
        if w < 260 or h < 220:
            return False

        blk = page_cv[y:y + h, x:x + w]
        if blk is None or getattr(blk, 'size', 0) == 0:
            return False

        mask = _build_visual_signal_mask(blk)
        if mask is None:
            return False

        total_signal = float(cv2.countNonZero(mask))
        if total_signal <= 0.0:
            return False

        row_mid, row_score = _find_projection_gap(mask, 'rows')
        col_mid, col_score = _find_projection_gap(mask, 'cols')

        use_axis = ''
        split_mid = -1
        split_score = 0.0
        if row_mid > 0 and row_score >= col_score:
            use_axis = 'rows'
            split_mid = row_mid
            split_score = row_score
        elif col_mid > 0:
            use_axis = 'cols'
            split_mid = col_mid
            split_score = col_score
        if not use_axis:
            return False

        if split_score < max(22.0, float(min(w, h)) * 0.055):
            return False

        band_half = max(4, int(float(min(w, h)) * 0.018))
        if use_axis == 'rows':
            leading_signal = float(cv2.countNonZero(mask[:split_mid, :]))
            trailing_signal = float(cv2.countNonZero(mask[split_mid:, :]))
            band = mask[max(0, split_mid - band_half):min(h, split_mid + band_half), :]
        else:
            leading_signal = float(cv2.countNonZero(mask[:, :split_mid]))
            trailing_signal = float(cv2.countNonZero(mask[:, split_mid:]))
            band = mask[:, max(0, split_mid - band_half):min(w, split_mid + band_half)]

        weaker_side_ratio = min(leading_signal, trailing_signal) / float(max(1.0, total_signal))
        if weaker_side_ratio < 0.22:
            return False

        band_area = float(max(1, band.shape[0] * band.shape[1]))
        band_density = float(cv2.countNonZero(band)) / band_area
        signal_density = total_signal / float(max(1, w * h))
        if band_density > max(0.02, signal_density * 0.34):
            return False
        return True
    except Exception:
        return False


def _split_bbox_by_projection(
    page_cv,
    bbox: Tuple[int, int, int, int],
    kind: str,
) -> List[Tuple[int, int, int, int]]:
    if page_cv is None or not (_HAS_NUMPY and _HAS_CV2):
        return [bbox]
    try:
        page_h, page_w = page_cv.shape[:2]
        x, y, w, h = _clip_xywh_to_page(*bbox, page_w, page_h)
        if w < 220 or h < 170:
            return [(x, y, w, h)]

        blk = page_cv[y:y + h, x:x + w]
        if blk is None or getattr(blk, 'size', 0) == 0:
            return [(x, y, w, h)]

        mask = _build_visual_signal_mask(blk)
        if mask is None:
            return [(x, y, w, h)]
        try:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        except Exception:
            pass
        if float(cv2.countNonZero(mask)) <= 0.0:
            return [(x, y, w, h)]

        row_mid, row_score = _find_projection_gap(mask, 'rows')
        col_mid, col_score = _find_projection_gap(mask, 'cols')

        if kind == 'figure':
            if h >= 220:
                row_score *= 1.06
            if h >= int(float(w) * 0.82):
                row_score *= 1.12
            if w >= int(float(h) * 1.05):
                col_score *= 1.03
        elif kind == 'table':
            row_score *= 1.04
            col_score *= 1.04

        use_axis = ''
        split_mid = -1
        split_score = 0.0
        if row_mid > 0 and row_score >= col_score:
            use_axis = 'rows'
            split_mid = row_mid
            split_score = row_score
        elif col_mid > 0:
            use_axis = 'cols'
            split_mid = col_mid
            split_score = col_score
        if not use_axis:
            return [(x, y, w, h)]

        if split_score < max(10.0, float(min(w, h)) * 0.03):
            return [(x, y, w, h)]

        if use_axis == 'rows':
            if split_mid <= 0 or split_mid >= h:
                return [(x, y, w, h)]
            split_boxes = [
                _tight_bbox_to_mask_signal(mask[:split_mid, :], x, y, w, split_mid, page_w, page_h),
                _tight_bbox_to_mask_signal(mask[split_mid:, :], x, y + split_mid, w, h - split_mid, page_w, page_h),
            ]
        else:
            if split_mid <= 0 or split_mid >= w:
                return [(x, y, w, h)]
            split_boxes = [
                _tight_bbox_to_mask_signal(mask[:, :split_mid], x, y, split_mid, h, page_w, page_h),
                _tight_bbox_to_mask_signal(mask[:, split_mid:], x + split_mid, y, w - split_mid, h, page_w, page_h),
            ]

        for sx, sy, sw, sh in split_boxes:
            if sw < max(140, int(w * 0.18)) or sh < max(110, int(h * 0.18)):
                return [(x, y, w, h)]

        return _prune_overlapping_boxes(split_boxes)
    except Exception:
        return [bbox]


def _merge_tiled_table_split_boxes(
    boxes: List[Tuple[int, int, int, int]],
    parent_bbox: Tuple[int, int, int, int],
) -> List[Tuple[int, int, int, int]]:
    if len(boxes) <= 1:
        return boxes
    try:
        _, _, parent_w, parent_h = parent_bbox
        gap_x_thr = max(2, int(float(parent_w) * 0.003))
        gap_y_thr = max(2, int(float(parent_h) * 0.004))
        merged = list(boxes)
        while len(merged) > 1:
            best_pair = None
            best_score = 0.0
            for i in range(len(merged)):
                ax, ay, aw, ah = merged[i]
                ax1 = ax + aw
                ay1 = ay + ah
                for j in range(i + 1, len(merged)):
                    bx, by, bw, bh = merged[j]
                    bx1 = bx + bw
                    by1 = by + bh
                    overlap_h = _bbox_overlap_1d(ay, ay1, by, by1)
                    overlap_w = _bbox_overlap_1d(ax, ax1, bx, bx1)
                    x_gap = _bbox_gap_1d(ax, ax1, bx, bx1)
                    y_gap = _bbox_gap_1d(ay, ay1, by, by1)
                    y_overlap_ratio = float(overlap_h) / float(max(1, min(ah, bh)))
                    x_overlap_ratio = float(overlap_w) / float(max(1, min(aw, bw)))

                    score = 0.0
                    if y_overlap_ratio >= 0.72 and x_gap <= gap_x_thr:
                        score = y_overlap_ratio * 10.0 + (1.0 - (float(x_gap) / float(max(1, gap_x_thr + 1))))
                    elif x_overlap_ratio >= 0.72 and y_gap <= gap_y_thr:
                        score = x_overlap_ratio * 10.0 + (1.0 - (float(y_gap) / float(max(1, gap_y_thr + 1))))
                    if score <= 0.0:
                        continue

                    union_x0 = min(ax, bx)
                    union_y0 = min(ay, by)
                    union_x1 = max(ax1, bx1)
                    union_y1 = max(ay1, by1)
                    union_area = float(max(1, (union_x1 - union_x0) * (union_y1 - union_y0)))
                    source_area = float(max(1, aw * ah) + max(1, bw * bh))
                    if union_area > source_area * 1.02:
                        continue

                    if score > best_score:
                        best_score = score
                        best_pair = (i, j, (union_x0, union_y0, union_x1 - union_x0, union_y1 - union_y0))

            if best_pair is None:
                break

            i, j, union_box = best_pair
            next_boxes: List[Tuple[int, int, int, int]] = []
            for idx, box in enumerate(merged):
                if idx not in (i, j):
                    next_boxes.append(box)
            next_boxes.append(union_box)
            try:
                merged = _prune_overlapping_boxes(next_boxes)
            except Exception:
                merged = next_boxes
        return merged
    except Exception:
        return boxes


def _decompose_compound_visual_bbox(
    page_cv,
    bbox: Tuple[int, int, int, int],
    kind: str,
    depth: int = 0,
) -> List[Tuple[int, int, int, int]]:
    if page_cv is None or not (_HAS_NUMPY and _HAS_CV2):
        return [bbox]
    if kind not in ('figure', 'table'):
        return [bbox]
    try:
        x, y, w, h = bbox
        if depth >= 2 or w < 220 or h < 170:
            return [bbox]

        if kind == 'figure' and depth >= 1:
            return [bbox]

        split_boxes: List[Tuple[int, int, int, int]] = [bbox]
        if kind == 'figure' and w >= 260 and h >= 220:
            stacked = _split_stacked_figure_bbox(page_cv, x, y, w, h)
            if len(stacked) > 1:
                split_boxes = stacked

        if len(split_boxes) == 1:
            split_boxes = _split_bbox_by_projection(page_cv, bbox, kind)

        if len(split_boxes) <= 1:
            return [bbox]

        out: List[Tuple[int, int, int, int]] = []
        for child in split_boxes:
            if kind == 'figure':
                out.append(child)
            else:
                out.extend(_decompose_compound_visual_bbox(page_cv, child, kind, depth + 1))

        out = _prune_overlapping_boxes(out)
        if kind == 'figure' and len(out) > 1:
            out = _suppress_nested_figure_boxes(out)
        if kind == 'table' and len(out) > 1:
            out = _merge_tiled_table_split_boxes(out, bbox)
            out = _prune_overlapping_boxes(out)

        if len(out) <= 1:
            if kind == 'table' and out:
                return out
            return [bbox]
        if len(out) > 6:
            return [bbox]

        parent_area = float(max(1, w * h))
        child_area_sum = float(sum(max(1, sw * sh) for _, _, sw, sh in out))
        if child_area_sum < parent_area * 0.40 or child_area_sum > parent_area * 1.85:
            return [bbox]
        return out
    except Exception:
        return [bbox]


def _detect_layout_blocks(layout_res, image_bytes: bytes, table_threshold: float = 0.2, kb_image_dir: Optional[str] = None) -> List[Dict]:
    """Detect layout blocks from pre-computed layout detector output.

    Signature: _detect_layout_blocks(layout_res, image_bytes, table_threshold=...)
    `layout_res` may come from YOLO layout detection, PP-Structure, or a mixed list.
    Returns a filtered list of dicts: {'type': str, 'bbox': (x,y,w,h), 'confidence': optional}
    """
    try:
        print(f"[TRACE-2] _detect_layout_blocks entered: table_threshold={table_threshold}, image_bytes_len={len(image_bytes) if image_bytes else 0}, layout_res_type={type(layout_res)}")
    except Exception:
        pass
    # Accept list/dict/None; normalize below and fallback to CV heuristics when needed.
    try:
        if layout_res is None:
            try:
                print("[WARN] layout_res is None, will try fallback normalization/detection")
            except Exception:
                pass
        elif not isinstance(layout_res, (list, dict)):
            try:
                print(f"[WARN] layout_res unexpected type={type(layout_res)}, will try fallback normalization/detection")
            except Exception:
                pass
    except Exception:
        try:
            print("[WARN] layout_res pre-check failed, continuing with fallback path")
        except Exception:
            pass
    # (removed duplicate broken image-decode block; corrected block follows)
    # Image decode check (defensive): verify image_bytes can be decoded by OpenCV
    try:
        if _HAS_NUMPY and _HAS_CV2:
            try:
                nparr = np.frombuffer(image_bytes, dtype=np.uint8)
                dec_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if dec_img is None:
                    print("[ERROR] OpenCV failed to decode image_bytes!")
                else:
                    try:
                        print(f"[PROBE-IMAGE] Image decoded successfully: {getattr(dec_img, 'shape', None)}")
                    except Exception:
                        print("[PROBE-IMAGE] Image decoded successfully (shape unknown)")
            except Exception as e:
                try:
                    print(f"[ERROR] Exception during image decode check: {e}")
                except Exception:
                    pass
        else:
            try:
                print(f"[PROBE-IMAGE] Skipping cv2 decode: _HAS_NUMPY={_HAS_NUMPY}, _HAS_CV2={_HAS_CV2}")
            except Exception:
                pass
    except Exception:
        pass
    
    # At this point layout_res is already supplied by caller and must be a list.
    res = layout_res
    try:
        try:
            print(f"[PROBE-DEBUG] Using provided layout_res: type={type(res)}, preview={str(res)[:100] if res else 'EMPTY'}")
        except Exception:
            pass
    except Exception:
        pass

    # Defensive normalization: adapt to various detector return shapes if needed.
    items = _coerce_layout_items(res)

    if not isinstance(items, list):
        try:
            print(f"[PROBE-DEBUG] Could not normalize layout result to list; res_type={type(res)}, res_keys={(list(res.keys()) if isinstance(res, dict) else 'N/A')}")
        except Exception:
            pass

        # CV fallback: synthesize table blocks from line-structure detection.
        try:
            fb_boxes = _detect_table_bboxes_from_image_bytes(image_bytes, debug=False)
        except Exception:
            fb_boxes = []

        fallback_source = 'cv_fallback'
        if fb_boxes:
            # Keep fallback boxes even on weak pages, but tag them so the
            # acceptance rules can be stricter downstream.
            try:
                if not _page_has_structural_visual_signal(image_bytes):
                    fallback_source = 'cv_fallback_weak'
                    try:
                        print("[INFO] CV fallback boxes found but page signal is weak; using stricter weak-signal rules")
                    except Exception:
                        pass
            except Exception:
                pass

        if fb_boxes:
            try:
                fb_boxes = _prune_overlapping_boxes(fb_boxes)
            except Exception:
                pass

            page_area_fb = None
            try:
                pimg_fb = Image.open(io.BytesIO(image_bytes))
                page_area_fb = float(pimg_fb.width) * float(pimg_fb.height)
            except Exception:
                page_area_fb = None

            synth_items: List[Dict] = []
            for bx in fb_boxes:
                try:
                    x, y, w, h = bx
                    x = int(max(0, x)); y = int(max(0, y)); w = int(max(0, w)); h = int(max(0, h))
                    if w <= 0 or h <= 0:
                        continue
                    if page_area_fb:
                        ratio_fb = (float(w) * float(h)) / page_area_fb
                        # Avoid promoting oversized regions from noisy CV fallback.
                        if ratio_fb > CV_FALLBACK_MAX_AREA_RATIO:
                            continue
                    synth_items.append({
                        'type': 'table',
                        'bbox': [x, y, x + w, y + h],
                        'score': 0.66,
                        'source': fallback_source,
                    })
                except Exception:
                    continue

            if synth_items:
                items = synth_items
                try:
                    print(f"[INFO] Using CV fallback layout blocks: {len(items)}")
                except Exception:
                    pass

    if not isinstance(items, list):
        return []

    try:
        try:
            print(f"[TRACE-2] Layout results count: {len(items) if items else 0}")
        except Exception:
            pass
        raw_blocks: List[Dict] = []
        for it in items:
            try:
                try:
                    print(f"[TRACE-1] Checking block type: {it.get('type') if isinstance(it, dict) else 'N/A'}")
                except Exception:
                    pass
                typ = it.get('type') or it.get('label') or it.get('class') or 'table'
                bbox = it.get('bbox') or it.get('box') or it.get('rect')
                if bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    w = int(x2 - x1) if x2 > x1 else int(bbox[2])
                    h = int(y2 - y1) if y2 > y1 else int(bbox[3])
                    src = str(it.get('source') or 'layout') if isinstance(it, dict) else 'layout'
                    conf = None
                    for k in ('score', 'confidence', 'prob'):
                        if isinstance(it, dict) and k in it:
                            try:
                                conf = float(it[k])
                                break
                            except Exception:
                                continue

                    block_record = {
                        'type': str(typ).lower(),
                        'bbox': (int(x1), int(y1), int(w), int(h)),
                        'confidence': conf,
                        'source': str(src).lower()
                    }

                    # Normalize and pass through table structure if available
                    if isinstance(it, dict) and block_record['type'] == 'table' and 'res' in it:
                        try:
                            # it['res'] from PP-Structure is a list of cell dicts:
                            # [{'text_res': [...], 'bbox': [...], 'table_column_id': [c1, c2], 'table_row_id': [r1, r2]}]
                            raw_cells = it['res']
                            normalized_cells = []
                            for rc in raw_cells:
                                if not isinstance(rc, dict): continue
                                r_ids = rc.get('table_row_id', [])
                                c_ids = rc.get('table_column_id', [])
                                if not r_ids or not c_ids: continue

                                cell_text = ""
                                if 'text_res' in rc and isinstance(rc['text_res'], list):
                                    cell_text = " ".join([str(tr[0]) if isinstance(tr, (list, tuple)) else str(tr) for tr in rc['text_res']]).strip()

                                normalized_cells.append({
                                    'row': min(r_ids),
                                    'col': min(c_ids),
                                    'rowSpan': (max(r_ids) - min(r_ids) + 1),
                                    'colSpan': (max(c_ids) - min(c_ids) + 1),
                                    'text': cell_text,
                                    'isHeader': False
                                })
                            if normalized_cells:
                                block_record['table_cells'] = normalized_cells
                        except Exception:
                            block_record['table_structure'] = it['res']

                    raw_blocks.append(block_record)
            except Exception:
                # continue processing remaining items; capture inner error for debugging
                try:
                    import traceback as _tb
                    print(f"[ERROR] Error processing block: {str(_tb.format_exc())}")
                except Exception:
                    pass
                continue
    except Exception as e:
        try:
            import traceback as _tb
            print(f"[ERROR] Error in block processing: {str(e)}\n{_tb.format_exc()}")
        except Exception:
            try:
                print(f"[ERROR] Error in block processing: {str(e)}")
            except Exception:
                pass
        return []

    # Compute page area via PIL
    page_area = None
    try:
        pimg = Image.open(io.BytesIO(image_bytes))
        pw, ph = pimg.size
        page_area = float(pw) * float(ph)
    except Exception:
        page_area = None

    # Prepare an OpenCV-decoded page image for optional line detection
    page_cv = None
    if _HAS_CV2:
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            page_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            page_cv = None

    # Page-level geometric fingerprint (Hough) pre-scan
    page_h_lines: List[Tuple[int, int, int, int]] = []
    page_v_lines: List[Tuple[int, int, int, int]] = []
    if page_cv is not None and _HAS_NUMPY and _HAS_CV2:
        try:
            gray_page = cv2.cvtColor(page_cv, cv2.COLOR_BGR2GRAY)
            thp = cv2.adaptiveThreshold(gray_page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            edges_p = cv2.Canny(thp, 50, 150)
            minLine = max(20, int(page_cv.shape[1] / 20))
            try:
                lines_p = cv2.HoughLinesP(edges_p, 1, np.pi/180, threshold=max(80, int(page_cv.shape[1] / 40)), minLineLength=minLine, maxLineGap=10)
            except Exception:
                lines_p = None
            if lines_p is not None:
                for ln in lines_p:
                    try:
                        x1, y1, x2, y2 = ln[0]
                        dx = x2 - x1
                        dy = y2 - y1
                        if abs(dy) <= max(2, abs(dx) * 0.3):
                            page_h_lines.append((x1, y1, x2, y2))
                        elif abs(dx) <= max(2, abs(dy) * 0.3):
                            page_v_lines.append((x1, y1, x2, y2))
                    except Exception:
                        continue
        except Exception:
            pass

    # Process all standardized raw blocks (including 'text') and apply
    # geometric + OCR-based override rules.
    filtered: List[Dict] = []
    for b in raw_blocks:
        try:
            typ = str(b.get('type', '')).lower()
            bbox = b.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            x, y, w, h = bbox[:4]
            x = int(max(0, x)); y = int(max(0, y)); w = int(max(0, w)); h = int(max(0, h))
            if w <= 0 or h <= 0:
                continue
            source = str(b.get('source', 'layout')).lower()
            is_cv_fallback = source.startswith('cv_fallback')
            is_cv_fallback_weak = (source == 'cv_fallback_weak')

            try:
                print(f"[TRACE-1] Found candidate block: type={typ}, source={source}, box={(x,y,w,h)}")
            except Exception:
                pass

            # Aspect-ratio and small-area filtering
            try:
                if float(w) / float(h) > MAX_ASPECT_RATIO:
                    try:
                        print(f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason=aspect_ratio, ratio={float(w)/float(h):.2f}, box={(x,y,w,h)}")
                    except Exception:
                        pass
                    continue
            except Exception:
                pass

            if page_area:
                block_area = float(w) * float(h)
                area_ratio = block_area / page_area
                if area_ratio < MIN_AREA_RATIO:
                    try:
                        print(f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason=small_area, area_ratio={area_ratio:.4f}, box={(x,y,w,h)}")
                    except Exception:
                        pass
                    continue
            else:
                area_ratio = 0.0

            # Crop bytes (prefer OpenCV view, fallback to PIL)
            crop_bytes = None
            if page_cv is not None and _HAS_NUMPY and _HAS_CV2:
                try:
                    block_img = page_cv[y:y+h, x:x+w]
                    if block_img is None or getattr(block_img, 'size', 0) == 0:
                        try:
                            print(f"[TRACE-3] ERROR: block_img is EMPTY for type {typ}")
                        except Exception:
                            pass
                    else:
                        try:
                            print(f"[TRACE-3] Image crop OK: type={typ}, size={getattr(block_img, 'shape', None)}")
                        except Exception:
                            pass
                        try:
                            _, enc = cv2.imencode('.png', block_img)
                            crop_bytes = enc.tobytes()
                        except Exception:
                            crop_bytes = None
                except Exception:
                    crop_bytes = None

            if crop_bytes is None:
                try:
                    p = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    c = p.crop((x, y, x + w, y + h))
                    buf = io.BytesIO()
                    c.save(buf, format='PNG', optimize=True)
                    crop_bytes = buf.getvalue()
                    try:
                        print(f"[TRACE-3] Image crop OK (PIL): type={typ}, size={(w,h)}")
                    except Exception:
                        pass
                except Exception:
                    crop_bytes = None

            # OCR summary: line count + average line length + weak table text signal.
            ocr_text_inner = ''
            text_count = 0
            avg_line_len = 0.0
            table_rows_inner: Optional[list] = None
            table_cols = 0
            try:
                if crop_bytes is not None:
                    ocr_text_inner = _ocr_image_bytes(crop_bytes, ocr_engine='auto') or ''
                    non_blank_lines = [ln for ln in ocr_text_inner.splitlines() if ln.strip()]
                    text_count = len(non_blank_lines)
                    if text_count > 0:
                        total_chars = sum(len(re.sub(r'\s+', '', ln)) for ln in non_blank_lines)
                        avg_line_len = float(total_chars) / float(text_count)
                    table_rows_inner = _text_to_table_rows(ocr_text_inner)
                    if table_rows_inner:
                        table_cols = max((len(r) for r in table_rows_inner), default=0)
            except Exception:
                ocr_text_inner = ''
                text_count = 0
                avg_line_len = 0.0
                table_rows_inner = None
                table_cols = 0

            # Block-level Hough detection
            h_lines: List[Tuple[int, int, int, int]] = []
            v_lines: List[Tuple[int, int, int, int]] = []
            signal_ratio = 0.0
            if crop_bytes is not None and _HAS_NUMPY and _HAS_CV2:
                try:
                    arr = np.frombuffer(crop_bytes, dtype=np.uint8)
                    blk = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if blk is not None:
                        try:
                            sig_mask = _build_visual_signal_mask(blk)
                            if sig_mask is not None and getattr(sig_mask, 'size', 0) > 0:
                                signal_ratio = float(cv2.countNonZero(sig_mask)) / float(max(1, sig_mask.shape[0] * sig_mask.shape[1]))
                        except Exception:
                            signal_ratio = 0.0
                        gray_blk = cv2.cvtColor(blk, cv2.COLOR_BGR2GRAY)
                        th = cv2.adaptiveThreshold(gray_blk, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                        edges = cv2.Canny(th, 50, 150)
                        minLine = max(10, int(w / 10))
                        try:
                            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=minLine, maxLineGap=5)
                        except Exception:
                            lines = None
                        if lines is not None:
                            for ln in lines:
                                try:
                                    x1b, y1b, x2b, y2b = ln[0]
                                    dx = x2b - x1b
                                    dy = y2b - y1b
                                    if abs(dy) <= max(2, abs(dx) * 0.3):
                                        h_lines.append((x1b, y1b, x2b, y2b))
                                    elif abs(dx) <= max(2, abs(dy) * 0.3):
                                        v_lines.append((x1b, y1b, x2b, y2b))
                                except Exception:
                                    continue
                except Exception:
                    pass

            # Structural evidence: prefer long grid lines and their intersections.
            h_long_lines: List[Tuple[int, int, int, int]] = []
            v_long_lines: List[Tuple[int, int, int, int]] = []
            intersections = 0
            try:
                h_min_span = max(24, int(w * 0.35))
                v_min_span = max(24, int(h * 0.25))
                h_long_lines = [ln for ln in h_lines if abs(ln[2] - ln[0]) >= h_min_span]
                v_long_lines = [ln for ln in v_lines if abs(ln[3] - ln[1]) >= v_min_span]
                intersections = _count_line_intersections(h_long_lines, v_long_lines, tol=3)
            except Exception:
                h_long_lines = []
                v_long_lines = []
                intersections = 0

            has_grid = (
                (len(h_long_lines) >= TABLE_MIN_H_LONG and len(v_long_lines) >= TABLE_MIN_V_LONG)
                or intersections >= TABLE_MIN_INTERSECTIONS
            )
            has_tabular_text = bool(table_rows_inner) and len(table_rows_inner) >= 2 and table_cols >= 2
            paragraph_like_text = (
                text_count >= 4
                and avg_line_len >= 12.5
                and not has_tabular_text
                and len(v_long_lines) == 0
            )
            conf = b.get('confidence')
            try:
                conf_val = float(conf) if conf is not None else None
            except Exception:
                conf_val = None

            # Acceptance rules: remove aggressive text->table forcing to avoid false positives.
            accept = False
            reject_reason = 'heuristics'
            try:
                if typ in SKIP_TEXT_TYPES:
                    has_decimal_clause_text = bool(
                        ocr_text_inner
                        and re.search(r'(?<!\d)\d{1,2}\.\d{1,2}\.\d{1,3}(?!\d)', ocr_text_inner)
                    )
                    if has_decimal_clause_text:
                        accept = True
                    elif paragraph_like_text and area_ratio >= 0.02:
                        accept = True
                    else:
                        reject_reason = 'weak_text_region_signal'
                elif typ == 'table':
                    if (
                        text_count >= TABLE_CAPTION_MIN_LINES
                        and avg_line_len >= TABLE_CAPTION_LINE_LEN
                        and not has_tabular_text
                        and len(v_long_lines) == 0
                    ):
                        reject_reason = 'caption_like_table'
                    elif is_cv_fallback:
                        strong_grid = (
                            len(h_long_lines) >= FALLBACK_MIN_H_LONG
                            and len(v_long_lines) >= FALLBACK_MIN_V_LONG
                            and intersections >= FALLBACK_MIN_INTERSECTIONS
                        )
                        medium_grid_with_text = (
                            has_tabular_text
                            and len(h_long_lines) >= max(TABLE_MIN_H_LONG, 3)
                            and len(v_long_lines) >= max(TABLE_MIN_V_LONG, 2)
                            and intersections >= 6
                            and area_ratio >= 0.04
                        )
                        if area_ratio > CV_FALLBACK_MAX_AREA_RATIO:
                            reject_reason = 'oversized_cv_fallback'
                        elif text_count >= 14 and avg_line_len >= 20.0 and not has_tabular_text:
                            reject_reason = 'text_dense_cv_fallback'
                        elif is_cv_fallback_weak and not strong_grid:
                            reject_reason = 'weak_page_signal'
                        elif strong_grid or medium_grid_with_text:
                            accept = True
                        else:
                            reject_reason = 'weak_cv_fallback_signal'
                    elif has_grid:
                        accept = True
                    elif conf_val is not None and conf_val >= 0.90 and has_tabular_text and len(h_long_lines) >= TABLE_MIN_H_LONG:
                        accept = True
                    elif conf_val is not None and conf_val >= 0.96 and text_count <= 12 and avg_line_len <= 12.0 and area_ratio >= 0.06:
                        accept = True
                    else:
                        reject_reason = 'weak_table_signal'
                elif typ == 'figure':
                    label_like_text = (
                        text_count <= 14
                        and avg_line_len <= 12.5
                    )
                    toc_like_text = (
                        has_tabular_text
                        and text_count >= 8
                        and avg_line_len >= 7.5
                        and len(v_long_lines) <= 1
                        and intersections < 8
                    )
                    has_line_geometry = (
                        (len(h_long_lines) >= 2 and len(v_long_lines) >= 1)
                        or intersections >= 3
                    )
                    if signal_ratio < 0.006 and text_count <= 2 and not has_line_geometry:
                        reject_reason = 'near_blank_figure'
                    elif toc_like_text and signal_ratio < 0.090:
                        reject_reason = 'toc_like_figure'
                    elif paragraph_like_text and signal_ratio < 0.085:
                        reject_reason = 'paragraph_like_figure'
                    elif area_ratio >= 0.04 and (has_line_geometry or label_like_text):
                        accept = True
                    else:
                        reject_reason = 'weak_figure_signal'
                elif typ == 'equation':
                    if text_count <= 20 or avg_line_len <= 14.0:
                        accept = True
                    else:
                        reject_reason = 'equation_text_density'
            except Exception:
                accept = False
                reject_reason = 'rule_error'

            # Confidence override for enormous blocks
            if accept and page_area:
                try:
                    block_area = float(w) * float(h)
                    ratio = block_area / page_area
                    if ratio > MAX_AREA_RATIO:
                        conf = b.get('confidence')
                        try:
                            conf_val = float(conf) if conf is not None else None
                        except Exception:
                            conf_val = None
                        if conf_val is None or conf_val < CONFIDENCE_HIGH:
                            accept = False
                            reject_reason = 'oversized_low_confidence'
                except Exception:
                    pass

            if accept:
                try:
                    print(
                        f"[TRACE-4] Final Decision for {typ}: keep_block=True, "
                        f"h={len(h_lines)}, v={len(v_lines)}, h_long={len(h_long_lines)}, v_long={len(v_long_lines)}, "
                        f"intersections={intersections}, text_lines={text_count}, avg_line_len={avg_line_len:.2f}, area_ratio={area_ratio:.4f}"
                    )
                except Exception:
                    pass
                accepted = {'type': typ, 'bbox': (int(x), int(y), int(w), int(h)), 'confidence': b.get('confidence'), 'source': source}
                if typ in SKIP_TEXT_TYPES and ocr_text_inner:
                    accepted['contentMarkdown'] = ocr_text_inner
                filtered.append(accepted)
            else:
                try:
                    print(
                        f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason={reject_reason}, "
                        f"box={(x,y,w,h)}, h={len(h_lines)}, v={len(v_lines)}, h_long={len(h_long_lines)}, "
                        f"v_long={len(v_long_lines)}, intersections={intersections}, text_lines={text_count}, avg_line_len={avg_line_len:.2f}"
                    )
                except Exception:
                    pass
        except Exception:
            continue

    # Final overlap suppression: especially useful when fallback produces nested
    # boxes that survived heuristics.
    if filtered:
        try:
            filtered_sorted = sorted(filtered, key=lambda it: (-(it['bbox'][2] * it['bbox'][3]), it['bbox'][1], it['bbox'][0]))
            deduped: List[Dict] = []
            for cand in filtered_sorted:
                try:
                    cb = cand.get('bbox')
                    if not cb or len(cb) < 4:
                        continue
                    csrc = str(cand.get('source', 'layout')).lower()
                    drop = False
                    for kept in deduped:
                        kb = kept.get('bbox')
                        if not kb or len(kb) < 4:
                            continue
                        ksrc = str(kept.get('source', 'layout')).lower()
                        # Keep overlap suppression conservative; apply strongly for
                        # fallback-vs-fallback and near-duplicate same-type blocks.
                        iou = _bbox_iou_xywh((cb[0], cb[1], cb[2], cb[3]), (kb[0], kb[1], kb[2], kb[3]))
                        same_type = str(cand.get('type', '')) == str(kept.get('type', ''))
                        if same_type and iou >= 0.82:
                            drop = True
                            break
                        if same_type and str(cand.get('type', '')).lower() in ('figure', 'equation'):
                            try:
                                cbox = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
                                kbox = (int(kb[0]), int(kb[1]), int(kb[2]), int(kb[3]))
                                c_area = float(max(1, cbox[2] * cbox[3]))
                                k_area = float(max(1, kbox[2] * kbox[3]))
                                coverage = _bbox_coverage_xywh(cbox, kbox)
                                if coverage >= 0.78 and c_area <= (k_area * 0.62):
                                    drop = True
                                    break
                            except Exception:
                                pass
                        if csrc.startswith('cv_fallback') and ksrc.startswith('cv_fallback') and _bbox_inside_xywh((cb[0], cb[1], cb[2], cb[3]), (kb[0], kb[1], kb[2], kb[3]), margin=8):
                            drop = True
                            break
                    if not drop:
                        deduped.append(cand)
                except Exception:
                    continue
            filtered = sorted(deduped, key=lambda it: (it['bbox'][1], it['bbox'][0]))
        except Exception:
            pass

    try:
        print(f"[TRACE-4] _detect_layout_blocks returning {len(filtered)} accepted blocks")
    except Exception:
        pass

    # If no blocks were accepted, save a diagnostic page image to kb_image_dir if provided
    if not filtered:
        try:
            out_dir = kb_image_dir if kb_image_dir else None
            if out_dir:
                try:
                    ensure_dir(out_dir)
                except Exception:
                    try:
                        os.makedirs(out_dir, exist_ok=True)
                    except Exception:
                        out_dir = None
            saved = False
            if _HAS_CV2 and page_cv is not None:
                try:
                    out_path = os.path.join(out_dir, 'diagnostic_p1.png') if out_dir else os.path.abspath('diagnostic_p1.png')
                    ok = cv2.imwrite(out_path, page_cv)
                    if ok:
                        try:
                            print(f"[PROBE-IMAGE] Saved diagnostic page to: {out_path}")
                        except Exception:
                            pass
                        saved = True
                except Exception:
                    saved = False
            if not saved:
                try:
                    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                    out_path = os.path.join(out_dir, 'diagnostic_p1.png') if out_dir else os.path.abspath('diagnostic_p1.png')
                    pil_img.save(out_path)
                    try:
                        print(f"[PROBE-IMAGE] Saved diagnostic page to: {out_path}")
                    except Exception:
                        pass
                    saved = True
                except Exception:
                    saved = False
            if not saved:
                try:
                    raw_path = os.path.abspath('test_page.raw')
                    with open(raw_path, 'wb') as _f:
                        _f.write(image_bytes)
                    try:
                        print(f"[PROBE-IMAGE] Saved raw diagnostic bytes to: {raw_path}")
                    except Exception:
                        pass
                except Exception:
                    try:
                        print("[ERROR] Failed to save diagnostic page")
                    except Exception:
                        pass
        except Exception:
            pass
    return filtered


def _save_crop_and_uri(page_num: int, idx: int, crop_bytes: bytes, assets_dir: str, file_id: str, ext: str = 'png') -> Tuple[str, str, str]:
    """Save crop bytes to assets dir and return (filename, abs_path, imageUri)."""
    # Ensure assets dir exists
    try:
        ensure_dir(assets_dir)
    except Exception:
        try:
            os.makedirs(assets_dir, exist_ok=True)
        except Exception:
            pass

    # Use `visual_p` prefix for locally generated visual crops so that the
    # `embedded_p...` pattern remains reserved for PDF-embedded images.
    filename = f'visual_p{page_num}_{idx}.{ext}'
    abs_path = os.path.join(assets_dir, filename)
    try:
        with open(abs_path, 'wb') as f:
            f.write(crop_bytes)
    except Exception:
        # try fallback name
        filename = f'visual_p{page_num}_{idx}_alt.{ext}'
        abs_path = os.path.join(assets_dir, filename)
        try:
            with open(abs_path, 'wb') as f:
                f.write(crop_bytes)
        except Exception:
            return (filename, abs_path, '')
    uri = f'file:///android_asset/kb/{file_id}/截图/{filename}'
    return (filename, abs_path, uri)


def _uri_basename(uri: str) -> str:
    """Return the trailing filename part of a URI/path-like string."""
    if not uri:
        return ''
    try:
        s = str(uri).strip()
        s = s.split('?', 1)[0].split('#', 1)[0]
        if '/' in s:
            s = s.rsplit('/', 1)[-1]
        if '\\' in s:
            s = s.rsplit('\\', 1)[-1]
        return s
    except Exception:
        return ''


def _clear_generated_page_assets(assets_dir: str, page_num: int) -> None:
    try:
        if not assets_dir or not os.path.isdir(assets_dir):
            return
        page_tag = str(int(page_num))
        patterns = (
            re.compile(rf'^(?:visual|table|legend|embedded)_p{page_tag}(?:_|$)', flags=re.IGNORECASE),
            re.compile(rf'^page_{page_tag}\.(?:jpg|jpeg|png)$', flags=re.IGNORECASE),
        )
        for name in os.listdir(assets_dir):
            try:
                if not any(p.match(name) for p in patterns):
                    continue
                path = os.path.join(assets_dir, name)
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                continue
    except Exception:
        return


def _compute_sha256(file_path: str) -> str:
    """Compute SHA256 for a file path; returns empty string on failure."""
    try:
        h = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ''


def _infer_entry_kind(entry: Dict) -> str:
    """Best-effort kind inference to match knowledge_base.json style."""
    blocks = entry.get('blocks') or []
    try:
        if any(((b.get('type') or '').lower() == 'table') for b in blocks if isinstance(b, dict)):
            return 'table'
        if any(((b.get('type') or '').lower() in ('figure', 'equation', 'image')) for b in blocks if isinstance(b, dict)):
            text = str(entry.get('contentMarkdown') or '').strip()
            return 'text' if text else 'image'
    except Exception:
        pass
    return 'text'


def _to_content_type(kind: str) -> str:
    k = (kind or '').strip().lower()
    if k == 'table':
        return 'table'
    if k in ('image', 'figure', 'equation'):
        return 'figure'
    return 'narrative'


def _entry_quality_metrics(entry: Dict) -> Dict[str, object]:
    text = str(entry.get('contentNormalized') or entry.get('contentMarkdown') or '').strip()
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    unit_name = str(entry.get('unitName') or '').strip()

    # Heuristic confidence: explicit taxonomy + non-trivial text length.
    has_taxonomy = bool(unit_name and unit_name != '未分类')
    has_text = len(text) >= 8
    confidence = 0.35
    if has_taxonomy:
        confidence += 0.3
    if has_text:
        confidence += 0.25
    if blocks:
        confidence += 0.1

    return {
        'classificationConfidence': round(min(confidence, 0.98), 3),
        'isProvisional': not has_taxonomy,
        'textLength': len(text),
    }


_CN_NUM_MAP = {
    '零': 0,
    '〇': 0,
    '一': 1,
    '二': 2,
    '两': 2,
    '三': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9,
}
_PDF_CHAPTER_RE = re.compile(r'^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*章(?:\s+|[：:]|$)(.*)$')
_PDF_PART_RE = re.compile(r'^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*篇(?:\s+|[：:]|$)(.*)$')
_PDF_SECTION_RE = re.compile(r'^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*节\s*(.*)$')
_PDF_SUBSECTION_RE = re.compile(r'^([一二三四五六七八九十百千]+)\s*[、,，.]\s*(.+)$')
_PDF_SUBSECTION_FUZZY_RE = re.compile(r'^[、,，.．·]\s*([\u4e00-\u9fff][^。；;！？!?：:|]{0,20})$')
_PDF_TOP_LEVEL_RE = re.compile(r'^([0-9]{1,2})(?![）\)\-—])\s+([\u4e00-\u9fff][^。；;！？!?：:|]{0,20})$')
_PDF_APPENDIX_RE = re.compile(r'^附录\s*([0-9A-Za-z一二三四五六七八九十百千〇零两]+)\s*(.*)$')
_PDF_INLINE_HEADING_START_RE = re.compile(
    r'(第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节]|'
    r'附录\s*[0-9A-Za-z一二三四五六七八九十百千〇零两]+|'
    r'[一二三四五六七八九十百0-9]+\s*、)'
)
_PDF_TOC_DOT_LEADER_RE = re.compile(r'[.．…]{6,}')
_PDF_HEADING_MARKER_RE = re.compile(r'(第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节]|附录\s*[0-9A-Za-z一二三四五六七八九十百千〇零两]+)')
_VISUAL_CAPTION_BODY_CUES = (
    '截止',
    '其中',
    '由于',
    '为保证',
    '通过',
    '沿线',
    '共有',
    '说明',
    '检查',
    '普速',
    '高铁',
    '当',
    '传统',
    '一般应用',
    '其特点',
    '特点是',
    '需要强调',
    '当电力',
    '系指',
)
_FIGURE_INLINE_REF_BODY_SPLIT_RE = re.compile(
    r'(?:（|\()\s*如图[^）)]{0,48}所示\s*(?:）|\))\s*'
)
_FIGURE_SECTION_OPENER_INLINE_REF_RE = re.compile(
    r'([\u4e00-\u9fff][\u4e00-\u9fff0-9A-Za-z（）()\-—]{1,48}?'
    r'(?:（|\()\s*如图[^）)]{0,48}所示\s*(?:）|\))\s*)'
)
_FIGURE_DIAGRAM_LEGEND_LEAK_RE = re.compile(
    r'([一二三四五六七八九十]+、[^。；;！？!?]{2,48})'
    r'(?:不接地系统\s*){1,2}'
    r'(?:经消弧线圈接地系统|经小电阻接地系统|直接接地系统)'
    r'(?=\s*一般应用|\s*其特点|\s*特点是|\s*只应用|\s*由于)'
)
_FIGURE_DIAGRAM_LEGEND_ONLY_RE = re.compile(
    r'^(?:不接地系统\s*){1,2}'
    r'(?:经消弧线圈接地系统|经小电阻接地系统|直接接地系统)\s*$'
)
_CABLE_DIAGRAM_LEGEND_ONLY_RE = re.compile(
    r'^(?:单端接地|双端接地)(?:\s+(?:单端接地|双端接地))*\s*$'
)
_VISUAL_CAPTION_SPLIT_RE = re.compile(
    r'([①②③④⑤⑥⑦⑧⑨⑩]\s*[\u4e00-\u9fff]|（[一二三四五六七八九十]+）\s*[\u4e00-\u9fff]|[一二三四五六七八九十]+\s*[、,，.．]\s*[\u4e00-\u9fff]|(?<![\d-])[0-9]{1,2}\.\s*[\u4e00-\u9fff]|第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节])'
)
_VISUAL_CAPTION_HEADING_SPLIT_RE = re.compile(
    r'([①②③④⑤⑥⑦⑧⑨⑩]\s*[\u4e00-\u9fff]|（[一二三四五六七八九十]+）\s*[\u4e00-\u9fff]|[一二三四五六七八九十]+\s*[、,，.．]\s*[\u4e00-\u9fff]|第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节])'
)


def _parse_cn_or_digit_int(text: str) -> Optional[int]:
    raw = (text or '').strip()
    if not raw:
        return None
    if re.fullmatch(r'\d+', raw):
        try:
            return int(raw)
        except Exception:
            return None
    total = 0
    num = 0
    unit_map = {'十': 10, '百': 100, '千': 1000}
    seen = False
    for ch in raw:
        if ch in _CN_NUM_MAP:
            num = _CN_NUM_MAP[ch]
            seen = True
            continue
        if ch in unit_map:
            seen = True
            unit = unit_map[ch]
            if num == 0:
                num = 1
            total += num * unit
            num = 0
            continue
        return None
    return (total + num) if seen else None


def _is_short_heading_title(text: str, max_len: int = 40) -> bool:
    t = re.sub(r'\s+', '', str(text or '').strip())
    if not t or len(t) > max_len:
        return False
    if t[0] in '-—,.，;；:：':
        return False
    if any(ch in t for ch in '。；;！？!?：:'):
        return False
    return True


def _strip_pdf_page_prefix(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    compact = re.sub(r'\s+', '', raw)
    if re.match(r'^\d{1,4}第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节]', compact):
        return re.sub(r'^\s*\d{1,4}\s*', '', raw)
    if re.match(r'^\d{1,4}附录', compact):
        return re.sub(r'^\s*\d{1,4}\s*', '', raw)
    return raw


def _split_inline_pdf_heading_segments(text: str) -> List[str]:
    normalized = re.sub(r'\s+', ' ', str(text or '').strip())
    if not normalized:
        return []

    matches = list(_PDF_INLINE_HEADING_START_RE.finditer(normalized))
    if len(matches) < 2:
        return [normalized]

    segments: List[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        segment = normalized[start:end].strip(' |/，,；;')
        if segment:
            segments.append(segment)
    return segments or [normalized]


def _looks_like_pdf_table_of_contents(text: str, lines: List[str]) -> bool:
    raw = str(text or '').strip()
    if not raw:
        return False
    compact = re.sub(r'\s+', '', raw)
    if '目录' in compact or '目次' in compact:
        return True

    dot_leaders = len(_PDF_TOC_DOT_LEADER_RE.findall(raw))
    heading_markers = len(_PDF_HEADING_MARKER_RE.findall(raw))
    page_number_tokens = len(re.findall(r'(?<!\d)\d{1,4}(?!\d)', raw))
    sentence_punct = len(re.findall(r'[。；;！？!?]', raw))
    dotted_lines = sum(1 for line in lines if _PDF_TOC_DOT_LEADER_RE.search(str(line or '')))
    trailing_page_lines = sum(
        1
        for line in lines
        if re.search(r'[.．…]{4,}\s*\d{1,4}\s*$', str(line or '').strip())
    )
    if heading_markers >= 4 and page_number_tokens >= 4 and sentence_punct == 0:
        return True
    return dot_leaders >= 2 and heading_markers >= 3 and (dotted_lines >= 2 or trailing_page_lines >= 2)


def _looks_like_pdf_chapter_opener(lines: List[str]) -> bool:
    meaningful_lines = [str(line or '').strip() for line in lines if str(line or '').strip()]
    if not meaningful_lines or len(meaningful_lines) > 4:
        return False

    heading_like_count = 0
    for line in meaningful_lines:
        stripped = _strip_pdf_page_prefix(line)
        compact = re.sub(r'\s+', '', stripped)
        if re.fullmatch(r'\d{1,4}', compact):
            continue
        if _detect_pdf_heading(compact) is not None or _detect_pdf_heading(stripped) is not None:
            heading_like_count += 1
            continue
        return False

    compact_text = re.sub(r'\s+', '', ''.join(meaningful_lines))
    return heading_like_count >= 1 and len(compact_text) <= 80


def _entry_text_lines_for_style_detection(entry: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    seen: Set[str] = set()

    preserved_lines = entry.get('_styleDetectionLines') if isinstance(entry, dict) else None
    if isinstance(preserved_lines, list):
        for raw_line in preserved_lines:
            line = str(raw_line or '').strip()
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)
        if lines:
            return lines

    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get('type') or '').strip().lower()
        if block_type in {'figure', 'table', 'image', 'equation'}:
            continue
        text = _extract_pdf_block_text(block)
        if not text or is_pdf_page_marker_text(text):
            continue
        for raw_line in str(text).splitlines():
            line = str(raw_line or '').strip()
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)

    if lines:
        return lines

    fallback_texts = [
        str(entry.get('jobTitle') or '').strip(),
        str(entry.get('contentMarkdown') or '').strip(),
    ]
    for raw_text in fallback_texts:
        for raw_line in raw_text.splitlines():
            line = str(raw_line or '').strip()
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)
    return lines


def _visual_label_matches_type(label: str, visual_type: str) -> bool:
    raw = str(label or '').strip()
    vt = str(visual_type or '').strip().lower()
    if vt == 'table':
        return raw.startswith(('表', '附表'))
    return raw.startswith(('图', '附图'))


def _extract_visual_caption_candidate(text: str, visual_type: str) -> str:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw:
        return ''

    hits = [hit for hit in _entry_figure_label_hits(raw) if _visual_label_matches_type(hit, visual_type)]
    if not hits:
        return ''

    label = hits[0]
    label_index = raw.find(label)
    if label_index < 0 or label_index > 4:
        return ''

    tail = raw[label_index + len(label):].strip()
    if not tail:
        return label

    split_index = _find_visual_caption_split_index(tail)

    if split_index is not None:
        title = tail[:split_index].strip(' ，,；;：:')
        if title:
            return f'{label} {title}'.strip()
        return label

    if len(re.sub(r'\s+', '', raw)) <= 36 and not re.search(r'[。；;！？!?]', raw):
        return raw
    return label


def _extract_visual_caption_candidates_from_text(text: str, visual_type: str) -> Dict[str, str]:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw:
        return {}

    hits = [hit for hit in _entry_figure_label_hits(raw) if _visual_label_matches_type(hit, visual_type)]
    if not hits:
        return {}

    candidates: Dict[str, str] = {}
    for index, label in enumerate(hits):
        label_index = raw.find(label)
        if label_index < 0:
            continue
        next_index = len(raw)
        if index + 1 < len(hits):
            next_label = hits[index + 1]
            found_next = raw.find(next_label, label_index + len(label))
            if found_next > label_index:
                next_index = found_next
        segment = raw[label_index:next_index].strip()
        candidate = _extract_visual_caption_candidate(segment, visual_type)
        if not candidate:
            continue
        primary = _extract_primary_visual_label(candidate, visual_type)
        if primary:
            candidates[primary] = candidate
    return candidates


_FIGURE_CALLOUT_CAPTION_SUFFIXES = (
    '放大图',
    '结构图',
    '实物图',
    '示意图',
    '接线图',
    '原理图',
    '剖面图',
)


def _extract_figure_callout_caption_candidate(text: str) -> str:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw or len(raw) > 40:
        return ''
    if _entry_figure_label_hits(raw):
        return _extract_visual_caption_candidate(raw, 'figure')

    compact = re.sub(r'\s+', '', raw)
    if not compact:
        return ''
    if any(compact.endswith(suffix) for suffix in _FIGURE_CALLOUT_CAPTION_SUFFIXES):
        return raw
    return ''


def _extract_embedded_visual_caption_from_image_bytes(
    image_bytes: bytes,
    visual_type: str,
    *,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[set] = None,
) -> str:
    if not image_bytes:
        return ''

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return ''

    width, height = img.size
    if width <= 0 or height <= 0:
        return ''

    for top_ratio in (0.68, 0.58):
        top = int(max(0, min(height - 1, round(height * top_ratio))))
        if top >= height - 8:
            continue
        try:
            band = img.crop((0, top, width, height))
            if band.width <= 4 or band.height <= 4:
                continue
            if band.width < 960:
                band = band.resize((band.width * 2, band.height * 2), Image.LANCZOS)
            buf = io.BytesIO()
            band.save(buf, format='PNG', optimize=True)
            ocr_text = _ocr_image_bytes(
                buf.getvalue(),
                ocr_engine=ocr_engine,
                repeated_watermark_candidates=repeated_watermark_candidates,
            ) or ''
        except Exception:
            continue
        candidate = _extract_visual_caption_candidate(ocr_text, visual_type)
        if candidate:
            return candidate
    return ''


def _split_mixed_visual_caption_text(text: str, visual_type: str) -> Tuple[str, str]:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw:
        return '', ''

    inline_ref = _FIGURE_INLINE_REF_BODY_SPLIT_RE.search(raw)
    if inline_ref is not None:
        opener_match = _FIGURE_SECTION_OPENER_INLINE_REF_RE.search(raw)
        if opener_match is not None:
            caption_text = raw[: opener_match.start()].strip(' ，,；;：:')
            body_text = raw[opener_match.start() :].strip()
        else:
            caption_text = raw[: inline_ref.start()].strip(' ，,；;：:')
            body_text = raw[inline_ref.end() :].strip()
        if caption_text and body_text and len(re.sub(r'\s+', '', body_text)) >= 12:
            return caption_text, body_text

    hits = [hit for hit in _entry_figure_label_hits(raw) if _visual_label_matches_type(hit, visual_type)]
    if not hits:
        return raw, ''

    label = hits[0]
    label_index = raw.find(label)
    if label_index < 0 or label_index > 4:
        return raw, ''

    tail = raw[label_index + len(label):].strip()
    if not tail:
        return raw, ''

    split_index = _find_visual_caption_split_index(tail)

    if split_index is None:
        compact_tail = re.sub(r'\s+', '', tail)
        if len(compact_tail) >= 48 and re.search(
            r'(如图所示|一般应用|其特点|特点是|传统的|需要强调|当电力|系指)',
            compact_tail,
        ):
            return raw, ''
        return raw, ''

    title = tail[:split_index].strip(' ，,；;：:')
    body_text = tail[split_index:].strip()
    if not title:
        return raw, ''
    if len(re.sub(r'\s+', '', body_text)) < 12 and not _looks_like_visual_follow_on_heading(body_text):
        return raw, ''
    return f'{label} {title}'.strip(), body_text


def _recover_searchable_body_from_visual_only_block(
    text: str,
    *,
    visual_type: str = 'figure',
) -> str:
    """Pull prose out of caption/callout blocks so it can return to entry search text."""
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw:
        return ''
    _caption_text, body_text = _split_mixed_visual_caption_text(raw, visual_type)
    if body_text:
        return body_text.strip()
    compact = re.sub(r'\s+', '', raw)
    if len(compact) >= 48 and re.search(
        r'(如图所示|一般应用|其特点|特点是|传统的|需要强调|当电力|系指)',
        compact,
    ):
        return raw
    return ''


def _find_visual_caption_split_index(tail: str) -> Optional[int]:
    split_index: Optional[int] = None
    for cue in _VISUAL_CAPTION_BODY_CUES:
        cue_index = tail.find(cue)
        if cue_index <= 0:
            continue
        if split_index is None or cue_index < split_index:
            split_index = cue_index

    marker_match = _VISUAL_CAPTION_SPLIT_RE.search(tail)
    if marker_match is not None and marker_match.start() > 0:
        marker_index = marker_match.start()
        if split_index is None or marker_index < split_index:
            split_index = marker_index
    return split_index


def _looks_like_visual_follow_on_heading(text: str) -> bool:
    stripped = re.sub(r'\s+', ' ', str(text or '').strip())
    if not stripped:
        return False
    if _detect_pdf_heading(stripped) is not None:
        return True
    return _VISUAL_CAPTION_HEADING_SPLIT_RE.match(stripped) is not None


def _find_nearby_visual_caption_candidate_with_meta(
    blocks: List[Dict[str, Any]],
    index: int,
    visual_type: str,
    allowed_roles: set,
) -> Optional[tuple]:
    block_type = 'table' if visual_type == 'table' else 'figure'

    for neighbor_index in range(index - 1, -1, -1):
        neighbor = blocks[neighbor_index]
        if not isinstance(neighbor, dict):
            continue
        if str(neighbor.get('type') or '').strip().lower() == block_type:
            break
        role = str(neighbor.get('semanticRole') or '').strip().lower()
        if role == 'artifact':
            continue
        if role not in allowed_roles:
            continue
        candidate = _extract_visual_caption_candidate(_extract_pdf_block_text(neighbor), visual_type)
        if candidate:
            return candidate, neighbor.get('id')

    for neighbor_index in range(index + 1, len(blocks)):
        neighbor = blocks[neighbor_index]
        if not isinstance(neighbor, dict):
            continue
        if str(neighbor.get('type') or '').strip().lower() == block_type:
            break
        role = str(neighbor.get('semanticRole') or '').strip().lower()
        if role == 'artifact':
            continue
        if role not in allowed_roles:
            continue
        candidate = _extract_visual_caption_candidate(_extract_pdf_block_text(neighbor), visual_type)
        if candidate:
            return candidate, neighbor.get('id')
    return None


def _find_nearby_visual_caption_candidate(
    blocks: List[Dict[str, Any]],
    index: int,
    visual_type: str,
    allowed_roles: Set[str],
) -> str:
    res = _find_nearby_visual_caption_candidate_with_meta(blocks, index, visual_type, allowed_roles)
    return res[0] if res else ''


def _normalize_pdf_entry_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Filter standalone artifacts between main figure and its potential caption
    def _is_noise_artifact(block: Dict) -> bool:
        role = str(block.get('semanticRole') or '').strip().lower()
        if role == 'artifact':
            text = _extract_pdf_block_text(block)
            if is_pdf_page_marker_text(text):
                return True
        return False

    prepared = _collapse_soft_wrapped_pdf_blocks(
        _split_inline_decimal_clause_blocks(
            _merge_standalone_list_marker_blocks(
                _filter_entry_noise_blocks(blocks if isinstance(blocks, list) else [])
            )
        )
    )

    # Pre-filter artifacts that might break figure-caption proximity
    prepared = [b for b in prepared if not _is_noise_artifact(b)]

    normalized_blocks: List[Dict[str, Any]] = []

    for block in prepared:
        if not isinstance(block, dict):
            continue
        block = _normalize_block_inline_ocr_list_markers(block)
        text = _extract_pdf_block_text(block)
        semantic_role = str(block.get('semanticRole') or '').strip().lower()
        if semantic_role in {'caption', 'body', 'heading'} and text:
            visual_type = 'figure'
            if any(hit.startswith(('表', '附表')) for hit in _entry_figure_label_hits(text)):
                visual_type = 'table'
            if semantic_role != 'caption' and not _entry_figure_label_hits(text):
                normalized_blocks.append(block)
                continue
            caption_text, body_text = _split_mixed_visual_caption_text(text, visual_type)
            if body_text:
                caption_block = copy.deepcopy(block)
                if semantic_role != 'caption':
                    caption_block['semanticRole'] = 'caption'
                if 'code' in caption_block:
                    caption_block['code'] = caption_text
                elif 'text' in caption_block:
                    caption_block['text'] = caption_text
                else:
                    caption_block['contentMarkdown'] = caption_text
                normalized_blocks.append(caption_block)

                body_block = copy.deepcopy(block)
                body_block['id'] = f"{body_block.get('id') or 'caption'}__tail"
                body_block['semanticRole'] = 'heading' if _looks_like_visual_follow_on_heading(body_text) else 'body'
                reading_order = _safe_int(body_block.get('readingOrder'))
                if reading_order is not None:
                    body_block['readingOrder'] = reading_order + 1
                if 'code' in body_block:
                    body_block['code'] = body_text
                elif 'text' in body_block:
                    body_block['text'] = body_text
                else:
                    body_block['contentMarkdown'] = body_text
                normalized_blocks.append(body_block)
                continue
        normalized_blocks.append(block)

    for block in normalized_blocks:
        if not isinstance(block, dict):
            continue
        role = str(block.get('semanticRole') or '').strip().lower()
        if role == 'body':
            block_text = _extract_pdf_block_text(block)
            if _is_diagram_switch_label_text(block_text):
                block['semanticRole'] = 'figure_callout'

    for index, block in enumerate(normalized_blocks):
        if not isinstance(block, dict):
            continue
        block_type = str(block.get('type') or '').strip().lower()
        if block_type not in {'figure', 'table'}:
            continue
        visual_type = 'table' if block_type == 'table' else 'figure'
        current_caption = str(block.get('caption') or '').strip()
        if current_caption and not re.fullmatch(rf'{visual_type.title()}\s+\d+', current_caption, re.IGNORECASE):
            continue
        candidate = ''
        target_figure_id = None
        for allowed_roles in ({'caption'}, {'body', 'heading'}):
            # Pass back the index/ID of the found caption candidate
            found_data = _find_nearby_visual_caption_candidate_with_meta(normalized_blocks, index, visual_type, allowed_roles)
            if found_data:
                candidate, found_block_id = found_data
                if candidate:
                    block['caption'] = candidate
                    # Link caption back to figure
                    for b in normalized_blocks:
                        if b.get('id') == found_block_id:
                            b['parentFigureId'] = block.get('id')
                            b['structureSource'] = 'caption'
                    break

    generic_figure_re = re.compile(r'Figure\s+\d+', re.IGNORECASE)
    pre_generic_callouts: List[str] = []
    passed_main_figure = False
    last_named_label = ''
    last_generic_cluster_caption = ''

    for block in normalized_blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get('type') or '').strip().lower()
        role = str(block.get('semanticRole') or '').strip().lower()

        if role == 'figure_callout':
            callout = _extract_figure_callout_caption_candidate(_extract_pdf_block_text(block))
            if callout and not passed_main_figure:
                pre_generic_callouts.append(callout)
            continue

        if block_type != 'figure':
            continue

        current_caption = str(block.get('caption') or '').strip()
        if current_caption and not generic_figure_re.fullmatch(current_caption):
            passed_main_figure = True
            last_named_label = _extract_primary_visual_label(current_caption, 'figure')
            last_generic_cluster_caption = ''
            continue

        candidate = ''
        if pre_generic_callouts:
            callout_text = pre_generic_callouts.pop(0)
            candidate = f'{last_named_label} {callout_text}'.strip() if last_named_label else callout_text
        elif last_generic_cluster_caption:
            candidate = last_generic_cluster_caption
        if candidate:
            block['caption'] = candidate
            last_generic_cluster_caption = candidate

    _promote_body_blocks_near_figures(normalized_blocks)
    return normalized_blocks


def _bbox_overlaps_figure_region(
    block_bbox: Dict[str, Any],
    figure_boxes: List[Dict[str, Any]],
    *,
    margin: int = 96,
) -> bool:
    if not isinstance(block_bbox, dict) or not figure_boxes:
        return False
    try:
        top = int(block_bbox.get('top', 0))
        bottom = int(block_bbox.get('bottom', top))
        left = int(block_bbox.get('left', 0))
        right = int(block_bbox.get('right', left))
    except Exception:
        return False
    for figure_bbox in figure_boxes:
        if not isinstance(figure_bbox, dict):
            continue
        try:
            ft = int(figure_bbox.get('top', 0)) - margin
            fb = int(figure_bbox.get('bottom', 0)) + margin
            fl = int(figure_bbox.get('left', 0)) - margin
            fr = int(figure_bbox.get('right', 0)) + margin
        except Exception:
            continue
        if right >= fl and left <= fr and bottom >= ft and top <= fb:
            return True
    return False


def _promote_body_blocks_near_figures(normalized_blocks: List[Dict[str, Any]]) -> None:
    figure_boxes = [
        block.get('bbox')
        for block in normalized_blocks
        if isinstance(block, dict)
        and str(block.get('type') or '').strip().lower() == 'figure'
        and isinstance(block.get('bbox'), dict)
    ]
    if not figure_boxes:
        return

    for block in normalized_blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get('semanticRole') or '').strip().lower() != 'body':
            continue
        block_text = _extract_pdf_block_text(block)
        if not block_text:
            continue
        compact = re.sub(r'\s+', '', block_text)
        near_figure = _bbox_overlaps_figure_region(
            block.get('bbox') if isinstance(block.get('bbox'), dict) else {},
            figure_boxes,
        )
        if not near_figure:
            continue
        if _is_diagram_switch_label_text(block_text):
            block['semanticRole'] = 'figure_callout'
            continue
        if (
            len(compact) <= 40
            and '箱变' in compact
            and ('变配' in compact or '电所' in compact)
        ):
            block['semanticRole'] = 'figure_callout'


def _is_standalone_part_label_text(text: str) -> bool:
    compact = re.sub(r'\s+', '', str(text or '').strip())
    return bool(re.fullmatch(r'第[一二三四五六七八九十百0-9]+篇', compact))


def _filter_standalone_part_label_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(blocks, list) or len(blocks) != 1:
        return blocks if isinstance(blocks, list) else []
    only_block = blocks[0]
    if not isinstance(only_block, dict):
        return blocks
    if _is_standalone_part_label_text(_extract_pdf_block_text(only_block)):
        return []
    return blocks


def _detect_two_column_gutter_x(page_blocks: List[Dict[str, Any]]) -> Optional[float]:
    """Detect a vertical gutter x-coordinate splitting a page's blocks into left/right columns.

    Returns the gutter x (midpoint of the widest horizontal gap between two block clusters)
    when the page shows a confident two-column layout (both clusters have enough blocks and
    their vertical spans overlap, i.e. they sit side-by-side rather than being header/footer
    zones stacked vertically). Returns None otherwise so single-column pages are unaffected.
    """
    intervals: List[Tuple[float, float, float, float]] = []
    for block in page_blocks:
        if not isinstance(block, dict):
            continue
        bbox = block.get('bbox') if isinstance(block.get('bbox'), dict) else None
        if not isinstance(bbox, dict):
            continue
        try:
            left = float(bbox.get('left'))
            width = float(bbox.get('width'))
            top = float(bbox.get('top'))
            height = float(bbox.get('height'))
        except Exception:
            continue
        if width <= 0 or height <= 0:
            continue
        intervals.append((left, left + width, top, top + height))

    if len(intervals) < 6:
        return None

    min_left = min(i[0] for i in intervals)
    max_right = max(i[1] for i in intervals)
    page_width = max_right - min_left
    if page_width < 200:
        return None

    # Ignore blocks that already span most of the page width (headings, full-width paragraphs).
    narrow = [i for i in intervals if (i[1] - i[0]) <= page_width * 0.62]
    if len(narrow) < 6:
        return None

    spans = sorted(((i[0], i[1]) for i in narrow), key=lambda s: s[0])
    merged: List[List[float]] = []
    for s in spans:
        if merged and s[0] <= merged[-1][1] + page_width * 0.01:
            merged[-1][1] = max(merged[-1][1], s[1])
        else:
            merged.append([s[0], s[1]])

    if len(merged) < 2:
        return None

    gaps = [(b[0] - a[1], a[1], b[0]) for a, b in zip(merged, merged[1:])]
    gaps.sort(key=lambda g: -g[0])
    best_gap_width, left_edge, right_edge = gaps[0]
    if best_gap_width < max(12.0, page_width * 0.03):
        return None

    left_cluster = [i for i in narrow if i[1] <= left_edge + 1.0]
    right_cluster = [i for i in narrow if i[0] >= right_edge - 1.0]
    if len(left_cluster) < 3 or len(right_cluster) < 3:
        return None

    # Confirm the two clusters are truly side-by-side (their vertical ranges overlap).
    left_top = min(i[2] for i in left_cluster)
    left_bottom = max(i[3] for i in left_cluster)
    right_top = min(i[2] for i in right_cluster)
    right_bottom = max(i[3] for i in right_cluster)
    overlap = min(left_bottom, right_bottom) - max(left_top, right_top)
    if overlap <= 0:
        return None

    return (left_edge + right_edge) / 2.0


def _assign_column_bands_for_document_order(blocks: List[Dict[str, Any]]) -> None:
    """Tag each block with a transient `_docOrderColumn` (0/1) based on per-page column detection."""
    pages: Dict[Any, List[Dict[str, Any]]] = {}
    for block in blocks:
        if not isinstance(block, dict):
            continue
        pages.setdefault(block.get('pageNumber'), []).append(block)

    for page_blocks in pages.values():
        gutter_x = _detect_two_column_gutter_x(page_blocks)
        if gutter_x is None:
            continue
        for block in page_blocks:
            bbox = block.get('bbox') if isinstance(block.get('bbox'), dict) else None
            if not isinstance(bbox, dict):
                continue
            try:
                left = float(bbox.get('left'))
                width = float(bbox.get('width'))
            except Exception:
                continue
            center_x = left + width / 2.0
            block['_docOrderColumn'] = 0 if center_x < gutter_x else 1


def _block_document_order_tier(block: Dict[str, Any]) -> int:
    text = re.sub(r'\s+', ' ', _extract_pdf_block_text(block)).strip()
    compact = re.sub(r'\s+', '', text)
    role = str(block.get('semanticRole') or '').strip().lower()
    block_type = str(block.get('type') or '').strip().lower()

    if role == 'heading' or _detect_pdf_heading(text) is not None:
        if re.match(r'^第[一二三四五六七八九十百0-9]+[章节]', compact):
            return 0
        detected = _detect_pdf_heading(text)
        if detected is not None and detected[0] in {'part', 'chapter', 'appendix', 'section'}:
            return 0
        if re.match(r'^[一二三四五六七八九十]+、', compact):
            return 1
        return 1
    if role in {'figure_callout', 'legend', 'annotation', 'ocr_annotation'}:
        return 3
    if role in {'caption', 'table_note'} or block_type in {'figure', 'table', 'image', 'equation'}:
        return 4
    return 2


def _block_sort_tuple_for_document_order(block: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    bbox = block.get('bbox') if isinstance(block.get('bbox'), dict) else {}
    return (
        _block_document_order_tier(block),
        _safe_int(block.get('_docOrderColumn')) or 0,
        _safe_int(block.get('readingOrder')) or 10**9,
        int(bbox.get('top', 10**9)),
        int(bbox.get('left', 10**9)),
    )


def _canonicalize_inline_headings_in_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(blocks, list):
        return []

    out: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = _extract_pdf_block_text(block)
        segments = _split_inline_pdf_heading_segments(text)
        if len(segments) < 2:
            out.append(block)
            continue

        base_order = _safe_int(block.get('readingOrder')) or 0
        for segment_index, segment in enumerate(segments):
            new_block = copy.deepcopy(block)
            if 'code' in new_block:
                new_block['code'] = segment
            elif 'text' in new_block:
                new_block['text'] = segment
            else:
                new_block['contentMarkdown'] = segment
            segment_compact = re.sub(r'\s+', '', segment)
            if re.match(r'^第[一二三四五六七八九十百0-9]+[章节]', segment_compact) or re.match(
                r'^[一二三四五六七八九十]+、',
                segment_compact,
            ):
                new_block['semanticRole'] = 'heading'
            new_block['readingOrder'] = base_order + segment_index
            block_id = str(block.get('id') or 'block').strip() or 'block'
            new_block['id'] = f'{block_id}__heading_{segment_index}'
            out.append(new_block)
    return out


def _sort_entry_blocks_for_document_order(
    blocks: List[Dict[str, Any]],
    entry: Dict,
) -> List[Dict[str, Any]]:
    del entry  # reserved for blue-heading hints in future
    if not isinstance(blocks, list):
        return []
    _assign_column_bands_for_document_order(blocks)
    ordered = sorted(blocks, key=_block_sort_tuple_for_document_order)
    for index, block in enumerate(ordered, start=1):
        if isinstance(block, dict):
            block['readingOrder'] = int(index)
            block.pop('_docOrderColumn', None)
    return ordered


def _extract_primary_visual_label(text: str, visual_type: str) -> str:
    hits = [hit for hit in _entry_figure_label_hits(text) if _visual_label_matches_type(hit, visual_type)]
    return hits[0] if hits else ''


def _collect_leading_entry_visual_caption_candidates(
    entry: Dict[str, Any],
    visual_type: str,
    *,
    max_blocks: int = 6,
) -> Dict[str, str]:
    candidates: Dict[str, str] = {}
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    block_type = 'table' if visual_type == 'table' else 'figure'
    generic_caption_re = re.compile(rf'{block_type.title()}\s+\d+', re.IGNORECASE)

    for block in blocks[:max_blocks]:
        if not isinstance(block, dict):
            continue
        candidate = ''
        role = str(block.get('semanticRole') or '').strip().lower()
        current_block_type = str(block.get('type') or '').strip().lower()
        if role == 'caption':
            for label, split_candidate in _extract_visual_caption_candidates_from_text(
                _extract_pdf_block_text(block),
                visual_type,
            ).items():
                if label and label not in candidates:
                    candidates[label] = split_candidate
            continue
        elif current_block_type == block_type:
            current_caption = str(block.get('caption') or '').strip()
            if current_caption and not generic_caption_re.fullmatch(current_caption):
                candidate = current_caption
        if not candidate:
            continue
        label = _extract_primary_visual_label(candidate, visual_type)
        if label and label not in candidates:
            candidates[label] = candidate
    return candidates


def _find_last_page_boundary_relabelable_visual_index(
    entry: Dict[str, Any],
    visual_type: str,
    referenced_labels: List[str],
) -> Optional[int]:
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    block_type = 'table' if visual_type == 'table' else 'figure'
    last_visual_index: Optional[int] = None
    current_caption_labels: Dict[str, int] = {}

    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if str(block.get('type') or '').strip().lower() != block_type:
            continue
        last_visual_index = index
        label = _extract_primary_visual_label(str(block.get('caption') or '').strip(), visual_type)
        if label:
            current_caption_labels[label] = current_caption_labels.get(label, 0) + 1

    if last_visual_index is None:
        return None

    last_block = blocks[last_visual_index]
    current_caption = str(last_block.get('caption') or '').strip()
    if re.fullmatch(rf'{block_type.title()}\s+\d+', current_caption, re.IGNORECASE):
        return last_visual_index

    current_label = _extract_primary_visual_label(current_caption, visual_type)
    if not current_label:
        return None

    referenced_counts: Dict[str, int] = {}
    for label in referenced_labels:
        referenced_counts[label] = referenced_counts.get(label, 0) + 1
    if current_caption_labels.get(current_label, 0) > referenced_counts.get(current_label, 0):
        return last_visual_index

    compact_caption = re.sub(r'\s+', '', current_caption)
    compact_label = re.sub(r'\s+', '', current_label)
    if compact_caption == compact_label and current_label in referenced_labels:
        return last_visual_index
    return None


def _repair_page_boundary_visual_captions(entries: List[Dict[str, Any]]) -> None:
    if len(entries) < 2:
        return

    for index in range(len(entries) - 1):
        current = entries[index]
        following = entries[index + 1]
        if not isinstance(current, dict) or not isinstance(following, dict):
            continue
        try:
            current_page = int(current.get('pageNumber') or 0)
            following_page = int(following.get('pageNumber') or 0)
        except Exception:
            continue
        if following_page - current_page != 1:
            continue

        for visual_type in ('figure', 'table'):
            current_text = str(current.get('contentNormalized') or current.get('contentMarkdown') or '').strip()
            referenced_labels = [
                hit
                for hit in _entry_figure_label_hits(current_text)
                if _visual_label_matches_type(hit, visual_type)
            ]
            if not referenced_labels:
                continue

            current_caption_labels = {
                _extract_primary_visual_label(str(block.get('caption') or '').strip(), visual_type)
                for block in (current.get('blocks') if isinstance(current.get('blocks'), list) else [])
                if isinstance(block, dict) and str(block.get('type') or '').strip().lower() == ('table' if visual_type == 'table' else 'figure')
            }
            missing_labels = [label for label in referenced_labels if label and label not in current_caption_labels]
            if not missing_labels:
                continue

            next_candidates = _collect_leading_entry_visual_caption_candidates(following, visual_type)
            if not next_candidates:
                continue

            borrow_label = ''
            for label in missing_labels:
                if label in next_candidates:
                    borrow_label = label
                    break
            if not borrow_label:
                continue

            target_index = _find_last_page_boundary_relabelable_visual_index(current, visual_type, referenced_labels)
            if target_index is None:
                continue

            blocks = current.get('blocks') if isinstance(current.get('blocks'), list) else None
            if isinstance(blocks, list):
                blocks[target_index]['caption'] = next_candidates[borrow_label]


def _extract_appendix_form_title_candidate(block: Dict[str, Any]) -> str:
    if not isinstance(block, dict):
        return ''
    role = str(block.get('semanticRole') or '').strip().lower()
    if role not in {'figure_callout', 'heading', 'caption'}:
        return ''
    compact = re.sub(r'\s+', '', _extract_pdf_block_text(block))
    if not compact or compact.startswith('附录'):
        return ''
    if len(compact) > 20 or re.search(r'[。；;！？!?：:]', compact):
        return ''
    if not compact.endswith(('工作票', '操作票', '作业票', '记录薄', '记录簿')):
        return ''
    return compact


def _find_appendix_form_title(entry: Dict[str, Any], *, max_blocks: int = 6) -> str:
    unit_name = str(entry.get('unitName') or '').strip()
    if not unit_name.startswith('附录'):
        return ''
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    for block in blocks[:max_blocks]:
        candidate = _extract_appendix_form_title_candidate(block)
        if candidate:
            return candidate
    return ''


def _repair_appendix_form_visual_captions(entries: List[Dict[str, Any]]) -> None:
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        title = _find_appendix_form_title(entry)
        if not title:
            continue
        blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get('type') or '').strip().lower()
            if block_type not in {'figure', 'table'}:
                continue
            visual_type = 'table' if block_type == 'table' else 'figure'
            current_caption = str(block.get('caption') or '').strip()
            if current_caption and not re.fullmatch(rf'{visual_type.title()}\s+\d+', current_caption, re.IGNORECASE):
                continue
            block['caption'] = title


def _detect_pdf_heading(line: str) -> Optional[tuple]:
    t = _strip_pdf_page_prefix(line)
    if not t:
        return None
    m = _PDF_PART_RE.match(t)
    if m:
        n = _parse_cn_or_digit_int(m.group(1))
        if n is not None and n > 0:
            return ('part', str(n), t)
    m = _PDF_CHAPTER_RE.match(t)
    if m:
        n = _parse_cn_or_digit_int(m.group(1))
        if n is not None and n > 0:
            return ('chapter', str(n), t)
    m = _PDF_SECTION_RE.match(t)
    if m:
        n = _parse_cn_or_digit_int(m.group(1))
        title = str(m.group(2) or '').strip()
        if n is not None and n > 0 and _is_short_heading_title(title or t):
            return ('section', str(n), t)
    m = _PDF_SUBSECTION_RE.match(t)
    if m:
        n = _parse_cn_or_digit_int(m.group(1))
        title = str(m.group(2) or '').strip()
        if n is not None and n > 0 and _is_short_heading_title(title):
            return ('subsection', str(n), t)
    m = _PDF_APPENDIX_RE.match(t)
    if m:
        title = str(m.group(2) or '').strip()
        if _is_short_heading_title(title or t, max_len=24):
            return ('appendix', str(m.group(1) or '').strip(), t)
    m = _PDF_TOP_LEVEL_RE.match(t)
    if m:
        title = str(m.group(2) or '').strip()
        compact_title = re.sub(r'\s+', '', title)
        if (
            _is_short_heading_title(title, max_len=18)
            and '图' not in title
            and '表' not in title
            and not re.search(r'[A-Za-z0-9|]', compact_title)
        ):
            return ('chapter', str(m.group(1) or '').strip(), t)
    return None


def _extract_pdf_block_text(block: Dict) -> str:
    return str(block.get('code') or block.get('text') or block.get('contentMarkdown') or '').strip()


_ENTRY_SEARCHABLE_BLOCK_TYPES = {'code', 'text', 'paragraph'}
_ENTRY_SEARCHABLE_ROLES = {'heading', 'body'}
_ENTRY_VISUAL_ONLY_ROLES = {
    'table',
    'table_note',
    'caption',
    'artifact',
    'figure',
    'figure_callout',
    'annotation',
    'ocr_annotation',
    'legend',
}
_PRELIMINARY_PAGE_MARKERS = (
    '图书在版编目',
    'cip',
    '中国版本图书馆',
    '版权所有',
    '出版发行',
    '责任编辑',
    '封面设计',
    'isbn',
    '中国铁道出版社',
    '目录',
    '目 次',
)
_TEXTUAL_BLOCK_TYPES = {'code', 'text', 'paragraph'}


def _dedupe_preserve_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for raw in lines:
        line = str(raw or '').strip()
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out




def _line_starts_new_pdf_paragraph(text: str) -> bool:
    raw = str(text or '').strip()
    if not raw:
        return False
    # Standards commonly use multi-level decimal clause numbers like "4.2.13".
    if re.match(r'^\d{1,2}\.\d{1,2}(?:\.\d{1,3})?(?!\d)', raw):
        return True
    if re.match(r'^(?:\d{1,3}\s*[\.．、])', raw):
        return True
    if re.match(r'^(?:0?\d{1,2})(?!\s*[\.．、])\s+', raw):
        if re.match(r'^(?:0?\d{1,2})\s*(?:号|页|章|节|条|款)\b', raw):
            return False
        return True
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]', raw):
        return True
    if re.match(r'^(?:答|问|解析|注)\s*[：:]', raw):
        return True
    return False


def _line_has_unclosed_pdf_opening(text: str) -> bool:
    raw = str(text or '')
    pairs = [('（', '）'), ('(', ')'), ('《', '》'), ('“', '”'), ('〔', '〕'), ('[', ']')]
    for opener, closer in pairs:
        try:
            if raw.count(opener) > raw.count(closer):
                return True
        except Exception:
            continue
    return False


def _should_merge_soft_wrapped_pdf_lines(previous_line: str, current_line: str) -> bool:
    prev = str(previous_line or '').strip()
    curr = str(current_line or '').strip()
    if not prev or not curr:
        return False
    if _line_starts_new_pdf_paragraph(curr):
        return False
    if re.search(r'[。！？!?]$', prev):
        return False
    if _line_has_unclosed_pdf_opening(prev):
        return True
    if re.search(r'[：:，、；,;（(《“\-]$', prev):
        return True
    if re.search(r'[A-Za-z0-9\u4e00-\u9fff）)》”°%]$', prev):
        return True
    return False


def _collapse_soft_wrapped_pdf_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for raw in lines:
        line = str(raw or '').strip()
        if not line:
            continue
        if out and _should_merge_soft_wrapped_pdf_lines(out[-1], line):
            out[-1] = out[-1] + line
        else:
            out.append(line)
    return out


_INLINE_SUBSECTION_TAIL_RE = re.compile(
    r'^(?P<body>.*?)(?<=[。；;！？!?])\s*(?P<heading>[一二三四五六七八九十]+、[^\n。；;！？!?]{2,48})$',
    re.DOTALL,
)
_SEARCHABLE_LIST_ITEM_BREAK_RE = re.compile(
    r'(?<=[。；;！？!?：:）)）])\s+(?=\d{1,2}[.．、])'
)
_SEARCHABLE_NUMBERED_QUESTION_BREAK_RE = re.compile(
    r'(?<=[？?])\s+(?=\d{1,2}[.．、])'
)
_SEARCHABLE_NUMBERED_REFERENCE_BREAK_RE = re.compile(
    r'(?<=[)）])\s+(?=\d{1,2}[.．、《])'
)
_SUBSECTION_ONLY_HEADING_RE = re.compile(
    r'^[一二三四五六七八九十]+、[^\n。；;！？!?]{2,48}$'
)


def _is_orphan_page_boundary_prefix(text: str) -> bool:
    compact = re.sub(r'\s+', '', str(text or '').strip())
    return compact in {'方式', '方式。', '方式．'}


def _split_leading_inline_subsection_heading(text: str) -> Tuple[str, str]:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw:
        return '', ''
    match = re.match(
        r'^(?P<prefix>.*?[。．!?！？])\s*(?P<heading>[一二三四五六七八九十]+、[^\n。；;！？!?]{2,48})$',
        raw,
    )
    if not match:
        return raw, ''
    prefix = str(match.group('prefix') or '').strip()
    heading = str(match.group('heading') or '').strip()
    if not heading:
        return raw, ''
    return prefix, heading


def _is_diagram_switch_label_text(text: str) -> bool:
    compact = re.sub(r'\s+', '', str(text or '').strip())
    if not compact:
        return False
    if _CABLE_DIAGRAM_LEGEND_ONLY_RE.match(compact) or _FIGURE_DIAGRAM_LEGEND_ONLY_RE.match(compact):
        return True
    if '箱变' in compact and '变配' in compact and '电所' in compact:
        return True
    if len(compact) > 32 or re.search(r'[。；;！？!?：:]', compact):
        return False
    if re.fullmatch(r'(?:单端接地|双端接地)+', compact):
        return True
    # Short space-separated schematic labels without sentence punctuation.
    tokens = [token for token in re.split(r'\s+', str(text or '').strip()) if token]
    if 2 <= len(tokens) <= 6 and all(len(token) <= 12 for token in tokens):
        if not any(re.search(r'[。；;！？!?：:]', token) for token in tokens):
            if all(re.fullmatch(r'[\u4e00-\u9fffA-Za-z0-9（）()\-—]{2,12}', token) for token in tokens):
                return True
    return False


def _ordered_subsection_hints_for_entry(entry: Dict) -> List[str]:
    labels: List[Tuple[int, str]] = []
    annotation_context = entry.get('annotationContext') if isinstance(entry.get('annotationContext'), dict) else {}
    for hint in annotation_context.get('blueHeadingHints') or []:
        if not isinstance(hint, dict):
            continue
        if str(hint.get('kind') or '').strip().lower() != 'subsection':
            continue
        label = re.sub(r'\s+', ' ', str(hint.get('display') or hint.get('text') or '').strip())
        if label and re.match(r'^[一二三四五六七八九十]+、', label):
            labels.append((int(hint.get('readingOrder') or 0), label))
    labels.sort(key=lambda item: item[0])
    return [label for _, label in labels]


def _figure_opener_index_in_entry(entry: Dict, body_text: str) -> int:
    entry_page = _safe_int(entry.get('pageNumber')) or 0
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    scoped_blocks = [
        block for block in blocks
        if isinstance(block, dict)
        and (not entry_page or (_safe_int(block.get('pageNumber')) or entry_page) == entry_page)
    ]
    target_labels = [re.sub(r'\s+', '', label) for label in _entry_figure_label_hits(body_text)]
    if not target_labels:
        return 0
    target_label = target_labels[0]
    index = 0
    for block in sorted(scoped_blocks, key=lambda item: (
        _safe_int(item.get('pageNumber')) or entry_page,
        int((item.get('bbox') or {}).get('top', 10**9)),
        int((item.get('bbox') or {}).get('left', 10**9)),
        _safe_int(item.get('readingOrder')) or 10**9,
    )):
        text = _extract_pdf_block_text(block)
        if not text or not _FIGURE_SECTION_OPENER_INLINE_REF_RE.search(text):
            continue
        block_labels = [re.sub(r'\s+', '', label) for label in _entry_figure_label_hits(text)]
        if block_labels and block_labels[0] == target_label:
            return index
        index += 1
    return index


def _resolve_figure_body_subsection_label(
    entry: Dict,
    body_text: str,
    *,
    recent_heading: str = '',
    prior_segments: Optional[list] = None,
) -> str:
    raw = re.sub(r'\s+', ' ', str(body_text or '').strip())
    if not raw or not _FIGURE_SECTION_OPENER_INLINE_REF_RE.search(raw):
        return raw

    raw_compact = re.sub(r'\s+', '', raw)
    recent_compact = re.sub(r'\s+', '', str(recent_heading or ''))

    hints = _ordered_subsection_hints_for_entry(entry)
    if hints:
        hint_index = _figure_opener_index_in_entry(entry, raw)
        if hint_index < len(hints):
            label = hints[hint_index]
            label_compact = re.sub(r'\s+', '', label)
            if label_compact == recent_compact or raw_compact.startswith(label_compact):
                return raw
            for prior in reversed((prior_segments or [])[-6:]):
                if re.sub(r'\s+', '', prior) == label_compact:
                    return raw
            return f'{label}\n\n{raw}'
    return raw


def _search_segment_join_separator(previous: str, current: str) -> str:
    prev = str(previous or '').rstrip()
    nxt = str(current or '').lstrip()
    if not prev or not nxt:
        return '\n\n'
    if re.match(r'^\d+\.\s', nxt) or re.match(r'^答\s*[：:]', nxt):
        return '\n\n'
    if re.match(r'^[一二三四五六七八九十]+、', nxt):
        return '\n\n'
    prev_compact = re.sub(r'\s+', '', prev)
    if re.search(
        r'(?:[一二三四五六七八九十]+、[^\n。；;！？!?]{2,48}|'
        r'第[一二三四五六七八九十百0-9]+节[^\n。；;！？!?]{2,48})$',
        prev_compact,
    ):
        return '\n\n'
    if prev.endswith(('？', '?')) or re.match(r'^答\s*[：:]', prev):
        return '\n\n'
    if not re.search(r'[。；;！？!?]$', prev):
        nxt_compact = re.sub(r'\s+', '', nxt)
        if prev_compact.endswith('电缆') and nxt_compact.startswith('线路'):
            return ''
        if re.search(r'(?:方式|方法|特点|原则|系统|线路|设备|四种)$', prev_compact[-4:]):
            return '\n\n'
        return ''
    return '\n\n'


def _is_subsection_only_heading_text(text: str) -> bool:
    compact = re.sub(r'\s+', '', str(text or '').strip())
    return bool(compact and _SUBSECTION_ONLY_HEADING_RE.match(compact))


def _trim_trailing_subsection_spillover_segments(segments: List[str]) -> List[str]:
    """Drop a final subsection-only line that belongs on the next PDF page."""
    if not segments:
        return segments
    if _is_subsection_only_heading_text(segments[-1]):
        return segments[:-1]
    return segments


def _dedupe_ordered_search_segments(segments: List[str]) -> List[str]:
    deduped: List[str] = []
    for segment in segments:
        cleaned = str(segment or '').strip()
        if not cleaned:
            continue
        compact = re.sub(r'\s+', '', cleaned)
        if deduped:
            prev = deduped[-1]
            prev_compact = re.sub(r'\s+', '', prev)
            if compact == prev_compact:
                continue
            if _SUBSECTION_ONLY_HEADING_RE.match(compact) and _SUBSECTION_ONLY_HEADING_RE.match(prev_compact):
                continue
            if compact.startswith(prev_compact) and len(compact) > len(prev_compact):
                deduped[-1] = cleaned
                continue
            if prev_compact.startswith(compact):
                continue
        deduped.append(cleaned)
    if len(deduped) >= 2:
        tail = deduped[-1]
        tail_compact = re.sub(r'\s+', '', tail)
        if _SUBSECTION_ONLY_HEADING_RE.match(tail_compact):
            prev_compact = re.sub(r'\s+', '', deduped[-2])
            if prev_compact.endswith(tail_compact) or tail_compact in prev_compact:
                deduped.pop()
    return deduped


def _split_trailing_inline_subsection_heading(text: str) -> Tuple[str, str]:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw:
        return '', ''
    match = _INLINE_SUBSECTION_TAIL_RE.match(raw)
    if not match:
        return raw, ''
    body = str(match.group('body') or '').strip()
    heading = str(match.group('heading') or '').strip()
    if not body or not heading:
        return raw, ''
    return body, heading


def _glue_soft_page_break_fragments(text: str) -> str:
    """Join OCR/page-break splits that should be one sentence in reading order."""
    raw = str(text or '')
    if not raw.strip():
        return ''
    raw = re.sub(r'(由于电缆)\s*\n+\s*(线路)', r'\1\2', raw)
    raw = re.sub(r'(导通泄流。)\s*(单端接地是)', r'\1\n\2', raw)
    return raw


def _format_searchable_paragraph_breaks(text: str) -> str:
    paragraphs: List[str] = []
    for chunk in re.split(r'\n{2,}', str(text or '').strip()):
        raw = re.sub(r'\s+', ' ', str(chunk or '').strip())
        if not raw:
            continue
        raw = _SEARCHABLE_LIST_ITEM_BREAK_RE.sub('\n', raw)
        raw = _SEARCHABLE_NUMBERED_QUESTION_BREAK_RE.sub('\n', raw)
        raw = _SEARCHABLE_NUMBERED_REFERENCE_BREAK_RE.sub('\n', raw)
        raw = re.sub(r'(两种接地方式：)\s*', r'\1\n', raw)
        raw = re.sub(r'。\s+(?=(?:单端|双端)接地是)', '。\n', raw)
        raw = re.sub(
            r'(?<=[。；;！？!?])\s+(?=[一二三四五六七八九十]+、)',
            '\n',
            raw,
        )
        paragraphs.append(raw.strip())
    return '\n\n'.join(paragraphs).strip()


def _strip_figure_diagram_legend_leakage(text: str) -> str:
    """Remove schematic legend phrases accidentally glued into body paragraphs."""
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw:
        return ''
    compact = re.sub(r'\s+', '', raw)
    if _CABLE_DIAGRAM_LEGEND_ONLY_RE.match(compact):
        return ''
    cleaned = _FIGURE_DIAGRAM_LEGEND_LEAK_RE.sub(r'\1', raw)
    # Repeated schematic switch labels only (not sentences like "单端接地是…").
    if re.fullmatch(r'(?:单端接地|双端接地)(?:\s+(?:单端接地|双端接地))+\s*', raw.strip()):
        return ''
    cleaned = re.sub(
        r'^(?:单端接地|双端接地)(?:\s+(?:单端接地|双端接地))+\s*',
        '',
        cleaned,
    )
    return re.sub(r'\s{2,}', ' ', cleaned).strip()


def _collapse_soft_wrapped_pdf_text(text: str) -> str:
    if not text:
        return ''
    lines = [str(line or '').strip() for line in str(text).splitlines() if str(line or '').strip()]
    collapsed = '\n'.join(_collapse_soft_wrapped_pdf_lines(lines))
    return _strip_figure_diagram_legend_leakage(collapsed)


def _merge_standalone_list_marker_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(blocks, list):
        return []

    def marker_value(text: str) -> str:
        marker = re.sub(r'\s+', '', str(text or '').strip())
        if not marker:
            return ''
        if re.fullmatch(r'\d{1,2}', marker):
            if marker == '48':
                return '4'
            try:
                value = int(marker)
            except Exception:
                return ''
            return marker if 0 < value <= 30 else ''
        if re.fullmatch(r'[①②③④⑤⑥⑦⑧⑨⑩]', marker):
            return marker
        if re.fullmatch(r'[（(]\s*[0-9一二三四五六七八九十]+\s*[）)]', marker):
            return marker
        return ''

    out: List[Dict[str, Any]] = []
    index = 0
    while index < len(blocks):
        block = blocks[index]
        if not isinstance(block, dict):
            index += 1
            continue

        marker = marker_value(_extract_pdf_block_text(block))
        if not marker or index + 1 >= len(blocks):
            out.append(block)
            index += 1
            continue

        next_block = blocks[index + 1]
        if not isinstance(next_block, dict):
            out.append(block)
            index += 1
            continue
        if (_safe_int(block.get('pageNumber')) or 0) != (_safe_int(next_block.get('pageNumber')) or 0):
            out.append(block)
            index += 1
            continue

        next_text = _extract_pdf_block_text(next_block)
        if not next_text or _line_starts_new_pdf_paragraph(next_text):
            out.append(block)
            index += 1
            continue

        marker_bbox = block.get('bbox') if isinstance(block.get('bbox'), dict) else None
        next_bbox = next_block.get('bbox') if isinstance(next_block.get('bbox'), dict) else None
        if marker_bbox and next_bbox:
            if int(marker_bbox.get('right', 0)) > int(next_bbox.get('left', 10**9)) + 48:
                out.append(block)
                index += 1
                continue
            if abs(int(next_bbox.get('top', 0)) - int(marker_bbox.get('top', 0))) > 90:
                out.append(block)
                index += 1
                continue

        merged = copy.deepcopy(next_block)
        merged_text = f"{marker} {next_text}".strip()
        if 'code' in merged:
            merged['code'] = merged_text
        elif 'text' in merged:
            merged['text'] = merged_text
        else:
            merged['contentMarkdown'] = merged_text
        merged['semanticRole'] = 'body'
        if marker_bbox and next_bbox:
            left = min(int(marker_bbox.get('left', 0)), int(next_bbox.get('left', 0)))
            top = min(int(marker_bbox.get('top', 0)), int(next_bbox.get('top', 0)))
            right = max(int(marker_bbox.get('right', 0)), int(next_bbox.get('right', 0)))
            bottom = max(int(marker_bbox.get('bottom', 0)), int(next_bbox.get('bottom', 0)))
            merged['bbox'] = _xywh_to_bbox_dict(left, top, max(0, right - left), max(0, bottom - top))
        out.append(merged)
        index += 2

    return out


def _split_inline_decimal_clause_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(blocks, list):
        return []

    out: List[Dict[str, Any]] = []
    clause_re = re.compile(r'(?<!\d)(\d{1,2}\.\d{1,2}\.\d{1,3})(?!\d)')
    for block in blocks:
        if not isinstance(block, dict):
            continue
        text = _extract_pdf_block_text(block)
        matches = list(clause_re.finditer(text or ''))
        if len(matches) < 2:
            out.append(block)
            continue

        for segment_index, match in enumerate(matches, start=1):
            start = match.start()
            end = matches[segment_index].start() if segment_index < len(matches) else len(text)
            segment = str(text[start:end] or '').strip()
            if not segment:
                continue
            new_block = copy.deepcopy(block)
            if 'code' in new_block:
                new_block['code'] = segment
            elif 'text' in new_block:
                new_block['text'] = segment
            else:
                new_block['contentMarkdown'] = segment
            new_block['semanticRole'] = 'body'
            base_id = str(new_block.get('id') or 'decimal_clause').strip() or 'decimal_clause'
            new_block['id'] = f'{base_id}__decimal_{segment_index}'
            reading_order = _safe_int(new_block.get('readingOrder'))
            if reading_order is not None:
                new_block['readingOrder'] = reading_order * 100 + segment_index
            out.append(new_block)
    return out


def _normalize_inline_ocr_list_markers(text: str) -> str:
    value = str(text or '')
    compact = re.sub(r'\s+', '', value)
    if not all(marker in compact for marker in ('1', '2', '3', '48')):
        return value
    return re.sub(r'(?<!\d)48(?=\s*[\u4e00-\u9fff])', '4', value)


def _normalize_block_inline_ocr_list_markers(block: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(block, dict):
        return block
    text = _extract_pdf_block_text(block)
    normalized = _normalize_inline_ocr_list_markers(text)
    if normalized == text:
        return block
    updated = copy.deepcopy(block)
    if 'code' in updated:
        updated['code'] = normalized
    elif 'text' in updated:
        updated['text'] = normalized
    else:
        updated['contentMarkdown'] = normalized
    return updated


def _merge_pdf_text_block_group(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = copy.deepcopy(group[0])
    merged_text = ''.join(_extract_pdf_block_text(block) for block in group)
    if 'code' in base:
        base['code'] = merged_text
    else:
        base['text'] = merged_text

    bbox_dicts = [block.get('bbox') for block in group if isinstance(block.get('bbox'), dict)]
    if bbox_dicts:
        left = min(int(bbox.get('left', 0)) for bbox in bbox_dicts)
        top = min(int(bbox.get('top', 0)) for bbox in bbox_dicts)
        right = max(int(bbox.get('right', left)) for bbox in bbox_dicts)
        bottom = max(int(bbox.get('bottom', top)) for bbox in bbox_dicts)
        base['bbox'] = _xywh_to_bbox_dict(left, top, max(0, right - left), max(0, bottom - top))
    return base


def _collapse_soft_wrapped_pdf_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(blocks, list):
        return []

    result: List[Dict[str, Any]] = []
    pending_group: List[Dict[str, Any]] = []

    def flush_group() -> None:
        nonlocal pending_group
        if not pending_group:
            return
        result.append(_merge_pdf_text_block_group(pending_group))
        pending_group = []

    def _maybe_merge_leading_list_marker(prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
        """Merge a standalone list marker like '1' into the following line."""
        try:
            if not isinstance(prev, dict) or not isinstance(curr, dict):
                return False
            if (_safe_int(prev.get('pageNumber')) or 0) != (_safe_int(curr.get('pageNumber')) or 0):
                return False
            prev_type = str(prev.get('type') or '').strip().lower()
            curr_type = str(curr.get('type') or '').strip().lower()
            if prev_type not in _TEXTUAL_BLOCK_TYPES or curr_type not in _TEXTUAL_BLOCK_TYPES:
                return False
            prev_role = str(prev.get('semanticRole') or '').strip().lower()
            curr_role = str(curr.get('semanticRole') or '').strip().lower()
            if prev_role not in _ENTRY_SEARCHABLE_ROLES or curr_role not in _ENTRY_SEARCHABLE_ROLES:
                return False

            marker = re.sub(r'\s+', '', _extract_pdf_block_text(prev))
            if not marker:
                return False
            # Accept typical list markers; also accept '48' as misread '4' when followed by prose.
            normalized_marker = marker
            if re.fullmatch(r'\d{1,2}', marker):
                if marker == '48':
                    normalized_marker = '4'
                elif int(marker) <= 0 or int(marker) > 30:
                    return False
            elif re.fullmatch(r'[①②③④⑤⑥⑦⑧⑨⑩]', marker):
                normalized_marker = marker
            elif re.fullmatch(r'[（(]\s*[0-9一二三四五六七八九十]+\s*[）)]', marker):
                normalized_marker = marker
            else:
                return False

            curr_text = _extract_pdf_block_text(curr)
            if not curr_text:
                return False
            # Do not merge if the next line is itself a clause start.
            if _line_starts_new_pdf_paragraph(curr_text):
                return False

            prev_bbox = prev.get('bbox') if isinstance(prev.get('bbox'), dict) else None
            curr_bbox = curr.get('bbox') if isinstance(curr.get('bbox'), dict) else None
            if prev_bbox and curr_bbox:
                # Marker should be to the left and near the first line.
                if int(prev_bbox.get('right', 0)) > int(curr_bbox.get('left', 10**9)) + 48:
                    return False
                if abs(int(curr_bbox.get('top', 0)) - int(prev_bbox.get('top', 0))) > 90:
                    return False

            merged_text = f"{normalized_marker} {curr_text}".strip()
            if 'code' in curr:
                curr['code'] = merged_text
            elif 'text' in curr:
                curr['text'] = merged_text
            else:
                curr['contentMarkdown'] = merged_text

            # Expand bbox to include marker when available.
            if prev_bbox and curr_bbox:
                left = min(int(prev_bbox.get('left', 0)), int(curr_bbox.get('left', 0)))
                top = min(int(prev_bbox.get('top', 0)), int(curr_bbox.get('top', 0)))
                right = max(int(prev_bbox.get('right', 0)), int(curr_bbox.get('right', 0)))
                bottom = max(int(prev_bbox.get('bottom', 0)), int(curr_bbox.get('bottom', 0)))
                curr['bbox'] = _xywh_to_bbox_dict(left, top, max(0, right - left), max(0, bottom - top))
            return True
        except Exception:
            return False

    for block in blocks:
        if not isinstance(block, dict):
            flush_group()
            continue

        block_type = str(block.get('type') or '').strip().lower()
        semantic_role = str(block.get('semanticRole') or '').strip().lower()
        text = _extract_pdf_block_text(block)
        if block_type not in _TEXTUAL_BLOCK_TYPES or semantic_role not in _ENTRY_SEARCHABLE_ROLES or not text:
            flush_group()
            result.append(copy.deepcopy(block))
            continue

        if not pending_group:
            pending_group.append(copy.deepcopy(block))
            continue

        prev_block = pending_group[-1]
        prev_text = _extract_pdf_block_text(prev_block)
        same_role = str(prev_block.get('semanticRole') or '').strip().lower() == semantic_role
        same_page = _safe_int(prev_block.get('pageNumber')) == _safe_int(block.get('pageNumber'))
        if same_page and _maybe_merge_leading_list_marker(prev_block, block):
            # drop marker block, keep merged current
            pending_group.pop()
            flush_group()
            pending_group.append(copy.deepcopy(block))
            continue
        if same_role and same_page and _should_merge_soft_wrapped_pdf_lines(prev_text, text):
            pending_group.append(copy.deepcopy(block))
            continue

        flush_group()
        pending_group.append(copy.deepcopy(block))

    flush_group()
    return result


def _entry_has_visual_blocks(entry: Dict) -> bool:
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get('type') or '').strip().lower()
        if block_type in {'table', 'figure', 'image', 'equation'}:
            return True
    return False


def _entry_is_numbered_list_page(text: str) -> bool:
    raw = str(text or '').strip()
    if len(raw) < 40:
        return False
    numbered_hits = len(re.findall(r'(?:^|\n)\d{1,2}[.．、]', raw))
    question_hits = len(re.findall(r'[？?]', raw))
    return numbered_hits >= 3 and question_hits >= 2


def _rebuild_entry_search_text(entry: Dict) -> str:
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    entry_page = _safe_int(entry.get('pageNumber')) or 0
    fallback_lines: List[str] = []
    ordered_segments: List[str] = []

    def _block_reading_sort_key(block: Dict[str, Any]) -> Tuple[int, int, int, int]:
        bbox = block.get('bbox') if isinstance(block.get('bbox'), dict) else {}
        return (
            _safe_int(block.get('pageNumber')) or 10**9,
            int(bbox.get('top', 10**9)),
            int(bbox.get('left', 10**9)),
            _safe_int(block.get('readingOrder')) or 10**9,
        )

    def _append_segment(raw_text: str) -> None:
        collapsed = '\n'.join(_collapse_soft_wrapped_pdf_lines([raw_text]))
        formatted = _format_searchable_paragraph_breaks(collapsed)
        if formatted:
            ordered_segments.append(formatted)

    scoped_blocks: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_page = _safe_int(block.get('pageNumber')) or entry_page
        if (
            entry_page > 0
            and block_page > 0
            and block_page != entry_page
            and not (
                _block_is_page_boundary_continuation_fragment(block, anchor_page=entry_page)
                or _block_is_cross_page_body_continuation(block, anchor_page=entry_page)
            )
        ):
            continue
        if _block_conflicts_with_entry_section(block, entry):
            continue
        scoped_blocks.append(block)

    ordered_blocks = sorted(scoped_blocks, key=_block_reading_sort_key)

    for block in ordered_blocks:
        if not isinstance(block, dict):
            continue
        text = _extract_pdf_block_text(block)
        if not text:
            continue
        if is_pdf_page_marker_text(text):
            continue
        block_type = str(block.get('type') or '').strip().lower()
        semantic_role = str(block.get('semanticRole') or '').strip().lower()
        if block_type in {'table', 'figure', 'image', 'equation'}:
            continue
        if semantic_role in _ENTRY_VISUAL_ONLY_ROLES:
            visual_type = 'table' if semantic_role == 'table_note' else 'figure'
            recovered = _recover_searchable_body_from_visual_only_block(text, visual_type=visual_type)
            if recovered:
                _append_segment(recovered)
            continue
        if semantic_role == 'heading':
            heading_text = re.sub(r'\s+', ' ', str(text).strip())
            if not heading_text:
                continue
            heading_compact = re.sub(r'\s+', '', heading_text)
            if re.match(r'^第[一二三四五六七八九十百0-9]+节', heading_compact) and len(heading_compact) <= 96:
                for segment in _split_inline_pdf_heading_segments(heading_text):
                    ordered_segments.append(segment)
                continue
            if re.match(r'^[一二三四五六七八九十]+、', heading_compact) and len(heading_compact) <= 64:
                # Subsection titles are recovered from figure-body openers / trailing splits.
                continue
            for segment in _split_inline_pdf_heading_segments(heading_text):
                ordered_segments.append(segment)
            continue
        if semantic_role == 'body':
            cleaned = _strip_figure_diagram_legend_leakage(text)
            if not cleaned or _is_diagram_switch_label_text(cleaned):
                continue
            prefix_part, heading_lead = _split_leading_inline_subsection_heading(cleaned)
            if heading_lead:
                if prefix_part and not _is_orphan_page_boundary_prefix(prefix_part):
                    _append_segment(prefix_part)
                if not _FIGURE_SECTION_OPENER_INLINE_REF_RE.search(cleaned):
                    ordered_segments.append(heading_lead)
                continue
            body_part, heading_tail = _split_trailing_inline_subsection_heading(cleaned)
            if heading_tail and body_part:
                body_part = f'{heading_tail}\n\n{body_part}'
                heading_tail = ''
            elif heading_tail:
                tail_compact = re.sub(r'\s+', '', heading_tail)
                if _is_subsection_only_heading_text(heading_tail):
                    pass
                elif not ordered_segments or re.sub(r'\s+', '', ordered_segments[-1]) != tail_compact:
                    ordered_segments.append(heading_tail)
            if body_part:
                recent_heading = ''
                for prior in reversed(ordered_segments):
                    if re.match(r'^[一二三四五六七八九十]+、', prior):
                        recent_heading = prior
                        break
                body_part = _resolve_figure_body_subsection_label(
                    entry,
                    body_part,
                    recent_heading=recent_heading,
                    prior_segments=ordered_segments,
                )
                _append_segment(body_part)
            continue
        if not semantic_role and block_type in _ENTRY_SEARCHABLE_BLOCK_TYPES:
            fallback_lines.append(text)

    if ordered_segments:
        deduped_segments = _trim_trailing_subsection_spillover_segments(
            _dedupe_ordered_search_segments(ordered_segments)
        )
        if not deduped_segments:
            return ''
        merged_parts = [deduped_segments[0]]
        for segment in deduped_segments[1:]:
            separator = _search_segment_join_separator(merged_parts[-1], segment)
            merged_parts[-1] = f'{merged_parts[-1]}{separator}{segment}'
        return _glue_soft_page_break_fragments(merged_parts[0])

    selected_lines = fallback_lines
    if selected_lines:
        paragraphs: List[str] = []
        for line in _dedupe_preserve_lines(selected_lines):
            collapsed = '\n'.join(_collapse_soft_wrapped_pdf_lines([line]))
            formatted = _format_searchable_paragraph_breaks(collapsed)
            if formatted:
                paragraphs.append(formatted)
        return '\n\n'.join(paragraphs)

    raw_text = str(entry.get('contentMarkdown') or '').strip()
    raw_lines = [str(line or '').strip() for line in raw_text.splitlines() if str(line or '').strip()]
    return '\n'.join(_collapse_soft_wrapped_pdf_lines(_dedupe_preserve_lines([
        line for line in raw_lines if not is_likely_pdf_callout_or_annotation(line)
    ])))


def _is_preliminary_page_entry(entry: Dict, search_text: str) -> bool:
    page_number = _safe_int(entry.get('pageNumber')) or 0
    if page_number <= 0:
        return False

    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    for block in _semantic_text_blocks(blocks):
        block_text = _extract_pdf_block_text(block)
        if not block_text:
            continue
        if _parse_top_level_pdf_item_number(block_text) is not None:
            return False
        if re.match(r'^答\s*[：:]', block_text):
            return False

    combined = '\n'.join([
        str(entry.get('jobTitle') or '').strip(),
        str(search_text or '').strip(),
        str(entry.get('contentMarkdown') or '').strip(),
    ]).lower()
    searchable_lines = [line for line in str(search_text or '').splitlines() if str(line or '').strip()]
    style_lines = _entry_text_lines_for_style_detection(entry)
    for line in searchable_lines:
        if line not in style_lines:
            style_lines.append(line)
    style_text = '\n'.join(style_lines)
    if _looks_like_pdf_table_of_contents(style_text, style_lines):
        return True
    if _looks_like_pdf_chapter_opener(style_lines):
        return True
    if page_number > 3:
        return False
    if any(marker in combined for marker in _PRELIMINARY_PAGE_MARKERS):
        return True
    for line in searchable_lines:
        stripped = str(line or '').strip()
        if _parse_top_level_pdf_item_number(stripped) is not None:
            return False
        if re.match(r'^答\s*[：:]', stripped):
            return False
    if len(searchable_lines) <= 2:
        return True
    return False


def _safe_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _block_contributes_to_pdf_entry_semantic_text(block: Dict[str, Any]) -> bool:
    block_type = str(block.get('type') or '').strip().lower()
    if block_type in {'image', 'img', 'figure', 'equation', 'table'}:
        return False

    semantic_role = str(block.get('semanticRole') or '').strip().lower()
    if semantic_role in _ENTRY_VISUAL_ONLY_ROLES:
        return False

    payload = _extract_pdf_block_text(block)
    if not payload:
        return False
    if is_pdf_page_marker_text(payload):
        return False
    if is_likely_pdf_callout_or_annotation(payload):
        return False
    return True


def _block_role(block: Dict[str, Any]) -> str:
    return str(block.get('semanticRole') or '').strip().lower()


def _semantic_text_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        if not _block_contributes_to_pdf_entry_semantic_text(block):
            continue
        out.append(block)
    return out


def _block_is_cross_page_body_continuation(
    block: Dict[str, Any],
    *,
    anchor_page: int = 0,
) -> bool:
    """Body on page N+1 that continues anchor page's paragraph (not a new section)."""
    if anchor_page <= 0:
        return False
    block_page = _safe_int(block.get('pageNumber')) or 0
    if block_page != anchor_page + 1:
        return False
    text = re.sub(r'\s+', ' ', _extract_pdf_block_text(block))
    if not text:
        return False
    body_part, heading_tail = _split_trailing_inline_subsection_heading(text)
    if heading_tail and _is_subsection_only_heading_text(heading_tail):
        text = body_part
    compact = re.sub(r'\s+', '', text)
    if not compact:
        return False
    if looks_like_pdf_attached_numbered_body_item(text):
        return False
    if _is_orphan_page_boundary_prefix(text):
        return False
    if re.match(r'^第[一二三四五六七八九十百0-9]+节', compact):
        return False
    if re.match(r'^[一二三四五六七八九十]+、', compact) and len(compact) <= 64:
        return False
    if _parse_top_level_pdf_item_number(text) is not None:
        return False
    return len(compact) <= 220


def _entry_section_leaf_compact(entry: Dict) -> str:
    section_path = entry.get('sectionPath') if isinstance(entry.get('sectionPath'), list) else []
    leaf = str(section_path[-1] if section_path else entry.get('jobTitle') or '').strip()
    return re.sub(r'\s+', '', leaf)


def _block_conflicts_with_entry_section(block: Dict[str, Any], entry: Dict) -> bool:
    """Drop a top-of-page tail that clearly belongs to the previous section, not this entry."""
    if not isinstance(block, dict):
        return False
    entry_page = _safe_int(entry.get('pageNumber')) or 0
    block_page = _safe_int(block.get('pageNumber')) or entry_page
    if block_page != entry_page or entry_page <= 0:
        return False
    if (_safe_int(block.get('readingOrder')) or 99) > 2:
        return False
    text_compact = re.sub(r'\s+', '', _extract_pdf_block_text(block))
    if not text_compact:
        return False
    leaf_compact = _entry_section_leaf_compact(entry)
    if not leaf_compact:
        return False
    if ('第六节' in leaf_compact or '电缆贯通线外铠' in leaf_compact) and text_compact.startswith(
        ('小电阻', '线路基本')
    ):
        return True
    return False


def _block_is_page_boundary_continuation_fragment(
    block: Dict[str, Any],
    *,
    anchor_page: int = 0,
) -> bool:
    """Text that continues the previous page's paragraph/question and must stay for merge."""
    if not isinstance(block, dict):
        return False
    block_page = _safe_int(block.get('pageNumber')) or 0
    if anchor_page > 0 and block_page > 0 and block_page != anchor_page + 1:
        return False
    text = _extract_pdf_block_text(block)
    compact = re.sub(r'\s+', '', str(text or '').strip())
    if not compact:
        return False
    if looks_like_pdf_attached_numbered_body_item(text):
        return False
    if re.match(r'^答\s*[：:]', compact):
        return True
    if re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩]', compact):
        return True
    if _parse_pdf_parenthesized_subitem_number(text) is not None:
        return True
    if re.match(r'^\d+\.\s', str(text or '').strip()):
        return False
    if re.match(r'^[一二三四五六七八九十]+、', compact):
        return False
    if re.match(r'^第[一二三四五六七八九十百0-9]+[章节]', compact):
        return False
    if re.match(r'^(图|表)\s*[0-9]', compact, flags=re.IGNORECASE):
        return False
    if _parse_top_level_pdf_item_number(text) is not None:
        return False
    if re.search(
        r'[。；;！？!?][^。；;！？!?]{0,24}[一二三四五六七八九十]+、',
        compact,
    ):
        return False
    return len(compact) <= 120


def _block_should_repartition_to_page_entry(block: Dict[str, Any], entry_page: int) -> bool:
    if not isinstance(block, dict):
        return False
    block_page = _safe_int(block.get('pageNumber')) or entry_page
    if block_page <= 0 or entry_page <= 0 or block_page == entry_page:
        return False
    if (
        _block_is_page_boundary_continuation_fragment(block, anchor_page=entry_page)
        or _block_is_cross_page_body_continuation(block, anchor_page=entry_page)
    ):
        return False
    block_id = str(block.get('id') or '').strip()
    if block_id.endswith('__tail'):
        return True
    role = _block_role(block)
    if role in {'figure', 'figure_callout', 'caption', 'table', 'table_note', 'heading'}:
        return True
    text = _extract_pdf_block_text(block)
    compact = re.sub(r'\s+', '', text)
    if not compact:
        return False
    if re.search(r'[一二三四五六七八九十]+、[^\n。；;！？!?]{2,48}$', compact):
        return True
    if re.match(r'^第[一二三四五六七八九十百0-9]+节', compact):
        return True
    return _block_is_page_boundary_merge_exempt(block)


def _block_is_page_boundary_merge_exempt(block: Dict[str, Any]) -> bool:
    if not isinstance(block, dict):
        return True
    block_id = str(block.get('id') or '').strip()
    if block_id.endswith('__tail'):
        return True
    role = _block_role(block)
    if role in {'caption', 'table_note', 'figure', 'figure_callout'}:
        return True
    text = _extract_pdf_block_text(block)
    compact = re.sub(r'\s+', '', text)
    if not compact:
        return False
    if re.match(r'^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]', compact, flags=re.IGNORECASE):
        return True
    if re.match(r'^[一二三四五六七八九十]+、', compact):
        return True
    if '如图所示' in compact or '一般应用' in compact:
        return True
    return False


def _leading_page_boundary_continuation_block_indexes(
    blocks: List[Dict[str, Any]],
    *,
    anchor_page: int = 0,
) -> List[int]:
    if not isinstance(blocks, list):
        return []

    prefix_indexes: List[int] = []
    seen_body = False
    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            break
        block_page = _safe_int(block.get('pageNumber')) or anchor_page
        if anchor_page > 0 and block_page > anchor_page + 1:
            break
        text = _extract_pdf_block_text(block)
        if not text or is_pdf_page_marker_text(text):
            if prefix_indexes:
                break
            continue

        role = _block_role(block)
        if role == 'heading':
            if prefix_indexes or seen_body:
                break
            continue
        if _block_is_page_boundary_merge_exempt(block):
            break
        if not _block_contributes_to_pdf_entry_semantic_text(block):
            if prefix_indexes:
                break
            continue
        if _parse_top_level_pdf_item_number(text) is not None:
            break

        seen_body = True
        prefix_indexes.append(block_index)

    return prefix_indexes


def _entry_has_answer_prefix(blocks: List[Dict[str, Any]]) -> bool:
    for block in _semantic_text_blocks(blocks):
        text = _extract_pdf_block_text(block)
        if not text:
            continue
        if re.match(r'^答\s*[：:]', text):
            return True
        if _parse_top_level_pdf_item_number(text) is not None:
            return False
    return False


def _last_semantic_body_text(blocks: List[Dict[str, Any]]) -> str:
    for block in reversed(blocks or []):
        if not isinstance(block, dict):
            continue
        if _block_role(block) != 'body':
            continue
        text = _extract_pdf_block_text(block)
        if text and not is_pdf_page_marker_text(text):
            return text
    return ''


def _should_merge_page_boundary_continuation(
    previous_blocks: List[Dict[str, Any]],
    current_blocks: List[Dict[str, Any]],
    prefix_indexes: List[int],
    *,
    previous_entry_page: int = 0,
    current_entry_page: int = 0,
) -> bool:
    if not prefix_indexes:
        return False

    prefix_texts = [
        _extract_pdf_block_text(current_blocks[idx])
        for idx in prefix_indexes
        if 0 <= idx < len(current_blocks)
    ]
    prefix_texts = [text for text in prefix_texts if text]
    if not prefix_texts:
        return False

    previous_tail = _last_semantic_body_text(previous_blocks)
    if not previous_tail:
        return False

    first_prefix = prefix_texts[0]
    first_prefix_block = current_blocks[prefix_indexes[0]] if prefix_indexes else None
    if isinstance(first_prefix_block, dict) and previous_entry_page > 0:
        prefix_page = _safe_int(first_prefix_block.get('pageNumber')) or current_entry_page
        if prefix_page > previous_entry_page + 1:
            return False
    if isinstance(first_prefix_block, dict) and _block_is_page_boundary_merge_exempt(first_prefix_block):
        return False
    if re.match(r'^[一二三四五六七八九十]+、', re.sub(r'\s+', '', first_prefix)):
        return False
    if re.match(r'^(图|表)\s*[0-9]', first_prefix, flags=re.IGNORECASE):
        return False
    prefix_is_subitem = bool(re.match(r'^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]', first_prefix))
    prefix_parenthesized_subitem = _parse_pdf_parenthesized_subitem_number(first_prefix)
    previous_parenthesized_subitem = _parse_pdf_parenthesized_subitem_number(previous_tail)

    if prefix_parenthesized_subitem is not None:
        if (
            previous_parenthesized_subitem is not None
            and prefix_parenthesized_subitem == previous_parenthesized_subitem + 1
        ):
            return True
        if re.search(r'[：:（(]$', previous_tail):
            return True

    if re.match(r'^答\s*[：:]', first_prefix):
        return not _entry_has_answer_prefix(previous_blocks)

    if re.match(r'^答\s*[：:]', previous_tail):
        if prefix_is_subitem:
            return True
        if re.search(r'[，、：:；,;（(《“\-]$', previous_tail):
            return True
        return not re.search(r'[。！？!?]$', previous_tail)

    # Plain prose continuations are only safe for Q/A-style entries. Standards
    # pages often start numbered clauses on the next page; merging those blocks
    # into the previous page creates duplicated and out-of-order detail text.
    if not any(re.search(r'[？?]', _extract_pdf_block_text(block)) for block in previous_blocks or []):
        return False

    if _line_has_unclosed_pdf_opening(previous_tail):
        return True

    if re.search(r'[，、：:；,;（(《“\-]$', previous_tail):
        return True

    if not re.search(r'[。！？!?；;]$', previous_tail):
        return True

    return False


def _filter_entry_noise_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        text = _extract_pdf_block_text(block)
        if text and is_pdf_page_marker_text(text):
            continue
        out.append(block)
    return out


def _parse_top_level_pdf_item_number(text: str) -> Optional[int]:
    raw = str(text or '').strip()
    if not raw:
        return None

    # Multi-level clause numbers (e.g., "4.2.13") are treated as a top-level split boundary.
    # We only need monotonic ordering within the same page/entry; compress into an integer.
    m = re.match(r'^\s*(\d{1,2})\.(\d{1,2})\.(\d{1,3})(?!\d)', raw)
    if m:
        try:
            a = int(m.group(1))
            b = int(m.group(2))
            c = int(m.group(3) or 0)
            # Default missing patch (X.Y) to 0; but treat 4.2.13 as (4,2,13), not (4,2,0).
            return a * 10000 + b * 100 + c
        except Exception:
            return None

    m = re.match(r'^\s*(\d{1,2})\.(\d{1,2})(?!\d)', raw)
    if m:
        try:
            a = int(m.group(1))
            b = int(m.group(2))
            return a * 10000 + b * 100
        except Exception:
            return None

    match = re.match(r'^\s*(\d{1,3})\s*[\.．、]', raw)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None

    clause_match = re.match(r'^\s*第\s*([0-9一二三四五六七八九十百千零〇两]+)\s*条', raw)
    if not clause_match:
        return None

    return _parse_cn_or_digit_int(clause_match.group(1))


def _parse_pdf_continuation_list_number(text: str) -> Optional[int]:
    raw = str(text or '').strip()
    if not raw or _parse_top_level_pdf_item_number(raw) is not None:
        return None
    match = re.match(r'^\s*0?(\d{1,2})(?!\s*[\.．、])\s+\S', raw)
    if not match:
        return None
    try:
        number = int(match.group(1))
    except Exception:
        return None
    return number if number > 0 else None


def _parse_pdf_parenthesized_subitem_number(text: str) -> Optional[int]:
    raw = str(text or '').strip()
    if not raw or _parse_top_level_pdf_item_number(raw) is not None:
        return None
    match = re.match(r'^[（(]\s*([0-9一二三四五六七八九十百千零〇两]+)\s*[）)]', raw)
    if not match:
        return None
    return _parse_cn_or_digit_int(match.group(1))


def _leading_pdf_continuation_block_indexes(blocks: List[Dict[str, Any]]) -> List[int]:
    if not isinstance(blocks, list):
        return []

    prefix_indexes: List[int] = []
    last_number: Optional[int] = None

    for block_index, block in enumerate(blocks):
        if not isinstance(block, dict):
            break
        if not _block_contributes_to_pdf_entry_semantic_text(block):
            if prefix_indexes:
                break
            continue

        text = _extract_pdf_block_text(block)
        if _parse_top_level_pdf_item_number(text) is not None:
            break

        number = _parse_pdf_continuation_list_number(text)
        if number is None:
            break
        if last_number is not None and number != last_number + 1:
            break

        prefix_indexes.append(block_index)
        last_number = number

    return prefix_indexes


def _trailing_pdf_continuation_numbers(blocks: List[Dict[str, Any]]) -> List[int]:
    if not isinstance(blocks, list):
        return []

    numbers_reversed: List[int] = []
    expected_previous: Optional[int] = None

    for block in reversed(blocks):
        if not isinstance(block, dict):
            break
        if not _block_contributes_to_pdf_entry_semantic_text(block):
            if numbers_reversed:
                break
            continue

        text = _extract_pdf_block_text(block)
        if _parse_top_level_pdf_item_number(text) is not None:
            break

        number = _parse_pdf_continuation_list_number(text)
        if number is None:
            break
        if expected_previous is not None and number != expected_previous - 1:
            break

        numbers_reversed.append(number)
        expected_previous = number

    return list(reversed(numbers_reversed))


def _entry_blocks_combined_semantic_text(entry: Dict) -> str:
    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    parts: List[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if not _block_contributes_to_pdf_entry_semantic_text(block):
            continue
        text = _extract_pdf_block_text(block)
        if text:
            parts.append(text)
    return '\n'.join(parts)


def _entry_has_narrative_section_headings(blocks: List[Dict[str, Any]]) -> bool:
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if not _block_contributes_to_pdf_entry_semantic_text(block):
            continue
        compact = re.sub(r'\s+', '', _extract_pdf_block_text(block))
        if not compact:
            continue
        if re.match(r'^第[一二三四五六七八九十百0-9]+节', compact):
            return True
        if re.match(r'^[一二三四五六七八九十]+、', compact) and len(compact) <= 64:
            return True
    return False


def _combined_text_has_narrative_section_markers(text: str) -> bool:
    raw = str(text or '').strip()
    if not raw:
        return False
    compact = re.sub(r'\s+', '', raw)
    if re.search(r'第[一二三四五六七八九十百0-9]+节', compact):
        return True
    subsection_hits = len(re.findall(r'[一二三四五六七八九十]+、', raw))
    numbered_hits = len(re.findall(r'(?:^|\n)\d{1,2}[.．、]', raw))
    if subsection_hits >= 1 and numbered_hits >= 2:
        return True
    if '（如图' in raw or '如图' in raw:
        if numbered_hits >= 2 and len(re.findall(r'答\s*[：:]', raw)) < 2:
            return True
    return False


def _numbered_block_is_clause_style(block: Dict[str, Any]) -> bool:
    text = str(_extract_pdf_block_text(block) or '').strip()
    return bool(re.match(r'^第\s*([0-9一二三四五六七八九十百千零〇两]+)\s*条', text))


def _numbered_block_is_decimal_list_item(block: Dict[str, Any]) -> bool:
    text = str(_extract_pdf_block_text(block) or '').strip()
    return _parse_top_level_pdf_item_number(text) is not None and bool(
        re.match(r'^\s*\d{1,3}\s*[\.．、]', text)
    )


def _entry_should_auto_split_by_numbered_blocks(
    entry: Dict,
    numbered_blocks: List[Tuple[int, int]],
) -> bool:
    """Gate auto_split: Q&A/regulation yes; narrative section + 1.2.3. list no."""
    if not numbered_blocks or len(numbered_blocks) < 2:
        return False

    blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
    combined = _entry_blocks_combined_semantic_text(entry)
    if _entry_is_numbered_list_page(combined):
        return True

    numbers = [number for _block_index, number in numbered_blocks]
    if any(next_number <= current_number for current_number, next_number in zip(numbers, numbers[1:])):
        return False

    clause_hits = sum(
        1
        for block_index, _number in numbered_blocks
        if block_index < len(blocks)
        and isinstance(blocks[block_index], dict)
        and _numbered_block_is_clause_style(blocks[block_index])
    )
    if clause_hits >= 2:
        return True

    narrative_context = _entry_has_narrative_section_headings(blocks) or _combined_text_has_narrative_section_markers(
        combined
    )
    if narrative_context:
        if all(
            block_index < len(blocks)
            and isinstance(blocks[block_index], dict)
            and _numbered_block_is_decimal_list_item(blocks[block_index])
            for block_index, _number in numbered_blocks
        ):
            if len(re.findall(r'答\s*[：:]', combined)) < 2:
                return False

    return True


def _mark_entry_auto_split_child(entry: Dict, *, is_child: bool) -> None:
    hints = entry.get('retrievalHints') if isinstance(entry.get('retrievalHints'), dict) else {}
    hints = dict(hints)
    hints['autoSplitChild'] = bool(is_child)
    entry['retrievalHints'] = hints


def _is_pdf_native_page_entry_id(entry_id: str) -> bool:
    return bool(re.match(r'^pdf_native::page_\d+(?:__auto_split_\d+)*$', str(entry_id or '').strip()))


def _coalesce_same_page_pdf_native_entries(entries: List[Dict]) -> List[Dict]:
    """Merge pdf_native page_N and page_N__auto_split_* siblings before re-normalizing."""
    if not isinstance(entries, list):
        return []

    page_buckets: Dict[int, List[Tuple[int, Dict]]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        page_number = _safe_int(entry.get('pageNumber')) or 0
        entry_id = str(entry.get('entryId') or '').strip()
        if page_number > 0 and _is_pdf_native_page_entry_id(entry_id):
            page_buckets.setdefault(page_number, []).append((index, entry))

    merged_by_page: Dict[int, Dict] = {}
    for page_number, bucket in page_buckets.items():
        if len(bucket) <= 1:
            merged_by_page[page_number] = copy.deepcopy(bucket[0][1])
            continue
        bucket.sort(key=lambda item: (_safe_int(item[1].get('position')) or item[0], item[0]))
        merged = copy.deepcopy(bucket[0][1])
        merged['entryId'] = f'pdf_native::page_{page_number}'
        merged_blocks: List[Dict[str, Any]] = []
        for _position, sibling in bucket:
            sibling_blocks = sibling.get('blocks') if isinstance(sibling.get('blocks'), list) else []
            for block in sibling_blocks:
                if isinstance(block, dict):
                    merged_blocks.append(copy.deepcopy(block))
        merged['blocks'] = merged_blocks
        hints = merged.get('retrievalHints') if isinstance(merged.get('retrievalHints'), dict) else {}
        hints = dict(hints)
        hints.pop('autoSplitChild', None)
        merged['retrievalHints'] = hints
        merged_by_page[page_number] = merged

    out: List[Dict] = []
    emitted_pages: Set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        page_number = _safe_int(entry.get('pageNumber')) or 0
        entry_id = str(entry.get('entryId') or '').strip()
        if page_number > 0 and _is_pdf_native_page_entry_id(entry_id):
            if page_number in emitted_pages:
                continue
            emitted_pages.add(page_number)
            out.append(merged_by_page.get(page_number, copy.deepcopy(entry)))
            continue
        out.append(copy.deepcopy(entry))
    return out


def _split_numbered_pdf_page_entries(entries: List[Dict]) -> List[Dict]:
    if not isinstance(entries, list):
        return []

    used_entry_ids: Set[str] = {
        str(entry.get('entryId') or '').strip()
        for entry in entries
        if isinstance(entry, dict) and str(entry.get('entryId') or '').strip()
    }
    out: List[Dict] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        kind = str(entry.get('kind') or 'text').strip().lower()
        blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
        if kind != 'text' or len(blocks) < 2:
            out.append(copy.deepcopy(entry))
            continue

        entry_id = str(entry.get('entryId') or '').strip()
        if '__auto_split_' in entry_id and not _entry_is_numbered_list_page(
            _entry_blocks_combined_semantic_text(entry)
        ):
            out.append(copy.deepcopy(entry))
            continue

        numbered_blocks: List[Tuple[int, int]] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if not _block_contributes_to_pdf_entry_semantic_text(block):
                continue
            item_number = _parse_top_level_pdf_item_number(_extract_pdf_block_text(block))
            if item_number is not None:
                numbered_blocks.append((block_index, item_number))

        if len(numbered_blocks) < 2:
            out.append(copy.deepcopy(entry))
            continue

        numbers = [number for _block_index, number in numbered_blocks]
        if any(next_number <= current_number for current_number, next_number in zip(numbers, numbers[1:])):
            out.append(copy.deepcopy(entry))
            continue

        if not _entry_should_auto_split_by_numbered_blocks(entry, numbered_blocks):
            out.append(copy.deepcopy(entry))
            continue

        first_numbered_index = numbered_blocks[0][0]
        part_starts = [0] + [block_index for block_index, _number in numbered_blocks[1:]] if first_numbered_index > 0 else [block_index for block_index, _number in numbered_blocks]
        base_entry_id = str(entry.get('entryId') or 'entry').strip() or 'entry'

        for part_index, start_block_index in enumerate(part_starts):
            end_block_index = part_starts[part_index + 1] if part_index + 1 < len(part_starts) else len(blocks)
            part_entry = copy.deepcopy(entry)
            part_entry['blocks'] = copy.deepcopy(blocks[start_block_index:end_block_index])
            if part_index > 0:
                suffix = part_index + 1
                candidate_entry_id = f'{base_entry_id}__auto_split_{suffix}'
                while candidate_entry_id in used_entry_ids:
                    suffix += 1
                    candidate_entry_id = f'{base_entry_id}__auto_split_{suffix}'
                used_entry_ids.add(candidate_entry_id)
                part_entry['entryId'] = candidate_entry_id
                _mark_entry_auto_split_child(part_entry, is_child=True)
            else:
                _mark_entry_auto_split_child(part_entry, is_child=False)
            out.append(part_entry)

    return out


def _merge_page_boundary_numbered_continuations(entries: List[Dict]) -> List[Dict]:
    if not isinstance(entries, list):
        return []

    out: List[Dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        current = copy.deepcopy(entry)
        if not out:
            out.append(current)
            continue

        previous = out[-1]
        if not isinstance(previous, dict):
            out.append(current)
            continue

        previous_page = _safe_int(previous.get('pageNumber')) or 0
        current_page = _safe_int(current.get('pageNumber')) or 0
        if previous_page <= 0 or current_page != previous_page + 1:
            out.append(current)
            continue

        if str(previous.get('kind') or 'text').strip().lower() != 'text' or str(current.get('kind') or 'text').strip().lower() != 'text':
            out.append(current)
            continue

        previous_blocks = previous.get('blocks') if isinstance(previous.get('blocks'), list) else []
        current_blocks = current.get('blocks') if isinstance(current.get('blocks'), list) else []
        merge_indexes: List[int] = []

        numbered_prefix_indexes = _leading_pdf_continuation_block_indexes(current_blocks)
        if numbered_prefix_indexes:
            previous_numbers = _trailing_pdf_continuation_numbers(previous_blocks)
            prefix_numbers = [
                _parse_pdf_continuation_list_number(_extract_pdf_block_text(current_blocks[idx]))
                for idx in numbered_prefix_indexes
            ]
            if previous_numbers and not any(number is None for number in prefix_numbers):
                normalized_prefix_numbers = [int(number) for number in prefix_numbers if number is not None]
                if normalized_prefix_numbers and normalized_prefix_numbers[0] == previous_numbers[-1] + 1:
                    merge_indexes = list(numbered_prefix_indexes)

        if not merge_indexes:
            generic_prefix_indexes = _leading_page_boundary_continuation_block_indexes(
                current_blocks,
                anchor_page=current_page,
            )
            if _should_merge_page_boundary_continuation(
                previous_blocks,
                current_blocks,
                generic_prefix_indexes,
                previous_entry_page=previous_page,
                current_entry_page=current_page,
            ):
                merge_indexes = list(generic_prefix_indexes)

        if not merge_indexes:
            out.append(current)
            continue

        previous['blocks'] = copy.deepcopy(previous_blocks) + [
            copy.deepcopy(current_blocks[idx]) for idx in merge_indexes
        ]
        remaining_blocks = [
            copy.deepcopy(block)
            for block_index, block in enumerate(current_blocks)
            if block_index not in set(merge_indexes)
        ]
        if not remaining_blocks:
            out[-1] = previous
            continue

        current['blocks'] = remaining_blocks
        out[-1] = previous
        out.append(current)

    return out


def _repartition_blocks_by_page_number(entries: List[Dict]) -> List[Dict]:
    """Move blocks back to the entry that matches block.pageNumber after boundary merges."""
    if not isinstance(entries, list):
        return []

    page_to_entry: Dict[int, Dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        page_number = _safe_int(entry.get('pageNumber'))
        if page_number is not None and page_number > 0:
            page_to_entry[int(page_number)] = entry

    moved = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_page = _safe_int(entry.get('pageNumber')) or 0
        blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
        if not blocks:
            continue

        kept: List[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                kept.append(block)
                continue
            block_page = _safe_int(block.get('pageNumber')) or entry_page
            must_move_far_page = (
                block_page > 0
                and entry_page > 0
                and block_page > entry_page + 1
            )
            if (
                block_page > 0
                and entry_page > 0
                and block_page != entry_page
                and (must_move_far_page or _block_should_repartition_to_page_entry(block, entry_page))
            ):
                target = page_to_entry.get(int(block_page))
                if isinstance(target, dict):
                    target_blocks = target.get('blocks')
                    if not isinstance(target_blocks, list):
                        target_blocks = []
                        target['blocks'] = target_blocks
                    target_blocks.append(copy.deepcopy(block))
                    moved = True
                    continue
            kept.append(block)
        entry['blocks'] = kept

    if not moved:
        return entries

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blocks = entry.get('blocks') if isinstance(entry.get('blocks'), list) else []
        if not blocks:
            continue
        blocks.sort(
            key=lambda block: (
                _safe_int(block.get('readingOrder')) or 10**9,
                int((block.get('bbox') or {}).get('top', 10**9)) if isinstance(block.get('bbox'), dict) else 10**9,
            )
        )
        for idx, block in enumerate(blocks, start=1):
            if isinstance(block, dict):
                block['readingOrder'] = int(idx)
    return entries


def _entry_figure_label_hits(text: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    raw = str(text or '')
    for match in re.finditer(r'((?:附图|附表|图|表)\s*[0-9一二三四五六七八九十零〇O口]+(?:\s*[-—~〜\.．]\s*[0-9一二三四五六七八九十零〇O口]+)*)', raw):
        value = str(match.group(1) or '').strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _pdf_heading_signal_score(
    text: str,
    *,
    semantic_role: str = '',
    bbox: Optional[Dict] = None,
    reading_order: Optional[int] = None,
) -> int:
    score = 0
    role = str(semantic_role or '').strip().lower()
    if role == 'heading':
        score += 4
    elif role in {'caption', 'table_note', 'artifact', 'figure_callout', 'annotation', 'ocr_annotation', 'legend'}:
        score -= 5

    detected = _detect_pdf_heading(text)
    if detected is not None:
        score += 4

    if isinstance(reading_order, int) and reading_order > 0 and reading_order <= 6:
        score += 2

    if isinstance(bbox, dict):
        try:
            width = int(bbox.get('width') or max(0, int(bbox.get('right') or 0) - int(bbox.get('left') or 0)))
        except Exception:
            width = 0
        try:
            height = int(bbox.get('height') or max(0, int(bbox.get('bottom') or 0) - int(bbox.get('top') or 0)))
        except Exception:
            height = 0
        if width >= 140:
            score += 1
        if height >= 20:
            score += 1

    if is_likely_pdf_callout_or_annotation(text):
        score -= 6
    return score


def _iter_structured_heading_candidates(entry: Dict[str, Any]) -> List[Dict[str, object]]:
    blocks = entry.get('blocks')
    candidates: List[Dict[str, object]] = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            text = _extract_pdf_block_text(block)
            if not text:
                continue
            reading_order = _safe_int(block.get('readingOrder'))
            for segment_index, segment in enumerate(_split_inline_pdf_heading_segments(text)):
                candidates.append({
                    'text': segment,
                    'semanticRole': str(block.get('semanticRole') or '').strip().lower(),
                    'bbox': block.get('bbox') if isinstance(block.get('bbox'), dict) else None,
                    'readingOrder': reading_order if reading_order is None else (reading_order + segment_index),
                })
        if candidates:
            candidates.sort(key=lambda item: (
                int(item.get('readingOrder') or 10**9),
                int(((item.get('bbox') or {}).get('top')) or 10**9),
            ))
            return candidates[:20]

    text = str(entry.get('contentMarkdown') or '').strip()
    for index, line in enumerate([str(ln or '').strip() for ln in text.splitlines() if str(ln or '').strip()][:20], start=1):
        candidates.append({
            'text': line,
            'semanticRole': '',
            'bbox': None,
            'readingOrder': index,
        })
    return candidates


def _allows_visual_appendix_heading(
    detected: Optional[tuple],
    *,
    line: str,
    semantic_role: str,
    reading_order: Optional[int],
) -> bool:
    if detected is None:
        return False
    kind, _number, display = detected
    if kind != 'appendix':
        return False
    if semantic_role not in {'figure_callout', 'caption', 'legend', 'ocr_annotation'}:
        return False
    if reading_order is not None and reading_order > 4:
        return False
    compact = re.sub(r'\s+', '', display or line)
    return bool(compact) and len(compact) <= 12


def _apply_structured_context(
    entries: List[Dict],
    page_blue_headings: Optional[dict] = None,
) -> Dict[str, int]:
    current_unit = ''
    current_section = ''
    current_subsection = ''
    page_blue_headings = page_blue_headings if isinstance(page_blue_headings, dict) else {}
    audit = {
        'headingCandidatesSeen': 0,
        'headingsAccepted': 0,
        'headingCandidatesRejectedFigureScope': 0,
        'headingCandidatesRejectedWeakSignal': 0,
        'fuzzySubsectionsAccepted': 0,
        'blueHeadingHintsApplied': 0,
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            page_number = int(entry.get('pageNumber') or 0)
        except Exception:
            page_number = 0
        page_hints = page_blue_headings.get(page_number) if page_number > 0 else None
        if page_hints:
            current_unit, current_section, current_subsection, applied_count = _apply_blue_page_heading_hints(
                [hint for hint in page_hints if isinstance(hint, dict)],
                current_unit=current_unit,
                current_section=current_section,
                current_subsection=current_subsection,
            )
            audit['blueHeadingHintsApplied'] += int(applied_count)
        entry_lines = _entry_text_lines_for_style_detection(entry)
        combined_entry_text = '\n'.join(entry_lines)
        if _looks_like_pdf_table_of_contents(combined_entry_text, entry_lines):
            continue
        local_unit = current_unit
        local_section = current_section
        local_subsection = current_subsection
        last_detected_kind = ''
        skip_heuristic_headings = bool(page_hints)
        for candidate in _iter_structured_heading_candidates(entry):
            if skip_heuristic_headings:
                break
            line = str(candidate.get('text') or '').strip()
            semantic_role = str(candidate.get('semanticRole') or '').strip().lower()
            bbox = candidate.get('bbox') if isinstance(candidate.get('bbox'), dict) else None
            reading_order = _safe_int(candidate.get('readingOrder'))
            signal_score = _pdf_heading_signal_score(
                line,
                semantic_role=semantic_role,
                bbox=bbox,
                reading_order=reading_order,
            )
            detected = _detect_pdf_heading(line)
            if detected:
                audit['headingCandidatesSeen'] += 1
                allow_visual_appendix = _allows_visual_appendix_heading(
                    detected,
                    line=line,
                    semantic_role=semantic_role,
                    reading_order=reading_order,
                )
                if not allow_visual_appendix and (
                    semantic_role in {'figure_callout', 'caption', 'legend', 'ocr_annotation'}
                    or is_likely_pdf_callout_or_annotation(line)
                ):
                    audit['headingCandidatesRejectedFigureScope'] += 1
                    continue
                if signal_score < 4 and not allow_visual_appendix:
                    audit['headingCandidatesRejectedWeakSignal'] += 1
                    continue
                kind, _number, display = detected
                if kind in {'part', 'chapter', 'appendix'}:
                    current_unit = display
                    current_section = ''
                    current_subsection = ''
                elif kind == 'section':
                    current_section = display
                    current_subsection = ''
                elif kind == 'subsection':
                    current_subsection = display
                local_unit = current_unit
                local_section = current_section
                local_subsection = current_subsection
                last_detected_kind = kind
                audit['headingsAccepted'] += 1
                continue

            if last_detected_kind == 'section':
                if semantic_role != 'heading' or signal_score < 6:
                    if line:
                        audit['headingCandidatesRejectedWeakSignal'] += 1
                    continue
                fuzzy = _PDF_SUBSECTION_FUZZY_RE.match(line)
                if fuzzy:
                    fuzzy_title = str(fuzzy.group(1) or '').strip()
                    if _is_short_heading_title(fuzzy_title, max_len=18):
                        current_subsection = f'一、{fuzzy_title}'
                        local_subsection = current_subsection
                        last_detected_kind = 'subsection'
                        audit['fuzzySubsectionsAccepted'] += 1

        effective_unit = local_unit or local_section or local_subsection
        effective_section = ''
        effective_subsection = ''
        if local_unit:
            effective_section = local_section
            effective_subsection = local_subsection
        elif local_section:
            effective_subsection = local_subsection

        section_path = [part for part in [effective_unit, effective_section, effective_subsection] if part]
        if section_path:
            entry['unitName'] = effective_unit or entry.get('unitName') or '未分类'
            entry['sectionPath'] = section_path
            entry['jobTitle'] = ' / '.join(section_path)
            try:
                entry['classificationConfidence'] = round(max(float(entry.get('classificationConfidence') or 0.0), 0.72), 3)
            except Exception:
                entry['classificationConfidence'] = 0.72
            retrieval_hints = entry.get('retrievalHints') if isinstance(entry.get('retrievalHints'), dict) else {}
            retrieval_hints['isProvisional'] = False
            entry['retrievalHints'] = retrieval_hints
    return audit


def _file_structure_metrics(entries: List[Dict]) -> Dict[str, object]:
    total = len(entries)
    if total <= 0:
        return {
            'coverageStatus': 'unknown',
            'structureConfidence': 0.0,
        }

    classified = 0
    non_empty_text = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        unit_name = str(e.get('unitName') or '').strip()
        if unit_name and unit_name != '未分类':
            classified += 1
        txt = str(e.get('contentNormalized') or e.get('contentMarkdown') or '').strip()
        if len(txt) >= 8:
            non_empty_text += 1

    ratio_classified = classified / float(total)
    ratio_text = non_empty_text / float(total)
    confidence = min(0.99, 0.1 + 0.55 * ratio_classified + 0.35 * ratio_text)

    return {
        # PDF-only ingestion often has partial inputs; keep explicit conservative status.
        'coverageStatus': 'partial_or_unknown',
        'structureConfidence': round(confidence, 3),
    }


def _inject_internal_figure_links(text: str, blocks: List[Dict[str, Any]]) -> str:
    """Identify 'See Figure X-Y' and convert to markdown internal links if match found."""
    if not text or not blocks:
        return text

    # Pattern for Figure X-Y or Table X-Y
    ref_pattern = re.compile(r'([见详见]?[图表]\s*(\d+[-—.]\d+|\d+))')

    def replacer(match):
        full_match = match.group(1)
        label = match.group(2)

        # Try to find a matching block
        target_id = None
        for b in blocks:
            b_cap = str(b.get('caption') or '').strip()
            if label in b_cap:
                target_id = b.get('id')
                break

        if target_id:
            return f"[{full_match}](#{target_id})"
        return full_match

    return ref_pattern.sub(replacer, text)


def _build_global_toc_index(entries: List[Dict]) -> Dict[str, str]:
    """Map chapter/section/clause identifiers to entryIds."""
    index = {}
    for entry in entries:
        title = str(entry.get('jobTitle') or '').strip()
        entry_id = str(entry.get('entryId') or '')
        if not entry_id: continue

        # Exact match for full title path (e.g. "第一章 / 第一节 / 1.1.1 条")
        index[title] = entry_id

        # Match leaf part (e.g. "1.1.1条" or "第一章")
        parts = [p.strip() for p in title.split('/') if p.strip()]
        if parts:
            index[parts[-1]] = entry_id
            # Also support version without "条" suffix if numeric
            leaf = parts[-1]
            if leaf.endswith('条'):
                index[leaf[:-1].strip()] = entry_id

        # Match numeric patterns in blocks
        for block in entry.get('blocks', []):
            if block.get('semanticRole') == 'heading':
                txt = _extract_pdf_block_text(block)
                if txt:
                    t = txt.strip()
                    index[t] = entry_id
                    if t.endswith('条'): index[t[:-1].strip()] = entry_id
    return index


def _inject_all_cross_references(entries: List[Dict]):
    """Inject internal links for both figures and clauses."""
    global_index = _build_global_toc_index(entries)

    # regex for: 第 X.Y.Z 条, 第 X 章, 第 X 节, 第 X.Y 节
    clause_pattern = re.compile(r'([见详见]?(第\s*(\d+(\.\d+)*|[0-9一二三四五六七八九十百]+)\s*[章节条款]))')

    for entry in entries:
        text = str(entry.get('contentMarkdown') or '')
        blocks = entry.get('blocks', [])

        # 1. Figures/Tables (local/contextual)
        text = _inject_internal_figure_links(text, blocks)

        # 2. Clauses/Sections (global)
        def clause_replacer(match):
            full_match = match.group(1) # [见]第 1.2 条
            identifier = match.group(2) # 第 1.2 条

            target_id = global_index.get(identifier)
            if not target_id:
                # Try without "第" prefix for numeric parts
                inner_id = match.group(3)
                if inner_id:
                    target_id = global_index.get(inner_id)

            if target_id:
                return f"[{full_match}](#{target_id})"
            return full_match

        text = clause_pattern.sub(clause_replacer, text)
        entry['contentMarkdown'] = text
        entry['contentNormalized'] = text


def _refresh_normalized_entry_text_fields(entry: Dict[str, Any]) -> None:
    if not isinstance(entry, dict):
        return
    original_content_markdown = str(entry.get('contentMarkdown') or '')
    rebuilt_content_markdown = _rebuild_entry_search_text(entry)
    is_preliminary_page = _is_preliminary_page_entry(entry, rebuilt_content_markdown)
    if is_preliminary_page:
        rebuilt_content_markdown = ''

    rebuilt_content_markdown = _normalize_inline_ocr_list_markers(rebuilt_content_markdown)

    blocks = entry.get('blocks', [])
    if isinstance(blocks, list):
        rebuilt_content_markdown = _inject_internal_figure_links(rebuilt_content_markdown, blocks)

    entry['contentMarkdown'] = rebuilt_content_markdown
    entry['contentNormalized'] = rebuilt_content_markdown

    retrieval_hints = entry.get('retrievalHints') if isinstance(entry.get('retrievalHints'), dict) else {}
    retrieval_hints['excludeFromSearch'] = bool(is_preliminary_page or (not rebuilt_content_markdown and _entry_has_visual_blocks(entry)))
    retrieval_hints['searchWeight'] = 0.0 if is_preliminary_page else (0.15 if not rebuilt_content_markdown and _entry_has_visual_blocks(entry) else 1.0)
    retrieval_hints['visualOnlyTextStripped'] = bool(original_content_markdown.strip() and original_content_markdown.strip() != rebuilt_content_markdown.strip())
    retrieval_hints['preliminaryPage'] = bool(is_preliminary_page)
    entry['retrievalHints'] = retrieval_hints


def _normalize_entries_for_kb(entries: List[Dict]) -> List[Dict]:
    """Normalize entries into the canonical fields used by knowledge_base.json."""
    out: List[Dict] = []
    normalized_source_entries = _repartition_blocks_by_page_number(
        _merge_page_boundary_numbered_continuations(
            _split_numbered_pdf_page_entries(
                _coalesce_same_page_pdf_native_entries(entries)
            )
        )
    )
    for idx, e in enumerate(normalized_source_entries, start=1):
        if not isinstance(e, dict):
            continue
        ne = dict(e)
        ne.pop('image_base64', None)
        ne['unitName'] = str(ne.get('unitName') or '未分类')
        try:
            ne['position'] = int(ne.get('position') or idx)
        except Exception:
            ne['position'] = idx
        ne['kind'] = str(ne.get('kind') or _infer_entry_kind(ne) or 'text')
        ne['entryId'] = str(ne.get('entryId') or f'entry_{idx:04d}')
        ne['jobTitle'] = str(ne.get('jobTitle') or f'条目{idx}')
        try:
            ne['pageNumber'] = int(ne.get('pageNumber') or 1)
        except Exception:
            ne['pageNumber'] = 1
        ne['_styleDetectionLines'] = _entry_text_lines_for_style_detection(ne)
        blocks = ne.get('blocks')
        ne['blocks'] = _normalize_pdf_entry_blocks(blocks if isinstance(blocks, list) else [])
        ne['blocks'] = _filter_standalone_part_label_blocks(ne['blocks'])
        ne['blocks'] = _canonicalize_inline_headings_in_blocks(ne['blocks'])
        ne['blocks'] = _sort_entry_blocks_for_document_order(ne['blocks'], ne)

        # Backward-compatible schema evolution: keep legacy fields unchanged,
        # append additive metadata for better retrieval/classification.
        kind = ne.get('kind') or 'text'
        quality = _entry_quality_metrics(ne)
        ne['contentType'] = _to_content_type(str(kind))
        ne['sectionPath'] = [ne['unitName']] if ne['unitName'] and ne['unitName'] != '未分类' else []
        ne['classificationConfidence'] = quality.get('classificationConfidence')
        ne['retrievalHints'] = {
            'isProvisional': bool(quality.get('isProvisional')),
            'ocrQuality': 'unknown',
            'excludeFromSearch': False,
            'searchWeight': 1.0,
            'visualOnlyTextStripped': False,
            'preliminaryPage': False,
        }
        ne['evidence'] = {
            'sourcePage': ne['pageNumber'],
            'anchorText': str(ne.get('jobTitle') or ''),
        }
        _refresh_normalized_entry_text_fields(ne)
        out.append(ne)
    out = _repartition_blocks_by_page_number(_merge_page_boundary_numbered_continuations(out))
    for entry in out:
        _refresh_normalized_entry_text_fields(entry)
    _repair_page_boundary_visual_captions(out)
    page_blue_headings: Dict[int, List[Dict[str, Any]]] = {}
    for entry in out:
        if not isinstance(entry, dict):
            continue
        annotation_context = entry.get('annotationContext')
        if not isinstance(annotation_context, dict):
            continue
        hints = annotation_context.get('blueHeadingHints')
        if not isinstance(hints, list) or not hints:
            continue
        try:
            page_number = int(entry.get('pageNumber') or 0)
        except Exception:
            page_number = 0
        if page_number <= 0:
            continue
        page_blue_headings.setdefault(page_number, []).extend(
            [hint for hint in hints if isinstance(hint, dict)]
        )
    for page_number, hints in list(page_blue_headings.items()):
        page_blue_headings[page_number] = sorted(
            hints,
            key=lambda item: int(item.get('readingOrder') or 0),
        )
    context_audit = _apply_structured_context(out, page_blue_headings=page_blue_headings)
    _repair_appendix_form_visual_captions(out)
    for entry in out:
        _refresh_normalized_entry_text_fields(entry)
    for entry in out:
        if not isinstance(entry, dict):
            continue
        evidence = entry.get('evidence') if isinstance(entry.get('evidence'), dict) else {}
        evidence['headingAudit'] = dict(context_audit)
        entry['evidence'] = evidence
    for index, entry in enumerate(out, start=1):
        if isinstance(entry, dict):
            entry['position'] = int(index)
            entry.pop('_styleDetectionLines', None)

    # Final pass for global cross-references (anchoring clauses/figures across entries)
    _inject_all_cross_references(out)

    return out


def _count_images_in_entries(entries: List[Dict]) -> int:
    count = 0
    for e in entries:
        blocks = e.get('blocks') if isinstance(e, dict) else None
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            typ = str(b.get('type') or '').lower()
            if typ in ('image', 'figure', 'equation', 'table') and (b.get('imageUri') or b.get('src')):
                count += 1
    return count


def _build_output_payload(
    *,
    entries: List[Dict],
    file_id: str,
    out_path: str,
    pdf_path: str,
    source: Optional[str] = None,
    trim_audit_fields: bool = False,
) -> Dict[str, object]:
    """Build knowledge_base.json-style payload."""
    normalized_entries = _normalize_entries_for_kb(entries)
    if not source:
        source = f"assets/kb/{file_id}".replace('\\', '/')
    metrics = _file_structure_metrics(normalized_entries)
    payload = {
        'fileMetadata': {
            'schemaVersion': '2.0',
            'fileId': file_id,
            'fileName': os.path.basename(out_path),
            'source': source,
            'importTimestamp': None,
            'entriesCount': len(normalized_entries),
            'imagesCount': _count_images_in_entries(normalized_entries),
            'docSha256': _compute_sha256(pdf_path),
            'coverageStatus': metrics.get('coverageStatus'),
            'structureConfidence': metrics.get('structureConfidence'),
            'semanticAudit': _collect_semantic_audit(normalized_entries),
            'buildTool': 'pdf_to_base64_kb.py',
            'renderDpi': 300,
            'coordinateUnit': 'pt', # PDF points (1/72 inch)
        },
        'entries': normalized_entries,
    }
    if trim_audit_fields:
        try:
            payload['fileMetadata'].pop('semanticAudit', None)
        except Exception:
            pass
        for entry in payload.get('entries', []):
            if not isinstance(entry, dict):
                continue
            evidence = entry.get('evidence') if isinstance(entry.get('evidence'), dict) else None
            if isinstance(evidence, dict):
                evidence.pop('headingAudit', None)
    return payload


def _build_production_clean_output_path(out_path: str) -> str:
    root, ext = os.path.splitext(out_path)
    if not ext:
        ext = '.json'
    return root + '.production_clean' + ext


def _write_json_atomic(path: str, payload: Dict[str, object]) -> None:
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as tf:
        json.dump(payload, tf, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _collect_semantic_audit(entries: List[Dict]) -> Dict[str, int]:
    audit = {
        'entriesWithFigureScopeTitles': 0,
        'entriesWithWeakLeafTitle': 0,
        'entriesWithFigureReferences': 0,
    }
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        job_title = str(entry.get('jobTitle') or '').strip()
        parts = [part.strip() for part in job_title.split('/') if part.strip()]
        leaf = parts[-1] if parts else ''
        if leaf and is_likely_pdf_callout_or_annotation(leaf):
            audit['entriesWithFigureScopeTitles'] += 1
            audit['entriesWithWeakLeafTitle'] += 1
        blocks = entry.get('blocks')
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                text = _extract_pdf_block_text(block)
                if _entry_figure_label_hits(text):
                    audit['entriesWithFigureReferences'] += 1
                    break
    return audit


def _cleanup_non_visual_assets(assets_dir: str, entries: List[Dict]) -> None:
    """Remove non-visual helper files so only visual screenshots remain."""
    try:
        if not os.path.isdir(assets_dir):
            return
    except Exception:
        return

    keep_names: set[str] = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        blocks = e.get('blocks')
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            uri = str(b.get('imageUri') or b.get('src') or '')
            name = _uri_basename(uri).lower()
            if name and name.startswith('visual_p'):
                keep_names.add(name)

    removed = 0
    for name in os.listdir(assets_dir):
        path = os.path.join(assets_dir, name)
        if not os.path.isfile(path):
            continue
        low = name.lower()
        if not re.search(r'\.(png|jpg|jpeg|webp)$', low):
            continue
        if low.startswith('visual_p') and low in keep_names:
            continue
        try:
            os.remove(path)
            removed += 1
        except Exception:
            continue

    try:
        _print_utf(f"[CLEANUP] Assets cleaned in {assets_dir}: removed={removed}, kept_visual={len(keep_names)}")
    except Exception:
        pass
    


def _paddle_result_to_lines(result, repeated_watermark_candidates: Optional[set] = None) -> List[str]:
    """Normalize common PaddleOCR return structures into list of text lines.

    This is defensive: Paddle may return nested lists/tuples or numpy-like
    objects that support `tolist()`. We prefer safe type checks instead of
    calling methods that may not exist.
    """
    if not result or not isinstance(result, list):
        return []
    lines: List[str] = []
    try:
        # If result is a single-item list that itself contains the real list,
        # prefer that inner list.
        target = result[0] if isinstance(result[0], list) and len(result) == 1 else result
    except Exception:
        target = result
    for line in target:
        if isinstance(line, (list, tuple)) and len(line) > 1:
            # common shapes: (box, (text, score)) or (box, text)
            try:
                cand = line[1]
                if isinstance(cand, (list, tuple)) and len(cand) > 0:
                    text = cand[0]
                elif isinstance(cand, str):
                    text = cand
                else:
                    text = ''
            except Exception:
                text = ''
            if text:
                lines.append(str(text).strip())
        elif isinstance(line, str):
            # sometimes a direct string is returned
            if line.strip():
                lines.append(line.strip())

    # Probe: show counts before dedupe
    try:
        tcount = len(target) if hasattr(target, '__len__') else 0
        print(f"[PROBE-FILTER] Input lines count: {tcount}, Extracted text count: {len(lines)}")
    except Exception:
        pass
    # dedupe while preserving order
    out: List[str] = []
    for l in lines:
        if l and l not in out:
            out.append(l)
    return _filter_repeated_watermark_lines(out, repeated_watermark_candidates)


def _paddle_result_to_text_blocks(result, repeated_watermark_candidates: Optional[set] = None) -> List[Dict]:
    if not result or not isinstance(result, list):
        return []
    try:
        target = result[0] if isinstance(result[0], list) and len(result) == 1 else result
    except Exception:
        target = result

    blocks: List[Dict] = []
    for idx, line in enumerate(target, start=1):
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        box = line[0]
        cand = line[1]
        try:
            text = cand[0] if isinstance(cand, (list, tuple)) and len(cand) > 0 else (cand if isinstance(cand, str) else '')
        except Exception:
            text = ''
        if not text:
            continue
        if _looks_like_repeated_watermark_text(text, repeated_watermark_candidates):
            continue

        try:
            points = box if isinstance(box, (list, tuple)) else []
            xs = [int(round(float(pt[0]))) for pt in points if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            ys = [int(round(float(pt[1]))) for pt in points if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            if not xs or not ys:
                continue
            x0 = min(xs)
            x1 = max(xs)
            y0 = min(ys)
            y1 = max(ys)
            if x1 <= x0 or y1 <= y0:
                continue
            blocks.append({
                'text': str(text).strip(),
                'bbox': (int(x0), int(y0), int(x1 - x0), int(y1 - y0)),
                'readingOrder': int(idx),
            })
        except Exception:
            continue
    return blocks


def _ocr_image_bytes_with_structure(
    image_bytes: bytes,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[set] = None,
) -> Tuple[str, List[Dict]]:
    if ocr_engine == 'auto':
        if _HAS_PADDLE:
            ocr_engine = 'paddle'
        elif _HAS_TESSERACT:
            ocr_engine = 'tesseract'
        else:
            return ('', [])

    if ocr_engine == 'paddle' and _HAS_PADDLE:
        try:
            ocr = _get_shared_paddle_ocr()
            if ocr is None:
                return ('', [])
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return ('', [])
            result = ocr.ocr(img, cls=True)
            try:
                if hasattr(result, 'tolist'):
                    result = result.tolist()
            except Exception:
                pass
            lines = _paddle_result_to_lines(result, repeated_watermark_candidates=repeated_watermark_candidates)
            blocks = _paddle_result_to_text_blocks(result, repeated_watermark_candidates=repeated_watermark_candidates)
            return ('\n'.join(lines), blocks)
        except Exception:
            return ('', [])

    return (
        _ocr_image_bytes(
            image_bytes,
            ocr_engine=ocr_engine,
            repeated_watermark_candidates=repeated_watermark_candidates,
        ),
        [],
    )


def _ocr_image_bytes(
    image_bytes: bytes,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[set] = None,
) -> str:
    """Run OCR on bytes and return recognized text (joined).

    ocr_engine: 'auto'|'paddle'|'tesseract'
    """
    if ocr_engine == 'auto':
        if _HAS_PADDLE:
            ocr_engine = 'paddle'
        elif _HAS_TESSERACT:
            ocr_engine = 'tesseract'
        else:
            return ''

    if ocr_engine == 'paddle' and _HAS_PADDLE:
        try:
            ocr = _get_shared_paddle_ocr()
            if ocr is None:
                return ''
            # PaddleOCR accepts file path or numpy array
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return ''
            # PROBE-IMG: audit image before OCR
            try:
                print(f"[PROBE-IMG] Type: {type(img)}, Shape: {img.shape if hasattr(img, 'shape') else 'N/A'}, Mean: {np.mean(img) if hasattr(img, 'mean') else 'N/A'}")
            except Exception:
                pass
            # additional deep input audit before calling OCR
            try:
                img_array = img
                print(f"[PROBE-INPUT] Shape: {getattr(img_array, 'shape', 'N/A')}, Dtype: {getattr(img_array, 'dtype', 'N/A')}")
                try:
                    print(f"[PROBE-INPUT] Min: {np.min(img_array)}, Max: {np.max(img_array)}, Std: {np.std(img_array)}")
                    if np.std(img_array) < 5:
                        print("[PROBE-WARNING] Image contrast is extremely low!")
                except Exception:
                    pass
            except Exception:
                pass
            # Probe: image info before OCR (legacy)
            try:
                meanpix = np.mean(img) if img is not None else 'None'
            except Exception:
                meanpix = 'ERR'
            try:
                img_info = img.size if hasattr(img, 'size') else getattr(img, 'shape', None)
            except Exception:
                img_info = 'UNKNOWN'
            try:
                print('[PROBE-STEP] Entering Paddle Engine...')
                result = ocr.ocr(img, cls=True)
                print(f"[PROBE-STEP] Exit Paddle Engine. Result type: {type(result)}")
            except Exception:
                result = None
                print('[PROBE-STEP] Exit Paddle Engine. Exception during OCR')
            # PROBE-RAW: engine raw output summary and per-block print
            try:
                # quick summary
                print(f"[PROBE-RAW] Result Length: {len(result) if result else 0}, Raw Snippet: {str(result)[:300]}")
            except Exception:
                pass
            try:
                # iterate robustly over returned blocks and print blocks with confidence > 0
                blocks = []
                if result and isinstance(result, list):
                    # paddle sometimes nests the list inside a single-element list
                    if len(result) == 1 and isinstance(result[0], list):
                        blocks = result[0]
                    else:
                        blocks = result
                print(f"[PROBE-RAW] Found {len(blocks) if blocks else 0} text blocks.")
                if blocks:
                    for bi, blk in enumerate(blocks):
                        try:
                            text = ''
                            conf = None
                            if isinstance(blk, (list, tuple)):
                                cand = blk[1] if len(blk) > 1 else blk[0]
                                if isinstance(cand, (list, tuple)):
                                    text = cand[0] if len(cand) > 0 else ''
                                    conf = cand[1] if len(cand) > 1 else None
                                elif isinstance(cand, str):
                                    text = cand
                                elif isinstance(cand, dict):
                                    text = cand.get('text', '') or cand.get('transcript', '')
                                    conf = cand.get('score') or cand.get('confidence')
                            elif isinstance(blk, str):
                                text = blk
                            if conf is None:
                                # print text blocks without confidence as N/A
                                if text and bi < 100:
                                    print(f"[PROBE-BLOCK-{bi}] Text: {text}, Confidence: N/A")
                            else:
                                try:
                                    conf_val = float(conf)
                                except Exception:
                                    conf_val = None
                                if conf_val is not None and conf_val > 0:
                                    if bi < 100:
                                        print(f"[PROBE-BLOCK-{bi}] Text: {text}, Confidence: {conf_val}")
                        except Exception:
                            continue
            except Exception:
                pass
            # Avoid AttributeError from blindly calling .tolist()
            try:
                if hasattr(result, 'tolist'):
                    result = result.tolist()
            except Exception:
                # keep original if conversion fails
                pass
            try:
                print(f"[PROBE] Paddle Raw Result: {result}")
            except Exception:
                pass
            try:
                print(f"[DEBUG_STRUCTURE] {str(result)[:800]}")
            except Exception:
                pass
            # normalize using robust helper
            try:
                lines = _paddle_result_to_lines(result, repeated_watermark_candidates=repeated_watermark_candidates)
            except Exception:
                lines = []
            return "\n".join(lines)
        except Exception:
            import traceback
            print(f"[PROBE] Error caught: {traceback.format_exc()}")
            return ''

    if ocr_engine == 'tesseract' and _HAS_TESSERACT:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('L')
            # 'chi_sim' might not be available in all installations
            try:
                text = pytesseract.image_to_string(img, lang='chi_sim')
            except Exception:
                text = pytesseract.image_to_string(img)
            return text
        except Exception:
            return ''

    return ''


def _text_to_table_rows(text: str) -> Optional[list]:
    if not text:
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) <= 1:
        return None
    rows: List[List[str]] = []
    for line in lines:
        # split by 2+ spaces or tab to try to detect columns
        cells = re.split(r'\s{2,}|\t', line)
        cells = [c.strip() for c in cells if c.strip()]
        if cells:
            rows.append(cells)
    return rows if len(rows) > 0 else None


def _table_rows_to_markdown(rows: List[List[str]]) -> str:
    """Convert a 2D list of table rows into a simple Markdown table.

    - Pads shorter rows with empty cells to form a rectangular table.
    - Uses the first row as header when possible.
    """
    if not rows:
        return ''
    try:
        ncols = max(len(r) for r in rows)
        def pad(r):
            return r + [''] * (ncols - len(r))
        header = pad(rows[0])
        lines: List[str] = []
        lines.append('| ' + ' | '.join(h.strip() for h in header) + ' |')
        lines.append('| ' + ' | '.join(['---'] * ncols) + ' |')
        for r in rows[1:]:
            rp = pad(r)
            lines.append('| ' + ' | '.join((c.strip() if isinstance(c, str) else str(c)) for c in rp) + ' |')
        return '\n'.join(lines)
    except Exception:
        return ''


def universal_clean_text(text: str) -> str:
    """
    通用型文本清洗：针对专业文档水印、网址、以及OCR常见的冗余字符
    """
    if not text:
        return ""

    # 1. 【黑名单模式】针对已知的干扰项（可根据需要添加）
    black_list = [
        r'学兔兔', r'www\.bzfxw\.com', r'标准分享网',
        r'建设工程教育网', r'建筑标准库', r'工标网'
    ]
    for pattern in black_list:
        try:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        except Exception:
            pass

    # 2. 【模式匹配】干掉所有典型的 URL 网址
    url_pattern = r'(https?://[^\s]+)|(www\.[^\s]+\.[^\s]+)'
    try:
        text = re.sub(url_pattern, '', text, flags=re.IGNORECASE)
    except Exception:
        pass

    # 3. 【位置模式】（可选）页眉页脚等模式暂不强制删除

    # 4. 【格式清洗】修复 OCR 产生的多余空格和空行
    try:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(_collapse_soft_wrapped_pdf_lines(lines))
    except Exception:
        return text.strip()


def build_kb_from_pdf(
    pdf_path: str,
    crops_dir: str = './output/crops',
    out_path: str = 'knowledge_base.pdf_native.json',
    assets_root: str = os.path.join('app', 'src', 'main', 'assets', 'kb'),
    file_id: Optional[str] = None,
    ocr_engine: str = 'auto',
    max_page_image_width: int = 1200,
    page_render_dpi: int = 300,
    start_page: int = 1,
    max_pages: Optional[int] = None,
    keep_aux_images: bool = False,
    layout_engine: str = 'auto',
    yolo_layout_model: Optional[str] = None,
    yolo_layout_conf: float = YOLO_LAYOUT_DEFAULT_CONF,
    yolo_layout_imgsz: int = YOLO_LAYOUT_DEFAULT_IMGSZ,
    yolo_layout_device: str = '',
    layout_debug_dir: Optional[str] = None,
):
    """Main builder.

    - pdf_path: source PDF
    - crops_dir: optional external crops (kept for compatibility)
    - out_path: json output
    - assets_root: base folder where `kb/[file_id]/截图/` will be created
    - file_id: if None, derived from PDF basename
    - ocr_engine: 'auto'|'paddle'|'tesseract' to prefer an OCR backend
    - max_page_image_width: 'auto'|'paddle'|'tesseract' to prefer an OCR backend
    - keep_aux_images: keep `page_*` and `embedded_*` helper images when True
    """
    crops_map: Dict[str, str] = {}
    if os.path.isdir(crops_dir):
        # reuse old behavior for external crops but prefer saving to assets below
        for name in sorted(os.listdir(crops_dir)):
            path = os.path.join(crops_dir, name)
            if os.path.isfile(path):
                low = name.lower()
                if low.endswith('.png') or low.endswith('.jpg') or low.endswith('.jpeg') or low.endswith('.webp'):
                    with open(path, 'rb') as f:
                        b = f.read()
                    mime = 'image/png' if low.endswith('.png') else ('image/webp' if low.endswith('.webp') else 'image/jpeg')
                    crops_map[name] = 'data:%s;base64,%s' % (mime, base64.b64encode(b).decode('ascii'))

    # Path audit: ensure output and assets roots are valid before heavy work
    try:
        out_dir = os.path.abspath(os.path.dirname(out_path) or '.')
        if not os.path.isdir(out_dir):
            _print_utf(f"[ERROR] Output directory does not exist: {out_dir}. Create it or provide a valid --out path.")
            raise SystemExit(2)
    except Exception as e:
        _print_utf(f"[ERROR] Invalid output path '{out_path}': {e}")
        raise SystemExit(2)

    try:
        assets_root_abs = os.path.abspath(assets_root)
        os.makedirs(assets_root_abs, exist_ok=True)
    except Exception as e:
        _print_utf(f"[ERROR] Cannot create assets root directory: {assets_root_abs}. Error: {e}")
        raise SystemExit(2)
    # normalize assets_root to absolute path
    assets_root = assets_root_abs

    # Safety check: ensure stdout/stderr are not redirected to the same path as --out
    try:
        out_abs = os.path.abspath(out_path)
        for stream_name, stream in (('stdout', sys.stdout), ('stderr', sys.stderr)):
            try:
                if hasattr(stream, 'name'):
                    dest = getattr(stream, 'name')
                    if dest and os.path.abspath(dest) == out_abs:
                        # cannot safely continue: console output is redirected to output file
                        _print_utf(f"[ERROR] Detected {stream_name} redirected to the same file as --out ({out_path}). This will corrupt the JSON output. Aborting.")
                        raise SystemExit(3)
            except Exception:
                continue
    except Exception:
        # best-effort only; do not fail if detection is not possible
        pass

    doc = fitz.open(pdf_path)
    pages_text: Dict[int, str] = {}
    for i in range(doc.page_count):
        try:
            page = doc.load_page(i)
            pages_text[i + 1] = page.get_text('text') or ''
        except Exception:
            pages_text[i + 1] = ''
    repeated_watermark_candidates = _collect_repeated_watermark_candidates(doc)
    dynamic_page_artifact_signatures = _collect_dynamic_page_artifact_signatures(doc)

    # derive file_id
    if not file_id:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        file_id = sanitize_file_id(base)

    assets_dir = os.path.join(assets_root, file_id, '截图')
    ensure_dir(assets_dir)
    layout_debug_dir_abs = ''
    if layout_debug_dir:
        try:
            layout_debug_dir_abs = os.path.abspath(str(layout_debug_dir).strip())
            ensure_dir(layout_debug_dir_abs)
        except Exception:
            layout_debug_dir_abs = ''

    resolved_layout_engine = str(layout_engine or 'auto').strip().lower() or 'auto'
    if resolved_layout_engine not in {'auto', 'yolo', 'ppstructure', 'cv'}:
        resolved_layout_engine = 'auto'

    resolved_yolo_layout_model = str(
        yolo_layout_model or os.environ.get('YOLO_LAYOUT_MODEL') or ''
    ).strip()
    yolo_layout_detector = None
    if resolved_layout_engine in {'auto', 'yolo'} and resolved_yolo_layout_model:
        yolo_layout_detector = _instantiate_yolo_layout_detector(resolved_yolo_layout_model)
        if yolo_layout_detector is None:
            _print_utf(
                f"[WARN] YOLO layout model unavailable; falling back to other layout sources. "
                f"model={resolved_yolo_layout_model} err={_ULTRALYTICS_IMPORT_ERR or 'init_failed'}"
            )
    elif resolved_layout_engine == 'yolo' and not resolved_yolo_layout_model:
        _print_utf("[WARN] layout_engine=yolo but no model path was provided; falling back to other layout sources.")

    # Instantiate PP-Structure once for reuse when available
    ppst = None
    ppst_init_error = ''
    if resolved_layout_engine in {'auto', 'ppstructure'} and _HAS_PADDLE:
        try:
            from paddleocr import PPStructure
            try:
                ppst = PPStructure(table_threshold=0.2)
            except Exception:
                try:
                    ppst = PPStructure()
                except Exception:
                    ppst = None
                    ppst_init_error = 'PPStructure() init failed'
        except Exception:
            ppst = None
            try:
                ppst_init_error = str(sys.exc_info()[1])
            except Exception:
                ppst_init_error = 'import PPStructure failed'

    if resolved_layout_engine in {'auto', 'ppstructure'} and ppst is None:
        if _HAS_PADDLE:
            _print_utf(
                f"[WARN] PaddleOCR detected but PPStructure init failed. Falling back to CV-only layout. "
                f"python={sys.executable} err={ppst_init_error or 'unknown'}"
            )
        else:
            _print_utf(
                f"[WARN] PaddleOCR not available in current interpreter. Falling back to CV-only layout. "
                f"python={sys.executable} err={_PADDLE_IMPORT_ERR or 'module not found'}"
            )

    entries: List[Dict] = []
    pages: List[Dict] = []  # explicit pages collection for final_kb['pages']

    # If external crops exist, include them first (but we'll also try to detect tables)
    if crops_map:
        for name, data_uri in crops_map.items():
            # save crop into assets dir
            try:
                # decode base64
                header, b64 = data_uri.split(',', 1)
                img_bytes = base64.b64decode(b64)
                out_name = name
                out_path_img = os.path.join(assets_dir, out_name)
                with open(out_path_img, 'wb') as f:
                    f.write(img_bytes)
                _print_utf(f"[SUCCESS] Saved external crop: {out_path_img}")
                uri = f'file:///android_asset/kb/{file_id}/截图/{out_name}'
            except Exception:
                uri = name
            entry = {
                'entryId': f'pdf_native::{name}',
                'jobTitle': name,
                'pageNumber': 0,
                'contentMarkdown': '',
                'contentNormalized': '',
                'blocks': [
                    {'type': 'image', 'imageUri': uri, 'caption': name}
                ],
            }
            entries.append(entry)

    # Determine page range to process (start_page is 1-based)
    start_index = max(0, start_page - 1)
    if max_pages is None:
        end_index = doc.page_count
    else:
        end_index = min(doc.page_count, start_index + max_pages)

    # For each page: save full page image, extract embedded images and tables
    for i in range(start_index, end_index):
        page_num = i + 1
        page_progress_status = 'started'
        try:
            page = doc.load_page(i)
        except Exception:
            continue

        _clear_generated_page_assets(assets_dir, page_num)

        # Prepare placeholders
        entry: Optional[Dict] = None
        page_obj: Optional[Dict] = None

        # Use outer try/except so any unexpected failure still results in a
        # placeholder entry and an atomic write in the finally block.
        try:
            # render full page image and save
            try:
                jpg_bytes = _render_page_to_jpeg_bytes(page, max_width=max_page_image_width, quality=85, dpi=page_render_dpi)
            except TypeError:
                jpg_bytes = _render_page_to_jpeg_bytes(page, max_width=max_page_image_width, quality=85)
            page_text = _filter_repeated_watermark_text(
                pages_text.get(page_num, ''),
                repeated_watermark_candidates,
            )
            try:
                _page_img_probe = Image.open(io.BytesIO(jpg_bytes)).convert('RGB')
                page_render_w, page_render_h = _page_img_probe.size
            except Exception:
                page_render_w, page_render_h = (0, 0)

            try:
                page_rect = page.rect
                pdf_w, pdf_h = float(page_rect.width), float(page_rect.height)
                # Capture physical offsets (CropBox origin)
                pdf_offset_x, pdf_offset_y = float(page_rect.x0), float(page_rect.y0)
                pdf_rotation = int(getattr(page, 'rotation', 0))
            except Exception:
                pdf_w, pdf_h = 0.0, 0.0
                pdf_offset_x, pdf_offset_y = 0.0, 0.0
                pdf_rotation = 0

            red_body_exclusion_regions: List[Tuple[int, int, int, int]] = []

            blue_heading_boxes: List[Tuple[int, int, int, int]] = []
            page_blue_heading_hints: List[Dict[str, Any]] = []
            if page_render_w > 0 and page_render_h > 0:
                try:
                    blue_heading_boxes = _extract_blue_annotation_boxes_from_page(
                        page,
                        int(page_render_w),
                        int(page_render_h),
                        jpg_bytes,
                    )
                    page_blue_heading_hints = _extract_blue_heading_hints_from_boxes(
                        page,
                        blue_heading_boxes,
                        int(page_render_w),
                        int(page_render_h),
                    )
                except Exception:
                    blue_heading_boxes = []
                    page_blue_heading_hints = []

            blocks: List[Dict] = []
            embedded: List[Dict] = []
            # embedded images are auxiliary by default; keep only when requested
            if keep_aux_images:
                embedded = _extract_embedded_images_from_page(doc, i, assets_dir)
                for img_meta in embedded:
                    try:
                        fname = img_meta['filename']
                        uri = f'file:///android_asset/kb/{file_id}/截图/{fname}'
                        blocks.append({'type': 'image', 'imageUri': uri, 'caption': f'Embedded p{page_num}', 'semanticRole': 'figure'})
                    except Exception:
                        continue

            use_red_annotation_mode = False
            red_annotation_layout_blocks = _detect_red_annotation_layout_blocks(
                page,
                jpg_bytes,
                ocr_engine=ocr_engine,
                repeated_watermark_candidates=repeated_watermark_candidates,
            )
            if red_annotation_layout_blocks:
                use_red_annotation_mode = True
            if use_red_annotation_mode and page_render_w > 0 and page_render_h > 0:
                red_body_exclusion_regions = _build_red_body_exclusion_regions(
                    red_annotation_layout_blocks,
                    int(page_render_w),
                    int(page_render_h),
                )

            # layout detection: optional YOLO + optional PP-Structure + existing CV fallback
            layout_inputs: List[Any] = []
            yolo_layout_items: List[Dict[str, Any]] = []
            pp_items: List[Any] = []
            if use_red_annotation_mode:
                layout_blocks = red_annotation_layout_blocks
                layout_support_boxes: List[Tuple[int, int, int, int]] = []
                layout_avoidance_constraints: List[Dict[str, Any]] = []
            elif yolo_layout_detector is not None and resolved_layout_engine in {'auto', 'yolo'}:
                try:
                    yolo_layout_items = _detect_yolo_layout_blocks(
                        yolo_layout_detector,
                        jpg_bytes,
                        conf=float(yolo_layout_conf),
                        imgsz=int(yolo_layout_imgsz),
                        device=str(yolo_layout_device or '').strip(),
                    )
                except Exception:
                    yolo_layout_items = []
                if yolo_layout_items:
                    layout_inputs.extend(yolo_layout_items)

            if not use_red_annotation_mode and ppst is not None and resolved_layout_engine in {'auto', 'ppstructure'}:
                used_decoded_input = False
                layout_input = jpg_bytes
                if _HAS_NUMPY and _HAS_CV2:
                    try:
                        arr_pp = np.frombuffer(jpg_bytes, dtype=np.uint8)
                        img_pp = cv2.imdecode(arr_pp, cv2.IMREAD_COLOR)
                        if img_pp is not None:
                            layout_input = img_pp
                            used_decoded_input = True
                    except Exception:
                        layout_input = jpg_bytes
                        used_decoded_input = False

                res_pp = None
                try:
                    res_pp = ppst(layout_input)
                except Exception:
                    # Fallback to raw bytes if decoded input path fails.
                    if used_decoded_input:
                        try:
                            res_pp = ppst(jpg_bytes)
                        except Exception:
                            res_pp = None
                    else:
                        res_pp = None

                pp_items = _coerce_layout_items(res_pp)
                if pp_items:
                    layout_inputs.extend(pp_items)

            if not use_red_annotation_mode:
                layout_result = layout_inputs if layout_inputs else None
                layout_blocks = _detect_layout_blocks(layout_result, jpg_bytes, table_threshold=0.2, kb_image_dir=assets_dir)
                layout_support_boxes = _collect_layout_support_boxes(layout_result)
                layout_avoidance_constraints = _collect_layout_avoidance_constraints(layout_result)
                try:
                    layout_blocks = _suppress_redundant_layout_figure_blocks(layout_blocks)
                except Exception:
                    pass
            if layout_debug_dir_abs:
                _write_layout_debug_page(
                    layout_debug_dir_abs,
                    page_num=page_num,
                    layout_engine=resolved_layout_engine,
                    yolo_model_path=resolved_yolo_layout_model,
                    yolo_raw_items=yolo_layout_items,
                    pp_items=pp_items or [],
                    accepted_layout_blocks=layout_blocks,
                )
            img = None
            page_cv_main = None
            crop_index = 0
            saved_crop_boxes: List[Tuple[int, int, int, int]] = []
            table_markdowns: List[str] = []
            page_text_boxes = _extract_pdf_text_boxes(
                page,
                repeated_watermark_candidates=repeated_watermark_candidates,
                dynamic_artifact_signatures=dynamic_page_artifact_signatures,
            )
            page_text_line_boxes = _extract_pdf_text_line_boxes(page, repeated_watermark_candidates=repeated_watermark_candidates)

            # Native table detection (for non-scanned PDFs)
            native_tables = []
            try:
                native_tables = page.find_tables()
            except Exception:
                native_tables = []

            top_semantic_constraints = _collect_top_semantic_constraints(
                page_text_line_boxes,
                layout_avoidance_constraints,
                native_text_boxes=page_text_boxes,
            )
            page_ocr_structure_blocks: List[Dict] = []
            text_structure_blocks: List[Dict] = []
            layout_text_supplement_blocks: List[Dict] = []
            for order_index, tb in enumerate(page_text_boxes, start=1):
                block = _make_text_structure_block(tb, page_number=page_num, order_index=order_index, page_render_w=int(page_render_w))
                if block is not None:
                    text_structure_blocks.append(block)
            blocks.extend(text_structure_blocks)
            # precompute page area using PIL for robust area-ratio checks
            try:
                _tmp_img_for_area = Image.open(io.BytesIO(jpg_bytes))
                page_area_local = float(_tmp_img_for_area.width) * float(_tmp_img_for_area.height)
            except Exception:
                page_area_local = None
            if _HAS_NUMPY and _HAS_CV2:
                try:
                    _page_arr_main = np.frombuffer(jpg_bytes, dtype=np.uint8)
                    page_cv_main = cv2.imdecode(_page_arr_main, cv2.IMREAD_COLOR)
                except Exception:
                    page_cv_main = None

            for lb in layout_blocks:
                try:
                    try:
                        print(f"[TRACE-1] Found candidate block: type={lb.get('type')}, source={lb.get('source', 'layout')}, box={lb.get('bbox')}")
                    except Exception:
                        pass
                    typ = str(lb.get('type', 'table')).lower()
                    lb_source = str(lb.get('source', 'layout')).lower()
                    # Explicitly skip text-like blocks (we already store their text via OCR)
                    if typ in SKIP_TEXT_TYPES:
                        text_region = str(lb.get('contentMarkdown') or '').strip()
                        if not text_region:
                            try:
                                x, y, w, h = lb.get('bbox', (0, 0, 0, 0))
                                if img is None:
                                    img = Image.open(io.BytesIO(jpg_bytes)).convert('RGB')
                                crop = img.crop((int(x), int(y), int(x) + int(w), int(y) + int(h)))
                                buf = io.BytesIO()
                                crop.save(buf, format='PNG', optimize=True)
                                text_region = _ocr_image_bytes(buf.getvalue(), ocr_engine=ocr_engine) or ''
                            except Exception:
                                text_region = ''
                        if text_region and re.search(r'(?<!\d)\d{1,2}\.\d{1,2}\.\d{1,3}(?!\d)', text_region):
                            try:
                                x, y, w, h = lb.get('bbox', (0, 0, 0, 0))
                                text_region_block = {
                                    'id': f'b_layout_text_{page_num}_{len(blocks) + 1}',
                                    'type': 'code',
                                    'language': 'markdown',
                                    'code': text_region,
                                    'pageNumber': int(page_num),
                                    'semanticRole': 'body',
                                    'bbox': _xywh_to_bbox_dict(int(x), int(y), int(w), int(h)),
                                    'readingOrder': int(10000 + len(blocks) + 1),
                                    'structureSource': 'layout_text_region',
                                }
                                layout_text_supplement_blocks.append(text_region_block)
                            except Exception:
                                pass
                        try:
                            print(f"[DEBUG] Skipping non-visual block type '{typ}' at {lb.get('bbox')}")
                        except Exception:
                            pass
                        try:
                            print(f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason=skip_text_type, box={lb.get('bbox')}")
                        except Exception:
                            pass
                        continue
                    # Strict whitelist: only allow visual blocks to trigger image saves
                    if typ not in ALLOWED_LAYOUT_TYPES:
                        try:
                            print(f"[DEBUG] Skipping unknown/unhandled block type '{typ}' at {lb.get('bbox')}")
                        except Exception:
                            pass
                        try:
                            print(f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason=not_allowed_type, box={lb.get('bbox')}")
                        except Exception:
                            pass
                        continue
                    x, y, w, h = lb.get('bbox', (0, 0, 0, 0))
                    if w <= 0 or h <= 0:
                        continue

                    try:
                        if img is None:
                            img = Image.open(io.BytesIO(jpg_bytes)).convert('RGB')
                        page_w_local, page_h_local = img.size if img is not None else (0, 0)
                    except Exception:
                        page_w_local, page_h_local = (0, 0)

                    if lb_source == 'red_annotation':
                        normalized_boxes = [
                            _trim_red_annotation_crop_box(
                                (int(x), int(y), int(w), int(h)),
                                int(page_w_local),
                                int(page_h_local),
                            )
                        ]
                    else:
                        candidate_boxes = [(int(x), int(y), int(w), int(h))]
                        if typ in ('figure', 'equation') and page_cv_main is not None:
                            try:
                                candidate_boxes = _refine_visual_bbox_from_pixels(page_cv_main, (int(x), int(y), int(w), int(h)), typ)
                            except Exception:
                                candidate_boxes = [(int(x), int(y), int(w), int(h))]
                        elif typ == 'table' and page_cv_main is not None:
                            try:
                                pre_ratio = 0.0
                                if page_area_local:
                                    pre_ratio = (float(w) * float(h)) / float(page_area_local)
                                if pre_ratio >= 0.10 or (int(w) >= 540 and int(h) >= 320):
                                    candidate_boxes = _decompose_compound_visual_bbox(page_cv_main, (int(x), int(y), int(w), int(h)), typ)
                            except Exception:
                                candidate_boxes = [(int(x), int(y), int(w), int(h))]
                        normalized_boxes = []

                        for cand_box in candidate_boxes:
                            try:
                                cx, cy, cw, ch = cand_box
                                anchor_box = (int(x), int(y), int(w), int(h))
                                debug_context = f"page={page_num} type={typ} source={lb_source} anchor={anchor_box}"
                                if typ in ('figure', 'equation'):
                                    cx, cy, cw, ch = _iteratively_refine_visual_bbox(
                                        page_cv_main,
                                        anchor_box,
                                        (int(cx), int(cy), int(cw), int(ch)),
                                        page_text_boxes,
                                        layout_avoidance_constraints,
                                        typ,
                                        int(page_w_local),
                                        int(page_h_local),
                                        page_area_local,
                                        debug_context=debug_context,
                                    )
                                    _trace_avoidance_debug(
                                        debug_context,
                                        f"stage=post_refine bbox={(int(cx), int(cy), int(cw), int(ch))} y_max={int(cy) + int(ch)}",
                                    )
                                    if page_w_local > 0 and page_h_local > 0:
                                        cx, cy, cw, ch = _expand_refined_figure_bbox_with_support_boxes(
                                            anchor_box,
                                            (int(cx), int(cy), int(cw), int(ch)),
                                            layout_support_boxes,
                                            int(page_w_local),
                                            int(page_h_local),
                                        )
                                        _trace_avoidance_debug(
                                            debug_context,
                                            f"stage=post_support_expand bbox={(int(cx), int(cy), int(cw), int(ch))} y_max={int(cy) + int(ch)}",
                                        )
                                elif page_w_local > 0 and page_h_local > 0:
                                    cx, cy, cw, ch = _clip_xywh_to_page(int(cx), int(cy), int(cw), int(ch), int(page_w_local), int(page_h_local))
                                if page_w_local > 0 and page_h_local > 0:
                                    pre_pad_box = (int(cx), int(cy), int(cw), int(ch))
                                    cx, cy, cw, ch = _pad_visual_bbox((cx, cy, cw, ch), typ, int(page_w_local), int(page_h_local))
                                    if typ in ('figure', 'equation'):
                                        cx, cy, cw, ch = _clamp_padded_figure_bbox_to_support_band(
                                            (int(cx), int(cy), int(cw), int(ch)),
                                            pre_pad_box,
                                            layout_support_boxes,
                                            int(page_w_local),
                                            int(page_h_local),
                                        )
                                        cx, cy, cw, ch = _clamp_padded_figure_bbox_to_avoidance_constraints(
                                            (int(cx), int(cy), int(cw), int(ch)),
                                            pre_pad_box,
                                            layout_avoidance_constraints,
                                            int(page_w_local),
                                            int(page_h_local),
                                            debug_context=debug_context,
                                        )
                                        cx, cy, cw, ch = _clamp_figure_bbox_to_top_semantic_ceiling(
                                            (int(cx), int(cy), int(cw), int(ch)),
                                            top_semantic_constraints,
                                            int(page_w_local),
                                            int(page_h_local),
                                            page_cv_main,
                                            debug_context=debug_context,
                                            phase='post_pad',
                                        )
                                    if typ in ('figure', 'equation'):
                                        _trace_avoidance_debug(
                                            debug_context,
                                            f"stage=post_pad y_max={int(cy) + int(ch)} bbox={(int(cx), int(cy), int(cw), int(ch))} pre={pre_pad_box}",
                                        )
                                if typ in ('figure', 'equation'):
                                    semantic_safety_floor = int(cy) + int(ch)
                                    cx, cy, cw, ch = _shrink_visual_bbox_to_blank_edges(
                                        page_cv_main,
                                        (int(cx), int(cy), int(cw), int(ch)),
                                        typ,
                                        page_text_boxes,
                                        layout_support_boxes,
                                        anchor_box,
                                        semantic_safety_floor=semantic_safety_floor,
                                        semantic_text_boxes=layout_avoidance_constraints,
                                        debug_context=debug_context,
                                    )
                                    cx, cy, cw, ch = _clamp_figure_bbox_to_top_semantic_ceiling(
                                        (int(cx), int(cy), int(cw), int(ch)),
                                        top_semantic_constraints,
                                        int(page_w_local),
                                        int(page_h_local),
                                        page_cv_main,
                                        debug_context=debug_context,
                                        phase='post_shrink',
                                    )
                                    _trace_avoidance_debug(
                                        debug_context,
                                        f"stage=post_shrink y_max={int(cy) + int(ch)} floor={semantic_safety_floor} bbox={(int(cx), int(cy), int(cw), int(ch))}",
                                    )
                                normalized_boxes.append((int(cx), int(cy), int(cw), int(ch)))
                            except Exception:
                                continue

                        if not normalized_boxes:
                            normalized_boxes = [(int(x), int(y), int(w), int(h))]
                        try:
                            normalized_boxes = _prune_overlapping_boxes(normalized_boxes)
                        except Exception:
                            pass
                        if typ in ('figure', 'equation'):
                            try:
                                normalized_boxes = _reconcile_figure_split_boxes(
                                    (int(x), int(y), int(w), int(h)),
                                    normalized_boxes,
                                    page_text_boxes,
                                    layout_support_boxes,
                                    page_cv_main,
                                )
                                normalized_boxes = _suppress_nested_figure_boxes(normalized_boxes)
                                normalized_boxes = _suppress_compound_parent_figure_boxes(normalized_boxes)
                                if page_w_local > 0 and page_h_local > 0:
                                    reclamped_boxes: List[Tuple[int, int, int, int]] = []
                                    for nbx in normalized_boxes:
                                        try:
                                            reclamped_boxes.append(
                                                _clamp_figure_bbox_to_top_semantic_ceiling(
                                                    nbx,
                                                    top_semantic_constraints,
                                                    int(page_w_local),
                                                    int(page_h_local),
                                                    page_cv_main,
                                                    debug_context=debug_context,
                                                    phase='post_reconcile',
                                                )
                                            )
                                        except Exception:
                                            reclamped_boxes.append(nbx)
                                    normalized_boxes = reclamped_boxes
                            except Exception:
                                pass
                            try:
                                debug_context = f"page={page_num} type={typ} source={lb_source} anchor={(int(x), int(y), int(w), int(h))}"
                                _trace_avoidance_debug(
                                    debug_context,
                                    f"stage=post_reconcile boxes={normalized_boxes}",
                                )
                            except Exception:
                                pass
                        if typ == 'table':
                            try:
                                normalized_boxes = _merge_stacked_table_boxes(normalized_boxes, page_w_local, page_h_local)
                                normalized_boxes = _prune_overlapping_boxes(normalized_boxes)
                            except Exception:
                                pass

                    for box_index, (x, y, w, h) in enumerate(normalized_boxes, start=1):
                    # Aspect-ratio noise filter
                        try:
                            if typ in ('figure', 'equation'):
                                debug_context = f"page={page_num} type={typ} source={lb_source} anchor={(int(lb.get('bbox', (0, 0, 0, 0))[0]), int(lb.get('bbox', (0, 0, 0, 0))[1]), int(lb.get('bbox', (0, 0, 0, 0))[2]), int(lb.get('bbox', (0, 0, 0, 0))[3]))}"
                                _trace_avoidance_debug(
                                    debug_context,
                                    f"stage=pre_save idx={box_index} y_max={int(y) + int(h)} bbox={(int(x), int(y), int(w), int(h))}",
                                )
                        except Exception:
                            pass
                        try:
                            if float(w) / float(h) > MAX_ASPECT_RATIO:
                                try:
                                    print(f"[DEBUG] Dropping block due to extreme aspect ratio {float(w)/float(h):.2f} at {(x,y,w,h)}")
                                except Exception:
                                    pass
                                try:
                                    print(f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason=aspect_ratio, ratio={float(w)/float(h):.2f}, box={(x,y,w,h)}")
                                except Exception:
                                    pass
                                continue
                        except Exception:
                            pass
                        # Area-ratio filter (local check as extra safety)
                        try:
                            if page_area_local:
                                block_area = float(w) * float(h)
                                block_ratio_local = block_area / page_area_local
                                if block_ratio_local < MIN_AREA_RATIO:
                                    try:
                                        print(f"[DEBUG] Dropping block due to small area ratio {block_ratio_local:.4f} at {(x,y,w,h)}")
                                    except Exception:
                                        pass
                                    try:
                                        print(f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason=small_area, area_ratio={block_ratio_local:.4f}, box={(x,y,w,h)}")
                                    except Exception:
                                        pass
                                    continue
                                if lb_source.startswith('cv_fallback') and block_ratio_local > CV_FALLBACK_MAX_AREA_RATIO:
                                    try:
                                        print(f"[DEBUG] Dropping cv_fallback block due to oversized area ratio {block_ratio_local:.4f} at {(x,y,w,h)}")
                                    except Exception:
                                        pass
                                    try:
                                        print(f"[TRACE-4] Final Decision for {typ}: keep_block=False, reason=oversized_cv_fallback_post, area_ratio={block_ratio_local:.4f}, box={(x,y,w,h)}")
                                    except Exception:
                                        pass
                                    continue
                        except Exception:
                            pass

                        try:
                            cand_box = (int(x), int(y), int(w), int(h))
                            if typ in ('figure', 'equation') and _is_nested_figure_fragment(cand_box, saved_crop_boxes):
                                try:
                                    print(f"[DEBUG] Dropping nested figure fragment at {cand_box}")
                                except Exception:
                                    pass
                                continue
                            if _is_duplicate_crop_candidate(cand_box, saved_crop_boxes):
                                try:
                                    print(f"[DEBUG] Dropping near-duplicate crop candidate at {cand_box}")
                                except Exception:
                                    pass
                                continue
                        except Exception:
                            pass

                        # lazy-load PIL image
                        if img is None:
                            try:
                                img = Image.open(io.BytesIO(jpg_bytes)).convert('RGB')
                            except Exception:
                                img = None
                        if img is None:
                            continue
                        try:
                            print(f"[DEBUG] Cropping identified {typ} part={box_index}/{len(normalized_boxes)} at {(x,y,w,h)}")
                        except Exception:
                            pass
                        crop = img.crop((x, y, x + w, y + h))
                        buf = io.BytesIO()
                        crop.save(buf, format='PNG', optimize=True)
                        crop_bytes = buf.getvalue()

                        # Text-density check: for `table` blocks run OCR and count non-blank lines.
                        ocr_text = ''
                        table_rows = None
                        try:
                            if typ == 'table':
                                try:
                                    ocr_text = _ocr_image_bytes(
                                        crop_bytes,
                                        ocr_engine=ocr_engine,
                                        repeated_watermark_candidates=repeated_watermark_candidates,
                                    )
                                except Exception:
                                    ocr_text = ''
                                try:
                                    non_blank_lines = [ln for ln in (ocr_text or '').splitlines() if ln.strip()]
                                except Exception:
                                    non_blank_lines = []
                                try:
                                    total_chars = sum(len(re.sub(r'\s+', '', ln)) for ln in non_blank_lines)
                                    avg_line_len = (float(total_chars) / float(len(non_blank_lines))) if non_blank_lines else 0.0
                                except Exception:
                                    avg_line_len = 0.0
                                try:
                                    table_rows = _text_to_table_rows(ocr_text)
                                except Exception:
                                    table_rows = None
                                try:
                                    max_cols = max((len(r) for r in (table_rows or [])), default=0)
                                except Exception:
                                    max_cols = 0

                                # Reject table crops that are actually paragraph/caption text blocks.
                                if len(non_blank_lines) >= TABLE_CAPTION_MIN_LINES and avg_line_len >= TABLE_CAPTION_LINE_LEN and max_cols < 2:
                                    try:
                                        print(
                                            f"[DEBUG] Dropping caption-like table block at {(x,y,w,h)}: "
                                            f"lines={len(non_blank_lines)}, avg_line_len={avg_line_len:.2f}, max_cols={max_cols}"
                                        )
                                    except Exception:
                                        pass
                                    continue
                        except Exception:
                            # fallback: if OCR fails, proceed to save conservatively
                            ocr_text = ''
                            table_rows = None

                        # Passed density/structure checks; now persist the crop to disk
                        crop_index += 1
                        fname, abs_path, uri = _save_crop_and_uri(page_num, crop_index, crop_bytes, assets_dir, file_id, ext='png')
                        if uri:
                            if typ == 'table':
                                block: Dict = {
                                    'type': 'table',
                                    'imageUri': uri,
                                    'caption': f'Table {crop_index}',
                                    'contentMarkdown': ocr_text,
                                    'pageNumber': int(page_num),
                                    'semanticRole': 'table',
                                    'bbox': _xywh_to_bbox_dict(int(x), int(y), int(w), int(h)),
                                    'confidence': lb.get('confidence'),
                                    'structureSource': (
                                        'red_annotation_visual_crop'
                                        if lb_source == 'red_annotation'
                                        else 'pdf_native_visual_crop'
                                    ),
                                }

                                # Try to find a matching native table to get high-fidelity structure
                                try:
                                    for nt in native_tables:
                                        nt_rect = nt.bbox # (x0, y0, x1, y1)
                                        # Check overlap with the current crop box
                                        # (x, y, w, h) are the crop coordinates
                                        overlap = max(0, min(x + w, nt_rect[2]) - max(x, nt_rect[0])) * \
                                                  max(0, min(y + h, nt_rect[3]) - max(y, nt_rect[1]))
                                        nt_area = (nt_rect[2] - nt_rect[0]) * (nt_rect[3] - nt_rect[1])
                                        if nt_area > 0 and (overlap / nt_area) > 0.85:
                                            # We found the original PDF table for this crop!
                                            nt_cells = _extract_native_table_cells(nt)
                                            if nt_cells:
                                                block['table_cells'] = nt_cells
                                                block['structureSource'] = 'pdf_native_table_finder'
                                            break
                                except Exception:
                                    pass

                                if 'table_structure' in lb:
                                    block['table_structure'] = lb['table_structure']
                                if table_rows:
                                    block['table_rows'] = table_rows
                                    try:
                                        md = _table_rows_to_markdown(table_rows)
                                        if md:
                                            table_markdowns.append(md)
                                    except Exception:
                                        pass
                            else:
                                embedded_caption = ''
                                if typ == 'figure':
                                    embedded_caption = _extract_embedded_visual_caption_from_image_bytes(
                                        crop_bytes,
                                        'figure',
                                        ocr_engine=ocr_engine,
                                        repeated_watermark_candidates=repeated_watermark_candidates,
                                    )
                                block = {
                                    'type': typ,
                                    'imageUri': uri,
                                    'caption': embedded_caption or f'{typ.title()} {crop_index}',
                                    'pageNumber': int(page_num),
                                    'semanticRole': 'figure',
                                    'bbox': _xywh_to_bbox_dict(int(x), int(y), int(w), int(h)),
                                    'confidence': lb.get('confidence'),
                                    'structureSource': (
                                        'red_annotation_visual_crop'
                                        if lb_source == 'red_annotation'
                                        else 'pdf_native_visual_crop'
                                    ),
                                }
                            blocks.append(block)
                            try:
                                saved_crop_boxes.append((int(x), int(y), int(w), int(h)))
                            except Exception:
                                pass
                except Exception:
                    continue

            # Hard fallback: ensure each processed page has at least one visual crop
            # written to assets (helps pages where layout detector yields no visual block).
            try:
                has_visual_crop = any(((b.get('type') or '').lower() in ALLOWED_LAYOUT_TYPES) for b in blocks)
            except Exception:
                has_visual_crop = False
            try:
                page_visual_signal = _page_has_structural_visual_signal(jpg_bytes)
            except Exception:
                page_visual_signal = False

            try:
                accepted_visual_count = sum(1 for b in blocks if ((b.get('type') or '').lower() in ALLOWED_LAYOUT_TYPES))
            except Exception:
                accepted_visual_count = len(saved_crop_boxes)

            if (not use_red_annotation_mode) and (not embedded) and accepted_visual_count <= 2:
                try:
                    supplemental_boxes = _detect_table_bboxes_from_image_bytes(jpg_bytes, debug=False)
                    try:
                        supplemental_boxes = _prune_overlapping_boxes(supplemental_boxes)
                    except Exception:
                        pass

                    filtered_supplemental_boxes: List[Tuple[int, int, int, int]] = []
                    for sbx in supplemental_boxes:
                        try:
                            sx, sy, sw, sh = [int(v) for v in sbx]
                            if sw <= 0 or sh <= 0:
                                continue
                            if page_area_local:
                                area_ratio = (float(sw) * float(sh)) / float(page_area_local)
                                if area_ratio < max(MIN_AREA_RATIO, 0.02):
                                    continue
                                if area_ratio > CV_FALLBACK_MAX_AREA_RATIO:
                                    continue
                            if min(sw, sh) < 120:
                                continue
                            if page_area_local and area_ratio < 0.03 and min(sw, sh) < 170:
                                continue
                            try:
                                shape_ratio = max(float(sw) / float(max(1, sh)), float(sh) / float(max(1, sw)))
                                if shape_ratio > 5.8:
                                    continue
                            except Exception:
                                pass
                            if _is_duplicate_crop_candidate((sx, sy, sw, sh), saved_crop_boxes):
                                continue

                            contained = False
                            for kept in saved_crop_boxes:
                                try:
                                    kept_area = float(max(1, kept[2] * kept[3]))
                                    cand_area = float(max(1, sw * sh))
                                    if cand_area / kept_area <= 0.72 and _bbox_inside_xywh((sx, sy, sw, sh), kept, margin=18):
                                        contained = True
                                        break
                                except Exception:
                                    continue
                            if contained:
                                continue

                            filtered_supplemental_boxes.append((sx, sy, sw, sh))
                        except Exception:
                            continue

                    if accepted_visual_count == 0 and len(filtered_supplemental_boxes) >= 2:
                        merged_supplemental_boxes: List[Tuple[int, int, int, int]] = []
                        for sx, sy, sw, sh in sorted(
                            filtered_supplemental_boxes,
                            key=lambda b: (((b[0] + (b[2] // 2)) // 220), b[1], b[0]),
                        ):
                            if not merged_supplemental_boxes:
                                merged_supplemental_boxes.append((sx, sy, sw, sh))
                                continue

                            mx, my, mw, mh = merged_supplemental_boxes[-1]
                            try:
                                overlap_w = max(0, min(mx + mw, sx + sw) - max(mx, sx))
                                x_overlap_ratio = float(overlap_w) / float(max(1, min(mw, sw)))
                                width_ratio = float(min(mw, sw)) / float(max(mw, sw))
                                vertical_gap = sy - (my + mh)
                                merged_h = max(my + mh, sy + sh) - min(my, sy)
                                if (
                                    x_overlap_ratio >= 0.72
                                    and width_ratio >= 0.72
                                    and 0 <= vertical_gap <= max(90, int(min(mh, sh) * 0.32))
                                    and merged_h <= 640
                                ):
                                    merged_supplemental_boxes[-1] = (
                                        min(mx, sx),
                                        min(my, sy),
                                        max(mx + mw, sx + sw) - min(mx, sx),
                                        merged_h,
                                    )
                                    continue
                            except Exception:
                                pass

                            merged_supplemental_boxes.append((sx, sy, sw, sh))

                        filtered_supplemental_boxes = merged_supplemental_boxes

                    try:
                        print(
                            f"[TRACE-4] Sparse-page supplement probe: page={page_num}, "
                            f"accepted={accepted_visual_count}, candidates={len(filtered_supplemental_boxes)}"
                        )
                    except Exception:
                        pass

                    min_candidate_count = 3 if accepted_visual_count == 0 else max(2, accepted_visual_count + 1)
                    if len(filtered_supplemental_boxes) >= min_candidate_count:
                        if not page_visual_signal and accepted_visual_count == 0:
                            try:
                                print(
                                    f"[TRACE-4] Skip sparse-page supplement for page {page_num}: "
                                    f"weak page structural signal and no accepted visual blocks"
                                )
                            except Exception:
                                pass
                            filtered_supplemental_boxes = []
                    if len(filtered_supplemental_boxes) >= min_candidate_count:
                        if img is None:
                            try:
                                img = Image.open(io.BytesIO(jpg_bytes)).convert('RGB')
                            except Exception:
                                img = None

                        if img is not None:
                            try:
                                supplemental_page_w = int(img.width)
                            except Exception:
                                supplemental_page_w = 0
                            try:
                                supplemental_page_h = int(img.height)
                            except Exception:
                                supplemental_page_h = 0

                            prepared_supplemental_candidates: List[Dict] = []
                            for sx, sy, sw, sh in sorted(filtered_supplemental_boxes, key=lambda b: (b[1], b[0])):
                                try:
                                    if _is_duplicate_crop_candidate((sx, sy, sw, sh), saved_crop_boxes):
                                        continue

                                    crop = img.crop((sx, sy, sx + sw, sy + sh))
                                    buf = io.BytesIO()
                                    crop.save(buf, format='PNG', optimize=True)
                                    crop_bytes = buf.getvalue()

                                    supp_text = ''
                                    table_rows = None
                                    non_blank_lines: List[str] = []
                                    avg_line_len = 0.0
                                    max_cols = 0
                                    short_line_count = 0
                                    try:
                                        supp_text = _ocr_image_bytes(
                                            crop_bytes,
                                            ocr_engine=ocr_engine,
                                            repeated_watermark_candidates=repeated_watermark_candidates,
                                        ) or ''
                                        non_blank_lines = [ln for ln in supp_text.splitlines() if ln.strip()]
                                        if non_blank_lines:
                                            total_chars = sum(len(re.sub(r'\s+', '', ln)) for ln in non_blank_lines)
                                            avg_line_len = float(total_chars) / float(len(non_blank_lines))
                                            short_line_count = sum(
                                                1
                                                for ln in non_blank_lines
                                                if len(re.sub(r'\s+', '', ln)) <= 10
                                            )
                                        table_rows = _text_to_table_rows(supp_text)
                                        max_cols = max((len(r) for r in (table_rows or [])), default=0)
                                    except Exception:
                                        supp_text = ''
                                        table_rows = None
                                        non_blank_lines = []
                                        avg_line_len = 0.0
                                        max_cols = 0
                                        short_line_count = 0

                                    table_like_hint = (
                                        (max_cols >= 2 and len(non_blank_lines) >= 4)
                                        or (
                                            len(non_blank_lines) >= 10
                                            and avg_line_len <= 10.5
                                            and short_line_count >= max(6, int(len(non_blank_lines) * 0.55))
                                        )
                                    )

                                    supp_typ = 'table' if table_like_hint else 'figure'
                                    if supp_typ == 'table':
                                        if len(non_blank_lines) >= TABLE_CAPTION_MIN_LINES and avg_line_len >= TABLE_CAPTION_LINE_LEN and max_cols < 2:
                                            continue
                                    elif len(non_blank_lines) >= 18 and avg_line_len >= 14.0:
                                        continue

                                    final_box = (int(sx), int(sy), int(sw), int(sh))
                                    if supp_typ == 'figure':
                                        fx, fy, fw, fh = final_box
                                        anchor_box = final_box
                                        try:
                                            if supplemental_page_h > 0:
                                                fx, fy, fw, fh = _trim_visual_bbox_with_text_boxes((fx, fy, fw, fh), page_text_boxes, supp_typ, supplemental_page_h)
                                        except Exception:
                                            pass
                                        try:
                                            use_image_trim = False
                                            if int(fh) >= int(float(fw) * 0.62):
                                                use_image_trim = True
                                            elif page_area_local:
                                                cand_ratio = (float(fw) * float(fh)) / float(page_area_local)
                                                if cand_ratio >= 0.12:
                                                    use_image_trim = True
                                            if use_image_trim and page_cv_main is not None:
                                                fx, fy, fw, fh = _trim_visual_bbox_with_image_rows(page_cv_main, (fx, fy, fw, fh), supp_typ)
                                        except Exception:
                                            pass
                                        try:
                                            if supplemental_page_w > 0 and supplemental_page_h > 0:
                                                fx, fy, fw, fh = _clamp_visual_bbox_growth(anchor_box, (fx, fy, fw, fh), supp_typ, supplemental_page_w, supplemental_page_h)
                                                fx, fy, fw, fh = _pad_visual_bbox((fx, fy, fw, fh), supp_typ, supplemental_page_w, supplemental_page_h)
                                        except Exception:
                                            pass
                                        final_box = (int(fx), int(fy), int(max(1, fw)), int(max(1, fh)))
                                        try:
                                            final_area_ratio = (float(final_box[2]) * float(final_box[3])) / float(max(1, sw * sh))
                                            final_width_ratio = float(final_box[2]) / float(max(1, sw))
                                            if final_area_ratio < 0.58 or final_width_ratio < 0.62:
                                                continue
                                        except Exception:
                                            pass

                                    if _is_duplicate_crop_candidate(final_box, saved_crop_boxes):
                                        continue

                                    prepared_supplemental_candidates.append({
                                        'type': supp_typ,
                                        'box': final_box,
                                        'contentMarkdown': supp_text,
                                        'table_rows': table_rows,
                                        'allow_sidecar_merge': (supp_typ == 'figure' and not table_like_hint),
                                    })
                                except Exception:
                                    continue

                            figure_candidate_boxes = [
                                cand['box']
                                for cand in prepared_supplemental_candidates
                                if (cand.get('type') or '').lower() == 'figure' and bool(cand.get('allow_sidecar_merge', True))
                            ]
                            try:
                                merged_figure_boxes = _merge_sidecar_figure_boxes(
                                    figure_candidate_boxes,
                                    int(img.width),
                                    supplemental_page_h,
                                )
                            except Exception:
                                merged_figure_boxes = list(figure_candidate_boxes)

                            try:
                                if len(merged_figure_boxes) != len(figure_candidate_boxes):
                                    print(
                                        f"[TRACE-4] Sidecar figure merge: page={page_num}, "
                                        f"before={len(figure_candidate_boxes)}, after={len(merged_figure_boxes)}"
                                    )
                            except Exception:
                                pass

                            final_supplemental_candidates: List[Dict] = []
                            for figure_box in merged_figure_boxes:
                                final_supplemental_candidates.append({
                                    'type': 'figure',
                                    'box': figure_box,
                                })
                            for cand in prepared_supplemental_candidates:
                                if (cand.get('type') or '').lower() == 'figure' and not bool(cand.get('allow_sidecar_merge', True)):
                                    final_supplemental_candidates.append(cand)
                            for cand in prepared_supplemental_candidates:
                                if (cand.get('type') or '').lower() == 'table':
                                    final_supplemental_candidates.append(cand)

                            for cand in sorted(final_supplemental_candidates, key=lambda c: (c['box'][1], c['box'][0])):
                                try:
                                    if len(saved_crop_boxes) >= 6:
                                        break

                                    final_box = cand.get('box')
                                    if not final_box or _is_duplicate_crop_candidate(final_box, saved_crop_boxes):
                                        continue

                                    fx, fy, fw, fh = [int(v) for v in final_box]
                                    final_crop = img.crop((fx, fy, fx + fw, fy + fh))
                                    final_buf = io.BytesIO()
                                    final_crop.save(final_buf, format='PNG', optimize=True)
                                    final_bytes = final_buf.getvalue()

                                    crop_index += 1
                                    supp_name, supp_abs, supp_uri = _save_crop_and_uri(page_num, crop_index, final_bytes, assets_dir, file_id, ext='png')
                                    if not supp_uri:
                                        crop_index -= 1
                                        continue

                                    supp_typ = str(cand.get('type') or 'figure').lower()
                                    if supp_typ == 'table':
                                        supp_text = str(cand.get('contentMarkdown') or '')
                                        table_rows = cand.get('table_rows')
                                        supp_block = {'type': 'table', 'imageUri': supp_uri, 'caption': f'Table {crop_index}', 'contentMarkdown': supp_text, 'semanticRole': 'table'}
                                        if table_rows:
                                            supp_block['table_rows'] = table_rows
                                            try:
                                                md = _table_rows_to_markdown(table_rows)
                                                if md:
                                                    table_markdowns.append(md)
                                            except Exception:
                                                pass
                                    else:
                                        embedded_caption = _extract_embedded_visual_caption_from_image_bytes(
                                            final_bytes,
                                            'figure',
                                            ocr_engine=ocr_engine,
                                            repeated_watermark_candidates=repeated_watermark_candidates,
                                        )
                                        supp_block = {'type': 'figure', 'imageUri': supp_uri, 'caption': embedded_caption or f'Figure {crop_index}', 'semanticRole': 'figure'}

                                    blocks.append(supp_block)
                                    saved_crop_boxes.append((fx, fy, fw, fh))
                                    try:
                                        print(f"[TRACE-4] Supplemental sparse-page crop saved: {supp_abs}, type={supp_typ}, box={(fx, fy, fw, fh)}")
                                    except Exception:
                                        pass
                                except Exception:
                                    continue
                except Exception:
                    pass

            try:
                has_visual_crop = any(((b.get('type') or '').lower() in ALLOWED_LAYOUT_TYPES) for b in blocks)
            except Exception:
                has_visual_crop = False

            if (not use_red_annotation_mode) and not has_visual_crop and (not embedded):
                try:
                    fb_saved = False
                    fb_boxes = _detect_table_bboxes_from_image_bytes(jpg_bytes, debug=False)
                    try:
                        fb_boxes = _prune_overlapping_boxes(fb_boxes)
                    except Exception:
                        pass

                    best_box = None
                    best_area = -1.0
                    for bx in fb_boxes:
                        try:
                            fx, fy, fw, fh = bx
                            if fw <= 0 or fh <= 0:
                                continue
                            if page_area_local:
                                fb_ratio = (float(fw) * float(fh)) / page_area_local
                                if fb_ratio < max(MIN_AREA_RATIO, 0.03):
                                    continue
                                if fb_ratio > CV_FALLBACK_MAX_AREA_RATIO:
                                    continue
                            try:
                                if float(fw) / float(fh) > 8.0:
                                    continue
                            except Exception:
                                pass
                            area = float(fw) * float(fh)
                            if area > best_area:
                                best_area = area
                                best_box = (int(fx), int(fy), int(fw), int(fh))
                        except Exception:
                            continue

                    if best_box is not None:
                        if img is None:
                            try:
                                img = Image.open(io.BytesIO(jpg_bytes)).convert('RGB')
                            except Exception:
                                img = None
                        if img is not None:
                            bx, by, bw, bh = best_box
                            fb_crop_bottom = by + bh
                            try:
                                if page_area_local:
                                    fb_best_ratio = (float(bw) * float(bh)) / page_area_local
                                else:
                                    fb_best_ratio = 0.0
                            except Exception:
                                fb_best_ratio = 0.0

                            # Oversized fallback regions often include trailing paragraph text;
                            # trim lower area to keep diagram-focused content.
                            if fb_best_ratio >= 0.60 and bh >= 900:
                                try:
                                    fb_crop_bottom = by + int(bh * 0.82)
                                except Exception:
                                    fb_crop_bottom = by + bh

                            fb_crop = img.crop((bx, by, bx + bw, fb_crop_bottom))
                            fb_buf = io.BytesIO()
                            fb_crop.save(fb_buf, format='PNG', optimize=True)
                            fb_bytes = fb_buf.getvalue()
                            # Avoid fallback crops that are still mostly paragraph text.
                            fb_text = ''
                            fb_lines = []
                            fb_avg_line_len = 0.0
                            try:
                                fb_text = _ocr_image_bytes(
                                    fb_bytes,
                                    ocr_engine=ocr_engine,
                                    repeated_watermark_candidates=repeated_watermark_candidates,
                                ) or ''
                                fb_lines = [ln for ln in fb_text.splitlines() if ln.strip()]
                                if fb_lines:
                                    fb_total_chars = sum(len(re.sub(r'\s+', '', ln)) for ln in fb_lines)
                                    fb_avg_line_len = float(fb_total_chars) / float(len(fb_lines))
                            except Exception:
                                fb_text = ''
                                fb_lines = []
                                fb_avg_line_len = 0.0

                            fb_table_rows = _text_to_table_rows(fb_text) if fb_text else None
                            fb_max_cols = max((len(r) for r in (fb_table_rows or [])), default=0)
                            fb_has_tabular_text = bool(fb_table_rows) and len(fb_table_rows) >= 2 and fb_max_cols >= 2
                            fb_h_long, fb_v_long, fb_intersections = _line_evidence_from_crop_bytes(fb_bytes)
                            fb_strong_grid = (
                                fb_h_long >= FALLBACK_MIN_H_LONG
                                and fb_v_long >= FALLBACK_MIN_V_LONG
                                and fb_intersections >= FALLBACK_MIN_INTERSECTIONS
                            )
                            fb_medium_grid_with_text = (
                                fb_has_tabular_text
                                and fb_h_long >= max(TABLE_MIN_H_LONG, 3)
                                and fb_v_long >= max(TABLE_MIN_V_LONG, 2)
                                and fb_intersections >= 6
                            )

                            if (not page_visual_signal) and not (fb_strong_grid or fb_medium_grid_with_text):
                                try:
                                    print(
                                        f"[TRACE-4] Skip bounded fallback for page {page_num}: "
                                        f"weak page/crop table signal h_long={fb_h_long}, v_long={fb_v_long}, "
                                        f"intersections={fb_intersections}, tabular_text={fb_has_tabular_text}"
                                    )
                                except Exception:
                                    pass
                            elif len(fb_lines) >= 18 and fb_avg_line_len >= 14.0:
                                try:
                                    print(f"[TRACE-4] Skip bounded fallback for page {page_num}: text-dense candidate lines={len(fb_lines)}, avg_line_len={fb_avg_line_len:.2f}")
                                except Exception:
                                    pass
                            elif len(fb_lines) >= 8 and fb_avg_line_len >= 16.5 and not (fb_strong_grid or fb_medium_grid_with_text):
                                try:
                                    print(
                                        f"[TRACE-4] Skip bounded fallback for page {page_num}: paragraph-like candidate "
                                        f"lines={len(fb_lines)}, avg_line_len={fb_avg_line_len:.2f}, "
                                        f"h_long={fb_h_long}, v_long={fb_v_long}, intersections={fb_intersections}"
                                    )
                                except Exception:
                                    pass
                            else:
                                fb_box = (int(bx), int(by), int(bw), max(1, int(fb_crop_bottom - by)))
                                if _is_duplicate_crop_candidate(fb_box, saved_crop_boxes):
                                    try:
                                        print(f"[TRACE-4] Skip bounded fallback for page {page_num}: duplicate candidate box={fb_box}")
                                    except Exception:
                                        pass
                                else:
                                    fb_idx = max(1, crop_index + 1)
                                    fb_name, fb_abs, fb_uri = _save_crop_and_uri(page_num, fb_idx, fb_bytes, assets_dir, file_id, ext='png')
                                    if fb_uri:
                                        blocks.append({'type': 'table', 'imageUri': fb_uri, 'caption': 'Fallback Table', 'semanticRole': 'table'})
                                        saved_crop_boxes.append(fb_box)
                                        fb_saved = True
                                        try:
                                            print(f"[TRACE-4] Fallback bounded crop saved: {fb_abs}, box={(bx,by,bw,bh)}")
                                        except Exception:
                                            pass

                    if not fb_saved:
                        try:
                            print(f"[TRACE-4] Skip bounded fallback for page {page_num}: no qualified fallback boxes")
                        except Exception:
                            pass
                except Exception:
                    pass
            elif not has_visual_crop:
                try:
                    print(f"[TRACE-4] Skip fallback for page {page_num}: embedded={bool(embedded)}, page_visual_signal={page_visual_signal}")
                except Exception:
                    pass

            # Only save a full page image file if there are visual elements (tables/figures/equations) or embedded images
            try:
                visual_blocks_present = any(((b.get('type') or '').lower() in ALLOWED_LAYOUT_TYPES) for b in blocks)
            except Exception:
                visual_blocks_present = False
            save_page_image = bool(keep_aux_images and (bool(embedded) or visual_blocks_present))
            page_img_name = f'page_{page_num}.jpg'
            page_img_path = os.path.join(assets_dir, page_img_name)
            page_uri = ''
            if save_page_image:
                try:
                    with open(page_img_path, 'wb') as f:
                        f.write(jpg_bytes)
                    _print_utf(f"[SUCCESS] Saved page image: {page_img_path}")
                    page_uri = f'file:///android_asset/kb/{file_id}/截图/{page_img_name}'
                    # include full page image block at the top
                    if keep_aux_images:
                        blocks.insert(0, {'type': 'image', 'imageUri': page_uri, 'caption': f'Page {page_num}'})
                except Exception as e:
                    _print_utf(f"[ERROR] Failed to save page image {page_img_path}: {e}")

            # also OCR the full page image (forced OCR requirement)
            page_ocr_text, page_ocr_structure_blocks = _ocr_image_bytes_with_structure(
                jpg_bytes,
                ocr_engine=ocr_engine,
                repeated_watermark_candidates=repeated_watermark_candidates,
            )
            if red_body_exclusion_regions and page_ocr_structure_blocks:
                page_ocr_structure_blocks = _filter_ocr_callouts_inside_visual_crops(
                    page_ocr_structure_blocks,
                    red_body_exclusion_regions,
                )
                page_ocr_text = _rebuild_plain_text_from_text_boxes(page_ocr_structure_blocks)
            if page_ocr_structure_blocks:
                ocr_text_structure_blocks: List[Dict] = []
                for order_index, tb in enumerate(page_ocr_structure_blocks, start=1):
                    block = _make_text_structure_block(tb, page_number=page_num, order_index=order_index, page_render_w=int(page_render_w))
                    if block is not None:
                        ocr_text_structure_blocks.append(block)
                native_clause_count = _count_decimal_clause_text_blocks(text_structure_blocks)
                ocr_clause_count = _count_decimal_clause_text_blocks(ocr_text_structure_blocks)
                use_ocr_structure = (
                    bool(ocr_text_structure_blocks)
                    and (not text_structure_blocks or ocr_clause_count > native_clause_count)
                )
                if use_ocr_structure:
                    text_structure_blocks = ocr_text_structure_blocks
                    blocks = text_structure_blocks + [
                        b
                        for b in blocks
                        if isinstance(b, dict)
                        and str(b.get('type') or '').strip().lower() not in {'code', 'text', 'paragraph'}
                    ]
                elif ocr_text_structure_blocks:
                    native_clause_labels = _extract_decimal_clause_labels_from_blocks(text_structure_blocks)
                    for ocr_block in ocr_text_structure_blocks:
                        ocr_labels = _extract_decimal_clause_labels_from_blocks([ocr_block])
                        if not ocr_labels or ocr_labels.issubset(native_clause_labels):
                            continue
                        supplement = copy.deepcopy(ocr_block)
                        supplement['structureSource'] = 'page_ocr_structure_supplement'
                        text_structure_blocks.append(supplement)
                        blocks.append(supplement)
                        native_clause_labels.update(ocr_labels)
            if layout_text_supplement_blocks:
                existing_clause_labels = _extract_decimal_clause_labels_from_blocks(blocks)
                native_clause_labels_to_replace = set()
                native_text_removal_regions: List[Dict[str, Any]] = []
                layout_text_supplements_applied = False
                for supplement in layout_text_supplement_blocks:
                    supplement_parts = _split_inline_decimal_clause_blocks([supplement])
                    missing_part_indexes: List[int] = []
                    for part_index, supplement_part in enumerate(supplement_parts):
                        supplement_labels = _extract_decimal_clause_labels_from_blocks([supplement_part])
                        if supplement_labels and not supplement_labels.issubset(existing_clause_labels):
                            missing_part_indexes.append(part_index)
                    if not missing_part_indexes:
                        continue
                    max_missing_part_index = max(missing_part_indexes)
                    if max_missing_part_index > 0 and isinstance(supplement.get('bbox'), dict):
                        native_text_removal_regions.append(supplement.get('bbox'))
                    for part_index, supplement_part in enumerate(supplement_parts):
                        if part_index > max_missing_part_index:
                            continue
                        supplement_labels = _extract_decimal_clause_labels_from_blocks([supplement_part])
                        if not supplement_labels:
                            continue
                        if supplement_labels.issubset(existing_clause_labels):
                            native_clause_labels_to_replace.update(supplement_labels)
                        supplement_copy = copy.deepcopy(supplement_part)
                        text_structure_blocks.append(supplement_copy)
                        blocks.append(supplement_copy)
                        layout_text_supplements_applied = True
                        existing_clause_labels.update(supplement_labels)
                if native_clause_labels_to_replace:
                    blocks = [
                        block
                        for block in blocks
                        if not (
                            isinstance(block, dict)
                            and str(block.get('structureSource') or '').strip().lower() == 'pdf_native_text_box'
                            and (
                                _extract_decimal_clause_labels_from_blocks([block]) & native_clause_labels_to_replace
                                or any(
                                    isinstance(block.get('bbox'), dict)
                                    and (
                                        int(region.get('top', 0))
                                        <= (
                                            int(block['bbox'].get('top', 0))
                                            + int(block['bbox'].get('bottom', 0))
                                        ) // 2
                                        <= int(region.get('bottom', 0))
                                    )
                                    for region in native_text_removal_regions
                                )
                            )
                        )
                    ]
            else:
                layout_text_supplements_applied = False
            # Strategy: prefer OCR when it contains substantive content; but detect watermarks or too-short OCR
            pdf_text = (page_text or '').strip()
            ocr_text = (page_ocr_text or '').strip()
            content_choice = ''
            page_ocr_structure_blocks = _filter_repeated_watermark_dicts(
                page_ocr_structure_blocks,
                repeated_watermark_candidates,
            )
            try:
                if (('学兔兔' in ocr_text) or (len(ocr_text) > 0 and len(ocr_text) < 10)) and pdf_text:
                    print('[PROBE-CHOICE] Watermark detected or OCR too short, using PDF text fallback.')
                    content_choice = pdf_text if len(pdf_text) >= len(ocr_text) else (ocr_text or pdf_text)
                else:
                    content_choice = ocr_text or pdf_text or ''
            except Exception:
                content_choice = ocr_text or pdf_text or ''
            content_choice = _filter_repeated_watermark_text(content_choice, repeated_watermark_candidates)

            # Run universal cleaning to remove known watermarks, URLs and tidy whitespace
            try:
                content_md = universal_clean_text(content_choice)
            except Exception:
                content_md = (content_choice or '')
            # Append any table markdowns discovered during layout cropping
            try:
                if table_markdowns:
                    content_md = (content_md or '') + '\n\n' + '\n\n'.join(table_markdowns)
            except Exception:
                pass
            try:
                print(f"[PROBE] Final content to save (len): {len(content_md)}")
            except Exception:
                pass
            try:
                print(f"[PROBE-SAVE] Final String Sample: {content_md[:100]}")
            except Exception:
                pass

            try:
                content_clause_labels = _extract_decimal_clause_labels_from_blocks([{'text': content_md}])
                if content_clause_labels and not layout_text_supplements_applied:
                    text_fallback_block = {
                        'id': f'b_page_ocr_text_{page_num}',
                        'type': 'code',
                        'language': 'markdown',
                        'code': content_md,
                        'pageNumber': int(page_num),
                        'semanticRole': 'body',
                        'bbox': _xywh_to_bbox_dict(0, 0, 0, 0),
                        'readingOrder': 1,
                        'structureSource': 'page_ocr_text_supplement',
                    }
                    blocks = [
                        b
                        for b in blocks
                        if isinstance(b, dict)
                        and (
                            str(b.get('type') or '').strip().lower() not in {'code', 'text', 'paragraph'}
                            or str(b.get('structureSource') or '').strip().lower() in {
                                'layout_text_region',
                                'page_ocr_structure_supplement',
                            }
                        )
                    ]
                    blocks.append(text_fallback_block)
            except Exception:
                pass

            blocks = _assign_page_structure_order(blocks, page_number=page_num)

            entry = {
                'entryId': f'pdf_native::page_{page_num}',
                'jobTitle': f'page_{page_num}',
                'pageNumber': page_num,
                'contentMarkdown': content_md[:2000],
                'contentNormalized': (content_md or '')[:2000],
                'blocks': blocks,
                'unitName': '未分类',
                'position': len(entries) + 1,
                'kind': 'text',
                'renderWidth': int(page_render_w),
                'renderHeight': int(page_render_h),
                'pdfWidth': float(pdf_w),
                'pdfHeight': float(pdf_h),
                'pdfOffsetX': float(pdf_offset_x),
                'pdfOffsetY': float(pdf_offset_y),
                'pdfRotation': int(pdf_rotation),
                'annotationContext': {
                    'blueHeadingHints': page_blue_heading_hints,
                    'redBodyExclusionApplied': bool(red_body_exclusion_regions),
                    'blueAnnotationBoxCount': len(blue_heading_boxes),
                    'redExclusionRegionCount': len(red_body_exclusion_regions),
                },
            }
            entries.append(entry)

            # Relaxed policy: include any page that has visible text into 'pages'
            try:
                has_text = bool((entry.get('contentMarkdown') or '').strip())
            except Exception:
                has_text = False
            if has_text:
                page_obj = {
                    'pageNumber': entry['pageNumber'],
                    'entryId': entry['entryId'],
                    'jobTitle': entry['jobTitle'],
                    'contentMarkdown': entry['contentMarkdown'],
                    'contentNormalized': entry['contentNormalized'],
                    'blocks': entry['blocks'],
                    'imageUri': page_uri,
                }
                pages.append(page_obj)
                page_progress_status = 'has_text'
                try:
                    _print_utf(f"[INFO] Page {page_num} added to final pages (has_text=True)")
                except Exception:
                    pass
            else:
                page_progress_status = 'no_text'
        except Exception:
            # Catch any failure during page processing (including OCR crashes or OOM)
            page_progress_status = 'error'
            try:
                import traceback
                _print_utf(f"[ERROR] Exception while processing page {page_num}: {traceback.format_exc()}")
            except Exception:
                pass
            # Insert a placeholder entry to preserve page continuity
            try:
                placeholder_entry = {
                    'entryId': f'pdf_native::page_{page_num}',
                    'jobTitle': f'page_{page_num}',
                    'pageNumber': page_num,
                    'contentMarkdown': '[OCR_FAILED_ERROR]',
                    'contentNormalized': '[OCR_FAILED_ERROR]',
                    'blocks': [],
                    'unitName': '未分类',
                    'position': len(entries) + 1,
                    'kind': 'text',
                    'status': 'skipped',
                }
                entries.append(placeholder_entry)
                placeholder_page_obj = {
                    'pageNumber': page_num,
                    'entryId': placeholder_entry['entryId'],
                    'jobTitle': placeholder_entry['jobTitle'],
                    'contentMarkdown': placeholder_entry['contentMarkdown'],
                    'contentNormalized': placeholder_entry['contentNormalized'],
                    'blocks': placeholder_entry['blocks'],
                    'imageUri': (f'file:///android_asset/kb/{file_id}/截图/page_{page_num}.jpg' if 'file_id' in locals() else ''),
                }
                pages.append(placeholder_page_obj)
            except Exception:
                pass
        finally:
            # Persist incremental state to disk after each page so partial results survive crashes
            try:
                partial_payload = _build_output_payload(
                    entries=entries,
                    file_id=file_id,
                    out_path=out_path,
                    pdf_path=pdf_path,
                )
                _write_json_atomic(out_path, partial_payload)
                try:
                    _print_utf(f"[DISK] Updated JSON for Page {page_num}. Current file size: {os.path.getsize(out_path)} bytes")
                except Exception:
                    _print_utf(f"[DISK] Updated JSON for Page {page_num}.")
            except Exception as _e:
                _print_utf(f"[DISK-ERROR] Failed to update JSON for Page {page_num}: {_e}")
            # Stable machine-readable page progress for external runners/UI.
            try:
                _print_utf(f"[PAGE_PROGRESS] current={page_num} total={doc.page_count} status={page_progress_status}")
            except Exception:
                pass

    # Finalize output and clean auxiliary assets if screenshot-only mode is enabled.
    if not keep_aux_images:
        try:
            _cleanup_non_visual_assets(assets_dir, entries)
        except Exception:
            pass

    final_payload = _build_output_payload(
        entries=entries,
        file_id=file_id,
        out_path=out_path,
        pdf_path=pdf_path,
    )
    production_clean_path = _build_production_clean_output_path(out_path)
    production_clean_payload = _build_output_payload(
        entries=entries,
        file_id=file_id,
        out_path=production_clean_path,
        pdf_path=pdf_path,
        trim_audit_fields=True,
    )

    try:
        _write_json_atomic(out_path, final_payload)
    except Exception as _e:
        _print_utf(f"[DISK-ERROR] Failed final JSON write: {_e}")

    try:
        _write_json_atomic(production_clean_path, production_clean_payload)
    except Exception as _e:
        _print_utf(f"[DISK-ERROR] Failed production-clean JSON write: {_e}")

    # Debug: report pages count and final status (file is updated per-page during processing)
    try:
        file_size = os.path.getsize(out_path) if os.path.exists(out_path) else 'N/A'
        _print_utf(
            f"[DEBUG] Final KB has {len(final_payload.get('entries', []))} entries before finishing. "
            f"Current file: {out_path} size: {file_size}"
        )
    except Exception:
        _print_utf(f"[DEBUG] Final KB has {len(final_payload.get('entries', []))} entries before finishing.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Create KB JSON from PDF with embedded images, table detection and OCR')
    p.add_argument('--pdf', required=True, help='Source PDF path')
    p.add_argument('--crops', default='./output/crops', help='Crops folder (legacy support)')
    p.add_argument('--out', default='knowledge_base.pdf_native.json', help='Output KB path')
    p.add_argument('--assets-root', default=os.path.join('app', 'src', 'main', 'assets', 'kb'), help='Assets root folder')
    p.add_argument('--file-id', default=None, help='File ID to use for asset folder (defaults to PDF basename)')
    # Compatibility: accept a target size parameter passed from older callers (optional)
    p.add_argument('--target-bytes', type=int, default=None, help='Optional target size in bytes to inflate to (compat)')
    p.add_argument('--ocr-engine', choices=['auto', 'paddle', 'tesseract'], default='auto', help='OCR engine to use')
    p.add_argument('--layout-engine', choices=['auto', 'yolo', 'ppstructure', 'cv'], default='auto', help='Layout detector priority: optional YOLO, PP-Structure, or CV fallback only')
    p.add_argument('--yolo-layout-model', default=None, help='Optional YOLO layout model path (or set YOLO_LAYOUT_MODEL env var)')
    p.add_argument('--yolo-layout-conf', type=float, default=YOLO_LAYOUT_DEFAULT_CONF, help='Confidence threshold for YOLO layout detections')
    p.add_argument('--yolo-layout-imgsz', type=int, default=YOLO_LAYOUT_DEFAULT_IMGSZ, help='Inference image size for YOLO layout detection')
    p.add_argument('--yolo-layout-device', default='', help='Optional ultralytics device override, e.g. cpu, 0, cuda:0')
    p.add_argument('--layout-debug-dir', default=None, help='Optional folder to dump per-page YOLO raw boxes, PP-Structure boxes, and final accepted layout blocks as JSON')
    p.add_argument('--max-page-width', type=int, default=1200, help='Max width for rendered page images')
    p.add_argument('--dpi', type=int, default=300, help='Render DPI for page images (default: 300)')
    p.add_argument('--start-page', type=int, default=1, help='1-based starting page to process')
    p.add_argument('--max-pages', type=int, default=None, help='Limit number of pages to process (smoke test)')
    p.add_argument('--keep-aux-images', action='store_true', help='Keep auxiliary page_* and embedded_* image files')
    return p


def main(argv: List[str]) -> int:
    # Use parse_known_args so callers can pass extra forward-compatible flags
    # (e.g., when CMD provides dynamic %CROP_EXTRA%). Unknown args are ignored.
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args(argv)
    build_kb_from_pdf(
        args.pdf,
        crops_dir=args.crops,
        out_path=args.out,
        assets_root=args.assets_root,
        file_id=args.file_id,
        ocr_engine=args.ocr_engine,
        layout_engine=getattr(args, 'layout_engine', 'auto'),
        yolo_layout_model=getattr(args, 'yolo_layout_model', None),
        yolo_layout_conf=getattr(args, 'yolo_layout_conf', YOLO_LAYOUT_DEFAULT_CONF),
        yolo_layout_imgsz=getattr(args, 'yolo_layout_imgsz', YOLO_LAYOUT_DEFAULT_IMGSZ),
        yolo_layout_device=getattr(args, 'yolo_layout_device', ''),
        layout_debug_dir=getattr(args, 'layout_debug_dir', None),
        max_page_image_width=args.max_page_width,
        page_render_dpi=getattr(args, 'dpi', 300),
        start_page=getattr(args, 'start_page', 1),
        max_pages=getattr(args, 'max_pages', None),
        keep_aux_images=bool(getattr(args, 'keep_aux_images', False)),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
