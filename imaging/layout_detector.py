import importlib
import logging
from typing import List, Dict, Any, Optional
try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2
except Exception:
    cv2 = None

logger = logging.getLogger(__name__)

# Allowed layout types for saving crops
ALLOWED_LAYOUT_TYPES = ('table', 'figure', 'equation')
# Explicitly skip these non-visual layout labels (they are handled via OCR text)
SKIP_TEXT_TYPES = ('text', 'title', 'header', 'footer')

_UltralyticsYOLO = None
_HAS_ULTRALYTICS = False
_ULTRALYTICS_IMPORT_ERR = ''

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

def instantiate_yolo_layout_detector(model_path: str):
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
            logger.error(f"Failed to import ultralytics: {ex}")
            return None
    if not _HAS_ULTRALYTICS or _UltralyticsYOLO is None:
        return None
    try:
        return _UltralyticsYOLO(model_path)
    except Exception as ex:
        logger.error(f"Failed to load YOLO model from {model_path}: {ex}")
        return None

def detect_yolo_layout_blocks(
    detector: Any,
    image_bytes: bytes,
    *,
    conf: float,
    imgsz: int,
    device: str,
) -> List[Dict[str, Any]]:
    if detector is None or not image_bytes:
        return []
    if cv2 is None or np is None:
        return []

    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as ex:
        logger.error(f"Error decoding image for YOLO: {ex}")
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
    except Exception as ex:
        logger.error(f"YOLO prediction error: {ex}")
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

def coerce_layout_items(layout_res: Any) -> Optional[List[Any]]:
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
