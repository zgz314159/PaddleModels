from typing import Any, Dict, List, Optional, Tuple, Set

def clip_xywh_to_page(x: int, y: int, w: int, h: int, page_w: int, page_h: int) -> Tuple[int, int, int, int]:
    x = int(max(0, min(x, max(0, page_w - 1))))
    y = int(max(0, min(y, max(0, page_h - 1))))
    w = int(max(1, min(w, max(1, page_w - x))))
    h = int(max(1, min(h, max(1, page_h - y))))
    return (x, y, w, h)

def bbox_overlap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))

def bbox_gap_1d(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, max(a0, b0) - min(a1, b1))

def coerce_bbox_to_xywh(bbox: Any) -> Optional[Tuple[int, int, int, int]]:
    if isinstance(bbox, dict):
        try:
            if {'left', 'top', 'right', 'bottom'}.issubset(set(bbox.keys())):
                left = int(round(float(bbox.get('left', 0))))
                top = int(round(float(bbox.get('top', 0))))
                right = int(round(float(bbox.get('right', left))))
                bottom = int(round(float(bbox.get('bottom', top))))
                return (left, top, max(1, right - left), max(1, bottom - top))
            if {'x0', 'y0', 'x1', 'y1'}.issubset(set(bbox.keys())):
                x0 = int(round(float(bbox.get('x0', 0))))
                y0 = int(round(float(bbox.get('y0', 0))))
                x1 = int(round(float(bbox.get('x1', x0))))
                y1 = int(round(float(bbox.get('y1', y0))))
                return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))
            if {'x', 'y', 'w', 'h'}.issubset(set(bbox.keys())):
                x = int(round(float(bbox.get('x', 0))))
                y = int(round(float(bbox.get('y', 0))))
                w = int(round(float(bbox.get('w', 0))))
                h = int(round(float(bbox.get('h', 0))))
                if w > 0 and h > 0:
                    return (x, y, w, h)
        except Exception:
            return None
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            vals = [int(round(float(v))) for v in bbox[:4]]
            if vals[2] > vals[0] and vals[3] > vals[1]:
                return (vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1])
            if vals[2] > 0 and vals[3] > 0:
                return (vals[0], vals[1], vals[2], vals[3])
        except Exception:
            return None
    return None

def xywh_to_bbox_dict(x: int, y: int, w: int, h: int) -> Dict[str, int]:
    return {
        'left': int(x),
        'top': int(y),
        'right': int(x + w),
        'bottom': int(y + h),
        'width': int(w),
        'height': int(h),
    }

def bbox_intersection_area_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    try:
        ax, ay, aw, ah = [int(v) for v in a]
        bx, by, bw, bh = [int(v) for v in b]
        x0 = max(ax, bx)
        y0 = max(ay, by)
        x1 = min(ax + aw, bx + bw)
        y1 = min(ay + ah, by + bh)
        if x1 <= x0 or y1 <= y0:
            return 0
        return int((x1 - x0) * (y1 - y0))
    except Exception:
        return 0

def bbox_overlap_ratio_xywh(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> float:
    try:
        _x, _y, iw, ih = [int(v) for v in inner]
        inner_area = max(1, iw * ih)
        return float(bbox_intersection_area_xywh(inner, outer)) / float(inner_area)
    except Exception:
        return 0.0

def bbox_center_inside_region(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int], *, min_center_overlap_ratio: float = 0.65) -> bool:
    try:
        x, y, w, h = [int(v) for v in inner]
        cx = x + w / 2.0
        cy = y + h / 2.0
        ox, oy, ow, oh = [int(v) for v in outer]
        if ow <= 0 or oh <= 0:
            return False
        if cx < ox or cy < oy or cx > ox + ow or cy > oy + oh:
            return False
        return bbox_overlap_ratio_xywh(inner, outer) >= float(min_center_overlap_ratio)
    except Exception:
        return False

def bbox_iou_xywh(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    try:
        inter = float(bbox_intersection_area_xywh(a, b))
        union = float(a[2] * a[3] + b[2] * b[3]) - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def bbox_inside_xywh(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int], margin: int = 6) -> bool:
    try:
        ix, iy, iw, ih = inner
        ox, oy, ow, oh = outer
        return (ix >= ox - margin and iy >= oy - margin and
                ix + iw <= ox + ow + margin and iy + ih <= oy + oh + margin)
    except Exception:
        return False

def bbox_coverage_xywh(inner: Tuple[int, int, int, int], outer: Tuple[int, int, int, int]) -> float:
    try:
        area_outer = float(outer[2] * outer[3])
        if area_outer <= 0: return 0.0
        inter = float(bbox_intersection_area_xywh(inner, outer))
        return inter / area_outer
    except Exception:
        return 0.0

def normalize_bbox_dict(value: Any) -> Optional[Dict[str, int]]:
    xywh = coerce_bbox_to_xywh(value)
    if xywh:
        return xywh_to_bbox_dict(*xywh)
    return None
