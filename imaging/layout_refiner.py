import re
import os
import io
import logging
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
    bbox_coverage_xywh,
    bbox_inside_xywh,
    bbox_iou_xywh,
    bbox_overlap_1d,
    clip_xywh_to_page,
    bbox_center_inside_region,
    bbox_overlap_ratio_xywh,
    coerce_bbox_to_xywh
)
from imaging.vision_utils import (
    build_visual_signal_mask
)
from imaging.ocr_engine import ocr_image_bytes

logger = logging.getLogger(__name__)

# Constants from main script
TEXT_TRIM_MARGIN_PX = 6
FALLBACK_MIN_H_LONG = 4
FALLBACK_MIN_V_LONG = 2
FALLBACK_MIN_INTERSECTIONS = 10
ALLOWED_LAYOUT_TYPES = ('table', 'figure', 'equation')
BLUE_ANNOTATION_BODY_EXCLUSION_OVERLAP_RATIO = 0.55

def normalize_figure_note_text(text: str) -> str:
    raw = str(text or '').strip()
    if not raw: return ''
    compact = re.sub(r'[\s\-_—–~〜·•,，.。:：;；|/\\]+', '', raw)
    return compact.strip()

def looks_like_figure_note_text(text: str) -> bool:
    compact = normalize_figure_note_text(text)
    if not compact: return False
    try:
        if re.match(r'^[（(]?[a-zA-Z0-9一二三四五六七八九十]+[）)](?:[-—:：.]|$)', compact):
            return True
        if re.match(r'^(?:[0-9]{1,2}|[a-zA-Z])[\-—:：][^。；;！？!?]{1,24}$', compact):
            return True
        if re.search(r'(?:^|[，,；;、])(?:[0-9]{1,2}|[a-zA-Z]|[（(][a-zA-Z0-9一二三四五六七八九十]+[）)])\s*[\-—:：]\s*[^。；;！？!?]{1,18}', compact):
            return True
        if len(compact) <= 36 and re.search(r'[（(][a-zA-Z0-9一二三四五六七八九十]+[）)]', compact):
            return True
    except Exception:
        return False
    return False

def estimate_text_span_ratio(tb: Dict, crop_w: int) -> float:
    try:
        bbox = tb.get('bbox')
        if not bbox or len(bbox) < 4: return 0.0
        tw = bbox[2]
        return float(tw) / float(max(1, crop_w))
    except Exception:
        return 0.0

def is_caption_like_text_block(tb: Dict, crop_w: int) -> bool:
    try:
        bbox = tb.get('bbox', (0, 0, 0, 0))
        bw = bbox[2]
        line_count = int(tb.get('line_count') or 0)
        avg_line_len = float(tb.get('avg_line_len') or 0.0)
        text = str(tb.get('text') or '').strip()
        width_ratio = (float(bw) / float(max(1, crop_w))) if crop_w > 0 else 1.0
        span_ratio = estimate_text_span_ratio(tb, crop_w)
        if re.match(r'^(图|表|Fig\.?|Figure|Table|注)\s*', text, flags=re.IGNORECASE):
            return True
        if looks_like_figure_note_text(text) and line_count <= 2 and span_ratio <= 0.96:
            return True
        if line_count <= 2 and avg_line_len <= 14.0 and width_ratio <= 0.78:
            return True
        if line_count <= 1 and width_ratio <= 0.88:
            return True
        if line_count <= 3 and avg_line_len <= 22.0 and span_ratio <= 0.86:
            if len(text) <= max(54, int(float(max(1, crop_w)) / 18.0)):
                if not re.search(r'[。；;！？!?].+[。；;！？!?]', text):
                    return True
    except Exception:
        return False
    return False

def is_compact_annotation_like_text_block(tb: Dict, crop_w: int) -> bool:
    try:
        bbox = tb.get('bbox', (0, 0, 0, 0))
        bw = bbox[2]
        line_count = int(tb.get('line_count') or 0)
        avg_line_len = float(tb.get('avg_line_len') or 0.0)
        text = str(tb.get('text') or '').strip()
        if not text:
            return False
        width_ratio = (float(bw) / float(max(1, crop_w))) if crop_w > 0 else 1.0
        span_ratio = estimate_text_span_ratio(tb, crop_w)
        if is_caption_like_text_block(tb, crop_w):
            return True
        if looks_like_figure_note_text(text) and line_count <= 4 and span_ratio <= 0.98 and len(normalize_figure_note_text(text)) <= 52:
            return True
        if line_count <= 3 and avg_line_len <= 18.0 and width_ratio <= 0.92 and span_ratio <= 0.94 and len(text) <= 42:
            if not re.search(r'[。；;！？!?]', text):
                return True
        if line_count <= 2 and width_ratio <= 0.72 and span_ratio <= 0.78 and len(text) <= 24:
            return True
    except Exception:
        return False
    return False

def collect_top_semantic_constraints(
    native_text_line_boxes: Optional[List[Dict]],
    avoidance_constraints: Optional[List[Dict]],
    native_text_boxes: Optional[List[Dict]] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, int, int, int, int]] = set()

    for line_box in native_text_line_boxes or []:
        try:
            bbox = line_box.get('bbox', (0, 0, 0, 0))
            x, y, w, h = bbox[:4]
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        key = ('pdf_text_line', int(x), int(y), int(w), int(h))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'type': 'text',
            'bbox': (int(x), int(y), int(w), int(h)),
            'source': 'pdf_text_line',
            'text': str(line_box.get('text') or ''),
        })

    if not out:
        for text_box in native_text_boxes or []:
            try:
                bbox = text_box.get('bbox', (0, 0, 0, 0))
                x, y, w, h = bbox[:4]
            except Exception:
                continue
            if w <= 0 or h <= 0:
                continue
            key = ('pdf_text_box', int(x), int(y), int(w), int(h))
            if key in seen:
                continue
            seen.add(key)
            out.append({
                'type': 'text',
                'bbox': (int(x), int(y), int(w), int(h)),
                'source': 'pdf_text_box',
                'text': str(text_box.get('text') or ''),
            })

    for constraint in avoidance_constraints or []:
        try:
            bbox = constraint.get('bbox', (0, 0, 0, 0))
            x, y, w, h = bbox[:4]
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        source = str(constraint.get('source') or 'layout').lower()
        constraint_type = str(constraint.get('type') or 'text').lower()
        key = (f'{source}:{constraint_type}', int(x), int(y), int(w), int(h))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'type': constraint_type,
            'bbox': (int(x), int(y), int(w), int(h)),
            'source': source,
        })

    return out

def collect_relevant_figure_support_regions(
    parent_bbox: Tuple[int, int, int, int],
    page_text_boxes: List[Dict],
    layout_support_boxes: Optional[List[Dict]] = None,
) -> List[Tuple[int, int, int, int]]:
    px, py, pw, ph = parent_bbox
    px1 = px + pw
    py1 = py + ph
    caption_margin = max(70, int(ph * 0.25))
    relevant: List[Tuple[int, int, int, int]] = []

    for tb in page_text_boxes or []:
        try:
            if not is_caption_like_text_block(tb, pw):
                continue
            bbox = tb.get('bbox', (0, 0, 0, 0))
            tx, ty, tw, th = bbox[:4]
            tx1 = tx + tw
            ty1 = ty + th
            x_overlap = bbox_overlap_1d(px, px1, tx, tx1)
            if x_overlap < max(40, int(min(tw, pw) * 0.35)):
                continue
            if ty > py1 + caption_margin:
                continue
            if ty1 < py + int(ph * 0.35):
                continue
            relevant.append((int(tx), int(ty), int(tw), int(th)))
        except Exception:
            continue

    for support in layout_support_boxes or []:
        try:
            bbox = support.get('bbox', (0,0,0,0))
            tx, ty, tw, th = bbox[:4]
            if tw <= 0 or th <= 0: continue
            tx1 = tx + tw
            ty1 = ty + th
            if ty > py1 + caption_margin:
                continue
            if ty1 < py + int(ph * 0.30): # support_top_limit
                continue
            x_overlap = bbox_overlap_1d(px, px1, tx, tx1)
            center_delta = abs((float(tx) + float(tx1)) * 0.5 - (float(px) + float(px1)) * 0.5)
            if x_overlap < max(28, int(min(tw, pw) * 0.18)) and center_delta > max(float(pw) * 0.42, float(tw) * 0.85):
                continue
            relevant.append((int(tx), int(ty), int(tw), int(th)))
        except Exception:
            continue
            
    deduped: List[Tuple[int, int, int, int]] = []
    seen_boxes: Set[Tuple[int, int, int, int]] = set()
    for box in sorted(relevant, key=lambda b: (b[1], b[0], b[2], b[3])):
        if box in seen_boxes:
            continue
        seen_boxes.add(box)
        deduped.append(box)
    return deduped

def suppress_nested_figure_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Keep complete figure boxes and suppress substantially covered fragments."""
    if len(boxes) < 2:
        return list(boxes)

    try:
        ordered = sorted(boxes, key=lambda b: (-(b[2] * b[3]), b[1], b[0]))
    except Exception:
        ordered = list(boxes)

    kept: List[Tuple[int, int, int, int]] = []
    for cand in ordered:
        try:
            cand_area = float(max(1, cand[2] * cand[3]))
        except Exception:
            continue

        drop = False
        for parent in kept:
            try:
                parent_area = float(max(1, parent[2] * parent[3]))
                area_ratio = cand_area / parent_area
                coverage = bbox_coverage_xywh(cand, parent)
                if coverage >= 0.78 and area_ratio <= 0.62:
                    drop = True
                    break
                if bbox_inside_xywh(cand, parent, margin=18) and area_ratio <= 0.80:
                    drop = True
                    break
            except Exception:
                continue
        if not drop:
            kept.append(cand)

    try:
        kept = sorted(kept, key=lambda b: (b[1], b[0]))
    except Exception:
        pass
    return kept

def suppress_compound_parent_figure_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Suppress large boxes that are mostly just containers for smaller discrete figures."""
    if len(boxes) < 3:
        return list(boxes)

    try:
        ordered = sorted(boxes, key=lambda b: (-(b[2] * b[3]), b[1], b[0]))
    except Exception:
        ordered = list(boxes)

    kept: List[Tuple[int, int, int, int]] = []
    for idx, cand in enumerate(ordered):
        try:
            cx, cy, cw, ch = cand
            cand_area = float(max(1, cw * ch))
        except Exception:
            kept.append(cand)
            continue

        support_boxes: List[Tuple[int, int, int, int]] = []
        covered_area = 0.0
        for jdx, other in enumerate(ordered):
            if idx == jdx:
                continue
            try:
                ox, oy, ow, oh = other
                other_area = float(max(1, ow * oh))
                if other_area >= cand_area * 0.90:
                    continue
                inter_w = bbox_overlap_1d(cx, cx + cw, ox, ox + ow)
                inter_h = bbox_overlap_1d(cy, cy + ch, oy, oy + oh)
                inter_area = float(max(0, inter_w * inter_h))
                if inter_area < cand_area * 0.18:
                    continue
                support_boxes.append(other)
                covered_area += inter_area
            except Exception:
                continue

        drop = False
        if len(support_boxes) >= 2 and covered_area >= cand_area * 0.88:
            x_centers = [float(b[0]) + (float(b[2]) * 0.5) for b in support_boxes]
            y_centers = [float(b[1]) + (float(b[3]) * 0.5) for b in support_boxes]
            if (max(y_centers) - min(y_centers)) >= max(90.0, float(ch) * 0.18):
                drop = True
            elif (max(x_centers) - min(x_centers)) >= max(90.0, float(cw) * 0.18):
                drop = True

        if not drop:
            kept.append(cand)

    try:
        kept = sorted(kept, key=lambda b: (b[1], b[0]))
    except Exception:
        pass
    return kept

def is_nested_figure_fragment(candidate: Tuple[int, int, int, int], kept_boxes: List[Tuple[int, int, int, int]]) -> bool:
    if not kept_boxes:
        return False
    try:
        cand_area = float(max(1, candidate[2] * candidate[3]))
    except Exception:
        return False

    for kept in kept_boxes:
        try:
            kept_area = float(max(1, kept[2] * kept[3]))
            area_ratio = cand_area / kept_area
            coverage = bbox_coverage_xywh(candidate, kept)
            if coverage >= 0.78 and area_ratio <= 0.62:
                return True
            if bbox_inside_xywh(candidate, kept, margin=18) and area_ratio <= 0.80:
                return True
        except Exception:
            continue
    return False

def prune_overlapping_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    """Remove near-duplicate / nested CV boxes while preserving distinct regions."""
    if not boxes:
        return []
    try:
        ordered = sorted(boxes, key=lambda b: (-(b[2] * b[3]), b[1], b[0]))
    except Exception:
        ordered = list(boxes)

    kept: List[Tuple[int, int, int, int]] = []
    for cand in ordered:
        try:
            cx, cy, cw, ch = cand
            if cw <= 0 or ch <= 0:
                continue
        except Exception:
            continue

        drop = False
        for k in kept:
            try:
                iou = bbox_iou_xywh(cand, k)
                if iou >= 0.82:
                    drop = True
                    break
                if bbox_inside_xywh(cand, k, margin=8):
                    drop = True
                    break
            except Exception:
                continue
        if not drop:
            kept.append(cand)

    try:
        kept = sorted(kept, key=lambda b: (b[1], b[0]))
    except Exception:
        pass
    return kept

def suppress_redundant_layout_figure_blocks(layout_blocks: List[Dict]) -> List[Dict]:
    if len(layout_blocks) < 3:
        return layout_blocks

    try:
        figure_indices = [idx for idx, lb in enumerate(layout_blocks) if str(lb.get('type') or '').strip().lower() == 'figure']
    except Exception:
        return layout_blocks
    if len(figure_indices) < 3:
        return layout_blocks

    drop_indices = set()
    for idx in figure_indices:
        try:
            cand = layout_blocks[idx]
            cb = cand.get('bbox', (0, 0, 0, 0))
            cx, cy, cw, ch = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
            cand_area = float(max(1, cw * ch))
        except Exception:
            continue

        support_count = 0
        covered_area = 0.0
        x_centers: List[float] = []
        y_centers: List[float] = []
        for jdx in figure_indices:
            if jdx == idx:
                continue
            try:
                other = layout_blocks[jdx]
                ob = other.get('bbox', (0, 0, 0, 0))
                ox, oy, ow, oh = (int(ob[0]), int(ob[1]), int(ob[2]), int(ob[3]))
                other_area = float(max(1, ow * oh))
                if other_area >= cand_area * 0.90:
                    continue
                inter_w = bbox_overlap_1d(cx, cx + cw, ox, ox + ow)
                inter_h = bbox_overlap_1d(cy, cy + ch, oy, oy + oh)
                inter_area = float(max(0, inter_w * inter_h))
                if inter_area < cand_area * 0.18:
                    continue
                support_count += 1
                covered_area += inter_area
                x_centers.append(float(ox) + (float(ow) * 0.5))
                y_centers.append(float(oy) + (float(oh) * 0.5))
            except Exception:
                continue

        if support_count < 2 or covered_area < cand_area * 0.88:
            continue
        if y_centers and (max(y_centers) - min(y_centers)) >= max(90.0, float(ch) * 0.18):
            drop_indices.add(idx)
            continue
        if x_centers and (max(x_centers) - min(x_centers)) >= max(90.0, float(cw) * 0.18):
            drop_indices.add(idx)

    if not drop_indices:
        return layout_blocks
    return [lb for idx, lb in enumerate(layout_blocks) if idx not in drop_indices]

def merge_sidecar_figure_boxes(boxes: List[Tuple[int, int, int, int]], page_w: int = 0, page_h: int = 0) -> List[Tuple[int, int, int, int]]:
    """Merge figure fragments when a narrow sidecar box belongs to the same drawing."""
    if len(boxes) < 2:
        return list(boxes)

    try:
        page_area = float(max(1, page_w * page_h)) if page_w > 0 and page_h > 0 else 0.0
    except Exception:
        page_area = 0.0

    try:
        merged = sorted(boxes, key=lambda b: (b[1], b[0]))
    except Exception:
        merged = list(boxes)

    changed = True
    while changed and len(merged) >= 2:
        changed = False
        for i in range(len(merged)):
            ax, ay, aw, ah = merged[i]
            ax1 = ax + aw
            ay1 = ay + ah
            for j in range(i + 1, len(merged)):
                bx, by, bw, bh = merged[j]
                bx1 = bx + bw
                by1 = by + bh

                overlap_h = bbox_overlap_1d(ay, ay1, by, by1)
                try:
                    y_overlap_ratio = float(overlap_h) / float(max(1, min(ah, bh)))
                except Exception:
                    y_overlap_ratio = 0.0
                if y_overlap_ratio < 0.58:
                    continue

                horizontal_gap = max(0, max(ax, bx) - min(ax1, bx1))
                if horizontal_gap > max(110, int(min(aw, bw) * 0.24)):
                    continue

                try:
                    width_ratio = float(min(aw, bw)) / float(max(aw, bw))
                except Exception:
                    width_ratio = 1.0
                if width_ratio > 0.72 and min(aw, bw) > 220:
                    continue

                union_x0 = min(ax, bx)
                union_y0 = min(ay, by)
                union_x1 = max(ax1, bx1)
                union_y1 = max(ay1, by1)
                union_w = union_x1 - union_x0
                union_h = union_y1 - union_y0
                union_area = float(max(1, union_w * union_h))

                inter_w = bbox_overlap_1d(ax, ax1, bx, bx1)
                inter_h = overlap_h
                inter_area = float(max(0, inter_w * inter_h))
                covered_area = float(max(1, aw * ah + bw * bh)) - inter_area
                fill_ratio = covered_area / union_area

                if fill_ratio < 0.46:
                    continue
                if page_area and (union_area / page_area) > 0.22:
                    continue

                merged[i] = (union_x0, union_y0, union_w, union_h)
                del merged[j]
                changed = True
                break
            if changed:
                break

    try:
        merged = sorted(merged, key=lambda b: (b[1], b[0]))
    except Exception:
        pass
    return merged

def merge_stacked_table_boxes(boxes: List[Tuple[int, int, int, int]], page_w: int = 0, page_h: int = 0) -> List[Tuple[int, int, int, int]]:
    """Merge table fragments that are vertically stacked or overlapping with near-identical columns."""
    if len(boxes) < 2:
        return list(boxes)

    try:
        page_area = float(max(1, page_w * page_h)) if page_w > 0 and page_h > 0 else 0.0
    except Exception:
        page_area = 0.0

    try:
        merged = sorted(boxes, key=lambda b: (b[1], b[0]))
    except Exception:
        merged = list(boxes)

    changed = True
    while changed and len(merged) >= 2:
        changed = False
        for i in range(len(merged)):
            ax, ay, aw, ah = merged[i]
            ax1 = ax + aw
            ay1 = ay + ah
            acx = ax + (aw / 2.0)
            for j in range(i + 1, len(merged)):
                bx, by, bw, bh = merged[j]
                bx1 = bx + bw
                by1 = by + bh
                bcx = bx + (bw / 2.0)

                x_overlap = bbox_overlap_1d(ax, ax1, bx, bx1)
                try:
                    x_overlap_ratio = float(x_overlap) / float(max(1, min(aw, bw)))
                except Exception:
                    x_overlap_ratio = 0.0
                if x_overlap_ratio < 0.80:
                    continue

                try:
                    width_ratio = float(min(aw, bw)) / float(max(aw, bw))
                except Exception:
                    width_ratio = 0.0
                if width_ratio < 0.78:
                    continue

                center_offset = abs(acx - bcx)
                if center_offset > max(18.0, float(min(aw, bw)) * 0.08):
                    continue

                overlap_v = bbox_overlap_1d(ay, ay1, by, by1)
                vertical_gap = max(0, max(ay, by) - min(ay1, by1))
                if overlap_v <= 0 and vertical_gap > max(12, int(min(ah, bh) * 0.10)):
                    continue

                strong_column_alignment = (
                    x_overlap_ratio >= 0.95 
                    and width_ratio >= 0.95 
                    and center_offset <= max(10.0, float(min(aw, bw)) * 0.04)
                    and vertical_gap <= 2
                )
                
                union_x0 = min(ax, bx)
                union_y0 = min(ay, by)
                union_x1 = max(ax1, bx1)
                union_y1 = max(ay1, by1)
                union_w = union_x1 - union_x0
                union_h = union_y1 - union_y0
                max_union_height_ratio = 2.25 if strong_column_alignment else 1.85
                if union_h > int(max(ah, bh) * max_union_height_ratio):
                    continue

                union_area = float(max(1, union_w * union_h))
                inter_w = x_overlap
                inter_h = bbox_overlap_1d(ay, ay1, by, by1)
                inter_area = float(max(0, inter_w * inter_h))
                covered_area = float(max(1, aw * ah + bw * bh)) - inter_area
                fill_ratio = covered_area / union_area
                if fill_ratio < 0.72:
                    continue
                max_page_ratio = 0.48 if strong_column_alignment else 0.34
                if page_area and (union_area / page_area) > max_page_ratio:
                    continue

                merged[i] = (union_x0, union_y0, union_w, union_h)
                del merged[j]
                changed = True
                break
            if changed:
                break
    
    try:
        merged = sorted(merged, key=lambda b: (b[1], b[0]))
    except Exception:
        pass
    return merged

def trim_visual_bbox_with_text_boxes(
    bbox: Tuple[int, int, int, int],
    text_boxes: List[Dict],
    kind: str,
    page_h: int,
) -> Tuple[int, int, int, int]:
    """Trim paragraph text from visual crops using native PDF text block geometry."""
    x, y, w, h = bbox
    x1 = x + w
    y1 = y + h
    if kind not in ('figure', 'equation') or not text_boxes:
        return bbox

    overlap_boxes: List[Dict] = []
    for tb in text_boxes:
        try:
            t_bbox = tb.get('bbox')
            if not t_bbox: continue
            tx, ty, tw, th = t_bbox[:4]
            tx1 = tx + tw
            x_ov = bbox_overlap_1d(x, x1, tx, tx1)
            if x_ov <= 0:
                continue
            x_ratio = float(x_ov) / float(max(1, min(w, tw)))
            if x_ratio < 0.35:
                is_note_like = is_compact_annotation_like_text_block(tb, w)
                tb_center = float(tx + tx1) * 0.5
                box_center = float(x + x1) * 0.5
                center_delta = abs(tb_center - box_center)
                center_limit = max(float(tw) * 0.9, float(w) * 0.28)
                if not (is_note_like and x_ov >= max(18, int(min(w, tw) * 0.15)) and center_delta <= center_limit):
                    continue
            overlap_boxes.append(tb)
        except Exception:
            continue

    if not overlap_boxes:
        return bbox

    below = sorted((tb for tb in overlap_boxes if tb.get('bbox')[1] >= y + int(h * 0.25)), key=lambda tb: tb.get('bbox')[1])
    kept_bottom = y1
    seen_note_band = False
    note_band_bottom = -1
    caption_tail_gap = max(TEXT_TRIM_MARGIN_PX * 3, int(h * 0.08))
    caption_tail_limit = max(170, int(h * 0.46))
    for tb in below:
        tx, ty, tw, th = tb.get('bbox')[:4]
        if ty >= kept_bottom:
            continue
        is_note_like = is_compact_annotation_like_text_block(tb, w)
        if is_note_like and not seen_note_band:
            note_band_bottom = ty + th
            kept_bottom = min(page_h, note_band_bottom + max(TEXT_TRIM_MARGIN_PX, int(h * 0.015)))
            seen_note_band = True
            continue
        if seen_note_band and is_note_like:
            if ty <= (note_band_bottom + caption_tail_gap) and (ty + th) <= (y1 + caption_tail_limit):
                note_band_bottom = max(note_band_bottom, ty + th)
                kept_bottom = min(page_h, note_band_bottom + max(TEXT_TRIM_MARGIN_PX, int(h * 0.015)))
                continue
        if ty > y + int(h * 0.18):
            kept_bottom = min(kept_bottom, max(y + max(80, int(h * 0.30)), ty - TEXT_TRIM_MARGIN_PX))
            break

    above = sorted((tb for tb in overlap_boxes if (tb.get('bbox')[1] + tb.get('bbox')[3]) <= y + int(h * 0.65)), key=lambda tb: tb.get('bbox')[1] + tb.get('bbox')[3], reverse=True)
    kept_top = y
    for tb in above:
        tx, ty, tw, th = tb.get('bbox')[:4]
        if is_caption_like_text_block(tb, w):
            continue
        if (ty + th) < y1 and (ty + th) >= y:
            kept_top = max(kept_top, ty + th + TEXT_TRIM_MARGIN_PX)
            break

    kept_left = x
    kept_right = x1
    side_boxes = sorted(overlap_boxes, key=lambda tb: (tb.get('bbox')[0], tb.get('bbox')[1]))
    for tb in side_boxes:
        tx, ty, tw, th = tb.get('bbox')[:4]
        if is_compact_annotation_like_text_block(tb, w):
            continue
        y_ov = bbox_overlap_1d(y, y1, ty, ty + th)
        if y_ov <= 0:
            continue
        y_ratio = float(y_ov) / float(max(1, min(h, th)))
        if y_ratio < 0.25:
            continue
        tb_right = tx + tw
        if tx <= x + int(w * 0.10) and tb_right <= x + int(w * 0.42):
            kept_left = max(kept_left, tb_right + TEXT_TRIM_MARGIN_PX)
            continue
        if tb_right >= x1 - int(w * 0.10) and tx >= x + int(w * 0.58):
            kept_right = min(kept_right, tx - TEXT_TRIM_MARGIN_PX)

    new_y = min(kept_top, kept_bottom - 1)
    new_x = min(kept_left, kept_right - 1)
    new_h = max(1, kept_bottom - new_y)
    new_w = max(1, kept_right - new_x)
    if new_h < max(80, int(h * 0.45)):
        return bbox
    if new_w < max(140, int(w * 0.52)):
        new_x = x
        new_w = w
    return (new_x, new_y, new_w, new_h)

def pad_visual_bbox(
    bbox: Tuple[int, int, int, int],
    kind: str,
    page_w: int,
    page_h: int,
) -> Tuple[int, int, int, int]:
    try:
        x, y, w, h = bbox
        if page_w <= 0 or page_h <= 0 or w <= 0 or h <= 0:
            return bbox
        pad_x = max(8, int(float(w) * 0.012))
        pad_y = max(8, int(float(h) * 0.015))
        pad_top = pad_y
        pad_bottom = pad_y
        if kind == 'table':
            pad_x = max(pad_x, 10)
            pad_top = max(pad_y, 10)
            pad_bottom = max(pad_y, 10)
        elif kind in ('figure', 'equation'):
            pad_top = max(pad_y, 10)
            pad_bottom = max(pad_y, int(float(h) * 0.045), 18)
            if kind == 'figure' and int(h) <= 420:
                pad_bottom = max(pad_bottom, int(float(h) * 0.12), 36)
        return clip_xywh_to_page(int(x) - pad_x, int(y) - pad_top, int(w) + (pad_x * 2), int(h) + pad_top + pad_bottom, page_w, page_h)
    except Exception:
        return bbox

def clamp_visual_bbox_growth(
    anchor_bbox: Tuple[int, int, int, int],
    candidate_bbox: Tuple[int, int, int, int],
    kind: str,
    page_w: int,
    page_h: int,
) -> Tuple[int, int, int, int]:
    try:
        ax, ay, aw, ah = anchor_bbox
        cx, cy, cw, ch = candidate_bbox
        if kind not in ('figure', 'equation', 'table'):
            return clip_xywh_to_page(cx, cy, cw, ch, page_w, page_h)

        grow_left = max(24, int(float(aw) * 0.08))
        grow_right = max(24, int(float(aw) * 0.08))
        grow_top = max(24, int(float(ah) * 0.08))
        grow_bottom = max(70, int(float(ah) * 0.24))
        if kind in ('figure', 'equation'):
            grow_left = max(14, int(float(aw) * 0.03))
            grow_right = max(14, int(float(aw) * 0.03))
            grow_top = max(14, int(float(ah) * 0.03))
            grow_bottom = max(110, int(float(ah) * 0.42))
        elif kind == 'table':
            grow_left = max(grow_left, 36)
            grow_right = max(grow_right, 36)
            grow_top = max(grow_top, 30)
            grow_bottom = max(grow_bottom, 90)

        min_x = ax - grow_left
        max_x1 = ax + aw + grow_right
        min_y = ay - grow_top
        max_y1 = ay + ah + grow_bottom

        nx0 = max(min_x, cx)
        ny0 = max(min_y, cy)
        nx1 = min(max_x1, cx + cw)
        ny1 = min(max_y1, cy + ch)
        if nx1 <= nx0 or ny1 <= ny0:
            return clip_xywh_to_page(cx, cy, cw, ch, page_w, page_h)
        return clip_xywh_to_page(nx0, ny0, nx1 - nx0, ny1 - ny0, page_w, page_h)
    except Exception:
        return candidate_bbox

def refine_visual_bbox_from_pixels(
    page_cv,
    bbox: Tuple[int, int, int, int],
    kind: str,
) -> List[Tuple[int, int, int, int]]:
    if cv2 is None or page_cv is None:
        return [bbox]
    try:
        page_h, page_w = page_cv.shape[:2]
        x, y, w, h = clip_xywh_to_page(*bbox, page_w, page_h)
        pad_x = max(12, int(w * 0.10))
        pad_y = max(12, int(h * 0.10))
        if kind in ('figure', 'equation'):
            pad_y = max(pad_y, int(h * 0.15), 18)
        sx = max(0, x - pad_x)
        sy = max(0, y - pad_y)
        ex = min(page_w, x + w + pad_x)
        ey = min(page_h, y + h + pad_y)
        roi = page_cv[sy:ey, sx:ex]
        if roi is None or getattr(roi, 'size', 0) == 0:
            return [bbox]

        mask = build_visual_signal_mask(roi)
        if mask is None:
            return [bbox]

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        seed_box = (x - sx, y - sy, w, h)
        seed_x0, seed_y0, seed_w, seed_h = seed_box
        seed_x1 = seed_x0 + seed_w
        seed_y1 = seed_y0 + seed_h
        roi_area = float((ex - sx) * (ey - sy))
        min_cc_area = max(20, int(roi_area * 0.00008))
        selected: List[Tuple[int, int, int, int]] = []

        for label in range(1, int(num_labels)):
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            ww = int(stats[label, cv2.CC_STAT_WIDTH])
            hh = int(stats[label, cv2.CC_STAT_HEIGHT])
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_cc_area or ww <= 0 or hh <= 0:
                continue
            x1 = x0 + ww
            y1 = y0 + hh
            overlap = bbox_overlap_1d(x0, x1, seed_x0, seed_x1) * bbox_overlap_1d(y0, y1, seed_y0, seed_y1)
            near_x = max(0, max(seed_x0 - x1, x0 - seed_x1))
            near_y = max(0, max(seed_y0 - y1, y0 - seed_y1))
            near_x_thr = max(10, int(seed_w * 0.08))
            near_y_thr = max(10, int(seed_h * 0.08))
            if kind in ('figure', 'equation'):
                near_y_thr = max(near_y_thr, int(seed_h * 0.16), 16)
            if overlap > 0 or (near_x <= near_x_thr and near_y <= near_y_thr):
                selected.append((x0, y0, ww, hh))

        if not selected:
            return [bbox]

        changed = True
        gap_x_thr = max(12, int(seed_w * 0.10))
        gap_y_thr = max(12, int(seed_h * 0.10))
        if kind in ('figure', 'equation'):
            gap_y_thr = max(gap_y_thr, int(seed_h * 0.16), 18)
        while changed:
            changed = False
            ux0 = min(c[0] for c in selected)
            uy0 = min(c[1] for c in selected)
            ux1 = max(c[0] + c[2] for c in selected)
            uy1 = max(c[1] + c[3] for c in selected)
            for label in range(1, int(num_labels)):
                x0 = int(stats[label, cv2.CC_STAT_LEFT])
                y0 = int(stats[label, cv2.CC_STAT_TOP])
                ww = int(stats[label, cv2.CC_STAT_WIDTH])
                hh = int(stats[label, cv2.CC_STAT_HEIGHT])
                area = int(stats[label, cv2.CC_STAT_AREA])
                cand = (x0, y0, ww, hh)
                if area < min_cc_area or cand in selected:
                    continue
                x1 = x0 + ww
                y1 = y0 + hh
                gap_x = max(0, max(ux0 - x1, x0 - ux1))
                gap_y = max(0, max(uy0 - y1, y0 - uy1))
                y_overlap = bbox_overlap_1d(y0, y1, uy0, uy1)
                x_overlap = bbox_overlap_1d(x0, x1, ux0, ux1)
                if (gap_x <= gap_x_thr and y_overlap >= max(8, int(min(hh, uy1 - uy0) * 0.15))) or (gap_y <= gap_y_thr and x_overlap >= max(8, int(min(ww, ux1 - ux0) * 0.15))):
                    selected.append(cand)
                    changed = True

        rx0 = min(c[0] for c in selected)
        ry0 = min(c[1] for c in selected)
        rx1 = max(c[0] + c[2] for c in selected)
        ry1 = max(c[1] + c[3] for c in selected)
        refined = clip_xywh_to_page(sx + rx0, sy + ry0, rx1 - rx0, ry1 - ry0, page_w, page_h)
        return [refined]
    except Exception:
        return [bbox]

def iteratively_refine_visual_bbox(
    page_cv,
    anchor_bbox: Tuple[int, int, int, int],
    candidate_bbox: Tuple[int, int, int, int],
    page_text_boxes: List[Dict],
    avoidance_constraints: Optional[List[Dict]],
    kind: str,
    page_w: int,
    page_h: int,
    page_area: Optional[float],
    debug_context: str = '',
) -> Tuple[int, int, int, int]:
    """Iteratively apply text-based trim and growth-clamping to refine a visual crop."""
    current = candidate_bbox
    for _ in range(3):
        prev = current
        cx, cy, cw, ch = current
        if page_w > 0 and page_h > 0:
            cx, cy, cw, ch = clip_xywh_to_page(int(cx), int(cy), int(cw), int(ch), int(page_w), int(page_h))
        
        if kind in ('figure', 'equation'):
            if page_h > 0:
                cx, cy, cw, ch = trim_visual_bbox_with_text_boxes((cx, cy, cw, ch), page_text_boxes, kind, int(page_h))
        
        current = (int(cx), int(cy), int(cw), int(ch))
        if current == prev:
            break
    return current

def expand_refined_figure_bbox_with_support_boxes(
    anchor_bbox: Tuple[int, int, int, int],
    candidate_bbox: Tuple[int, int, int, int],
    layout_support_boxes: List[Dict],
    page_w: int,
    page_h: int,
) -> Tuple[int, int, int, int]:
    if not layout_support_boxes:
        return candidate_bbox
    try:
        ax, ay, aw, ah = anchor_bbox
        cx, cy, cw, ch = candidate_bbox

        relevant = collect_relevant_figure_support_regions(anchor_bbox, [], layout_support_boxes)
        if not relevant:
            return candidate_bbox

        cx1 = cx + cw
        cy1 = cy + ch
        ax1 = ax + aw
        ay1 = ay + ah
        support_gap_max = max(96, int(ah * 0.18), int(ch * 0.28))
        support_band_gap = max(24, int(ah * 0.07))
        support_bottom_limit = min(page_h, max(ay1, cy1 + support_gap_max))
        selected: List[Tuple[int, int, int, int]] = []
        band_bottom = cy1

        for tx, ty, tw, th in sorted(relevant, key=lambda b: (b[1], b[0])):
            if ty + th < cy + int(ch * 0.45):
                continue
            gap_y = ty - cy1
            if gap_y > support_gap_max:
                continue
            if ty > support_bottom_limit:
                continue
            x_overlap = bbox_overlap_1d(cx, cx1, tx, tx + tw)
            anchor_overlap = bbox_overlap_1d(ax, ax1, tx, tx + tw)
            center_delta = abs((float(tx) + float(tx + tw)) * 0.5 - (float(cx) + float(cx1)) * 0.5)
            if x_overlap < max(24, int(min(cw, tw) * 0.16)):
                if anchor_overlap < max(32, int(min(aw, tw) * 0.22)):
                    continue
                if center_delta > max(float(cw) * 0.60, float(tw) * 0.85):
                    continue
            if selected and ty > band_bottom + support_band_gap:
                break
            selected.append((tx, ty, tw, th))
            band_bottom = max(band_bottom, ty + th)

        if not selected:
            return candidate_bbox

        nx0 = cx
        nx1 = cx1
        ny1 = cy1
        x_pad = max(8, int(float(aw) * 0.02))
        min_x = max(0, ax - x_pad)
        max_x1 = min(page_w, ax1 + x_pad)
        for tx, ty, tw, th in selected:
            nx0 = min(nx0, tx)
            nx1 = max(nx1, tx + tw)
            ny1 = max(ny1, ty + th)
        nx0 = max(min_x, nx0)
        nx1 = min(max_x1, nx1)
        ny1 = min(page_h, ny1)
        return clip_xywh_to_page(int(nx0), int(cy), int(nx1 - nx0), int(ny1 - cy), page_w, page_h)
    except Exception:
        return candidate_bbox

def clamp_padded_figure_bbox_to_support_band(
    padded_bbox: Tuple[int, int, int, int],
    unpadded_bbox: Tuple[int, int, int, int],
    layout_support_boxes: List[Dict],
    page_w: int,
    page_h: int,
) -> Tuple[int, int, int, int]:
    if not layout_support_boxes:
        return padded_bbox
    try:
        px, py, pw, ph = padded_bbox
        ux, uy, uw, uh = unpadded_bbox
        ux1 = ux + uw
        uy1 = uy + uh
        matched: List[Tuple[int, int, int, int]] = []
        for support in layout_support_boxes:
            bbox = support.get('bbox', (0, 0, 0, 0))
            sx, sy, sw, sh = bbox[:4]
            if sw <= 0 or sh <= 0:
                continue
            if sy < uy + int(uh * 0.55):
                continue
            if abs((sy + sh) - uy1) > max(10, int(sh * 0.60), TEXT_TRIM_MARGIN_PX + 2):
                continue
            x_overlap = bbox_overlap_1d(ux, ux1, sx, sx + sw)
            if x_overlap < max(24, int(min(uw, sw) * 0.16)):
                continue
            matched.append((int(sx), int(sy), int(sw), int(sh)))
        if not matched:
            return padded_bbox
        keep_bottom = max(sy + sh for _sx, sy, _sw, sh in matched) + max(4, TEXT_TRIM_MARGIN_PX)
        current_bottom = py + ph
        if keep_bottom >= current_bottom:
            return padded_bbox
        return clip_xywh_to_page(int(px), int(py), int(pw), int(keep_bottom - py), page_w, page_h)
    except Exception:
        return padded_bbox

def clamp_padded_figure_bbox_to_avoidance_constraints(
    padded_bbox: Tuple[int, int, int, int],
    unpadded_bbox: Tuple[int, int, int, int],
    avoidance_constraints: Optional[List[Dict]],
    page_w: int,
    page_h: int,
    debug_context: str = '',
) -> Tuple[int, int, int, int]:
    if not avoidance_constraints:
        return padded_bbox
    try:
        px, py, pw, ph = padded_bbox
        ux, uy, uw, uh = unpadded_bbox
        if pw <= 0 or ph <= 0 or uw <= 0 or uh <= 0:
            return padded_bbox

        current_top = py
        current_bottom = py + ph
        target_text_bottom: Optional[int] = None
        target_text_top: Optional[int] = None
        
        top_band_bottom = py + int(ph * 0.28)
        bottom_band_top = py + int(ph * 0.72)
        proximity_limit = max(6, min(24, int(ph * 0.05)))
        intrusion_limit = max(proximity_limit, min(36, int(ph * 0.12)))
        min_overlap_px = max(48, int(pw * 0.40))
        anchor_guard = max(6, int(uh * 0.04))
        anchor_top = uy
        anchor_bottom = uy + uh

        for constraint in avoidance_constraints:
            bbox = constraint.get('bbox', (0,0,0,0))
            tx, ty, tw, th = bbox[:4]
            if tw <= 0 or th <= 0: continue
            
            x_overlap = bbox_overlap_1d(px, px + pw, tx, tx + tw)
            if x_overlap < min_overlap_px: continue
            
            constraint_bottom = ty + th
            top_edge_delta = current_top - constraint_bottom
            if constraint_bottom <= top_band_bottom and constraint_bottom <= (anchor_top + anchor_guard):
                if top_edge_delta <= proximity_limit and abs(top_edge_delta) <= intrusion_limit:
                    if target_text_bottom is None or constraint_bottom > target_text_bottom:
                        target_text_bottom = constraint_bottom
                        
            edge_delta = ty - current_bottom
            if ty >= bottom_band_top and ty >= (anchor_bottom - anchor_guard):
                if edge_delta <= proximity_limit and abs(edge_delta) <= intrusion_limit:
                    if target_text_top is None or ty < target_text_top:
                        target_text_top = ty

        proposed_top = py
        proposed_bottom = current_bottom
        if target_text_bottom is not None:
            proposed_top = max(proposed_top, target_text_bottom + TEXT_TRIM_MARGIN_PX)
        if target_text_top is not None:
            proposed_bottom = min(proposed_bottom, target_text_top - TEXT_TRIM_MARGIN_PX)
            
        new_h = proposed_bottom - proposed_top
        if new_h > 0:
            area_reduction_ratio = 1.0 - (float(new_h) / float(ph))
            if area_reduction_ratio <= 0.20:
                return clip_xywh_to_page(int(px), int(proposed_top), int(pw), int(new_h), page_w, page_h)
        return padded_bbox
    except Exception:
        return padded_bbox

def trim_visual_bbox_with_image_rows(
    page_cv,
    bbox: Tuple[int, int, int, int],
    kind: str,
) -> Tuple[int, int, int, int]:
    """Fallback trim for scanned PDFs without reliable native text boxes."""
    if cv2 is None or page_cv is None:
        return bbox
    if kind not in ('figure', 'equation'):
        return bbox

    try:
        ph, pw = page_cv.shape[:2]
        x, y, w, h = clip_xywh_to_page(*bbox, pw, ph)
        if w < 220 or h < 170:
            return (x, y, w, h)

        roi = page_cv[y:y + h, x:x + w]
        if roi is None or roi.size == 0:
            return (x, y, w, h)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 9)

        k_w = max(26, int(w * 0.085))
        text_lines = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 3)), iterations=1)
        text_lines = cv2.morphologyEx(text_lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)), iterations=1)

        contours, _ = cv2.findContours(text_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            lx, ly, lw, lh = cv2.boundingRect(cnt)
            if lw < max(44, int(w * 0.22)): continue
            if lh > max(34, int(h * 0.10)): continue
            if ly < int(h * 0.16): continue
            line_boxes.append((lx, ly, lw, lh))

        if len(line_boxes) < 3: return (x, y, w, h)

        line_boxes = sorted(line_boxes, key=lambda t: t[1])
        gap_thr = max(12, int(h * 0.035))
        clusters = []
        if line_boxes:
            cur_cluster = [line_boxes[0]]
            for cur in line_boxes[1:]:
                prev_bottom = cur_cluster[-1][1] + cur_cluster[-1][3]
                if (cur[1] - prev_bottom) <= gap_thr:
                    cur_cluster.append(cur)
                else:
                    clusters.append(cur_cluster)
                    cur_cluster = [cur]
            clusters.append(cur_cluster)

        if not clusters: return (x, y, w, h)

        def _cluster_metrics(cluster):
            cy1 = max(c[1] + c[3] for c in cluster)
            width_cov = float(sum(c[2] for c in cluster)) / float(max(1, len(cluster) * w))
            return cy1, width_cov

        edge_gap_thr = max(18, int(h * 0.05))
        top_y1, top_width_cov = _cluster_metrics(clusters[0])
        if top_y1 <= int(h * 0.18) and len(clusters) >= 2 and (min(c[1] for c in clusters[1]) - top_y1) >= edge_gap_thr and top_width_cov >= 0.56:
            new_y = y + min(h - 1, top_y1 + TEXT_TRIM_MARGIN_PX)
            return clip_xywh_to_page(x, int(new_y), w, int((y + h) - new_y), pw, ph)

        bot_y0 = min(c[1] for c in clusters[-1])
        bot_y1, bot_width_cov = _cluster_metrics(clusters[-1])
        prev_gap = (bot_y0 - max(c[1] + c[3] for c in clusters[-2])) if len(clusters) >= 2 else 0
        if (bot_y0 >= int(h * 0.88) and bot_width_cov <= 0.18) or (bot_y0 >= int(h * 0.72) and bot_width_cov >= 0.58 and prev_gap >= edge_gap_thr):
            new_h = max(int(h * 0.35), min(bot_y0 - TEXT_TRIM_MARGIN_PX, h - 1))
            return (x, y, w, int(new_h))

        return (x, y, w, h)
    except Exception:
        return bbox

def shrink_visual_bbox_to_blank_edges(
    page_cv,
    bbox: Tuple[int, int, int, int],
    kind: str,
    page_text_boxes: Optional[List[Dict]] = None,
    layout_support_boxes: Optional[List[Dict]] = None,
    anchor_bbox: Optional[Tuple[int, int, int, int]] = None,
    semantic_safety_floor: Optional[int] = None,
    semantic_text_boxes: Optional[List[Dict]] = None,
    debug_context: str = '',
) -> Tuple[int, int, int, int]:
    """Trim stray edge signal until a real blank band is reached."""
    if cv2 is None or page_cv is None or kind not in ('figure', 'equation'):
        return bbox

    try:
        ph, pw = page_cv.shape[:2]
        x, y, w, h = clip_xywh_to_page(*bbox, pw, ph)
        if w < 120 or h < 120: return (x, y, w, h)

        roi = page_cv[y:y + h, x:x + w]
        mask = build_visual_signal_mask(roi)
        if mask is None or mask.size == 0: return (x, y, w, h)

        total_density = float(cv2.countNonZero(mask)) / float(max(1, mask.size))
        if total_density <= 0.004: return (x, y, w, h)

        blank_thr = max(0.0025, min(0.016, total_density * 0.42))
        signal_thr = max(blank_thr * 2.4, min(0.032, total_density * 1.05))

        def _scan_side(side):
            limit = max(8, min(48, int(w * 0.1))) if side in ('left', 'right') else max(8, min(56, int(h * 0.1)))
            seen_signal = False
            for offset in range(limit):
                if side == 'left': region = mask[:, offset:offset+2]
                elif side == 'right': region = mask[:, w-offset-2:w-offset]
                elif side == 'top': region = mask[offset:offset+2, :]
                else: region = mask[h-offset-2:h-offset, :]
                
                density = float(cv2.countNonZero(region)) / float(max(1, region.size))
                if density >= signal_thr: seen_signal = True
                elif seen_signal and density <= blank_thr: return offset
            return 0

        l, r, t, b = _scan_side('left'), _scan_side('right'), _scan_side('top'), _scan_side('bottom')
        return clip_xywh_to_page(x+l, y+t, w-l-r, h-t-b, pw, ph)
    except Exception:
        return bbox

def figure_box_looks_self_contained(page_cv, bbox: Tuple[int, int, int, int]) -> bool:
    if cv2 is None or page_cv is None: return True
    try:
        ph, pw = page_cv.shape[:2]
        x, y, w, h = clip_xywh_to_page(*bbox, pw, ph)
        if w < 120 or h < 120: return True
        roi = page_cv[y:y + h, x:x + w]
        mask = build_visual_signal_mask(roi)
        if mask is None: return True
        
        density = float(cv2.countNonZero(mask)) / float(max(1, mask.size))
        if density <= 0.01: return True
        
        edge_w, edge_h = max(6, int(w*0.035)), max(6, int(h*0.035))
        edges = [mask[:edge_h, :], mask[h-edge_h:, :], mask[:, :edge_w], mask[:, w-edge_w:]]
        strong = sum(1 for e in edges if (float(cv2.countNonZero(e))/max(1, e.size)) >= max(0.028, density*1.45))
        return strong < 3
    except Exception: return True
