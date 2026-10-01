"""Native PyMuPDF table detection + native/CV merge helpers (Phase 2A).

Pure merge/suppress helpers are PDF-free for unit tests. Detection requires fitz.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from imaging.bbox_utils import bbox_overlap_ratio_xywh
from pipeline.canonical_ir import BBox, DocBlock

try:
    import fitz
except ImportError:
    fitz = None  # type: ignore

try:
    from imaging.table_processor import extract_native_table_cells
except ImportError:
    extract_native_table_cells = None  # type: ignore


def cells_to_table_rows(cells: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """Convert cell dicts (row/col/text) into a dense row-major string matrix."""
    if not cells:
        return []
    max_r = max(int(c.get("row") or 0) for c in cells)
    max_c = max(int(c.get("col") or 0) for c in cells)
    grid: List[List[str]] = [["" for _ in range(max_c + 1)] for _ in range(max_r + 1)]
    for c in cells:
        try:
            r = int(c.get("row") or 0)
            col = int(c.get("col") or 0)
        except (TypeError, ValueError):
            continue
        if 0 <= r <= max_r and 0 <= col <= max_c:
            grid[r][col] = str(c.get("text") or "")
    return grid


def escape_markdown_cell(value: str) -> str:
    """Safe cell text for Markdown table pipes/newlines."""
    s = str(value if value is not None else "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\n", " ").replace("|", "\\|")
    return s.strip()


def table_rows_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    """Deterministic Markdown table from row matrix. Empty → ''."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    norm = [list(r) + [""] * (width - len(r)) for r in rows]
    # Header separator after first row (or synthetic if single row).
    header = norm[0]
    body = norm[1:] if len(norm) > 1 else []
    lines = [
        "| " + " | ".join(escape_markdown_cell(c) for c in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(escape_markdown_cell(c) for c in row) + " |")
    return "\n".join(lines)


def _bbox_xywh(b: Any) -> Tuple[float, float, float, float]:
    if isinstance(b, DocBlock):
        return (b.bbox.x, b.bbox.y, b.bbox.w, b.bbox.h)
    if isinstance(b, dict):
        bb = b.get("bbox") or b
        if isinstance(bb, dict):
            return (float(bb.get("x", 0)), float(bb.get("y", 0)), float(bb.get("w", 0)), float(bb.get("h", 0)))
        if isinstance(bb, (list, tuple)) and len(bb) >= 4:
            # xyxy or xywh ambiguous — treat as xyxy if right>left style large
            x0, y0, x1, y1 = map(float, bb[:4])
            if x1 > x0 and y1 > y0 and (x1 - x0) > 0 and x0 < 10000:
                # Heuristic: PyMuPDF find_tables returns (x0,y0,x1,y1)
                return (x0, y0, x1 - x0, y1 - y0)
            return (x0, y0, x1, y1)
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        x0, y0, x1, y1 = map(float, b[:4])
        return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))
    return (0.0, 0.0, 0.0, 0.0)


def boxes_overlap_xywh(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    min_ratio: float = 0.3,
) -> bool:
    """True when either box covers ≥ min_ratio of its own area by the intersection."""
    ia = (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    ib = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    # bbox_overlap_ratio_xywh(inner, outer): fraction of inner covered by outer
    r1 = bbox_overlap_ratio_xywh(ia, ib)
    r2 = bbox_overlap_ratio_xywh(ib, ia)
    return r1 >= min_ratio or r2 >= min_ratio


def merge_native_and_visual_tables(
    native_tables: Sequence[DocBlock],
    visual_tables: Sequence[DocBlock],
    *,
    min_overlap: float = 0.3,
) -> List[DocBlock]:
    """Match native structured tables with CV visual table bboxes.

    - Overlap → single block: native cells win; visual bbox kept for crop if larger.
    - Unmatched native → structured (or image_only if no cells).
    - Unmatched CV → image_only only (never structured).
    """
    used_visual: set = set()
    merged: List[DocBlock] = []

    for nt in native_tables:
        cells = list(nt.metadata.get("cells") or [])
        structure = "structured" if cells else "image_only"
        n_box = (nt.bbox.x, nt.bbox.y, nt.bbox.w, nt.bbox.h)
        match_idx = None
        for i, vt in enumerate(visual_tables):
            if i in used_visual:
                continue
            v_box = (vt.bbox.x, vt.bbox.y, vt.bbox.w, vt.bbox.h)
            if boxes_overlap_xywh(n_box, v_box, min_overlap):
                match_idx = i
                break
        meta = dict(nt.metadata)
        meta["structureStatus"] = structure
        meta["detectionSource"] = meta.get("detectionSource") or "pymupdf_native"
        if match_idx is not None:
            used_visual.add(match_idx)
            vt = visual_tables[match_idx]
            # Prefer larger visual bbox for crop when it better covers the table.
            if vt.bbox.w * vt.bbox.h > nt.bbox.w * nt.bbox.h:
                bbox = BBox(vt.bbox.x, vt.bbox.y, vt.bbox.w, vt.bbox.h)
            else:
                bbox = nt.bbox
            meta["visualMatched"] = True
        else:
            bbox = nt.bbox
            meta["visualMatched"] = False
        meta["rows"] = int(meta.get("rows") or _rows_from_cells(cells))
        meta["cols"] = int(meta.get("cols") or _cols_from_cells(cells))
        merged.append(
            DocBlock(
                id=nt.id,
                type="table",
                text=nt.text,
                bbox=bbox,
                page_number=nt.page_number,
                reading_order=nt.reading_order,
                confidence=nt.confidence,
                source=nt.source,
                metadata=meta,
            )
        )

    for i, vt in enumerate(visual_tables):
        if i in used_visual:
            continue
        meta = dict(vt.metadata)
        meta["structureStatus"] = "image_only"
        meta["detectionSource"] = meta.get("detectionSource") or vt.source or "cv_fallback"
        meta["cells"] = meta.get("cells") or []
        meta["rows"] = int(meta.get("rows") or 0)
        meta["cols"] = int(meta.get("cols") or 0)
        # Never mark cv_fallback_weak as structured.
        if meta.get("source") == "cv_fallback_weak":
            meta["structureStatus"] = "image_only"
        merged.append(
            DocBlock(
                id=vt.id,
                type="table",
                text=vt.text,
                bbox=vt.bbox,
                page_number=vt.page_number,
                reading_order=vt.reading_order,
                confidence=vt.confidence,
                source=vt.source,
                metadata=meta,
            )
        )

    return merged


def _rows_from_cells(cells: Sequence[Dict[str, Any]]) -> int:
    if not cells:
        return 0
    return max(int(c.get("row") or 0) for c in cells) + 1


def _cols_from_cells(cells: Sequence[Dict[str, Any]]) -> int:
    if not cells:
        return 0
    return max(int(c.get("col") or 0) for c in cells) + 1


def should_suppress_native_text(
    text_box: Tuple[float, float, float, float],
    table_meta: Dict[str, Any],
    *,
    overlap_threshold: float = 0.8,
) -> bool:
    """Only structured tables may suppress overlapping native text.

    image_only / unknown structure → keep text (avoid silent content loss).
    """
    status = str(table_meta.get("structureStatus") or "")
    if status != "structured":
        return False
    t_box = table_meta.get("_bbox_xywh")
    if not t_box:
        # Prefer explicit fields if present
        if all(k in table_meta for k in ("x", "y", "w", "h")):
            t_box = (table_meta["x"], table_meta["y"], table_meta["w"], table_meta["h"])
        else:
            return False
    ratio = bbox_overlap_ratio_xywh(
        (int(text_box[0]), int(text_box[1]), int(text_box[2]), int(text_box[3])),
        (int(t_box[0]), int(t_box[1]), int(t_box[2]), int(t_box[3])),
    )
    return ratio >= overlap_threshold


def suppress_text_for_structured_tables(
    native_blocks: Sequence[DocBlock],
    table_blocks: Sequence[DocBlock],
    *,
    overlap_threshold: float = 0.8,
) -> List[DocBlock]:
    """Drop native text blocks highly overlapped ONLY by structured tables."""
    structured_boxes = []
    for tb in table_blocks:
        if (tb.metadata or {}).get("structureStatus") == "structured":
            structured_boxes.append((tb.bbox.x, tb.bbox.y, tb.bbox.w, tb.bbox.h))

    if not structured_boxes:
        return list(native_blocks)

    kept: List[DocBlock] = []
    for nb in native_blocks:
        if nb.type in ("table", "image", "figure"):
            kept.append(nb)
            continue
        if nb.type not in ("text", "heading", "caption"):
            kept.append(nb)
            continue
        n_box = (nb.bbox.x, nb.bbox.y, nb.bbox.w, nb.bbox.h)
        suppress = False
        for s_box in structured_boxes:
            ratio = bbox_overlap_ratio_xywh(
                (int(n_box[0]), int(n_box[1]), int(n_box[2]), int(n_box[3])),
                (int(s_box[0]), int(s_box[1]), int(s_box[2]), int(s_box[3])),
            )
            if ratio >= overlap_threshold:
                suppress = True
                break
        if not suppress:
            kept.append(nb)
    return kept


def detect_native_table_blocks(page: Any, page_number: int) -> List[DocBlock]:
    """Run page.find_tables() and convert to Canonical IR table DocBlocks.

    Returns [] when fitz/table extraction unavailable — never fabricates cells.
    """
    if fitz is None or extract_native_table_cells is None or page is None:
        return []
    try:
        finder = page.find_tables()
        tables = list(getattr(finder, "tables", None) or [])
    except Exception:
        return []

    blocks: List[DocBlock] = []
    for idx, table in enumerate(tables):
        try:
            bbox_xyxy = tuple(table.bbox)  # x0,y0,x1,y1
            x0, y0, x1, y1 = (float(v) for v in bbox_xyxy[:4])
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            if w <= 0 or h <= 0:
                continue
            cells = extract_native_table_cells(table) or []
            rows = 0
            cols = 0
            try:
                rows = int(getattr(table, "row_count", 0) or 0)
                cols = int(getattr(table, "col_count", 0) or 0)
            except Exception:
                rows = _rows_from_cells(cells)
                cols = _cols_from_cells(cells)
            if not rows:
                rows = _rows_from_cells(cells)
            if not cols:
                cols = _cols_from_cells(cells)
            structure = "structured" if cells else "image_only"
            blocks.append(
                DocBlock(
                    id=f"p{page_number}_tbl{idx + 1}",
                    type="table",
                    text="",
                    bbox=BBox(x0, y0, w, h),
                    page_number=page_number,
                    reading_order=0,
                    source="pymupdf_native",
                    metadata={
                        "cells": cells,
                        "rows": rows,
                        "cols": cols,
                        "structureStatus": structure,
                        "detectionSource": "pymupdf_native",
                    },
                )
            )
        except Exception:
            continue
    return blocks
