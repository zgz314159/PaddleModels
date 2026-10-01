"""Legacy vs v2 table cell structure diff (diagnosis only — no algorithm tuning)."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value if value is not None else "")).strip()


def _canonical_rows_grid(container: Dict[str, Any]) -> List[Any]:
    """Canonical 2D grid. Prefer a list-valued `rows` (even empty); else list `table_rows`."""
    rows = container.get("rows")
    if isinstance(rows, list):
        return rows
    table_rows = container.get("table_rows")
    if isinstance(table_rows, list):
        return table_rows
    return []


def _legacy_numeric_count(value: Any) -> Optional[int]:
    """Return a genuine numeric row/col count; None for lists/dicts/None/non-numeric strings."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _bbox_overlap_ratio(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    """Fraction of a's area covered by intersection with b (xywh)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0:
        return 0.0
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax + aw, bx + bw)
    iy1 = min(ay + ah, by + bh)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area = aw * ah
    return inter / area if area > 0 else 0.0


def _block_bbox_xywh(block: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bb = block.get("bbox")
    if isinstance(bb, dict) and {"left", "top", "right", "bottom"} <= set(bb):
        left = float(bb["left"])
        top = float(bb["top"])
        right = float(bb["right"])
        bottom = float(bb["bottom"])
        return (left, top, right - left, bottom - top)
    if isinstance(bb, dict) and {"x", "y", "w", "h"} <= set(bb):
        return (float(bb["x"]), float(bb["y"]), float(bb["w"]), float(bb["h"]))
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        x0, y0, x1, y1 = map(float, bb[:4])
        return (x0, y0, x1 - x0, y1 - y0)
    return None


def _table_geometry_stats(table: Dict[str, Any], side: str) -> Dict[str, Any]:
    """Logical grid vs physical cells — never count empty span placeholders as physical."""
    rows_list = _canonical_rows_grid(table)
    grid_rows = len(rows_list)
    grid_cols = max((len(r) for r in rows_list if isinstance(r, list)), default=0)

    raw_cells = table.get("cells_raw")
    if raw_cells is None and side == "v2":
        raw_cells = table.get("cells")
    if raw_cells is None and isinstance(table.get("cells"), list):
        raw_cells = table.get("cells")

    span_data_available = isinstance(raw_cells, list) and len(raw_cells) > 0

    if span_data_available:
        physical = len(raw_cells)
        nonempty = sum(
            1 for c in raw_cells
            if isinstance(c, dict) and _norm_text(c.get("text"))
        )
        spanned = sum(
            1 for c in raw_cells
            if isinstance(c, dict)
            and (int(c.get("rowSpan") or 1) > 1 or int(c.get("colSpan") or 1) > 1)
        )
        # Prefer declared rows/cols or derive from cells
        if grid_rows == 0:
            grid_rows = max(
                (int(c.get("row") or 0) + max(1, int(c.get("rowSpan") or 1))
                 for c in raw_cells if isinstance(c, dict)),
                default=0,
            )
        if grid_cols == 0:
            grid_cols = max(
                (int(c.get("col") or 0) + max(1, int(c.get("colSpan") or 1))
                 for c in raw_cells if isinstance(c, dict)),
                default=0,
            )
        # Honor explicit NUMERIC rows/cols counts only (never int() a 2D grid).
        legacy_rows = _legacy_numeric_count(table.get("rows"))
        if legacy_rows:
            grid_rows = legacy_rows
        legacy_cols = _legacy_numeric_count(table.get("cols"))
        if legacy_cols:
            grid_cols = legacy_cols
    else:
        # Legacy (or v2 without cells): count non-empty text slots as split units.
        physical = 0
        nonempty = 0
        spanned = 0
        for row in rows_list:
            if not isinstance(row, list):
                continue
            for text in row:
                if _norm_text(text):
                    physical += 1
                    nonempty += 1

    return {
        "logical_rows": int(grid_rows),
        "logical_cols": int(grid_cols),
        "grid_slots": int(grid_rows) * int(grid_cols),
        "physical_cells": int(physical),
        "nonempty_cells": int(nonempty),
        "spanned_cells": int(spanned),
        "span_data_available": bool(span_data_available),
    }


def collect_legacy_tables(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    for ei, entry in enumerate((kb or {}).get("entries") or []):
        for bi, block in enumerate(entry.get("blocks") or []):
            if not isinstance(block, dict) or block.get("type") != "table":
                continue
            grid = _canonical_rows_grid(block)
            nonempty = 0
            for row in grid:
                if isinstance(row, list):
                    nonempty += sum(1 for c in row if _norm_text(c))
            stats = _table_geometry_stats(
                {
                    "table_rows": grid,
                    "cells": block.get("cells"),
                    "cells_raw": None,
                    "rows": _legacy_numeric_count(block.get("rows")),
                    "cols": _legacy_numeric_count(block.get("cols")),
                },
                side="legacy",
            )
            tables.append(
                {
                    "side": "legacy",
                    "entry_index": ei,
                    "block_index": bi,
                    "id": block.get("id"),
                    "page_number": int(block.get("pageNumber") or 0),
                    "bbox": _block_bbox_xywh(block),
                    "rows": stats["logical_rows"] or len(grid),
                    "cols": stats["logical_cols"],
                    "nonempty_cells": nonempty,
                    "table_rows": [[str(c) for c in r] if isinstance(r, list) else [] for r in grid],
                    "cells_raw": None,  # legacy has no geometric cells → span unavailable
                    "geometry": stats,
                }
            )
    return tables


def collect_v2_tables(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    """From v2 KB table blocks (prefer cells[] for physical geometry)."""
    tables: List[Dict[str, Any]] = []
    for ei, entry in enumerate((kb or {}).get("entries") or []):
        for bi, block in enumerate(entry.get("blocks") or []):
            if not isinstance(block, dict) or block.get("type") != "table":
                continue
            grid = _canonical_rows_grid(block)
            cells = block.get("cells")
            stats = _table_geometry_stats(
                {
                    "table_rows": grid,
                    "cells": cells,
                    "rows": _legacy_numeric_count(block.get("rows")),
                    "cols": _legacy_numeric_count(block.get("cols")),
                },
                side="v2",
            )
            nonempty = stats["nonempty_cells"]
            if not stats["span_data_available"]:
                nonempty = 0
                for row in grid:
                    if isinstance(row, list):
                        nonempty += sum(1 for c in row if _norm_text(c))
            tables.append(
                {
                    "side": "v2",
                    "entry_index": ei,
                    "block_index": bi,
                    "id": block.get("id"),
                    "page_number": int(block.get("pageNumber") or 0),
                    "bbox": _block_bbox_xywh(block),
                    "rows": stats["logical_rows"] or len(grid),
                    "cols": stats["logical_cols"],
                    "nonempty_cells": nonempty,
                    "table_rows": [[str(c) for c in r] if isinstance(r, list) else [] for r in grid],
                    "cells_raw": cells if isinstance(cells, list) else None,
                    "structureStatus": block.get("structureStatus"),
                    "geometry": stats,
                }
            )
    return tables


def match_tables(
    legacy_tables: Sequence[Dict[str, Any]],
    v2_tables: Sequence[Dict[str, Any]],
    *,
    min_overlap: float = 0.3,
) -> List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]]:
    """Match by pageNumber + bbox overlap; fallback page + order."""
    pairs: List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]] = []
    used_v: set = set()

    for lt in legacy_tables:
        match = None
        method = "unmatched"
        lb = lt.get("bbox")
        if lb is not None:
            for i, vt in enumerate(v2_tables):
                if i in used_v:
                    continue
                if vt.get("page_number") != lt.get("page_number"):
                    continue
                vb = vt.get("bbox")
                if vb is None:
                    continue
                if _bbox_overlap_ratio(lb, vb) >= min_overlap or _bbox_overlap_ratio(vb, lb) >= min_overlap:
                    match = i
                    method = "page+bbox"
                    break
        if match is None:
            # Fallback: page + order — pair i-th legacy on page with i-th v2 on page.
            leg_idx = 0
            for j, prev in enumerate(legacy_tables):
                if prev is lt:
                    leg_idx = sum(
                        1
                        for t2 in legacy_tables[:j]
                        if t2.get("page_number") == lt.get("page_number")
                    )
                    break
            v_on_page = [
                (i, t)
                for i, t in enumerate(v2_tables)
                if t.get("page_number") == lt.get("page_number")
            ]
            if leg_idx < len(v_on_page):
                vi = v_on_page[leg_idx][0]
                if vi not in used_v:
                    match = vi
                    method = "page+order"

        if match is not None:
            used_v.add(match)
            pairs.append((lt, v2_tables[match], method))
        else:
            pairs.append((lt, None, method))

    for i, vt in enumerate(v2_tables):
        if i not in used_v:
            pairs.append((None, vt, "unmatched"))
    return pairs


def _expand_rows_cols(table: Dict[str, Any]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """Flatten table_rows (and cells if present) to keyed cells."""
    cells: Dict[Tuple[int, int], Dict[str, Any]] = {}
    rows = table.get("table_rows") or []
    for r, row in enumerate(rows):
        if not isinstance(row, list):
            continue
        for c, text in enumerate(row):
            cells[(r, c)] = {
                "row": r,
                "col": c,
                "rowSpan": 1,
                "colSpan": 1,
                "text": _norm_text(text),
                "raw_text": str(text) if text is not None else "",
            }
    # Overlay explicit cells when present (legacy may use cells list)
    raw_cells = table.get("cells_raw")
    if isinstance(raw_cells, list):
        for cell in raw_cells:
            if not isinstance(cell, dict):
                continue
            try:
                r = int(cell.get("row") or 0)
                c = int(cell.get("col") or 0)
            except (TypeError, ValueError):
                continue
            cells[(r, c)] = {
                "row": r,
                "col": c,
                "rowSpan": int(cell.get("rowSpan") or 1),
                "colSpan": int(cell.get("colSpan") or 1),
                "text": _norm_text(cell.get("text")),
                "raw_text": str(cell.get("text") or ""),
                "bbox": cell.get("bbox"),
            }
    return cells


def cells_diff(
    legacy_table: Dict[str, Any],
    v2_table: Dict[str, Any],
) -> Dict[str, Any]:
    leg = _expand_rows_cols(legacy_table)
    v2 = _expand_rows_cols(v2_table)

    only_legacy: List[Dict[str, Any]] = []
    only_v2: List[Dict[str, Any]] = []
    span_or_pos_diff: List[Dict[str, Any]] = []
    matched_same: List[Dict[str, Any]] = []

    all_keys = sorted(set(leg) | set(v2), key=lambda k: (k[0], k[1]))
    # Also match by normalized text across positions when row/col differ
    leg_by_text: Dict[str, List[Tuple[int, int]]] = {}
    for k, cell in leg.items():
        leg_by_text.setdefault(cell["text"], []).append(k)
    v2_by_text: Dict[str, List[Tuple[int, int]]] = {}
    for k, cell in v2.items():
        v2_by_text.setdefault(cell["text"], []).append(k)

    matched_leg_keys: set = set()
    matched_v2_keys: set = set()

    for key in all_keys:
        lc = leg.get(key)
        vc = v2.get(key)
        if lc and vc:
            matched_leg_keys.add(key)
            matched_v2_keys.add(key)
            diffs = []
            if lc["rowSpan"] != vc["rowSpan"]:
                diffs.append("rowSpan")
            if lc["colSpan"] != vc["colSpan"]:
                diffs.append("colSpan")
            if lc["text"] != vc["text"]:
                diffs.append("text")
            if diffs:
                span_or_pos_diff.append(
                    {
                        "key": list(key),
                        "diff_fields": diffs,
                        "legacy": lc,
                        "v2": vc,
                    }
                )
            else:
                matched_same.append({"key": list(key), "text": lc["text"]})
        elif lc and not vc:
            # try text match elsewhere in v2
            candidates = [k for k in v2_by_text.get(lc["text"], []) if k not in matched_v2_keys]
            if candidates:
                vk = candidates[0]
                matched_v2_keys.add(vk)
                matched_leg_keys.add(key)
                span_or_pos_diff.append(
                    {
                        "key": list(key),
                        "diff_fields": ["position"],
                        "legacy": lc,
                        "v2": {**v2[vk], "matched_key": list(vk)},
                    }
                )
            else:
                only_legacy.append(lc)
                matched_leg_keys.add(key)
        elif vc and not lc:
            candidates = [k for k in leg_by_text.get(vc["text"], []) if k not in matched_leg_keys]
            if candidates:
                lk = candidates[0]
                matched_leg_keys.add(lk)
                matched_v2_keys.add(key)
                span_or_pos_diff.append(
                    {
                        "key": list(key),
                        "diff_fields": ["position"],
                        "legacy": {**leg[lk], "matched_key": list(lk)},
                        "v2": vc,
                    }
                )
            else:
                only_v2.append(vc)
                matched_v2_keys.add(key)

    return {
        "legacy_nonempty_cells": sum(1 for c in leg.values() if c["text"]),
        "v2_nonempty_cells": sum(1 for c in v2.values() if c["text"]),
        "legacy_cell_count": len(leg),
        "v2_cell_count": len(v2),
        "only_legacy": only_legacy,
        "only_v2": only_v2,
        "span_or_position_diffs": span_or_pos_diff,
        "matched_identical_count": len(matched_same),
        "legacy_rows": legacy_table.get("rows"),
        "legacy_cols": legacy_table.get("cols"),
        "v2_rows": v2_table.get("rows"),
        "v2_cols": v2_table.get("cols"),
        "legacy_geometry": legacy_table.get("geometry"),
        "v2_geometry": v2_table.get("geometry"),
        "counting_note": (
            "legacy nonempty counts text-split slots from table_rows; "
            "v2 physical_cells counts geometric cells with spans when cells[] present. "
            "These are different denominators — do not force them equal."
        ),
    }


def _markdown_preview(rows: Sequence[Sequence[str]], limit_rows: int = 8) -> str:
    if not rows:
        return ""
    lines = []
    for row in list(rows)[:limit_rows]:
        lines.append("| " + " | ".join(_norm_text(c) for c in row) + " |")
    if len(rows) > limit_rows:
        lines.append(f"... ({len(rows) - limit_rows} more rows)")
    return "\n".join(lines)


def build_table_diff_report(
    legacy_kb: Dict[str, Any],
    v2_kb: Dict[str, Any],
    *,
    page_number: Optional[int] = None,
) -> Dict[str, Any]:
    leg_tables = collect_legacy_tables(legacy_kb)
    v2_tables = collect_v2_tables(v2_kb)
    if page_number is not None:
        leg_tables = [t for t in leg_tables if t["page_number"] == page_number]
        v2_tables = [t for t in v2_tables if t["page_number"] == page_number]

    pairs = match_tables(leg_tables, v2_tables)
    comparisons = []
    for lt, vt, method in pairs:
        if lt and vt:
            cell_diff = cells_diff(lt, vt)
            comparisons.append(
                {
                    "match_method": method,
                    "legacy": {
                        "id": lt.get("id"),
                        "page_number": lt.get("page_number"),
                        "rows": lt.get("rows"),
                        "cols": lt.get("cols"),
                        "nonempty_cells": lt.get("nonempty_cells"),
                        "bbox": lt.get("bbox"),
                        "geometry": lt.get("geometry"),
                    },
                    "v2": {
                        "id": vt.get("id"),
                        "page_number": vt.get("page_number"),
                        "rows": vt.get("rows"),
                        "cols": vt.get("cols"),
                        "nonempty_cells": vt.get("nonempty_cells"),
                        "bbox": vt.get("bbox"),
                        "structureStatus": vt.get("structureStatus"),
                        "geometry": vt.get("geometry"),
                    },
                    "cells": cell_diff,
                    "legacy_markdown_preview": _markdown_preview(lt.get("table_rows") or []),
                    "v2_markdown_preview": _markdown_preview(vt.get("table_rows") or []),
                }
            )
        elif lt:
            comparisons.append(
                {
                    "match_method": method,
                    "legacy": {"id": lt.get("id"), "page_number": lt.get("page_number")},
                    "v2": None,
                    "note": "legacy-only table",
                }
            )
        elif vt:
            comparisons.append(
                {
                    "match_method": method,
                    "legacy": None,
                    "v2": {"id": vt.get("id"), "page_number": vt.get("page_number")},
                    "note": "v2-only table",
                }
            )

    # Summary of nonempty cell delta for primary paired tables
    cell_count_notes: List[str] = []
    for c in comparisons:
        if c.get("cells"):
            cd = c["cells"]
            lg = cd.get("legacy_geometry") or {}
            vg = cd.get("v2_geometry") or {}
            cell_count_notes.append(
                f"page={c['legacy'].get('page_number')}: "
                f"legacy_text_slots={cd['legacy_nonempty_cells']} "
                f"(span_data={lg.get('span_data_available', False)}) | "
                f"v2_physical={vg.get('physical_cells', cd['v2_nonempty_cells'])} "
                f"grid_slots={vg.get('grid_slots')} "
                f"spanned={vg.get('spanned_cells')} "
                f"span_data={vg.get('span_data_available')} | "
                f"only_legacy={len(cd['only_legacy'])} "
                f"only_v2={len(cd['only_v2'])} "
                f"span_or_pos={len(cd['span_or_position_diffs'])} | "
                f"counting_note=legacy 23-style counts are text-split units; "
                f"v2 physical cells use geometry+spans — do not equate"
            )

    return {
        "generated_by": "pipeline.compatibility.table_diff",
        "page_filter": page_number,
        "legacy_table_count": len(leg_tables),
        "v2_table_count": len(v2_tables),
        "comparisons": comparisons,
        "cell_count_notes": cell_count_notes,
        "note": (
            "Diagnosis only — does not decide which side is correct. "
            "Counts alone are insufficient; inspect only_* and span diffs."
        ),
    }
