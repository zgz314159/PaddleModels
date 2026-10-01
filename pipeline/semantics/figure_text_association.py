"""Same-page figure ↔ native text association (caption / legend) — Phase 2E.

Pure semantic post-pass. Does not extract images (FigureAdapter) and does not
modify legacy. Runs after page blocks are assembled, before SemanticProjector.

Phase 2E.1: global deterministic pair matching; tightened legend eligibility.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from pipeline.canonical_ir import DocBlock, DocPage

# --- Centralized geometry thresholds (points) ---
MAX_VERTICAL_GAP_PT = 48.0
MIN_HORIZONTAL_OVERLAP_RATIO = 0.20
CENTER_ALIGN_TOLERANCE = 0.35
LEGEND_MAX_VERTICAL_GAP_PT = 40.0
MIN_BELOW_FIGURE_FRACTION = 0.85

# Reference patterns that must never become caption.
_REF_PATTERNS = (
    re.compile(r"见图\s*\d+", re.IGNORECASE),
    re.compile(r"如图\s*\d+", re.IGNORECASE),
    re.compile(r"[（(]\s*图\s*\d+\s*[）)]"),
    re.compile(r"参见图\s*\d+", re.IGNORECASE),
    re.compile(r"图\s*\d+\s*所示"),
)

# Panel markers: (a), （ｂ）, (1), （1） — used for strength checks, not alone as body→legend.
_LEGEND_PART_RE = re.compile(r"[（(][a-zA-Z０-９1-9ａ-ｚA-Z][）)]")
_LEGEND_LETTER_RE = re.compile(r"[（(]([a-zA-Zａ-ｚＡ-Ｚ])[）)]")

_CAPTION_START_RE = re.compile(r"^图\s*[0-9０-９一二三四五六七八九十]+")


def _block_bbox_xyxy(block: DocBlock) -> Tuple[float, float, float, float]:
    b = block.bbox
    return (b.x, b.y, b.x + b.w, b.y + b.h)


def _block_text(block: DocBlock) -> str:
    return (block.text or "").strip()


def _role(block: DocBlock) -> str:
    return str((block.metadata or {}).get("semanticRole") or "").strip()


def _source(block: DocBlock) -> str:
    return str(getattr(block, "source", "") or "").strip()


def is_reference_only(text: str) -> bool:
    """True if text is primarily a cross-reference (见图N / （图N）), not a caption."""
    t = (text or "").strip()
    if not t:
        return True
    for pat in _REF_PATTERNS:
        if pat.search(t):
            return True
    return False


def _looks_like_body_sentence(text: str) -> bool:
    t = text.strip()
    if re.search(r"[。；;，,：:]", t) and len(t) > 30:
        return True
    if re.match(r"^\d+\.\d+", t):
        return True
    if len(t) > 60:
        return True
    return False


def has_strong_legend_markers(text: str) -> bool:
    """Strong multi-panel evidence: ≥2 distinct (a)/(b)-style markers.

    Single (1) / (a) alone is NOT strong (prevents body list upgrade).
    """
    t = re.sub(r"\s+", "", text or "")
    if not t or len(t) > 120:
        return False
    if is_reference_only(t):
        return False
    parts = _LEGEND_PART_RE.findall(t)
    if len(parts) >= 2:
        return True
    letters = {m.group(1).lower() for m in _LEGEND_LETTER_RE.finditer(t)}
    if len(letters) >= 2:
        return True
    return False


def looks_like_legend(text: str) -> bool:
    """Kept for API compat: strong multi-panel legend markers only."""
    return has_strong_legend_markers(text)


def looks_like_primary_caption(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 80:
        return False
    if is_reference_only(t):
        return False
    if _looks_like_body_sentence(t):
        return False
    if _CAPTION_START_RE.match(re.sub(r"\s+", "", t)):
        return True
    return False


def _horizontal_ok(
    fbb: Tuple[float, float, float, float],
    tbb: Tuple[float, float, float, float],
) -> bool:
    fx0, _, fx1, _ = fbb
    tx0, _, tx1, _ = tbb
    fw = max(1e-6, fx1 - fx0)
    tw = max(0.0, tx1 - tx0)
    inter = max(0.0, min(fx1, tx1) - max(fx0, tx0))
    if inter / fw >= MIN_HORIZONTAL_OVERLAP_RATIO:
        return True
    if inter / max(tw, 1e-6) >= MIN_HORIZONTAL_OVERLAP_RATIO:
        return True
    fcx = (fx0 + fx1) / 2.0
    tcx = (tx0 + tx1) / 2.0
    if abs(fcx - tcx) <= CENTER_ALIGN_TOLERANCE * fw:
        return True
    return False


def _vertical_gap_ok(
    fbb: Tuple[float, float, float, float],
    tbb: Tuple[float, float, float, float],
    max_gap: float,
) -> bool:
    _, fy0, _, fy1 = fbb
    _, ty0, _, _ = tbb
    fh = max(1e-6, fy1 - fy0)
    if ty0 < fy0 + MIN_BELOW_FIGURE_FRACTION * fh:
        if ty0 < fy0 + 0.5 * fh:
            return False
    gap = ty0 - fy1
    return gap >= -4.0 and gap <= max_gap


def _candidate_kind(block: DocBlock) -> Optional[str]:
    """Return 'caption' | 'legend' | None.

    Only native-sourced text blocks are eligible (OCR never masquerades as native).
    """
    text = _block_text(block)
    if not text:
        return None
    # Must be native source — OCR/visual blocks are not association candidates.
    if _source(block) != "native":
        return None
    if block.type not in ("text", "heading", "caption"):
        if _role(block) not in ("caption", "legend", "figure_callout"):
            return None
    if is_reference_only(text):
        return None

    role = _role(block)

    # --- caption ---
    if role == "caption" or block.type == "caption":
        if not is_reference_only(text):
            return "caption"
        return None

    # --- legend (tightened) ---
    if role == "legend":
        return "legend"
    if role == "figure_callout":
        # Only strong multi-panel markers upgrade figure_callout → legend.
        if has_strong_legend_markers(text):
            return "legend"
        return None
    # body / empty / other roles: NEVER legend (no distance-only upgrade)
    # body also never caption by proximity.
    return None


def _pair_sort_key(
    kind: str,
    fbb: Tuple[float, float, float, float],
    tbb: Tuple[float, float, float, float],
    role: str,
    fig_id: str,
    text_id: str,
) -> Tuple:
    """Global deterministic order: role priority, explicit role, geometry, stable ids.

    Lower sorts first. Distance before figure id so closer figure wins regardless
    of input list order.
    """
    _, fy0, _, fy1 = fbb
    _, ty0, _, _ = tbb
    gap = max(0.0, ty0 - fy1)
    fcx = (fbb[0] + fbb[2]) / 2.0
    tcx = (tbb[0] + tbb[2]) / 2.0
    center_d = abs(fcx - tcx)
    kind_rank = {"caption": 0, "legend": 1}.get(kind, 2)
    role_rank = 0 if role in ("caption", "legend") else 1
    return (kind_rank, role_rank, gap, center_d, fig_id, text_id)


def associate_page_texts(page: DocPage) -> Dict[str, Any]:
    """Global deterministic association of same-page figures with native texts."""
    figures = [b for b in page.blocks if b.type in ("figure", "image")]
    texts = [
        b
        for b in page.blocks
        if b.type not in ("figure", "image", "table") and (b.text or "").strip()
    ]

    stats = {
        "figures": len(figures),
        "caption_candidates": 0,
        "legend_candidates": 0,
        "captions_linked": 0,
        "legends_linked": 0,
        "unmatched_caption_candidates": 0,
    }

    # Build eligible candidates (native only, kind-classified)
    cand: List[Dict[str, Any]] = []
    for t in texts:
        kind = _candidate_kind(t)
        if kind is None:
            continue
        if kind == "caption":
            stats["caption_candidates"] += 1
        else:
            stats["legend_candidates"] += 1
        cand.append(
            {
                "block": t,
                "kind": kind,
                "text": _block_text(t),
                "role": _role(t),
                "id": t.id,
            }
        )

    if not figures:
        stats["unmatched_caption_candidates"] = sum(
            1 for c in cand if c["kind"] == "caption"
        )
        return stats

    # Enumerate all valid (figure, candidate) pairs
    pairs: List[Tuple[Tuple, DocBlock, Dict[str, Any]]] = []
    for fig in figures:
        fbb = _block_bbox_xyxy(fig)
        for c in cand:
            tbb = _block_bbox_xyxy(c["block"])
            if not _vertical_gap_ok(fbb, tbb, MAX_VERTICAL_GAP_PT):
                continue
            if not _horizontal_ok(fbb, tbb):
                continue
            gap = max(0.0, tbb[1] - fbb[3])
            if c["kind"] == "legend" and gap > LEGEND_MAX_VERTICAL_GAP_PT:
                continue
            key = _pair_sort_key(c["kind"], fbb, tbb, c["role"], fig.id, c["id"])
            pairs.append((key, fig, c))

    # Global sort — independent of figure/text input order
    pairs.sort(key=lambda item: item[0])

    used_text_ids: set = set()
    fig_has_caption: set = set()
    fig_assocs: Dict[str, List[Dict[str, Any]]] = {f.id: [] for f in figures}

    for _key, fig, c in pairs:
        if c["id"] in used_text_ids:
            continue
        if c["kind"] == "caption" and fig.id in fig_has_caption:
            continue
        used_text_ids.add(c["id"])
        entry = {
            "kind": c["kind"],
            "blockId": c["id"],
            "text": c["text"],
            "source": "native",
            "method": "same_page_geometry",
        }
        fig_assocs[fig.id].append(entry)
        if c["kind"] == "caption":
            fig_has_caption.add(fig.id)
            stats["captions_linked"] += 1
        else:
            stats["legends_linked"] += 1
            # Upgrade role only for strong-legend native blocks
            if has_strong_legend_markers(c["text"]) or c["role"] == "legend":
                md = c["block"].metadata if isinstance(c["block"].metadata, dict) else {}
                md = dict(md)
                md["semanticRole"] = "legend"
                c["block"].metadata = md

    # Write associations: stable sort by (kind rank, blockId)
    kind_rank = {"caption": 0, "legend": 1}
    for fig in figures:
        assoc = sorted(
            fig_assocs.get(fig.id, []),
            key=lambda a: (kind_rank.get(a["kind"], 9), a["blockId"]),
        )
        md = fig.metadata if isinstance(fig.metadata, dict) else {}
        md = dict(md)
        md["figureTextAssociations"] = assoc
        fig.metadata = md

    stats["unmatched_caption_candidates"] = sum(
        1 for c in cand if c["kind"] == "caption" and c["id"] not in used_text_ids
    )
    return stats


def associate_ir_figures(ir: Any) -> Dict[str, Any]:
    """Post-pass over all pages of a CanonicalIR (fresh or cache-loaded).

    Raises on failure — this is a correctness stage; callers must not swallow.
    """
    totals = {
        "figures_with_native_caption": 0,
        "figures_with_native_legend": 0,
        "native_captions_linked": 0,
        "native_legends_linked": 0,
        "unmatched_caption_candidates": 0,
        "pages_processed": 0,
    }
    pages = getattr(ir, "pages", None) or []
    for page in pages:
        st = associate_page_texts(page)
        totals["pages_processed"] += 1
        totals["native_captions_linked"] += st.get("captions_linked", 0)
        totals["native_legends_linked"] += st.get("legends_linked", 0)
        totals["unmatched_caption_candidates"] += st.get(
            "unmatched_caption_candidates", 0
        )
        for b in page.blocks:
            if b.type not in ("figure", "image"):
                continue
            assocs = (b.metadata or {}).get("figureTextAssociations") or []
            kinds = {a.get("kind") for a in assocs if isinstance(a, dict)}
            if "caption" in kinds:
                totals["figures_with_native_caption"] += 1
            if "legend" in kinds:
                totals["figures_with_native_legend"] += 1
    return totals
