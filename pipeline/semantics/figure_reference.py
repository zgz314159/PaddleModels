"""Document-level figure label index + cross-page reference resolver — Phase 2G.

Runs after native association and optional OCR caption pass, before
SemanticProjector. Pure IR post-pass (no OCR, no cache): every run
recomputes labels/references within the currently loaded page range only.

Fail-fast: exceptions propagate — callers must not swallow.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline.canonical_ir import DocBlock, DocPage
from imaging.text_utils import extract_figure_label_pairs

FIGURE_REFERENCE_REPORT = "figure_reference_index.json"
METHOD = "exact_label_index"

# Body roles that must never be treated as reference sources.
_EXCLUDED_REFERENCE_ROLES = frozenset(
    {"caption", "legend", "figure_callout", "artifact", "ocr_annotation"}
)

_REF_STATUS_RESOLVED = "resolved"
_REF_STATUS_UNRESOLVED = "unresolved"
_REF_STATUS_AMBIGUOUS = "ambiguous"


def stable_reference_id(source_block_id: str, normalized_label: str) -> str:
    """Deterministic reference id from source block + normalized label."""
    seed = f"{source_block_id}|{normalized_label}"
    return "ref_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def _is_figure_block(block: DocBlock) -> bool:
    return block.type in ("figure", "image")


def _is_reference_source(block: DocBlock) -> bool:
    """Only body-type text blocks; never caption/legend/callout/artifact/OCR caption."""
    if block.type != "text":
        return False
    md = block.metadata if isinstance(block.metadata, dict) else {}
    role = str(md.get("semanticRole") or "").strip()
    if role in _EXCLUDED_REFERENCE_ROLES:
        return False
    # OCR-created caption blocks are type=caption (already excluded) — belt and
    # braces for any ocr-sourced caption-role text.
    if str(getattr(block, "source", "") or "") == "ocr" and role == "caption":
        return False
    if not (block.text or "").strip():
        return False
    return True


def _block_center(block: DocBlock) -> Tuple[float, float]:
    b = block.bbox
    return (b.x + b.w / 2.0, b.y + b.h / 2.0)


def _geometric_distance(a: DocBlock, b: DocBlock) -> float:
    ax, ay = _block_center(a)
    bx, by = _block_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _make_label_entry(
    raw_label: str,
    normalized_label: str,
    *,
    source: str,
    method: str,
    confidence: float,
) -> Dict[str, Any]:
    conf = max(0.0, min(1.0, float(confidence)))
    return {
        "rawLabel": raw_label,
        "normalizedLabel": normalized_label,
        "source": source,
        "method": method,
        "confidence": conf,
    }


def _label_sort_key(entry: Dict[str, Any]) -> Tuple:
    return (
        str(entry.get("normalizedLabel") or ""),
        str(entry.get("source") or ""),
        str(entry.get("method") or ""),
    )


def merge_figure_labels(
    existing: Sequence[Dict[str, Any]],
    incoming: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge by normalizedLabel; native evidence wins over OCR on conflict."""
    by_norm: Dict[str, Dict[str, Any]] = {}
    source_rank = {"native": 0, "ocr": 1}

    def _consider(entry: Dict[str, Any]) -> None:
        norm = str(entry.get("normalizedLabel") or "").strip()
        if not norm:
            return
        prev = by_norm.get(norm)
        if prev is None:
            by_norm[norm] = dict(entry)
            return
        prev_rank = source_rank.get(str(prev.get("source") or ""), 9)
        new_rank = source_rank.get(str(entry.get("source") or ""), 9)
        if new_rank < prev_rank:
            by_norm[norm] = dict(entry)
        # equal rank → keep first (stable)

    for e in existing or []:
        if isinstance(e, dict):
            _consider(e)
    for e in incoming or []:
        if isinstance(e, dict):
            _consider(e)
    return sorted(by_norm.values(), key=_label_sort_key)


def _labels_from_native_captions(fig: DocBlock) -> List[Dict[str, Any]]:
    """Trusted labels from native caption associations on this figure."""
    md = fig.metadata if isinstance(fig.metadata, dict) else {}
    assocs = md.get("figureTextAssociations") or []
    out: List[Dict[str, Any]] = []
    for a in assocs:
        if not isinstance(a, dict):
            continue
        if a.get("kind") != "caption":
            continue
        if a.get("source") == "ocr":
            # OCR caption labels are written by the OCR pass itself.
            continue
        text = str(a.get("text") or "")
        method = str(a.get("method") or "same_page_geometry")
        for raw, norm in extract_figure_label_pairs(text):
            out.append(
                _make_label_entry(
                    raw,
                    norm,
                    source="native",
                    method=method,
                    confidence=1.0,
                )
            )
    return out


def collect_trusted_labels(ir: Any) -> int:
    """Write/merge metadata.figureLabels on every figure in the loaded IR.

    Sources already on the figure (OCR pass) + native caption associations.
    Dedup by normalizedLabel; native wins over OCR. Returns total label count.
    """
    total = 0
    pages = getattr(ir, "pages", None) or []
    for page in pages:
        for block in page.blocks:
            if not _is_figure_block(block):
                continue
            md = block.metadata if isinstance(block.metadata, dict) else {}
            existing = list(md.get("figureLabels") or [])
            native = _labels_from_native_captions(block)
            merged = merge_figure_labels(existing, native)
            md = dict(md)
            md["figureLabels"] = merged
            block.metadata = md
            total += len(merged)
    return total


def build_label_index(ir: Any) -> Dict[str, List[Dict[str, Any]]]:
    """normalizedLabel → sorted target descriptors within current page range."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    pages = getattr(ir, "pages", None) or []
    for page in pages:
        p_num = page.page_number if isinstance(page, DocPage) else page["page_number"]
        for block in page.blocks:
            if not _is_figure_block(block):
                continue
            md = block.metadata if isinstance(block.metadata, dict) else {}
            for lab in md.get("figureLabels") or []:
                if not isinstance(lab, dict):
                    continue
                norm = str(lab.get("normalizedLabel") or "").strip()
                if not norm:
                    continue
                index.setdefault(norm, []).append(
                    {
                        "figureId": block.id,
                        "pageNumber": int(p_num),
                        "block": block,
                        "label": dict(lab),
                    }
                )
    # Deterministic target order within each label.
    for norm in index:
        index[norm].sort(key=lambda t: (t["pageNumber"], t["figureId"]))
    return index


def resolve_targets(
    normalized_label: str,
    source_block: DocBlock,
    source_page: int,
    index: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, Optional[Dict[str, Any]], str, Optional[int]]:
    """Exact index lookup. Returns (status, target|None, reason, page_delta|None).

    Rules:
    - no target → unresolved
    - unique → resolved
    - multi: prefer same-page nearest by geometric distance
    - no same-page → min |pageDelta|
    - best distance still tied → ambiguous (never pick first)
    """
    targets = index.get(normalized_label) or []
    if not targets:
        return _REF_STATUS_UNRESOLVED, None, "no_label_in_index", None

    if len(targets) == 1:
        t = targets[0]
        delta = int(t["pageNumber"]) - int(source_page)
        return _REF_STATUS_RESOLVED, t, "unique_target", delta

    same_page = [t for t in targets if int(t["pageNumber"]) == int(source_page)]
    pool = same_page if same_page else targets

    if same_page:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for t in pool:
            dist = _geometric_distance(source_block, t["block"])
            scored.append((dist, t))
        best_d = min(d for d, _ in scored)
        winners = [t for d, t in scored if d == best_d]
        if len(winners) == 1:
            t = winners[0]
            delta = int(t["pageNumber"]) - int(source_page)
            return _REF_STATUS_RESOLVED, t, "nearest_same_page", delta
        return (
            _REF_STATUS_AMBIGUOUS,
            None,
            f"tied_same_page_distance:{len(winners)}",
            None,
        )

    # Cross-page only: min absolute page delta.
    scored_p: List[Tuple[int, Dict[str, Any]]] = []
    for t in pool:
        delta = abs(int(t["pageNumber"]) - int(source_page))
        scored_p.append((delta, t))
    best_delta = min(d for d, _ in scored_p)
    winners = [t for d, t in scored_p if d == best_delta]
    if len(winners) == 1:
        t = winners[0]
        signed = int(t["pageNumber"]) - int(source_page)
        return _REF_STATUS_RESOLVED, t, "nearest_page", signed
    return (
        _REF_STATUS_AMBIGUOUS,
        None,
        f"tied_page_distance:{len(winners)}",
        None,
    )


def resolve_figure_references(ir: Any) -> Dict[str, Any]:
    """Full Phase 2G pass: labels → index → extract refs → resolve → write back.

    Raises on unexpected failure (fail-fast). Returns report dict including
    stats consumed by the CLI consistency gate and figure_reference_index.json.
    """
    pages = getattr(ir, "pages", None) or []
    if not pages:
        page_numbers: List[int] = []
    else:
        page_numbers = sorted(
            int(p.page_number if isinstance(p, DocPage) else p["page_number"])
            for p in pages
        )
    page_range = (
        {"start": page_numbers[0], "end": page_numbers[-1]}
        if page_numbers
        else None
    )

    # 1) Ensure figureLabels (OCR-written + native captions) on all figures.
    figure_labels_indexed = collect_trusted_labels(ir)

    # 2) Document-level exact index (current loaded range only).
    index = build_label_index(ir)

    label_index_out: List[Dict[str, Any]] = []
    for norm in sorted(index.keys()):
        for t in index[norm]:
            lab = t["label"]
            label_index_out.append(
                {
                    "normalizedLabel": norm,
                    "rawLabel": lab.get("rawLabel"),
                    "source": lab.get("source"),
                    "method": lab.get("method"),
                    "confidence": lab.get("confidence"),
                    "figureId": t["figureId"],
                    "pageNumber": t["pageNumber"],
                }
            )

    # 3) Extract references from body text blocks + resolve.
    resolved_edges: List[Dict[str, Any]] = []
    unresolved_refs: List[Dict[str, Any]] = []
    ambiguous_refs: List[Dict[str, Any]] = []
    found = 0
    n_resolved = 0
    n_unresolved = 0
    n_ambiguous = 0

    # Collect (page, block) pairs first for deterministic processing order.
    body_blocks: List[Tuple[int, DocBlock]] = []
    for page in pages:
        p_num = int(
            page.page_number if isinstance(page, DocPage) else page["page_number"]
        )
        for block in page.blocks:
            if _is_reference_source(block):
                body_blocks.append((p_num, block))
    body_blocks.sort(key=lambda pb: (pb[0], pb[1].id))

    for source_page, src in body_blocks:
        pairs = extract_figure_label_pairs(src.text or "")
        # Dedup within this source block by normalizedLabel.
        seen_norm: set = set()
        refs: List[Dict[str, Any]] = []
        for raw, norm in pairs:
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            found += 1
            ref_id = stable_reference_id(src.id, norm)
            status, target, reason, page_delta = resolve_targets(
                norm, src, source_page, index
            )
            entry: Dict[str, Any] = {
                "referenceId": ref_id,
                "rawLabel": raw,
                "normalizedLabel": norm,
                "status": status,
                "method": METHOD,
            }
            if status == _REF_STATUS_RESOLVED and target is not None:
                entry["targetFigureId"] = target["figureId"]
                entry["targetPageNumber"] = int(target["pageNumber"])
                entry["pageDelta"] = int(page_delta or 0)
                n_resolved += 1
                back = {
                    "referenceId": ref_id,
                    "sourceBlockId": src.id,
                    "sourcePageNumber": int(source_page),
                    "rawLabel": raw,
                    "normalizedLabel": norm,
                }
                tmd = (
                    target["block"].metadata
                    if isinstance(target["block"].metadata, dict)
                    else {}
                )
                tmd = dict(tmd)
                existing_back = list(tmd.get("referencedBy") or [])
                if not any(
                    isinstance(b, dict) and b.get("referenceId") == ref_id
                    for b in existing_back
                ):
                    existing_back.append(back)
                existing_back.sort(
                    key=lambda b: (
                        str(b.get("referenceId") or ""),
                        str(b.get("sourceBlockId") or ""),
                    )
                )
                tmd["referencedBy"] = existing_back
                target["block"].metadata = tmd
                resolved_edges.append(dict(entry))
            elif status == _REF_STATUS_AMBIGUOUS:
                entry["reason"] = reason
                n_ambiguous += 1
                ambiguous_refs.append(dict(entry))
            else:
                entry["reason"] = reason
                n_unresolved += 1
                unresolved_refs.append(dict(entry))
            refs.append(entry)

        # Stable order of references on the source block.
        refs.sort(key=lambda r: (r["normalizedLabel"], r["referenceId"]))
        if refs:
            md = src.metadata if isinstance(src.metadata, dict) else {}
            md = dict(md)
            md["figureReferences"] = refs
            src.metadata = md

    referenced_figures = 0
    for page in pages:
        for block in page.blocks:
            if not _is_figure_block(block):
                continue
            md = block.metadata if isinstance(block.metadata, dict) else {}
            if md.get("referencedBy"):
                referenced_figures += 1

    stats = {
        "figure_labels_indexed": figure_labels_indexed,
        "figure_references_found": found,
        "figure_references_resolved": n_resolved,
        "figure_references_unresolved": n_unresolved,
        "figure_references_ambiguous": n_ambiguous,
        "referenced_figures": referenced_figures,
    }

    return {
        "method": METHOD,
        "pageRange": page_range,
        "pageNumbers": page_numbers,
        "figureLabelIndex": label_index_out,
        "resolved": sorted(
            resolved_edges,
            key=lambda e: (e.get("normalizedLabel") or "", e.get("referenceId") or ""),
        ),
        "unresolved": sorted(
            unresolved_refs,
            key=lambda e: (e.get("normalizedLabel") or "", e.get("referenceId") or ""),
        ),
        "ambiguous": sorted(
            ambiguous_refs,
            key=lambda e: (e.get("normalizedLabel") or "", e.get("referenceId") or ""),
        ),
        "stats": stats,
    }
