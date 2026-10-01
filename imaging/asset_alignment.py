import re
from typing import Any, Dict, List, Optional, Set, Tuple

from imaging.text_utils import safe_str
from imaging.pdf_analyzer import (
    basename_from_uri,
    entry_page_hint,
    parse_page_number
)
from imaging.table_processor import (
    entry_has_table_block_missing_snapshot,
    entry_existing_image_uris
)
from imaging.kb_utils import (
    make_image_block,
    insert_block,
    copy_bbox_dict
)
from imaging.semantic_matcher import (
    build_entry_semantic_profile,
    candidate_indices_for_page_window,
    score_manifest_signal_against_entry,
    find_target_entry_index,
    normalize_for_similarity
)

def attach_snapshot_to_first_missing_table_block(
    entry: Dict[str, Any],
    asset_uri: str,
    bbox: Optional[Dict[str, Any]],
) -> bool:
    """Find the first table block in an entry that is missing a snapshot and attach the given URI."""
    blocks = entry.get("blocks") if isinstance(entry, dict) else None
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict): continue
        if safe_str(block.get("type")).strip().lower() != "table": continue
        
        snapshot_uri = safe_str(block.get("imageUri") or block.get("snapshotUri")).strip()
        images = block.get("images")
        has_inline_images = isinstance(images, list) and any(
            isinstance(image, dict) and safe_str(image.get("imageUri") or image.get("src")).strip()
            for image in images
        )
        
        if snapshot_uri or has_inline_images: continue
        
        block["imageUri"] = asset_uri
        if bbox:
            block["imageBBox"] = copy_bbox_dict(bbox)
        return True
    return False

def find_asset_alignment_target_entry_index(
    kb: Dict[str, Any],
    record: Dict[str, Any],
) -> Tuple[Optional[int], int, int, float]:
    """Search for the most likely KB entry that an orphaned manifest asset should belong to."""
    entries = kb.get("entries")
    if not isinstance(entries, list) or not entries:
        return (None, 0, 0, 0.0)

    page_to_entries: Dict[int, List[int]] = {}
    profiles: Dict[int, Dict[str, Any]] = {}
    from imaging.pdf_analyzer import collect_entry_pages
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict): continue
        profiles[idx] = build_entry_semantic_profile(entry)
        for page_num in set(collect_entry_pages(entry)):
            page_to_entries.setdefault(int(page_num), []).append(int(idx))

    page_number = parse_page_number(record.get("pageNumber"))
    item_kind = safe_str(record.get("itemKind")).strip().lower() or "legend"
    
    # We use the same semantic extraction logic as manifest items
    signals: List[str] = []
    for key in ("anchorText", "headingText", "captionText"):
        val = safe_str(record.get(key)).strip()
        if val: signals.append(val)
        
    if not signals:
        fallback_heading = safe_str(record.get("headingText") or record.get("anchorText") or record.get("captionText")).strip()
        if fallback_heading and isinstance(page_number, int) and page_number > 0:
            best_idx = find_target_entry_index(entries, page_number, "best", heading_text=fallback_heading, prefer_heading=True)
            return (best_idx, 0, 0, 0.0)
        return (None, 0, 0, 0.0)

    candidate_indices = (
        candidate_indices_for_page_window(page_to_entries, page_number, radius=8)
        if isinstance(page_number, int) and page_number > 0
        else []
    )
    if not candidate_indices:
        candidate_indices = list(profiles.keys())

    best_idx: Optional[int] = None
    best_score = 0
    best_evidence = 0
    best_similarity = 0.0
    best_page_gap = 10**9

    for idx in candidate_indices:
        entry = entries[idx]
        profile = profiles.get(idx) or build_entry_semantic_profile(entry)
        local_best_score = 0
        local_best_evidence = 0
        local_best_similarity = 0.0
        for signal in signals:
            score, evidence, similarity = score_manifest_signal_against_entry(
                signal,
                entry,
                profile,
                item_kind=item_kind,
                cutoff_ratio=0.5,
                page_number=page_number,
            )
            if score > local_best_score:
                local_best_score = int(score)
                local_best_evidence = int(evidence)
                local_best_similarity = float(similarity)

        entry_page = entry_page_hint(entry)
        page_gap = abs(int(entry_page) - int(page_number)) if isinstance(entry_page, int) and isinstance(page_number, int) else 10**9
        
        if local_best_score <= 0: continue

        is_better = False
        if local_best_score > best_score:
            is_better = True
        elif local_best_score == best_score and local_best_evidence > best_evidence:
            is_better = True
        elif local_best_score == best_score and local_best_evidence == best_evidence and local_best_similarity > best_similarity:
            is_better = True
        elif (
            local_best_score == best_score
            and local_best_evidence == best_evidence
            and abs(local_best_similarity - best_similarity) < 1e-9
            and page_gap < best_page_gap
        ):
            is_better = True

        if is_better:
            best_idx = int(idx)
            best_score = int(local_best_score)
            best_evidence = int(local_best_evidence)
            best_similarity = float(local_best_similarity)
            best_page_gap = int(page_gap)

    min_score = 220 if item_kind == "table" else 180
    if best_idx is None or best_score < min_score:
        return (None, best_score, best_evidence, best_similarity)
    return (best_idx, best_score, best_evidence, best_similarity)

def attach_asset_alignment_record_to_entry(
    kb: Dict[str, Any],
    record: Dict[str, Any],
    entry_index: int,
    *,
    debug: bool,
) -> bool:
    """Physically attach an orphaned asset to the selected KB entry."""
    entries = kb.get("entries")
    if not isinstance(entries, list) or entry_index < 0 or entry_index >= len(entries):
        return False
    entry = entries[entry_index]
    if not isinstance(entry, dict):
        return False

    asset_uri = safe_str(record.get("assetUri")).strip()
    if not asset_uri:
        return False

    if asset_uri in entry_existing_image_uris(entry):
        return True

    block_page = parse_page_number(record.get("pageNumber")) or int(entry_page_hint(entry) or 1)
    caption_text = safe_str(record.get("captionText")).strip()
    anchor_text = safe_str(record.get("anchorText")).strip()
    heading_text = safe_str(record.get("headingText")).strip()
    item_kind = safe_str(record.get("itemKind")).strip().lower() or "legend"
    block_bbox = copy_bbox_dict(record.get("bbox"))

    if not caption_text:
        caption_text = heading_text or anchor_text or ("表格原件（续）" if item_kind == "table" else "图件续页")

    attached = False
    if item_kind == "table" and entry_has_table_block_missing_snapshot(entry):
        attached = attach_snapshot_to_first_missing_table_block(entry, asset_uri, block_bbox)

    if not attached:
        block = make_image_block(asset_uri=asset_uri, page_number=block_page, caption=caption_text, bbox=block_bbox)
        # Note: _insert_image_block_after_anchor_text is still in main script for now as it's very specific
        # We will fallback to a standard insert if the anchor search is not available or fails
        insert_block(
            entry=entry,
            block=block,
            page_number=block_page,
            insert_mode=safe_str(record.get("insertMode")).strip() or "after-page-text",
        )

    return True

def looks_like_pure_figure_caption_signal(text: str) -> bool:
    raw = safe_str(text).strip()
    if not raw: return False
    compact = re.sub(r"\s+", "", raw)
    return bool(re.match(r"^(?:\d+)?[图表][0-9一二三四五六七八九十百千万—\-~～]+", compact))

def should_skip_asset_alignment_record(record: Dict[str, Any]) -> bool:
    item_kind = safe_str(record.get("itemKind")).strip().lower() or "legend"
    if item_kind == "table": return False

    raw_signals = [
        safe_str(record.get("anchorText")).strip(),
        safe_str(record.get("headingText")).strip(),
        safe_str(record.get("captionText")).strip(),
    ]
    normalized_signals = {
        normalize_for_similarity(text)
        for text in raw_signals
        if normalize_for_similarity(text)
    }
    if len(normalized_signals) != 1: return False

    repeated_signal = next(iter(normalized_signals), "")
    if not repeated_signal: return False

    for text in raw_signals:
        if normalize_for_similarity(text) == repeated_signal and looks_like_pure_figure_caption_signal(text):
            return True
    return False

def run_asset_alignment_check(
    kb: Dict[str, Any],
    *,
    inserted_assets: List[Dict[str, Any]],
    debug: bool = False,
) -> Dict[str, int]:
    """Verify that all assets inserted during merge are still present in the KB; reattach if orphaned."""
    metrics = {
        "checkedInsertedAssets": 0,
        "missingInsertedAssets": 0,
        "reattachedInsertedAssets": 0,
        "skippedWeakAnchorAssets": 0,
        "unresolvedInsertedAssets": 0,
    }
    if not inserted_assets:
        return metrics

    from imaging.table_processor import collect_kb_image_basenames
    referenced = collect_kb_image_basenames(kb)
    seen: Set[str] = set()
    for record in inserted_assets:
        if not isinstance(record, dict): continue
        basename = safe_str(record.get("basename") or basename_from_uri(record.get("assetUri"))).strip()
        if not basename or basename in seen: continue
        seen.add(basename)
        metrics["checkedInsertedAssets"] += 1
        if basename in referenced: continue

        metrics["missingInsertedAssets"] += 1
        if should_skip_asset_alignment_record(record):
            metrics["skippedWeakAnchorAssets"] += 1
            continue

        best_idx, best_score, _, _ = find_asset_alignment_target_entry_index(kb, record)
        if best_idx is None:
            metrics["unresolvedInsertedAssets"] += 1
            continue

        if attach_asset_alignment_record_to_entry(kb, record, best_idx, debug=debug):
            referenced.add(basename)
            metrics["reattachedInsertedAssets"] += 1
        else:
            metrics["unresolvedInsertedAssets"] += 1

    return metrics
