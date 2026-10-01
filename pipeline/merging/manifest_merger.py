import math
import re
import copy
from typing import Any, Dict, List, Optional, Set, Tuple
from imaging.text_utils import safe_str, get_text_fingerprint
from imaging.pdf_analyzer import (
    parse_page_number,
    collect_entry_pages,
    entry_dominant_page,
    entry_page_hint,
    basename_from_uri,
)
from imaging.kb_utils import (
    pick_first_str,
    caption_from_manifest_item,
    semantic_texts_from_manifest_item,
    get_manifest_item_bbox,
    bbox_sort_key_from_item,
    MergeScope,
    entry_in_scope,
    make_image_block,
    insert_block,
    insert_block_after_markdown_code,
    get_block_bbox
)
from imaging.semantic_matcher import (
    manifest_item_semantic_kind_hint,
    build_entry_semantic_profile,
    score_manifest_signal_against_entry,
    order_score,
    find_target_entry_index,
    candidate_indices_for_page_window,
    entry_has_image_ref_markers
)
from imaging.table_processor import (
    entry_has_table_block,
    entry_existing_image_uris,
    merge_table_fragments_before_appendix_fill,
    reroute_existing_legend_images_into_tables as _reroute_existing_legend_images_into_tables,
    bridge_previous_legend_page_into_tables as _bridge_previous_legend_page_into_tables
)

from pipeline.merging.merger_stats import MergeStats, record_inserted_manifest_asset

def _page_anchor_order_score(item_index: int, entry_index: int, page_number: Optional[int], 
                             page_item_order_hint: Dict[int, Tuple[int, int]], 
                             page_entry_order_hint: Dict[int, Tuple[int, int]]) -> float:
    if not isinstance(page_number, int) or page_number <= 0:
        return 0.0
    item_hint = page_item_order_hint.get(int(item_index))
    entry_hint = page_entry_order_hint.get(int(entry_index))
    if item_hint is None or entry_hint is None:
        return 0.0
    item_rank, item_total = item_hint
    entry_rank, entry_total = entry_hint
    if item_total <= 1 or entry_total <= 1:
        return 0.0
    i_rank = float(item_rank) / float(max(1, item_total - 1))
    e_rank = float(entry_rank) / float(max(1, entry_total - 1))
    return max(0.0, 1.0 - abs(i_rank - e_rank))

def apply_merge(kb: Dict[str, Any], item_index: int, item: Dict[str, Any], entry_index: int, 
                stats: MergeStats, dry_run: bool, debug_merge: bool,
                page_item_order_hint: Dict[int, Tuple[int, int]],
                insert_mode: str) -> bool:
    entries = kb.get("entries")
    entry = entries[entry_index]
    asset_uri = pick_first_str(item, ("assetUri", "asset_uri", "imageUri", "image_uri", "src")).strip()
    if not asset_uri:
        return False

    page_number = parse_page_number(item.get("pageNumber"))
    has_page = isinstance(page_number, int) and page_number > 0
    caption_text = caption_from_manifest_item(item)
    anchor_text = safe_str(item.get("anchorText")).strip()
    heading_text = safe_str(item.get("headingText")).strip()
    block_bbox = get_manifest_item_bbox(item)
    item_page_hint = page_item_order_hint.get(int(item_index))

    # Simplified kind hint
    manifest_item_kind = manifest_item_semantic_kind_hint(item)
    
    existing_uris = entry_existing_image_uris(entry)
    if asset_uri in existing_uris:
        stats.skipped_duplicate += 1
        return False

    block_page = int(page_number) if has_page else int(entry_page_hint(entry) or 1)
    block = make_image_block(asset_uri=asset_uri, page_number=block_page, caption=caption_text, bbox=block_bbox)

    if not dry_run:
        rank_hint = None
        total_hint = None
        if item_page_hint is not None:
            rank_hint, total_hint = item_page_hint
        insert_block(
            entry,
            block,
            block_page,
            insert_mode,
            page_rank_hint=rank_hint,
            page_total_hint=total_hint,
        )

    stats.inserted += 1
    record_inserted_manifest_asset(
        stats,
        item_index=item_index,
        item=item,
        asset_uri=asset_uri,
        page_number=block_page,
        caption_text=caption_text,
        anchor_text=anchor_text,
        heading_text=heading_text,
        item_kind=manifest_item_kind,
        target_entry_id=safe_str(entry.get("entryId")).strip(),
        insert_mode=insert_mode,
        bbox=block_bbox,
    )
    return True

def merge_manifest_into_kb(
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    insert_mode: str,
    page_match: str,
    prefer_heading: bool,
    dry_run: bool,
    scope: Optional[MergeScope],
    gap_max_pages: int = 3,
    enable_appendix_fill: bool = True,
    similarity_threshold: float = 0.5,
    debug_merge: bool = False,
    *,
    assets_root: str = "",
    pdf_native: bool = False,
    source_native_kb: Optional[Dict[str, Any]] = None,
) -> MergeStats:
    stats = MergeStats()
    entries = kb.get("entries")
    items = manifest.get("items")
    if not isinstance(entries, list) or not isinstance(items, list):
        return stats

    # Precompute hints and profiles
    page_to_entries: Dict[int, List[int]] = {}
    scoped_entry_profiles: Dict[int, Dict[str, Any]] = {}
    scoped_indices_all = []
    
    for idx, e in enumerate(entries):
        if scope is not None and not entry_in_scope(e, scope):
            continue
        scoped_indices_all.append(idx)
        scoped_entry_profiles[idx] = build_entry_semantic_profile(e)
        for p in set(collect_entry_pages(e)):
            page_to_entries.setdefault(p, []).append(idx)

    page_item_order_hint: Dict[int, Tuple[int, int]] = {}
    page_to_item_indices: Dict[int, List[int]] = {}
    for item_i, it in enumerate(items):
        pn = parse_page_number(it.get("pageNumber"))
        if isinstance(pn, int) and pn > 0:
            page_to_item_indices.setdefault(pn, []).append(item_i)
            
    for page_num, idxs in page_to_item_indices.items():
        sorted_item_indices = sorted(idxs, key=lambda i: bbox_sort_key_from_item(items[i], i))
        total = len(sorted_item_indices)
        for rank, idx in enumerate(sorted_item_indices):
            page_item_order_hint[int(idx)] = (int(rank), int(total))

    page_entry_order_hint: Dict[int, Tuple[int, int]] = {}
    for page_num, idxs in page_to_entries.items():
        sorted_indices = sorted(idxs, key=lambda i: (int(entries[i].get("position", 10**9)), i))
        total = len(sorted_indices)
        for rank, idx in enumerate(sorted_indices):
            page_entry_order_hint[int(idx)] = (int(rank), int(total))

    def _best_semantic_entry_index(
        item_i: int,
        item: Dict[str, Any],
        page_number: Optional[int],
        candidate_indices: List[int],
        item_kind: str,
    ) -> Optional[int]:
        signals = semantic_texts_from_manifest_item(item)
        if not signals or not candidate_indices:
            return None

        scored: List[Tuple[float, int, int, float, int]] = []
        for entry_idx in candidate_indices:
            entry = entries[entry_idx]
            profile = scoped_entry_profiles.get(entry_idx) or build_entry_semantic_profile(entry)
            best_score = 0
            best_evidence = 0
            best_similarity = 0.0
            for signal in signals:
                score, evidence, similarity = score_manifest_signal_against_entry(
                    signal,
                    entry,
                    profile,
                    item_kind=item_kind,
                    cutoff_ratio=similarity_threshold,
                    page_number=page_number,
                )
                if score > best_score:
                    best_score = score
                    best_evidence = evidence
                    best_similarity = similarity

            page_order_score = _page_anchor_order_score(
                item_i,
                entry_idx,
                page_number,
                page_item_order_hint,
                page_entry_order_hint,
            )
            total_score = float(best_score) + page_order_score * 75.0
            if best_evidence > 0 or best_similarity >= similarity_threshold:
                scored.append((total_score, best_evidence, entry_idx, best_similarity, entry_idx))

        if not scored:
            return None

        scored.sort(key=lambda item: (-item[0], -item[1], -item[3], item[4]))
        best_total, best_evidence, best_idx, best_similarity, _ = scored[0]
        min_score = max(120.0, float(similarity_threshold) * 220.0)
        if best_total >= min_score or best_evidence >= 2 or best_similarity >= similarity_threshold:
            return int(best_idx)
        return None

    # Phase 1: precise semantic match, with page/order fallbacks.
    used_manifest_indices = set()
    for item_i, it in enumerate(items):
        stats.items_total += 1
        pn = parse_page_number(it.get("pageNumber"))
        if pn: stats.items_with_page += 1
        
        asset_uri = pick_first_str(it, ("assetUri", "asset_uri", "imageUri", "image_uri", "src"))
        if asset_uri: stats.items_with_uri += 1
        
        if not asset_uri:
            stats.skipped_missing_fields += 1
            continue

        item_kind = manifest_item_semantic_kind_hint(it)
        candidate_indices: List[int]
        if isinstance(pn, int) and pn > 0:
            candidate_indices = candidate_indices_for_page_window(page_to_entries, pn, radius=gap_max_pages)
        else:
            candidate_indices = list(scoped_indices_all)

        if scope is not None:
            candidate_indices = [idx for idx in candidate_indices if entry_in_scope(entries[idx], scope)]
        if not candidate_indices:
            stats.skipped_no_entry_for_page += 1
            continue

        target_idx = _best_semantic_entry_index(item_i, it, pn, candidate_indices, item_kind)
        if target_idx is None and isinstance(pn, int) and pn > 0:
            allowed_pages = set(page_to_entries.keys()) if page_match == "any" else set(range(max(1, pn - gap_max_pages), pn + gap_max_pages + 1))
            target_idx = find_target_entry_index(
                entries,
                pn,
                page_match,
                heading_text=safe_str(it.get("headingText") or it.get("anchorText")).strip(),
                prefer_heading=prefer_heading,
                candidate_indices=candidate_indices,
                allowed_pages=allowed_pages,
            )
        if target_idx is None:
            target_idx = candidate_indices[0] if candidate_indices else None
        if target_idx is None:
            stats.skipped_no_entry_for_page += 1
            continue

        if apply_merge(
            kb,
            item_i,
            it,
            target_idx,
            stats,
            dry_run,
            debug_merge,
            page_item_order_hint,
            insert_mode,
        ):
            used_manifest_indices.add(item_i)

    # Post processing
    if enable_appendix_fill:
        merge_table_fragments_before_appendix_fill(
            kb=kb,
            manifest=manifest,
            used_manifest_indices=used_manifest_indices,
            stats=stats,
            dry_run=dry_run,
            debug_merge=debug_merge,
        )
    _reroute_existing_legend_images_into_tables(kb, manifest, dry_run, debug_merge)
    _bridge_previous_legend_page_into_tables(
        kb=kb,
        dry_run=dry_run,
        debug_merge=debug_merge,
    )
    
    return stats
