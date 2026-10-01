import logging
import os
import re
import shutil
from typing import List, Dict, Any, Optional, Tuple, Set, Sequence
try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None

from imaging.text_utils import collapse_match_text, ocr_match_tokens, safe_str, figure_table_label_variants
from imaging.pdf_analyzer import (
    basename_from_uri, 
    entry_page_hint, 
    physical_page_from_asset_uri,
    resolve_manifest_local_path,
    parse_page_number,
    entry_reference_labels
)
from imaging.ocr_engine import ocr_image_file
from imaging.bbox_utils import normalize_bbox_dict
from imaging.kb_utils import (
    insert_block, 
    make_image_block, 
    remove_image_references_by_basename,
    record_inserted_manifest_asset
)

logger = logging.getLogger(__name__)

def is_table_snapshot_uri(uri: str) -> bool:
    """Check if the URI filename identifies it as a table snapshot."""
    fn = basename_from_uri(uri).lower()
    if not fn:
        return False
    return fn.startswith("table_") or fn.startswith("tablepos") or fn.startswith("tablep")

def entry_has_table_block(entry: Dict[str, Any]) -> bool:
    """Check if the entry contains any block of type 'table'."""
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "").strip().lower() == "table":
            return True
    return False

def first_table_block(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the first block of type 'table' found in the entry."""
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "").strip().lower() == "table":
            return block
    return None

def table_entry_text(entry: Dict[str, Any]) -> str:
    """Aggregate text from an entry that is useful for table matching."""
    parts: List[str] = []
    job_title = str(entry.get("jobTitle") or "").strip()
    if job_title:
        parts.append(job_title)
    content_norm = str(entry.get("contentNormalized") or "").strip()
    if content_norm:
        parts.append(content_norm)
    
    table_block = first_table_block(entry)
    if isinstance(table_block, dict):
        caption = str(table_block.get("caption") or table_block.get("title") or "").strip()
        if caption:
            parts.append(caption)
    return " ".join(part for part in parts if part)

def score_table_fragment_match(item_text: str, entry: Dict[str, Any], *, page_number: int) -> Tuple[int, int]:
    """Score how well an OCR fragment matches a given table entry."""
    entry_text = table_entry_text(entry)
    entry_compact = collapse_match_text(entry_text)
    if not entry_compact:
        return (0, 0)

    tokens = ocr_match_tokens(item_text)
    if not tokens:
        compact = collapse_match_text(item_text)
        if len(compact) >= 6 and compact in entry_compact:
            tokens = [compact]
    if not tokens:
        return (0, 0)

    score = 0
    hits = 0
    for token in tokens:
        if token in entry_compact:
            score += min(len(token), 12)
            hits += 1

    if hits <= 0:
        return (0, 0)

    entry_page = entry_page_hint(entry)
    if isinstance(entry_page, int) and entry_page > 0 and page_number > 0:
        delta = abs(int(page_number) - int(entry_page))
        if delta <= 1:
            score += 6
        elif delta <= 3:
            score += 3
        elif delta <= 6:
            score += 1

    return (int(score), int(hits))

def snapshot_covers_fragment(snapshot_text: str, fragment_text: str) -> bool:
    """Determine if a table snapshot's OCR text contains/covers a fragment's text."""
    snap_compact = collapse_match_text(snapshot_text)
    frag_compact = collapse_match_text(fragment_text)
    if len(snap_compact) < 8 or len(frag_compact) < 6:
        return False
    if frag_compact in snap_compact:
        return True

    frag_tokens = ocr_match_tokens(fragment_text)
    if not frag_tokens:
        return False

    total = sum(len(token) for token in frag_tokens)
    if total <= 0:
        return False
    matched = sum(len(token) for token in frag_tokens if token in snap_compact)
    ratio = float(matched) / float(total)
    
    if matched >= 8 and ratio >= 0.45:
        return True
    if len(frag_tokens) >= 3 and matched >= 12 and ratio >= 0.40:
        return True
    return False

def resolve_table_snapshot_local_path(manifest: Dict[str, Any], uri: str) -> str:
    """Resolve local path for a table snapshot URI."""
    out_dir = str(manifest.get("outDir") or "").strip()
    base = basename_from_uri(uri)
    if out_dir and base:
        candidate = os.path.join(out_dir, base)
        if os.path.exists(candidate):
            return candidate
    return ""

def next_table_continuation_index(entry: Dict[str, Any], page_number: int) -> int:
    """Find the next available index for a table continuation image on a given page."""
    page_i = int(page_number or 1)
    max_idx = 0
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return 1
    for block in blocks:
        if not isinstance(block, dict):
            continue
        uri = str(block.get("imageUri") or block.get("image_uri") or block.get("src") or "").strip()
        if not uri:
            continue
        match = re.search(r"_p(\d+).*_idx(\d+)\.(png|jpg|jpeg|webp)$", uri, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            p = int(match.group(1))
            idx = int(match.group(2))
        except Exception:
            continue
        if p == page_i and idx > max_idx:
            max_idx = idx
    return max_idx + 1

def find_existing_table_continuation_uri(entry: Dict[str, Any], page_number: int, position_hint: int) -> str:
    """Look for an existing continuation image URI that matches page and position."""
    page_i = int(page_number or 1)
    pos_i = int(position_hint or 0)
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return ""

    matches: List[Tuple[int, str]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        uri = str(block.get("imageUri") or block.get("image_uri") or block.get("src") or "").strip()
        if not uri:
            continue
        match = re.search(r"table_p(\d+)_pos(\d+)_idx(\d+)\.(png|jpg|jpeg|webp)$", basename_from_uri(uri), flags=re.IGNORECASE)
        if not match:
            continue
        try:
            page_v = int(match.group(1))
            pos_v = int(match.group(2))
            idx_v = int(match.group(3))
        except Exception:
            continue
        if page_v != page_i:
            continue
        if pos_i > 0 and pos_v != pos_i:
            continue
        matches.append((idx_v, uri))

    if not matches:
        return ""
    matches.sort(key=lambda item: item[0])
    return matches[0][1]

def table_entry_position_hint(entry: Dict[str, Any]) -> int:
    """Extract a numeric position hint from entry ID or title."""
    entry_id = str(entry.get("entryId") or "").strip()
    match = re.search(r"::table_(\d+)$", entry_id)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass

    job_title = str(entry.get("jobTitle") or "").strip()
    match = re.search(r"表格#\s*(\d+)", job_title)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass

    position = entry.get("position")
    try:
        return int(position)
    except Exception:
        return 0

def materialize_table_continuation_asset_uri(
    manifest: Dict[str, Any],
    item: Dict[str, Any],
    entry: Dict[str, Any],
) -> Optional[str]:
    """Copy a table fragment from manifest to KB assets and return its asset URI."""
    src_path = resolve_manifest_local_path(manifest, item)
    if not src_path:
        return None

    out_dir = str(manifest.get("outDir") or "").strip()
    file_id = str(manifest.get("fileId") or "").strip()
    page_number = int(parse_page_number(item.get("pageNumber")) or 1)
    pos_i = table_entry_position_hint(entry)
    
    existing_uri = find_existing_table_continuation_uri(entry, page_number, pos_i)
    if existing_uri:
        return existing_uri
        
    cont_idx = next_table_continuation_index(entry, page_number)
    if pos_i > 0:
        file_name = f"table_p{page_number}_pos{pos_i}_idx{cont_idx}.png"
    else:
        file_name = f"table_p{page_number}_idx{cont_idx}.png"
        
    dst_path = os.path.join(out_dir, file_name)
    if not os.path.exists(dst_path):
        try:
            shutil.copyfile(src_path, dst_path)
        except Exception as ex:
            logger.error(f"Failed to copy table continuation {src_path} -> {dst_path}: {ex}")
            return None

    return f"file:///android_asset/kb/{file_id}/截图/{file_name}"

def score_table_visual_match(source_bbox: Dict[str, Any], target_bbox: Dict[str, Any]) -> int:
    """Score visual similarity between two table bounding boxes (width and left alignment)."""
    score = 0
    src_width = float(source_bbox.get("width") or 0)
    dst_width = float(target_bbox.get("width") or 0)
    if src_width > 0 and dst_width > 0:
        width_ratio = abs(src_width - dst_width) / max(src_width, dst_width)
        if width_ratio <= 0.06:
            score += 6
        elif width_ratio <= 0.12:
            score += 4
        elif width_ratio <= 0.20:
            score += 2

    src_left = float(source_bbox.get("left") or 0)
    dst_left = float(target_bbox.get("left") or 0)
    if src_left > 0 and dst_left > 0:
        left_delta = abs(src_left - dst_left)
        if left_delta <= 30:
            score += 4
        elif left_delta <= 80:
            score += 2
    return score

def entry_has_table_block_missing_snapshot(entry: Dict[str, Any]) -> bool:
    """Check if the entry has a table block that is missing its primary imageUri or inline images."""
    blocks = entry.get("blocks") if isinstance(entry, dict) else None
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "").strip().lower() != "table":
            continue
        snapshot_uri = str(block.get("imageUri") or block.get("snapshotUri") or "").strip()
        images = block.get("images")
        has_inline_images = isinstance(images, list) and any(
            isinstance(image, dict) and str(image.get("imageUri") or image.get("src") or "").strip()
            for image in images
        )
        if not snapshot_uri and not has_inline_images:
            return True
    return False

def entry_existing_image_uris(entry: Dict[str, Any]) -> Set[str]:
    """Collect all image URIs currently referenced in an entry's blocks."""
    uris: Set[str] = set()
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return uris
    for b in blocks:
        if not isinstance(b, dict):
            continue
        # Check standard fields
        for key in ("imageUri", "image_uri", "src", "uri"):
            uri = b.get(key)
            if isinstance(uri, str) and uri.strip():
                uris.add(uri.strip())
        # Check nested images
        inline_images = b.get("images")
        if isinstance(inline_images, list):
            for img in inline_images:
                if not isinstance(img, dict): continue
                uri = img.get("imageUri") or img.get("src")
                if isinstance(uri, str) and uri.strip():
                    uris.add(uri.strip())
    return uris

def collect_kb_image_basenames(kb: Dict[str, Any]) -> Set[str]:
    """Collect basenames of all images referenced in the KB."""
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return set()
    
    basenames: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict): continue
        for uri in entry_existing_image_uris(entry):
            base = basename_from_uri(uri).strip().lower()
            if base: basenames.add(base)
            
    return basenames

def is_legend_snapshot_uri(uri: str) -> bool:
    """Check if the URI filename identifies it as a legend snapshot."""
    fn = basename_from_uri(uri).lower()
    if not fn: return False
    if fn.startswith("legend_"): return True
    from imaging.pdf_analyzer import parse_visual_snapshot_page_idx
    page, idx = parse_visual_snapshot_page_idx(fn)
    return page is not None and idx is not None

def snapshot_family(uri: str) -> str:
    """Determine if a snapshot URI belongs to 'legend' or 'table' family."""
    if is_legend_snapshot_uri(uri): return "legend"
    if is_table_snapshot_uri(uri): return "table"
    return ""

def sort_table_entry_images_after_merge(kb: Dict[str, Any], *, debug: bool = False) -> int:
    """Ensure that table snapshots and continuation images are ordered deterministically by page and index."""
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    def _parse_order(uri: str, fallback_page: int, fallback_pos: int) -> Tuple[int, int, str, int]:
        s = safe_str(uri).strip()
        page = fallback_page if fallback_page > 0 else 10**9
        idx = 10**9
        try:
            m = re.search(r"_p(\d+)_", s)
            if m: page = int(m.group(1))
        except Exception: pass
        try:
            m = re.search(r"_idx(\d+)\.(png|jpg|jpeg|webp)$", s, flags=re.IGNORECASE)
            if m: idx = int(m.group(1))
        except Exception: pass
        fn = basename_from_uri(s)
        return (int(page), int(idx), fn, int(fallback_pos))

    changed = 0
    for e in entries:
        if not isinstance(e, dict): continue
        kind = safe_str(e.get("kind")).strip().lower()
        jt = safe_str(e.get("jobTitle")).strip()
        looks_like_table = (kind == "table") or jt.startswith("表格#")
        if not looks_like_table: continue
        
        blocks = e.get("blocks")
        if not isinstance(blocks, list) or not blocks: continue

        first_table_idx = None
        for i, b in enumerate(blocks):
            if not isinstance(b, dict): continue
            if safe_str(b.get("type")).strip().lower() == "table":
                first_table_idx = int(i)
                break
        if first_table_idx is None: continue

        prefix = blocks[: first_table_idx + 1]
        rest = blocks[first_table_idx + 1 :]

        # Find the first table block and its current snapshot.
        table_block = prefix[first_table_idx]
        prev_snapshot_uri = safe_str(table_block.get("imageUri") or table_block.get("image_uri")).strip()

        images: List[Dict[str, Any]] = []
        others: List[Dict[str, Any]] = []
        for b in rest:
            if not isinstance(b, dict): continue
            if safe_str(b.get("type")).strip().lower() == "image":
                images.append(b)
            else:
                others.append(b)

        # Decide the primary snapshot family based on presence of 'legend' screenshots.
        legend_present = is_legend_snapshot_uri(prev_snapshot_uri)
        for b in images:
            if is_legend_snapshot_uri(safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src"))):
                legend_present = True
                break

        preferred_family = "legend" if legend_present else "table"
        candidates: List[Tuple[Tuple[int, int, str, int], str]] = []

        if prev_snapshot_uri and snapshot_family(prev_snapshot_uri) == preferred_family:
            candidates.append((_parse_order(prev_snapshot_uri, 1, -1), prev_snapshot_uri))

        for pos, b in enumerate(images):
            uri = safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
            if not uri or snapshot_family(uri) != preferred_family: continue
            
            fallback_page = parse_page_number(b.get("pageNumber")) or 1
            candidates.append((_parse_order(uri, fallback_page, pos), uri))

        if not candidates: continue
        
        candidates.sort(key=lambda t: t[0])
        chosen_snapshot_uri = candidates[0][1]
        chosen_fn = basename_from_uri(chosen_snapshot_uri)

        if chosen_snapshot_uri and chosen_snapshot_uri != prev_snapshot_uri:
            table_block["imageUri"] = chosen_snapshot_uri
            
            # Keep old as ImageBlock if same family
            prev_fn = basename_from_uri(prev_snapshot_uri)
            if prev_snapshot_uri and prev_fn and prev_fn != chosen_fn and snapshot_family(prev_snapshot_uri) == preferred_family:
                pn = parse_page_number(prev_snapshot_uri) or 1
                images.append(make_image_block(asset_uri=prev_snapshot_uri, page_number=pn, caption="表格原件（续）"))

        # Remove duplicate of chosen and filter family
        images = [b for b in images if basename_from_uri(safe_str(b.get("imageUri") or b.get("src"))) != chosen_fn and snapshot_family(safe_str(b.get("imageUri") or b.get("src"))) == preferred_family]

        # Sort remaining continuation images
        if len(images) > 1:
            ordered = []
            for pos, b in enumerate(images):
                uri = safe_str(b.get("imageUri") or b.get("src")).strip()
                fallback_page = parse_page_number(b.get("pageNumber")) or 1
                ordered.append((_parse_order(uri, fallback_page, pos), b))
            ordered.sort(key=lambda t: t[0])
            images = [item[1] for item in ordered]

        e["blocks"] = prefix + images + others
        changed += 1

    return changed

def rebind_table_snapshot_uris_by_content(
    *,
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    debug_merge: bool,
) -> int:
    """Rebind table snapshots to better matching entries based on OCR content similarity."""
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    table_indices = [
        idx for idx, entry in enumerate(entries)
        if isinstance(entry, dict) and entry_has_table_block(entry)
    ]
    if len(table_indices) <= 1:
        return 0

    ocr_cache: Dict[str, str] = {}

    def cached_ocr(path: str) -> str:
        if not path:
            return ""
        if path not in ocr_cache:
            ocr_cache[path] = ocr_image_file(path)
        return ocr_cache[path]

    rebound = 0
    for source_idx in table_indices:
        source_entry = entries[source_idx]
        if not isinstance(source_entry, dict):
            continue
        source_table = first_table_block(source_entry)
        if not isinstance(source_table, dict):
            continue
        snapshot_uri = safe_str(source_table.get("imageUri") or source_table.get("image_uri")).strip()
        if not snapshot_uri:
            continue

        local_path = resolve_table_snapshot_local_path(manifest, snapshot_uri)
        if not local_path:
            continue
        snapshot_text = cached_ocr(local_path)
        if len(collapse_match_text(snapshot_text)) < 6:
            continue

        page_number = int(physical_page_from_asset_uri(snapshot_uri) or entry_page_hint(source_entry) or 1)
        current_score, current_hits = score_table_fragment_match(snapshot_text, source_entry, page_number=page_number)

        best_idx = source_idx
        best_score = int(current_score)
        best_hits = int(current_hits)
        for target_idx in table_indices:
            target_entry = entries[target_idx]
            if not isinstance(target_entry, dict):
                continue
            score, hits = score_table_fragment_match(snapshot_text, target_entry, page_number=page_number)
            if score > best_score or (score == best_score and hits > best_hits):
                best_idx = int(target_idx)
                best_score = int(score)
                best_hits = int(hits)

        if best_idx == source_idx:
            continue
        if best_hits < 2 or best_score < 10:
            continue
        if best_score < current_score + 6 and best_hits < current_hits + 2:
            continue

        target_entry = entries[best_idx]
        if not isinstance(target_entry, dict):
            continue
        target_table = first_table_block(target_entry)
        if not isinstance(target_table, dict):
            continue

        target_existing = entry_existing_image_uris(target_entry)
        if snapshot_uri in target_existing:
            source_table.pop("imageUri", None)
            source_table.pop("image_uri", None)
            rebound += 1
            if debug_merge:
                print(
                    f"[MergeDebug] TableSnapshot content-rebind duplicate: {basename_from_uri(snapshot_uri)} -> "
                    f"entryId={safe_str(target_entry.get('entryId')).strip()}"
                )
            continue

        prev_target_uri = safe_str(target_table.get("imageUri") or target_table.get("image_uri")).strip()
        if prev_target_uri and prev_target_uri != snapshot_uri and prev_target_uri not in target_existing:
            prev_page = int(physical_page_from_asset_uri(prev_target_uri) or entry_page_hint(target_entry) or 1)
            insert_block(
                entry=target_entry,
                block=make_image_block(asset_uri=prev_target_uri, page_number=prev_page, caption="表格原件（续）"),
                page_number=prev_page,
                insert_mode="append",
            )

        target_table["imageUri"] = snapshot_uri
        source_table.pop("imageUri", None)
        source_table.pop("image_uri", None)
        rebound += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableSnapshot content-rebind: {basename_from_uri(snapshot_uri)} "
                f"srcEntry={safe_str(source_entry.get('entryId')).strip()} -> "
                f"targetEntry={safe_str(target_entry.get('entryId')).strip()} score={best_score} hits={best_hits} "
                f"prevScore={current_score} prevHits={current_hits}"
            )

    return int(rebound)


def reroute_existing_legend_images_into_tables(
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    dry_run: bool = False,
    debug_merge: bool = False,
) -> int:
    """Compatibility wrapper for rerouting existing table/legend snapshots by content."""
    if dry_run:
        return 0
    return rebind_table_snapshot_uris_by_content(
        kb=kb,
        manifest=manifest,
        debug_merge=debug_merge,
    )


def merge_table_fragments_before_appendix_fill(
    *,
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    used_manifest_indices: Set[int],
    stats: Any,
    dry_run: bool,
    debug_merge: bool,
) -> int:
    """Identify legend/table fragments in manifest and route them to matching table entries in KB."""
    entries = kb.get("entries")
    items = manifest.get("items")
    if not isinstance(entries, list) or not isinstance(items, list):
        return 0

    table_indices = [
        idx for idx, entry in enumerate(entries)
        if isinstance(entry, dict) and entry_has_table_block(entry)
    ]
    if not table_indices:
        return 0

    ocr_cache: Dict[str, str] = {}

    def cached_ocr(path: str) -> str:
        if not path:
            return ""
        if path not in ocr_cache:
            ocr_cache[path] = ocr_image_file(path)
        return ocr_cache[path]

    handled = 0
    for item_index, item in enumerate(items):
        if item_index in used_manifest_indices:
            continue
        if not isinstance(item, dict):
            continue

        # Try multiple URI keys
        uri = ""
        for key in ("assetUri", "asset_uri", "imageUri", "image_uri", "src"):
            val = item.get(key)
            if val:
                uri = str(val).strip()
                break
        if not uri:
            continue

        base = basename_from_uri(uri).lower()
        kind = safe_str(item.get("kind")).strip().lower()
        is_legend_like = kind == "legend" or base.startswith("legend_")
        if not is_legend_like:
            continue

        local_path = resolve_manifest_local_path(manifest, item)
        if not local_path:
            continue

        item_text = cached_ocr(local_path)
        if len(collapse_match_text(item_text)) < 6:
            continue

        page_number = int(parse_page_number(item.get("pageNumber")) or 1)
        best_idx: Optional[int] = None
        best_score = 0
        best_hits = 0
        for entry_index in table_indices:
            entry = entries[entry_index]
            if not isinstance(entry, dict):
                continue
            score, hits = score_table_fragment_match(item_text, entry, page_number=page_number)
            if score > best_score or (score == best_score and hits > best_hits):
                best_idx = int(entry_index)
                best_score = int(score)
                best_hits = int(hits)

        if best_idx is None:
            continue
        if best_hits < 2 and best_score < 10:
            continue

        entry = entries[best_idx]
        if not isinstance(entry, dict):
            continue

        existing_uris = entry_existing_image_uris(entry)
        if uri in existing_uris:
            used_manifest_indices.add(int(item_index))
            if hasattr(stats, 'skipped_duplicate'): stats.skipped_duplicate += 1
            handled += 1
            continue

        table_block = first_table_block(entry)
        snapshot_uri = ""
        if isinstance(table_block, dict):
            snapshot_uri = safe_str(table_block.get("imageUri") or table_block.get("image_uri")).strip()
        snapshot_path = resolve_table_snapshot_local_path(manifest, snapshot_uri)
        snapshot_text = cached_ocr(snapshot_path) if snapshot_path else ""
        snapshot_base = basename_from_uri(snapshot_uri).lower()
        # current_is_legend = base.startswith("legend_")
        snapshot_is_legend = snapshot_base.startswith("legend_")

        if snapshot_text and snapshot_is_legend and snapshot_covers_fragment(snapshot_text, item_text):
            # If already covered by primary snapshot, skip.
            # (Removal logic removed as it belongs to the caller's KB cleanup pass if needed)
            used_manifest_indices.add(int(item_index))
            if hasattr(stats, 'skipped_duplicate'): stats.skipped_duplicate += 1
            handled += 1
            if debug_merge:
                print(
                    f"[MergeDebug] TableFragment duplicate: item#{item_index} {basename_from_uri(uri)} -> "
                    f"entryId={safe_str(entry.get('entryId')).strip()} jobTitle={safe_str(entry.get('jobTitle')).strip()[:60]}"
                )
            continue

        if not dry_run:
            block = make_image_block(asset_uri=uri, page_number=page_number, caption="表格原件（续）")
            insert_block(entry=entry, block=block, page_number=page_number, insert_mode="append")

        used_manifest_indices.add(int(item_index))
        if hasattr(stats, 'inserted'): stats.inserted += 1
        record_inserted_manifest_asset(
            stats,
            item_index=item_index,
            item=item,
            asset_uri=uri,
            page_number=page_number,
            caption_text="表格原件（续）",
            anchor_text=safe_str(item.get("anchorText")).strip(),
            heading_text=safe_str(item.get("headingText")).strip(),
            item_kind="table",
            target_entry_id=safe_str(entry.get("entryId")).strip(),
            insert_mode="append",
        )
        handled += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableFragment reroute: item#{item_index} {basename_from_uri(uri)} -> "
                    f"entryId={safe_str(entry.get('entryId')).strip()} as {basename_from_uri(uri)}"
            )

    return int(handled)

def bridge_table_snapshot_prev_page_figure_continuations(
    *,
    kb: Dict[str, Any],
    debug_merge: bool,
) -> int:
    """Bridge table snapshots to preceding entries if they look like continuations from the previous page."""
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    moved = 0
    for source_idx, source_entry in enumerate(entries):
        if not isinstance(source_entry, dict) or not entry_has_table_block(source_entry):
            continue
        source_table = first_table_block(source_entry)
        if not isinstance(source_table, dict):
            continue

        snapshot_uri = safe_str(source_table.get("imageUri") or source_table.get("image_uri")).strip()
        if not snapshot_uri:
            continue
        page_number = int(physical_page_from_asset_uri(snapshot_uri) or 0)
        if page_number <= 1:
            continue

        source_bbox = normalize_bbox_dict(source_table.get("bbox")) if isinstance(source_table.get("bbox"), dict) else {}

        ranked: List[Tuple[int, int]] = []
        for target_idx in range(0, int(source_idx)):
            target_entry = entries[target_idx]
            if not isinstance(target_entry, dict) or entry_has_table_block(target_entry):
                continue
            blocks = target_entry.get("blocks")
            if not isinstance(blocks, list):
                continue

            score = 0
            prev_page_hits = 0
            visual_prev_hits = 0
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if safe_str(block.get("type")).strip().lower() != "image":
                    continue
                block_uri = safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
                block_page = int(block.get("pageNumber") or physical_page_from_asset_uri(block_uri) or 0)
                if block_page != page_number - 1:
                    continue

                prev_page_hits += 1
                if basename_from_uri(block_uri).lower().startswith(f"visual_p{page_number - 1}_"):
                    visual_prev_hits += 1

                block_bbox = normalize_bbox_dict(block.get("bbox")) if isinstance(block.get("bbox"), dict) else {}
                score += score_table_visual_match(source_bbox, block_bbox)

            if prev_page_hits <= 0:
                continue

            score += 5 * min(visual_prev_hits, 1)
            if prev_page_hits == 1:
                score += 2

            labels = entry_reference_labels(target_entry)
            if labels:
                score += 2

            content_blob = " ".join(
                [
                    safe_str(target_entry.get("contentMarkdown")),
                    safe_str(target_entry.get("contentNormalized")),
                    safe_str(target_entry.get("jobTitle")),
                ]
            )
            compact_content = collapse_match_text(content_blob)
            if "明细表" in compact_content:
                score += 8

            distance = int(source_idx) - int(target_idx)
            score -= min(max(distance // 15, 0), 6)
            ranked.append((int(score), int(target_idx)))

        if not ranked:
            continue

        ranked.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_target_idx = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -999
        if best_score < 10:
            continue
        if second_score > -999 and (best_score - second_score) < 3:
            continue

        target_entry = entries[best_target_idx]
        if not isinstance(target_entry, dict):
            continue
        if snapshot_uri in entry_existing_image_uris(target_entry):
            source_table.pop("imageUri", None)
            source_table.pop("image_uri", None)
            moved += 1
            continue

        insert_block(
            entry=target_entry,
            block=make_image_block(asset_uri=snapshot_uri, page_number=page_number, caption="图件续页"),
            page_number=page_number,
            insert_mode="append",
        )
        source_table.pop("imageUri", None)
        source_table.pop("image_uri", None)
        moved += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableSnapshot prev-page-bridge: {basename_from_uri(snapshot_uri)} "
                f"srcEntry={safe_str(source_entry.get('entryId')).strip()} -> "
                f"targetEntry={safe_str(target_entry.get('entryId')).strip()} score={best_score}"
            )

    return int(moved)

def bridge_previous_legend_page_into_tables(
    *,
    kb: Dict[str, Any],
    dry_run: bool,
    debug_merge: bool,
) -> int:
    """Bridge legend snapshots to table entries if they belong to the same sequence across pages."""
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    def _parse_legend_page(uri: str) -> Optional[int]:
        match = re.search(r"legend_p(\d+)_idx\d+", basename_from_uri(uri), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    page_to_occurrences: Dict[int, List[Tuple[int, str]]] = {}
    table_pages_in_use: Set[int] = set()
    table_targets: List[Tuple[int, int, str]] = []

    for entry_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        is_table = entry_has_table_block(entry)
        entry_legend_uris: List[str] = []
        table_block = first_table_block(entry) if is_table else None
        if isinstance(table_block, dict):
            primary_uri = safe_str(table_block.get("imageUri") or table_block.get("image_uri")).strip()
            if _parse_legend_page(primary_uri) is not None:
                entry_legend_uris.append(primary_uri)

        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_uri = safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
            page = _parse_legend_page(block_uri)
            if page is None:
                continue
            page_to_occurrences.setdefault(int(page), []).append((int(entry_idx), block_uri))
            if is_table:
                table_pages_in_use.add(int(page))
                entry_legend_uris.append(block_uri)

        if is_table and entry_legend_uris:
            unique_legend_uris = sorted(
                set(entry_legend_uris),
                key=lambda uri: (_parse_legend_page(uri) or 10**9, basename_from_uri(uri)),
            )
            primary_uri = unique_legend_uris[0]
            primary_page = _parse_legend_page(primary_uri)
            if primary_page is not None:
                table_pages_in_use.add(int(primary_page))
                table_targets.append((int(primary_page), int(entry_idx), primary_uri))

    moved = 0
    for primary_page, entry_idx, primary_uri in sorted(table_targets, key=lambda item: (item[0], item[1])):
        prev_page = int(primary_page) - 1
        if prev_page <= 0 or prev_page in table_pages_in_use:
            continue
        candidates = page_to_occurrences.get(prev_page, [])
        candidates = [(src_idx, uri) for src_idx, uri in candidates if int(src_idx) != int(entry_idx)]
        if len(candidates) != 1:
            continue

        src_idx, uri = candidates[0]
        target_entry = entries[entry_idx]
        if not isinstance(target_entry, dict):
            continue
        if uri in entry_existing_image_uris(target_entry):
            continue

        if not dry_run:
            remove_image_references_by_basename(kb, uri, keep_entry_idx=entry_idx)
            insert_block(
                entry=target_entry,
                block=make_image_block(asset_uri=uri, page_number=prev_page, caption="表格原件（续）"),
                page_number=prev_page,
                insert_mode="append",
            )

        table_pages_in_use.add(prev_page)
        moved += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableLegend bridge-prev-page: {basename_from_uri(uri)} -> "
                f"entryId={safe_str(target_entry.get('entryId')).strip()}"
            )

    return int(moved)

def detect_table_bboxes_from_image_bytes(image_bytes: bytes, debug: bool = False) -> List[Tuple[int, int, int, int]]:
    """Return list of bounding boxes (x,y,w,h) that likely contain tables.
    Uses computer vision (line detection) to identify table structures.
    """
    if cv2 is None or np is None:
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
            boxes.append((int(x), int(y), int(w), int(h)))
        
        # sort by top-left y then x
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        return boxes
    except Exception as ex:
        logger.error(f"Error in CV table detection: {ex}")
        return []

def _merge_close_coords(values: Sequence[float], tol: float = 0.5) -> List[float]:
    """Sort and merge nearly-equal boundary coordinates within tolerance."""
    if not values:
        return []
    ordered = sorted(float(v) for v in values)
    merged = [ordered[0]]
    for v in ordered[1:]:
        if abs(v - merged[-1]) > tol:
            merged.append(v)
        # else: keep existing boundary (within tol)
    return merged


def map_cell_rect_to_grid(
    rect: Sequence[float],
    xs: Sequence[float],
    ys: Sequence[float],
    tol: float = 0.5,
) -> Tuple[int, int, int, int]:
    """Map an (x0,y0,x1,y1) rect to (row, col, rowSpan, colSpan) via boundary grids."""
    if len(rect) < 4 or not xs or not ys:
        raise ValueError("empty rect or boundary lists")

    def _edge_index(val: float, edges: Sequence[float]) -> int:
        best = 0
        best_d = abs(float(edges[0]) - val)
        for i, e in enumerate(edges):
            d = abs(float(e) - val)
            if d <= tol:
                return i
            if d < best_d:
                best_d = d
                best = i
        return best

    x0, y0, x1, y1 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    c0 = _edge_index(x0, xs)
    c1 = _edge_index(x1, xs)
    r0 = _edge_index(y0, ys)
    r1 = _edge_index(y1, ys)
    if c1 <= c0:
        c1 = min(c0 + 1, len(xs) - 1)
        if c1 <= c0:
            c1 = c0 + 1
    if r1 <= r0:
        r1 = min(r0 + 1, len(ys) - 1)
        if r1 <= r0:
            r1 = r0 + 1
    return r0, c0, max(1, r1 - r0), max(1, c1 - c0)


def build_native_table_grid(table: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build physical cells from table.rows[].cells[] geometry.

    Returns (cells, warnings).

    - Uses row.cells[col] bbox tuples (not the flat table.cells index order).
    - None slots are span coverage — never treated as colspan by themselves.
    - Merged rects are emitted once.
    - Text comes from table.extract() at the anchor (or first non-None in span).
    - bbox is None only if geometry is unreliable (warning recorded).
    """
    warnings: List[str] = []
    try:
        text_matrix = table.extract()
    except Exception as ex:
        return [], [f"extract failed: {ex}"]
    if not text_matrix:
        return [], ["empty extract matrix"]

    rows_attr = getattr(table, "rows", None)
    if rows_attr is None:
        return [], ["table.rows unavailable"]

    # Collect non-null cell rects with their first-seen grid position.
    raw_rects: List[Tuple[int, int, Tuple[float, float, float, float]]] = []
    try:
        for ri, row in enumerate(rows_attr):
            cells_row = list(getattr(row, "cells", []) or [])
            for ci, cell in enumerate(cells_row):
                if cell is None:
                    continue
                # cell is typically (x0, y0, x1, y1)
                if not isinstance(cell, (tuple, list)) or len(cell) < 4:
                    continue
                rect = (float(cell[0]), float(cell[1]), float(cell[2]), float(cell[3]))
                raw_rects.append((ri, ci, rect))
    except Exception as ex:
        return [], [f"rows/cells walk failed: {ex}"]

    if not raw_rects:
        return [], ["no cell rects"]

    xs: List[float] = []
    ys: List[float] = []
    for _, _, (x0, y0, x1, y1) in raw_rects:
        xs.extend([x0, x1])
        ys.extend([y0, y1])
    xs = _merge_close_coords(xs)
    ys = _merge_close_coords(ys)
    if len(xs) < 2 or len(ys) < 2:
        return [], ["insufficient unique boundaries"]

    n_rows_logical = max(len(text_matrix), len(ys) - 1)
    n_cols_logical = max(max(len(r) for r in text_matrix), len(xs) - 1)

    seen_keys = set()
    cells: List[Dict[str, Any]] = []
    for ri, ci, rect in raw_rects:
        # Dedupe identical merged rects (same geometry → one physical cell).
        key = (round(rect[0], 1), round(rect[1], 1), round(rect[2], 1), round(rect[3], 1))
        if key in seen_keys:
            continue
        seen_keys.add(key)

        try:
            r0, c0, row_span, col_span = map_cell_rect_to_grid(rect, xs, ys)
        except ValueError as ex:
            warnings.append(f"grid map failed at r{ri}c{ci}: {ex}")
            continue

        # Clamp into logical matrix
        if r0 >= n_rows_logical or c0 >= n_cols_logical:
            warnings.append(
                f"cell rect out of logical grid at r{ri}c{ci} → ({r0},{c0}); bbox omitted"
            )
            bbox_out = None
        else:
            bbox_out = [float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])]

        # Text: prefer extract at source position; else anchor; else first non-None in span.
        text_val: Any = None
        if 0 <= ri < len(text_matrix) and 0 <= ci < len(text_matrix[ri]):
            text_val = text_matrix[ri][ci]
        if text_val is None and 0 <= r0 < len(text_matrix) and 0 <= c0 < len(text_matrix[r0]):
            text_val = text_matrix[r0][c0]
        if text_val is None:
            for rr in range(r0, min(r0 + row_span, len(text_matrix))):
                rowm = text_matrix[rr]
                for cc in range(c0, min(c0 + col_span, len(rowm))):
                    if rowm[cc] is not None:
                        text_val = rowm[cc]
                        break
                if text_val is not None:
                    break

        # Rowspan placeholders (None in extract under the span) must NOT inflate colSpan.
        # Geometry already computed colSpan from rect width — keep it.

        cell_alignment = "left"
        if r0 == 0:
            cell_alignment = "center"
        elif text_val and str(text_val).strip().replace(".", "", 1).isdigit():
            cell_alignment = "center"

        if bbox_out is None:
            warnings.append(f"missing bbox for cell ({r0},{c0})")

        cells.append(
            {
                "row": int(r0),
                "col": int(c0),
                "rowSpan": int(row_span),
                "colSpan": int(col_span),
                "text": str(text_val if text_val is not None else "").strip(),
                "isHeader": r0 == 0,
                "alignment": cell_alignment,
                "bbox": bbox_out,
            }
        )

    cells.sort(key=lambda c: (c["row"], c["col"], c.get("colSpan") or 0))
    # Sanity: rowSpan/colSpan >= 1
    for c in cells:
        if c["rowSpan"] < 1 or c["colSpan"] < 1:
            warnings.append(
                f"invalid span at ({c['row']},{c['col']}) "
                f"rowSpan={c['rowSpan']} colSpan={c['colSpan']}"
            )
            c["rowSpan"] = max(1, int(c["rowSpan"]))
            c["colSpan"] = max(1, int(c["colSpan"]))
    return cells, warnings


def extract_native_table_cells(table: Any) -> List[Dict[str, Any]]:
    """Convert PyMuPDF native table to physical cells with geometry-based spans.

    Uses table.rows[].cells[] (not flat table.cells index arithmetic).
    """
    try:
        cells, warnings = build_native_table_grid(table)
        for w in warnings:
            logger.warning(f"native table cell geometry: {w}")
        return cells
    except Exception as ex:
        logger.error(f"Error extracting native table cells: {ex}")
        return []
