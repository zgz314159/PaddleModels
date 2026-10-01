import hashlib
import math
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from imaging.text_utils import safe_str
from imaging.pdf_analyzer import basename_from_uri, parse_page_number

def pick_first_str(it: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    """Return the first non-empty string value found for the given keys in a dictionary."""
    for k in keys:
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def caption_from_manifest_item(it: Dict[str, Any]) -> str:
    """Extract a suitable caption for a manifest item (image/table)."""
    # DOCX shapes manifest: anchorText
    # PDF crop manifest: kind/method
    caption = pick_first_str(it, ("anchorText", "headingText", "caption", "title", "kind"))
    return caption or ""

def semantic_texts_from_manifest_item(it: Dict[str, Any]) -> List[str]:
    """Extract all text signals from a manifest item for semantic matching."""
    from imaging.semantic_matcher import has_meaningful_anchor_text, normalize_for_similarity
    from imaging.text_utils import get_text_fingerprint
    texts: List[str] = []
    seen: Set[str] = set()
    for key in ("anchorText", "headingText", "caption", "title"):
        text = safe_str(it.get(key)).strip()
        if not has_meaningful_anchor_text(text):
            continue
        fp = get_text_fingerprint(text) or normalize_for_similarity(text)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        texts.append(text)
    return texts

def get_block_bbox(block: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract bounding box from a block, normalizing common field names."""
    from imaging.bbox_utils import normalize_bbox_dict
    return normalize_bbox_dict(block.get("bbox") or block.get("box") or block.get("cropBox") or block.get("rect"))

def get_manifest_item_bbox(item: Dict[str, Any]) -> Optional[Dict[str, int]]:
    """Extract bounding box from a manifest item."""
    from imaging.bbox_utils import normalize_bbox_dict
    return normalize_bbox_dict(item.get("bbox") or item.get("box") or item.get("cropBox") or item.get("rect"))

def bbox_sort_key_from_item(item: Dict[str, Any], item_index: int) -> Tuple[int, int, int]:
    """Generate a stable sort key for manifest items based on their physical position."""
    bbox = get_manifest_item_bbox(item)
    if isinstance(bbox, dict):
        return (int(bbox.get("top", 10**9)), int(bbox.get("left", 10**9)), int(item_index))
    return (10**9, 10**9, int(item_index))

def get_block_text_payload(block: Dict[str, Any]) -> str:
    """Extract text content from various block types (text, code, markdown, etc.)."""
    for key in ("code", "text", "contentMarkdown", "content", "caption"):
        value = safe_str(block.get(key)).strip()
        if value:
            return value
    return ""

def copy_bbox_dict(bbox: Any) -> Optional[Dict[str, int]]:
    """Deep copy a bbox dictionary, ensuring all values are integers."""
    if not isinstance(bbox, dict):
        return None
    out: Dict[str, int] = {}
    for key in ("left", "top", "right", "bottom", "width", "height"):
        try:
            out[key] = int(bbox.get(key, 0))
        except Exception:
            out[key] = 0
    return out

from dataclasses import dataclass, field

@dataclass
class MergeScope:
    unit_name_contains: List[str]
    job_title_contains: List[str]
    entry_id_prefix: List[str]

def entry_in_scope(entry: Dict[str, Any], scope: MergeScope) -> bool:
    """Check if a KB entry matches the defined merge scope."""
    if scope.unit_name_contains:
        unit = safe_str(entry.get("unitName"))
        if not any(s in unit for s in scope.unit_name_contains):
            return False

    if scope.job_title_contains:
        jt = safe_str(entry.get("jobTitle"))
        if not any(s in jt for s in scope.job_title_contains):
            return False

    if scope.entry_id_prefix:
        eid = safe_str(entry.get("entryId"))
        if not any(eid.startswith(p) for p in scope.entry_id_prefix):
            return False

    return True

def make_image_block(asset_uri: str, page_number: int, caption: str, bbox: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a standardized Image block for the KB entry."""
    norm = f"image|{asset_uri}|{page_number}|{caption}".encode("utf-8")
    digest = hashlib.sha1(norm).hexdigest()[:12]
    block: Dict[str, Any] = {
        "id": f"b_shape_{digest}",
        "type": "image",
        "src": asset_uri,
        "imageUri": asset_uri,
        "pageNumber": page_number,
        "semanticRole": "figure",
    }
    caption = caption.strip()
    if caption:
        block["caption"] = caption
    if isinstance(bbox, dict):
        block["bbox"] = {
            "left": int(bbox.get("left", 0)),
            "top": int(bbox.get("top", 0)),
            "right": int(bbox.get("right", 0)),
            "bottom": int(bbox.get("bottom", 0)),
            "width": int(bbox.get("width", 0)),
            "height": int(bbox.get("height", 0)),
        }
    return block

def make_markdown_code_block(text: str, page_number: int) -> Dict[str, Any]:
    """Create a standardized Markdown code block for the KB entry."""
    code = safe_str(text)
    digest = hashlib.sha1(f"code|markdown|{page_number}|{code}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"b_code_{digest}",
        "type": "code",
        "language": "markdown",
        "code": code,
        "pageNumber": int(page_number or 1),
    }

def get_block_page_number(block: Dict[str, Any]) -> Optional[int]:
    """Helper to extract page number from a block, with fallback parsing."""
    return parse_page_number(block.get("pageNumber"))

def insert_block(
    entry: Dict[str, Any],
    block: Dict[str, Any],
    page_number: int,
    insert_mode: str,
    *,
    page_rank_hint: Optional[int] = None,
    page_total_hint: Optional[int] = None,
) -> None:
    """Insert a block into an entry's blocks list based on various positioning modes."""
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        blocks = []
        entry["blocks"] = blocks

    def _same_page_indices(text_only: bool) -> List[int]:
        out: List[int] = []
        for i, existing in enumerate(blocks):
            if not isinstance(existing, dict):
                continue
            if get_block_page_number(existing) != page_number:
                continue
            if text_only:
                t = safe_str(existing.get("type")).strip().lower()
                if t not in {"text", "p", "paragraph", "h1", "h2", "h3", "heading1", "heading2", "heading3", "quote", "blockquote", "code"}:
                    continue
            out.append(i)
        return out

    def _rank_anchor_index(text_only: bool) -> Optional[int]:
        if page_rank_hint is None or page_total_hint is None or int(page_total_hint) <= 0:
            return None
        candidates = _same_page_indices(text_only=text_only)
        if not candidates:
            return None
        slot_count = len(candidates)
        mapped = int(math.floor(((int(page_rank_hint) + 1) * slot_count - 1) / max(1, int(page_total_hint))))
        mapped = max(0, min(slot_count - 1, mapped))
        return int(candidates[mapped])

    if insert_mode == "append":
        blocks.append(block)
        return

    if insert_mode == "after-page":
        last_idx = _rank_anchor_index(text_only=False)
        if last_idx is None:
            last_idx = -1
            for i, b in enumerate(blocks):
                if not isinstance(b, dict):
                    continue
                if get_block_page_number(b) == page_number:
                    last_idx = i
        if last_idx < 0:
            blocks.append(block)
        else:
            blocks.insert(last_idx + 1, block)
        return

    if insert_mode == "after-page-text":
        insert_after = _rank_anchor_index(text_only=True)
        if insert_after is None:
            last_text_idx = -1
            last_any_idx = -1
            for i, b in enumerate(blocks):
                if not isinstance(b, dict):
                    continue
                if get_block_page_number(b) != page_number:
                    continue
                last_any_idx = i
                t = safe_str(b.get("type")).strip().lower()
                if t in {"text", "p", "paragraph", "h1", "h2", "h3", "heading1", "heading2", "heading3", "quote", "blockquote", "code"}:
                    last_text_idx = i
            insert_after = last_text_idx if last_text_idx >= 0 else last_any_idx
        if insert_after is None or insert_after < 0:
            blocks.append(block)
        else:
            blocks.insert(insert_after + 1, block)
        return

    raise ValueError(f"Unknown insert_mode: {insert_mode}")

def insert_block_after_markdown_code(entry: Dict[str, Any], block: Dict[str, Any]) -> None:
    """Insert block right after the first markdown code block (best for 附件类)."""
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        blocks = []
        entry["blocks"] = blocks

    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            continue
        if safe_str(b.get("type")).strip().lower() == "code" and safe_str(b.get("language")).strip().lower() == "markdown":
            blocks.insert(i + 1, block)
            return

    blocks.append(block)

def remove_image_references_by_basename(
    kb: Dict[str, Any],
    image_name: str,
    *,
    keep_entry_idx: Optional[int] = None,
) -> int:
    """Remove references to an image (by basename) across the whole KB, optionally keeping one entry."""
    target = (basename_from_uri(image_name) or safe_str(image_name)).strip().lower()
    if not target:
        return 0

    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    removed = 0
    for entry_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if keep_entry_idx is not None and int(entry_idx) == int(keep_entry_idx):
            continue

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        kept_blocks: List[Any] = []
        changed = False
        for block in blocks:
            if not isinstance(block, dict):
                kept_blocks.append(block)
                continue

            uri = safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src") or block.get("uri")).strip()
            block_name = basename_from_uri(uri).strip().lower()
            if block_name == target:
                changed = True
                removed += 1
                continue
            kept_blocks.append(block)

        if changed:
            entry["blocks"] = kept_blocks
            
    return removed

def record_inserted_manifest_asset(
    stats: Any,
    *,
    item_index: int,
    item: Dict[str, Any],
    asset_uri: str,
    page_number: Optional[int],
    caption_text: str,
    anchor_text: str,
    heading_text: str,
    item_kind: str,
    target_entry_id: str,
    insert_mode: str,
    bbox: Optional[Dict[str, int]] = None,
) -> None:
    """Record a successful manifest asset insertion into stats."""
    basename = basename_from_uri(asset_uri)
    if not basename:
        return
    if not hasattr(stats, 'inserted_assets'):
        return
        
    stats.inserted_assets.append({
        "itemIndex": int(item_index),
        "assetUri": safe_str(asset_uri).strip(),
        "basename": basename,
        "pageNumber": parse_page_number(page_number),
        "captionText": safe_str(caption_text).strip(),
        "anchorText": safe_str(anchor_text).strip(),
        "headingText": safe_str(heading_text).strip(),
        "itemKind": safe_str(item_kind).strip().lower(),
        "targetEntryId": safe_str(target_entry_id).strip(),
        "insertMode": safe_str(insert_mode).strip() or "after-page-text",
        "bbox": copy_bbox_dict(bbox),
    })
