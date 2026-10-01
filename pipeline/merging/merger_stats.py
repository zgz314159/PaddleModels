from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from imaging.text_utils import safe_str
from imaging.pdf_analyzer import parse_page_number, basename_from_uri
from imaging.kb_utils import copy_bbox_dict
from imaging.semantic_matcher import manifest_item_semantic_kind_hint

@dataclass
class MergeStats:
    items_total: int = 0
    items_with_page: int = 0
    items_with_uri: int = 0
    inserted: int = 0
    skipped_duplicate: int = 0
    skipped_no_entry_for_page: int = 0
    skipped_missing_fields: int = 0
    skipped_out_of_scope: int = 0
    ambiguous_pages: int = 0
    inserted_assets: List[Dict[str, Any]] = field(default_factory=list)

def record_inserted_manifest_asset(
    stats: MergeStats,
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
    bbox: Optional[Dict[str, Any]] = None,
) -> None:
    basename = basename_from_uri(asset_uri)
    if not basename:
        return
    stats.inserted_assets.append(
        {
            "itemIndex": int(item_index),
            "assetUri": safe_str(asset_uri).strip(),
            "basename": basename,
            "pageNumber": parse_page_number(page_number),
            "captionText": safe_str(caption_text).strip(),
            "anchorText": safe_str(anchor_text).strip(),
            "headingText": safe_str(heading_text).strip(),
            "itemKind": safe_str(item_kind).strip().lower() or manifest_item_semantic_kind_hint(item),
            "targetEntryId": safe_str(target_entry_id).strip(),
            "insertMode": safe_str(insert_mode).strip() or "after-page-text",
            "bbox": copy_bbox_dict(bbox),
        }
    )
