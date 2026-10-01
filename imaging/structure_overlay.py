import re
from typing import Any, Dict, List, Optional, Tuple

from imaging.text_utils import (
    normalize_for_search_like_app,
    get_text_fingerprint,
    safe_str,
    ocr_match_tokens
)
from imaging.pdf_analyzer import (
    entry_page_hint,
    basename_from_uri
)
from imaging.semantic_matcher import (
    semantic_similarity_ratio
)
from imaging.kb_utils import (
    get_block_page_number,
    get_block_text_payload
)

def extract_source_structure_blocks_by_page(source_native_kb: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    """Group blocks from a native PDF KB by their physical page number, sorted by reading order."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    entries = source_native_kb.get("entries")
    if not isinstance(entries, list):
        return out

    for entry in entries:
        if not isinstance(entry, dict): continue
        entry_page = entry_page_hint(entry)
        blocks = entry.get("blocks")
        if not isinstance(blocks, list): continue
        
        for order_index, block in enumerate(blocks, start=1):
            if not isinstance(block, dict): continue
            page_number = get_block_page_number(block) or entry_page
            if not isinstance(page_number, int) or page_number <= 0: continue
            
            bbox = block.get("bbox") or block.get("box") or block.get("cropBox") or block.get("rect")
            if not isinstance(bbox, dict): continue
            
            block_type = safe_str(block.get("type")).strip().lower() or "unknown"
            out.setdefault(int(page_number), []).append({
                "type": block_type,
                "pageNumber": int(page_number),
                "bbox": dict(bbox),
                "readingOrder": int(block.get("readingOrder") or order_index),
                "text": get_block_text_payload(block),
                "imageUri": safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip(),
                "semanticRole": safe_str(block.get("semanticRole")).strip()
            })

    for page_number, blocks in out.items():
        blocks.sort(key=lambda b: (
            int(b.get("readingOrder") or 10**9),
            int((b.get("bbox") or {}).get("top", 10**9)),
            int((b.get("bbox") or {}).get("left", 10**9))
        ))
    return out

def score_structure_block_match(target_text: str, source_block: Dict[str, Any]) -> int:
    """Score matching quality between a target text and a source structural block."""
    source_text = safe_str(source_block.get("text")).strip()
    if not target_text or not source_text:
        return 0

    target_norm = normalize_for_search_like_app(target_text).replace("\n", " ").strip()
    source_norm = normalize_for_search_like_app(source_text).replace("\n", " ").strip()
    target_compact = re.sub(r"\s+", "", target_norm)
    source_compact = re.sub(r"\s+", "", source_norm)
    target_fp = get_text_fingerprint(target_text)
    source_fp = get_text_fingerprint(source_text)

    score = 0
    if target_norm and source_norm and (target_norm in source_norm or source_norm in target_norm):
        score += 900
    elif target_compact and source_compact and (target_compact in source_compact or source_compact in target_compact):
        score += 760

    if target_fp and source_fp and (target_fp in source_fp or source_fp in target_fp):
        score += 950

    token_hits = 0
    haystacks = [source_compact, source_fp]
    for token in ocr_match_tokens(target_text)[:12]:
        compact = re.sub(r"\s+", "", token)
        if not compact or len(compact) < 2: continue
        if any(compact in hay for hay in haystacks if hay):
            token_hits += 1
    score += int(token_hits * 55)

    score += int(semantic_similarity_ratio(target_text, source_text, cutoff_ratio=0.35) * 220.0)
    return int(score)

def overlay_structure_bboxes_from_source(kb: Dict[str, Any], source_native_kb: Dict[str, Any], *, debug: bool = False) -> Dict[str, int]:
    """Overlay high-precision bounding boxes from a native PDF extraction onto a heuristic KB (e.g. OCR-based)."""
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return {"textBlocks": 0, "visualBlocks": 0}

    page_blocks = extract_source_structure_blocks_by_page(source_native_kb)
    if not page_blocks:
        return {"textBlocks": 0, "visualBlocks": 0}

    text_updated = 0
    visual_updated = 0

    for entry in entries:
        if not isinstance(entry, dict): continue
        entry_page = entry_page_hint(entry)
        if not isinstance(entry_page, int) or entry_page <= 0: continue
        source_blocks = page_blocks.get(int(entry_page)) or []
        if not source_blocks: continue

        blocks = entry.get("blocks")
        if not isinstance(blocks, list): continue

        unused_text_indices = [
            idx for idx, b in enumerate(source_blocks)
            if b.get("type") in {"text", "code", "paragraph"}
        ]

        for block in blocks:
            if not isinstance(block, dict): continue
            block_type = safe_str(block.get("type")).strip().lower()
            
            # 1. Visual block alignment (Images, Tables)
            if block_type in {"image", "figure", "equation", "table"}:
                # Skip if already has a valid bbox
                existing_bbox = block.get("bbox") or block.get("box")
                if isinstance(existing_bbox, dict): continue
                
                uri = safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
                if not uri: continue
                block_base = basename_from_uri(uri)
                
                for source_block in source_blocks:
                    if basename_from_uri(safe_str(source_block.get("imageUri"))) == block_base:
                        bbox = source_block.get("bbox")
                        if isinstance(bbox, dict):
                            block["bbox"] = dict(bbox)
                            block["readingOrder"] = int(source_block.get("readingOrder") or 0)
                            if source_block.get("semanticRole") and not block.get("semanticRole"):
                                block["semanticRole"] = source_block["semanticRole"]
                            visual_updated += 1
                        break
                continue

            # 2. Text block alignment
            if block_type not in {"code", "text", "paragraph"}: continue
            if isinstance(block.get("bbox") or block.get("box"), dict): continue
            
            target_text = get_block_text_payload(block)
            if not target_text: continue

            best_idx: Optional[int] = None
            best_score = 0
            for source_idx in unused_text_indices:
                source_block = source_blocks[source_idx]
                score = score_structure_block_match(target_text, source_block)
                if score > best_score:
                    best_score = score
                    best_idx = source_idx

            if best_idx is not None and best_score >= 180:
                source_block = source_blocks[best_idx]
                bbox = source_block.get("bbox")
                if isinstance(bbox, dict):
                    block["bbox"] = dict(bbox)
                    block["readingOrder"] = int(source_block.get("readingOrder") or 0)
                    block["structureSource"] = "pdf_native_text_box_overlay"
                    if source_block.get("semanticRole") and not block.get("semanticRole"):
                        block["semanticRole"] = source_block["semanticRole"]
                    text_updated += 1
                    unused_text_indices.remove(best_idx)

    return {"textBlocks": int(text_updated), "visualBlocks": int(visual_updated)}
