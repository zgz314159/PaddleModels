import re
import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple

from imaging.text_utils import safe_str, normalize_anchor_key, figure_table_label_variants
from imaging.pdf_analyzer import basename_from_uri, entry_page_hint, parse_visual_snapshot_page_idx
from imaging.kb_utils import make_image_block, insert_block, remove_image_references_by_basename

def make_figure_node_id(entry: Dict[str, Any], label_normalized: str, ordinal: int) -> str:
    """Generate a unique ID for a figure node."""
    eid = safe_str(entry.get("entryId")).strip()
    seed = f"fig|{eid}|{label_normalized}|{ordinal}".encode("utf-8")
    return f"fn_{hashlib.sha1(seed).hexdigest()[:12]}"

def choose_figure_node_caption(caption_texts: List[str], label_normalized: str) -> str:
    """Pick the best caption for a figure node from a list of candidates."""
    if not caption_texts: return ""
    
    # 1. Prefer text that contains the label
    for text in caption_texts:
        if label_normalized in normalize_anchor_key(text):
            return text.strip()
            
    # 2. Prefer the longest text
    return max(caption_texts, key=len).strip()

def choose_figure_node_label(caption_texts: List[str], fallback_label: str, fallback_normalized: str) -> Tuple[str, str]:
    """Extract or pick the best label for a figure node."""
    for text in caption_texts:
        variants = figure_table_label_variants(text)
        if variants:
            # Pick first matching variant
            return variants[0], variants[0] # Simplification for now
            
    return fallback_label, fallback_normalized

def move_image_block_between_entries(
    *,
    entries: List[Dict[str, Any]],
    source_entry_idx: int,
    block_idx: int,
    target_entry_idx: int,
) -> bool:
    """Move an image block from one entry to another."""
    if source_entry_idx < 0 or source_entry_idx >= len(entries): return False
    if target_entry_idx < 0 or target_entry_idx >= len(entries): return False
    
    source_entry = entries[source_entry_idx]
    target_entry = entries[target_entry_idx]
    
    blocks = source_entry.get("blocks")
    if not isinstance(blocks, list) or block_idx < 0 or block_idx >= len(blocks):
        return False
        
    block = blocks.pop(block_idx)
    target_blocks = target_entry.get("blocks")
    if not isinstance(target_blocks, list):
        target_blocks = []
        target_entry["blocks"] = target_blocks
        
    target_blocks.append(block)
    return True

def rebuild_entry_figure_nodes(kb: Dict[str, Any], *, debug: bool = False) -> int:
    """Re-scan KB entries and rebuild their 'figureNodes' metadata based on current blocks."""
    entries = kb.get("entries")
    if not isinstance(entries, list): return 0
    
    updated = 0
    for entry in entries:
        if not isinstance(entry, dict): continue
        
        blocks = entry.get("blocks")
        if not isinstance(blocks, list): continue
        
        # Logic to group image blocks into figure nodes goes here...
        # (This is a complex function, I will port it in chunks if needed)
        # For now, let's keep it simple or port it fully if it's manageable.
        pass
        
    return updated

# (Full port of _rebuild_entry_figure_nodes and _rebind_figure_images_to_reference_entries would go here)
