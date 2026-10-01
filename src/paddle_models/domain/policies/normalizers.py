import re
from typing import List, Optional

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not text:
        return ""
    # Remove redundant whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_figure_scope_title(text: str) -> bool:
    """
    Detects if text looks like a figure or table title.
    Based on _is_figure_scope_title_fragment in legacy code.
    """
    raw = text.strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact or len(compact) > 32:
        return False
        
    # Rule: matches "图 X-X-X" or "表 X-X-X"
    if re.match(r"^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+", compact):
        return True
        
    # Standard technical labels
    if any(token in compact for token in ("中心线", "尺寸", "侧视图", "平面图")):
        return True
        
    return False

def extract_item_number(text: str) -> Optional[int]:
    """
    Extracts leading item number (e.g. from "第 1 条").
    """
    match = re.search(r"^第\s*(\d+)\s*条", text)
    if match:
        return int(match.group(1))
    return None

def is_structural_title(text: str) -> bool:
    """
    Detects if text looks like a structural title (Chapter, Section, etc.).
    """
    raw = text.strip()
    # matches "第一章 ...", "第一节 ...", "1.1 ...", etc.
    if re.match(r"^(第[一二三四五六七八九十]+[章节]|([0-9]+\.)+[0-9]+)\s+", raw):
        return True
    return False

def strip_artifact_tails(text: str) -> str:
    """
    Removes common page artifacts (e.g. "- 1 -") or trailing noise from blocks.
    Based on _strip_mixed_body_figure_artifact_tail in legacy code.
    """
    if not text:
        return ""
    
    # 1. Strip page numbers like "- 1 -"
    text = re.sub(r"\s*-\s*\d+\s*-\s*$", "", text)
    
    # 2. Strip trailing dots/noise often found in OCR
    text = text.rstrip(".·• ")
    
    return text

def is_likely_artifact(text: str) -> bool:
    """
    Heuristic to detect if a block is likely a header/footer artifact.
    """
    compact = re.sub(r"\s+", "", text)
    # Very short digits or isolated single words at edges
    if re.fullmatch(r"^\d+$", compact):
        return True
    return False
