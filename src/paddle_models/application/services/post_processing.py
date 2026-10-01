import re
from typing import List, Dict, Any
from paddle_models.domain.models import Document, Page, Block

class PostProcessor:
    """
    Applies product-level rules and cleanup to the Document IR.
    """
    def __init__(self):
        pass

    def process(self, document: Document) -> Document:
        for page in document.pages:
            self._merge_related_blocks(page)
            self._cleanup_text(page)
        return document

    def _merge_related_blocks(self, page: Page):
        """
        Merge blocks that are very close vertically and likely part of the same paragraph.
        """
        if not page.blocks:
            return

        # Sort by reading order then Y coordinate
        sorted_blocks = sorted(page.blocks, key=lambda b: (b.reading_order, b.bbox.y))
        
        merged = []
        if not sorted_blocks: return
        
        current = sorted_blocks[0]
        for next_b in sorted_blocks[1:]:
            # Simple heuristic: if same type, same source, and very close vertically
            dist = next_b.bbox.y - (current.bbox.y + current.bbox.h)
            
            # If distance is small (e.g. < 10 points) and they are horizontally aligned
            is_same_column = abs(next_b.bbox.x - current.bbox.x) < 20
            
            if (current.type == next_b.type == "text" and 
                current.source == next_b.source and 
                dist < 10 and is_same_column):
                
                # Merge text
                current.text += " " + next_b.text
                # Merge bbox
                x = min(current.bbox.x, next_b.bbox.x)
                y = min(current.bbox.y, next_b.bbox.y)
                w = max(current.bbox.x + current.bbox.w, next_b.bbox.x + next_b.bbox.w) - x
                h = max(current.bbox.y + current.bbox.h, next_b.bbox.y + next_b.bbox.h) - y
                current.bbox.x, current.bbox.y, current.bbox.w, current.bbox.h = x, y, w, h
                
                # Update confidence (average)
                current.confidence = (current.confidence + next_b.confidence) / 2
            else:
                merged.append(current)
                current = next_b
        
        merged.append(current)
        page.blocks = merged

    def _cleanup_text(self, page: Page):
        for block in page.blocks:
            # Remove common OCR artifacts or noise
            block.text = self._clean_string(block.text)

    def _clean_string(self, text: str) -> str:
        if not text: return ""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove trailing/leading garbage
        text = text.strip()
        return text
