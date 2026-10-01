import re
from typing import List, Optional
from paddle_models.domain.models import Block, LayoutBlock

class MergingPolicy:
    def should_match(self, text_block: Block, layout_block: LayoutBlock) -> bool:
        """
        Determines if a text block (e.g. caption) matches a layout block (e.g. figure).
        """
        raise NotImplementedError

class ProximityMergingPolicy(MergingPolicy):
    def __init__(self, vertical_threshold: float = 50.0):
        self.vertical_threshold = vertical_threshold

    def should_match(self, text_block: Block, layout_block: LayoutBlock) -> bool:
        # Same page check
        if text_block.page_number != layout_block.metadata.get('page_number', 0):
            # If layout_block doesn't have page_number in metadata, it might be in context
            pass

        # Vertical proximity: text is usually below the figure (caption) or above (table title)
        text_bbox = text_block.bbox
        layout_bbox = layout_block.bbox

        # Check if text is close to layout block vertically
        # Simplified: check distance between closest edges
        dist_top = abs(text_bbox.y - (layout_bbox.y + layout_bbox.h))
        dist_bottom = abs(layout_bbox.y - (text_bbox.y + text_bbox.h))

        return min(dist_top, dist_bottom) < self.vertical_threshold

def is_figure_caption(text: str) -> bool:
    """
    Heuristic: does the text look like a figure caption (e.g. '图 1-1-1 ...')?
    """
    clean_text = text.strip()
    return bool(re.match(r"^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+", clean_text))
