import re
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from paddle_models.domain.models import Document, Block, Page

class FigureCanonicalizationPolicy:
    """
    Groups visual blocks (figures/tables) across pages and entries.
    Resolves captions and creates unified figure nodes.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    def apply(self, document: Document):
        # 1. Identify all visual blocks
        visual_blocks = []
        for page in document.pages:
            for block in page.blocks:
                if block.type in ("figure", "table"):
                    visual_blocks.append(block)

        # 2. Extract labels and group (simplified version)
        # In a full implementation, we would look for "图 X-Y" in the text or surrounding text.
        for block in visual_blocks:
            label = self._extract_label(block.text)
            if label:
                block.metadata["label"] = label
                block.metadata["label_normalized"] = self._normalize_label(label)

        # 3. Resolve Captions (if block text is empty, look at neighbor blocks)
        # This is already partially handled in build_document.py via ProximityMergingPolicy.

    def _extract_label(self, text: str) -> Optional[str]:
        if not text: return None
        # Matches "图 1-1" or "表 2"
        match = re.search(r"((?:图|表)\s*\d+[-.\d]*)", text)
        if match:
            return match.group(1)
        return None

    def _normalize_label(self, label: str) -> str:
        return re.sub(r"\s+", "", label).replace(" ", "").lower()
