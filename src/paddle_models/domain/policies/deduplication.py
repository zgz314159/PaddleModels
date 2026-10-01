from typing import Dict, Any, List, Set, Tuple
from paddle_models.domain.models import Document, Block
import os

class ImageDeduplicationPolicy:
    """
    Identifies and removes duplicate image blocks across a document.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    def apply(self, document: Document) -> Dict[str, int]:
        seen_by_key: Dict[str, str] = {} # key -> first_block_id
        removed_count = 0
        duplicate_keys = set()

        for page in document.pages:
            kept_blocks = []
            for block in page.blocks:
                if block.type not in ("image", "figure", "table"):
                    kept_blocks.append(block)
                    continue

                uri = block.metadata.get("image_uri")
                if not uri:
                    kept_blocks.append(block)
                    continue

                key = self._canonical_image_key(uri)
                if not key:
                    kept_blocks.append(block)
                    continue

                if key in seen_by_key:
                    removed_count += 1
                    duplicate_keys.add(key)
                    if self.debug:
                        print(f"[Dedupe] Duplicate image {key} found. Original: {seen_by_key[key]}, Duplicate: {block.id}")
                    continue

                seen_by_key[key] = block.id
                block.metadata["canonical_image_key"] = key
                kept_blocks.append(block)
            
            page.blocks = kept_blocks

        return {
            "duplicateImageBlocksRemoved": removed_count,
            "duplicateImageKeys": len(duplicate_keys)
        }

    def _canonical_image_key(self, uri: str) -> str:
        # Extract filename from URI
        s = uri.split("?")[0].split("#")[0]
        return os.path.basename(s).strip().lower()
