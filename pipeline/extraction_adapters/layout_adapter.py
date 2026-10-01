import io
from typing import List, Dict, Any, Optional
from pipeline.canonical_ir import DocBlock, BBox
from models.run_context import RunContext
from imaging.layout_engine import detect_layout_blocks

class LayoutAdapter:
    """Adapter for layout analysis (tables, figures, equations)."""
    def __init__(self, context: RunContext):
        self.context = context
        # In a full implementation, we would load YOLO models here via BuildProfile
        self.yolo_detector = None 
        
    def detect_blocks(self, image_bytes: bytes, page_number: int, native_blocks: Optional[List] = None) -> List:
        """Detects visual blocks like tables and figures from page image."""
        # Map DocBlocks to Dict for layout_engine
        text_boxes = []
        if native_blocks:
            for nb in native_blocks:
                text_boxes.append({
                    'bbox': (nb.bbox.x, nb.bbox.y, nb.bbox.w, nb.bbox.h),
                    'text': nb.text,
                    'isBold': nb.metadata.get('isBold', False)
                })

        # In shadow mode, we might not have a loaded YOLO model, 
        # so detect_layout_blocks will fallback to CV detection.
        raw_blocks = detect_layout_blocks(None, image_bytes, page_text_boxes=text_boxes)
        
        doc_blocks = []
        for idx, rb in enumerate(raw_blocks):
            x, y, w, h = rb['bbox']
            btype = rb['type']
            
            doc_blocks.append(DocBlock(
                id=f"p{page_number}_v{idx}",
                type=btype,
                text="", # Visual blocks often have empty text until OCR'd
                bbox=BBox(float(x), float(y), float(w), float(h)),
                page_number=page_number,
                reading_order=0, # Will be set by a sorter later
                confidence=rb.get('confidence', 0.5),
                source=rb.get('source', 'cv_fallback'),
                metadata={"visual": True}
            ))
            
        return doc_blocks
