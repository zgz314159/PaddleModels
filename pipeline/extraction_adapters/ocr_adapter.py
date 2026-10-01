from typing import List, Dict, Any, Optional
from pipeline.canonical_ir import DocPage, DocBlock, BBox
from models.run_context import RunContext
from imaging.ocr_engine import ocr_image_bytes_with_structure

class OCRAdapter:
    """Adapter for OCR-based text extraction."""
    def __init__(self, context: RunContext):
        self.context = context
        
    def extract_page(self, page_number: int, image_bytes: bytes) -> DocPage:
        """Extracts text blocks from a single page using OCR."""
        text, raw_blocks = ocr_image_bytes_with_structure(image_bytes)
        
        # We need to know the image size to set DocPage width/height
        # In a real scenario, this would be passed in or calculated
        doc_page = DocPage(page_number=page_number, width=0, height=0, method="ocr")
        
        for idx, rb in enumerate(raw_blocks):
            x, y, w, h = rb['bbox']
            doc_page.blocks.append(DocBlock(
                id=f"p{page_number}_ocr{idx}",
                type="text",
                text=rb['text'],
                bbox=BBox(float(x), float(y), float(w), float(h)),
                page_number=page_number,
                reading_order=rb.get('readingOrder', idx),
                source="ocr",
                confidence=rb.get('confidence', 0.8)
            ))
        return doc_page
