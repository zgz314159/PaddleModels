import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from paddle_models.domain.models import LayoutBlock, LayoutResult, BBox

class DoclingAdapter:
    def __init__(self):
        self.converter = None

    def _init_converter(self):
        if self.converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.document_converter import DocumentConverter
            # Docling handles the whole document conversion
            self.converter = DocumentConverter()

    def convert_document(self, pdf_path: Path) -> Dict[int, List[LayoutBlock]]:
        """
        Converts the entire document and returns a map of page number to layout blocks.
        """
        self._init_converter()
        result = self.converter.convert(pdf_path)
        
        pages_layout = {}
        
        # docling result has a 'document' property (DoclingDocument)
        doc = result.document
        
        # Iterate over elements (blocks)
        for element, context in doc.iterate_items():
            # context.page is 1-based page number
            page_no = context.page_no
            if page_no not in pages_layout:
                pages_layout[page_no] = []
                
            # Get bbox if available
            # Note: Docling coordinates might need conversion to match our IR
            bbox = None
            if hasattr(element, "prov") and element.prov:
                prov = element.prov[0] # Take first provenance
                # prov.bbox is usually [x1, y1, x2, y2] in points
                b = prov.bbox
                bbox = BBox(x=b.l, y=b.t, w=b.r - b.l, h=b.b - b.t)
            
            # Map type
            label = "text"
            if hasattr(element, "label"):
                label = element.label.name.lower()
            
            pages_layout[page_no].append(LayoutBlock(
                type=label,
                bbox=bbox if bbox else BBox(0,0,0,0),
                confidence=1.0,
                source="docling",
                metadata={"text": element.text if hasattr(element, "text") else ""}
            ))
            
        return pages_layout
