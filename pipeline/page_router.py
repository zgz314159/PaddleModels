from typing import List, Dict, Any, Optional
from models.run_context import RunContext
from pipeline.extraction_adapters.native_adapter import NativeAdapter
from pipeline.extraction_adapters.ocr_adapter import OCRAdapter

class PageRouter:
    """Routes PDF pages to the most appropriate extraction strategy."""
    
    def __init__(self, context: RunContext):
        self.context = context
        self.native_adapter = NativeAdapter(context)
        self.ocr_adapter = OCRAdapter(context)
        
    def route_page(self, page_number: int) -> str:
        """Determines the best strategy for a given page."""
        # Check native text coverage for this specific page
        # In a real scenario, we might use the probe_coverage from native_adapter 
        # or a per-page check.
        
        # For now, a simple heuristic:
        # If native extraction yields many blocks, use native.
        # Otherwise, consider OCR or Hybrid.
        return "hybrid" # Default to hybrid to get the best of both

    def close(self):
        self.native_adapter.close()
