from typing import Dict, Any, Optional
import fitz
from paddle_models.infrastructure.pdf.probe import probe_page

class PageRouter:
    def __init__(self, force_method: Optional[str] = None):
        self.force_method = force_method

    def route(self, page: fitz.Page) -> str:
        """
        Determines the extraction method for a given page.
        Returns "native", "ocr", or "hybrid".
        """
        if self.force_method:
            return self.force_method

        info = probe_page(page)
        
        # Heuristics:
        # 1. If has significant native text, use native.
        if info["has_native_text"] and info["text_length"] > 100:
            return "native"
        
        # 2. If no text but has images, definitely OCR.
        if info["text_length"] < 20 and info["image_count"] > 0:
            return "ocr"
            
        # 3. Otherwise hybrid or default to ocr if no text at all.
        if info["text_length"] == 0:
            return "ocr"
            
        return "hybrid"

    def get_page_info(self, page: fitz.Page) -> Dict[str, Any]:
        return probe_page(page)
