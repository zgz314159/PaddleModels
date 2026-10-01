from typing import Dict, Any
from paddle_models.domain.models import Document

class DocumentAuditService:
    """
    Computes quality metrics and semantic audit for a Document.
    """
    def compute_audit(self, document: Document) -> Dict[str, Any]:
        metrics = {
            "page_count": len(document.pages),
            "block_count": 0,
            "table_count": 0,
            "image_count": 0,
            "native_text_pages": 0,
            "ocr_pages": 0,
        }

        for page in document.pages:
            metrics["block_count"] += len(page.blocks)
            metrics["table_count"] += len([b for b in page.blocks if b.type == "table"])
            metrics["image_count"] += len([b for b in page.blocks if b.type in ("image", "figure")])
            
            if page.method == "native":
                metrics["native_text_pages"] += 1
            elif page.method == "ocr":
                metrics["ocr_pages"] += 1

        return metrics
