from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float
    
    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

@dataclass
class DocBlock:
    id: str
    type: str  # "text", "table", "image", "heading", etc.
    text: str
    bbox: BBox
    page_number: int
    reading_order: int
    confidence: float = 1.0
    source: str = "native"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "page_number": self.page_number,
            "reading_order": self.reading_order,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata
        }

@dataclass
class DocPage:
    page_number: int
    width: float
    height: float
    blocks: List[DocBlock] = field(default_factory=list)
    rotation: int = 0
    method: str = "native" # "native", "ocr", "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "blocks": [b.to_dict() for b in self.blocks],
            "rotation": self.rotation,
            "method": self.method
        }

@dataclass
class CanonicalIR:
    document_id: str
    schema_version: str = "2.0"
    pages: List[DocPage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sha256: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "schema_version": self.schema_version,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata,
            "sha256": self.sha256
        }
