from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class BBox:
    x: float
    y: float
    w: float
    h: float

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.w, self.h]

@dataclass
class Block:
    id: str
    type: str # text, title, table, image, etc.
    text: str
    bbox: BBox
    page_number: int
    reading_order: int
    confidence: float = 1.0
    source: str = "native" # native, ocr, ai
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TableCell:
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: Optional[BBox] = None

@dataclass
class Table:
    id: str
    rows: int
    cols: int
    cells: List[TableCell]
    bbox: BBox
    page_number: int
    caption: Optional[str] = None
    image_uri: Optional[str] = None

@dataclass
class Page:
    page_number: int
    width: float
    height: float
    blocks: List[Block] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "native" # native, ocr, hybrid
    rotation: int = 0
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

@dataclass
class LayoutBlock:
    type: str # table, figure, text, title, etc.
    bbox: BBox
    confidence: float
    source: str # yolo, paddle, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LayoutResult:
    blocks: List[LayoutBlock] = field(default_factory=list)
    page_number: int = 0
    engine_name: str = ""
    engine_version: str = ""

@dataclass
class Document:
    document_id: str
    schema_version: str = "2.0"
    pages: List[Page] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sha256: Optional[str] = None
