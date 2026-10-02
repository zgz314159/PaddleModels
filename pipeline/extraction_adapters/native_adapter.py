try:
    import fitz
except ImportError:
    fitz = None
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from pipeline.canonical_ir import DocPage, DocBlock, BBox
from models.run_context import RunContext
from pipeline.semantics.role_inference import (
    infer_text_structure_semantic_role,
    is_positional_page_marker,
)
from imaging.watermark_utils import (
    collect_repeated_watermark_candidates,
    looks_like_repeated_watermark_text
)

class NativeAdapter:
    """Adapter for high-precision native PDF text extraction."""
    def __init__(self, context: RunContext):
        if fitz is None:
            raise ImportError("PyMuPDF (fitz) is required for NativeAdapter: pip install PyMuPDF")
        self.context = context
        self.doc = None
        self._watermarks: Set[str] = set()
        self._dynamic_artifacts: Set[str] = set()
        self._initialized = False

    def _ensure_doc(self):
        """Open the PDF handle lazily without running full-document scans."""
        if not self.doc:
            self.doc = fitz.open(self.context.input_path)
        return self.doc

    def _ensure_initialized(self):
        if self._initialized:
            return
        self.doc = self._ensure_doc()
        
        # Collect artifacts (watermarks and dynamic headers/footers)
        self._watermarks = collect_repeated_watermark_candidates(self.doc)
        self._dynamic_artifacts = self._collect_dynamic_page_artifact_signatures()
        self._initialized = True

    def _collect_dynamic_page_artifact_signatures(self) -> Set[str]:
        """Detect page-level artifacts like running page numbers."""
        page_hits: Dict[str, Set[int]] = {}
        page_count = self.doc.page_count
        if page_count <= 2:
            return set()

        threshold = max(3, min(40, (page_count + 4) // 5))

        for page_index in range(page_count):
            try:
                page = self.doc.load_page(page_index)
                page_height = float(page.rect.height)
                data = page.get_text('dict') or {}
            except Exception:
                continue
            
            top_zone_limit = page_height * 0.06
            bottom_zone_limit = page_height * 0.94
            blocks = data.get('blocks', [])
            for blk in blocks:
                if blk.get('type') != 0: continue # 0 is text
                bbox = blk.get('bbox')
                y0, y1 = bbox[1], bbox[3]
                
                zone = None
                if y1 <= top_zone_limit: zone = 'top'
                elif y0 >= bottom_zone_limit: zone = 'bottom'
                
                if not zone: continue
                
                text = "".join(["".join([s['text'] for s in l['spans']]) for l in blk.get('lines', [])]).strip()
                compact = re.sub(r'\s+', '', text)
                if not compact or len(compact) > 40: continue
                
                normalized = re.sub(r'\d+', '#', compact)
                signature = f'{zone}|{normalized}'
                page_hits.setdefault(signature, set()).add(page_index + 1)

        return {sig for sig, pages in page_hits.items() if len(pages) >= threshold}

    def _is_artifact(
        self,
        text: str,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float,
    ) -> bool:
        compact = re.sub(r'\s+', '', text)
        if not compact: return True

        # Check watermarks
        if looks_like_repeated_watermark_text(text, self._watermarks):
            return True

        # Positional page markers (edge pure digits / decorated page numbers)
        if is_positional_page_marker(text, bbox, page_width, page_height):
            return True

        # Check dynamic artifacts
        y0, y1 = bbox[1], bbox[3]
        zone = None
        if y1 <= page_height * 0.06: zone = 'top'
        elif y0 >= page_height * 0.94: zone = 'bottom'

        if zone:
            normalized = re.sub(r'\d+', '#', compact)
            if f'{zone}|{normalized}' in self._dynamic_artifacts:
                return True

        return False

    def extract_page(self, page_number: int) -> DocPage:
        """Extracts native text and image blocks from a single page."""
        self._ensure_initialized()
        page_idx = page_number - 1
        try:
            page = self.doc.load_page(page_idx)
            page_rect = page.rect
            width, height = page_rect.width, page_rect.height
            
            doc_page = DocPage(
                page_number=page_number,
                width=width,
                height=height,
                method="native"
            )
            
            # Text extraction
            data = page.get_text("dict")
            blocks = data.get("blocks", [])
            
            reading_order = 0
            for blk in blocks:
                if blk.get("type") == 0: # Text
                    bbox_raw = blk.get("bbox") # x0, y0, x1, y1
                    lines = blk.get("lines", [])
                    if not lines: continue

                    line_texts = []
                    total_len = 0
                    is_bold = False
                    font_sizes = []

                    for line in lines:
                        span_texts = []
                        for s in line.get("spans", []):
                            txt = s.get("text", "")
                            span_texts.append(txt)
                            if "bold" in s.get("font", "").lower():
                                is_bold = True
                            font_sizes.append(s.get("size", 0))
                        
                        l_text = "".join(span_texts)
                        line_texts.append(l_text)
                        total_len += len(l_text.strip())
                    
                    full_text = " ".join(line_texts).strip()
                    if not full_text: continue

                    if self._is_artifact(full_text, bbox_raw, width, height):
                        continue

                    avg_line_len = total_len / len(lines)

                    # Convert bbox to x, y, w, h
                    x, y, x1, y1 = bbox_raw
                    w, h = x1 - x, y1 - y

                    # Determine role using v2 path (role_inference + page_context)
                    tb_info = {
                        "text": full_text,
                        "bbox": (x, y, w, h),
                        "line_count": len(lines),
                        "avg_line_len": avg_line_len,
                        "is_bold": is_bold,
                        "font_size": max(font_sizes) if font_sizes else 0,
                    }
                    page_context = {
                        "page_number": page_number,
                        "page_width": width,
                        "page_height": height,
                    }
                    role = infer_text_structure_semantic_role(tb_info, page_context)

                    reading_order += 1

                    doc_block = DocBlock(
                        id=f"p{page_number}_b{reading_order}",
                        type="text" if role not in ("heading", "caption") else role,
                        text=full_text,
                        bbox=BBox(x, y, w, h),
                        page_number=page_number,
                        reading_order=reading_order,
                        source="native",
                        metadata={
                            "semanticRole": role,
                            "isBold": is_bold,
                            "fontSize": tb_info["font_size"]
                        }
                    )
                    doc_page.blocks.append(doc_block)

            # Image/figure assets are owned by pipeline/extraction_adapters/figure_adapter.py
            # Table detection is owned by pipeline/extraction_adapters/table_adapter.py

            return doc_page
        except Exception as e:
            print(f"Error extracting page {page_number}: {e}")
            return DocPage(page_number=page_number, width=0, height=0, method="error")

    def probe_page_native_chars(self, page_number: int) -> int:
        """Cheap, read-only per-page native-text probe used for routing only.

        Counts non-whitespace characters returned by a plain text read of a
        single page. It deliberately does NOT run the artifact scan, full
        block extraction, rendering, OCR, or any model inference, so routing
        never repeats expensive per-page work.

        Raises ValueError for out-of-range pages; underlying fitz errors from
        loading the page / reading its text propagate to the caller.
        """
        doc = self._ensure_doc()
        page_count = doc.page_count
        if page_number < 1 or page_number > page_count:
            raise ValueError(
                f"page {page_number} out of range (1..{page_count})"
            )
        page = doc.load_page(page_number - 1)
        text = page.get_text("text") or ""
        return len(re.sub(r"\s+", "", text))

    def probe_coverage(self) -> float:
        """Returns the ratio of pages with native text."""
        if not self.doc:
            self.doc = fitz.open(self.context.input_path)
        
        text_pages = 0
        total_pages = self.doc.page_count
        for page in self.doc:
            if page.get_text().strip():
                text_pages += 1
        return text_pages / total_pages if total_pages > 0 else 0

    def render_page(self, page_number: int, dpi: int = 72) -> bytes:
        """Renders a page to PNG bytes."""
        self._ensure_initialized()
        page_idx = page_number - 1
        page = self.doc.load_page(page_idx)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")

    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None
