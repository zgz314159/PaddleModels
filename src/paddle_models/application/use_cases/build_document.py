import fitz
import time
import hashlib
import dataclasses
from pathlib import Path
from typing import List, Optional
from paddle_models.core.context import RunContext
from paddle_models.core.cache import PageCache
from paddle_models.domain.models import Document, Page, Block, BBox, LayoutBlock
from paddle_models.infrastructure.pdf.probe import probe_page
from paddle_models.infrastructure.pdf.native_extractor import extract_native_blocks
from paddle_models.infrastructure.pdf.renderer import render_page_to_image
from paddle_models.infrastructure.pdf.ocr_adapters import PaddleOcrAdapter, TesseractOcrAdapter
from paddle_models.infrastructure.pdf.layout_adapter import YoloLayoutAdapter, PaddleLayoutAdapter
from paddle_models.infrastructure.pdf.docling_adapter import DoclingAdapter
from paddle_models.infrastructure.pdf.table_adapters import PaddleTableAdapter
from paddle_models.infrastructure.pdf.cv_adapter import CvLayoutAdapter
from paddle_models.infrastructure.pdf.asset_manager import AssetManager
from paddle_models.application.services.router import PageRouter
from paddle_models.application.services.post_processing import PostProcessor
from paddle_models.application.services.audit import DocumentAuditService
from paddle_models.domain.policies.deduplication import ImageDeduplicationPolicy
from paddle_models.utils.geometry import scale_bbox
from paddle_models.domain.policies.merging_rules import ProximityMergingPolicy, is_figure_caption
from paddle_models.domain.policies.normalizers import is_structural_title, strip_artifact_tails, is_likely_artifact
from paddle_models.application.services.incremental import IncrementalService

def get_file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def build_document_from_pdf(
    pdf_path: str,
    context: RunContext,
    pages: Optional[List[int]] = None,
    layout_engine: str = "paddle",
    layout_model_path: Optional[str] = None
) -> Document:
    pdf_path_obj = Path(pdf_path)
    doc = fitz.open(pdf_path)
    file_sha256 = get_file_sha256(pdf_path)
    document = Document(document_id=context.file_id, sha256=file_sha256)

    cache = PageCache(context.output_root / "cache")
    incremental_service = IncrementalService(cache)
    
    # Pre-initialize adapters to reuse them across pages
    ocr_adapter = PaddleOcrAdapter(use_gpu=context.capability.has_gpu, lang=context.profile.parameters.get("ocr_lang", "ch"))
    table_adapter = PaddleTableAdapter(use_gpu=context.capability.has_gpu)
    page_router = PageRouter()
    asset_manager = AssetManager(context.assets_kb_root / context.file_id)
    merging_policy = ProximityMergingPolicy(vertical_threshold=60.0)

    layout_adapter = None
    docling_layout_map = None
    
    if layout_engine == "yolo" and layout_model_path:
        layout_adapter = YoloLayoutAdapter(layout_model_path)
    elif layout_engine == "paddle":
        layout_adapter = PaddleLayoutAdapter()
    elif layout_engine == "cv":
        layout_adapter = CvLayoutAdapter()
    elif layout_engine == "docling":
        print("[*] Running Docling full-document analysis...")
        docling_adapter = DoclingAdapter()
        docling_layout_map = docling_adapter.convert_document(pdf_path_obj)

    page_indices = pages if pages else range(len(doc))

    for i in page_indices:
        if i >= len(doc): continue
        page_start_time = time.time()
        page = doc[i]

        # Phase 1: Probe & Route
        probe_start = time.time()
        info = page_router.get_page_info(page)
        method = page_router.route(page)
        probe_time = (time.time() - probe_start) * 1000

        # Check Cache (Incremental)
        cached_data = incremental_service.get_cached_page_data(file_sha256, info["page_number"], method)
        if cached_data:
            print(f"[*] Page {info['page_number']}: Skip (Cache hit)")
            # Reconstruct from cache
            blocks = [Block(**b) for b in cached_data["blocks"]]
            for b in blocks:
                if isinstance(b.bbox, dict): b.bbox = BBox(**b.bbox)
            
            ir_page = Page(
                page_number=info["page_number"],
                width=info["width"],
                height=info["height"],
                blocks=blocks,
                method=method,
                rotation=info["rotation"]
            )
            document.pages.append(ir_page)
            continue

        print(f"[*] Page {info['page_number']}: Extracting ({method})...")
        
        # Render page
        render_start = time.time()
        dpi = 300
        page_image = render_page_to_image(page, dpi=dpi)
        img_h, img_w = page_image.shape[:2]
        render_time = (time.time() - render_start) * 1000

        # Scale factor from pixel to point (72 DPI)
        point_scale_x = page.rect.width / img_w
        point_scale_y = page.rect.height / img_h

        # Phase 2: Extract Text
        text_start = time.time()
        if method == "native":
            blocks = extract_native_blocks(page)
        else:
            blocks = ocr_adapter.extract_blocks(page_image, info["page_number"])
            for b in blocks:
                b.bbox = scale_bbox(b.bbox, point_scale_x, point_scale_y)
        text_time = (time.time() - text_start) * 1000

        # Basic Title Detection
        for b in blocks:
            if is_structural_title(b.text):
                b.type = "title"

        # Phase 4: Sanitation
        sanitation_start = time.time()
        final_blocks = []
        for b in blocks:
            if is_likely_artifact(b.text): continue
            b.text = strip_artifact_tails(b.text)
            if b.text.strip(): final_blocks.append(b)
        blocks = final_blocks
        sanitation_time = (time.time() - sanitation_start) * 1000

        # Phase 3: Layout & Assets
        layout_start = time.time()
        visual_blocks = []
        
        # Determine layout results for this page
        current_layout_blocks = []
        if layout_adapter:
            layout_res = layout_adapter.detect(page_image, info["page_number"])
            current_layout_blocks = layout_res.blocks
        elif docling_layout_map and info["page_number"] in docling_layout_map:
            current_layout_blocks = docling_layout_map[info["page_number"]]
            
        for lb_idx, lb in enumerate(current_layout_blocks):
            if lb.type in ('table', 'figure', 'image', 'picture'):
                # Handle potential mapping issues for Docling point coordinates vs our 300 DPI crop
                # For now assume Docling points (72 DPI) and we need to crop from 300 DPI page_image
                crop_bbox = lb.bbox
                if lb.source == "docling":
                    # Scale point to pixel (300 DPI)
                    crop_bbox = scale_bbox(lb.bbox, 300/72, 300/72)
                    
                uri = asset_manager.crop_and_save(page_image, crop_bbox, f"p{info['page_number']}_v{lb_idx}")
                
                # IR block should always be in points (72 DPI)
                scaled_bbox = lb.bbox
                if lb.source != "docling": # if it was from pixels (Paddle/Yolo/CV)
                    scaled_bbox = scale_bbox(lb.bbox, point_scale_x, point_scale_y)
                
                vb = Block(
                    id=f"p{info['page_number']}_v{lb_idx}",
                    type="table" if lb.type == "table" else "figure",
                    text=lb.metadata.get("text", ""), 
                    bbox=scaled_bbox,
                    page_number=info["page_number"],
                    reading_order=1000 + lb_idx,
                    source=lb.source,
                    metadata={"image_uri": uri, "is_visual": True}
                )

                if lb.type == 'table':
                    x, y, w, h = int(lb.bbox.x), int(lb.bbox.y), int(lb.bbox.w), int(lb.bbox.h)
                    table_img = page_image[y:y+h, x:x+w]
                    table_struct = table_adapter.recognize_table(table_img, info["page_number"], vb.id)
                    if table_struct:
                        vb.metadata.update({
                            "table_rows": table_struct.rows,
                            "table_cols": table_struct.cols,
                            "table_cells": [dataclasses.asdict(c) for c in table_struct.cells]
                        })
                visual_blocks.append(vb)
        
        # Merging
        for vb in visual_blocks:
            for tb in blocks:
                if tb.type == "text" and is_figure_caption(tb.text):
                    vb.metadata['page_number'] = info["page_number"]
                    lb_proxy = LayoutBlock(type=vb.type, bbox=vb.bbox, confidence=vb.confidence, source=vb.source, metadata=vb.metadata)
                    if merging_policy.should_match(tb, lb_proxy):
                        vb.text = tb.text
                        break
        
        blocks.extend(visual_blocks)
        layout_time = (time.time() - layout_start) * 1000

        ir_page = Page(
            page_number=info["page_number"],
            width=info["width"],
            height=info["height"],
            blocks=blocks,
            method=method,
            rotation=info["rotation"],
            processing_time_ms=(time.time() - page_start_time) * 1000
        )
        # Add detailed timings to metadata
        ir_page.warnings.append(f"Timings(ms): probe={probe_time:.1f}, render={render_time:.1f}, text={text_time:.1f}, sanitation={sanitation_time:.1f}, layout={layout_time:.1f}")
        
        document.pages.append(ir_page)

        # Set cache
        cache.set(file_sha256, info["page_number"], method, {
            "blocks": [dataclasses.asdict(b) for b in ir_page.blocks]
        })

    doc.close()

    # Phase 5: Post-Processing & Policies
    post_processor = PostProcessor()
    document = post_processor.process(document)

    dedupe_policy = ImageDeduplicationPolicy(debug=context.debug)
    dedupe_stats = dedupe_policy.apply(document)
    document.metadata["dedupe_stats"] = dedupe_stats

    # Phase 6: Audit
    audit_service = DocumentAuditService()
    document.metadata["audit"] = audit_service.compute_audit(document)

    return document
