import argparse
import sys
import os
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from models.run_context import RunContext, BuildProfile
from pipeline.extraction_adapters.layout_adapter import LayoutAdapter
from pipeline.extraction_adapters.table_adapter import (
    detect_native_table_blocks,
    merge_native_and_visual_tables,
    suppress_text_for_structured_tables,
)
from pipeline.extraction_adapters.figure_adapter import process_page_figures
from pipeline.semantics.figure_text_association import associate_ir_figures
from pipeline.semantic_projector import SemanticProjector
from pipeline.canonical_ir import CanonicalIR
from imaging.bbox_utils import bbox_overlap_ratio_xywh
from pipeline.page_router import DEFAULT_MIN_NATIVE_CHARS, PageRouter
from pipeline.page_cache import PageCache

try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    jsonschema = None
    HAS_JSONSCHEMA = False

def main():
    parser = argparse.ArgumentParser(description="PaddleModels v2 Pipeline Runner")
    parser.add_argument("input", help="Path to input PDF/DOCX")
    parser.add_argument("--mode", choices=["legacy", "v2", "shadow"], default="v2", help="Run mode")
    parser.add_argument("--output-base", default="outputs", help="Base directory for outputs")
    parser.add_argument("--pages", help="Page range to process (e.g. 1-10)", default=None)
    parser.add_argument("--strategy", choices=["heading", "page"], default="heading", help="Grouping strategy")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
        
    profile = BuildProfile(name="default", grouping_strategy=args.strategy)
    ctx = RunContext.create(str(input_path), args.output_base, profile)
    
    print(f"Starting run {ctx.run_id} in {args.mode} mode...")
    print(f"Input: {ctx.input_path}")
    print(f"SHA256: {ctx.get_input_sha256()}")
    
    kb_v2 = None
    if args.mode == "v2" or args.mode == "shadow":
        _, kb_v2 = run_v2(ctx, args.pages)
        
    if args.mode == "legacy" or args.mode == "shadow":
        run_legacy(ctx)

    if args.mode == "shadow" and kb_v2:
        run_shadow_compare(ctx, kb_v2)

def run_v2(ctx: RunContext, page_range_str: Optional[str] = None):
    print("Running v2 extraction...")
    
    # Cache and Router
    cache_dir = ctx.get_cache_dir()
    router = PageRouter(ctx)
    page_cache = PageCache(
        cache_dir,
        input_sha256=ctx.get_input_sha256(),
        profile_config=ctx.profile.page_ir_cache_config(),
        router_config={
            "min_native_chars": getattr(
                router, "min_native_chars", DEFAULT_MIN_NATIVE_CHARS
            )
        },
    )
    
    # adapters
    native = router.native_adapter
    ocr = router.ocr_adapter
    layout = LayoutAdapter(ctx)
    
    coverage = native.probe_coverage()
    print(f"Native text coverage: {coverage:.2%}")
    
    doc_ir = CanonicalIR(document_id=ctx.input_path.stem, sha256=ctx.get_input_sha256())
    
    # Asset directory
    asset_dir = ctx.output_dir / "shots"
    asset_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine page range
    import fitz
    doc = fitz.open(ctx.input_path)
    total_pages = doc.page_count
    doc.close()
    
    start_page = 1
    end_page = total_pages
    
    if page_range_str:
        try:
            parts = page_range_str.split('-')
            start_page = int(parts[0])
            if len(parts) > 1:
                end_page = int(parts[1])
        except Exception as e:
            print(f"Error parsing page range: {e}")
            
    print(f"Processing pages {start_page} to {end_page} of {total_pages}...")
    
    for p_num in range(start_page, end_page + 1):
        # Route first so native and ocr pages get distinct cache identities.
        strategy = router.route_page(p_num)
        print(f"Page {p_num}: Strategy {strategy}")

        # A hit requires schema, fingerprint, input SHA, page number and
        # strategy to all match; any mismatch or corruption is a safe miss.
        page_obj = page_cache.load(p_num, strategy)
        if page_obj is not None:
            # Cache stores IR only — rehydrate missing shot assets for this run dir.
            try:
                missing_assets = [
                    b for b in page_obj.blocks
                    if b.type in ("table", "image", "figure")
                    and isinstance(b.metadata.get("imageUri"), str)
                    and b.metadata["imageUri"].startswith("shots/")
                    and not (ctx.output_dir / b.metadata["imageUri"]).exists()
                ]
                if missing_assets:
                    dpi_c = 144
                    scale_c = 72.0 / dpi_c
                    img_b = native.render_page(p_num, dpi=dpi_c)
                    from PIL import Image as _Image
                    import io as _io
                    page_img_c = _Image.open(_io.BytesIO(img_b))
                    pw, ph = page_img_c.size
                    for b in missing_assets:
                        try:
                            x0 = max(0, int(b.bbox.x / scale_c))
                            y0 = max(0, int(b.bbox.y / scale_c))
                            x1 = min(pw, int((b.bbox.x + b.bbox.w) / scale_c))
                            y1 = min(ph, int((b.bbox.y + b.bbox.h) / scale_c))
                            if x1 <= x0 or y1 <= y0:
                                raise ValueError("invalid crop")
                            cropped = page_img_c.crop((x0, y0, x1, y1))
                            rel = b.metadata["imageUri"]
                            out_p = ctx.output_dir / rel
                            out_p.parent.mkdir(parents=True, exist_ok=True)
                            cropped.save(out_p)
                            if not out_p.exists() or out_p.stat().st_size <= 0:
                                raise ValueError("empty crop")
                            b.metadata["assetWidth"] = int(cropped.size[0])
                            b.metadata["assetHeight"] = int(cropped.size[1])
                        except Exception as ae:
                            print(f"[WARN] cache asset rehydrate failed {b.id}: {ae}")
                            b.metadata.setdefault("cropWarnings", []).append(str(ae))
            except Exception as re_hydrate_e:
                print(f"[WARN] cache asset pass failed page={p_num}: {re_hydrate_e}")

            doc_ir.pages.append(page_obj)
            print(f"Page {p_num}: Loaded from cache")
            continue

        # Native extraction always as base if strategy allows
        page_ir = native.extract_page(p_num)

        # Render for Visual Analysis (144 DPI)
        dpi = 144
        img_bytes = native.render_page(p_num, dpi=dpi)

        # Visual Detection with native block guidance
        visual_blocks = layout.detect_blocks(img_bytes, p_num, native_blocks=page_ir.blocks)

        # OCR if needed (full-page OCR is profile-gated: smoke disables it;
        # Phase 2F figure-caption OCR stays independent of this gate)
        if (
            strategy in ("ocr", "hybrid")
            and getattr(ctx.profile, "ocr_enabled", True)
            and not any(b.type != "image" for b in page_ir.blocks)
        ):
            ocr_page = ocr.extract_page(p_num, img_bytes)
            page_ir.blocks.extend(ocr_page.blocks)
            page_ir.method = "hybrid" if page_ir.blocks else "ocr"

        # Coordinate mapping (Visual Pixels -> Native Points)
        scale = 72.0 / dpi
        for vb in visual_blocks:
            vb.bbox.x *= scale
            vb.bbox.y *= scale
            vb.bbox.w *= scale
            vb.bbox.h *= scale

        # Native structured tables (PyMuPDF) + merge with CV visual tables
        native_tables: List = []
        try:
            fitz_doc_page = native.doc.load_page(p_num - 1) if native.doc else None
            native_tables = detect_native_table_blocks(fitz_doc_page, p_num)
        except Exception as e:
            print(f"Page {p_num}: native table detect failed: {e}")
            native_tables = []

        visual_tables = [b for b in visual_blocks if b.type == "table"]
        table_blocks = merge_native_and_visual_tables(native_tables, visual_tables)
        for tb in table_blocks:
            tb.metadata["_bbox_xywh"] = (tb.bbox.x, tb.bbox.y, tb.bbox.w, tb.bbox.h)

        # Suppress native text ONLY inside structured tables (image_only keeps text)
        non_text_native = [b for b in page_ir.blocks if b.type not in ("text", "heading", "caption")]
        text_native = [b for b in page_ir.blocks if b.type in ("text", "heading", "caption")]
        kept_text = suppress_text_for_structured_tables(text_native, table_blocks)
        # Also drop text that sits inside pure visual tables when structured was merged;
        # for image_only tables keep text (already handled by suppress_text_for_structured_tables).
        # Visual non-table blocks still prune heavily-overlapped native text against CV tables
        # ONLY when those CV tables were upgraded to structured via merge (already in table_blocks).
        # Unmatched image_only CV tables: do NOT suppress (prevents content loss).

        # Merge and Prune
        final_blocks = []
        # visual non-table blocks first
        for vb in visual_blocks:
            if vb.type != "table":
                final_blocks.append(vb)
        # merged tables
        final_blocks.extend(table_blocks)
        # native non-text (images etc.)
        final_blocks.extend(non_text_native)
        # native text kept after structured-only suppression
        final_blocks.extend(kept_text)

        # De-dupe native tables already represented (native.extract_page may add find_tables)
        # Remove native table blocks from non_text_native that duplicate our table_blocks ids/bboxes.
        table_id_set = {tb.id for tb in table_blocks}
        final_blocks = [
            b for b in final_blocks
            if not (b.type == "table" and b.id not in table_id_set and b.source == "pymupdf_native")
            or b.id in table_id_set
            or b.source not in ("pymupdf_native",)
        ]
        # Simpler: keep only table_blocks for type==table
        final_blocks = [b for b in final_blocks if b.type != "table"] + table_blocks
        # Drop any residual native image placeholders — FigureAdapter owns figures.
        final_blocks = [b for b in final_blocks if b.type not in ("image", "figure")]

        # Figure assets via FigureAdapter (embedded preferred, visual crop fallback)
        try:
            fig_blocks, fig_warns, fig_stats = process_page_figures(
                page_number=p_num,
                page_width=page_ir.width,
                page_height=page_ir.height,
                doc=native.doc,
                visual_blocks=visual_blocks,
                asset_dir=asset_dir,
                page_image_bytes=img_bytes,
                scale=scale,
                dpi=dpi,
            )
            final_blocks.extend(fig_blocks)
            for w in fig_warns:
                print(f"[WARN] figure: {w}")
            print(
                f"Page {p_num}: figures embedded={fig_stats['embedded_candidates']} "
                f"visual={fig_stats['visual_candidates']} matched={fig_stats['matched']} "
                f"kept={fig_stats['embedded_kept']} cropped={fig_stats['cropped']} "
                f"unique_assets={fig_stats['unique_assets_written']} "
                f"dedup_hits={fig_stats['duplicate_asset_hits']} excluded={fig_stats['excluded']}"
            )
        except Exception as fig_e:
            print(f"[WARN] figure adapter failed page={p_num}: {fig_e}")

        # Table asset crops only (figures already handled by FigureAdapter)
        from PIL import Image
        import io
        page_img = Image.open(io.BytesIO(img_bytes))
        page_w_px, page_h_px = page_img.size

        for b in final_blocks:
            if b.type == "table":
                try:
                    x0 = max(0, int(b.bbox.x / scale))
                    y0 = max(0, int(b.bbox.y / scale))
                    x1 = min(page_w_px, int((b.bbox.x + b.bbox.w) / scale))
                    y1 = min(page_h_px, int((b.bbox.y + b.bbox.h) / scale))
                    if x1 <= x0 or y1 <= y0:
                        raise ValueError(f"invalid crop box {(x0, y0, x1, y1)}")
                    cropped = page_img.crop((x0, y0, x1, y1))
                    filename = f"{b.id}.png"
                    out_path = asset_dir / filename
                    cropped.save(out_path)
                    if not out_path.exists() or out_path.stat().st_size <= 0:
                        raise ValueError("crop file empty")
                    if cropped.size[0] <= 0 or cropped.size[1] <= 0:
                        raise ValueError("crop size zero")
                    b.metadata["imageUri"] = f"shots/{filename}"
                    b.metadata["assetWidth"] = int(cropped.size[0])
                    b.metadata["assetHeight"] = int(cropped.size[1])
                except Exception as crop_e:
                    warn = f"table crop failed page={p_num} id={b.id}: {crop_e}"
                    print(f"[WARN] {warn}")
                    b.metadata.setdefault("cropWarnings", []).append(str(crop_e))
                    b.metadata.pop("imageUri", None)

        # Re-sort by reading order (approximate by top, then left)
        # Using a small tolerance for Y to group same-line blocks
        page_ir.blocks = sorted(final_blocks, key=lambda b: (int(b.bbox.y / 4) * 4, b.bbox.x))
        for idx, b in enumerate(page_ir.blocks):
            b.reading_order = idx + 1
            
        print(f"Page {p_num}: {len(page_ir.blocks)} blocks ({len(visual_blocks)} visual)")
        
        # Save to cache (atomic, fingerprinted envelope)
        page_cache.store(p_num, strategy, page_ir)
        doc_ir.pages.append(page_ir)
    
    router.close()

    # Phase 2E/2E.1: figure ↔ native caption/legend association.
    # Correctness stage — exceptions propagate to run_v2 failure path (no silent OK).
    assoc_totals = associate_ir_figures(doc_ir)
    print(
        f"figure-text association: caption_figures={assoc_totals['figures_with_native_caption']} "
        f"legend_figures={assoc_totals['figures_with_native_legend']} "
        f"captions={assoc_totals['native_captions_linked']} "
        f"legends={assoc_totals['native_legends_linked']} "
        f"unmatched_captions={assoc_totals['unmatched_caption_candidates']}"
    )
    ctx.profile.parameters = dict(getattr(ctx.profile, "parameters", None) or {})
    ctx.profile.parameters["figureTextAssociation"] = assoc_totals

    # Phase 2F: figure-internal caption OCR (only when explicitly requested).
    # off → zero OCR calls, no extra outputs, prior hashes unchanged.
    fc_engine = getattr(ctx.profile, "figure_caption_ocr", "off") or "off"
    if fc_engine != "off":
        from imaging.caption_utils import extract_visual_caption_candidate
        from imaging.figure_caption_ocr import DEFAULT_LANGUAGE, EXTRACTOR_VERSION
        from imaging.text_utils import extract_figure_table_labels
        from pipeline.semantics.figure_caption_ocr import (
            FIGURE_CAPTION_OCR_REPORT,
            run_figure_caption_ocr_pass,
        )

        cache_root = Path(
            os.environ.get("PADDLE_CACHE_ROOT", ".cache")
        )
        fc_report = run_figure_caption_ocr_pass(
            doc_ir,
            run_dir=ctx.output_dir,
            engine=fc_engine,
            language=DEFAULT_LANGUAGE,
            cache_root=cache_root,
            label_hits_fn=extract_figure_table_labels,
            extract_candidate_fn=extract_visual_caption_candidate,
            extractor_version=EXTRACTOR_VERSION,
        )
        report_path = ctx.output_dir / FIGURE_CAPTION_OCR_REPORT
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(fc_report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        st = fc_report["stats"]
        print(
            f"figure caption OCR ({fc_engine}): attempted={st['figure_caption_ocr_attempted']} "
            f"cache_hits={st['figure_caption_ocr_cache_hits']} "
            f"linked={st['ocr_captions_linked']} "
            f"rejected={st['ocr_captions_rejected']} "
            f"ambiguous={st['ocr_captions_ambiguous']} "
            f"elapsed_ms={st['ocr_caption_elapsed_ms']}"
        )
        ctx.profile.parameters["figureCaptionOcr"] = st
        ctx.profile.parameters["figureCaptionOcrReport"] = str(report_path)

    # Phase 2G: figure label index + cross-page reference resolution.
    # Always runs (native labels work without OCR). Fail-fast — no swallow.
    from pipeline.semantics.figure_reference import (
        FIGURE_REFERENCE_REPORT,
        resolve_figure_references,
    )

    fr_report = resolve_figure_references(doc_ir)
    fr_path = ctx.output_dir / FIGURE_REFERENCE_REPORT
    with open(fr_path, "w", encoding="utf-8") as f:
        json.dump(fr_report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    fr_st = fr_report["stats"]
    print(
        f"figure references: labels={fr_st['figure_labels_indexed']} "
        f"found={fr_st['figure_references_found']} "
        f"resolved={fr_st['figure_references_resolved']} "
        f"unresolved={fr_st['figure_references_unresolved']} "
        f"ambiguous={fr_st['figure_references_ambiguous']} "
        f"referenced_figs={fr_st['referenced_figures']}"
    )
    ctx.profile.parameters = dict(getattr(ctx.profile, "parameters", None) or {})
    ctx.profile.parameters["figureReference"] = fr_st
    ctx.profile.parameters["figureReferenceReport"] = str(fr_path)

    # Semantic Projection
    is_pdf_input = ctx.input_path.suffix.lower() == ".pdf"
    projector = SemanticProjector(
        ctx.input_path.stem,
        strategy=ctx.profile.grouping_strategy,
        pdf_source_name=(ctx.input_path.name if is_pdf_input else None),
    )
    kb_final = projector.project(doc_ir)
    
    # Save outputs
    ir_output_path = ctx.output_dir / "knowledge_base.v2.json"
    kb_output_path = ctx.output_dir / "knowledge_base.json"
    with open(ir_output_path, "w", encoding="utf-8") as f:
        json.dump(doc_ir.to_dict(), f, indent=2, ensure_ascii=False)
    with open(kb_output_path, "w", encoding="utf-8") as f:
        json.dump(kb_final, f, indent=2, ensure_ascii=False)
        
    print(f"IR output saved to {ir_output_path}")
    print(f"KB output saved to {kb_output_path}")
    print("Done v2 extraction.")
    return doc_ir, kb_final

import subprocess

def run_legacy(ctx: RunContext):
    print("Running legacy pipeline...")
    legacy_script = Path(__file__).parent.parent / "tools" / "pdf_to_base64_kb.py"
    legacy_output = ctx.output_dir / "legacy"
    legacy_output.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(legacy_script),
        str(ctx.input_path),
        "--out", str(legacy_output / "knowledge_base.json"),
        "--assets-root", str(legacy_output / "assets")
    ]
    
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Done legacy pipeline.")
    except subprocess.CalledProcessError as e:
        print(f"Error running legacy pipeline: {e}")

def run_shadow_compare(ctx: RunContext, kb_v2: Dict[str, Any]):
    print("\n" + "="*60)
    print("SHADOW COMPARISON REPORT")
    print("="*60)
    
    # Try to find legacy output from this run first
    legacy_kb_path = ctx.output_dir / "legacy" / "knowledge_base.json"
    
    # Fallback to baseline for railway if it's the target
    if not legacy_kb_path.exists() and "铁路电力" in ctx.input_path.name:
        legacy_kb_path = Path(__file__).parent.parent / "outputs" / "baseline_railway" / "railway_kb.json"
    
    if legacy_kb_path.exists():
        with open(legacy_kb_path, "r", encoding="utf-8") as f:
            kb_legacy = json.load(f)
        
        print(f"Legacy Source: {legacy_kb_path}")
        
        # Metrics collection
        def get_metrics(kb):
            num_pages = len(kb["fileMetadata"].get("pageSizes", {}))
            num_entries = len(kb["entries"])
            num_tables = 0
            num_images = 0
            total_text_len = 0
            
            for entry in kb["entries"]:
                total_text_len += len(entry.get("contentMarkdown", ""))
                for block in entry.get("blocks", []):
                    btype = block["type"]
                    if btype == "table":
                        num_tables += 1
                    elif btype == "image":
                        num_images += 1
            return {
                "pages": num_pages,
                "entries": num_entries,
                "tables": num_tables,
                "images": num_images,
                "text_chars": total_text_len
            }
            
        leg_m = get_metrics(kb_legacy)
        v2_m = get_metrics(kb_v2)
        
        print(f"{'Metric':<20} | {'Legacy':<15} | {'v2':<15} | {'Diff':<10}")
        print("-" * 65)
        for k in leg_m:
            l_val = leg_m[k]
            v_val = v2_m[k]
            diff = v_val - l_val
            pct = (diff / l_val * 100) if l_val != 0 else 0
            print(f"{k:<20} | {l_val:<15} | {v_val:<15} | {diff:<10} ({pct:+.1f}%)")
            
        # Structure check
        print("\nStructure Validation:")
        schema_path = Path(__file__).parent.parent / "contracts" / "knowledge_base_schema_v2.json"
        if schema_path.exists():
            if HAS_JSONSCHEMA:
                try:
                    with open(schema_path, "r", encoding="utf-8") as f:
                        schema = json.load(f)
                    jsonschema.validate(instance=kb_v2, schema=schema)
                    print("[OK] v2 output matches schema.")
                except Exception as e:
                    print(f"[FAIL] v2 output schema violation: {e}")
            else:
                print("[SKIP] jsonschema not installed.")
        else:
            print("[WARN] Schema file not found.")
            
    else:
        print(f"No legacy KB found at {legacy_kb_path} for comparison.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
