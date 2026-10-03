# SCRIPT_FUNCTION_MAP

Mapping of core functions to their implementation files.

| Function | File | Node Type | Description |
|---|---|---|---|
| `_get_shared_paddle_ocr` | [pdf_to_base64_kb.py](file:///C:/Users/zgz31/Desktop/PaddleModels/tools/pdf_to_base64_kb.py) | Proxy | Wrapper for ocr_factory |
| `get_shared_paddle_ocr` | [ocr_factory.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/ocr_factory.py) | Implementation | PaddleOCR singleton factory |
| `_instantiate_yolo_layout_detector` | [pdf_to_base64_kb.py](file:///C:/Users/zgz31/Desktop/PaddleModels/tools/pdf_to_base64_kb.py) | Proxy | Wrapper for layout_detector |
| `instantiate_yolo_layout_detector` | [layout_detector.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/layout_detector.py) | Implementation | YOLO model loader |
| `_detect_yolo_layout_blocks` | [pdf_to_base64_kb.py](file:///C:/Users/zgz31/Desktop/PaddleModels/tools/pdf_to_base64_kb.py) | Proxy | Wrapper for layout_detector |
| `detect_yolo_layout_blocks` | [layout_detector.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/layout_detector.py) | Implementation | YOLO inference logic |
| `_extract_native_table_cells` | [pdf_to_base64_kb.py](file:///C:/Users/zgz31/Desktop/PaddleModels/tools/pdf_to_base64_kb.py) | Proxy | Wrapper for table_processor |
| `extract_native_table_cells` | [table_processor.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/table_processor.py) | Implementation | PDFMiner table cell extractor |
| `_detect_table_bboxes_from_image_bytes` | [pdf_to_base64_kb.py](file:///C:/Users/zgz31/Desktop/PaddleModels/tools/pdf_to_base64_kb.py) | Proxy | Wrapper for table_processor |
| `detect_table_bboxes_from_image_bytes` | [table_processor.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/table_processor.py) | Implementation | CV-based table detection |
| `ocr_image_bytes` | [ocr_engine.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/ocr_engine.py) | Implementation | Main OCR entry point |
| `page_has_structural_visual_signal` | [vision_utils.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/vision_utils.py) | Implementation | Structural line detection |
| `collect_repeated_watermark_candidates` | [watermark_utils.py](file:///C:/Users/zgz31/Desktop/PaddleModels/imaging/watermark_utils.py) | Implementation | Watermark identification |
| `run_v2` | [v2_runner.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/v2_runner.py) | Entry | Main v2 pipeline execution loop |
| `route_page` | [page_router.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/page_router.py) | Implementation | Per-page native/ocr routing decision (never hybrid); deterministic ocr fallback |
| `probe_page_native_chars` | [native_adapter.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/extraction_adapters/native_adapter.py) | Implementation | Cheap per-page native text probe (non-whitespace char count) used only for routing |
| `extract_page` | [native_adapter.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/extraction_adapters/native_adapter.py) | Implementation | High-precision native PDF extraction |
| `detect_blocks` | [layout_adapter.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/extraction_adapters/layout_adapter.py) | Implementation | Visual block detection (tables/figures) |
| `project` | [semantic_projector.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/semantic_projector.py) | Implementation | Canonical IR to Knowledge Base projection |
| `_canonical_table_grid` | [main.py](file:///C:/Users/zgz31/Desktop/PaddleModels/src/paddle_models/cli/main.py) | Implementation | Resolve table block canonical 2D grid (prefer list `rows`, else `table_rows`) |
| `_canonical_rows_grid` | [table_diff.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/compatibility/table_diff.py) | Implementation | Resolve canonical 2D grid for table diff (prefer list `rows`, else `table_rows`) |
| `_legacy_numeric_count` | [table_diff.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/compatibility/table_diff.py) | Implementation | Return numeric row/col count only; None for lists/dicts/non-numeric |
| `page_ir_cache_config` | [run_context.py](file:///C:/Users/zgz31/Desktop/PaddleModels/models/run_context.py) | Implementation | Stable IR-affecting subset of BuildProfile folded into the page-cache fingerprint |
| `compute_page_fingerprint` | [page_cache.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/page_cache.py) | Implementation | SHA-256 page-cache identity over schema version, IR profile config, router threshold and route strategy |
| `PageCache.load` | [page_cache.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/page_cache.py) | Implementation | Validate a page-cache metadata envelope; return DocPage on a full match, else a safe miss. Long-path aware; a location-level OSError disables the cache for the run |
| `PageCache.store` | [page_cache.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/page_cache.py) | Implementation | Atomically write the fingerprinted page-cache envelope (temp file + os.replace), addressed via `fs_path`; disables the cache after one warning when the location is unusable |
| `page_ir_from_dict` | [page_cache.py](file:///C:/Users/zgz31/Desktop/PaddleModels/pipeline/page_cache.py) | Implementation | Reconstruct a DocPage from its plain-dict Canonical IR form |
| `check_ir_schema` | [check_project.py](file:///C:/Users/zgz31/Desktop/PaddleModels/scripts/check_project.py) | Implementation | Validate the Canonical IR schema (contracts/knowledge-base.v2.schema.json) exists and is valid JSON |
| `check_kb_schema` | [check_project.py](file:///C:/Users/zgz31/Desktop/PaddleModels/scripts/check_project.py) | Implementation | Validate the Android KB schema (contracts/knowledge_base_schema_v2.json) exists and is valid JSON |
| `_repo_root` | [main.py](file:///C:/Users/zgz31/Desktop/PaddleModels/src/paddle_models/cli/main.py) | Implementation | Return the source-checkout root when running from a git checkout, else None (installed wheels have none) |
| `_contracts_dir` | [main.py](file:///C:/Users/zgz31/Desktop/PaddleModels/src/paddle_models/cli/main.py) | Implementation | Locate the contract schemas via the installed `contracts` package (importlib.resources) with a checkout fallback |
| `_resolve_contract` | [main.py](file:///C:/Users/zgz31/Desktop/PaddleModels/src/paddle_models/cli/main.py) | Implementation | Resolve a contract file by name against the packaged contracts directory (installed or checkout) |
| `_tool_script` | [main.py](file:///C:/Users/zgz31/Desktop/PaddleModels/src/paddle_models/cli/main.py) | Implementation | Locate a bundled `tools/` script (installed `tools` package or checkout) for the legacy adapter |
| `get_cache_dir` | [run_context.py](file:///C:/Users/zgz31/Desktop/PaddleModels/models/run_context.py) | Implementation | `<PADDLE_CACHE_ROOT>/<full input SHA>/<profile>`; created via `fs_path` and tolerant of an uncreatable location (cache is optional) |
| `fs_path` | [fs_paths.py](file:///C:/Users/zgz31/Desktop/PaddleModels/utils/fs_paths.py) | Implementation | Windows extended-length (`\\?\`) path form that lifts MAX_PATH for the cache; no-op on other platforms |
