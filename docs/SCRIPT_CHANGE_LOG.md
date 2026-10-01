# SCRIPT_CHANGE_LOG

| Date | Task | Files Modified | Description |
|---|---|---|---|
| 2026-09-09 | Refactor: Modularize imaging subsystem | `pdf_to_base64_kb.py`, `imaging/*` | Extracted BBox utils, Watermark utils, OCR engine, and Vision utils. Reduced main script by ~2000 lines. |
| 2026-09-14 | Phase 1 & 2: v2 Pipeline Infrastructure | `pipeline/v2_runner.py`, `pipeline/extraction_adapters/*`, `pipeline/canonical_ir.py`, `models/run_context.py` | Established new v2 pipeline with unique CLI, RunContext, and Canonical IR. Implemented NativeAdapter with artifact filtering and shadow mode. |
| 2026-09-15 | Phase 3: Asset Convergence & Contract | `pipeline/semantic_projector.py`, `v2_runner.py`, `docs/ANDROID_CONTRACT_v2.md`, `docs/kb_v2_schema.json` | Implemented SemanticProjector for Android-compatible JSON export. Added image extraction and linking to shots/ directory. Formalized v2 schema. |
| 2026-10-02 | Fix: Normalize v2 table rows contract | `pipeline/semantic_projector.py`, `contracts/knowledge_base_schema_v2.json`, `docs/ANDROID_CONTRACT_v2.md`, `tests/test_table_slice.py` | v2 table blocks now emit 2D `rows` (anchor-only grid) and no longer write integer `rows`/`cols` or `table_rows`; aligned the CLI-enforced KB schema; added row-contract regression tests. |
| 2026-10-02 | Fix: Align v2 table rows consumers | `src/paddle_models/cli/main.py`, `pipeline/compatibility/table_diff.py`, `tests/test_table_slice.py`, `docs/SCRIPT_FUNCTION_MAP.md` | `_classify_table_status`/`kb_metrics_from_obj` and `collect_v2_tables`/`_table_geometry_stats` now consume canonical 2D `rows` first (empty list respected), fall back to legacy `table_rows`, and never `int()` a grid; added consumer-closure regression tests. |
