# SCRIPT_CHANGE_LOG

| Date | Task | Files Modified | Description |
|---|---|---|---|
| 2026-09-09 | Refactor: Modularize imaging subsystem | `pdf_to_base64_kb.py`, `imaging/*` | Extracted BBox utils, Watermark utils, OCR engine, and Vision utils. Reduced main script by ~2000 lines. |
| 2026-09-14 | Phase 1 & 2: v2 Pipeline Infrastructure | `pipeline/v2_runner.py`, `pipeline/extraction_adapters/*`, `pipeline/canonical_ir.py`, `models/run_context.py` | Established new v2 pipeline with unique CLI, RunContext, and Canonical IR. Implemented NativeAdapter with artifact filtering and shadow mode. |
| 2026-09-15 | Phase 3: Asset Convergence & Contract | `pipeline/semantic_projector.py`, `v2_runner.py`, `docs/ANDROID_CONTRACT_v2.md`, `docs/kb_v2_schema.json` | Implemented SemanticProjector for Android-compatible JSON export. Added image extraction and linking to shots/ directory. Formalized v2 schema. |
