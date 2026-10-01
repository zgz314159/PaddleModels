#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from imaging.text_utils import safe_str as _safe_str
from imaging.kb_utils import MergeScope, entry_in_scope
from imaging.structure_overlay import overlay_structure_bboxes_from_source
from imaging.asset_alignment import run_asset_alignment_check

# Modularized imports
from pipeline.cleaning.normalization import (
    dedupe_figure_image_blocks,
    cleanup_empty_split_entries,
    sanitize_mixed_text_entry_artifact_tails,
    cleanup_known_text_only_figure_reference_entries,
    collect_kb_image_basenames,
    prune_unreferenced_screenshot_files
)
from pipeline.alignment.figure_processor import (
    canonicalize_figure_nodes_across_entries,
    rebuild_entry_figure_nodes,
    normalize_visual_snapshot_uris,
    rebind_figure_images_to_reference_entries,
    prune_unreferenced_image_only_figure_nodes
)
from pipeline.alignment.source_syncer import (
    sync_top_level_entries_from_source_sections,
    repair_degraded_numbered_entries_from_source_sections
)
from pipeline.merging.manifest_merger import merge_manifest_into_kb
from pipeline.export.kb_exporter import (
    atomic_write_json,
    refresh_file_metadata,
    strip_embedded_image_payloads
)

def _read_json(path: str) -> Any:
    import json
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Merge manifest items into KB")
    p.add_argument("--kb", required=True, help="Path to knowledge_base.json")
    p.add_argument("--manifest", required=True, help="Path to manifest JSON")
    p.add_argument("--out", help="Output path (atomic)")
    p.add_argument("--debug-merge", action="store_true", help="Print debug info")
    p.add_argument("--dry-run", action="store_true", help="Do not write output")
    # Add other arguments as needed for legacy compatibility
    return p

def main(argv: List[str]) -> int:
    args = build_arg_parser().parse_args(argv)
    
    kb_path = args.kb
    manifest_path = args.manifest
    out_path = args.out or kb_path

    kb = _read_json(kb_path)
    manifest = _read_json(manifest_path)
    
    # Core Orchestration
    stats = merge_manifest_into_kb(
        kb=kb,
        manifest=manifest,
        insert_mode="after-page",
        page_match="best",
        prefer_heading=True,
        dry_run=args.dry_run,
        scope=None,
        debug_merge=args.debug_merge
    )
    
    # Cleanup & Normalization
    cleanup_empty_split_entries(kb, debug=args.debug_merge)
    dedupe_figure_image_blocks(kb, debug=args.debug_merge)
    
    # Alignment & Processing
    canonicalize_figure_nodes_across_entries(kb, debug=args.debug_merge)
    
    # Final Export
    if not args.dry_run:
        refresh_file_metadata(kb)
        strip_embedded_image_payloads(kb)
        atomic_write_json(out_path, kb)
        print(f"Wrote merged KB to {out_path}")
        
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
