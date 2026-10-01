"""Legacy KB contract adapter — fill missing block ids only (Phase 2B).

Does not modify tools/pdf_to_base64_kb.py. Input dict is never mutated.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

from models.entry_model import stable_block_id

ADAPTER_NAME = "legacy_kb_contract_v1"


def _block_seed(
    file_id: str,
    entry_id: str,
    block_index: int,
    block: Dict[str, Any],
) -> str:
    """Deterministic seed — blockIndex participates so identical blocks stay unique."""
    btype = str(block.get("type") or "")
    page = str(block.get("pageNumber") if block.get("pageNumber") is not None else "")
    uri = str(block.get("imageUri") or block.get("src") or "")
    caption = str(block.get("caption") or "")
    return "|".join(
        [
            file_id,
            entry_id,
            str(block_index),
            btype,
            page,
            uri,
            caption,
        ]
    )


def normalize_legacy_kb_contract(kb: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (new_kb, report). Only fills missing/blank block `id` fields.

    report = {
      "applied": bool,
      "transforms": {"missing_block_ids": N},
      "changed_paths": ["entries.0.blocks.10.id", ...],
      "adapter": ADAPTER_NAME,
    }
    """
    new_kb = copy.deepcopy(kb) if isinstance(kb, dict) else {}
    report: Dict[str, Any] = {
        "adapter": ADAPTER_NAME,
        "applied": False,
        "transforms": {"missing_block_ids": 0},
        "changed_paths": [],
    }

    if not isinstance(new_kb, dict):
        return new_kb, report

    file_meta = new_kb.get("fileMetadata") or {}
    file_id = str(file_meta.get("fileId") or "knowledge_base")
    entries = new_kb.get("entries")
    if not isinstance(entries, list):
        return new_kb, report

    missing = 0
    changed: List[str] = []

    for ei, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("entryId") or f"entry_{ei}")
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for bi, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            raw_id = block.get("id")
            if isinstance(raw_id, str) and raw_id.strip():
                continue  # preserve existing non-empty id
            seed = _block_seed(file_id, entry_id, bi, block)
            block["id"] = stable_block_id(seed)
            missing += 1
            changed.append(f"entries.{ei}.blocks.{bi}.id")

    report["transforms"]["missing_block_ids"] = missing
    report["changed_paths"] = changed
    report["applied"] = missing > 0
    return new_kb, report


def normalize_legacy_kb_contract_file_payload(payload: Any) -> Tuple[Any, Dict[str, Any]]:
    """Convenience wrapper for loaded JSON payloads."""
    if not isinstance(payload, dict):
        return payload, {
            "adapter": ADAPTER_NAME,
            "applied": False,
            "transforms": {"missing_block_ids": 0},
            "changed_paths": [],
        }
    return normalize_legacy_kb_contract(payload)
