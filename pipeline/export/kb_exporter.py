import json
import os
from typing import Any, Dict, Set
from imaging.text_utils import safe_str as _safe_str

def atomic_write_json(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)

def refresh_file_metadata(kb: Dict[str, Any]) -> None:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return

    image_uris: Set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        table_image_uri = _safe_str(entry.get("tableImageUri")).strip()
        if table_image_uri:
            image_uris.add(table_image_uri)

        entry_image_uris = entry.get("imageUris")
        if isinstance(entry_image_uris, list):
            for uri in entry_image_uris:
                text = _safe_str(uri).strip()
                if text:
                    image_uris.add(text)

        blocks = entry.get("blocks")
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                text = _safe_str(
                    block.get("imageUri")
                    or block.get("image_uri")
                    or block.get("src")
                    or block.get("uri")
                ).strip()
                if text:
                    image_uris.add(text)

    metadata = kb.get("fileMetadata")
    if not isinstance(metadata, dict):
        metadata = {}
        kb["fileMetadata"] = metadata

    metadata["entriesCount"] = len(entries)
    metadata["imagesCount"] = len(image_uris)

def strip_embedded_image_payloads(kb: Dict[str, Any]) -> None:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if "image_base64" in block:
                block.pop("image_base64")
            if "base64" in block:
                block.pop("base64")

def export_final_kb(kb: Dict[str, Any], output_path: str) -> None:
    refresh_file_metadata(kb)
    strip_embedded_image_payloads(kb)
    atomic_write_json(output_path, kb)
