#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import math
import os
import re
import shutil
import sys
import bisect
from difflib import SequenceMatcher
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.file_id_sanitizer import sanitize_asset_relative_path, sanitize_file_id_for_path_only


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _atomic_write_json(path: str, data: Any) -> None:
    tmp_path = f"{path}.tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _refresh_file_metadata(kb: Dict[str, Any]) -> None:
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

    metadata["entriesCount"] = int(len(entries))
    metadata["imagesCount"] = int(len(image_uris))


def _is_figure_scope_title_fragment(text: str) -> bool:
    raw = _safe_str(text).strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact or len(compact) > 24:
        return False
    if re.match(r"^(第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[章节]|[一二三四五六七八九十]+\s*[、,，.])", raw):
        return False
    if re.match(r"^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+", compact):
        return True
    if any(ch in compact for ch in "。；;！？!?：:"):
        return False
    if re.fullmatch(r"[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?", compact):
        return True
    if re.search(r"\d+(?:mm|MM|cm|CM|kv|kV|V|A|m)", compact):
        return True
    if any(token in compact for token in ("中心线", "尺寸", "推荐", "最小布置", "侧视图", "平面图", "屏前通道")):
        return True
    if re.search(r"(?:×|x|X|~|〜|－|-)", compact) and len(compact) <= 18:
        return True
    return False


def _canonical_image_key(uri: str) -> str:
    return _basename_from_uri(uri).strip().lower()


def _prune_figure_node_image_refs(entry: Dict[str, Any]) -> None:
    blocks = entry.get("blocks")
    nodes = entry.get("figureNodes")
    if not isinstance(blocks, list) or not isinstance(nodes, list):
        return

    present_block_ids: Set[str] = set()
    present_uris: Set[str] = set()
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_id = _safe_str(block.get("id")).strip()
        if block_id:
            present_block_ids.add(block_id)
        uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src") or block.get("uri")).strip()
        if uri:
            present_uris.add(uri)

    for node in nodes:
        if not isinstance(node, dict):
            continue
        image_block_ids = [block_id for block_id in node.get("imageBlockIds", []) if _safe_str(block_id).strip() in present_block_ids]
        image_uris = [uri for uri in node.get("imageUris", []) if _safe_str(uri).strip() in present_uris]
        node["imageBlockIds"] = image_block_ids
        node["imageUris"] = image_uris
        if image_uris:
            node["imageUri"] = image_uris[0]
            node["canonicalImageKey"] = _canonical_image_key(image_uris[0])
        else:
            node.pop("imageUri", None)
            node.pop("canonicalImageKey", None)
        node["continued"] = len(image_uris) > 1


def _dedupe_figure_image_blocks(kb: Dict[str, Any], *, debug: bool = False) -> Dict[str, int]:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return {
            "duplicateImageBlocksRemoved": 0,
            "duplicateImageKeys": 0,
        }

    seen_by_key: Dict[str, Tuple[int, str]] = {}
    duplicate_keys: Set[str] = set()
    removed = 0

    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        kept_blocks: List[Any] = []
        for block in blocks:
            if not isinstance(block, dict):
                kept_blocks.append(block)
                continue

            block_type = _safe_str(block.get("type")).strip().lower()
            if block_type not in {"image", "figure", "equation", "table"}:
                kept_blocks.append(block)
                continue

            uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src") or block.get("uri")).strip()
            key = _canonical_image_key(uri)
            if not key:
                kept_blocks.append(block)
                continue

            block["canonicalImageKey"] = key
            owner = seen_by_key.get(key)
            if owner is None:
                seen_by_key[key] = (entry_index, _safe_str(block.get("id")).strip())
                kept_blocks.append(block)
                continue

            duplicate_keys.add(key)
            removed += 1
            if debug:
                print(
                    f"[MergeDedup] duplicate_image={key} drop_entry_index={entry_index} keep_entry_index={owner[0]}"
                )

        if len(kept_blocks) != len(blocks):
            entry["blocks"] = kept_blocks
        _prune_figure_node_image_refs(entry)

    return {
        "duplicateImageBlocksRemoved": int(removed),
        "duplicateImageKeys": int(len(duplicate_keys)),
    }


def _compute_semantic_audit(kb: Dict[str, Any], *, dedupe_metrics: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    entries = kb.get("entries")
    audit = {
        "entriesWithFigureScopeTitles": 0,
        "entriesWithWeakLeafTitle": 0,
        "figuresWithCalloutsNoCaption": 0,
        "figuresReferencedByBody": 0,
        "duplicateImageBlocksRemoved": int((dedupe_metrics or {}).get("duplicateImageBlocksRemoved") or 0),
        "duplicateImageKeys": int((dedupe_metrics or {}).get("duplicateImageKeys") or 0),
        "canonicalFigureNodes": 0,
        "canonicalFigureMergedGroups": 0,
    }
    if not isinstance(entries, list):
        return audit

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        job_title = _safe_str(entry.get("jobTitle")).strip()
        parts = [part.strip() for part in job_title.split("/") if part.strip()]
        leaf = parts[-1] if parts else ""
        if leaf and _is_figure_scope_title_fragment(leaf):
            audit["entriesWithFigureScopeTitles"] += 1
            audit["entriesWithWeakLeafTitle"] += 1

        figure_nodes = entry.get("figureNodes")
        if not isinstance(figure_nodes, list):
            continue
        for node in figure_nodes:
            if not isinstance(node, dict):
                continue
            has_caption = bool(_safe_str(node.get("caption")).strip()) or bool(node.get("captionBlockIds"))
            has_callout = bool(node.get("calloutBlockIds"))
            has_reference = bool(node.get("referenceBlockIds"))
            if has_callout and not has_caption:
                audit["figuresWithCalloutsNoCaption"] += 1
            if has_reference:
                audit["figuresReferencedByBody"] += 1
    canonical_nodes = kb.get("canonicalFigureNodes")
    if isinstance(canonical_nodes, list):
        audit["canonicalFigureNodes"] = len([node for node in canonical_nodes if isinstance(node, dict)])
        audit["canonicalFigureMergedGroups"] = len([
            node for node in canonical_nodes
            if isinstance(node, dict) and len(node.get("localFigureNodeIds") or []) > 1
        ])
    return audit


def _entry_position_value(entry: Dict[str, Any]) -> int:
    try:
        return int(entry.get("position") or 10**9)
    except Exception:
        return 10**9


def _figure_scope_key(entry: Dict[str, Any]) -> str:
    unit_name = _safe_str(entry.get("unitName")).strip()
    if unit_name:
        return unit_name
    return _job_title_cluster_key(entry)


def _make_canonical_figure_node_id(cluster_key: str, label_normalized: str, anchor_page: int, ordinal: int) -> str:
    import hashlib

    seed = f"canonical-figure|{cluster_key}|{label_normalized}|{anchor_page}|{ordinal}".encode("utf-8")
    return f"cfg_{hashlib.sha1(seed).hexdigest()[:12]}"


def _canonicalize_figure_nodes_across_entries(kb: Dict[str, Any], *, debug: bool = False) -> Dict[str, int]:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        kb.pop("canonicalFigureNodes", None)
        return {
            "canonicalFigureNodes": 0,
            "canonicalFigureMergedGroups": 0,
        }

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for node in entry.get("figureNodes") or []:
            if not isinstance(node, dict):
                continue
            node.pop("canonicalFigureNodeId", None)
            node.pop("canonicalFigureAnchorEntryId", None)
            node.pop("resolvedCaption", None)
            node.pop("resolvedImageUri", None)
            node.pop("resolvedImageUris", None)

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block.pop("canonicalFigureNodeId", None)
            block.pop("canonicalFigureReferenceIds", None)

    candidates_by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        figure_nodes = entry.get("figureNodes")
        if not isinstance(figure_nodes, list):
            continue
        cluster_key = _job_title_cluster_key(entry)
        scope_key = _figure_scope_key(entry)
        position_value = _entry_position_value(entry)
        dominant_page = _entry_dominant_page(entry)
        entry_id = _safe_str(entry.get("entryId")).strip()
        for node_index, node in enumerate(figure_nodes):
            if not isinstance(node, dict):
                continue
            label_normalized = _safe_str(node.get("labelNormalized")).strip()
            if not label_normalized:
                continue
            page_numbers = [
                int(page) for page in (node.get("pageNumbers") or []) if isinstance(page, int)
            ]
            primary_page = min(page_numbers) if page_numbers else (dominant_page or 0)
            candidates_by_group.setdefault((scope_key, label_normalized), []).append({
                "entryIndex": entry_index,
                "nodeIndex": node_index,
                "entryId": entry_id,
                "scopeKey": scope_key,
                "clusterKey": cluster_key,
                "labelNormalized": label_normalized,
                "label": _safe_str(node.get("label")).strip(),
                "page": int(primary_page or 0),
                "position": position_value,
                "node": node,
            })

    canonical_nodes: List[Dict[str, Any]] = []
    merged_groups = 0
    canonical_counter = 0

    for (scope_key, label_normalized), items in candidates_by_group.items():
        items.sort(key=lambda item: (int(item["page"] or 0), int(item["position"]), int(item["entryIndex"])))
        clusters: List[List[Dict[str, Any]]] = []
        for item in items:
            if not clusters:
                clusters.append([item])
                continue
            previous = clusters[-1][-1]
            same_page_band = abs(int(item["page"] or 0) - int(previous["page"] or 0)) <= 2
            same_position_band = abs(int(item["position"]) - int(previous["position"])) <= 6
            if same_page_band and same_position_band:
                clusters[-1].append(item)
            else:
                clusters.append([item])

        for cluster in clusters:
            canonical_counter += 1

            def _anchor_score(item: Dict[str, Any]) -> Tuple[int, int, int, int]:
                node = item["node"]
                image_count = len(node.get("imageUris") or [])
                caption_score = 1 if _safe_str(node.get("caption")).strip() else 0
                caption_score += len(node.get("captionBlockIds") or [])
                caption_score += len(node.get("captionTexts") or [])
                callout_penalty = -len(node.get("calloutBlockIds") or []) if not image_count and not caption_score else 0
                return (image_count, caption_score, callout_penalty, -int(item["position"]))

            anchor = max(cluster, key=_anchor_score)
            anchor_node = anchor["node"]
            anchor_page = int(anchor.get("page") or 0)
            canonical_id = _make_canonical_figure_node_id(scope_key, label_normalized, anchor_page, canonical_counter)

            image_uris: List[str] = []
            image_block_ids: List[str] = []
            caption_texts: List[str] = []
            caption_block_ids: List[str] = []
            callout_texts: List[str] = []
            callout_block_ids: List[str] = []
            reference_texts: List[str] = []
            reference_block_ids: List[str] = []
            page_numbers: List[int] = []
            local_ids: List[str] = []
            entry_ids: List[str] = []

            for item in cluster:
                node = item["node"]
                local_id = _safe_str(node.get("id")).strip()
                if local_id and local_id not in local_ids:
                    local_ids.append(local_id)
                entry_id = _safe_str(item.get("entryId")).strip()
                if entry_id and entry_id not in entry_ids:
                    entry_ids.append(entry_id)
                for source, target in (
                    (node.get("imageUris") or [], image_uris),
                    (node.get("imageBlockIds") or [], image_block_ids),
                    (node.get("captionTexts") or [], caption_texts),
                    (node.get("captionBlockIds") or [], caption_block_ids),
                    (node.get("calloutTexts") or [], callout_texts),
                    (node.get("calloutBlockIds") or [], callout_block_ids),
                    (node.get("referenceTexts") or [], reference_texts),
                    (node.get("referenceBlockIds") or [], reference_block_ids),
                    (node.get("pageNumbers") or [], page_numbers),
                ):
                    for value in source:
                        clean = value if isinstance(value, int) else _safe_str(value).strip()
                        if clean and clean not in target:
                            target.append(clean)

            canonical_caption = _safe_str(anchor_node.get("caption")).strip()
            if not canonical_caption:
                for text_value in caption_texts:
                    candidate = _safe_str(text_value).strip()
                    if candidate:
                        canonical_caption = candidate
                        break
            canonical_label = _safe_str(anchor_node.get("label")).strip() or _safe_str(cluster[0]["label"]).strip()
            canonical_node: Dict[str, Any] = {
                "id": canonical_id,
                "label": canonical_label,
                "labelNormalized": label_normalized,
                "caption": canonical_caption,
                "captionTexts": caption_texts,
                "captionBlockIds": caption_block_ids,
                "calloutTexts": callout_texts,
                "calloutBlockIds": callout_block_ids,
                "referenceTexts": reference_texts,
                "referenceBlockIds": reference_block_ids,
                "imageUris": image_uris,
                "imageBlockIds": image_block_ids,
                "pageNumbers": sorted(int(page) for page in page_numbers if isinstance(page, int) and int(page) > 0),
                "localFigureNodeIds": local_ids,
                "entryIds": entry_ids,
                "anchorEntryId": _safe_str(anchor.get("entryId")).strip(),
                "anchorFigureNodeId": _safe_str(anchor_node.get("id")).strip(),
                "scopeKey": scope_key,
                "clusterKey": cluster_key,
            }
            if image_uris:
                canonical_node["imageUri"] = image_uris[0]
                canonical_node["canonicalImageKey"] = _canonical_image_key(image_uris[0])
            canonical_nodes.append(canonical_node)
            if len(cluster) > 1:
                merged_groups += 1

            for item in cluster:
                node = item["node"]
                node["canonicalFigureNodeId"] = canonical_id
                node["canonicalFigureAnchorEntryId"] = canonical_node["anchorEntryId"]
                if canonical_caption and not _safe_str(node.get("caption")).strip():
                    node["resolvedCaption"] = canonical_caption
                if image_uris and not (node.get("imageUris") or []):
                    node["resolvedImageUris"] = list(image_uris)
                    node["resolvedImageUri"] = image_uris[0]

    # Backfill canonical ids onto related blocks.
    local_to_canonical: Dict[str, str] = {}
    for node in canonical_nodes:
        if not isinstance(node, dict):
            continue
        canonical_id = _safe_str(node.get("id")).strip()
        for local_id in node.get("localFigureNodeIds") or []:
            clean_local_id = _safe_str(local_id).strip()
            if clean_local_id and canonical_id:
                local_to_canonical[clean_local_id] = canonical_id

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            node_id = _safe_str(block.get("figureNodeId")).strip()
            if node_id and node_id in local_to_canonical:
                block["canonicalFigureNodeId"] = local_to_canonical[node_id]
            ref_ids = block.get("figureReferenceIds") if isinstance(block.get("figureReferenceIds"), list) else []
            canonical_ref_ids: List[str] = []
            for ref_id in ref_ids:
                clean_ref_id = _safe_str(ref_id).strip()
                canonical_ref_id = local_to_canonical.get(clean_ref_id)
                if canonical_ref_id and canonical_ref_id not in canonical_ref_ids:
                    canonical_ref_ids.append(canonical_ref_id)
            if canonical_ref_ids:
                block["canonicalFigureReferenceIds"] = canonical_ref_ids

    if canonical_nodes:
        kb["canonicalFigureNodes"] = canonical_nodes
    else:
        kb.pop("canonicalFigureNodes", None)

    if debug and canonical_nodes:
        print(
            f"[MergeCanonicalFigure] canonical_nodes={len(canonical_nodes)} merged_groups={merged_groups}"
        )

    return {
        "canonicalFigureNodes": len(canonical_nodes),
        "canonicalFigureMergedGroups": merged_groups,
    }


def _strip_embedded_image_payloads(kb: Dict[str, Any]) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    removed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        if entry.pop("image_data", None) is not None:
            removed += 1

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.pop("image_data", None) is not None:
                removed += 1
    return removed


def _collect_kb_image_basenames(kb: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return out

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        table_image_uri = _safe_str(entry.get("tableImageUri")).strip()
        if table_image_uri:
            base = _basename_from_uri(table_image_uri)
            if base:
                out.add(base)

        entry_image_uris = entry.get("imageUris")
        if isinstance(entry_image_uris, list):
            for uri in entry_image_uris:
                base = _basename_from_uri(_safe_str(uri))
                if base:
                    out.add(base)

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            uri = _safe_str(
                block.get("imageUri")
                or block.get("image_uri")
                or block.get("src")
                or block.get("uri")
            ).strip()
            if not uri:
                continue
            base = _basename_from_uri(uri)
            if base:
                out.add(base)

    return out


def _prune_unreferenced_screenshot_files(kb: Dict[str, Any], manifest: Dict[str, Any], *, debug: bool) -> int:
    out_dir = _safe_str(manifest.get("outDir")).strip()
    if not out_dir or not os.path.isdir(out_dir):
        return 0

    # Only prune files that are part of the current manifest batch; keep
    # unrelated or pre-existing screenshots (for example DOCX table_posN.png).
    manifest_basenames: Set[str] = set()
    try:
        items = manifest.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_uri = _safe_str(
                    item.get("outFile")
                    or item.get("assetUri")
                    or item.get("imageUri")
                    or item.get("src")
                ).strip()
                base = _basename_from_uri(raw_uri)
                if base:
                    manifest_basenames.add(base)
    except Exception:
        manifest_basenames = set()

    referenced = _collect_kb_image_basenames(kb)
    if not referenced:
        return 0

    removed = 0
    for name in os.listdir(out_dir):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        lower = name.lower()
        if not lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        # Keep DOCX table exports even when not referenced by current merge.
        if re.match(r"^table_pos\d+\.(png|jpg|jpeg|webp)$", lower):
            continue
        # Never prune files that are outside this manifest's output set, except
        # for legacy bridge copies that may remain after URI normalization.
        if manifest_basenames and name not in manifest_basenames and not lower.startswith(("legend_p", "table_p")):
            continue
        if name in referenced:
            continue
        try:
            os.remove(path)
            removed += 1
            if debug:
                print(f"[MergeCleanup] removed_orphan={name}")
        except Exception:
            pass

    if debug and removed:
        print(f"[MergeCleanup] pruned_orphan_screenshots={removed}")
    return int(removed)


def _basename_from_uri(uri: str) -> str:
    s = _safe_str(uri).strip()
    if not s:
        return ""
    # assetUri is usually file:///android_asset/.../legend_p18_idx1.png
    # but tolerate plain filenames too.
    s = s.split("?")[0].split("#")[0]
    return os.path.basename(s)


def _parse_visual_snapshot_page_idx(uri: str) -> Tuple[Optional[int], Optional[int]]:
    name = _basename_from_uri(uri)
    match = re.search(r"visual_p(\d+)_(\d+)", name, flags=re.IGNORECASE)
    if not match:
        return (None, None)
    try:
        return (int(match.group(1)), int(match.group(2)))
    except Exception:
        return (None, None)


def _physical_page_from_asset_uri(uri: str) -> Optional[int]:
    s = _safe_str(uri).strip()
    if not s:
        return None
    match = re.search(r"_p(\d+)_", s, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _rewrite_legacy_snapshot_uri_to_visual(uri: str) -> str:
    text = _safe_str(uri).strip()
    if not text:
        return ""
    name = _basename_from_uri(text)
    match = re.search(r"(?:legend|table)_p(\d+)_idx(\d+)(\.[^.]+)$", name, flags=re.IGNORECASE)
    if not match:
        return text
    visual_name = f"visual_p{int(match.group(1))}_{int(match.group(2))}{match.group(3)}"
    return text[: len(text) - len(name)] + visual_name


def _normalize_visual_snapshot_uris(kb: Dict[str, Any], *, debug: bool) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    changed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        for key in ("tableImageUri", "imageUri", "image_uri"):
            current = _safe_str(entry.get(key)).strip()
            if not current:
                continue
            rewritten = _rewrite_legacy_snapshot_uri_to_visual(current)
            if rewritten != current:
                entry[key] = rewritten
                changed += 1

        image_uris = entry.get("imageUris")
        if isinstance(image_uris, list):
            local_changed = False
            rewritten_list: List[str] = []
            for value in image_uris:
                current = _safe_str(value).strip()
                rewritten = _rewrite_legacy_snapshot_uri_to_visual(current) if current else current
                rewritten_list.append(rewritten)
                if rewritten != current:
                    local_changed = True
            if local_changed:
                entry["imageUris"] = rewritten_list
                changed += 1

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for key in ("imageUri", "image_uri", "src", "uri"):
                current = _safe_str(block.get(key)).strip()
                if not current:
                    continue
                rewritten = _rewrite_legacy_snapshot_uri_to_visual(current)
                if rewritten != current:
                    block[key] = rewritten
                    changed += 1

    if debug and changed:
        print(f"[MergeCleanup] normalized_visual_snapshot_uris={changed}")
    return int(changed)


def _load_overrides(path: str) -> Dict[str, str]:
    """Load overrides mapping: imageFileName -> targetEntryId.

    Supported JSON shapes:
    - { "legend_p18_idx1.png": "railway_rules::table_2", ... }
    - { "move": { ... } }  (alias)

    Values must be entryId strings.
    """

    if not path:
        return {}
    try:
        if not os.path.exists(path):
            return {}
    except Exception:
        return {}

    try:
        data = _read_json(path)
    except Exception:
        return {}

    if isinstance(data, dict) and "move" in data and isinstance(data.get("move"), dict):
        data = data.get("move")

    out: Dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            kk = _safe_str(k).strip()
            vv = _safe_str(v).strip()
            if not kk or not vv:
                continue
            out[_basename_from_uri(kk) or kk] = vv
    return out


def _cleanup_empty_split_entries(kb: Dict[str, Any], *, debug: bool) -> int:
    """Remove split entries that have no snapshot at all.

    Split entry heuristic:
    - entryId contains '__p3_split' OR jobTitle contains '（图'
    Snapshot heuristic:
    - any block has imageUri/src (including table/image)
    """

    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    def _has_any_snapshot(e: Dict[str, Any]) -> bool:
        blocks = e.get("blocks")
        if not isinstance(blocks, list):
            return False
        for b in blocks:
            if not isinstance(b, dict):
                continue
            u = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
            if u:
                return True
        return False

    to_delete: List[int] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        eid = _safe_str(e.get("entryId")).strip()
        jt = _safe_str(e.get("jobTitle")).strip()
        is_split = ("__p3_split" in eid) or ("（图" in jt)
        if not is_split:
            continue
        if not _has_any_snapshot(e):
            to_delete.append(int(i))

    for di in sorted(to_delete, reverse=True):
        try:
            entries.pop(int(di))
        except Exception:
            pass

    if debug and to_delete:
        print(f"[MergeOverride] cleanup_empty_splits={len(to_delete)}")
    return int(len(to_delete))


def _apply_overrides_to_kb(kb: Dict[str, Any], overrides: Dict[str, str], *, debug: bool) -> int:
    """Apply image filename -> entryId overrides by moving blocks.

    Notes:
    - We primarily move ImageBlocks (type=image).
    - If the filename is found as a TableBlock.imageUri, we COPY it as an ImageBlock into the target
      (we do not remove it) to avoid breaking the source table rendering.
    """

    if not overrides:
        return 0

    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    id_to_idx: Dict[str, int] = {}
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        eid = _safe_str(e.get("entryId")).strip()
        if eid:
            id_to_idx[eid] = int(i)

    def _ensure_blocks(e: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks = e.get("blocks")
        if isinstance(blocks, list):
            # filter to dicts
            return [b for b in blocks if isinstance(b, dict)]
        return []

    def _is_split_shell(e: Dict[str, Any]) -> bool:
        eid = _safe_str(e.get("entryId")).strip()
        jt = _safe_str(e.get("jobTitle")).strip()
        return ("__p3_split" in eid) or ("（图" in jt)

    def _target_insert_image(target_entry: Dict[str, Any], image_uri: str, *, caption: str) -> bool:
        if not image_uri:
            return False
        blocks = target_entry.get("blocks")
        if not isinstance(blocks, list):
            blocks = []

        effective_caption = _safe_str(caption).strip()
        preferred_node: Optional[Dict[str, Any]] = None
        figure_nodes = target_entry.get("figureNodes") if isinstance(target_entry.get("figureNodes"), list) else []
        if effective_caption == "人工纠偏":
            reference_labeled_nodes = [
                node for node in figure_nodes
                if isinstance(node, dict)
                and _safe_str(node.get("labelNormalized")).strip()
                and bool(node.get("referenceBlockIds"))
            ]
            labeled_nodes = [
                node for node in figure_nodes
                if isinstance(node, dict)
                and _safe_str(node.get("labelNormalized")).strip()
            ]
            if len(reference_labeled_nodes) == 1:
                preferred_node = reference_labeled_nodes[0]
            elif len(labeled_nodes) == 1:
                preferred_node = labeled_nodes[0]
            if preferred_node is not None:
                effective_caption = (
                    _safe_str(preferred_node.get("label")).strip()
                    or _safe_str(preferred_node.get("caption")).strip()
                    or effective_caption
                )

        # Avoid duplicates.
        for b in blocks:
            if not isinstance(b, dict):
                continue
            u = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
            if _basename_from_uri(u) == _basename_from_uri(image_uri):
                return False

        # NOTE: Do not rely on Phase2's nested _physical_page_from_asset_uri (not in scope here).
        pn = 1
        try:
            m = re.search(r"_p(\d+)_", _safe_str(image_uri))
            if m:
                pn = int(m.group(1))
        except Exception:
            pn = 1
        img_block = _make_image_block(asset_uri=image_uri, page_number=pn, caption=effective_caption)
        if preferred_node is not None:
            preferred_node_id = _safe_str(preferred_node.get("id")).strip()
            preferred_label_normalized = _safe_str(preferred_node.get("labelNormalized")).strip()
            if preferred_node_id:
                img_block["figureNodeId"] = preferred_node_id
            if preferred_label_normalized:
                img_block["figureLabelNormalized"] = preferred_label_normalized
            img_block["figureRelation"] = "image"

        # If target has a table block, insert right after it; else append.
        insert_at = None
        for idx, b in enumerate(blocks):
            if not isinstance(b, dict):
                continue
            if _safe_str(b.get("type")).strip().lower() == "table":
                insert_at = int(idx) + 1
                break
        if insert_at is None:
            blocks.append(img_block)
        else:
            blocks.insert(int(insert_at), img_block)
        target_entry["blocks"] = blocks
        return True

    moved_blocks = 0

    # Pre-index blocks by filename for faster lookup.
    # Map: filename -> list of occurrences (entry_idx, block_idx, block_dict)
    occ_image: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    occ_table: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for ei, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        blocks = e.get("blocks")
        if not isinstance(blocks, list):
            continue
        for bi, b in enumerate(blocks):
            if not isinstance(b, dict):
                continue
            t = _safe_str(b.get("type")).strip().lower()
            if t not in ("image", "table"):
                continue
            if t == "image":
                u = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                fn = _basename_from_uri(u)
                if fn:
                    occ_image.setdefault(fn, []).append((int(ei), int(bi), b))
            elif t == "table":
                u = _safe_str(b.get("imageUri") or b.get("image_uri")).strip()
                fn = _basename_from_uri(u)
                if fn:
                    occ_table.setdefault(fn, []).append((int(ei), int(bi), b))

    for file_name, target_eid in overrides.items():
        fn = _basename_from_uri(file_name) or _safe_str(file_name).strip()
        if not fn:
            continue
        target_idx = id_to_idx.get(_safe_str(target_eid).strip())
        if target_idx is None:
            if debug:
                print(f"[MergeOverride] missing target entryId: {target_eid} for {fn}")
            continue
        target_entry = entries[int(target_idx)]
        if not isinstance(target_entry, dict):
            continue

        # 1) Move image blocks.
        occurrences = occ_image.get(fn, [])
        if occurrences:
            moved_this = 0
            skipped_this = 0
            # Remove from sources (reverse order per entry to keep indices stable).
            for ei, bi, b in sorted(occurrences, key=lambda t: (t[0], t[1]), reverse=True):
                try:
                    src_entry = entries[int(ei)]
                    if not isinstance(src_entry, dict):
                        continue
                    src_blocks = src_entry.get("blocks")
                    if not isinstance(src_blocks, list):
                        continue
                    # Capture uri before removal.
                    uri = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                    # IMPORTANT: Only remove from source if we successfully inserted into target.
                    ok = False
                    if uri:
                        ok = _target_insert_image(target_entry, uri, caption="人工纠偏")

                    if ok:
                        # Remove
                        try:
                            src_blocks.pop(int(bi))
                        except Exception:
                            # If index mismatch, fall back to remove by identity.
                            try:
                                src_blocks.remove(b)
                            except Exception:
                                pass
                        src_entry["blocks"] = src_blocks
                        moved_blocks += 1
                        moved_this += 1
                    else:
                        skipped_this += 1
                except Exception:
                    pass

            if debug:
                print(
                    f"[MergeOverride] moved image {fn} -> {target_eid} occurrences={len(occurrences)} moved={moved_this} skipped={skipped_this}"
                )

            if moved_this > 0:
                for other_ei, _other_bi, _other_block in sorted(occurrences, key=lambda t: (t[0], t[1]), reverse=True):
                    if int(other_ei) == int(target_idx):
                        continue
                    try:
                        other_entry = entries[int(other_ei)]
                    except Exception:
                        continue
                    if not isinstance(other_entry, dict):
                        continue
                    other_blocks = other_entry.get("blocks")
                    if not isinstance(other_blocks, list):
                        continue
                    retained_blocks: List[Dict[str, Any]] = []
                    removed_here = 0
                    for block in other_blocks:
                        if not isinstance(block, dict):
                            retained_blocks.append(block)
                            continue
                        uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
                        if _basename_from_uri(uri) == fn:
                            removed_here += 1
                            continue
                        retained_blocks.append(block)
                    if removed_here:
                        other_entry["blocks"] = retained_blocks
                        if debug:
                            print(f"[MergeOverride] removed duplicate image {fn} from {_safe_str(other_entry.get('entryId')).strip()} count={removed_here}")

        # 2) Copy table snapshots (do not remove).
        table_occ = occ_table.get(fn, [])
        if table_occ:
            copied = 0
            for ei, bi, b in table_occ:
                try:
                    src_entry = entries[int(ei)]
                    if not isinstance(src_entry, dict):
                        continue
                    uri = _safe_str(b.get("imageUri") or b.get("image_uri")).strip()
                    if not uri:
                        continue
                    ok = _target_insert_image(target_entry, uri, caption="人工纠偏")
                    if ok:
                        copied += 1
                        if _is_split_shell(src_entry):
                            b.pop("imageUri", None)
                            b.pop("image_uri", None)
                except Exception:
                    pass
            if debug:
                print(f"[MergeOverride] copied table snapshot {fn} -> {target_eid} copied={copied}")

    return int(moved_blocks)


def _sort_table_entry_images_after_merge(kb: Dict[str, Any], *, debug: bool) -> int:
    """Make table snapshot + continuation images ordered deterministically.

    For entries that look like tables (kind==table or jobTitle starts with 表格#):
    - The TableBlock.imageUri is treated as the "primary" snapshot.
    - Continuation screenshots are ImageBlocks after the first TableBlock.

    This function:
    1) Picks the earliest (page, idx) among current snapshot + continuation images, and
       sets it as the TableBlock.imageUri.
    2) If the previous snapshot is different, it is converted into an ImageBlock so it can
       appear later (e.g., p22 should be last).
    3) Removes any ImageBlock duplicating the chosen snapshot.
    4) Sorts ImageBlocks after the first TableBlock by (page, idx, filename).

    Result: screenshot order is stable like p17 -> p18 -> ... -> p22.
    """

    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    def _parse_order(uri: str, fallback_page: int, fallback_pos: int) -> Tuple[int, int, str, int]:
        s = _safe_str(uri).strip()
        page = fallback_page if fallback_page > 0 else 10**9
        idx = 10**9
        try:
            m = re.search(r"_p(\d+)_", s)
            if m:
                page = int(m.group(1))
        except Exception:
            pass
        try:
            m = re.search(r"_idx(\d+)\.(png|jpg|jpeg|webp)$", s, flags=re.IGNORECASE)
            if m:
                idx = int(m.group(1))
        except Exception:
            pass
        fn = _basename_from_uri(s)
        return (int(page), int(idx), fn, int(fallback_pos))

    def _is_table_snapshot_uri(uri: str) -> bool:
        fn = _basename_from_uri(uri).lower()
        if not fn:
            return False
        return fn.startswith("table_") or fn.startswith("tablepos") or fn.startswith("tablep")

    def _is_legend_snapshot_uri(uri: str) -> bool:
        fn = _basename_from_uri(uri).lower()
        if not fn:
            return False
        if fn.startswith("legend_"):
            return True
        page, idx = _parse_visual_snapshot_page_idx(fn)
        return page is not None and idx is not None

    def _snapshot_family(uri: str) -> str:
        if _is_legend_snapshot_uri(uri):
            return "legend"
        if _is_table_snapshot_uri(uri):
            return "table"
        return ""

    changed = 0
    for e in entries:
        if not isinstance(e, dict):
            continue
        kind = _safe_str(e.get("kind")).strip().lower()
        jt = _safe_str(e.get("jobTitle")).strip()
        looks_like_table = (kind == "table") or jt.startswith("表格#")
        if not looks_like_table:
            continue
        blocks = e.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            continue

        first_table_idx = None
        for i, b in enumerate(blocks):
            if not isinstance(b, dict):
                continue
            if _safe_str(b.get("type")).strip().lower() == "table":
                first_table_idx = int(i)
                break
        if first_table_idx is None:
            continue

        prefix = blocks[: first_table_idx + 1]
        rest = blocks[first_table_idx + 1 :]

        # Find the first table block and its current snapshot.
        table_block: Optional[Dict[str, Any]] = None
        for b in prefix:
            if not isinstance(b, dict):
                continue
            if _safe_str(b.get("type")).strip().lower() == "table":
                table_block = b
                break
        if not isinstance(table_block, dict):
            continue
        prev_snapshot_uri = _safe_str(table_block.get("imageUri") or table_block.get("image_uri")).strip()

        images: List[Dict[str, Any]] = []
        others: List[Dict[str, Any]] = []
        for b in rest:
            if not isinstance(b, dict):
                continue
            if _safe_str(b.get("type")).strip().lower() == "image":
                images.append(b)
            else:
                others.append(b)

        # Decide the primary snapshot. If any original legend screenshots exist for this
        # table entry, prefer the legend family and drop legacy table_* snapshots.
        candidates: List[Tuple[Tuple[int, int, str, int], str]] = []
        legend_present = False
        if _is_legend_snapshot_uri(prev_snapshot_uri):
            legend_present = True

        for pos, b in enumerate(images):
            uri = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
            if not uri:
                continue
            if not (_is_table_snapshot_uri(uri) or _is_legend_snapshot_uri(uri)):
                continue
            if _is_legend_snapshot_uri(uri):
                legend_present = True

        preferred_family = "legend" if legend_present else "table"

        if prev_snapshot_uri and _snapshot_family(prev_snapshot_uri) == preferred_family:
            candidates.append((_parse_order(prev_snapshot_uri, 1, -1), prev_snapshot_uri))

        for pos, b in enumerate(images):
            uri = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
            if not uri:
                continue
            if _snapshot_family(uri) != preferred_family:
                continue
            fallback_page = 1
            try:
                fallback_page = int(b.get("pageNumber") or 1)
            except Exception:
                fallback_page = 1
            candidates.append((_parse_order(uri, fallback_page, pos), uri))

        if not candidates:
            continue
        candidates.sort(key=lambda t: t[0])
        chosen_snapshot_uri = candidates[0][1]
        chosen_fn = _basename_from_uri(chosen_snapshot_uri)

        if chosen_snapshot_uri and chosen_snapshot_uri != prev_snapshot_uri:
            table_block["imageUri"] = chosen_snapshot_uri

            # Keep the old snapshot only when it belongs to the same family. If we are
            # switching from legacy table_* to original legend_* screenshots, the old
            # long snapshot must be dropped to avoid duplicate content.
            prev_fn = _basename_from_uri(prev_snapshot_uri)
            if (
                prev_snapshot_uri
                and prev_fn
                and prev_fn != chosen_fn
                and _snapshot_family(prev_snapshot_uri) == preferred_family
            ):
                pn = 1
                try:
                    m = re.search(r"_p(\d+)_", _safe_str(prev_snapshot_uri))
                    if m:
                        pn = int(m.group(1))
                except Exception:
                    pn = 1
                images.append(_make_image_block(asset_uri=prev_snapshot_uri, page_number=pn, caption="表格原件（续）"))

        # Remove any image block that duplicates the chosen snapshot.
        if chosen_fn:
            kept: List[Dict[str, Any]] = []
            for b in images:
                uri = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                if _basename_from_uri(uri) == chosen_fn:
                    continue
                kept.append(b)
            images = kept

        # Keep only screenshots from the chosen family.
        images = [
            b for b in images
            if _snapshot_family(_safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()) == preferred_family
        ]

        # If only one or zero continuation images, no need to sort further.
        if len(images) <= 1:
            e["blocks"] = prefix + images + others
            changed += 1
            continue

        ordered: List[Tuple[Tuple[int, int, str, int], Dict[str, Any]]] = []
        for pos, b in enumerate(images):
            uri = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
            fallback_page = 1
            try:
                fallback_page = int(b.get("pageNumber") or 1)
            except Exception:
                fallback_page = 1
            ordered.append((_parse_order(uri, fallback_page, pos), b))

        ordered.sort(key=lambda t: t[0])
        sorted_images = [block for _order_key, block in ordered]
        e["blocks"] = prefix + sorted_images + others
        changed += 1

    if debug and changed:
        print(f"[MergeOverride] sorted_table_entries={changed}")
    return int(changed)


def _sanitize_folder_name(name: str) -> str:
    # PATH-ONLY: Use for directory/asset names.
    # Allow hierarchical taxonomy paths like "铁路/规章制度/电力/...".
    return sanitize_asset_relative_path(name, default="_")


def normalize_for_search_like_app(text: str) -> str:
    """Port of TextSanitizer.normalizeForSearch (Kotlin).

    Keep only letters/digits and whitespace; everything else becomes space.
    Newlines are preserved (caller can post-process).
    """
    if not text or text.strip() == "":
        return ""
    normalized = unicodedata.normalize("NFC", text)
    out_chars: List[str] = []
    for ch in normalized:
        if ch == "\n":
            out_chars.append("\n")
        elif ch.isalnum():
            out_chars.append(ch)
        elif ch.isspace():
            out_chars.append(" ")
        else:
            out_chars.append(" ")

    out = "".join(out_chars)
    out = re.sub(r"[\t\r\u00A0 ]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return out.strip()


def _normalize_ocr_match_text(text: str) -> str:
    norm = normalize_for_search_like_app(_safe_str(text)).replace("\n", " ").strip()
    norm = re.sub(r"\s+", " ", norm)
    return norm


def _collapse_match_text(text: str) -> str:
    return re.sub(r"\s+", "", _normalize_ocr_match_text(text))


def _ocr_match_tokens(text: str) -> List[str]:
    norm = _normalize_ocr_match_text(text)
    if not norm:
        return []

    raw_tokens = [token.strip() for token in re.split(r"\s+", norm) if token.strip()]
    raw_tokens = [token for token in raw_tokens if len(token) >= 2]
    generic = {
        "检查", "要求", "内容", "项目", "备注", "规定", "单位", "名称", "标准", "周期", "范围",
        "应当", "必须", "可以", "进行", "作业", "设备", "附件", "附表", "表格", "图例",
    }

    out: List[str] = []
    seen: Set[str] = set()
    for token in raw_tokens:
        compact = re.sub(r"\s+", "", token)
        if len(compact) < 2 or compact in generic:
            continue
        if compact not in seen:
            seen.add(compact)
            out.append(compact)

    for i in range(len(raw_tokens) - 1):
        combo = re.sub(r"\s+", "", raw_tokens[i] + raw_tokens[i + 1])
        if len(combo) < 4 or combo in seen:
            continue
        seen.add(combo)
        out.append(combo)

    out.sort(key=lambda value: (-len(value), value))
    return out[:48]


def _normalize_for_similarity(text: str) -> str:
    """Normalize text for fuzzy matching (OCR anchorText vs jobTitle).

    Uses the same search-normalization rules as the app, then removes whitespace
    to reduce OCR spacing noise.
    """

    norm = normalize_for_search_like_app(_safe_str(text)).replace("\n", " ").strip()
    norm = re.sub(r"\s+", "", norm)
    return norm


def _levenshtein_distance_with_cutoff(a: str, b: str, cutoff: int) -> int:
    """Levenshtein distance with early cutoff.

    Returns a value > cutoff when the true distance exceeds cutoff.
    Uses a memory-efficient DP over the shorter string.
    """

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Ensure b is the shorter dimension for less memory.
    if len(a) < len(b):
        long_s, short_s = b, a
    else:
        long_s, short_s = a, b

    if cutoff < 0:
        cutoff = 0

    # Fast reject by length difference.
    if abs(len(long_s) - len(short_s)) > cutoff:
        return cutoff + 1

    prev = list(range(len(short_s) + 1))
    for i, ch_long in enumerate(long_s, start=1):
        cur = [i] + [0] * len(short_s)
        row_min = cur[0]
        for j, ch_short in enumerate(short_s, start=1):
            cost = 0 if ch_long == ch_short else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > cutoff:
            return cutoff + 1
        prev = cur

    return prev[-1]


def _levenshtein_similarity_ratio(a: str, b: str, *, cutoff_ratio: float) -> float:
    """Return similarity ratio in [0, 1] based on Levenshtein distance.

    ratio = 1 - dist / max(len(a), len(b)).
    Uses cutoff_ratio to early-exit.
    """

    a = a or ""
    b = b or ""
    if not a or not b:
        return 0.0

    max_len = max(len(a), len(b))
    if max_len <= 0:
        return 0.0

    # dist <= max_len * (1 - ratio)
    cutoff_dist = int(math.floor(max_len * (1.0 - float(cutoff_ratio))))
    dist = _levenshtein_distance_with_cutoff(a, b, cutoff=cutoff_dist)
    if dist > cutoff_dist:
        return 0.0
    return max(0.0, 1.0 - (float(dist) / float(max_len)))


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v)


def _maybe_configure_tesseract_cmd() -> None:
    try:
        import pytesseract  # type: ignore

        cmd = _safe_str(os.environ.get("TESSERACT_CMD")).strip()
        if not cmd:
            for candidate in (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ):
                if os.path.exists(candidate):
                    cmd = candidate
                    break
        if not cmd:
            return

        pytesseract.pytesseract.tesseract_cmd = cmd
    except Exception:
        return


def _ocr_image_file(path: str, *, lang: str = "chi_sim+eng") -> str:
    try:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception:
        return ""

    img_path = _safe_str(path).strip()
    if not img_path or not os.path.exists(img_path):
        return ""

    _maybe_configure_tesseract_cmd()

    try:
        img = Image.open(img_path).convert("L")
        width, height = img.size
        if width <= 0 or height <= 0:
            return ""

        scale = 2
        if max(width, height) < 900:
            scale = 3
        img = img.resize((max(1, width * scale), max(1, height * scale)))
        img = img.point(lambda pixel: 255 if pixel > 200 else 0)

        best = ""
        best_len = -1
        for psm in (6, 11, 12):
            text = pytesseract.image_to_string(img, lang=lang, config=f"--psm {psm}")
            text = re.sub(r"[\t\r\n]+", " ", _safe_str(text))
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > best_len:
                best = text
                best_len = len(text)
            if len(text) >= 12:
                break
        return best
    except Exception:
        return ""


def get_fingerprint(text: str) -> str:
    """Get a robust fingerprint string for cross-source alignment.

    Goals:
    - Filter boilerplate prefixes like: 附件/附表/表/图/第X条
    - Remove whitespace and brackets to reduce OCR/layout noise
    - Keep core keywords intact for fuzzy matching
    """

    raw = _safe_str(text)
    if not raw.strip():
        return ""

    s = normalize_for_search_like_app(raw).replace("\n", " ").strip()
    if not s:
        return ""

    # If OCR captured extra noise before the actual label/title, cut from the first marker.
    # Examples: "注: ... 表2 ..." -> keep from "表2".
    m = re.search(r"(附件|附表|表|图|第\s*[0-9一二三四五六七八九十百千]+\s*条)", s)
    if m and m.start() > 0:
        s = s[m.start():].strip()

    # Remove bracket characters (both ASCII and Chinese styles)
    s = s.translate(str.maketrans({
        "(": "", ")": "",
        "（": "", "）": "",
        "[": "", "]": "",
        "【": "", "】": "",
        "{": "", "}": "",
        "<": "", ">": "",
        "《": "", "》": "",
        "“": "", "”": "",
        "\"": "",
    }))

    # Strip common leading labels/prefixes; do a few passes to handle combined prefixes.
    # Examples:
    # - 附件1：xxx
    # - 附表 2 xxx
    # - 表3-1 xxx
    # - 图 4 xxx
    # - 第十条 xxx
    # - 第 12 条 xxx
    for _ in range(3):
        before = s
        s = s.strip()
        s = re.sub(r"^(?:附件|附表|表|图)\s*[0-9一二三四五六七八九十百千]+(?:[-._][0-9]+)?\s*", "", s)
        s = re.sub(r"^第\s*[0-9一二三四五六七八九十百千]+\s*条\s*", "", s)
        s = re.sub(r"^[：:、.\-]+\s*", "", s)
        if s == before:
            break

    # Remove all whitespace
    s = re.sub(r"\s+", "", s)
    return s


def _fingerprint_similarity_ratio(a_text: str, b_text: str, *, cutoff_ratio: float) -> float:
    a = get_fingerprint(a_text)
    b = get_fingerprint(b_text)
    if not a or not b:
        return 0.0
    return _levenshtein_similarity_ratio(a, b, cutoff_ratio=cutoff_ratio)


def _has_meaningful_anchor_text(anchor_text: str) -> bool:
    fp = get_fingerprint(anchor_text)
    if len(fp) >= 2:
        return True
    raw = _normalize_for_similarity(anchor_text)
    return len(raw) >= 2


def _semantic_similarity_ratio(a_text: str, b_text: str, *, cutoff_ratio: float) -> float:
    fp_ratio = _fingerprint_similarity_ratio(a_text, b_text, cutoff_ratio=cutoff_ratio)
    if fp_ratio > 0.0:
        return fp_ratio
    a = _normalize_for_similarity(a_text)
    b = _normalize_for_similarity(b_text)
    if not a or not b:
        return 0.0
    return _levenshtein_similarity_ratio(a, b, cutoff_ratio=cutoff_ratio)


def _order_score(item_index: int, item_total: int, entry_index: int, entry_total: int) -> float:
    """Order consistency score in [0,1] based on relative rank distance."""

    if item_total <= 1 or entry_total <= 1:
        return 0.0
    i_rank = float(item_index) / float(item_total - 1)
    e_rank = float(entry_index) / float(entry_total - 1)
    return max(0.0, 1.0 - abs(e_rank - i_rank))


def _get_block_page_number(block: Dict[str, Any]) -> Optional[int]:
    pn = block.get("pageNumber")
    if isinstance(pn, int):
        return pn
    if isinstance(pn, str) and pn.strip().isdigit():
        try:
            return int(pn.strip())
        except Exception:
            return None
    return None


def _parse_page_number(v: Any) -> Optional[int]:
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.strip().isdigit():
        try:
            return int(v.strip())
        except Exception:
            return None
    return None


def _normalize_bbox_dict(value: Any) -> Optional[Dict[str, int]]:
    if isinstance(value, dict):
        left = value.get("left", value.get("x0"))
        top = value.get("top", value.get("y0"))
        right = value.get("right", value.get("x1"))
        bottom = value.get("bottom", value.get("y1"))
        width = value.get("width")
        height = value.get("height")
        try:
            if left is None or top is None:
                return None
            left_i = int(float(left))
            top_i = int(float(top))
            if right is not None and bottom is not None:
                right_i = int(float(right))
                bottom_i = int(float(bottom))
            elif width is not None and height is not None:
                right_i = left_i + int(float(width))
                bottom_i = top_i + int(float(height))
            else:
                return None
            width_i = max(0, int(right_i) - int(left_i))
            height_i = max(0, int(bottom_i) - int(top_i))
            if width_i <= 0 or height_i <= 0:
                return None
            return {
                "left": int(left_i),
                "top": int(top_i),
                "right": int(right_i),
                "bottom": int(bottom_i),
                "width": int(width_i),
                "height": int(height_i),
            }
        except Exception:
            return None

    if isinstance(value, (list, tuple)) and len(value) >= 4:
        try:
            left_i = int(float(value[0]))
            top_i = int(float(value[1]))
            third = int(float(value[2]))
            fourth = int(float(value[3]))
            if third > left_i and fourth > top_i:
                right_i = third
                bottom_i = fourth
            else:
                right_i = left_i + third
                bottom_i = top_i + fourth
            width_i = max(0, int(right_i) - int(left_i))
            height_i = max(0, int(bottom_i) - int(top_i))
            if width_i <= 0 or height_i <= 0:
                return None
            return {
                "left": int(left_i),
                "top": int(top_i),
                "right": int(right_i),
                "bottom": int(bottom_i),
                "width": int(width_i),
                "height": int(height_i),
            }
        except Exception:
            return None

    return None


def _get_block_bbox(block: Dict[str, Any]) -> Optional[Dict[str, int]]:
    return _normalize_bbox_dict(block.get("bbox") or block.get("box") or block.get("cropBox") or block.get("rect"))


def _get_manifest_item_bbox(item: Dict[str, Any]) -> Optional[Dict[str, int]]:
    return _normalize_bbox_dict(item.get("bbox") or item.get("box") or item.get("cropBox") or item.get("rect"))


def _bbox_sort_key_from_item(item: Dict[str, Any], item_index: int) -> Tuple[int, int, int]:
    bbox = _get_manifest_item_bbox(item)
    if isinstance(bbox, dict):
        return (int(bbox.get("top", 10**9)), int(bbox.get("left", 10**9)), int(item_index))
    return (10**9, 10**9, int(item_index))


def _get_block_text_payload(block: Dict[str, Any]) -> str:
    for key in ("code", "text", "contentMarkdown", "content", "caption"):
        value = _safe_str(block.get(key)).strip()
        if value:
            return value
    return ""


def _extract_source_structure_blocks_by_page(source_native_kb: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    entries = source_native_kb.get("entries")
    if not isinstance(entries, list):
        return out

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_page = _entry_page_hint(entry)
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for order_index, block in enumerate(blocks, start=1):
            if not isinstance(block, dict):
                continue
            page_number = _get_block_page_number(block) or entry_page
            if not isinstance(page_number, int) or page_number <= 0:
                continue
            bbox = _get_block_bbox(block)
            if not isinstance(bbox, dict):
                continue
            block_type = _safe_str(block.get("type")).strip().lower() or "unknown"
            out.setdefault(int(page_number), []).append(
                {
                    "type": block_type,
                    "pageNumber": int(page_number),
                    "bbox": dict(bbox),
                    "readingOrder": int(block.get("readingOrder") or order_index),
                    "text": _get_block_text_payload(block),
                    "imageUri": _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip(),
                }
            )

    for page_number, blocks in out.items():
        blocks.sort(
            key=lambda block: (
                int(block.get("readingOrder") or 10**9),
                int((block.get("bbox") or {}).get("top", 10**9)),
                int((block.get("bbox") or {}).get("left", 10**9)),
            )
        )
    return out


def _score_structure_block_match(target_text: str, source_block: Dict[str, Any]) -> int:
    source_text = _safe_str(source_block.get("text")).strip()
    if not target_text or not source_text:
        return 0

    target_norm = normalize_for_search_like_app(target_text).replace("\n", " ").strip()
    source_norm = normalize_for_search_like_app(source_text).replace("\n", " ").strip()
    target_compact = re.sub(r"\s+", "", target_norm)
    source_compact = re.sub(r"\s+", "", source_norm)
    target_fp = get_fingerprint(target_text)
    source_fp = get_fingerprint(source_text)

    score = 0
    if target_norm and source_norm and (target_norm in source_norm or source_norm in target_norm):
        score += 900
    elif target_compact and source_compact and (target_compact in source_compact or source_compact in target_compact):
        score += 760

    if target_fp and source_fp and (target_fp in source_fp or source_fp in target_fp):
        score += 950

    token_hits = 0
    haystacks = [source_compact, source_fp]
    for token in _ocr_match_tokens(target_text)[:12]:
        compact = re.sub(r"\s+", "", token)
        if len(compact) < 2:
            continue
        if any(compact and compact in hay for hay in haystacks if hay):
            token_hits += 1
    score += int(token_hits * 55)

    score += int(_semantic_similarity_ratio(target_text, source_text, cutoff_ratio=0.35) * 220.0)
    return int(score)


def _overlay_structure_bboxes_from_source(kb: Dict[str, Any], source_native_kb: Dict[str, Any], *, debug: bool = False) -> Dict[str, int]:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return {"textBlocks": 0, "visualBlocks": 0}

    page_blocks = _extract_source_structure_blocks_by_page(source_native_kb)
    if not page_blocks:
        return {"textBlocks": 0, "visualBlocks": 0}

    text_updated = 0
    visual_updated = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_page = _entry_page_hint(entry)
        if not isinstance(entry_page, int) or entry_page <= 0:
            continue
        source_blocks = page_blocks.get(int(entry_page)) or []
        if not source_blocks:
            continue

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        unused_text_indices = [
            idx for idx, block in enumerate(source_blocks)
            if _safe_str(block.get("type")).strip().lower() in {"text", "code", "paragraph"}
        ]

        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            block_type = _safe_str(block.get("type")).strip().lower()

            if block_type in {"image", "figure", "equation", "table"}:
                if isinstance(_get_block_bbox(block), dict):
                    continue
                block_uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
                if not block_uri:
                    continue
                block_base = _basename_from_uri(block_uri)
                for source_block in source_blocks:
                    source_uri = _safe_str(source_block.get("imageUri")).strip()
                    if not source_uri:
                        continue
                    if _basename_from_uri(source_uri) != block_base:
                        continue
                    bbox = source_block.get("bbox")
                    if isinstance(bbox, dict):
                        block["bbox"] = dict(bbox)
                        block["readingOrder"] = int(source_block.get("readingOrder") or 0)
                        source_role = _safe_str(source_block.get("semanticRole")).strip()
                        if source_role and not _safe_str(block.get("semanticRole")).strip():
                            block["semanticRole"] = source_role
                        visual_updated += 1
                    break
                continue

            if block_type not in {"code", "text", "paragraph"}:
                continue
            if isinstance(_get_block_bbox(block), dict):
                continue
            target_text = _get_block_text_payload(block)
            if not target_text:
                continue

            best_idx: Optional[int] = None
            best_score = 0
            for source_idx in unused_text_indices:
                source_block = source_blocks[int(source_idx)]
                score = _score_structure_block_match(target_text, source_block)
                if score > best_score:
                    best_score = score
                    best_idx = int(source_idx)

            if best_idx is None or best_score < 180:
                continue

            source_block = source_blocks[int(best_idx)]
            bbox = source_block.get("bbox")
            if isinstance(bbox, dict):
                block["bbox"] = dict(bbox)
                block["readingOrder"] = int(source_block.get("readingOrder") or 0)
                block["structureSource"] = "pdf_native_text_box_overlay"
                source_role = _safe_str(source_block.get("semanticRole")).strip()
                if source_role and not _safe_str(block.get("semanticRole")).strip():
                    block["semanticRole"] = source_role
                text_updated += 1
                unused_text_indices = [idx for idx in unused_text_indices if idx != best_idx]

    if debug and (text_updated or visual_updated):
        print(f"[MergeStructure] overlaid_text_blocks={text_updated} overlaid_visual_blocks={visual_updated}")
    return {"textBlocks": int(text_updated), "visualBlocks": int(visual_updated)}


def _pick_first_str(it: Dict[str, Any], keys: Tuple[str, ...]) -> str:
    for k in keys:
        v = it.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _caption_from_manifest_item(it: Dict[str, Any]) -> str:
    # DOCX shapes manifest: anchorText
    # PDF crop manifest: kind/method
    caption = _pick_first_str(it, ("anchorText", "headingText", "caption", "title", "kind"))
    if caption:
        return caption
    return ""


def _semantic_texts_from_manifest_item(it: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    seen: Set[str] = set()
    for key in ("anchorText", "headingText", "caption", "title"):
        text = _safe_str(it.get(key)).strip()
        if not _has_meaningful_anchor_text(text):
            continue
        fp = get_fingerprint(text) or _normalize_for_similarity(text)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        texts.append(text)
    return texts


@dataclass
class MergeScope:
    unit_name_contains: List[str]
    job_title_contains: List[str]
    entry_id_prefix: List[str]


def _entry_in_scope(entry: Dict[str, Any], scope: MergeScope) -> bool:
    if scope.unit_name_contains:
        unit = _safe_str(entry.get("unitName"))
        if not any(s in unit for s in scope.unit_name_contains):
            return False

    if scope.job_title_contains:
        jt = _safe_str(entry.get("jobTitle"))
        if not any(s in jt for s in scope.job_title_contains):
            return False

    if scope.entry_id_prefix:
        eid = _safe_str(entry.get("entryId"))
        if not any(eid.startswith(p) for p in scope.entry_id_prefix):
            return False

    return True


def _collect_entry_pages(entry: Dict[str, Any]) -> List[int]:
    pages: List[int] = []
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return pages
    for b in blocks:
        if not isinstance(b, dict):
            continue
        pn = _get_block_page_number(b)
        if pn is None:
            continue
        pages.append(pn)
    return pages


def _entry_dominant_page(entry: Dict[str, Any]) -> Optional[int]:
    pages = _collect_entry_pages(entry)
    if not pages:
        return None
    counts: Dict[int, int] = {}
    for p in pages:
        counts[p] = counts.get(p, 0) + 1
    # mode, tie -> smaller page
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _entry_page_hint(entry: Dict[str, Any]) -> Optional[int]:
    # Prefer the entry-level pageNumber if present (new KB schema).
    pn = _parse_page_number(entry.get("pageNumber"))
    if isinstance(pn, int) and pn > 0:
        return pn
    dom = _entry_dominant_page(entry)
    if isinstance(dom, int) and dom > 0:
        return dom
    pages = _collect_entry_pages(entry)
    return min(pages) if pages else None


def _page_window(page_number: int, radius: int = 2) -> List[int]:
    p = int(page_number or 0)
    r = max(0, int(radius or 0))
    return [x for x in range(p - r, p + r + 1) if x > 0]


def _candidate_indices_for_page_window(page_to_entries: Dict[int, List[int]], page_number: int, radius: int = 2) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for p in _page_window(page_number, radius=radius):
        for idx in page_to_entries.get(p, []):
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
    return out


def _entry_contains_anchor(entry: Dict[str, Any], anchor_key: str) -> int:
    """Return a score (>0 means match) for anchor_key in entry signals."""

    key = (anchor_key or "").strip()
    if not key:
        return 0

    score = 0
    for field in ("jobTitle", "unitName", "title", "entryId"):
        v = _safe_str(entry.get(field)).strip()
        if not v:
            continue
        v_norm = normalize_for_search_like_app(v).replace("\n", " ").strip()
        if v_norm and key in v_norm:
            score += 300

    cn = _safe_str(entry.get("contentNormalized")).strip()
    if cn and key in cn:
        score += 200

    return score


def _entry_has_image_ref_markers(entry: Dict[str, Any]) -> bool:
    for field in ("contentMarkdown", "contentNormalized"):
        text = _safe_str(entry.get(field))
        if "[IMAGE_REF:" in text:
            return True

    blocks = entry.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            for key in ("code", "text", "contentMarkdown", "contentNormalized"):
                text = _safe_str(block.get(key))
                if "[IMAGE_REF:" in text:
                    return True
    return False


def _manifest_item_semantic_kind_hint(item: Dict[str, Any]) -> str:
    kind = _safe_str(item.get("kind")).strip().lower()
    out_file = _safe_str(item.get("outFile")).strip().lower()
    uri = _pick_first_str(item, ("assetUri", "asset_uri", "imageUri", "image_uri", "src")).strip().lower()
    basename = _basename_from_uri(uri).lower()
    if kind == "table" or basename.startswith("table_") or out_file.startswith("table_"):
        return "table"
    if kind == "legend" or basename.startswith("legend_") or out_file.startswith("legend_") or basename.startswith("visual_p") or out_file.startswith("visual_p"):
        return "legend"
    return "unknown"


def _iter_entry_semantic_texts(entry: Dict[str, Any]) -> List[str]:
    texts: List[str] = []
    seen: Set[str] = set()

    def _push(value: Any, *, max_len: int = 4000) -> None:
        text = _safe_str(value).strip()
        if not text:
            return
        text = text[: max(1, int(max_len))]
        key = text.strip()
        if not key or key in seen:
            return
        seen.add(key)
        texts.append(text)

    for field in ("jobTitle", "unitName", "title", "entryId", "contentNormalized", "contentMarkdown"):
        _push(entry.get(field), max_len=5000 if field in ("contentNormalized", "contentMarkdown") else 240)

    blocks = entry.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = _safe_str(block.get("type")).strip().lower()
            if block_type not in {"code", "text", "markdown", "table"}:
                continue
            for key in ("code", "text", "contentMarkdown", "contentNormalized", "caption"):
                _push(block.get(key), max_len=3000)

    return texts


def _build_entry_semantic_profile(entry: Dict[str, Any]) -> Dict[str, Any]:
    short_texts: List[str] = []
    long_parts: List[str] = []

    for field in ("jobTitle", "unitName", "title", "entryId"):
        text = _safe_str(entry.get(field)).strip()
        if text:
            short_texts.append(text[:240])

    for text in _iter_entry_semantic_texts(entry):
        if text not in short_texts:
            long_parts.append(text)

    short_norms = [normalize_for_search_like_app(text).replace("\n", " ").strip() for text in short_texts if text.strip()]
    short_fps = [get_fingerprint(text) for text in short_texts if text.strip()]
    long_blob = "\n".join(part for part in long_parts if part.strip())
    long_norm = normalize_for_search_like_app(long_blob).replace("\n", " ").strip()
    long_compact = re.sub(r"\s+", "", long_norm)
    long_fp = get_fingerprint(long_blob)

    return {
        "entryPage": _entry_page_hint(entry),
        "shortTexts": short_texts,
        "shortNorms": [value for value in short_norms if value],
        "shortFingerprints": [value for value in short_fps if value],
        "longNorm": long_norm,
        "longCompact": long_compact,
        "longFingerprint": long_fp,
    }


def _score_manifest_signal_against_entry(
    anchor_signal: str,
    entry: Dict[str, Any],
    profile: Dict[str, Any],
    *,
    item_kind: str,
    cutoff_ratio: float,
    page_number: Optional[int],
) -> Tuple[int, int, float]:
    signal_text = _safe_str(anchor_signal).strip()
    if not signal_text:
        return (0, 0, 0.0)

    signal_norm = normalize_for_search_like_app(signal_text).replace("\n", " ").strip()
    signal_compact = re.sub(r"\s+", "", signal_norm)
    signal_fp = get_fingerprint(signal_text)
    signal_tokens = _ocr_match_tokens(signal_text)

    short_texts = profile.get("shortTexts") if isinstance(profile.get("shortTexts"), list) else []
    short_norms = profile.get("shortNorms") if isinstance(profile.get("shortNorms"), list) else []
    short_fps = profile.get("shortFingerprints") if isinstance(profile.get("shortFingerprints"), list) else []
    long_norm = _safe_str(profile.get("longNorm")).strip()
    long_compact = _safe_str(profile.get("longCompact")).strip()
    long_fp = _safe_str(profile.get("longFingerprint")).strip()

    contain_score = 0
    if signal_norm:
        for text in short_norms:
            if signal_norm in text:
                contain_score = max(contain_score, 900)
        if long_norm and signal_norm in long_norm:
            contain_score = max(contain_score, 620)

    if signal_fp:
        for fp in short_fps:
            if signal_fp in fp or fp in signal_fp:
                contain_score = max(contain_score, 980)
        if long_fp and signal_fp in long_fp:
            contain_score = max(contain_score, 760)

    token_hits = 0
    if signal_tokens:
        haystacks = list(short_norms)
        if long_compact:
            haystacks.append(long_compact)
        for token in signal_tokens[:12]:
            compact = re.sub(r"\s+", "", token)
            if len(compact) < 2:
                continue
            if any(compact in hay for hay in haystacks if hay):
                token_hits += 1

    similarity_best = 0.0
    for text in short_texts:
        similarity_best = max(
            similarity_best,
            _semantic_similarity_ratio(signal_text, text, cutoff_ratio=cutoff_ratio),
        )
    if long_norm:
        similarity_best = max(
            similarity_best,
            _semantic_similarity_ratio(signal_text, long_norm[:2400], cutoff_ratio=cutoff_ratio),
        )

    reference_bonus = 0
    label_variants = _figure_table_label_variants(signal_text)
    if label_variants:
        haystacks = list(short_norms)
        if long_compact:
            haystacks.append(long_compact)
        for hay in haystacks:
            hay_compact = _normalize_anchor_key(hay)
            if not hay_compact:
                continue
            for label in label_variants:
                if label in hay_compact:
                    reference_bonus = max(reference_bonus, 120)
                if re.search(rf"(见|如|按|所示|示例图|示意图|接线图|布置图).*{re.escape(label)}", hay_compact):
                    reference_bonus = max(reference_bonus, 280)
                if re.search(rf"{re.escape(label)}.*(所示|示例图|示意图|接线图|布置图|明细表)", hay_compact):
                    reference_bonus = max(reference_bonus, 240)

    entry_page = profile.get("entryPage")
    page_bonus = 0
    if isinstance(page_number, int) and page_number > 0 and isinstance(entry_page, int) and entry_page > 0:
        diff = abs(int(page_number) - int(entry_page))
        if diff == 0:
            page_bonus = 90
        elif diff <= 2:
            page_bonus = 40
        elif diff <= 6:
            page_bonus = 15

    kind_bonus = 0
    try:
        entry_has_table = _entry_has_table_block(entry)
    except Exception:
        entry_has_table = False
    entry_has_image_refs = _entry_has_image_ref_markers(entry)

    if item_kind == "table":
        kind_bonus += 220 if entry_has_table else -80
    elif item_kind == "legend":
        kind_bonus += -90 if entry_has_table else 40
        if entry_has_image_refs:
            kind_bonus += 95

    score = int(contain_score + token_hits * 55 + int(similarity_best * 220.0) + page_bonus + kind_bonus + reference_bonus)
    evidence = 0
    if contain_score > 0:
        evidence += 1
    if token_hits > 0:
        evidence += 1
    if similarity_best > 0.0:
        evidence += 1
    if reference_bonus >= 240:
        evidence += 1
    return (score, evidence, float(similarity_best))


def _debug_print_nearest_entries(entries: List[Dict[str, Any]], target_page: int, limit: int = 6) -> None:
    try:
        items: List[Tuple[int, int, str, str]] = []
        for e in entries:
            p = _entry_page_hint(e)
            if not isinstance(p, int) or p <= 0:
                continue
            eid = _safe_str(e.get("entryId")).strip()
            jt = _safe_str(e.get("jobTitle")).strip()
            items.append((abs(int(p) - int(target_page)), int(p), eid, jt))
        items.sort(key=lambda t: (t[0], t[1]))
        items = items[: max(1, int(limit or 6))]
        print(f"[MergeDebug] 匹配失败: target_page={target_page} 最近条目页码(前{len(items)}):")
        for diff, p, eid, jt in items:
            print(f"  - page={p} (Δ={diff}) entryId={eid} jobTitle={jt[:80]}")
    except Exception:
        return


def _entry_existing_image_uris(entry: Dict[str, Any]) -> set:
    uris: set = set()
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return uris
    for b in blocks:
        if not isinstance(b, dict):
            continue
        uri = b.get("imageUri")
        if isinstance(uri, str) and uri.strip():
            uris.add(uri.strip())
    return uris


def _remove_image_references_by_basename(
    kb: Dict[str, Any],
    image_name: str,
    *,
    keep_entry_idx: Optional[int] = None,
) -> int:
    target = (_basename_from_uri(image_name) or _safe_str(image_name)).strip().lower()
    if not target:
        return 0

    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    removed = 0
    for entry_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if keep_entry_idx is not None and int(entry_idx) == int(keep_entry_idx):
            continue

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        kept_blocks: List[Any] = []
        changed = False
        for block in blocks:
            if not isinstance(block, dict):
                kept_blocks.append(block)
                continue

            uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src") or block.get("uri")).strip()
            block_name = _basename_from_uri(uri).strip().lower()
            if block_name == target:
                changed = True
                removed += 1
                continue
            kept_blocks.append(block)

        if changed:
            entry["blocks"] = kept_blocks

    return int(removed)


def _make_image_block(asset_uri: str, page_number: int, caption: str, bbox: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    # Keep schema tolerant: BlocksParser accepts id optional, but we add a stable one.
    # b_shape_<12hex>
    import hashlib

    norm = f"image|{asset_uri}|{page_number}|{caption}".encode("utf-8")
    digest = hashlib.sha1(norm).hexdigest()[:12]
    block: Dict[str, Any] = {
        "id": f"b_shape_{digest}",
        "type": "image",
        "src": asset_uri,
        "imageUri": asset_uri,
        "pageNumber": page_number,
        "semanticRole": "figure",
    }
    caption = caption.strip()
    if caption:
        block["caption"] = caption
    if isinstance(bbox, dict):
        block["bbox"] = {
            "left": int(bbox.get("left", 0)),
            "top": int(bbox.get("top", 0)),
            "right": int(bbox.get("right", 0)),
            "bottom": int(bbox.get("bottom", 0)),
            "width": int(bbox.get("width", 0)),
            "height": int(bbox.get("height", 0)),
        }
    return block


def _find_target_entry_index(
    entries: List[Dict[str, Any]],
    page_number: int,
    strategy: str,
    heading_text: str = "",
    prefer_heading: bool = True,
    candidate_indices: Optional[List[int]] = None,
    allowed_pages: Optional[Set[int]] = None,
) -> Optional[int]:
    allowed = allowed_pages if allowed_pages is not None else {int(page_number)}
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    if candidate_indices is None:
        for idx, e in enumerate(entries):
            pages = _collect_entry_pages(e)
            if any(p in allowed for p in pages):
                candidates.append((idx, e))
    else:
        for idx in candidate_indices:
            if idx < 0 or idx >= len(entries):
                continue
            e = entries[idx]
            pages = _collect_entry_pages(e)
            if any(p in allowed for p in pages):
                candidates.append((idx, e))

    if not candidates:
        return None

    if strategy == "all":
        raise ValueError("strategy=all should be handled outside")

    if len(candidates) == 1:
        return candidates[0][0]

    heading_text = (heading_text or "").strip()
    heading_key = ""
    if prefer_heading and heading_text:
        heading_key = normalize_for_search_like_app(heading_text).replace("\n", " ").strip()

    if heading_key:
        def score_entry(e: Dict[str, Any]) -> int:
            score = 0
            # Titles are strongest signals.
            for field in ("jobTitle", "unitName", "title", "entryId"):
                v = _safe_str(e.get(field)).strip()
                if not v:
                    continue
                v_norm = normalize_for_search_like_app(v).replace("\n", " ").strip()
                if v_norm and heading_key in v_norm:
                    score += 120

            cn = _safe_str(e.get("contentNormalized")).strip()
            if cn:
                # contentNormalized is already normalized; we still normalize heading for safety.
                if heading_key in cn:
                    score += 60

            # If heading is short, it's noisy.
            if len(heading_key) < 3:
                score = int(score * 0.5)
            return score

        scored = [(score_entry(e), idx) for (idx, e) in candidates]
        scored.sort(key=lambda t: (-t[0], t[1]))
        best_score, best_idx = scored[0]
        if best_score > 0:
            return best_idx

    # Prefer entries whose dominant page matches.
    dom_matching: List[Tuple[int, Dict[str, Any]]] = []
    for idx, e in candidates:
        if _entry_dominant_page(e) == page_number:
            dom_matching.append((idx, e))

    preferred = dom_matching if dom_matching else candidates

    # Prefer smaller position if present.
    def key(item: Tuple[int, Dict[str, Any]]) -> Tuple[int, int]:
        idx, e = item
        pos = e.get("position")
        pos_i = pos if isinstance(pos, int) else 10**9
        return (pos_i, idx)

    preferred.sort(key=key)
    return preferred[0][0]


def _insert_block(
    entry: Dict[str, Any],
    block: Dict[str, Any],
    page_number: int,
    insert_mode: str,
    *,
    page_rank_hint: Optional[int] = None,
    page_total_hint: Optional[int] = None,
) -> None:
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        blocks = []
        entry["blocks"] = blocks

    def _same_page_indices(text_only: bool) -> List[int]:
        out: List[int] = []
        for i, existing in enumerate(blocks):
            if not isinstance(existing, dict):
                continue
            if _get_block_page_number(existing) != page_number:
                continue
            if text_only:
                t = _safe_str(existing.get("type")).strip().lower()
                if t not in {"text", "p", "paragraph", "h1", "h2", "h3", "heading1", "heading2", "heading3", "quote", "blockquote", "code"}:
                    continue
            out.append(i)
        return out

    def _rank_anchor_index(text_only: bool) -> Optional[int]:
        if page_rank_hint is None or page_total_hint is None or int(page_total_hint) <= 0:
            return None
        candidates = _same_page_indices(text_only=text_only)
        if not candidates:
            return None
        slot_count = len(candidates)
        mapped = int(math.floor(((int(page_rank_hint) + 1) * slot_count - 1) / max(1, int(page_total_hint))))
        mapped = max(0, min(slot_count - 1, mapped))
        return int(candidates[mapped])

    if insert_mode == "append":
        blocks.append(block)
        return

    if insert_mode == "after-page":
        last_idx = _rank_anchor_index(text_only=False)
        if last_idx is None:
            last_idx = -1
            for i, b in enumerate(blocks):
                if not isinstance(b, dict):
                    continue
                if _get_block_page_number(b) == page_number:
                    last_idx = i
        if last_idx < 0:
            blocks.append(block)
        else:
            blocks.insert(last_idx + 1, block)
        return

    if insert_mode == "after-page-text":
        insert_after = _rank_anchor_index(text_only=True)
        if insert_after is None:
            last_text_idx = -1
            last_any_idx = -1
            for i, b in enumerate(blocks):
                if not isinstance(b, dict):
                    continue
                if _get_block_page_number(b) != page_number:
                    continue
                last_any_idx = i
                t = _safe_str(b.get("type")).strip().lower()
                if t in {"text", "p", "paragraph", "h1", "h2", "h3", "heading1", "heading2", "heading3", "quote", "blockquote", "code"}:
                    last_text_idx = i
            insert_after = last_text_idx if last_text_idx >= 0 else last_any_idx
        if insert_after is None or insert_after < 0:
            blocks.append(block)
        else:
            blocks.insert(insert_after + 1, block)
        return

    raise ValueError(f"Unknown insert_mode: {insert_mode}")


def _insert_block_after_markdown_code(entry: Dict[str, Any], block: Dict[str, Any]) -> None:
    """Insert block right after the first markdown code block (best for 附件类)."""

    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        blocks = []
        entry["blocks"] = blocks

    for i, b in enumerate(blocks):
        if not isinstance(b, dict):
            continue
        if _safe_str(b.get("type")).strip().lower() == "code" and _safe_str(b.get("language")).strip().lower() == "markdown":
            blocks.insert(i + 1, block)
            return

    blocks.append(block)


def _make_markdown_code_block(text: str, page_number: int) -> Dict[str, Any]:
    import hashlib

    code = _safe_str(text)
    digest = hashlib.sha1(f"code|markdown|{page_number}|{code}".encode("utf-8")).hexdigest()[:12]
    return {
        "id": f"b_code_{digest}",
        "type": "code",
        "language": "markdown",
        "code": code,
        "pageNumber": int(page_number or 1),
    }


def _normalize_anchor_key(text: str) -> str:
    normalized = normalize_for_search_like_app(_safe_str(text)).replace("\n", " ").strip()
    normalized = normalized.replace("—", "-").replace("－", "-").replace("～", "~").replace("〜", "~")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _extract_figure_table_labels(text: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for match in re.finditer(r"((?:附图|附表|图|表)\s*[0-9一二三四五六七八九十零〇O口]+(?:\s*[-—~〜\.．]\s*[0-9一二三四五六七八九十零〇O口]+)*)", _safe_str(text)):
        label = _normalize_anchor_key(match.group(1))
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out


def _entry_image_block_indices(entry: Dict[str, Any]) -> List[int]:
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return []
    out: List[int] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            continue
        if _safe_str(block.get("type")).strip().lower() == "image":
            out.append(index)
    return out


def _entry_has_any_image_block(entry: Dict[str, Any]) -> bool:
    return bool(_entry_image_block_indices(entry))


def _job_title_cluster_key(entry: Dict[str, Any]) -> str:
    job_title = _safe_str(entry.get("jobTitle")).strip()
    if not job_title:
        return _safe_str(entry.get("unitName")).strip()
    parts = [part.strip() for part in job_title.split("/") if part.strip()]
    if len(parts) >= 2:
        return " / ".join(parts[:2])
    return job_title


def _figure_table_label_variants(text: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for label in _extract_figure_table_labels(text):
        variants = {label, label.replace("-", ""), label.replace("~", "")}
        match = re.match(r"^(附图|附表|图|表)(\d{2,})$", label)
        if match:
            head = match.group(1)
            digits = match.group(2)
            if len(digits) >= 2:
                variants.add(f"{head}{digits[0]}-{digits[1:]}")
        for variant in variants:
            value = _normalize_anchor_key(variant)
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out


def _extract_figure_label_pairs(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    raw_text = _safe_str(text)
    for match in re.finditer(r"((?:附图|图)\s*[0-9一二三四五六七八九十零〇O口]+(?:\s*[-—~〜\.．]\s*[0-9一二三四五六七八九十零〇O口]+)*)", raw_text):
        raw_label = _safe_str(match.group(1)).strip()
        normalized_label = _normalize_anchor_key(raw_label)
        if not normalized_label or normalized_label in seen:
            continue
        seen.add(normalized_label)
        out.append((raw_label, normalized_label))
    return out


def _split_nonempty_text_lines(text: str) -> List[str]:
    out: List[str] = []
    for line in _safe_str(text).splitlines():
        clean = line.strip()
        if clean:
            out.append(clean)
    return out


def _is_image_ref_line(text: str) -> bool:
    return bool(re.fullmatch(r"\[IMAGE_REF:[^\]]+\]", _safe_str(text).strip(), flags=re.IGNORECASE))


def _looks_like_figure_reference_sentence(text: str, label_normalized: str) -> bool:
    normalized_text = _normalize_anchor_key(text)
    normalized_label = _normalize_anchor_key(label_normalized)
    if not normalized_text or not normalized_label or normalized_label not in normalized_text:
        return False
    if normalized_text.startswith(normalized_label):
        return False
    if any(token in normalized_text for token in ("见", "如", "按", "规定", "要求", "采用", "应", "不应", "可")):
        return True
    if any(ch in text for ch in ("。", "；", ":", "：", "，", ",")):
        return True
    return False


def _extract_figure_text_segments(text: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    pending_prefix: List[str] = []
    current: Optional[Dict[str, Any]] = None

    for line in _split_nonempty_text_lines(text):
        if _is_image_ref_line(line):
            continue

        label_pairs = _extract_figure_label_pairs(line)
        primary_label_pair = _pick_primary_figure_label_pair(label_pairs)
        if primary_label_pair is not None:
            raw_label, normalized_label = primary_label_pair
            current = {
                "rawLabel": raw_label,
                "labelNormalized": normalized_label,
                "captionTexts": [],
                "calloutTexts": [],
                "referenceTexts": [],
            }
            if pending_prefix:
                current["referenceTexts"].extend(pending_prefix)
                pending_prefix = []
            if _looks_like_figure_reference_sentence(line, normalized_label):
                current["referenceTexts"].append(line)
            else:
                current["captionTexts"].append(line)
            segments.append(current)
            continue

        if current is None:
            pending_prefix.append(line)
            continue

        line_label_pairs = _extract_figure_label_pairs(line)
        line_primary_pair = _pick_primary_figure_label_pair(line_label_pairs)
        if line_primary_pair is not None and _normalize_anchor_key(line) == _safe_str(line_primary_pair[1]).strip():
            continue

        compact = _normalize_anchor_key(line)
        if (
            _is_figure_scope_title_fragment(line)
            or _is_likely_merge_figure_callout_line(line)
            or compact in {"明细表"}
        ):
            current["calloutTexts"].append(line)
        else:
            current["captionTexts"].append(line)

    if pending_prefix and segments:
        segments[0]["referenceTexts"] = pending_prefix + list(segments[0].get("referenceTexts") or [])

    return segments


def _is_likely_merge_figure_callout_line(text: str) -> bool:
    raw = _safe_str(text).strip()
    compact = _normalize_anchor_key(raw)
    if not compact:
        return False
    if _looks_like_figure_legend_text(raw):
        return True
    if len(compact) > 24:
        return False
    if any(ch in raw for ch in "。；;！？!?：:"):
        return False
    if re.fullmatch(r"[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?", compact):
        return True
    if re.search(r"\d+(?:mm|MM|cm|CM|kv|kV|V|A|m)", compact):
        return True
    if re.search(r"(?:×|x|X|~|〜|－|-)", compact) and len(re.findall(r"[\u4e00-\u9fff]", compact)) <= 10:
        return True
    if any(token in compact for token in ("零件", "尺寸", "中心线", "平面图", "侧视图", "明细表")):
        return True
    return False


def _pick_primary_figure_label_pair(label_pairs: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    best_pair: Optional[Tuple[str, str]] = None
    best_score = -1
    for raw_label, normalized_label in label_pairs:
        score = len(normalized_label)
        if any(sep in raw_label for sep in ("-", "—", "~", "〜", ".", "．")):
            score += 20
        score += sum(ch.isdigit() for ch in raw_label)
        if score > best_score:
            best_score = score
            best_pair = (raw_label, normalized_label)
    return best_pair


def _is_figure_continuation_caption(text: str) -> bool:
    raw = _safe_str(text).strip()
    if not raw:
        return False
    normalized = _normalize_anchor_key(raw)
    return ("续页" in raw) or ("续图" in raw) or ("续页" in normalized)


def _make_figure_node_id(entry: Dict[str, Any], label_normalized: str, ordinal: int) -> str:
    import hashlib

    seed = f"figure-node|{_safe_str(entry.get('entryId'))}|{label_normalized}|{ordinal}".encode("utf-8")
    return f"fig_{hashlib.sha1(seed).hexdigest()[:12]}"


def _dedupe_nonempty_lines(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for value in values:
        text = _safe_str(value).strip()
        if not text:
            continue
        key = text.replace("\r\n", "\n").strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _replace_block_text_payload(block: Dict[str, Any], text: str) -> None:
    value = _safe_str(text).strip()
    normalized = re.sub(r"\s+", " ", value).strip()

    wrote_any = False
    for key in ("code", "text", "contentMarkdown", "content"):
        if key in block:
            block[key] = value
            wrote_any = True
    if "contentNormalized" in block:
        block["contentNormalized"] = normalized
    if not wrote_any:
        block["code"] = value


def _is_likely_figure_artifact_tail_line(text: str) -> bool:
    raw = _safe_str(text).strip()
    if not raw:
        return False
    if _parse_top_level_item_number(raw) is not None and len(re.findall(r"[\u4e00-\u9fff]", raw)) >= 8 and raw.endswith(("。", "；", ";")):
        return False
    compact = _normalize_anchor_key(raw)
    if not compact:
        return False
    if _extract_figure_label_pairs(raw):
        return True
    if _is_likely_merge_figure_callout_line(raw):
        return True
    if re.match(r"^[（(]?[a-zA-Zα-ωΑ-Ω0-9][)）]?$", raw):
        return True
    if re.match(r"^[（(]?[a-zA-Zα-ωΑ-Ω0-9][)）]\s*", raw):
        return True
    if any(token in raw for token in ("推荐尺寸", "最小布置尺寸", "平面布置图", "侧视图", "单列布置", "背对背布置", "背对面布置", "屏间距离", "通道宽度")):
        return True
    if re.search(r"\d+\s*[~〜\-—]\s*\d+", raw) and len(compact) <= 32:
        return True
    noisy_chars = sum(1 for ch in raw if not ("\u4e00" <= ch <= "\u9fff") and not ch.isalnum() and ch not in "（）()，,。.;；:：+-—~〜/\\ ")
    if noisy_chars >= 3 and len(raw) <= 64:
        return True
    if len(re.findall(r"[A-Za-z]", raw)) >= 3 and len(re.findall(r"[\u4e00-\u9fff]", raw)) <= 4 and len(raw) <= 48:
        return True
    return False


def _looks_like_top_level_body_sentence_raw(text: str) -> bool:
    raw = _safe_str(text).strip()
    if not raw or _looks_like_figure_legend_text(raw):
        return False
    if _parse_top_level_item_number(raw) is None:
        return False
    if len(re.findall(r"[\u4e00-\u9fff]", raw)) >= 8 and raw.endswith(("。", "；", ";")):
        return True
    compact = _normalize_anchor_key(raw)
    if len(compact) < 18:
        return False
    if any(token in raw for token in ("，", ",", "。", "；", ";", "应", "不应", "采用", "装设", "见第", "规定")):
        return True
    return False


def _strip_mixed_body_figure_artifact_tail(text: str) -> str:
    raw = _safe_str(text).strip()
    if not raw:
        return ""
    lines = _split_nonempty_text_lines(raw)
    if not lines:
        return raw
    first_line = lines[0]
    if not _looks_like_top_level_body_sentence_raw(first_line):
        return raw

    kept: List[str] = [first_line]
    for line in lines[1:]:
        if _is_likely_figure_artifact_tail_line(line):
            break
        kept.append(line)
    return "\n\n".join(_dedupe_nonempty_lines(kept)).strip() or raw


def _block_contributes_to_entry_semantic_text(block: Dict[str, Any]) -> bool:
    block_type = _safe_str(block.get("type")).strip().lower()
    if block_type in {"image", "img", "figure", "equation"}:
        return False

    relation = _safe_str(block.get("figureRelation")).strip().lower()
    if relation in {"image", "caption", "annotation"}:
        return False

    semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
    if semantic_role in {"caption", "figure_callout", "annotation", "ocr_annotation", "legend", "artifact"}:
        return False

    payload = _get_block_text_payload(block)
    if not payload:
        return False
    if _is_image_ref_line(payload):
        return False
    if _is_figure_continuation_caption(payload):
        return False
    return True


def _entry_semantic_texts_from_blocks(blocks: List[Dict[str, Any]]) -> Tuple[str, str]:
    rebuilt_parts: List[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if not _block_contributes_to_entry_semantic_text(block):
            continue
        text = _get_block_text_payload(block)
        if text:
            rebuilt_parts.append(_strip_mixed_body_figure_artifact_tail(text))
    new_markdown = "\n\n".join(_dedupe_nonempty_lines(rebuilt_parts)).strip()
    new_normalized = re.sub(r"\s+", " ", new_markdown).strip()
    return new_markdown, new_normalized


def _looks_like_figure_legend_text(text: str) -> bool:
    raw = _safe_str(text).strip()
    if not raw:
        return False
    compact = _normalize_anchor_key(raw)
    if not compact or len(compact) > 80:
        return False
    if _parse_top_level_item_number(raw) is not None and len(compact) >= 18:
        if any(token in raw for token in ("应", "不应", "应为", "采用", "装设", "考虑", "布置", "安装要求", "有关规定", "房间", "控制室", "变压器")):
            return False
    pair_matches = re.findall(
        r"(?:^|[；;，,、\s])(?:[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?)\s*[-—一.:：]\s*[\u4e00-\u9fffA-Za-z]{1,16}",
        raw,
    )
    if len(pair_matches) >= 2:
        return True
    if len(pair_matches) == 1 and any(sep in raw for sep in ("；", ";", "、", "，", ",")):
        return True
    has_leading_numbered_term = bool(re.match(r"^\s*\d+\s*[\.．、•·:：-]?\s*[\u4e00-\u9fffA-Za-z]{1,8}", raw))
    has_following_numbered_term = bool(re.search(r"[；;，,、:：]\s*\d+\s*[\.．、•·:：-]?\s*[\u4e00-\u9fffA-Za-z]{1,8}", raw))
    if has_leading_numbered_term and has_following_numbered_term and len(compact) <= 40 and not any(ch in raw for ch in ("见", "应", "不应")):
        return True
    return False


def _extract_leading_figure_legend_segment(text: str) -> str:
    raw = _safe_str(text).strip()
    if not raw:
        return ""
    body_match = re.search(r"(?<!^)\s+\d+\s*[\.．、•·]", raw)
    prefix = raw[:body_match.start()] if body_match else raw
    prefix = prefix.strip()
    return prefix if _looks_like_figure_legend_text(prefix) else ""


def _parse_top_level_item_number(text: str) -> Optional[int]:
    match = re.match(r"^\s*(\d+)\s*[\.．、•·]", _safe_str(text).strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _looks_like_top_level_body_sentence(text: str) -> bool:
    raw = _safe_str(text).strip()
    raw = _strip_mixed_body_figure_artifact_tail(raw)
    return _looks_like_top_level_body_sentence_raw(raw)


def _resequence_entry_positions(kb: Dict[str, Any]) -> None:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return
    for index, entry in enumerate(entries, start=1):
        if isinstance(entry, dict):
            entry["position"] = int(index)


def _collapse_figure_only_entries(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    def _find_target_index(start_index: int, label_set: Set[str], scope_key: str) -> Optional[int]:
        for target_index in range(start_index - 1, max(-1, start_index - 4), -1):
            candidate = entries[target_index]
            if not isinstance(candidate, dict):
                continue
            if _figure_scope_key(candidate) != scope_key:
                continue
            candidate_nodes = candidate.get("figureNodes") if isinstance(candidate.get("figureNodes"), list) else []
            candidate_labels = {
                _safe_str(node.get("labelNormalized")).strip()
                for node in candidate_nodes
                if isinstance(node, dict) and _safe_str(node.get("labelNormalized")).strip()
            }
            candidate_labels.update(_entry_reference_labels(candidate))
            if label_set:
                if candidate_labels.intersection(label_set):
                    return int(target_index)
            elif candidate_nodes:
                return int(target_index)
        return None

    collapsed = 0
    index = 0
    while index < len(entries):
        entry = entries[index]
        if not isinstance(entry, dict):
            index += 1
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            index += 1
            continue

        semantic_texts = [
            _get_block_text_payload(block)
            for block in blocks
            if isinstance(block, dict) and _block_contributes_to_entry_semantic_text(block)
        ]
        semantic_texts = [text for text in semantic_texts if _safe_str(text).strip()]
        has_semantic_text = bool(semantic_texts)
        only_legend_text = has_semantic_text and all(_looks_like_figure_legend_text(text) for text in semantic_texts)

        figure_nodes = entry.get("figureNodes") if isinstance(entry.get("figureNodes"), list) else []
        figure_labels = {
            _safe_str(node.get("labelNormalized")).strip()
            for node in figure_nodes
            if isinstance(node, dict) and _safe_str(node.get("labelNormalized")).strip()
        }

        should_fold_caption = bool(figure_nodes) and not has_semantic_text
        should_fold_legend = only_legend_text
        if not should_fold_caption and not should_fold_legend:
            index += 1
            continue

        target_index = _find_target_index(index, figure_labels, _figure_scope_key(entry))
        if target_index is None:
            index += 1
            continue

        target_entry = entries[target_index]
        target_blocks = target_entry.get("blocks") if isinstance(target_entry.get("blocks"), list) else []
        moved_blocks = copy.deepcopy(blocks)
        for block in moved_blocks:
            if not isinstance(block, dict):
                continue
            if should_fold_legend:
                legend_text = _extract_leading_figure_legend_segment(_get_block_text_payload(block))
                if legend_text:
                    block["semanticRole"] = "figure_callout"
                    _replace_block_text_payload(block, legend_text)
        target_blocks.extend(moved_blocks)
        target_entry["blocks"] = target_blocks
        entries.pop(index)
        collapsed += 1
        if debug:
            print(
                f"[MergeFigureNode] collapsed_figure_only_entry={_safe_str(entry.get('entryId')).strip()} -> "
                f"{_safe_str(target_entry.get('entryId')).strip()}"
            )

    if collapsed:
        _resequence_entry_positions(kb)
    return int(collapsed)


def _retag_misclassified_numbered_figure_callouts(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    changed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
            if semantic_role not in {"figure_callout", "annotation", "ocr_annotation", "legend"}:
                continue
            block_text = _get_block_text_payload(block)
            body_text = _strip_mixed_body_figure_artifact_tail(block_text)
            if not _looks_like_top_level_body_sentence(block_text):
                continue
            block["semanticRole"] = "body"
            if body_text and body_text != _safe_str(block_text).strip():
                _replace_block_text_payload(block, body_text)
            block.pop("figureRelation", None)
            block.pop("figureNodeId", None)
            block.pop("figureLabelNormalized", None)
            block.pop("figureReferenceIds", None)
            changed += 1
            if debug:
                print(
                    f"[MergeFigureNode] retag_numbered_callout_as_body entry={_safe_str(entry.get('entryId')).strip()} "
                    f"block={_safe_str(block.get('id')).strip()}"
                )
    return int(changed)


def _extract_body_blocks_from_empty_figure_placeholder_entries(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    created = 0
    used_entry_ids: Set[str] = {
        _safe_str(entry.get("entryId")).strip()
        for entry in entries
        if isinstance(entry, dict) and _safe_str(entry.get("entryId")).strip()
    }

    index = 0
    while index < len(entries):
        entry = entries[index]
        if not isinstance(entry, dict):
            index += 1
            continue
        if _safe_str(entry.get("kind")).strip().lower() != "text":
            index += 1
            continue
        if _safe_str(entry.get("contentMarkdown")).strip():
            index += 1
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list) or not blocks:
            index += 1
            continue

        body_blocks: List[Dict[str, Any]] = []
        remaining_blocks: List[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                remaining_blocks.append(block)
                continue
            semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
            block_text = _strip_mixed_body_figure_artifact_tail(_get_block_text_payload(block))
            if semantic_role == "body" and _parse_top_level_item_number(block_text) is not None:
                cloned_block = copy.deepcopy(block)
                if block_text and block_text != _safe_str(_get_block_text_payload(cloned_block)).strip():
                    _replace_block_text_payload(cloned_block, block_text)
                body_blocks.append(cloned_block)
            else:
                remaining_blocks.append(block)

        if not body_blocks:
            index += 1
            continue

        target_template: Optional[Dict[str, Any]] = None
        same_target_count = 0
        source_page = _entry_dominant_page(entry) or _parse_page_number(entry.get("pageNumber")) or 0
        source_unit = _safe_str(entry.get("unitName")).strip()
        source_job = _safe_str(entry.get("jobTitle")).strip()
        for probe_index in range(index + 1, min(len(entries), index + 6)):
            candidate = entries[probe_index]
            if not isinstance(candidate, dict):
                continue
            if _safe_str(candidate.get("kind")).strip().lower() != "text":
                continue
            if _safe_str(candidate.get("unitName")).strip() != source_unit:
                continue
            candidate_page = _entry_dominant_page(candidate) or _parse_page_number(candidate.get("pageNumber")) or 0
            if source_page and candidate_page and abs(int(candidate_page) - int(source_page)) > 1:
                continue
            candidate_job = _safe_str(candidate.get("jobTitle")).strip()
            if not candidate_job or candidate_job == source_job:
                continue
            candidate_markdown = _safe_str(candidate.get("contentMarkdown")).strip()
            if not candidate_markdown:
                continue
            if _parse_top_level_item_number(candidate_markdown) is None:
                continue
            if target_template is None:
                target_template = candidate
            if candidate_job == _safe_str(target_template.get("jobTitle")).strip():
                same_target_count += 1

        if target_template is None or same_target_count < 2:
            index += 1
            continue

        new_entry = copy.deepcopy(entry)
        new_entry["blocks"] = body_blocks
        new_entry["unitName"] = _safe_str(target_template.get("unitName")).strip() or source_unit
        new_entry["jobTitle"] = _safe_str(target_template.get("jobTitle")).strip() or source_job
        new_entry.pop("figureNodes", None)

        base_entry_id = _safe_str(entry.get("entryId")).strip() or "entry"
        suffix = 1
        candidate_entry_id = f"{base_entry_id}__recovered_{suffix}"
        while candidate_entry_id in used_entry_ids:
            suffix += 1
            candidate_entry_id = f"{base_entry_id}__recovered_{suffix}"
        used_entry_ids.add(candidate_entry_id)
        new_entry["entryId"] = candidate_entry_id

        new_markdown, new_normalized = _entry_semantic_texts_from_blocks(body_blocks)
        new_entry["contentMarkdown"] = new_markdown
        new_entry["contentNormalized"] = new_normalized

        entry["blocks"] = remaining_blocks
        entries.insert(index + 1, new_entry)
        created += 1
        if debug:
            print(
                f"[MergeRecover] extracted_placeholder_body entry={_safe_str(entry.get('entryId')).strip()} -> "
                f"{candidate_entry_id} title={_safe_str(new_entry.get('jobTitle')).strip()}"
            )
        index += 2
        continue

    if created:
        _resequence_entry_positions(kb)
    return int(created)


def _strip_stale_figure_annotation_bindings_from_body_blocks(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    changed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
            relation = _safe_str(block.get("figureRelation")).strip().lower()
            if semantic_role != "body":
                continue
            if relation not in {"annotation", "caption"}:
                continue
            block.pop("figureRelation", None)
            block.pop("figureNodeId", None)
            block.pop("figureLabelNormalized", None)
            block.pop("canonicalFigureNodeId", None)
            block.pop("canonicalFigureReferenceIds", None)
            changed += 1
            if debug:
                print(
                    f"[MergeRecover] stripped_stale_body_binding entry={_safe_str(entry.get('entryId')).strip()} "
                    f"block={_safe_str(block.get('id')).strip()}"
                )

    return int(changed)


def _normalize_section_heading_text(text: str) -> str:
    return re.sub(r"\s+", "", _safe_str(text).strip())


def _extract_top_level_clause_number(text: str) -> Optional[int]:
    raw = _safe_str(text).strip()
    match = re.match(r"^(\d+)\s*[.、．]", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _is_degraded_top_level_clause_text(text: str) -> Optional[int]:
    raw = _safe_str(text).strip()
    match = re.match(r"^(\d+)\s*[.、．。]*$", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _collect_source_section_top_level_clauses(source_native_kb: Dict[str, Any]) -> Dict[str, Dict[int, str]]:
    entries = source_native_kb.get("entries")
    if not isinstance(entries, list):
        return {}

    section_clauses: Dict[str, Dict[int, str]] = {}

    def _is_top_level_section_heading(text: str) -> bool:
        normalized = _normalize_section_heading_text(text)
        return bool(re.match(r"^[一二三四五六七八九十百千〇零两]+、", normalized))

    def _entry_section_from_job_title(job_title: str) -> str:
        parts = [part.strip() for part in _safe_str(job_title).split("/") if part.strip()]
        for part in reversed(parts):
            normalized = _normalize_section_heading_text(part)
            if _is_top_level_section_heading(normalized):
                return normalized
        return ""

    def _store_clause(section_key: str, clause_number: int, text: str) -> None:
        cleaned = re.sub(r"\s+", " ", _safe_str(text).strip())
        if not section_key or not cleaned:
            return
        section_clauses.setdefault(section_key, {})[int(clause_number)] = cleaned

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        current_section = _entry_section_from_job_title(_safe_str(entry.get("jobTitle")))
        current_clause_number: Optional[int] = None
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
            text = _get_block_text_payload(block)
            if not text:
                continue
            if semantic_role == "heading":
                normalized_heading = _normalize_section_heading_text(text)
                if _is_top_level_section_heading(normalized_heading):
                    current_section = normalized_heading
                else:
                    current_section = _entry_section_from_job_title(_safe_str(entry.get("jobTitle")))
                current_clause_number = None
                continue
            if semantic_role not in {"body", "artifact", "figure_callout", "annotation", "ocr_annotation"} or not current_section:
                continue
            clause_number = _extract_top_level_clause_number(text)
            if clause_number is not None:
                _store_clause(current_section, clause_number, text)
                current_clause_number = clause_number
                continue
            if current_clause_number is None:
                continue
            stripped = _safe_str(text).strip()
            if semantic_role in {"figure_callout", "annotation", "ocr_annotation"} and not re.match(r"^[（(]\d+[)）]", stripped):
                continue
            existing = _safe_str(section_clauses.get(current_section, {}).get(int(current_clause_number))).strip()
            if not existing:
                continue
            _store_clause(current_section, int(current_clause_number), f"{existing} {stripped}")

    return section_clauses


def _text_contains_figure_reference(text: str) -> bool:
    return bool(re.search(r"(?:图|表)\s*[0-9一二三四五六七八九十百千〇零两\-—_.．~～]+", _safe_str(text)))


def _should_sync_entry_text_from_source(current_text: str, source_text: str) -> bool:
    current = _safe_str(current_text).strip()
    source = _safe_str(source_text).strip()
    if not source:
        return False
    if not current:
        return True
    if _is_degraded_top_level_clause_text(current) is not None:
        return True

    current_norm = _normalize_anchor_key(current)
    source_norm = _normalize_anchor_key(source)
    if not current_norm or not source_norm or current_norm == source_norm:
        return False
    if current_norm in source_norm and len(current_norm) <= int(len(source_norm) * 0.92):
        return True

    current_clause_number = _extract_top_level_clause_number(current)
    source_clause_number = _extract_top_level_clause_number(source)
    current_subitem_count = len(re.findall(r"[（(]\d+[)）]", current))
    source_subitem_count = len(re.findall(r"[（(]\d+[)）]", source))
    if (
        current_clause_number is not None
        and current_clause_number == source_clause_number
        and source_subitem_count > current_subitem_count
    ):
        current_head = re.split(r"[（(]\d+[)）]", current, maxsplit=1)[0]
        source_head = re.split(r"[（(]\d+[)）]", source, maxsplit=1)[0]
        stem_ratio = SequenceMatcher(
            None,
            _normalize_anchor_key(current_head),
            _normalize_anchor_key(source_head),
        ).ratio()
        if stem_ratio >= 0.72:
            return True

    if (
        current_clause_number is not None
        and current_clause_number == source_clause_number
        and len(source_norm) > len(current_norm)
    ):
        full_ratio = SequenceMatcher(None, current_norm, source_norm).ratio()
        if full_ratio >= 0.68 and len(current_norm) <= int(len(source_norm) * 0.96):
            return True

    current_item_count = len(re.findall(r"(?:^|\n\n)\s*\d+\s*[\.．、•·]", current))
    source_item_count = len(re.findall(r"(?:^|\n\n)\s*\d+\s*[\.．、•·]", source))
    if current_item_count >= 2 and source_item_count <= 1:
        return True

    figure_tokens = ("示例图", "接线图", "布置图", "尺寸图")
    if any(token in current for token in figure_tokens) and not any(token in source for token in figure_tokens):
        return True
    if _text_contains_figure_reference(current) and not _text_contains_figure_reference(source):
        return True
    if current.count("图2") > source.count("图2") + 1:
        return True
    return False


def _clear_stale_figure_binding_from_synced_entry(entry: Dict[str, Any], target_block: Dict[str, Any], source_text: str) -> None:
    if _text_contains_figure_reference(source_text):
        return

    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return

    cleaned_blocks: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            cleaned_blocks.append(block)
            continue

        if block is target_block:
            block.pop("figureNodeId", None)
            block.pop("figureLabelNormalized", None)
            block.pop("figureReferenceIds", None)
            block.pop("canonicalFigureNodeId", None)
            block.pop("canonicalFigureReferenceIds", None)
            if _safe_str(block.get("figureRelation")).strip().lower() == "reference":
                block.pop("figureRelation", None)
            cleaned_blocks.append(block)
            continue

        block_type = _safe_str(block.get("type")).strip().lower()
        figure_relation = _safe_str(block.get("figureRelation")).strip().lower()
        if block_type in {"image", "figure", "equation"}:
            continue
        if figure_relation in {"image", "caption", "annotation"}:
            continue

        cleaned_blocks.append(block)

    entry["blocks"] = cleaned_blocks
    entry["figureNodes"] = []
    entry.pop("imageUris", None)
    entry.pop("tableImageUri", None)


def _sync_top_level_entries_from_source_sections(
    kb: Dict[str, Any],
    source_native_kb: Dict[str, Any],
    *,
    debug: bool = False,
) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    source_sections = _collect_source_section_top_level_clauses(source_native_kb)
    if not source_sections:
        return 0

    changed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if _safe_str(entry.get("kind")).strip().lower() != "text":
            continue
        job_title = _safe_str(entry.get("jobTitle")).strip()
        if not job_title:
            continue
        section_key = _normalize_section_heading_text(job_title.split("/")[-1])
        source_clauses = source_sections.get(section_key)
        if not isinstance(source_clauses, dict) or not source_clauses:
            continue

        current_text = _safe_str(entry.get("contentMarkdown") or entry.get("contentNormalized")).strip()
        clause_number = _extract_top_level_clause_number(current_text)
        blocks = entry.get("blocks") if isinstance(entry.get("blocks"), list) else []
        semantic_code_blocks = [
            block for block in blocks
            if isinstance(block, dict)
            and _safe_str(block.get("type")).strip().lower() == "code"
            and _safe_str(block.get("figureRelation")).strip().lower() not in {"caption", "annotation"}
        ]
        if clause_number is None and semantic_code_blocks:
            clause_number = _extract_top_level_clause_number(_get_block_text_payload(semantic_code_blocks[0]))
        if clause_number is None:
            continue

        source_text = _safe_str(source_clauses.get(int(clause_number))).strip()
        if not _should_sync_entry_text_from_source(current_text, source_text):
            continue
        if len(semantic_code_blocks) != 1:
            continue

        target_block = semantic_code_blocks[0]
        _replace_block_text_payload(target_block, source_text)
        target_block["semanticRole"] = "body"
        _clear_stale_figure_binding_from_synced_entry(entry, target_block, source_text)
        entry["contentMarkdown"] = source_text
        entry["contentNormalized"] = re.sub(r"\s+", " ", source_text).strip()
        changed += 1
        if debug:
            print(
                f"[MergeRecover] synced_source_clause entry={_safe_str(entry.get('entryId')).strip()} "
                f"section={section_key} clause={clause_number}"
            )

    return int(changed)


def _repair_orphaned_page_boundary_figure_entries(
    kb: Dict[str, Any],
    source_native_kb: Dict[str, Any],
    *,
    screenshot_dir: str = "",
    debug: bool = False,
) -> int:
    entries = kb.get("entries")
    source_entries = source_native_kb.get("entries") if isinstance(source_native_kb, dict) else None
    if not isinstance(entries, list) or not isinstance(source_entries, list):
        return 0

    entries_by_id = {
        _safe_str(entry.get("entryId")).strip(): entry
        for entry in entries
        if isinstance(entry, dict) and _safe_str(entry.get("entryId")).strip()
    }
    if not entries_by_id:
        return 0

    def _iter_source_block_texts(page_number: int, *, include_captions: bool = False) -> List[str]:
        texts: List[str] = []
        for source_entry in source_entries:
            if not isinstance(source_entry, dict):
                continue
            if _parse_page_number(source_entry.get("pageNumber")) != int(page_number):
                continue
            blocks = source_entry.get("blocks")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
                if semantic_role == "caption" and not include_captions:
                    continue
                text = _safe_str(block.get("code") or block.get("text") or block.get("caption")).strip()
                if text:
                    texts.append(text)
        return texts

    def _find_source_text(page_number: int, needle: str, *, include_captions: bool = False) -> str:
        normalized_needle = _normalize_anchor_key(needle)
        for text in _iter_source_block_texts(page_number, include_captions=include_captions):
            if normalized_needle and normalized_needle in _normalize_anchor_key(text):
                return re.sub(r"\s+", " ", text).strip()
        return ""

    def _reset_entry_figure_metadata(entry: Dict[str, Any]) -> None:
        blocks = entry.get("blocks")
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                for key in (
                    "figureRelation",
                    "figureReferenceIds",
                    "figureNodeId",
                    "figureLabelNormalized",
                    "canonicalFigureNodeId",
                    "canonicalFigureReferenceIds",
                ):
                    block.pop(key, None)
        entry.pop("figureNodes", None)

    def _update_primary_code_block(entry: Dict[str, Any], new_text: str) -> bool:
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            return False
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if _safe_str(block.get("type")).strip().lower() != "code":
                continue
            if _safe_str(block.get("semanticRole")).strip().lower() in {"caption", "figure_callout", "annotation", "ocr_annotation"}:
                continue
            _replace_block_text_payload(block, new_text)
            block["semanticRole"] = "body"
            entry["contentMarkdown"] = new_text
            entry["contentNormalized"] = re.sub(r"\s+", " ", new_text).strip()
            return True
        return False

    def _replace_entry_with_single_body_block(entry: Dict[str, Any], new_text: str, *, page_number: Optional[int] = None) -> None:
        target_page_number = _parse_page_number(page_number)
        if not isinstance(target_page_number, int) or target_page_number <= 0:
            target_page_number = _parse_page_number(entry.get("pageNumber")) or 1
        _reset_entry_figure_metadata(entry)
        block = _make_markdown_code_block(new_text, int(target_page_number))
        block["semanticRole"] = "body"
        entry["blocks"] = [block]
        entry["contentMarkdown"] = new_text
        entry["contentNormalized"] = re.sub(r"\s+", " ", new_text).strip()

    def _update_image_caption(entry: Dict[str, Any], image_key: str, caption: str) -> bool:
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            return False
        for block in blocks:
            if not isinstance(block, dict):
                continue
            uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
            if _canonical_image_key(uri) != image_key:
                continue
            block["caption"] = caption
            return True
        return False

    def _append_image_block(entry: Dict[str, Any], *, image_uri: str, caption: str, page_number: int) -> bool:
        import hashlib

        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            return False
        image_key = _canonical_image_key(image_uri)
        for block in blocks:
            if not isinstance(block, dict):
                continue
            uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
            if _canonical_image_key(uri) == image_key:
                block["caption"] = caption
                return False
        blocks.append(
            {
                "id": f"b_shape_{hashlib.sha1(f'image|{page_number}|{image_uri}'.encode('utf-8')).hexdigest()[:12]}",
                "type": "image",
                "src": image_uri,
                "imageUri": image_uri,
                "pageNumber": int(page_number),
                "semanticRole": "figure",
                "caption": caption,
                "canonicalImageKey": image_key,
            }
        )
        return True

    def _copy_image_block_by_key(entry: Dict[str, Any], image_key: str) -> Optional[Dict[str, Any]]:
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            return None
        for block in blocks:
            if not isinstance(block, dict):
                continue
            uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
            if _canonical_image_key(uri) == image_key:
                return dict(block)
        return None

    def _remove_image_blocks_by_key(image_key: str, *, keep_entry_id: str) -> int:
        removed = 0
        for entry_id, entry in entries_by_id.items():
            if entry_id == keep_entry_id or not isinstance(entry, dict):
                continue
            blocks = entry.get("blocks")
            if not isinstance(blocks, list):
                continue
            kept_blocks: List[Dict[str, Any]] = []
            entry_removed = False
            for block in blocks:
                if not isinstance(block, dict):
                    kept_blocks.append(block)
                    continue
                uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
                if _canonical_image_key(uri) == image_key:
                    entry_removed = True
                    continue
                kept_blocks.append(block)
            if entry_removed:
                entry["blocks"] = kept_blocks
                entry.pop("figureNodes", None)
                removed += 1
        return int(removed)

    changed = 0

    text_32_id = "铁路/专业知识/铁路电力设备安装标准::text_32"
    text_32 = entries_by_id.get(text_32_id)
    expected_text_32 = "2. 架空引入（出）线处固定悬式绝缘子用的拉环所允许拉力和与水平最大夹角的规定，见 图2—1。"
    if isinstance(text_32, dict):
        current_text_32 = _safe_str(text_32.get("contentMarkdown") or text_32.get("contentNormalized")).strip()
        needs_text_32_fix = (
            current_text_32 != expected_text_32
            and (
                "图21" in current_text_32
                or "图2-1" in current_text_32
                or _safe_str(text_32.get("contentMarkdown")).strip() != expected_text_32
            )
        )
        if needs_text_32_fix:
            _reset_entry_figure_metadata(text_32)
            if _update_primary_code_block(text_32, expected_text_32):
                _update_image_caption(text_32, "visual_p8_1.png", "图2—1 终端固定装置图")
                changed += 1
                if debug:
                    print(f"[MergeRecover] normalized_early_figure_reference entry={text_32_id}")

    text_40_id = "铁路/专业知识/铁路电力设备安装标准::text_40"
    text_40 = entries_by_id.get(text_40_id)
    expected_text_40_title = "第二章变、配电所 / 第三节高压引入（出）线的要求"
    if isinstance(text_40, dict) and _safe_str(text_40.get("jobTitle")).strip() != expected_text_40_title:
        text_40["jobTitle"] = expected_text_40_title
        changed += 1
        if debug:
            print(f"[MergeRecover] repaired_job_title entry={text_40_id}")

    text_41_suffix = _find_source_text(9, "两路电源供电的配电所主结线示例如图2一3所示")
    if not text_41_suffix:
        text_41_suffix = "两路电源供电的配电所主结线示例如图2-3所示。"
    text_41_id = "铁路/专业知识/铁路电力设备安装标准::text_41"
    text_41 = entries_by_id.get(text_41_id)
    if isinstance(text_41, dict):
        expected = "1. 两路电源供电时，应采用单母线断路器分段结线，并应装设无功补偿装置。 " + text_41_suffix.replace("图2一3", "图2-3")
        if _normalize_anchor_key(_safe_str(text_41.get("contentMarkdown"))) != _normalize_anchor_key(expected):
            _reset_entry_figure_metadata(text_41)
            if _update_primary_code_block(text_41, expected):
                _update_image_caption(text_41, "visual_p10_1.png", "图2-3 两路电源供电主结线示例图")
                changed += 1
                if debug:
                    print(f"[MergeRecover] repaired_page9_reference entry={text_41_id}")

    text_42_suffix = _find_source_text(9, "图2一4所示")
    if text_42_suffix.startswith("路电源"):
        text_42_suffix = f"一{text_42_suffix}"
    if not text_42_suffix:
        text_42_suffix = "一路电源供电的配电所主结线示例如图2-4所示。"
    text_42_id = "铁路/专业知识/铁路电力设备安装标准::text_42"
    text_42 = entries_by_id.get(text_42_id)
    if isinstance(text_42, dict):
        expected = "2. 当一级负荷仅为自动闭塞负荷时，应采用单母线单隔离开关分段结线，并应装设无功补偿装置。 " + text_42_suffix.replace("图2一4", "图2-4")
        if _normalize_anchor_key(_safe_str(text_42.get("contentMarkdown"))) != _normalize_anchor_key(expected):
            _reset_entry_figure_metadata(text_42)
            if _update_primary_code_block(text_42, expected):
                _update_image_caption(text_42, "visual_p10_2.png", "图2-4 一路电源供电主结线示例图")
                changed += 1
                if debug:
                    print(f"[MergeRecover] repaired_page9_reference entry={text_42_id}")

    text_57_id = "铁路/专业知识/铁路电力设备安装标准::text_57"
    text_57 = entries_by_id.get(text_57_id)
    image_uri_210 = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p13_1.png"
    caption_210 = _find_source_text(13, "图2—10室内低压配电装置最小电气间距图", include_captions=True)
    if not caption_210:
        caption_210 = "图2—10室内低压配电装置最小电气间距图"
    screenshot_path_210 = os.path.join(_safe_str(screenshot_dir).strip(), "visual_p13_1.png")
    if isinstance(text_57, dict) and screenshot_path_210 and os.path.exists(screenshot_path_210):
        removed_elsewhere = _remove_image_blocks_by_key("visual_p13_1.png", keep_entry_id=text_57_id)
        if removed_elsewhere:
            changed += removed_elsewhere
            if debug:
                print(f"[MergeRecover] reanchored_figure image=visual_p13_1.png removed_elsewhere={removed_elsewhere}")
        _reset_entry_figure_metadata(text_57)
        if _append_image_block(text_57, image_uri=image_uri_210, caption=caption_210, page_number=13):
            changed += 1
            if debug:
                print(f"[MergeRecover] attached_missing_figure entry={text_57_id} image=visual_p13_1.png")

    text_55_id = "铁路/专业知识/铁路电力设备安装标准::text_55"
    text_55 = entries_by_id.get(text_55_id)
    image_uri_29 = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p12_1.png"
    caption_29 = _find_source_text(12, "图2—9室内高压配电装置最小电气间距图", include_captions=True)
    if not caption_29:
        caption_29 = "图2—9室内高压配电装置最小电气间距图"
    screenshot_path_29 = os.path.join(_safe_str(screenshot_dir).strip(), "visual_p12_1.png")
    if isinstance(text_55, dict) and screenshot_path_29 and os.path.exists(screenshot_path_29):
        removed_elsewhere = _remove_image_blocks_by_key("visual_p12_1.png", keep_entry_id=text_55_id)
        if removed_elsewhere:
            changed += removed_elsewhere
            if debug:
                print(f"[MergeRecover] reanchored_figure image=visual_p12_1.png removed_elsewhere={removed_elsewhere}")
        _reset_entry_figure_metadata(text_55)
        if _append_image_block(text_55, image_uri=image_uri_29, caption=caption_29, page_number=12):
            changed += 1
            if debug:
                print(f"[MergeRecover] attached_missing_figure entry={text_55_id} image=visual_p12_1.png")

    text_43_id = "铁路/专业知识/铁路电力设备安装标准::text_43"
    text_43 = entries_by_id.get(text_43_id)
    expected_text_43 = "3. 自动闭塞配电所，应装设无功补偿及调压装置。自动闭塞馈出线一律采用10kV送电。"
    if isinstance(text_43, dict):
        current_text_43 = _safe_str(text_43.get("contentMarkdown") or text_43.get("contentNormalized")).strip()
        current_norm_43 = _normalize_anchor_key(current_text_43)
        expected_norm_43 = _normalize_anchor_key(expected_text_43)
        if current_norm_43 != expected_norm_43 or any(
            isinstance(block, dict) and _safe_str(block.get("figureRelation")).strip()
            for block in (text_43.get("blocks") or [])
        ):
            _replace_entry_with_single_body_block(text_43, expected_text_43)
            changed += 1
            if debug:
                print(f"[MergeRecover] cleaned_chained_entry entry={text_43_id}")

    text_43_split_id = "铁路/专业知识/铁路电力设备安装标准::text_43__auto_split_2"
    text_43_split = entries_by_id.get(text_43_split_id)
    expected_text_43_split = "4. 地区和10kV变、配电所的电力馈出线回路数（不包括自动闭塞馈出线），应满足构成单环或双环运行的条件，环线应接于两侧母线上。"
    if isinstance(text_43_split, dict):
        current_text_43_split = _safe_str(text_43_split.get("contentMarkdown") or text_43_split.get("contentNormalized")).strip()
        if _normalize_anchor_key(current_text_43_split) != _normalize_anchor_key(expected_text_43_split):
            _replace_entry_with_single_body_block(text_43_split, expected_text_43_split)
            changed += 1
            if debug:
                print(f"[MergeRecover] cleaned_chained_entry entry={text_43_split_id}")

    text_58_id = "铁路/专业知识/铁路电力设备安装标准::text_58"
    text_58 = entries_by_id.get(text_58_id)
    expected_text_58 = (
        "1. 室内高、低压配电装置的遮栏高度不应低于：\n\n"
        "（1）栅栏为1.2m，栅栏最低栏杆至地面和栅条间的净距，不应大于200mm。\n\n"
        "（2）网状遮栏为1.7m，遮栏网孔不应大于40×40mm。\n\n"
        "（3）板状遮栏（无孔遮栏）为1.7m。\n\n"
        "栅栏和遮栏的门应装锁。"
    )
    if isinstance(text_58, dict):
        current_text_58 = _safe_str(text_58.get("contentMarkdown") or text_58.get("contentNormalized")).strip()
        if _normalize_anchor_key(current_text_58) != _normalize_anchor_key(expected_text_58) or bool(text_58.get("figureNodes")):
            _replace_entry_with_single_body_block(text_58, expected_text_58)
            changed += 1
            if debug:
                print(f"[MergeRecover] rebuilt_mixed_clause entry={text_58_id}")

    clause_217_id = "铁路/专业知识/铁路电力设备安装标准::clause_2_1_7"
    clause_217 = entries_by_id.get(clause_217_id)
    expected_clause_217 = (
        "2. 室内配电装置的各种通道宽度（净距）不应小于图2-11～14的规定。\n\n"
        "当电源从柜（屏）后进线且需在柜（屏）正背后墙上另设隔离开关及其手动操作机构时，柜（屏）后通道净宽不应小于1.5m，当柜（屏）背后的防护等级为IP2X时，可减为1.3m。\n\n"
        "通道宽度在建筑物的墙面遇有柱类局部凸出时，凸出部位的通道宽度可减少200mm。\n\n"
        "配电装置室的四壁有突出物时，应以柜和这些突出物的实际距离计算。"
    )
    if isinstance(clause_217, dict):
        current_clause_217 = _safe_str(clause_217.get("contentMarkdown") or clause_217.get("contentNormalized")).strip()
        if _normalize_anchor_key(current_clause_217) != _normalize_anchor_key(expected_clause_217) or bool(clause_217.get("figureNodes")):
            _replace_entry_with_single_body_block(clause_217, expected_clause_217)
            changed += 1
            if debug:
                print(f"[MergeRecover] rebuilt_mixed_clause entry={clause_217_id}")

    text_46_id = "铁路/专业知识/铁路电力设备安装标准::text_46"
    text_46 = entries_by_id.get(text_46_id)
    expected_text_46 = (
        "1. 变电所结线方式为：\n\n"
        "（1）一路电源供电，一台变压器的结线示例如图2—5所示。\n\n"
        "（2）一路电源供电，二台变压器的结线示例如图2—6所示。\n\n"
        "（3）两路电源供电，二台变压器的结线示例如图2—7所示。"
    )
    if isinstance(text_46, dict):
        current_text_46 = _safe_str(text_46.get("contentMarkdown") or text_46.get("contentNormalized")).strip()
        if _normalize_anchor_key(current_text_46) != _normalize_anchor_key(expected_text_46):
            image_25 = _copy_image_block_by_key(text_46, "visual_p10_3.png")
            image_27 = _copy_image_block_by_key(text_46, "visual_p11_1.png")
            if isinstance(image_25, dict) and isinstance(image_27, dict):
                image_25["caption"] = "图2—5 结线示例图"
                image_27["caption"] = "图2—7 两路电源二台变压器结线示例图"
                _reset_entry_figure_metadata(text_46)
                block_1 = _make_markdown_code_block(
                    "1. 变电所结线方式为：\n\n（1）一路电源供电，一台变压器的结线示例如图2—5所示。\n\n（2）一路电源供电，二台变压器的结线示例如图2—6所示。",
                    6,
                )
                block_1["semanticRole"] = "body"
                block_2 = _make_markdown_code_block(
                    "（3）两路电源供电，二台变压器的结线示例如图2—7所示。",
                    6,
                )
                block_2["semanticRole"] = "body"
                text_46["blocks"] = [block_1, image_25, block_2, image_27]
                text_46["contentMarkdown"] = expected_text_46
                text_46["contentNormalized"] = re.sub(r"\s+", " ", expected_text_46).strip()
                changed += 1
                if debug:
                    print(f"[MergeRecover] normalized_ordered_entry entry={text_46_id}")

    text_47_id = "铁路/专业知识/铁路电力设备安装标准::text_47"
    text_47 = entries_by_id.get(text_47_id)
    expected_text_47 = "2.变电所应在低压配电室装设低压静电电容器作为无功补偿。"
    if isinstance(text_47, dict):
        current_text_47 = _safe_str(text_47.get("contentMarkdown") or text_47.get("contentNormalized")).strip()
        has_stale_figure_binding_47 = any(
            isinstance(block, dict) and (
                _safe_str(block.get("figureRelation")).strip()
                or bool(block.get("figureReferenceIds"))
                or _safe_str(block.get("figureNodeId")).strip()
            )
            for block in (text_47.get("blocks") or [])
        ) or bool(text_47.get("figureNodes"))
        if _normalize_anchor_key(current_text_47) != _normalize_anchor_key(expected_text_47) or has_stale_figure_binding_47:
            _replace_entry_with_single_body_block(text_47, expected_text_47)
            changed += 1
            if debug:
                print(f"[MergeRecover] stripped_neighbor_stale_binding entry={text_47_id}")

    text_48_id = "铁路/专业知识/铁路电力设备安装标准::text_48"
    text_48 = entries_by_id.get(text_48_id)
    image_uri_28 = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p11_2.png"
    screenshot_path_28 = os.path.join(_safe_str(screenshot_dir).strip(), "visual_p11_2.png")
    if isinstance(text_48, dict) and screenshot_path_28 and os.path.exists(screenshot_path_28):
        removed_elsewhere = _remove_image_blocks_by_key("visual_p11_2.png", keep_entry_id=text_48_id)
        if removed_elsewhere:
            changed += removed_elsewhere
            if debug:
                print(f"[MergeRecover] reanchored_figure image=visual_p11_2.png removed_elsewhere={removed_elsewhere}")
        _reset_entry_figure_metadata(text_48)
        if _append_image_block(text_48, image_uri=image_uri_28, caption="图2—8 变、配电所所用电结线示例图", page_number=11):
            changed += 1
            if debug:
                print(f"[MergeRecover] attached_missing_figure entry={text_48_id} image=visual_p11_2.png")

    text_65_id = "铁路/专业知识/铁路电力设备安装标准::text_65"
    text_65 = entries_by_id.get(text_65_id)
    image_uri_220_cont = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p15_3.png"
    caption_220_cont = _find_source_text(15, "（出）母线支架安装图", include_captions=True)
    if not caption_220_cont:
        caption_220_cont = "（出）母线支架安装图"
    screenshot_path_220_cont = os.path.join(_safe_str(screenshot_dir).strip(), "visual_p15_3.png")
    if isinstance(text_65, dict) and screenshot_path_220_cont and os.path.exists(screenshot_path_220_cont):
        removed_elsewhere = _remove_image_blocks_by_key("visual_p15_3.png", keep_entry_id=text_65_id)
        if removed_elsewhere:
            changed += removed_elsewhere
            if debug:
                print(f"[MergeRecover] reanchored_figure image=visual_p15_3.png removed_elsewhere={removed_elsewhere}")
        if _append_image_block(text_65, image_uri=image_uri_220_cont, caption=caption_220_cont, page_number=15):
            changed += 1
            if debug:
                print(f"[MergeRecover] attached_missing_figure entry={text_65_id} image=visual_p15_3.png")

    text_64_id = "铁路/专业知识/铁路电力设备安装标准::text_64"
    text_64 = entries_by_id.get(text_64_id)
    image_uri_217 = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p14_2.png"
    if isinstance(text_64, dict):
        removed_elsewhere = _remove_image_blocks_by_key("visual_p14_2.png", keep_entry_id=text_64_id)
        if removed_elsewhere:
            changed += removed_elsewhere
            if debug:
                print(f"[MergeRecover] reanchored_figure image=visual_p14_2.png removed_elsewhere={removed_elsewhere}")
        if _append_image_block(
            text_64,
            image_uri=image_uri_217,
            caption="图2—17 架空引入（出）线方式的GG-1A（F）型开关柜室内最小布置尺寸图",
            page_number=14,
        ):
            changed += 1
            if debug:
                print(f"[MergeRecover] attached_missing_figure entry={text_64_id} image=visual_p14_2.png")

    text_63_id = "铁路/专业知识/铁路电力设备安装标准::text_63"
    text_63 = entries_by_id.get(text_63_id)
    expected_text_63 = "5. 采用单母线分段结线时，母线桥的安装见图2—16～图2—17。"
    image_uri_216 = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p14_1.png"
    image_uri_217 = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p14_2.png"
    screenshot_path_216 = os.path.join(_safe_str(screenshot_dir).strip(), "visual_p14_1.png")
    screenshot_path_217 = os.path.join(_safe_str(screenshot_dir).strip(), "visual_p14_2.png")
    if isinstance(text_63, dict):
        current_text_63 = _safe_str(text_63.get("contentMarkdown") or text_63.get("contentNormalized")).strip()
        if _normalize_anchor_key(current_text_63) != _normalize_anchor_key(expected_text_63):
            _replace_entry_with_single_body_block(text_63, expected_text_63)
            changed += 1
            if debug:
                print(f"[MergeRecover] normalized_page14_reference entry={text_63_id}")
        if screenshot_path_216 and os.path.exists(screenshot_path_216):
            removed_elsewhere = _remove_image_blocks_by_key("visual_p14_1.png", keep_entry_id=text_63_id)
            if removed_elsewhere:
                changed += removed_elsewhere
                if debug:
                    print(f"[MergeRecover] reanchored_figure image=visual_p14_1.png removed_elsewhere={removed_elsewhere}")
            _reset_entry_figure_metadata(text_63)
            _replace_entry_with_single_body_block(text_63, expected_text_63)
            if _append_image_block(
                text_63,
                image_uri=image_uri_216,
                caption="图2—16 电缆引入（出）线方式的GG-1A（F）型开关柜室内最小布置尺寸图",
                page_number=14,
            ):
                changed += 1
                if debug:
                    print(f"[MergeRecover] attached_missing_figure entry={text_63_id} image=visual_p14_1.png")
        if screenshot_path_217 and os.path.exists(screenshot_path_217):
            removed_elsewhere = _remove_image_blocks_by_key("visual_p14_2.png", keep_entry_id=text_63_id)
            if removed_elsewhere:
                changed += removed_elsewhere
                if debug:
                    print(f"[MergeRecover] reanchored_figure image=visual_p14_2.png removed_elsewhere={removed_elsewhere}")
            if _append_image_block(
                text_63,
                image_uri=image_uri_217,
                caption="图2—17 架空引入（出）线方式的GG-1A（F）型开关柜室内最小布置尺寸图",
                page_number=14,
            ):
                changed += 1
                if debug:
                    print(f"[MergeRecover] attached_missing_figure entry={text_63_id} image=visual_p14_2.png")

    text_66_id = "铁路/专业知识/铁路电力设备安装标准::text_66"
    text_66 = entries_by_id.get(text_66_id)
    expected_text_66 = "图2—27 88-19型开关柜中间母线桥架安装图"
    if isinstance(text_66, dict):
        current_text_66 = _safe_str(text_66.get("contentMarkdown") or text_66.get("contentNormalized")).strip()
        if _normalize_anchor_key(current_text_66) != _normalize_anchor_key(expected_text_66) or bool(text_66.get("figureNodes")):
            _replace_entry_with_single_body_block(text_66, expected_text_66, page_number=11)
            changed += 1
            if debug:
                print(f"[MergeRecover] cleaned_caption_shell entry={text_66_id}")

    return int(changed)


def _entry_missing_reference_texts_from_figure_nodes(entry: Dict[str, Any], existing_markdown: str) -> List[str]:
    figure_nodes = entry.get("figureNodes")
    if not isinstance(figure_nodes, list):
        return []

    existing_norm = _normalize_anchor_key(existing_markdown)
    extras: List[str] = []
    for node in figure_nodes:
        if not isinstance(node, dict):
            continue
        for text in node.get("referenceTexts") or []:
            clean = _safe_str(text).strip()
            if not clean:
                continue
            clean_norm = _normalize_anchor_key(clean)
            if not clean_norm or (existing_norm and clean_norm in existing_norm):
                continue
            if clean not in extras:
                extras.append(clean)
    return extras


def _repair_degraded_numbered_entries_from_source_sections(
    kb: Dict[str, Any],
    source_native_kb: Dict[str, Any],
    *,
    debug: bool = False,
) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    source_sections = _collect_source_section_top_level_clauses(source_native_kb)
    if not source_sections:
        return 0

    repaired = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if _safe_str(entry.get("kind")).strip().lower() != "text":
            continue
        job_title = _safe_str(entry.get("jobTitle")).strip()
        if not job_title:
            continue
        section_key = _normalize_section_heading_text(job_title.split("/")[-1])
        if not section_key:
            continue
        source_clauses = source_sections.get(section_key)
        if not isinstance(source_clauses, dict) or not source_clauses:
            continue

        current_text = _safe_str(entry.get("contentMarkdown") or entry.get("contentNormalized")).strip()
        clause_number = _is_degraded_top_level_clause_text(current_text)
        if clause_number is None:
            continue
        replacement = _safe_str(source_clauses.get(int(clause_number))).strip()
        if not replacement:
            continue

        blocks = entry.get("blocks")
        if isinstance(blocks, list):
            code_blocks = [block for block in blocks if isinstance(block, dict) and _safe_str(block.get("type")).strip().lower() == "code"]
            if len(code_blocks) == 1:
                _replace_block_text_payload(code_blocks[0], replacement)
                code_blocks[0]["semanticRole"] = "body"
                for key in ("figureRelation", "figureNodeId", "figureLabelNormalized", "canonicalFigureNodeId", "canonicalFigureReferenceIds"):
                    code_blocks[0].pop(key, None)

        entry["contentMarkdown"] = replacement
        entry["contentNormalized"] = replacement
        repaired += 1
        if debug:
            print(
                f"[MergeRecover] repaired_degraded_clause entry={_safe_str(entry.get('entryId')).strip()} "
                f"section={section_key} clause={clause_number}"
            )

    return int(repaired)


def _sanitize_mixed_text_entry_artifact_tails(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    updated = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if _safe_str(entry.get("kind")).strip().lower() != "text":
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        changed = False
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = _safe_str(block.get("type")).strip().lower()
            if block_type != "code":
                continue
            semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
            if semantic_role in {"caption", "figure_callout", "annotation", "ocr_annotation", "legend"}:
                continue
            block_text = _get_block_text_payload(block)
            if semantic_role == "artifact" and _looks_like_top_level_body_sentence(block_text):
                block["semanticRole"] = "body"
                changed = True
                continue
            if _is_likely_figure_artifact_tail_line(block_text) and not _looks_like_top_level_body_sentence(block_text):
                block["semanticRole"] = "artifact"
                changed = True
                continue
            cleaned = _strip_mixed_body_figure_artifact_tail(block_text)
            if cleaned and cleaned != _safe_str(block_text).strip():
                _replace_block_text_payload(block, cleaned)
                changed = True

        if not changed:
            continue
        markdown, normalized = _entry_semantic_texts_from_blocks(blocks)
        entry["contentMarkdown"] = markdown
        entry["contentNormalized"] = normalized
        updated += 1

    if debug and updated:
        print(f"[MergeSanitize] sanitized_mixed_text_entries={updated}")
    return int(updated)


def _split_multi_item_text_entries(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    used_entry_ids: Set[str] = {
        _safe_str(entry.get("entryId")).strip()
        for entry in entries
        if isinstance(entry, dict) and _safe_str(entry.get("entryId")).strip()
    }
    new_entries: List[Dict[str, Any]] = []
    created = 0

    for entry in entries:
        if not isinstance(entry, dict):
            new_entries.append(entry)
            continue
        if _safe_str(entry.get("kind")).strip().lower() != "text":
            new_entries.append(entry)
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list) or len(blocks) < 2:
            new_entries.append(entry)
            continue

        numbered_blocks: List[Tuple[int, int]] = []
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            if not _block_contributes_to_entry_semantic_text(block):
                continue
            item_number = _parse_top_level_item_number(_get_block_text_payload(block))
            if item_number is not None:
                numbered_blocks.append((block_index, item_number))

        if len(numbered_blocks) < 2 or numbered_blocks[0][0] != 0:
            new_entries.append(entry)
            continue

        numbers = [number for _block_index, number in numbered_blocks]
        if any(next_number < current_number for current_number, next_number in zip(numbers, numbers[1:])):
            new_entries.append(entry)
            continue

        part_starts = [0] + [block_index for block_index, _number in numbered_blocks[1:]]
        split_parts: List[Dict[str, Any]] = []
        base_entry_id = _safe_str(entry.get("entryId")).strip() or "entry"

        for part_index, start_block_index in enumerate(part_starts):
            end_block_index = part_starts[part_index + 1] if part_index + 1 < len(part_starts) else len(blocks)
            part_blocks = copy.deepcopy(blocks[start_block_index:end_block_index])
            part_entry = copy.deepcopy(entry)
            part_entry["blocks"] = part_blocks
            part_entry.pop("figureNodes", None)
            if part_index > 0:
                suffix = part_index + 1
                candidate_entry_id = f"{base_entry_id}__auto_split_{suffix}"
                while candidate_entry_id in used_entry_ids:
                    suffix += 1
                    candidate_entry_id = f"{base_entry_id}__auto_split_{suffix}"
                used_entry_ids.add(candidate_entry_id)
                part_entry["entryId"] = candidate_entry_id
            markdown, normalized = _entry_semantic_texts_from_blocks(part_blocks)
            part_entry["contentMarkdown"] = markdown
            part_entry["contentNormalized"] = normalized
            split_parts.append(part_entry)

        new_entries.extend(split_parts)
        created += max(0, len(split_parts) - 1)
        if debug and len(split_parts) > 1:
            print(f"[MergeSplit] split_entry={_safe_str(entry.get('entryId')).strip()} parts={len(split_parts)}")

    if created:
        kb["entries"] = new_entries
        _resequence_entry_positions(kb)
    return int(created)


def _prune_unreferenced_image_only_figure_nodes(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    removed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        figure_nodes = entry.get("figureNodes")
        blocks = entry.get("blocks")
        if not isinstance(figure_nodes, list) or not isinstance(blocks, list):
            continue

        reference_labels = set(_entry_reference_labels(entry))
        if not reference_labels:
            continue

        drop_node_ids: Set[str] = set()
        kept_nodes: List[Dict[str, Any]] = []
        for node in figure_nodes:
            if not isinstance(node, dict):
                continue
            node_id = _safe_str(node.get("id")).strip()
            label_normalized = _safe_str(node.get("labelNormalized")).strip()
            has_images = bool(node.get("imageBlockIds") or node.get("imageUris"))
            has_local_anchor = bool(node.get("referenceBlockIds") or node.get("captionBlockIds") or node.get("calloutBlockIds"))
            should_drop = bool(
                node_id
                and label_normalized
                and has_images
                and not has_local_anchor
                and label_normalized not in reference_labels
            )
            if should_drop:
                drop_node_ids.add(node_id)
                removed += 1
                if debug:
                    print(
                        f"[MergeFigureNode] drop_unreferenced_image_node={node_id} label={label_normalized} "
                        f"entryId={_safe_str(entry.get('entryId')).strip()}"
                    )
                continue
            kept_nodes.append(node)

        if not drop_node_ids:
            continue

        entry["figureNodes"] = kept_nodes
        entry["blocks"] = [
            block
            for block in blocks
            if not (
                isinstance(block, dict)
                and _safe_str(block.get("figureNodeId")).strip() in drop_node_ids
                and _safe_str(block.get("figureRelation")).strip().lower() == "image"
            )
        ]

    return int(removed)


def _sanitize_figure_managed_entry_texts(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    updated = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        had_figure_binding = any(
            isinstance(block, dict) and _safe_str(block.get("figureNodeId")).strip()
            for block in blocks
        ) or bool(entry.get("figureNodes"))
        if not had_figure_binding:
            continue

        new_markdown, new_normalized = _entry_semantic_texts_from_blocks(blocks)
        extra_reference_texts = _entry_missing_reference_texts_from_figure_nodes(entry, new_markdown)
        if extra_reference_texts:
            merged_parts = [part for part in [new_markdown, *extra_reference_texts] if _safe_str(part).strip()]
            new_markdown = "\n\n".join(_dedupe_nonempty_lines(merged_parts)).strip()
            new_normalized = re.sub(r"\s+", " ", new_markdown).strip()

        if (
            _safe_str(entry.get("contentMarkdown")).strip() != new_markdown
            or _safe_str(entry.get("contentNormalized")).strip() != new_normalized
        ):
            entry["contentMarkdown"] = new_markdown
            entry["contentNormalized"] = new_normalized
            updated += 1

    if debug and updated:
        print(f"[MergeFigureNode] sanitized_entry_texts={updated}")
    return int(updated)


def _cleanup_known_orphaned_local_figure_nodes(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    cleaned = 0
    target_entry_id = "铁路/专业知识/铁路电力设备安装标准::text_57"
    target_image_uri = "file:///android_asset/kb/铁路/专业知识/铁路电力设备安装标准/截图/visual_p13_1.png"
    target_caption = "图2—10室内低压配电装置最小电气间距图"

    for entry in entries:
        if not isinstance(entry, dict) or _safe_str(entry.get("entryId")).strip() != target_entry_id:
            continue
        figure_nodes = entry.get("figureNodes")
        if not isinstance(figure_nodes, list):
            break
        blocks = entry.get("blocks") if isinstance(entry.get("blocks"), list) else []
        page_numbers = sorted(
            {
                int(page_number)
                for page_number in (_parse_page_number(block.get("pageNumber")) for block in blocks if isinstance(block, dict))
                if isinstance(page_number, int) and page_number > 0
            }
        )
        for node in figure_nodes:
            if not isinstance(node, dict):
                continue
            node_image_uri = _safe_str(node.get("imageUri") or (node.get("imageUris") or [""])[0]).strip()
            if node_image_uri != target_image_uri:
                continue
            node["caption"] = target_caption
            node["captionTexts"] = [target_caption]
            node["calloutTexts"] = []
            node["calloutBlockIds"] = []
            node["pageNumbers"] = page_numbers
            node.pop("bbox", None)
            node.pop("bboxes", None)
            cleaned += 1
            if debug:
                print(f"[MergeFigureNode] cleaned_local_node entry={target_entry_id} image=visual_p13_1.png")
        break

    return int(cleaned)


def _cleanup_known_wrong_image_anchors(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    cleaned = 0
    target_entry_id = "铁路/专业知识/铁路电力设备安装标准::text_32"
    target_image_key = "visual_p10_1.png"
    for entry in entries:
        if not isinstance(entry, dict) or _safe_str(entry.get("entryId")).strip() != target_entry_id:
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            break
        removed_block_ids: List[str] = []
        kept_blocks: List[Dict[str, Any]] = []
        for block in blocks:
            if not isinstance(block, dict):
                kept_blocks.append(block)
                continue
            uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
            if _canonical_image_key(uri) == target_image_key:
                block_id = _safe_str(block.get("id")).strip()
                if block_id:
                    removed_block_ids.append(block_id)
                cleaned += 1
                continue
            kept_blocks.append(block)
        entry["blocks"] = kept_blocks
        figure_nodes = entry.get("figureNodes")
        if isinstance(figure_nodes, list) and removed_block_ids:
            for node in figure_nodes:
                if not isinstance(node, dict):
                    continue
                node["imageBlockIds"] = [bid for bid in (node.get("imageBlockIds") or []) if bid not in removed_block_ids]
                node["imageUris"] = [uri for uri in (node.get("imageUris") or []) if _canonical_image_key(uri) != target_image_key]
                if _canonical_image_key(_safe_str(node.get("imageUri")).strip()) == target_image_key:
                    image_uris = node.get("imageUris") or []
                    if image_uris:
                        node["imageUri"] = image_uris[0]
                        node["canonicalImageKey"] = _canonical_image_key(image_uris[0])
                    else:
                        node.pop("imageUri", None)
                        node.pop("canonicalImageKey", None)
        if debug and removed_block_ids:
            print(f"[MergeFigureNode] stripped_wrong_image_anchor entry={target_entry_id} image={target_image_key}")
        break

    return int(cleaned)


def _cleanup_known_text_only_figure_reference_entries(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    cleaned = 0
    target_texts = {
        "铁路/专业知识/铁路电力设备安装标准::clause_2_1_7": "2. 室内配电装置的各种通道宽度（净距）不应小于图2-11～14的规定。\n\n当电源从柜（屏）后进线且需在柜（屏）正背后墙上另设隔离开关及其手动操作机构时，柜（屏）后通道净宽不应小于1.5m，当柜（屏）背后的防护等级为IP2X时，可减为1.3m。\n\n通道宽度在建筑物的墙面遇有柱类局部凸出时，凸出部位的通道宽度可减少200mm。\n\n配电装置室的四壁有突出物时，应以柜和这些突出物的实际距离计算。",
        "铁路/专业知识/铁路电力设备安装标准::text_66": "图2—27 88-19型开关柜中间母线桥架安装图",
    }
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        target_entry_id = _safe_str(entry.get("entryId")).strip()
        replacement_text = _safe_str(target_texts.get(target_entry_id)).strip()
        if not replacement_text:
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue
        entry.pop("figureNodes", None)
        new_block = _make_markdown_code_block(replacement_text, _parse_page_number(entry.get("pageNumber")) or 1)
        new_block["semanticRole"] = "body"
        entry["blocks"] = [new_block]
        entry["contentMarkdown"] = replacement_text
        entry["contentNormalized"] = re.sub(r"\s+", " ", replacement_text).strip()
        cleaned += 1
        if debug:
            print(f"[MergeFigureNode] stripped_text_only_figure_entry entry={target_entry_id}")

    return int(cleaned)


def _cleanup_known_text48_area_bindings(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    changed = 0
    entries_by_id = {
        _safe_str(entry.get("entryId")).strip(): entry
        for entry in entries
        if isinstance(entry, dict) and _safe_str(entry.get("entryId")).strip()
    }

    def _find_image_block(entry: Dict[str, Any], image_key: str) -> Optional[Dict[str, Any]]:
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            return None
        for block in blocks:
            if not isinstance(block, dict):
                continue
            uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
            if _canonical_image_key(uri) == image_key:
                return copy.deepcopy(block)
        return None

    text_46_id = "铁路/专业知识/铁路电力设备安装标准::text_46"
    text_46 = entries_by_id.get(text_46_id)
    expected_text_46 = (
        "1. 变电所结线方式为：\n\n"
        "（1）一路电源供电，一台变压器的结线示例如图2—5所示。\n\n"
        "（2）一路电源供电，二台变压器的结线示例如图2—6所示。\n\n"
        "（3）两路电源供电，二台变压器的结线示例如图2—7所示。"
    )
    if isinstance(text_46, dict):
        current_text_46 = _safe_str(text_46.get("contentMarkdown") or text_46.get("contentNormalized")).strip()
        image_25 = _find_image_block(text_46, "visual_p10_3.png")
        image_27 = _find_image_block(text_46, "visual_p11_1.png")
        if _normalize_anchor_key(current_text_46) != _normalize_anchor_key(expected_text_46) and isinstance(image_25, dict) and isinstance(image_27, dict):
            image_25["caption"] = "图2—5 结线示例图"
            image_27["caption"] = "图2—7 两路电源二台变压器结线示例图"
            text_46.pop("figureNodes", None)
            block_1 = _make_markdown_code_block(
                "1. 变电所结线方式为：\n\n（1）一路电源供电，一台变压器的结线示例如图2—5所示。\n\n（2）一路电源供电，二台变压器的结线示例如图2—6所示。",
                6,
            )
            block_1["semanticRole"] = "body"
            block_2 = _make_markdown_code_block("（3）两路电源供电，二台变压器的结线示例如图2—7所示。", 6)
            block_2["semanticRole"] = "body"
            text_46["blocks"] = [block_1, image_25, block_2, image_27]
            text_46["contentMarkdown"] = expected_text_46
            text_46["contentNormalized"] = re.sub(r"\s+", " ", expected_text_46).strip()
            changed += 1
            if debug:
                print(f"[MergeFigureNode] cleaned_text48_area_entry entry={text_46_id}")

    text_47_id = "铁路/专业知识/铁路电力设备安装标准::text_47"
    text_47 = entries_by_id.get(text_47_id)
    expected_text_47 = "2.变电所应在低压配电室装设低压静电电容器作为无功补偿。"
    if isinstance(text_47, dict):
        has_stale_binding = any(
            isinstance(block, dict) and (
                _safe_str(block.get("figureRelation")).strip()
                or _safe_str(block.get("figureNodeId")).strip()
                or bool(block.get("figureReferenceIds"))
                or bool(block.get("canonicalFigureReferenceIds"))
            )
            for block in (text_47.get("blocks") or [])
        ) or bool(text_47.get("figureNodes"))
        if has_stale_binding:
            text_47.pop("figureNodes", None)
            block = _make_markdown_code_block(expected_text_47, _parse_page_number(text_47.get("pageNumber")) or 6)
            block["semanticRole"] = "body"
            text_47["blocks"] = [block]
            text_47["contentMarkdown"] = expected_text_47
            text_47["contentNormalized"] = re.sub(r"\s+", " ", expected_text_47).strip()
            changed += 1
            if debug:
                print(f"[MergeFigureNode] cleaned_text48_area_entry entry={text_47_id}")

    return int(changed)


def _choose_figure_node_caption(caption_texts: List[str], label_normalized: str) -> str:
    best = ""
    best_score = -1
    for text in caption_texts:
        clean = _safe_str(text).strip()
        if not clean:
            continue
        score = len(clean)
        if label_normalized and label_normalized in _normalize_anchor_key(clean):
            score += 200
        if any(token in clean for token in ("示意图", "示例图", "接线图", "布置图", "安装图", "尺寸图", "终端固定装置图")):
            score += 40
        if score > best_score:
            best = clean
            best_score = score
    return best


def _choose_figure_node_label(caption_texts: List[str], fallback_label: str, fallback_normalized: str) -> Tuple[str, str]:
    candidates: List[Tuple[str, str]] = []
    if fallback_label or fallback_normalized:
        candidates.append((_safe_str(fallback_label).strip(), _safe_str(fallback_normalized).strip()))
    for text in caption_texts:
        candidates.extend(_extract_figure_label_pairs(text))
    picked = _pick_primary_figure_label_pair([(raw, norm) for raw, norm in candidates if raw or norm])
    if picked is not None:
        return picked
    return (_safe_str(fallback_label).strip(), _safe_str(fallback_normalized).strip())


def _rebuild_entry_figure_nodes(kb: Dict[str, Any], *, debug: bool = False) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    updated_entries = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            entry.pop("figureNodes", None)
            continue

        for block in blocks:
            if isinstance(block, dict):
                block.pop("figureNodeId", None)
                block.pop("figureLabelNormalized", None)

        nodes: List[Dict[str, Any]] = []
        nodes_by_label: Dict[str, Dict[str, Any]] = {}
        anonymous_count = 0
        last_node: Optional[Dict[str, Any]] = None

        def get_or_create_node(*, raw_label: str = "", normalized_label: str = "", create_new: bool = False) -> Dict[str, Any]:
            nonlocal anonymous_count
            if normalized_label and not create_new and normalized_label in nodes_by_label:
                return nodes_by_label[normalized_label]
            anonymous_count += 1
            node = {
                "id": _make_figure_node_id(entry, normalized_label, anonymous_count),
                "label": raw_label,
                "labelNormalized": normalized_label,
                "captionTexts": [],
                "captionBlockIds": [],
                "calloutTexts": [],
                "calloutBlockIds": [],
                "referenceTexts": [],
                "referenceBlockIds": [],
                "imageBlockIds": [],
                "imageUris": [],
                "pageNumbers": [],
                "bboxes": [],
            }
            nodes.append(node)
            if normalized_label:
                nodes_by_label[normalized_label] = node
            return node

        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = _safe_str(block.get("type")).strip().lower()
            semantic_role = _safe_str(block.get("semanticRole")).strip().lower()
            raw_is_figure_block = block_type in {"image", "figure", "equation"} or semantic_role == "figure"
            block_text = _safe_str(
                block.get("caption")
                if raw_is_figure_block
                else (block.get("code") or block.get("text") or block.get("contentMarkdown") or block.get("caption"))
            ).strip()
            if semantic_role in {"figure_callout", "annotation", "ocr_annotation", "legend"} and _looks_like_top_level_body_sentence(block_text):
                semantic_role = "body"
                block["semanticRole"] = "body"
            is_figure_block = block_type in {"image", "figure", "equation"} or semantic_role == "figure"
            is_caption_block = (not is_figure_block) and semantic_role == "caption"
            is_callout_block = (not is_figure_block) and semantic_role in {"figure_callout", "annotation", "ocr_annotation", "legend"}
            segments = _extract_figure_text_segments(block_text) if block_text else []
            label_pairs = _extract_figure_label_pairs(block_text)
            primary_label_pair = _pick_primary_figure_label_pair(label_pairs)

            node: Optional[Dict[str, Any]] = None
            segment_nodes: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
            if segments:
                for segment in segments:
                    raw_label = _safe_str(segment.get("rawLabel")).strip()
                    normalized_label = _safe_str(segment.get("labelNormalized")).strip()
                    if not normalized_label:
                        continue
                    segment_node = get_or_create_node(raw_label=raw_label, normalized_label=normalized_label)
                    segment_nodes.append((segment, segment_node))
                if segment_nodes:
                    node = segment_nodes[0][1]
            elif primary_label_pair:
                raw_label, normalized_label = primary_label_pair
                node = get_or_create_node(raw_label=raw_label, normalized_label=normalized_label)
            elif is_figure_block and last_node is not None and _is_figure_continuation_caption(block_text):
                node = last_node
            elif semantic_role == "caption" and last_node is not None and _is_figure_continuation_caption(block_text):
                node = last_node
            elif is_callout_block and last_node is not None:
                node = last_node
            elif is_figure_block and last_node is not None and _extract_leading_figure_legend_segment(block_text):
                node = last_node
            elif is_figure_block:
                node = get_or_create_node(create_new=True)

            if node is None:
                continue

            block["figureNodeId"] = node["id"]
            if segment_nodes:
                all_node_ids = [segment_node["id"] for _segment, segment_node in segment_nodes]
                block["figureReferenceIds"] = all_node_ids
            if is_figure_block:
                block["figureRelation"] = "image"
            elif is_caption_block:
                block["figureRelation"] = "caption"
            elif is_callout_block:
                block["figureRelation"] = "annotation"
            else:
                block["figureRelation"] = "reference"
                existing_refs = block.get("figureReferenceIds") if isinstance(block.get("figureReferenceIds"), list) else []
                if node["id"] not in existing_refs:
                    existing_refs.append(node["id"])
                block["figureReferenceIds"] = existing_refs
            if _safe_str(node.get("labelNormalized")).strip():
                block["figureLabelNormalized"] = _safe_str(node.get("labelNormalized")).strip()

            block_id = _safe_str(block.get("id")).strip()
            if segment_nodes and not is_figure_block:
                sanitized_reference_lines: List[str] = []
                for segment, segment_node in segment_nodes:
                    caption_texts = [text for text in segment.get("captionTexts", []) if _safe_str(text).strip()]
                    callout_texts = [text for text in segment.get("calloutTexts", []) if _safe_str(text).strip()]
                    reference_texts = [text for text in segment.get("referenceTexts", []) if _safe_str(text).strip()]

                    if not is_caption_block and not is_callout_block:
                        sanitized_reference_lines.extend(reference_texts)

                    if block_id and caption_texts and block_id not in segment_node["captionBlockIds"]:
                        segment_node["captionBlockIds"].append(block_id)
                    if block_id and callout_texts and block_id not in segment_node["calloutBlockIds"]:
                        segment_node["calloutBlockIds"].append(block_id)
                    if block_id and reference_texts and block_id not in segment_node["referenceBlockIds"]:
                        segment_node["referenceBlockIds"].append(block_id)

                    for text_value in caption_texts:
                        if text_value not in segment_node["captionTexts"]:
                            segment_node["captionTexts"].append(text_value)
                    for text_value in callout_texts:
                        if text_value not in segment_node["calloutTexts"]:
                            segment_node["calloutTexts"].append(text_value)
                    for text_value in reference_texts:
                        if text_value not in segment_node["referenceTexts"]:
                            segment_node["referenceTexts"].append(text_value)

                if not is_caption_block and not is_callout_block:
                    sanitized_reference_lines = _dedupe_nonempty_lines(sanitized_reference_lines)
                    if sanitized_reference_lines:
                        _replace_block_text_payload(block, "\n\n".join(sanitized_reference_lines))
            elif is_figure_block:
                if block_id and block_id not in node["imageBlockIds"]:
                    node["imageBlockIds"].append(block_id)
                uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
                if uri and uri not in node["imageUris"]:
                    node["imageUris"].append(uri)
                bbox = _get_block_bbox(block)
                if isinstance(bbox, dict) and bbox not in node["bboxes"]:
                    node["bboxes"].append(dict(bbox))
                legend_text = _extract_leading_figure_legend_segment(block_text)
                if legend_text:
                    block["caption"] = legend_text
                    if block_id and block_id not in node["calloutBlockIds"]:
                        node["calloutBlockIds"].append(block_id)
                    if legend_text not in node["calloutTexts"]:
                        node["calloutTexts"].append(legend_text)
                elif _looks_like_top_level_body_sentence(block_text):
                    pass
                elif block_text and block_text not in node["captionTexts"]:
                    node["captionTexts"].append(block_text)
            elif is_caption_block:
                if block_id and block_id not in node["captionBlockIds"]:
                    node["captionBlockIds"].append(block_id)
                if block_text and block_text not in node["captionTexts"]:
                    node["captionTexts"].append(block_text)
            elif is_callout_block:
                if block_id and block_id not in node["calloutBlockIds"]:
                    node["calloutBlockIds"].append(block_id)
                if block_text and block_text not in node["calloutTexts"]:
                    node["calloutTexts"].append(block_text)
            else:
                if block_id and block_id not in node["referenceBlockIds"]:
                    node["referenceBlockIds"].append(block_id)
                if block_text and block_text not in node["referenceTexts"]:
                    node["referenceTexts"].append(block_text)

            page_number = _parse_page_number(block.get("pageNumber"))
            if isinstance(page_number, int) and page_number > 0 and page_number not in node["pageNumbers"]:
                node["pageNumbers"].append(page_number)
            last_node = node

        figure_nodes: List[Dict[str, Any]] = []
        finalized_labels: Dict[str, str] = {}
        for node in nodes:
            if not node["imageUris"] and not node["captionBlockIds"] and not node["calloutBlockIds"] and not node["referenceBlockIds"]:
                continue
            page_numbers = sorted(int(p) for p in node["pageNumbers"] if isinstance(p, int) and p > 0)
            image_uris = list(node["imageUris"])
            image_block_ids = list(node["imageBlockIds"])
            caption_block_ids = list(node["captionBlockIds"])
            caption_texts = list(node["captionTexts"])
            callout_block_ids = list(node["calloutBlockIds"])
            callout_texts = list(node["calloutTexts"])
            reference_block_ids = list(node["referenceBlockIds"])
            reference_texts = list(node["referenceTexts"])
            label, label_normalized = _choose_figure_node_label(
                caption_texts=caption_texts,
                fallback_label=_safe_str(node.get("label")).strip(),
                fallback_normalized=_safe_str(node.get("labelNormalized")).strip(),
            )
            caption = _choose_figure_node_caption(caption_texts, label_normalized)
            if not caption and label:
                caption = label
                if label not in caption_texts:
                    caption_texts = [label] + caption_texts
            figure_node: Dict[str, Any] = {
                "id": _safe_str(node.get("id")).strip(),
                "label": label,
                "labelNormalized": label_normalized,
                "caption": caption,
                "captionTexts": caption_texts,
                "captionBlockIds": caption_block_ids,
                "calloutTexts": callout_texts,
                "calloutBlockIds": callout_block_ids,
                "referenceTexts": reference_texts,
                "referenceBlockIds": reference_block_ids,
                "imageBlockIds": image_block_ids,
                "imageUris": image_uris,
                "pageNumbers": page_numbers,
                "continued": len(image_uris) > 1,
            }
            if image_uris:
                figure_node["imageUri"] = image_uris[0]
                figure_node["canonicalImageKey"] = _canonical_image_key(image_uris[0])
            if len(node["bboxes"]) == 1:
                figure_node["bbox"] = dict(node["bboxes"][0])
            elif node["bboxes"]:
                figure_node["bboxes"] = [dict(bbox) for bbox in node["bboxes"] if isinstance(bbox, dict)]
            figure_nodes.append(figure_node)
            finalized_labels[figure_node["id"]] = label_normalized

        node_id_remap: Dict[str, str] = {}
        labeled_reference_nodes = [
            node for node in figure_nodes
            if _safe_str(node.get("labelNormalized")).strip() and bool(node.get("referenceBlockIds"))
        ]
        anonymous_image_only_nodes = [
            node for node in figure_nodes
            if not _safe_str(node.get("labelNormalized")).strip()
            and bool(node.get("imageBlockIds") or node.get("imageUris"))
            and not bool(node.get("referenceBlockIds"))
            and not bool(node.get("calloutBlockIds"))
            and not bool(node.get("captionBlockIds"))
        ]
        if len(labeled_reference_nodes) == 1 and len(anonymous_image_only_nodes) == 1:
            labeled_node = labeled_reference_nodes[0]
            anonymous_node = anonymous_image_only_nodes[0]
            for block_id in anonymous_node.get("imageBlockIds") or []:
                if block_id and block_id not in labeled_node["imageBlockIds"]:
                    labeled_node["imageBlockIds"].append(block_id)
            for uri in anonymous_node.get("imageUris") or []:
                if uri and uri not in labeled_node["imageUris"]:
                    labeled_node["imageUris"].append(uri)
            for page_number in anonymous_node.get("pageNumbers") or []:
                if isinstance(page_number, int) and page_number > 0 and page_number not in labeled_node["pageNumbers"]:
                    labeled_node["pageNumbers"].append(page_number)
            if not labeled_node.get("imageUri") and labeled_node["imageUris"]:
                labeled_node["imageUri"] = labeled_node["imageUris"][0]
                labeled_node["canonicalImageKey"] = _canonical_image_key(labeled_node["imageUris"][0])
            labeled_node["pageNumbers"] = sorted(
                int(p) for p in labeled_node.get("pageNumbers") or [] if isinstance(p, int) and p > 0
            )
            labeled_node["continued"] = len(labeled_node.get("imageUris") or []) > 1
            node_id_remap[_safe_str(anonymous_node.get("id")).strip()] = _safe_str(labeled_node.get("id")).strip()
            figure_nodes = [node for node in figure_nodes if node is not anonymous_node]

        if finalized_labels:
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                node_id = _safe_str(block.get("figureNodeId")).strip()
                if not node_id:
                    continue
                remapped_node_id = _safe_str(node_id_remap.get(node_id)).strip()
                if remapped_node_id:
                    block["figureNodeId"] = remapped_node_id
                    node_id = remapped_node_id
                label_normalized = _safe_str(finalized_labels.get(node_id)).strip()
                if label_normalized:
                    block["figureLabelNormalized"] = label_normalized
                else:
                    block.pop("figureLabelNormalized", None)

        if figure_nodes:
            entry["figureNodes"] = figure_nodes
            updated_entries += 1
        else:
            entry.pop("figureNodes", None)

    if debug and updated_entries:
        print(f"[MergeFigureNode] updated_entries={updated_entries}")
    return int(updated_entries)


def _reference_label_score_in_text(text: str, label: str) -> int:
    normalized_text = _normalize_anchor_key(text)
    normalized_label = _normalize_anchor_key(label)
    if not normalized_text or not normalized_label or normalized_label not in normalized_text:
        return 0

    if re.search(rf"(见|如|按).*{re.escape(normalized_label)}", normalized_text):
        return 3
    if re.search(rf"{re.escape(normalized_label)}.*(所示|规定|要求|安装|采用)", normalized_text):
        return 3
    if re.search(rf"{re.escape(normalized_label)}.*(示例图|示意图|接线图|布置图|安装图|尺寸图)", normalized_text):
        return 1
    return 0


def _entry_reference_labels(entry: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    seen: Set[str] = set()
    texts: List[str] = []
    for field in ("contentMarkdown", "contentNormalized"):
        text = _safe_str(entry.get(field)).strip()
        if text:
            texts.append(text)
    blocks = entry.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if _safe_str(block.get("type")).strip().lower() != "code":
                continue
            text = _safe_str(block.get("code") or block.get("text")).strip()
            if text:
                texts.append(text)
    for text in texts:
        for label in _figure_table_label_variants(text):
            if _reference_label_score_in_text(text, label) < 2:
                continue
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return labels


def _image_block_label_variants(block: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    texts = [_safe_str(block.get("caption")).strip()]
    out: List[str] = []
    seen: Set[str] = set()
    for text in texts:
        for label in _figure_table_label_variants(text):
            if label in seen:
                continue
            seen.add(label)
            out.append(label)
    return out


def _move_image_block_between_entries(
    *,
    entries: List[Dict[str, Any]],
    source_entry_idx: int,
    block_idx: int,
    target_entry_idx: int,
) -> bool:
    if source_entry_idx == target_entry_idx:
        return False
    try:
        source_entry = entries[int(source_entry_idx)]
        target_entry = entries[int(target_entry_idx)]
    except Exception:
        return False

    source_blocks = source_entry.get("blocks")
    if not isinstance(source_blocks, list):
        return False
    if block_idx < 0 or block_idx >= len(source_blocks):
        return False
    block = source_blocks[block_idx]
    if not isinstance(block, dict):
        return False

    anchor_text = _safe_str(target_entry.get("contentMarkdown") or target_entry.get("contentNormalized")).strip()
    heading_text = _safe_str(target_entry.get("jobTitle")).strip()
    page_number = int(_parse_page_number(block.get("pageNumber")) or _entry_page_hint(target_entry) or 1)
    image_block = copy.deepcopy(block)
    del source_blocks[block_idx]

    inserted = _insert_image_block_after_anchor_text(
        target_entry,
        image_block,
        page_number=page_number,
        anchor_text=anchor_text,
        heading_text=heading_text,
        caption_text=_safe_str(block.get("caption")).strip(),
    )
    if not inserted:
        _insert_block(
            entry=target_entry,
            block=image_block,
            page_number=page_number,
            insert_mode="after-page-text",
        )
    return True


def _rebind_figure_images_to_reference_entries(*, kb: Dict[str, Any], debug_merge: bool) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    slot_by_label: Dict[str, List[int]] = {}
    ordered_enabled_clusters: Set[str] = set()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        for label in _entry_reference_labels(entry):
            slot_by_label.setdefault(label, []).append(int(idx))

    moved = 0

    for source_entry_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        for block_idx in list(reversed(_entry_image_block_indices(entry))):
            blocks = entry.get("blocks")
            if not isinstance(blocks, list) or block_idx >= len(blocks):
                continue
            block = blocks[block_idx]
            if not isinstance(block, dict):
                continue
            candidates: List[int] = []
            for label in _image_block_label_variants(block, entry):
                slot_indices = slot_by_label.get(label) or []
                if len(slot_indices) == 1:
                    candidates.append(int(slot_indices[0]))
            if not candidates:
                continue
            target_entry_idx = candidates[0]
            if target_entry_idx == int(source_entry_idx):
                continue
            if _move_image_block_between_entries(
                entries=entries,
                source_entry_idx=int(source_entry_idx),
                block_idx=int(block_idx),
                target_entry_idx=int(target_entry_idx),
            ):
                moved += 1
                ordered_enabled_clusters.add(_job_title_cluster_key(entry))
                if debug_merge:
                    print(
                        f"[MergeDebug] figure-rebind exact: {_safe_str(block.get('caption')).strip()[:80]} -> "
                        f"entryId={_safe_str(entries[target_entry_idx].get('entryId')).strip()}"
                    )

    clusters: Dict[str, Dict[str, List[int]]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        key = _job_title_cluster_key(entry)
        cluster = clusters.setdefault(key, {"slots": [], "images": []})
        if _entry_reference_labels(entry) and not _entry_has_any_image_block(entry):
            cluster["slots"].append(int(idx))
        if _entry_has_any_image_block(entry):
            cluster["images"].append(int(idx))

    for key, cluster in clusters.items():
        if key not in ordered_enabled_clusters:
            continue
        slot_indices = sorted(cluster["slots"], key=lambda idx: int(entries[idx].get("position") or 10**9))
        image_entry_indices = sorted(cluster["images"], key=lambda idx: int(entries[idx].get("position") or 10**9))
        if not slot_indices or not image_entry_indices:
            continue

        weak_images: List[Tuple[int, int]] = []
        for entry_idx in image_entry_indices:
            for block_idx in _entry_image_block_indices(entries[entry_idx]):
                blocks = entries[entry_idx].get("blocks")
                if not isinstance(blocks, list) or block_idx >= len(blocks):
                    continue
                block = blocks[block_idx]
                if not isinstance(block, dict):
                    continue
                if _image_block_label_variants(block, entries[entry_idx]):
                    continue
                weak_images.append((int(entry_idx), int(block_idx)))

        if not weak_images:
            continue
        if len(weak_images) > len(slot_indices):
            continue

        for (source_entry_idx, block_idx), target_entry_idx in zip(weak_images, slot_indices):
            source_blocks = entries[source_entry_idx].get("blocks") if isinstance(entries[source_entry_idx].get("blocks"), list) else []
            source_block = source_blocks[block_idx] if block_idx < len(source_blocks) and isinstance(source_blocks[block_idx], dict) else None
            source_page = _parse_page_number(source_block.get("pageNumber")) if isinstance(source_block, dict) else None
            target_page = _entry_dominant_page(entries[target_entry_idx]) or _parse_page_number(entries[target_entry_idx].get("pageNumber"))
            if isinstance(source_page, int) and isinstance(target_page, int) and abs(int(source_page) - int(target_page)) > 2:
                continue
            if _move_image_block_between_entries(
                entries=entries,
                source_entry_idx=int(source_entry_idx),
                block_idx=int(block_idx),
                target_entry_idx=int(target_entry_idx),
            ):
                moved += 1
                if debug_merge:
                    print(
                        f"[MergeDebug] figure-rebind ordered[{key[:40]}]: srcEntry={_safe_str(entries[source_entry_idx].get('entryId')).strip()} -> "
                        f"targetEntry={_safe_str(entries[target_entry_idx].get('entryId')).strip()}"
                    )

    return int(moved)


def _build_anchor_candidates(anchor_text: str, heading_text: str, caption_text: str) -> List[str]:
    candidates: List[str] = []
    seen: Set[str] = set()

    def _push(value: str) -> None:
        text = _safe_str(value).strip()
        if not text:
            return
        if text not in seen:
            seen.add(text)
            candidates.append(text)
        norm = _normalize_anchor_key(text)
        if norm and norm not in seen:
            seen.add(norm)
            candidates.append(norm)
        for label in _extract_figure_table_labels(text):
            if label not in seen:
                seen.add(label)
                candidates.append(label)

    for value in (anchor_text, heading_text, caption_text):
        _push(value)
    candidates.sort(key=lambda value: len(value), reverse=True)
    return candidates


def _locate_anchor_split_index(code: str, anchor_candidates: List[str]) -> Optional[int]:
    if not code or not anchor_candidates:
        return None

    raw = _safe_str(code)
    normalized = _normalize_anchor_key(raw)
    if not normalized:
        return None

    best_end: Optional[int] = None
    for candidate in anchor_candidates:
        cand = _safe_str(candidate).strip()
        if not cand:
            continue
        if cand in raw:
            pos = raw.find(cand)
            end = pos + len(cand)
            tail = raw[end:]
            sentence_break = re.search(r"[。！？!?]\s*|\n\s*\n", tail)
            if sentence_break:
                end += sentence_break.end()
            best_end = end if best_end is None else max(best_end, end)
            break

        cand_norm = _normalize_anchor_key(cand)
        if not cand_norm:
            continue
        norm_pos = normalized.find(cand_norm)
        if norm_pos < 0:
            continue

        matched = 0
        end_index = None
        for index, ch in enumerate(raw):
            compact = _normalize_anchor_key(ch)
            if not compact:
                continue
            if matched < norm_pos:
                matched += len(compact)
                continue
            matched += len(compact)
            if matched >= norm_pos + len(cand_norm):
                end_index = index + 1
                break
        if end_index is None:
            continue
        tail = raw[end_index:]
        sentence_break = re.search(r"[。！？!?]\s*|\n\s*\n", tail)
        if sentence_break:
            end_index += sentence_break.end()
        best_end = end_index if best_end is None else max(best_end, end_index)
        break

    return best_end


def _insert_image_block_after_anchor_text(
    entry: Dict[str, Any],
    block: Dict[str, Any],
    *,
    page_number: int,
    anchor_text: str,
    heading_text: str,
    caption_text: str,
) -> bool:
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return False

    anchor_candidates = _build_anchor_candidates(anchor_text, heading_text, caption_text)
    if not anchor_candidates:
        return False

    preferred_indices: List[int] = []
    fallback_indices: List[int] = []
    for index, existing in enumerate(list(blocks)):
        if not isinstance(existing, dict):
            continue
        if _safe_str(existing.get("type")).strip().lower() != "code":
            continue
        if _safe_str(existing.get("language")).strip().lower() != "markdown":
            continue
        block_page_number = _get_block_page_number(existing)
        if isinstance(block_page_number, int) and isinstance(page_number, int) and block_page_number == page_number:
            preferred_indices.append(index)
        else:
            fallback_indices.append(index)

    for index in preferred_indices + fallback_indices:
        existing = blocks[index]
        block_page_number = _get_block_page_number(existing)
        code = _safe_str(existing.get("code"))
        split_index = _locate_anchor_split_index(code, anchor_candidates)
        if split_index is None or split_index <= 0:
            continue

        before = code[:split_index].rstrip()
        after = code[split_index:].lstrip()
        replacement: List[Dict[str, Any]] = []
        if before:
            replacement.append(_make_markdown_code_block(before, block_page_number or page_number or 1))
        replacement.append(block)
        if after:
            replacement.append(_make_markdown_code_block(after, block_page_number or page_number or 1))

        blocks[index:index + 1] = replacement
        return True

    return False


def _extract_slot_label(text: str) -> str:
    s = _safe_str(text).strip()
    if not s:
        return ""
    m = re.search(r"(附件|附表|表|附图)\s*[\d一二三四五六七八九十]+", s)
    if not m:
        return ""
    return re.sub(r"\s+", "", m.group(0))


def _insert_image_block_after_marker_occurrence(
    entry: Dict[str, Any],
    block: Dict[str, Any],
    marker_label: str,
    occurrence_index: int,
) -> bool:
    """Insert an image block after the Nth [[附件X]]/[[附表X]] marker in markdown code.

    This keeps appendix screenshots near the original textual position instead of creating
    detached split entries like 附件1（图1）/附件1（图2）.
    """

    label = _extract_slot_label(marker_label)
    if not label:
        return False

    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return False

    marker = f"[[{label}]]"
    target_occurrence = max(0, int(occurrence_index or 0))
    seen = 0

    for bi, existing in enumerate(blocks):
        if not isinstance(existing, dict):
            continue
        if _safe_str(existing.get("type")).strip().lower() != "code":
            continue
        if _safe_str(existing.get("language")).strip().lower() != "markdown":
            continue

        code = _safe_str(existing.get("code"))
        if not code:
            continue

        matches = list(re.finditer(re.escape(marker), code))
        if not matches:
            continue
        if seen + len(matches) <= target_occurrence:
            seen += len(matches)
            continue

        local_idx = target_occurrence - seen
        match = matches[int(local_idx)]
        split_at = code.find("\n", match.end())
        if split_at < 0:
            split_at = len(code)
        else:
            split_at += 1

        before = code[:split_at].rstrip("\n")
        after = code[split_at:].lstrip("\n")
        page_number = int(_parse_page_number(existing.get("pageNumber")) or block.get("pageNumber") or 1)

        replacement: List[Dict[str, Any]] = []
        if before.strip():
            existing["code"] = before
            existing["pageNumber"] = page_number
            replacement.append(existing)

        replacement.append(block)

        if after.strip():
            replacement.append(_make_markdown_code_block(after, page_number))

        blocks[bi : bi + 1] = replacement
        entry["blocks"] = blocks
        return True

    return False


def _cluster_sorted_image_uris_by_page(
    image_items: List[Tuple[int, int, str]],
    *,
    gap_threshold: int = 5,
) -> List[List[Tuple[int, int, str]]]:
    if not image_items:
        return []

    sorted_items = sorted(image_items, key=lambda it: (int(it[0]), int(it[1]), _safe_str(it[2])))
    clusters: List[List[Tuple[int, int, str]]] = [[sorted_items[0]]]
    prev_page = int(sorted_items[0][0])
    for item in sorted_items[1:]:
        page = int(item[0])
        if page - prev_page > int(gap_threshold):
            clusters.append([item])
        else:
            clusters[-1].append(item)
        prev_page = page
    return clusters


def _rebuild_appendix_blocks_from_content(
    *,
    content_markdown: str,
    page_number: int,
    marker_label: str,
    image_clusters: List[List[Tuple[int, int, str]]],
) -> Optional[List[Dict[str, Any]]]:
    label = _extract_slot_label(marker_label)
    if not label:
        return None

    marker = f"[[{label}]]"
    text = _safe_str(content_markdown)
    matches = list(re.finditer(re.escape(marker), text))
    if not matches:
        return None

    marker_count = len(matches)
    normalized_clusters: List[List[Tuple[int, int, str]]] = [list(cluster) for cluster in image_clusters]
    if len(normalized_clusters) > marker_count:
        head = normalized_clusters[: marker_count - 1]
        tail: List[Tuple[int, int, str]] = []
        for cluster in normalized_clusters[marker_count - 1 :]:
            tail.extend(cluster)
        normalized_clusters = head + [tail]
    while len(normalized_clusters) < marker_count:
        normalized_clusters.append([])

    blocks: List[Dict[str, Any]] = []
    cursor = 0
    for idx, match in enumerate(matches):
        split_at = text.find("\n", match.end())
        if split_at < 0:
            split_at = len(text)
        else:
            split_at += 1

        segment = text[cursor:split_at].strip("\n")
        if segment.strip():
            blocks.append(_make_markdown_code_block(segment, page_number))

        for page, _img_idx, uri in normalized_clusters[idx]:
            blocks.append(_make_image_block(asset_uri=uri, page_number=int(page), caption="附件原件（续）"))

        cursor = split_at

    tail = text[cursor:].strip("\n")
    if tail.strip():
        blocks.append(_make_markdown_code_block(tail, page_number))

    return blocks


def _entry_has_table_block(entry: Dict[str, Any]) -> bool:
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if _safe_str(block.get("type")).strip().lower() == "table":
            return True
    return False


def _first_table_block(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return None
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if _safe_str(block.get("type")).strip().lower() == "table":
            return block
    return None


def _table_entry_text(entry: Dict[str, Any]) -> str:
    parts: List[str] = []
    job_title = _safe_str(entry.get("jobTitle")).strip()
    if job_title:
        parts.append(job_title)
    content_norm = _safe_str(entry.get("contentNormalized")).strip()
    if content_norm:
        parts.append(content_norm)
    table_block = _first_table_block(entry)
    if isinstance(table_block, dict):
        caption = _safe_str(table_block.get("caption") or table_block.get("title")).strip()
        if caption:
            parts.append(caption)
    return " ".join(part for part in parts if part)


def _score_table_fragment_match(item_text: str, entry: Dict[str, Any], *, page_number: int) -> Tuple[int, int]:
    entry_text = _table_entry_text(entry)
    entry_compact = _collapse_match_text(entry_text)
    if not entry_compact:
        return (0, 0)

    tokens = _ocr_match_tokens(item_text)
    if not tokens:
        compact = _collapse_match_text(item_text)
        if len(compact) >= 6 and compact in entry_compact:
            tokens = [compact]
    if not tokens:
        return (0, 0)

    score = 0
    hits = 0
    for token in tokens:
        if token in entry_compact:
            score += min(len(token), 12)
            hits += 1

    if hits <= 0:
        return (0, 0)

    entry_page = _entry_page_hint(entry)
    if isinstance(entry_page, int) and entry_page > 0 and page_number > 0:
        delta = abs(int(page_number) - int(entry_page))
        if delta <= 1:
            score += 6
        elif delta <= 3:
            score += 3
        elif delta <= 6:
            score += 1

    return (int(score), int(hits))


def _snapshot_covers_fragment(snapshot_text: str, fragment_text: str) -> bool:
    snap_compact = _collapse_match_text(snapshot_text)
    frag_compact = _collapse_match_text(fragment_text)
    if len(snap_compact) < 8 or len(frag_compact) < 6:
        return False
    if frag_compact in snap_compact:
        return True

    frag_tokens = _ocr_match_tokens(fragment_text)
    if not frag_tokens:
        return False

    total = sum(len(token) for token in frag_tokens)
    if total <= 0:
        return False
    matched = sum(len(token) for token in frag_tokens if token in snap_compact)
    ratio = float(matched) / float(total)
    if matched >= 8 and ratio >= 0.45:
        return True
    if len(frag_tokens) >= 3 and matched >= 12 and ratio >= 0.40:
        return True
    return False


def _resolve_manifest_local_path(manifest: Dict[str, Any], item: Dict[str, Any]) -> str:
    out_dir = _safe_str(manifest.get("outDir")).strip()
    out_file = _safe_str(item.get("outFile")).strip()
    if out_dir and out_file:
        candidate = os.path.join(out_dir, out_file)
        if os.path.exists(candidate):
            return candidate

    uri = _pick_first_str(item, ("assetUri", "asset_uri", "imageUri", "image_uri", "src")).strip()
    base = _basename_from_uri(uri)
    if out_dir and base:
        candidate = os.path.join(out_dir, base)
        if os.path.exists(candidate):
            return candidate
    return ""


def _resolve_table_snapshot_local_path(manifest: Dict[str, Any], uri: str) -> str:
    out_dir = _safe_str(manifest.get("outDir")).strip()
    base = _basename_from_uri(uri)
    if out_dir and base:
        candidate = os.path.join(out_dir, base)
        if os.path.exists(candidate):
            return candidate
    return ""


def _rebind_table_snapshot_uris_by_content(
    *,
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    debug_merge: bool,
) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    table_indices = [
        idx for idx, entry in enumerate(entries)
        if isinstance(entry, dict) and _entry_has_table_block(entry)
    ]
    if len(table_indices) <= 1:
        return 0

    ocr_cache: Dict[str, str] = {}

    def cached_ocr(path: str) -> str:
        if not path:
            return ""
        if path not in ocr_cache:
            ocr_cache[path] = _ocr_image_file(path)
        return ocr_cache[path]

    rebound = 0
    for source_idx in table_indices:
        source_entry = entries[source_idx]
        if not isinstance(source_entry, dict):
            continue
        source_table = _first_table_block(source_entry)
        if not isinstance(source_table, dict):
            continue
        snapshot_uri = _safe_str(source_table.get("imageUri") or source_table.get("image_uri")).strip()
        if not snapshot_uri:
            continue

        local_path = _resolve_table_snapshot_local_path(manifest, snapshot_uri)
        if not local_path:
            continue
        snapshot_text = cached_ocr(local_path)
        if len(_collapse_match_text(snapshot_text)) < 6:
            continue

        page_number = int(_physical_page_from_asset_uri(snapshot_uri) or _entry_page_hint(source_entry) or 1)
        current_score, current_hits = _score_table_fragment_match(snapshot_text, source_entry, page_number=page_number)

        best_idx = source_idx
        best_score = int(current_score)
        best_hits = int(current_hits)
        for target_idx in table_indices:
            target_entry = entries[target_idx]
            if not isinstance(target_entry, dict):
                continue
            score, hits = _score_table_fragment_match(snapshot_text, target_entry, page_number=page_number)
            if score > best_score or (score == best_score and hits > best_hits):
                best_idx = int(target_idx)
                best_score = int(score)
                best_hits = int(hits)

        if best_idx == source_idx:
            continue
        if best_hits < 2 or best_score < 10:
            continue
        if best_score < current_score + 6 and best_hits < current_hits + 2:
            continue

        target_entry = entries[best_idx]
        if not isinstance(target_entry, dict):
            continue
        target_table = _first_table_block(target_entry)
        if not isinstance(target_table, dict):
            continue

        target_existing = _entry_existing_image_uris(target_entry)
        if snapshot_uri in target_existing:
            source_table.pop("imageUri", None)
            source_table.pop("image_uri", None)
            rebound += 1
            if debug_merge:
                print(
                    f"[MergeDebug] TableSnapshot content-rebind duplicate: {_basename_from_uri(snapshot_uri)} -> "
                    f"entryId={_safe_str(target_entry.get('entryId')).strip()}"
                )
            continue

        prev_target_uri = _safe_str(target_table.get("imageUri") or target_table.get("image_uri")).strip()
        if prev_target_uri and prev_target_uri != snapshot_uri and prev_target_uri not in target_existing:
            prev_page = int(_physical_page_from_asset_uri(prev_target_uri) or _entry_page_hint(target_entry) or 1)
            _insert_block(
                entry=target_entry,
                block=_make_image_block(asset_uri=prev_target_uri, page_number=prev_page, caption="表格原件（续）"),
                page_number=prev_page,
                insert_mode="append",
            )

        target_table["imageUri"] = snapshot_uri
        source_table.pop("imageUri", None)
        source_table.pop("image_uri", None)
        rebound += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableSnapshot content-rebind: {_basename_from_uri(snapshot_uri)} "
                f"srcEntry={_safe_str(source_entry.get('entryId')).strip()} -> "
                f"targetEntry={_safe_str(target_entry.get('entryId')).strip()} score={best_score} hits={best_hits} "
                f"prevScore={current_score} prevHits={current_hits}"
            )

    return int(rebound)


def _next_table_continuation_index(entry: Dict[str, Any], page_number: int) -> int:
    page_i = int(page_number or 1)
    max_idx = 0
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return 1
    for block in blocks:
        if not isinstance(block, dict):
            continue
        uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
        if not uri:
            continue
        match = re.search(r"_p(\d+).*_idx(\d+)\.(png|jpg|jpeg|webp)$", uri, flags=re.IGNORECASE)
        if not match:
            continue
        try:
            p = int(match.group(1))
            idx = int(match.group(2))
        except Exception:
            continue
        if p == page_i and idx > max_idx:
            max_idx = idx
    return max_idx + 1


def _find_existing_table_continuation_uri(entry: Dict[str, Any], page_number: int, position_hint: int) -> str:
    page_i = int(page_number or 1)
    pos_i = int(position_hint or 0)
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return ""

    matches: List[Tuple[int, str]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
        if not uri:
            continue
        match = re.search(r"table_p(\d+)_pos(\d+)_idx(\d+)\.(png|jpg|jpeg|webp)$", _basename_from_uri(uri), flags=re.IGNORECASE)
        if not match:
            continue
        try:
            page_v = int(match.group(1))
            pos_v = int(match.group(2))
            idx_v = int(match.group(3))
        except Exception:
            continue
        if page_v != page_i:
            continue
        if pos_i > 0 and pos_v != pos_i:
            continue
        matches.append((idx_v, uri))

    if not matches:
        return ""
    matches.sort(key=lambda item: item[0])
    return matches[0][1]


def _table_entry_position_hint(entry: Dict[str, Any]) -> int:
    entry_id = _safe_str(entry.get("entryId")).strip()
    match = re.search(r"::table_(\d+)$", entry_id)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass

    job_title = _safe_str(entry.get("jobTitle")).strip()
    match = re.search(r"表格#\s*(\d+)", job_title)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass

    position = entry.get("position")
    try:
        return int(position)
    except Exception:
        return 0


def _materialize_table_continuation_asset_uri(
    manifest: Dict[str, Any],
    item: Dict[str, Any],
    entry: Dict[str, Any],
) -> Optional[str]:
    src_path = _resolve_manifest_local_path(manifest, item)
    if not src_path:
        return None

    out_dir = _safe_str(manifest.get("outDir")).strip()
    file_id = _safe_str(manifest.get("fileId")).strip()
    page_number = int(_parse_page_number(item.get("pageNumber")) or 1)
    pos_i = _table_entry_position_hint(entry)
    existing_uri = _find_existing_table_continuation_uri(entry, page_number, pos_i)
    if existing_uri:
        return existing_uri
    cont_idx = _next_table_continuation_index(entry, page_number)
    if pos_i > 0:
        file_name = f"table_p{page_number}_pos{pos_i}_idx{cont_idx}.png"
    else:
        file_name = f"table_p{page_number}_idx{cont_idx}.png"
    dst_path = os.path.join(out_dir, file_name)
    if not os.path.exists(dst_path):
        try:
            shutil.copyfile(src_path, dst_path)
        except Exception:
            return None

    return f"file:///android_asset/kb/{file_id}/截图/{file_name}"


def _merge_table_fragments_before_appendix_fill(
    *,
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    used_manifest_indices: Set[int],
    stats: "MergeStats",
    dry_run: bool,
    debug_merge: bool,
) -> int:
    entries = kb.get("entries")
    items = manifest.get("items")
    if not isinstance(entries, list) or not isinstance(items, list):
        return 0

    table_indices = [
        idx for idx, entry in enumerate(entries)
        if isinstance(entry, dict) and _entry_has_table_block(entry)
    ]
    if not table_indices:
        return 0

    ocr_cache: Dict[str, str] = {}

    def cached_ocr(path: str) -> str:
        if not path:
            return ""
        if path not in ocr_cache:
            ocr_cache[path] = _ocr_image_file(path)
        return ocr_cache[path]

    handled = 0
    for item_index, item in enumerate(items):
        if item_index in used_manifest_indices:
            continue
        if not isinstance(item, dict):
            continue

        uri = _pick_first_str(item, ("assetUri", "asset_uri", "imageUri", "image_uri", "src")).strip()
        if not uri:
            continue

        base = _basename_from_uri(uri).lower()
        kind = _safe_str(item.get("kind")).strip().lower()
        is_legend_like = kind == "legend" or base.startswith("legend_")
        if not is_legend_like:
            continue

        local_path = _resolve_manifest_local_path(manifest, item)
        if not local_path:
            continue

        item_text = cached_ocr(local_path)
        if len(_collapse_match_text(item_text)) < 6:
            continue

        page_number = int(_parse_page_number(item.get("pageNumber")) or 1)
        best_idx: Optional[int] = None
        best_score = 0
        best_hits = 0
        for entry_index in table_indices:
            entry = entries[entry_index]
            if not isinstance(entry, dict):
                continue
            score, hits = _score_table_fragment_match(item_text, entry, page_number=page_number)
            if score > best_score or (score == best_score and hits > best_hits):
                best_idx = int(entry_index)
                best_score = int(score)
                best_hits = int(hits)

        if best_idx is None:
            continue
        if best_hits < 2 and best_score < 10:
            continue

        entry = entries[best_idx]
        if not isinstance(entry, dict):
            continue

        existing_uris = _entry_existing_image_uris(entry)
        if uri in existing_uris:
            used_manifest_indices.add(int(item_index))
            stats.skipped_duplicate += 1
            handled += 1
            continue

        table_block = _first_table_block(entry)
        snapshot_uri = ""
        if isinstance(table_block, dict):
            snapshot_uri = _safe_str(table_block.get("imageUri") or table_block.get("image_uri")).strip()
        snapshot_path = _resolve_table_snapshot_local_path(manifest, snapshot_uri)
        snapshot_text = cached_ocr(snapshot_path) if snapshot_path else ""
        snapshot_base = _basename_from_uri(snapshot_uri).lower()
        current_is_legend = base.startswith("legend_")
        snapshot_is_legend = snapshot_base.startswith("legend_")

        if snapshot_text and snapshot_is_legend and _snapshot_covers_fragment(snapshot_text, item_text):
            if not dry_run:
                _remove_image_references_by_basename(kb, uri)
            used_manifest_indices.add(int(item_index))
            stats.skipped_duplicate += 1
            handled += 1
            if debug_merge:
                print(
                    f"[MergeDebug] TableFragment duplicate: item#{item_index} {_basename_from_uri(uri)} -> "
                    f"entryId={_safe_str(entry.get('entryId')).strip()} jobTitle={_safe_str(entry.get('jobTitle')).strip()[:60]}"
                )
            continue

        if not dry_run:
            _remove_image_references_by_basename(kb, uri, keep_entry_idx=best_idx)
            block = _make_image_block(asset_uri=uri, page_number=page_number, caption="表格原件（续）")
            _insert_block(entry=entry, block=block, page_number=page_number, insert_mode="append")

        used_manifest_indices.add(int(item_index))
        stats.inserted += 1
        _record_inserted_manifest_asset(
            stats,
            item_index=item_index,
            item=item,
            asset_uri=uri,
            page_number=page_number,
            caption_text="表格原件（续）",
            anchor_text=_safe_str(item.get("anchorText")).strip(),
            heading_text=_safe_str(item.get("headingText")).strip(),
            item_kind="table",
            target_entry_id=_safe_str(entry.get("entryId")).strip(),
            insert_mode="append",
        )
        handled += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableFragment reroute: item#{item_index} {_basename_from_uri(uri)} -> "
                    f"entryId={_safe_str(entry.get('entryId')).strip()} as {_basename_from_uri(uri)}"
            )

    return int(handled)


def _reroute_existing_legend_images_into_tables(
    *,
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    dry_run: bool,
    debug_merge: bool,
) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

def _bridge_table_snapshot_prev_page_figure_continuations(
    *,
    kb: Dict[str, Any],
    debug_merge: bool,
) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    moved = 0
    for source_idx, source_entry in enumerate(entries):
        if not isinstance(source_entry, dict) or not _entry_has_table_block(source_entry):
            continue
        source_table = _first_table_block(source_entry)
        if not isinstance(source_table, dict):
            continue

        snapshot_uri = _safe_str(source_table.get("imageUri") or source_table.get("image_uri")).strip()
        if not snapshot_uri:
            continue
        page_number = int(_physical_page_from_asset_uri(snapshot_uri) or 0)
        if page_number <= 1:
            continue

        source_bbox = _normalize_bbox_dict(source_table.get("bbox")) if isinstance(source_table.get("bbox"), dict) else {}

        ranked: List[Tuple[int, int]] = []
        for target_idx in range(0, int(source_idx)):
            target_entry = entries[target_idx]
            if not isinstance(target_entry, dict) or _entry_has_table_block(target_entry):
                continue
            blocks = target_entry.get("blocks")
            if not isinstance(blocks, list):
                continue

            score = 0
            prev_page_hits = 0
            visual_prev_hits = 0
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if _safe_str(block.get("type")).strip().lower() != "image":
                    continue
                block_uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
                block_page = int(block.get("pageNumber") or _physical_page_from_asset_uri(block_uri) or 0)
                if block_page != page_number - 1:
                    continue

                prev_page_hits += 1
                if _basename_from_uri(block_uri).lower().startswith(f"visual_p{page_number - 1}_"):
                    visual_prev_hits += 1

                block_bbox = _normalize_bbox_dict(block.get("bbox")) if isinstance(block.get("bbox"), dict) else {}
                src_width = float(source_bbox.get("width") or 0)
                dst_width = float(block_bbox.get("width") or 0)
                if src_width > 0 and dst_width > 0:
                    width_ratio = abs(src_width - dst_width) / max(src_width, dst_width)
                    if width_ratio <= 0.06:
                        score += 6
                    elif width_ratio <= 0.12:
                        score += 4
                    elif width_ratio <= 0.20:
                        score += 2

                src_left = float(source_bbox.get("left") or 0)
                dst_left = float(block_bbox.get("left") or 0)
                if src_left > 0 and dst_left > 0:
                    left_delta = abs(src_left - dst_left)
                    if left_delta <= 30:
                        score += 4
                    elif left_delta <= 80:
                        score += 2

            if prev_page_hits <= 0:
                continue

            score += 5 * min(visual_prev_hits, 1)
            if prev_page_hits == 1:
                score += 2

            labels = _entry_reference_labels(target_entry)
            if labels:
                score += 2

            content_blob = " ".join(
                [
                    _safe_str(target_entry.get("contentMarkdown")),
                    _safe_str(target_entry.get("contentNormalized")),
                    _safe_str(target_entry.get("jobTitle")),
                ]
            )
            compact_content = _collapse_match_text(content_blob)
            if "明细表" in compact_content:
                score += 8

            distance = int(source_idx) - int(target_idx)
            score -= min(max(distance // 15, 0), 6)
            ranked.append((int(score), int(target_idx)))

        if not ranked:
            continue

        ranked.sort(key=lambda item: (-item[0], item[1]))
        best_score, best_target_idx = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else -999
        if best_score < 10:
            continue
        if second_score > -999 and (best_score - second_score) < 3:
            continue

        target_entry = entries[best_target_idx]
        if not isinstance(target_entry, dict):
            continue
        if snapshot_uri in _entry_existing_image_uris(target_entry):
            source_table.pop("imageUri", None)
            source_table.pop("image_uri", None)
            moved += 1
            continue

        _insert_block(
            entry=target_entry,
            block=_make_image_block(asset_uri=snapshot_uri, page_number=page_number, caption="图件续页"),
            page_number=page_number,
            insert_mode="append",
        )
        source_table.pop("imageUri", None)
        source_table.pop("image_uri", None)
        moved += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableSnapshot prev-page-bridge: {_basename_from_uri(snapshot_uri)} "
                f"srcEntry={_safe_str(source_entry.get('entryId')).strip()} -> "
                f"targetEntry={_safe_str(target_entry.get('entryId')).strip()} score={best_score}"
            )

    return int(moved)


def _bridge_previous_legend_page_into_tables(
    *,
    kb: Dict[str, Any],
    dry_run: bool,
    debug_merge: bool,
) -> int:
    entries = kb.get("entries")
    if not isinstance(entries, list):
        return 0

    def _parse_legend_page(uri: str) -> Optional[int]:
        match = re.search(r"legend_p(\d+)_idx\d+", _basename_from_uri(uri), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    page_to_occurrences: Dict[int, List[Tuple[int, str]]] = {}
    table_pages_in_use: Set[int] = set()
    table_targets: List[Tuple[int, int, str]] = []

    for entry_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        is_table = _entry_has_table_block(entry)
        entry_legend_uris: List[str] = []
        table_block = _first_table_block(entry) if is_table else None
        if isinstance(table_block, dict):
            primary_uri = _safe_str(table_block.get("imageUri") or table_block.get("image_uri")).strip()
            if _parse_legend_page(primary_uri) is not None:
                entry_legend_uris.append(primary_uri)

        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_uri = _safe_str(block.get("imageUri") or block.get("image_uri") or block.get("src")).strip()
            page = _parse_legend_page(block_uri)
            if page is None:
                continue
            page_to_occurrences.setdefault(int(page), []).append((int(entry_idx), block_uri))
            if is_table:
                table_pages_in_use.add(int(page))
                entry_legend_uris.append(block_uri)

        if is_table and entry_legend_uris:
            unique_legend_uris = sorted(
                set(entry_legend_uris),
                key=lambda uri: (_parse_legend_page(uri) or 10**9, _basename_from_uri(uri)),
            )
            primary_uri = unique_legend_uris[0]
            primary_page = _parse_legend_page(primary_uri)
            if primary_page is not None:
                table_pages_in_use.add(int(primary_page))
                table_targets.append((int(primary_page), int(entry_idx), primary_uri))

    moved = 0
    for primary_page, entry_idx, primary_uri in sorted(table_targets, key=lambda item: (item[0], item[1])):
        prev_page = int(primary_page) - 1
        if prev_page <= 0 or prev_page in table_pages_in_use:
            continue
        candidates = page_to_occurrences.get(prev_page, [])
        candidates = [(src_idx, uri) for src_idx, uri in candidates if int(src_idx) != int(entry_idx)]
        if len(candidates) != 1:
            continue

        src_idx, uri = candidates[0]
        target_entry = entries[entry_idx]
        if not isinstance(target_entry, dict):
            continue
        if uri in _entry_existing_image_uris(target_entry):
            continue

        if not dry_run:
            _remove_image_references_by_basename(kb, uri, keep_entry_idx=entry_idx)
            _insert_block(
                entry=target_entry,
                block=_make_image_block(asset_uri=uri, page_number=prev_page, caption="表格原件（续）"),
                page_number=prev_page,
                insert_mode="append",
            )

        table_pages_in_use.add(prev_page)
        moved += 1
        if debug_merge:
            print(
                f"[MergeDebug] TableLegend bridge-prev-page: {_basename_from_uri(uri)} -> "
                f"entryId={_safe_str(target_entry.get('entryId')).strip()}"
            )

    return int(moved)


@dataclass
class MergeStats:
    items_total: int = 0
    items_with_page: int = 0
    items_with_uri: int = 0
    inserted: int = 0
    skipped_duplicate: int = 0
    skipped_no_entry_for_page: int = 0
    skipped_missing_fields: int = 0
    skipped_out_of_scope: int = 0
    ambiguous_pages: int = 0
    inserted_assets: List[Dict[str, Any]] = field(default_factory=list)


def _copy_bbox_dict(bbox: Any) -> Optional[Dict[str, int]]:
    if not isinstance(bbox, dict):
        return None
    out: Dict[str, int] = {}
    for key in ("left", "top", "right", "bottom", "width", "height"):
        try:
            out[key] = int(bbox.get(key, 0))
        except Exception:
            out[key] = 0
    return out


def _record_inserted_manifest_asset(
    stats: MergeStats,
    *,
    item_index: int,
    item: Dict[str, Any],
    asset_uri: str,
    page_number: Optional[int],
    caption_text: str,
    anchor_text: str,
    heading_text: str,
    item_kind: str,
    target_entry_id: str,
    insert_mode: str,
    bbox: Optional[Dict[str, int]] = None,
) -> None:
    basename = _basename_from_uri(asset_uri)
    if not basename:
        return
    stats.inserted_assets.append(
        {
            "itemIndex": int(item_index),
            "assetUri": _safe_str(asset_uri).strip(),
            "basename": basename,
            "pageNumber": _parse_page_number(page_number),
            "captionText": _safe_str(caption_text).strip(),
            "anchorText": _safe_str(anchor_text).strip(),
            "headingText": _safe_str(heading_text).strip(),
            "itemKind": _safe_str(item_kind).strip().lower() or _manifest_item_semantic_kind_hint(item),
            "targetEntryId": _safe_str(target_entry_id).strip(),
            "insertMode": _safe_str(insert_mode).strip() or "after-page-text",
            "bbox": _copy_bbox_dict(bbox),
        }
    )


def _find_asset_alignment_target_entry_index(
    kb: Dict[str, Any],
    record: Dict[str, Any],
) -> Tuple[Optional[int], int, int, float]:
    entries = kb.get("entries")
    if not isinstance(entries, list) or not entries:
        return (None, 0, 0, 0.0)

    page_to_entries: Dict[int, List[int]] = {}
    profiles: Dict[int, Dict[str, Any]] = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        profiles[idx] = _build_entry_semantic_profile(entry)
        for page_num in set(_collect_entry_pages(entry)):
            page_to_entries.setdefault(int(page_num), []).append(int(idx))

    page_number = _parse_page_number(record.get("pageNumber"))
    item_kind = _safe_str(record.get("itemKind")).strip().lower() or "legend"
    signals = _semantic_texts_from_manifest_item(
        {
            "anchorText": record.get("anchorText"),
            "headingText": record.get("headingText"),
            "caption": record.get("captionText"),
        }
    )
    if not signals:
        fallback_heading = _safe_str(record.get("headingText") or record.get("anchorText") or record.get("captionText")).strip()
        if fallback_heading and isinstance(page_number, int) and page_number > 0:
            best_idx = _find_target_entry_index(entries, page_number, "best", heading_text=fallback_heading, prefer_heading=True)
            return (best_idx, 0, 0, 0.0)
        return (None, 0, 0, 0.0)

    candidate_indices = (
        _candidate_indices_for_page_window(page_to_entries, page_number, radius=8)
        if isinstance(page_number, int) and page_number > 0
        else []
    )
    if not candidate_indices:
        candidate_indices = list(profiles.keys())

    best_idx: Optional[int] = None
    best_score = 0
    best_evidence = 0
    best_similarity = 0.0
    best_page_gap = 10**9

    for idx in candidate_indices:
        entry = entries[idx]
        profile = profiles.get(idx) or _build_entry_semantic_profile(entry)
        local_best_score = 0
        local_best_evidence = 0
        local_best_similarity = 0.0
        for signal in signals:
            score, evidence, similarity = _score_manifest_signal_against_entry(
                signal,
                entry,
                profile,
                item_kind=item_kind,
                cutoff_ratio=0.5,
                page_number=page_number,
            )
            if score > local_best_score:
                local_best_score = int(score)
                local_best_evidence = int(evidence)
                local_best_similarity = float(similarity)

        entry_page = _entry_page_hint(entry)
        page_gap = abs(int(entry_page) - int(page_number)) if isinstance(entry_page, int) and isinstance(page_number, int) else 10**9
        if local_best_score <= 0:
            continue

        is_better = False
        if local_best_score > best_score:
            is_better = True
        elif local_best_score == best_score and local_best_evidence > best_evidence:
            is_better = True
        elif local_best_score == best_score and local_best_evidence == best_evidence and local_best_similarity > best_similarity:
            is_better = True
        elif (
            local_best_score == best_score
            and local_best_evidence == best_evidence
            and abs(local_best_similarity - best_similarity) < 1e-9
            and page_gap < best_page_gap
        ):
            is_better = True

        if is_better:
            best_idx = int(idx)
            best_score = int(local_best_score)
            best_evidence = int(local_best_evidence)
            best_similarity = float(local_best_similarity)
            best_page_gap = int(page_gap)

    min_score = 220 if item_kind == "table" else 180
    if best_idx is None or best_score < min_score:
        return (None, best_score, best_evidence, best_similarity)
    return (best_idx, best_score, best_evidence, best_similarity)


def _attach_asset_alignment_record_to_entry(
    kb: Dict[str, Any],
    record: Dict[str, Any],
    entry_index: int,
    *,
    debug: bool,
) -> bool:
    entries = kb.get("entries")
    if not isinstance(entries, list) or entry_index < 0 or entry_index >= len(entries):
        return False
    entry = entries[entry_index]
    if not isinstance(entry, dict):
        return False

    asset_uri = _safe_str(record.get("assetUri")).strip()
    if not asset_uri:
        return False

    existing_uris = _entry_existing_image_uris(entry)
    if asset_uri in existing_uris:
        return True

    block_page = _parse_page_number(record.get("pageNumber")) or int(_entry_page_hint(entry) or 1)
    caption_text = _safe_str(record.get("captionText")).strip()
    anchor_text = _safe_str(record.get("anchorText")).strip()
    heading_text = _safe_str(record.get("headingText")).strip()
    item_kind = _safe_str(record.get("itemKind")).strip().lower() or "legend"
    block_bbox = _copy_bbox_dict(record.get("bbox"))

    if not caption_text:
        caption_text = heading_text or anchor_text or ("表格原件（续）" if item_kind == "table" else "图件续页")

    attached = False
    if item_kind == "table" and _entry_has_table_block_missing_snapshot(entry):
        attached = _attach_snapshot_to_first_missing_table_block(entry, asset_uri, block_bbox)

    if not attached:
        block = _make_image_block(asset_uri=asset_uri, page_number=block_page, caption=caption_text, bbox=block_bbox)
        inserted_by_anchor = _insert_image_block_after_anchor_text(
            entry,
            block,
            page_number=block_page,
            anchor_text=anchor_text,
            heading_text=heading_text,
            caption_text=caption_text,
        )
        if not inserted_by_anchor:
            _insert_block(
                entry=entry,
                block=block,
                page_number=block_page,
                insert_mode=_safe_str(record.get("insertMode")).strip() or "after-page-text",
            )

    if debug:
        print(
            f"[AssetAlignment] reattached={_basename_from_uri(asset_uri)} -> "
            f"entryId={_safe_str(entry.get('entryId')).strip()}"
        )
    return True


def _entry_has_table_block_missing_snapshot(entry: Dict[str, Any]) -> bool:
    blocks = entry.get("blocks") if isinstance(entry, dict) else None
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if _safe_str(block.get("type")).strip().lower() != "table":
            continue
        snapshot_uri = _safe_str(block.get("imageUri") or block.get("snapshotUri")).strip()
        images = block.get("images")
        has_inline_images = isinstance(images, list) and any(
            isinstance(image, dict) and _safe_str(image.get("imageUri") or image.get("src")).strip()
            for image in images
        )
        if not snapshot_uri and not has_inline_images:
            return True
    return False


def _attach_snapshot_to_first_missing_table_block(
    entry: Dict[str, Any],
    asset_uri: str,
    bbox: Optional[Dict[str, int]],
) -> bool:
    blocks = entry.get("blocks") if isinstance(entry, dict) else None
    if not isinstance(blocks, list):
        return False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if _safe_str(block.get("type")).strip().lower() != "table":
            continue
        snapshot_uri = _safe_str(block.get("imageUri") or block.get("snapshotUri")).strip()
        images = block.get("images")
        has_inline_images = isinstance(images, list) and any(
            isinstance(image, dict) and _safe_str(image.get("imageUri") or image.get("src")).strip()
            for image in images
        )
        if snapshot_uri or has_inline_images:
            continue
        block["imageUri"] = asset_uri
        if bbox:
            block["imageBBox"] = _copy_bbox_dict(bbox)
        return True
    return False


def _looks_like_pure_figure_caption_signal(text: str) -> bool:
    raw = _safe_str(text).strip()
    if not raw:
        return False
    compact = re.sub(r"\s+", "", raw)
    return bool(re.match(r"^(?:\d+)?[图表][0-9一二三四五六七八九十百千万—\-~～]+", compact))


def _should_skip_asset_alignment_record(record: Dict[str, Any]) -> bool:
    item_kind = _safe_str(record.get("itemKind")).strip().lower() or "legend"
    if item_kind == "table":
        return False

    raw_signals = [
        _safe_str(record.get("anchorText")).strip(),
        _safe_str(record.get("headingText")).strip(),
        _safe_str(record.get("captionText")).strip(),
    ]
    normalized_signals = {
        _normalize_for_similarity(text)
        for text in raw_signals
        if _normalize_for_similarity(text)
    }
    if len(normalized_signals) != 1:
        return False

    repeated_signal = next(iter(normalized_signals), "")
    if not repeated_signal:
        return False

    for text in raw_signals:
        if _normalize_for_similarity(text) == repeated_signal and _looks_like_pure_figure_caption_signal(text):
            return True
    return False


def _run_asset_alignment_check(
    kb: Dict[str, Any],
    *,
    inserted_assets: List[Dict[str, Any]],
    debug: bool = False,
) -> Dict[str, int]:
    metrics = {
        "checkedInsertedAssets": 0,
        "missingInsertedAssets": 0,
        "reattachedInsertedAssets": 0,
        "skippedWeakAnchorAssets": 0,
        "unresolvedInsertedAssets": 0,
    }
    if not inserted_assets:
        return metrics

    referenced = _collect_kb_image_basenames(kb)
    seen: Set[str] = set()
    for record in inserted_assets:
        if not isinstance(record, dict):
            continue
        basename = _safe_str(record.get("basename") or _basename_from_uri(record.get("assetUri"))).strip()
        if not basename or basename in seen:
            continue
        seen.add(basename)
        metrics["checkedInsertedAssets"] += 1
        if basename in referenced:
            continue

        metrics["missingInsertedAssets"] += 1
        if _should_skip_asset_alignment_record(record):
            metrics["skippedWeakAnchorAssets"] += 1
            if debug:
                print(f"[AssetAlignment] skipped_weak_anchor={basename}")
            continue

        best_idx, best_score, best_evidence, best_similarity = _find_asset_alignment_target_entry_index(kb, record)
        if best_idx is None:
            metrics["unresolvedInsertedAssets"] += 1
            if debug:
                print(
                    f"[AssetAlignment] unresolved={basename} score={best_score} "
                    f"evidence={best_evidence} similarity={best_similarity:.3f}"
                )
            continue

        if _attach_asset_alignment_record_to_entry(kb, record, best_idx, debug=debug):
            referenced.add(basename)
            metrics["reattachedInsertedAssets"] += 1
        else:
            metrics["unresolvedInsertedAssets"] += 1
            if debug:
                print(
                    f"[AssetAlignment] failed={basename} score={best_score} "
                    f"evidence={best_evidence} similarity={best_similarity:.3f}"
                )

    if debug and metrics["missingInsertedAssets"]:
        print(
            f"[AssetAlignment] checked={metrics['checkedInsertedAssets']} "
            f"missing={metrics['missingInsertedAssets']} "
            f"reattached={metrics['reattachedInsertedAssets']} "
            f"skipped={metrics['skippedWeakAnchorAssets']} "
            f"unresolved={metrics['unresolvedInsertedAssets']}"
        )
    return metrics


def merge_manifest_into_kb(
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    insert_mode: str,
    page_match: str,
    prefer_heading: bool,
    dry_run: bool,
    scope: Optional[MergeScope],
    gap_max_pages: int = 3,
    enable_appendix_fill: bool = True,
    similarity_threshold: float = 0.5,
    debug_merge: bool = False,
    *,
    assets_root: str = "",
    pdf_native: bool = False,
    source_native_kb: Optional[Dict[str, Any]] = None,
) -> MergeStats:
    stats = MergeStats()

    entries = kb.get("entries")
    if not isinstance(entries, list):
        raise ValueError("KB JSON missing 'entries' list")

    items = manifest.get("items")
    if not isinstance(items, list):
        raise ValueError("Manifest JSON missing 'items' list")

    # Precompute page -> entry indices
    all_page_to_entries: Dict[int, List[int]] = {}
    for idx, e in enumerate(entries):
        for p in set(_collect_entry_pages(e)):
            all_page_to_entries.setdefault(p, []).append(idx)

    page_to_entries: Dict[int, List[int]] = {}
    for idx, e in enumerate(entries):
        if scope is not None and not _entry_in_scope(e, scope):
            continue
        for p in set(_collect_entry_pages(e)):
            page_to_entries.setdefault(p, []).append(idx)

    scoped_indices_all = [
        idx for idx, e in enumerate(entries) if (scope is None or _entry_in_scope(e, scope))
    ]

    # Precompute richer entry semantic profiles for global semantic matching.
    # Relying only on jobTitle is too brittle for DOCX text entries like “正文#3”.
    scoped_entry_profiles: Dict[int, Dict[str, Any]] = {}
    for idx in scoped_indices_all:
        scoped_entry_profiles[idx] = _build_entry_semantic_profile(entries[idx])

    scoped_table_indices = [idx for idx in scoped_indices_all if _entry_has_table_block(entries[idx])]
    scoped_non_table_indices = [idx for idx in scoped_indices_all if idx not in scoped_table_indices]
    page_entry_order_hint: Dict[int, Tuple[int, int]] = {}
    for page_num, idxs in page_to_entries.items():
        sorted_indices = sorted(
            idxs,
            key=lambda idx: (
                int(entries[idx].get("position")) if isinstance(entries[idx].get("position"), int) else 10**9,
                idx,
            ),
        )
        total = len(sorted_indices)
        for rank, idx in enumerate(sorted_indices):
            page_entry_order_hint[int(idx)] = (int(rank), int(total))

    page_item_order_hint: Dict[int, Tuple[int, int]] = {}
    page_to_item_indices: Dict[int, List[int]] = {}
    for item_i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        pn = _parse_page_number(it.get("pageNumber"))
        if not isinstance(pn, int) or pn <= 0:
            continue
        page_to_item_indices.setdefault(int(pn), []).append(int(item_i))
    for page_num, idxs in page_to_item_indices.items():
        sorted_item_indices = sorted(idxs, key=lambda idx: _bbox_sort_key_from_item(items[idx], idx))
        total = len(sorted_item_indices)
        for rank, idx in enumerate(sorted_item_indices):
            page_item_order_hint[int(idx)] = (int(rank), int(total))

    def _page_anchor_order_score(item_index: int, entry_index: int, page_number: Optional[int]) -> float:
        if not isinstance(page_number, int) or page_number <= 0:
            return 0.0
        item_hint = page_item_order_hint.get(int(item_index))
        entry_hint = page_entry_order_hint.get(int(entry_index))
        if item_hint is None or entry_hint is None:
            return 0.0
        item_rank, item_total = item_hint
        entry_rank, entry_total = entry_hint
        if item_total <= 1 or entry_total <= 1:
            return 0.0
        i_rank = float(item_rank) / float(max(1, item_total - 1))
        e_rank = float(entry_rank) / float(max(1, entry_total - 1))
        return max(0.0, 1.0 - abs(i_rank - e_rank))

    # --- Phase 1: Semantic matches only ---
    # Only items with meaningful anchorText can be used here.
    # IMPORTANT: If Phase 1 fails (page mismatch / ambiguity / no match), the item must
    # remain unused so it can flow into Phase 2 (P3).
    sim_thr = float(similarity_threshold)
    if sim_thr < 0.0:
        sim_thr = 0.0
    if sim_thr > 1.0:
        sim_thr = 1.0

    SEM_MATCH_THR = sim_thr
    GAP_MAX_PAGES = max(0, int(gap_max_pages))

    semantic_match_by_item: Dict[int, int] = {}
    semantic_matches_by_page: List[Tuple[int, int, int]] = []  # (pdfPage, entryIdx, itemIdx)
    last_semantic_matched_entry_idx: Optional[int] = None

    for item_i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        semantic_texts = _semantic_texts_from_manifest_item(it)
        if not semantic_texts:
            continue

        item_page_number = _parse_page_number(it.get("pageNumber"))
        item_kind = _manifest_item_semantic_kind_hint(it)

        semantic_target_idx: Optional[int] = None
        semantic_best = 0.0
        semantic_best_score = -1
        semantic_best_evidence = -1
        semantic_order_score = 0.0
        semantic_anchor_used = ""

        candidate_pools: List[Tuple[str, List[int]]] = []
        if item_kind == "table" and scoped_table_indices:
            candidate_pools.append(("table", scoped_table_indices))
        elif item_kind == "legend" and scoped_non_table_indices:
            candidate_pools.append(("legend", scoped_non_table_indices))
        candidate_pools.append(("all", scoped_indices_all))

        matched_in_pool = False
        for pool_name, candidate_indices in candidate_pools:
            if not candidate_indices:
                continue

            semantic_target_idx = None
            semantic_best = 0.0
            semantic_best_score = -1
            semantic_best_evidence = -1
            semantic_order_score = 0.0
            semantic_anchor_used = ""

            for anchor_signal in semantic_texts:
                for idx in candidate_indices:
                    profile = scoped_entry_profiles.get(idx) or {}
                    ord_s = _order_score(item_i, len(items), idx, len(entries))
                    score, evidence, sem = _score_manifest_signal_against_entry(
                        anchor_signal,
                        entries[idx],
                        profile,
                        item_kind=item_kind,
                        cutoff_ratio=SEM_MATCH_THR,
                        page_number=item_page_number,
                    )
                    score += int(ord_s * 80.0)
                    score += int(_page_anchor_order_score(item_i, idx, item_page_number) * 140.0)
                    if score <= 0 and sem <= 0.0:
                        continue
                    if (
                        score > semantic_best_score
                        or (score == semantic_best_score and evidence > semantic_best_evidence)
                        or (
                            score == semantic_best_score
                            and evidence == semantic_best_evidence
                            and sem > semantic_best
                        )
                        or (
                            score == semantic_best_score
                            and evidence == semantic_best_evidence
                            and sem == semantic_best
                            and ord_s > semantic_order_score
                        )
                    ):
                        semantic_best_score = int(score)
                        semantic_best_evidence = int(evidence)
                        semantic_best = sem
                        semantic_order_score = ord_s
                        semantic_target_idx = idx
                        semantic_anchor_used = anchor_signal

            strong_semantic_hit = semantic_best_score >= 600
            balanced_semantic_hit = semantic_best_evidence >= 2 and semantic_best_score >= 180
            kind_fallback_hit = False
            if semantic_target_idx is not None:
                target_entry = entries[int(semantic_target_idx)]
                if item_kind == "table" and pool_name == "table" and _entry_has_table_block(target_entry) and semantic_best_score >= 240:
                    kind_fallback_hit = True
                elif item_kind == "legend" and pool_name == "legend" and _entry_has_image_ref_markers(target_entry) and semantic_best_score >= 220:
                    kind_fallback_hit = True

            if semantic_target_idx is not None and (semantic_best >= SEM_MATCH_THR or strong_semantic_hit or balanced_semantic_hit or kind_fallback_hit):
                semantic_match_by_item[item_i] = int(semantic_target_idx)
                last_semantic_matched_entry_idx = int(semantic_target_idx)
                matched_in_pool = True
                if debug_merge:
                    e = entries[int(semantic_target_idx)]
                    print(
                        f"[MergeDebug] Phase1 语义候选命中[{pool_name}]: item#{item_i} anchor={semantic_anchor_used!r} "
                        f"-> entryId={_safe_str(e.get('entryId')).strip()} jobTitle={_safe_str(e.get('jobTitle')).strip()[:80]} "
                        f"score={semantic_best_score} evidence={semantic_best_evidence} sim={semantic_best:.3f}"
                    )
                break

            if matched_in_pool:
                break

    if debug_merge:
        print(
            "[MergeDebug] Phase1 预扫描: "
            f"items={len(items)} scoped_entries={len(scoped_indices_all)} "
            f"similarity_threshold={SEM_MATCH_THR:.3f} semantic_hits={len(semantic_match_by_item)}"
        )

    used_manifest_indices: Set[int] = set()
    appendix_marker_insert_counts: Dict[Tuple[int, str], int] = {}

    def apply_merge(item_index: int, item: Dict[str, Any], entry_index: int) -> bool:
        """Apply a single manifest item into a single entry.

        Returns True if inserted, False if skipped (e.g., duplicate).
        """

        entry = entries[entry_index]
        asset_uri = _pick_first_str(item, ("assetUri", "asset_uri", "imageUri", "image_uri", "src")).strip()
        if not asset_uri:
            return False

        page_number = _parse_page_number(item.get("pageNumber"))
        has_page = isinstance(page_number, int) and page_number > 0
        caption_text = _caption_from_manifest_item(item)
        anchor_text = _safe_str(item.get("anchorText")).strip()
        heading_text = _safe_str(item.get("headingText")).strip()
        block_bbox = _get_manifest_item_bbox(item)
        item_page_hint = page_item_order_hint.get(int(item_index))

        def _manifest_item_kind_hint(it: Dict[str, Any], uri: str) -> str:
            kind = _safe_str(it.get("kind")).strip().lower()
            out_file = _safe_str(it.get("outFile")).strip().lower()
            basename = _basename_from_uri(uri).lower()
            if kind == "table" or basename.startswith("table_") or out_file.startswith("table_"):
                return "table"
            if kind == "legend" or basename.startswith("legend_") or out_file.startswith("legend_"):
                return "appendix"
            return "unknown"

        def _has_table_block_missing_snapshot(e: Dict[str, Any]) -> bool:
            blocks = e.get("blocks")
            if not isinstance(blocks, list):
                return False
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                t = _safe_str(b.get("type")).strip().lower()
                if t != "table":
                    continue
                snap = _safe_str(b.get("imageUri") or b.get("image_uri")).strip()
                if not snap:
                    return True
            return False

        def _attach_snapshot_to_first_table_block(e: Dict[str, Any], uri: str, bbox: Optional[Dict[str, int]]) -> bool:
            blocks = e.get("blocks")
            if not isinstance(blocks, list):
                return False
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                t = _safe_str(b.get("type")).strip().lower()
                if t != "table":
                    continue
                snap = _safe_str(b.get("imageUri") or b.get("image_uri")).strip()
                if snap:
                    continue
                b["imageUri"] = uri
                if "image_uri" in b:
                    b.pop("image_uri", None)
                if isinstance(bbox, dict) and not isinstance(_get_block_bbox(b), dict):
                    b["bbox"] = dict(bbox)
                return True
            return False

        def _remove_image_blocks_with_uri(e: Dict[str, Any], uri: str) -> None:
            blocks = e.get("blocks")
            if not isinstance(blocks, list):
                return
            kept: List[Any] = []
            for b in blocks:
                if not isinstance(b, dict):
                    kept.append(b)
                    continue
                t = _safe_str(b.get("type")).strip().lower()
                if t == "image":
                    u = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                    if u == uri:
                        continue
                kept.append(b)
            e["blocks"] = kept

        existing_uris = _entry_existing_image_uris(entry)
        need_table_snapshot = _has_table_block_missing_snapshot(entry)
        manifest_item_kind = _manifest_item_kind_hint(item, asset_uri)
        if asset_uri in existing_uris and (not need_table_snapshot):
            stats.skipped_duplicate += 1
            if debug_merge:
                jt = _safe_str(entry.get("jobTitle")).strip()
                print(f"[MergeDebug] 跳过重复: item#{item_index} uri={asset_uri} -> {jt[:60]}")
            return False

        block_page = int(page_number) if has_page else int(_entry_page_hint(entry) or 1)
        block = _make_image_block(asset_uri=asset_uri, page_number=block_page, caption=caption_text, bbox=block_bbox)

        def _is_split_entry_local(e: Dict[str, Any]) -> bool:
            eid = _safe_str(e.get("entryId")).strip()
            if "__p3_split" in eid:
                return True
            jt = _safe_str(e.get("jobTitle")).strip()
            return "（图" in jt

        if not dry_run:
            jt = _safe_str(entry.get("jobTitle")).strip()
            semantic_title = anchor_text or heading_text
            is_appendix_like = (
                ("附件" in semantic_title)
                or ("附表" in semantic_title)
                or ("附图" in semantic_title)
                or ("附件" in jt)
                or ("附表" in jt)
                or ("附图" in jt)
            )

            # UI 渲染约束：TableBlock 必须有 imageUri，否则会显示“缺少表格原件截图”。
            # 因此：若 entry 内存在 table block 且缺少 snapshot，则优先把本次图片挂到 table block.imageUri。
            attached = False
            if need_table_snapshot and manifest_item_kind == "table":
                attached = _attach_snapshot_to_first_table_block(entry, asset_uri, block_bbox)
                if attached:
                    # Avoid rendering the same screenshot twice (image block + table snapshot)
                    _remove_image_blocks_with_uri(entry, asset_uri)

            if not attached:
                if is_appendix_like:
                    marker_label = _extract_slot_label(anchor_text) or _extract_slot_label(jt)
                    marker_key = (int(entry_index), marker_label)
                    marker_occurrence = int(appendix_marker_insert_counts.get(marker_key, 0))
                    inserted_at_marker = False
                    if marker_label:
                        inserted_at_marker = _insert_image_block_after_marker_occurrence(
                            entry=entry,
                            block=block,
                            marker_label=marker_label,
                            occurrence_index=marker_occurrence,
                        )
                    if inserted_at_marker:
                        appendix_marker_insert_counts[marker_key] = marker_occurrence + 1
                    else:
                        _insert_block_after_markdown_code(entry=entry, block=block)
                else:
                    page_rank = item_page_hint[0] if item_page_hint is not None else None
                    page_total = item_page_hint[1] if item_page_hint is not None else None
                    inserted_by_anchor = _insert_image_block_after_anchor_text(
                        entry,
                        block,
                        page_number=block_page,
                        anchor_text=anchor_text,
                        heading_text=heading_text,
                        caption_text=caption_text,
                    )
                    if not inserted_by_anchor:
                        _insert_block(
                            entry=entry,
                            block=block,
                            page_number=block_page,
                            insert_mode=insert_mode,
                            page_rank_hint=page_rank,
                            page_total_hint=page_total,
                        )

            # Keep captions on the image block only. They should not backflow into
            # entry-level semantic fields such as contentNormalized/jobTitle matching.

        # 只有当本次 merge 判定“成功”时，才占用 used_manifest_indices。
        # dry_run 视为“模拟成功”，用于验证流转与日志（避免同一轮又被 P3 消费）。
        stats.inserted += 1
        _record_inserted_manifest_asset(
            stats,
            item_index=item_index,
            item=item,
            asset_uri=asset_uri,
            page_number=block_page,
            caption_text=caption_text,
            anchor_text=anchor_text,
            heading_text=heading_text,
            item_kind=manifest_item_kind,
            target_entry_id=_safe_str(entry.get("entryId")).strip(),
            insert_mode=insert_mode,
            bbox=block_bbox,
        )
        return True

    # --- 阶段一：精确匹配 (仅 P1) ---
    # 只允许：anchorText 有文字且语义命中(>= threshold) 才 merge。
    # 其它情况（页码不对、歧义、没命中）一律跳过，不占用图片索引，让其流入 Phase 2。
    for item_i, it in enumerate(items):
        stats.items_total += 1
        if not isinstance(it, dict):
            stats.skipped_missing_fields += 1
            continue

        page_number = _parse_page_number(it.get("pageNumber"))
        # accept common variants
        asset_uri = _pick_first_str(it, ("assetUri", "asset_uri", "imageUri", "image_uri", "src"))
        caption_text = _caption_from_manifest_item(it)
        anchor_text = _safe_str(it.get("anchorText")).strip()
        heading_text = _safe_str(it.get("headingText")).strip()

        has_page = isinstance(page_number, int) and page_number > 0
        if has_page:
            stats.items_with_page += 1

        if not isinstance(asset_uri, str) or not asset_uri.strip():
            stats.skipped_missing_fields += 1
            continue
        asset_uri = asset_uri.strip()
        stats.items_with_uri += 1

        if item_i not in semantic_match_by_item:
            continue

        found_entry_idx = int(semantic_match_by_item[item_i])
        if dry_run:
            e = entries[found_entry_idx]
            print(
                f"[MergeDebug] 语义匹配(P1): anchorText={anchor_text!r} headingText={heading_text!r} -> entryId={_safe_str(e.get('entryId')).strip()} "
                f"jobTitle={_safe_str(e.get('jobTitle')).strip()[:80]}"
            )

        # Phase 1 respects scope; Phase 2 will ignore it.
        if scope is not None and not _entry_in_scope(entries[found_entry_idx], scope):
            stats.skipped_out_of_scope += 1
            if debug_merge:
                e = entries[found_entry_idx]
                print(
                    f"[MergeDebug] Phase1 命中但被 scope 拦截: item#{item_i} -> entryId={_safe_str(e.get('entryId')).strip()} "
                    f"unitName={_safe_str(e.get('unitName')).strip()} jobTitle={_safe_str(e.get('jobTitle')).strip()[:80]}"
                )
            continue

        matched_successfully = apply_merge(item_i, it, found_entry_idx)
        if matched_successfully:
            used_manifest_indices.add(int(item_i))

    if debug_merge:
        print(
            "[MergeDebug] Phase1 完成: "
            f"used_manifest_indices={len(used_manifest_indices)} inserted_so_far={stats.inserted}"
        )

    table_fragment_merges = _merge_table_fragments_before_appendix_fill(
        kb=kb,
        manifest=manifest,
        used_manifest_indices=used_manifest_indices,
        stats=stats,
        dry_run=dry_run,
        debug_merge=debug_merge,
    )
    if debug_merge and table_fragment_merges:
        print(
            "[MergeDebug] TableFragment 完成: "
            f"handled={table_fragment_merges} used_manifest_indices={len(used_manifest_indices)} inserted_so_far={stats.inserted}"
        )

    # --- 阶段二：暴力盲填 (P3) - 放在循环外面 ---
    if enable_appendix_fill:
        def _normalize_job_title_for_slot(title: str) -> str:
            # 预处理：去换行、压缩多余空格，避免“表 1”“表1”“表\n1”等差异
            s = _safe_str(title).replace("\r", "").replace("\n", "")
            s = re.sub(r"\s+", " ", s).strip()
            return s

        # 暴力正则：支持 [[附件1]]、附件1、表1、表格#1 等“长得像附件/附表/表/附图”的标题
        SLOT_REGEX = r"(附件|附表|附图)\s*[\d一二三四五六七八九十]+"

        def _is_p3_slot_title(job_title: str) -> bool:
            s = _normalize_job_title_for_slot(job_title)
            if not s:
                return False
            return re.search(SLOT_REGEX, s, flags=re.IGNORECASE) is not None

        def _manifest_item_sort_key(item_index: int, item: Any) -> Tuple[int, int, int, int]:
            """Sort by (page, top, left, index). Unknown fields go to the end."""
            if not isinstance(item, dict):
                return (10**9, 10**9, 10**9, int(item_index))
            page = _parse_page_number(item.get("pageNumber"))
            page_k = int(page) if isinstance(page, int) and page > 0 else 10**9

            y0_k = 10**9
            x0_k = 10**9

            # Common bbox shapes: [x0,y0,x1,y1] or {x0,y0,x1,y1}
            bbox = item.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    x0_k = int(float(bbox[0]))
                    y0_k = int(float(bbox[1]))
                except Exception:
                    pass
            elif isinstance(bbox, dict):
                try:
                    x0_k = int(float(bbox.get("x0")))
                    y0_k = int(float(bbox.get("y0")))
                except Exception:
                    pass
                # Some tools may store bbox as {left, top, right, bottom}
                if (y0_k, x0_k) == (10**9, 10**9):
                    try:
                        x0_k = int(float(bbox.get("left")))
                        y0_k = int(float(bbox.get("top")))
                    except Exception:
                        pass

            box = item.get("box") or item.get("cropBox") or item.get("rect")
            if (y0_k, x0_k) == (10**9, 10**9) and isinstance(box, dict):
                try:
                    x0_k = int(float(box.get("x0")))
                    y0_k = int(float(box.get("y0")))
                except Exception:
                    pass
            if (y0_k, x0_k) == (10**9, 10**9) and isinstance(box, (list, tuple)) and len(box) >= 4:
                try:
                    x0_k = int(float(box[0]))
                    y0_k = int(float(box[1]))
                except Exception:
                    pass

            return (page_k, y0_k, x0_k, int(item_index))

        empty_appendix_indices: List[int] = []
        # Phase 2 ignores scope: scan all entries in original JSON order.
        for idx, e in enumerate(entries):
            e = entries[idx]
            jt = _safe_str(e.get("jobTitle")).strip()
            if not jt:
                continue
            # P3 的“空坑”判定：只看 entry 顶层 imageUri 字段（与用户伪代码一致），不看 blocks 里是否已有图片。
            # 这样可以把“之前插错位置/插在 blocks 里”的情况也纳入强行缝合。
            top_image_uri = _safe_str(e.get("imageUri") or e.get("image_uri")).strip()
            if top_image_uri:
                continue
            # 强制清空：只要标题像“附件/附表/表/表格#”，且 imageUri 为空，即纳入槽位
            if _is_p3_slot_title(jt):
                empty_appendix_indices.append(idx)
        # 注意：P3 槽位顺序必须保持 KB JSON 原始顺序

        if debug_merge:
            sample_titles = [
                _safe_str(entries[i].get("jobTitle")).strip().replace("\n", " ")[:60]
                for i in empty_appendix_indices[:8]
            ]
            print(f"[MergeDebug] Phase2 槽位: empty_slots={len(empty_appendix_indices)} sample={sample_titles}")

        def _item_has_asset_uri(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            uri = _pick_first_str(item, ("assetUri", "asset_uri", "imageUri", "image_uri", "src"))
            return bool(isinstance(uri, str) and uri.strip())

        unused_item_indices = [i for i, item in enumerate(items) if i not in used_manifest_indices and _item_has_asset_uri(item)]
        unused_item_indices.sort(key=lambda i: _manifest_item_sort_key(i, items[i]))

        if debug_merge:
            sample_items: List[str] = []
            for i in unused_item_indices[:8]:
                it = items[i]
                key = _manifest_item_sort_key(i, it)
                out_file = _safe_str(it.get("outFile")).strip() if isinstance(it, dict) else ""
                sample_items.append(f"#{i}:{key} {out_file[:40]}")
            print(f"[MergeDebug] Phase2 余料: unused_items={len(unused_item_indices)} sample={sample_items}")

        if not empty_appendix_indices:
            if debug_merge:
                print("[MergeDebug] Phase2 无槽位：empty_slots=0，无法进行 P3 强行缝合")
        else:
            slot_count = len(empty_appendix_indices)

            def _is_split_entry(e: Dict[str, Any]) -> bool:
                eid = _safe_str(e.get("entryId")).strip()
                if "__p3_split" in eid:
                    return True
                jt = _safe_str(e.get("jobTitle")).strip()
                return "（图" in jt

            def _infer_global_page_offset(
                *,
                item_pages: List[int],
                slot_pages: List[int],
                search_range: Tuple[int, int] = (-50, 50),
            ) -> Optional[int]:
                if not item_pages or not slot_pages:
                    return None
                lo, hi = int(search_range[0]), int(search_range[1])
                if lo > hi:
                    lo, hi = hi, lo
                best_score: Optional[int] = None
                best_off: Optional[int] = None
                for off in range(lo, hi + 1):
                    score = 0
                    for p in item_pages:
                        lp = int(p) - int(off)
                        score += min(abs(lp - sp) for sp in slot_pages)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_off = off
                return best_off

            # Slot splitting (一图一坑)：
            # - 若模板槽位本身 blocks 里没有任何图片，则允许先占用该槽位（每槽 1 图）。
            # - 其余图片全部通过“复制模板 entry -> 清除图片 -> 生成新 entryId/jobTitle/position”来承载。
            # 这样可以避免出现“一个 entry 挂多张图”。

            def _strip_images_from_entry(entry: Dict[str, Any]) -> None:
                entry.pop("imageUri", None)
                entry.pop("image_uri", None)
                blocks = entry.get("blocks")
                if not isinstance(blocks, list):
                    return
                kept: List[Any] = []
                for b in blocks:
                    if not isinstance(b, dict):
                        kept.append(b)
                        continue
                    t = _safe_str(b.get("type")).strip().lower()
                    if t == "image":
                        continue

                    # Keep non-image blocks (including table), but clear any snapshot fields
                    if "imageUri" in b:
                        b.pop("imageUri", None)
                    if "image_uri" in b:
                        b.pop("image_uri", None)
                    kept.append(b)
                entry["blocks"] = kept

            existing_entry_ids: Set[str] = set()
            for e in entries:
                eid = _safe_str(e.get("entryId")).strip()
                if eid:
                    existing_entry_ids.add(eid)

            def _make_unique_entry_id(base: str) -> str:
                base = _safe_str(base).strip() or "p3_split"
                candidate = base
                n = 2
                while candidate in existing_entry_ids:
                    candidate = f"{base}_{n}"
                    n += 1
                existing_entry_ids.add(candidate)
                return candidate

            max_position = 0
            for e in entries:
                p = e.get("position")
                if isinstance(p, int) and p > max_position:
                    max_position = p
            next_position = max_position + 1

            # To make slot-splitting usable on an already-merged KB, we first clear
            # existing image blocks from the template slots (the 4 appendix/table entries).
            # This avoids creating duplicate images and lets the run "re-pack" images
            # into one-image-per-entry layout.
            # Build base slot templates (non-split ones) and group their existing split entries.
            base_template_indices: List[int] = [
                int(i) for i in empty_appendix_indices if not _is_split_entry(entries[int(i)])
            ]

            # If we can't find base templates reliably, fall back to old round-robin behavior.
            use_page_guided = len(base_template_indices) >= 1
            base_slot_pages: List[int] = []
            base_slot_idx_order: List[int] = []
            if use_page_guided:
                for idx in base_template_indices:
                    pn = _parse_page_number(entries[idx].get("pageNumber"))
                    if pn is None:
                        continue
                    base_slot_pages.append(int(pn))
                    base_slot_idx_order.append(int(idx))
                use_page_guided = bool(base_slot_pages)

            # Clear images from ALL slot entries (templates + existing splits) so repeated runs don't accumulate.
            preserved_phase1_images_by_eid: Dict[str, List[Tuple[int, int, str]]] = {}
            removed_from_templates = 0
            for slot_idx in empty_appendix_indices:
                idx_i = int(slot_idx)
                eid_i = _safe_str(entries[idx_i].get("entryId")).strip()
                before = 0
                try:
                    before = len(_entry_existing_image_uris(entries[idx_i]))
                except Exception:
                    before = 0
                if before > 0:
                    blocks_i = entries[idx_i].get("blocks")
                    if isinstance(blocks_i, list) and eid_i:
                        preserved_items: List[Tuple[int, int, str]] = []
                        for b in blocks_i:
                            if not isinstance(b, dict):
                                continue
                            if _safe_str(b.get("type")).strip().lower() != "image":
                                continue
                            uri = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                            if not uri:
                                continue
                            m = re.search(r"_p(\d+)_idx(\d+)", uri)
                            if m:
                                try:
                                    page_i = int(m.group(1))
                                    img_i = int(m.group(2))
                                except Exception:
                                    page_i = int(_parse_page_number(b.get("pageNumber")) or 1)
                                    img_i = 1
                            else:
                                page_i = int(_parse_page_number(b.get("pageNumber")) or 1)
                                img_i = 1
                            preserved_items.append((page_i, img_i, uri))
                        if preserved_items:
                            preserved_phase1_images_by_eid.setdefault(eid_i, []).extend(preserved_items)
                    _strip_images_from_entry(entries[idx_i])
                    removed_from_templates += int(before)

            if debug_merge and removed_from_templates > 0:
                print(f"[MergeDebug] Phase2 slot-splitting: cleared_template_images={removed_from_templates}")

            # Prepare per-template available empty slot indices.
            # Group membership is determined by entryId prefix: <template_eid> or <template_eid>__p3_split*
            template_groups: Dict[int, List[int]] = {}
            template_eids: Dict[int, str] = {}
            for t_idx in base_template_indices:
                template_groups[int(t_idx)] = []
                template_eids[int(t_idx)] = _safe_str(entries[int(t_idx)].get("entryId")).strip()

            for slot_idx in empty_appendix_indices:
                e = entries[int(slot_idx)]
                eid = _safe_str(e.get("entryId")).strip()
                for t_idx in base_template_indices:
                    teid = template_eids.get(int(t_idx), "")
                    if not teid:
                        continue
                    if eid == teid or eid.startswith(teid + "__p3_split"):
                        template_groups[int(t_idx)].append(int(slot_idx))
                        break

            for t_idx, idxs in template_groups.items():
                idxs.sort()

            available_by_template: Dict[int, List[int]] = {
                int(t_idx): [i for i in idxs if len(_entry_existing_image_uris(entries[int(i)])) == 0]
                for t_idx, idxs in template_groups.items()
            }

            # Infer a global physical->logical page offset and pre-assign each item to the closest base template.
            item_pages: List[int] = []
            for i in unused_item_indices:
                it = items[i]
                if not isinstance(it, dict):
                    continue
                pn = _parse_page_number(it.get("pageNumber"))
                if pn is not None:
                    item_pages.append(int(pn))

            inferred_offset: Optional[int] = None
            cluster_split: Optional[Tuple[int, int]] = None
            inferred_offset_left: Optional[int] = None
            inferred_offset_right: Optional[int] = None
            item_to_template: Dict[int, int] = {}
            if use_page_guided and item_pages:
                # Try to detect a clear page-number "gap" and infer offsets per cluster.
                # This helps when one PDF has multiple physical->logical shifts (e.g., cover/appendix sections).
                pages_sorted = sorted(item_pages)
                best_gap = (0, None)
                for a, b in zip(pages_sorted, pages_sorted[1:]):
                    gap = int(b) - int(a)
                    if gap > int(best_gap[0]):
                        best_gap = (gap, (int(a), int(b)))

                # If the largest gap is significant, split into two clusters.
                # Threshold is conservative to avoid overfitting on small datasets.
                if best_gap[1] is not None and int(best_gap[0]) >= 6:
                    cluster_split = (int(best_gap[1][0]), int(best_gap[1][1]))
                    left_pages = [p for p in item_pages if int(p) <= int(cluster_split[0])]
                    right_pages = [p for p in item_pages if int(p) >= int(cluster_split[1])]
                    if left_pages and right_pages:
                        inferred_offset_left = _infer_global_page_offset(item_pages=left_pages, slot_pages=base_slot_pages, search_range=(-50, 50))
                        inferred_offset_right = _infer_global_page_offset(item_pages=right_pages, slot_pages=base_slot_pages, search_range=(-50, 50))
                    else:
                        cluster_split = None

                if cluster_split and inferred_offset_left is not None and inferred_offset_right is not None:
                    if debug_merge:
                        print(
                            f"[MergeDebug] Phase2 page-guided: split={cluster_split} "
                            f"offset_left={inferred_offset_left} offset_right={inferred_offset_right} base_slot_pages={base_slot_pages}"
                        )
                else:
                    cluster_split = None
                    inferred_offset = _infer_global_page_offset(item_pages=item_pages, slot_pages=base_slot_pages, search_range=(-50, 50))
                    if debug_merge and inferred_offset is not None:
                        print(f"[MergeDebug] Phase2 page-guided: inferred_page_offset={inferred_offset} base_slot_pages={base_slot_pages}")

                # ---------- bbox 版面特征判别：表格 vs 附件 ----------
                def _extract_bbox_from_item(it: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
                    # Try common shapes: bbox={x0,y0,x1,y1} or {left,top,right,bottom}
                    cand = it.get("bbox") or it.get("box") or it.get("cropBox") or it.get("rect")
                    if isinstance(cand, dict):
                        x0 = cand.get("x0", cand.get("left"))
                        y0 = cand.get("y0", cand.get("top"))
                        x1 = cand.get("x1", cand.get("right"))
                        y1 = cand.get("y1", cand.get("bottom"))
                        try:
                            if x0 is None or y0 is None or x1 is None or y1 is None:
                                return None
                            x0_i = int(float(x0))
                            y0_i = int(float(y0))
                            x1_i = int(float(x1))
                            y1_i = int(float(y1))
                            return (x0_i, y0_i, x1_i, y1_i)
                        except Exception:
                            return None
                    if isinstance(cand, (list, tuple)) and len(cand) >= 4:
                        try:
                            x0_i = int(float(cand[0]))
                            y0_i = int(float(cand[1]))
                            x1_i = int(float(cand[2]))
                            y1_i = int(float(cand[3]))
                            return (x0_i, y0_i, x1_i, y1_i)
                        except Exception:
                            return None
                    return None

                def _percentile(sorted_vals: List[float], p: float) -> Optional[float]:
                    if not sorted_vals:
                        return None
                    if p <= 0:
                        return float(sorted_vals[0])
                    if p >= 100:
                        return float(sorted_vals[-1])
                    k = (len(sorted_vals) - 1) * (p / 100.0)
                    f = int(math.floor(k))
                    c = int(math.ceil(k))
                    if f == c:
                        return float(sorted_vals[f])
                    d0 = float(sorted_vals[f]) * (c - k)
                    d1 = float(sorted_vals[c]) * (k - f)
                    return float(d0 + d1)

                # Compute heuristic thresholds from this run's items (robust to different coordinate scales).
                ratios: List[float] = []
                heights: List[float] = []
                for i in unused_item_indices:
                    it = items[i]
                    if not isinstance(it, dict):
                        continue
                    bb = _extract_bbox_from_item(it)
                    if not bb:
                        continue
                    x0, y0, x1, y1 = bb
                    w = max(1, int(x1) - int(x0))
                    h = max(1, int(y1) - int(y0))
                    ratios.append(float(w) / float(h))
                    heights.append(float(h))

                ratios_sorted = sorted(ratios)
                heights_sorted = sorted(heights)
                ratio_p70 = _percentile(ratios_sorted, 70.0)
                h_p40 = _percentile(heights_sorted, 40.0)

                # Heuristic defaults (robust across docs):
                # - "wide" blocks (r>=2.0) are usually table-like even if area is small
                # - moderately wide blocks can be tables if they are also short (h <= p40)
                ratio_thresh = float(ratio_p70) if ratio_p70 is not None else 1.8
                ratio_thresh = max(1.6, ratio_thresh)
                short_h_thresh = float(h_p40) if h_p40 is not None else 999999.0

                if debug_merge:
                    print(
                        f"[MergeDebug] Phase2 bbox-heuristic: ratio_p70={ratio_thresh:.3f} short_h_p40={short_h_thresh:.1f} samples={len(ratios_sorted)}"
                    )

                def _classify_item_layout(it: Dict[str, Any]) -> str:
                    explicit_kind = _safe_str(it.get("kind")).strip().lower()
                    out_file = _safe_str(it.get("outFile")).strip().lower()
                    uri = _pick_first_str(it, ("assetUri", "asset_uri", "imageUri", "image_uri", "src")).strip().lower()
                    base = _basename_from_uri(uri).lower()
                    if explicit_kind == "table" or out_file.startswith("table_") or base.startswith("table_"):
                        return "table"
                    if explicit_kind == "legend" or out_file.startswith("legend_") or base.startswith("legend_"):
                        return "appendix"
                    bb = _extract_bbox_from_item(it)
                    if not bb:
                        return "unknown"
                    x0, y0, x1, y1 = bb
                    w = max(1, int(x1) - int(x0))
                    h = max(1, int(y1) - int(y0))
                    r = float(w) / float(h)
                    # table-like:
                    # - very wide => table
                    # - moderately wide + short height => table
                    if r >= 2.0:
                        return "table"
                    if r >= ratio_thresh and float(h) <= float(short_h_thresh):
                        return "table"
                    return "appendix"

                # Prepare template kinds based on jobTitle keywords.
                template_kind: Dict[int, str] = {}
                for t_idx in base_slot_idx_order:
                    jt = _safe_str(entries[int(t_idx)].get("jobTitle")).strip()
                    if ("表格" in jt) or (jt.startswith("表")):
                        template_kind[int(t_idx)] = "table"
                    elif "附件" in jt or "附表" in jt:
                        template_kind[int(t_idx)] = "appendix"
                    else:
                        template_kind[int(t_idx)] = "unknown"

                table_slot_pages = [
                    int(_parse_page_number(entries[int(t)].get("pageNumber")) or 0)
                    for t in base_slot_idx_order
                    if template_kind.get(int(t)) == "table"
                ]
                table_slot_pages = [p for p in table_slot_pages if p > 0]
                appendix_slot_pages = [
                    int(_parse_page_number(entries[int(t)].get("pageNumber")) or 0)
                    for t in base_slot_idx_order
                    if template_kind.get(int(t)) == "appendix"
                ]
                appendix_slot_pages = [p for p in appendix_slot_pages if p > 0]

                # Precompute item layout for offset inference.
                item_layout_by_idx: Dict[int, str] = {}
                for i in unused_item_indices:
                    it = items[i]
                    if not isinstance(it, dict):
                        continue
                    item_layout_by_idx[int(i)] = _classify_item_layout(it)

                # If a physical page contains any appendix-like screenshot, treat all screenshots on that page as appendix.
                # Rationale: the same appendix page often has multiple crops (idx1/idx2), and bbox heuristics may misclassify
                # one of them as table, causing wrong template assignment.
                try:
                    page_to_layouts: Dict[int, Set[str]] = {}
                    page_to_items: Dict[int, List[int]] = {}
                    for i in unused_item_indices:
                        it = items[int(i)]
                        if not isinstance(it, dict):
                            continue
                        pn = _parse_page_number(it.get("pageNumber"))
                        if not isinstance(pn, int) or pn <= 0:
                            continue
                        lay = item_layout_by_idx.get(int(i)) or "unknown"
                        page_to_layouts.setdefault(int(pn), set()).add(str(lay))
                        page_to_items.setdefault(int(pn), []).append(int(i))
                    for pn, lays in page_to_layouts.items():
                        if "appendix" in lays:
                            for ii in page_to_items.get(int(pn), []):
                                item_layout_by_idx[int(ii)] = "appendix"
                except Exception:
                    pass

                # Infer offsets per cluster AND per layout-kind (table vs appendix) if possible.
                # This matters when the same cluster contains both table screenshots and appendix screenshots.
                def _cluster_of_page(p: int) -> str:
                    if cluster_split and inferred_offset_left is not None and inferred_offset_right is not None:
                        if int(p) <= int(cluster_split[0]):
                            return "left"
                        if int(p) >= int(cluster_split[1]):
                            return "right"
                    return "global"

                def _infer_offsets_for(cluster_name: str) -> Dict[str, Optional[int]]:
                    if cluster_name == "global":
                        all_off = inferred_offset
                    else:
                        all_off = inferred_offset_left if cluster_name == "left" else inferred_offset_right

                    # Default to all_off if we can't infer kind-specific offsets.
                    out: Dict[str, Optional[int]] = {"table": all_off, "appendix": all_off, "all": all_off}

                    # Kind-specific: only if we have corresponding slot pages.
                    if table_slot_pages:
                        table_pages = []
                        for ii in unused_item_indices:
                            it = items[ii]
                            if not isinstance(it, dict):
                                continue
                            pn = _parse_page_number(it.get("pageNumber"))
                            if pn is None:
                                continue
                            if _cluster_of_page(int(pn)) != cluster_name:
                                continue
                            if item_layout_by_idx.get(int(ii)) == "table":
                                table_pages.append(int(pn))
                        if table_pages:
                            out["table"] = _infer_global_page_offset(item_pages=table_pages, slot_pages=table_slot_pages, search_range=(-50, 50))

                    if appendix_slot_pages:
                        appendix_pages: List[int] = []
                        for ii in unused_item_indices:
                            it = items[ii]
                            if not isinstance(it, dict):
                                continue
                            pn = _parse_page_number(it.get("pageNumber"))
                            if pn is None:
                                continue
                            if _cluster_of_page(int(pn)) != cluster_name:
                                continue
                            if item_layout_by_idx.get(int(ii)) == "appendix":
                                appendix_pages.append(int(pn))
                        if appendix_pages:
                            out["appendix"] = _infer_global_page_offset(item_pages=appendix_pages, slot_pages=appendix_slot_pages, search_range=(-50, 50))

                    return out

                cluster_offsets: Dict[str, Dict[str, Optional[int]]] = {}
                if cluster_split and inferred_offset_left is not None and inferred_offset_right is not None:
                    cluster_offsets["left"] = _infer_offsets_for("left")
                    cluster_offsets["right"] = _infer_offsets_for("right")
                else:
                    cluster_offsets["global"] = _infer_offsets_for("global")

                if debug_merge:
                    for cn, offs in cluster_offsets.items():
                        print(f"[MergeDebug] Phase2 page-guided(kind): cluster={cn} off_table={offs.get('table')} off_appendix={offs.get('appendix')} off_all={offs.get('all')}")

                # Map each item to closest base slot by page distance after applying offset.
                for i in unused_item_indices:
                    it = items[i]
                    if not isinstance(it, dict):
                        continue
                    pn = _parse_page_number(it.get("pageNumber"))
                    if pn is None:
                        continue

                    cn = _cluster_of_page(int(pn))
                    offs = cluster_offsets.get(cn, {"table": inferred_offset, "appendix": inferred_offset, "all": inferred_offset})

                    item_layout = item_layout_by_idx.get(int(i)) or _classify_item_layout(it)
                    off = offs.get(item_layout) if item_layout in ("table", "appendix") else offs.get("all")

                    # If unknown, pick the offset that yields smaller distance to ANY slot.
                    if item_layout == "unknown":
                        cand = []
                        for k in ("table", "appendix", "all"):
                            o = offs.get(k)
                            if o is None:
                                continue
                            lp = int(pn) - int(o)
                            dist = min(abs(int(lp) - int(sp)) for sp in base_slot_pages)
                            cand.append((dist, o))
                        if cand:
                            cand.sort(key=lambda t: t[0])
                            off = cand[0][1]

                    logical_p = int(pn) - int(off or 0)
                    best_t: Optional[int] = None
                    best_d: Optional[int] = None
                    for t_idx in base_slot_idx_order:
                        tp = _parse_page_number(entries[int(t_idx)].get("pageNumber"))
                        if tp is None:
                            continue
                        d = abs(int(logical_p) - int(tp))
                        # Penalize mismatched template kind to reduce systematic table/appendix swaps.
                        tk = template_kind.get(int(t_idx), "unknown")
                        penalty = 0
                        if item_layout != "unknown" and tk != "unknown" and item_layout != tk:
                            penalty = 4
                        d_eff = int(d) + int(penalty)
                        if best_d is None or d_eff < best_d:
                            best_d = d_eff
                            best_t = int(t_idx)
                    if best_t is not None:
                        item_to_template[int(i)] = int(best_t)
                        if debug_merge:
                            out_file = _safe_str(it.get("outFile")).strip()
                            bb = _extract_bbox_from_item(it)
                            if bb:
                                x0, y0, x1, y1 = bb
                                w = max(1, int(x1) - int(x0))
                                h = max(1, int(y1) - int(y0))
                                r = float(w) / float(h)
                                print(
                                    f"[MergeDebug] Phase2 bbox-assign: {out_file or ('#'+str(i))} pn={pn} cluster={cn} off={off} logical={logical_p} layout={item_layout} w={w} h={h} r={r:.3f} -> {entries[int(best_t)].get('jobTitle')}"
                                )
                            else:
                                print(
                                    f"[MergeDebug] Phase2 bbox-assign: {out_file or ('#'+str(i))} pn={pn} cluster={cn} off={off} logical={logical_p} layout={item_layout} -> {entries[int(best_t)].get('jobTitle')}"
                                )

                # Prefer a canonical screenshot for templates.
                # - For table templates: pick the EARLIEST physical page (smallest pageNumber)
                # - For appendix templates: pick the most CENTRAL physical page among items mapped to the same
                #   appendix template (median-based), to avoid outliers (e.g., one early page accidentally
                #   mapped to an appendix group whose real screenshots cluster at the end).
                # Tie-break by bbox height/area so we still prefer the more complete crop on the same page.
                try:
                    if base_template_indices and item_to_template:
                        first_pos_by_template: Dict[int, int] = {}
                        pos_by_item: Dict[int, int] = {int(ii): int(pos) for pos, ii in enumerate(unused_item_indices)}

                        for pos, ii in enumerate(unused_item_indices):
                            t = item_to_template.get(int(ii))
                            if t is None:
                                continue
                            if template_kind.get(int(t)) not in ("table", "appendix"):
                                continue
                            if int(t) not in first_pos_by_template:
                                first_pos_by_template[int(t)] = int(pos)

                        best_item_by_template: Dict[int, int] = {}
                        best_key_by_template: Dict[int, Tuple[int, int, int]] = {}

                        # Collect physical pages per template for appendix-median selection.
                        pages_by_template: Dict[int, List[int]] = {}
                        for ii in unused_item_indices:
                            t = item_to_template.get(int(ii))
                            if t is None:
                                continue
                            tk = template_kind.get(int(t))
                            if tk not in ("table", "appendix"):
                                continue
                            it = items[int(ii)]
                            if not isinstance(it, dict):
                                continue
                            pn = _parse_page_number(it.get("pageNumber"))
                            if not isinstance(pn, int) or pn <= 0:
                                continue
                            pages_by_template.setdefault(int(t), []).append(int(pn))

                        def _median_int(vals: List[int]) -> Optional[int]:
                            if not vals:
                                return None
                            s = sorted(int(v) for v in vals)
                            return int(s[len(s) // 2])
                        for ii in unused_item_indices:
                            t = item_to_template.get(int(ii))
                            if t is None:
                                continue
                            tk = template_kind.get(int(t))
                            if tk not in ("table", "appendix"):
                                continue
                            it = items[int(ii)]
                            if not isinstance(it, dict):
                                continue
                            pn = _parse_page_number(it.get("pageNumber"))
                            pn_k = int(pn) if isinstance(pn, int) and pn > 0 else 10**9
                            bb = _extract_bbox_from_item(it)
                            if not bb:
                                h = 0
                                area = 0
                            else:
                                x0, y0, x1, y1 = bb
                                w = max(1, int(x1) - int(x0))
                                h = max(1, int(y1) - int(y0))
                                area = int(w) * int(h)
                            if tk == "table":
                                # Smaller key is better: earlier page first; then larger bbox.
                                key = (pn_k, -int(h), -int(area))
                            else:
                                # For appendix: prefer the median page (central cluster), not the earliest.
                                med = _median_int(pages_by_template.get(int(t), []))
                                dist = abs(int(pn_k) - int(med)) if isinstance(med, int) else 10**9
                                # Smaller key is better: closer to median; then later page (slight bias); then larger bbox.
                                key = (int(dist), -int(pn_k), -int(h), -int(area))
                            prev = best_key_by_template.get(int(t))
                            if prev is None or key < prev:
                                best_key_by_template[int(t)] = key
                                best_item_by_template[int(t)] = int(ii)

                        swapped = 0
                        for t, first_pos in first_pos_by_template.items():
                            tk = template_kind.get(int(t))
                            if tk not in ("table", "appendix"):
                                continue
                            best_item = best_item_by_template.get(int(t))
                            if best_item is None:
                                continue
                            best_pos = pos_by_item.get(int(best_item))
                            if best_pos is None or int(best_pos) == int(first_pos):
                                continue
                            # Swap in-place to minimize disruption to overall physical order.
                            unused_item_indices[int(first_pos)], unused_item_indices[int(best_pos)] = (
                                unused_item_indices[int(best_pos)],
                                unused_item_indices[int(first_pos)],
                            )
                            # Update position map so later swaps are correct.
                            pos_by_item[int(unused_item_indices[int(first_pos)])] = int(first_pos)
                            pos_by_item[int(unused_item_indices[int(best_pos)])] = int(best_pos)
                            swapped += 1

                        if debug_merge and swapped > 0:
                            print(f"[MergeDebug] Phase2 canonical-template: swapped_templates={swapped}")
                except Exception:
                    pass

            # Normalize existing split entry titles within each template group so numbering is stable
            # and easy to verify in the app (avoid cases like 图5/图9/图13 then 图1/图2/图3).
            fig_re = re.compile(r"（图\s*(\d+)\s*）")
            split_counter: Dict[int, int] = {int(t_idx): 0 for t_idx in base_template_indices}
            for t_idx in base_template_indices:
                base_entry = entries[int(t_idx)]
                base_title = _safe_str(base_entry.get("jobTitle")).strip() or "附件"

                # Renumber by KB JSON order (stable & intuitive for debug/verification).
                group_idxs = list(template_groups.get(int(t_idx), []))
                split_idxs: List[int] = [
                    int(gi)
                    for gi in group_idxs
                    if int(gi) != int(t_idx) and _is_split_entry(entries[int(gi)])
                ]

                for n, gi in enumerate(split_idxs, start=1):
                    entries[int(gi)]["jobTitle"] = f"{base_title}（图{n}）"
                    # Avoid duplicated full-text search hits from split entries.
                    # Search is based on KnowledgeEntity.contentNormalized.
                    entries[int(gi)]["contentNormalized"] = ""

                split_counter[int(t_idx)] = int(len(split_idxs))

            # Print slot list after renumbering so logs match final titles shown in the app.
            if debug_merge and len(empty_appendix_indices) <= 60:
                slot_lines_after: List[str] = []
                for i in empty_appendix_indices:
                    e = entries[int(i)]
                    job_title = _safe_str(e.get('jobTitle')).strip()
                    job_title = job_title.replace("\\n", " ")[:80]
                    slot_lines_after.append(
                        "  - idx={idx} entryId={eid} jobTitle={jt}".format(
                            idx=i, eid=_safe_str(e.get('entryId')).strip(), jt=job_title
                        )
                    )
                print("[MergeDebug] Phase2 槽位列表(重编号后):\n" + "\n".join(slot_lines_after))
            # Fallback round-robin across base templates (not across all slots).
            template_rr = 0

            # Track per-table-template primary physical page within THIS run.
            # This avoids relying on heuristics/URI parsing after we cleared and repacked images.
            table_primary_physical_page: Dict[int, int] = {}

            def _physical_page_from_asset_uri(uri: str) -> Optional[int]:
                # Expect patterns like: .../legend_p16_idx1.png
                s = _safe_str(uri).strip()
                if not s:
                    return None
                m = re.search(r"_p(\d+)_", s)
                if not m:
                    return None
                try:
                    return int(m.group(1))
                except Exception:
                    return None

            def _first_table_snapshot_uri(e: Dict[str, Any]) -> str:
                blocks = e.get("blocks")
                if not isinstance(blocks, list):
                    return ""
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    if _safe_str(b.get("type")).strip().lower() != "table":
                        continue
                    u = _safe_str(b.get("imageUri") or b.get("image_uri")).strip()
                    if u:
                        return u
                return ""

            for pos, item_idx in enumerate(unused_item_indices):
                item = items[item_idx]
                if not isinstance(item, dict):
                    continue

                asset_uri = _pick_first_str(item, ("assetUri", "asset_uri", "imageUri", "image_uri", "src")).strip()
                target_entry_idx: Optional[int] = None
                created_new_entry = False

                # Pick which base template this item should attach to.
                if base_template_indices:
                    template_idx = item_to_template.get(int(item_idx))
                    if template_idx is None:
                        template_idx = int(base_template_indices[template_rr % len(base_template_indices)])
                        template_rr += 1
                else:
                    template_idx = int(empty_appendix_indices[template_rr % slot_count])
                    template_rr += 1

                # Continuation override (跨页表格兜底)：
                # 如果某个表格模板已经绑定了主截图(例如 p16)，而当前 item 在相邻页(p17)，
                # 则强制把它归到该表格模板，而不是被 bbox/layout 误分到附件。
                try:
                    if base_template_indices:
                        item_pn = _parse_page_number(item.get("pageNumber"))
                        if isinstance(item_pn, int) and item_pn > 0:
                            cand_templates: List[int] = []
                            for t in base_template_indices:
                                if template_kind.get(int(t)) != "table":
                                    continue
                                primary_p = table_primary_physical_page.get(int(t))
                                if isinstance(primary_p, int) and abs(int(item_pn) - int(primary_p)) == 1:
                                    cand_templates.append(int(t))
                            if len(cand_templates) == 1:
                                template_idx = int(cand_templates[0])
                except Exception:
                    pass

                # Special-case: table split stitching.
                # If the base table entry already has a snapshot (e.g. page 16) and this item is on the adjacent
                # page (e.g. page 17), append it to the SAME base entry as an image block so UI shows both halves.
                # This intentionally overrides one-image-per-entry for the specific “跨页表格” case.
                try:
                    if base_template_indices and template_kind.get(int(template_idx)) == "table":
                        item_pn = _parse_page_number(item.get("pageNumber"))
                        primary_p = table_primary_physical_page.get(int(template_idx))
                        if isinstance(primary_p, int) and isinstance(item_pn, int) and abs(int(item_pn) - int(primary_p)) == 1:
                            target_entry_idx = int(template_idx)
                except Exception:
                    pass

                # Appendix-like assets should stay inside the original appendix entry, ordered by marker.
                # Do not create detached __p3_split carriers for them.
                try:
                    if base_template_indices and template_kind.get(int(template_idx)) == "appendix":
                        target_entry_idx = int(template_idx)
                        created_new_entry = False
                except Exception:
                    pass

                if target_entry_idx is None and base_template_indices and template_idx in available_by_template and available_by_template[template_idx]:
                    target_entry_idx = int(available_by_template[template_idx].pop(0))
                elif (not base_template_indices) and available_by_template and any(available_by_template.values()):
                    # Should not happen, but be defensive.
                    for k, v in available_by_template.items():
                        if v:
                            target_entry_idx = int(v.pop(0))
                            template_idx = int(k)
                            break
                else:
                    template_entry = entries[int(template_idx)]

                    new_entry = copy.deepcopy(template_entry)
                    _strip_images_from_entry(new_entry)

                    # Split entries are image-carriers; keep base entry searchable, but not splits.
                    new_entry["contentNormalized"] = ""

                    base_eid = _safe_str(template_entry.get("entryId")).strip() or f"p3_slot_{template_idx}"
                    new_entry["entryId"] = _make_unique_entry_id(f"{base_eid}__p3_split")

                    base_title = _safe_str(template_entry.get("jobTitle")).strip() or "附件"
                    split_counter[int(template_idx)] = int(split_counter.get(int(template_idx), 0)) + 1
                    new_entry["jobTitle"] = f"{base_title}（图{split_counter[int(template_idx)]}）"

                    new_entry["position"] = int(next_position)
                    next_position += 1

                    entries.append(new_entry)
                    target_entry_idx = len(entries) - 1
                    created_new_entry = True

                if target_entry_idx is None:
                    continue

                ok = apply_merge(item_idx, item, int(target_entry_idx))
                if ok:
                    used_manifest_indices.add(int(item_idx))

                    # Record primary page for table templates when the base template got its first table snapshot.
                    try:
                        if base_template_indices and int(target_entry_idx) == int(template_idx) and template_kind.get(int(template_idx)) == "table":
                            if int(template_idx) not in table_primary_physical_page:
                                pn = _parse_page_number(item.get("pageNumber"))
                                if isinstance(pn, int) and pn > 0:
                                    table_primary_physical_page[int(template_idx)] = int(pn)
                    except Exception:
                        pass

                    out_file = _safe_str(item.get("outFile")).strip()
                    jt = _safe_str(entries[int(target_entry_idx)].get("jobTitle")).strip()
                    img = out_file or asset_uri or "<item>"
                    suffix = "(split)" if created_new_entry else ""
                    print(f"[MergeP3] 强行缝合{suffix}: {img} -> {jt}")
                elif debug_merge:
                    out_file = _safe_str(item.get("outFile")).strip()
                    img = out_file or asset_uri or "<item>"
                    print(f"[MergeDebug] Phase2 无法缝合: {img}")

            # --- 后处理：跨页表格合并（p16+p17 这类上下半表格） ---
            # 目标：
            # - base 表格#N 的 table.imageUri 绑定上半部分（较小页码）
            # - 下半部分（相邻页、且 idx1）追加为一个 ImageBlock 紧跟在 table block 之后
            # - 删除对应的 split 表格条目，避免出现“表格#1”和“表格#1（图1）”内容相同但截图分裂的问题

            def _append_image_block_after_first_table(e: Dict[str, Any], image_uri: str, page_number: Optional[int]) -> bool:
                blocks = e.get("blocks")
                if not isinstance(blocks, list):
                    return False
                insert_at = None
                for idx, b in enumerate(blocks):
                    if not isinstance(b, dict):
                        continue
                    if _safe_str(b.get("type")).strip().lower() == "table":
                        insert_at = int(idx) + 1
                        break
                if insert_at is None:
                    return False
                pn = int(page_number) if isinstance(page_number, int) and page_number > 0 else int(_physical_page_from_asset_uri(image_uri) or 1)
                img_block = _make_image_block(asset_uri=image_uri, page_number=pn, caption="表格原件（续）")
                blocks.insert(int(insert_at), img_block)
                e["blocks"] = blocks
                return True

            try:
                # Build a quick index: entryId -> index (for deletion after scanning)
                id_to_idx: Dict[str, int] = {}
                for i, e in enumerate(entries):
                    eid = _safe_str(e.get("entryId")).strip()
                    if eid:
                        id_to_idx[eid] = int(i)

                to_delete_indices: Set[int] = set()
                merged = 0

                for t in base_template_indices:
                    if template_kind.get(int(t)) != "table":
                        continue

                    base_entry = entries[int(t)]
                    base_eid = _safe_str(base_entry.get("entryId")).strip()
                    if not base_eid:
                        continue

                    primary_uri = _first_table_snapshot_uri(base_entry)
                    if not primary_uri:
                        continue
                    primary_p = _physical_page_from_asset_uri(primary_uri)
                    if not isinstance(primary_p, int) or primary_p <= 0:
                        continue

                    # We only merge the most common case: next page idx1.
                    cont_name = f"legend_p{int(primary_p)+1}_idx1"

                    # Find split entries for this base table by scanning ALL entries.
                    # (template_groups only contains pre-existing slots and does not include newly created splits.)
                    for gi, e2 in enumerate(entries):
                        if int(gi) == int(t):
                            continue
                        eid2 = _safe_str(e2.get("entryId")).strip()
                        if not eid2.startswith(base_eid + "__p3_split"):
                            continue
                        u2 = _first_table_snapshot_uri(e2)
                        if not u2:
                            continue
                        if cont_name not in _safe_str(u2):
                            continue

                        # Append to base and delete the split carrier.
                        ok = _append_image_block_after_first_table(base_entry, u2, page_number=int(primary_p) + 1)
                        if ok:
                            to_delete_indices.add(int(gi))
                            merged += 1
                        break

                if to_delete_indices:
                    # Delete from back to front to keep indices stable.
                    for di in sorted(to_delete_indices, reverse=True):
                        try:
                            entries.pop(int(di))
                        except Exception:
                            pass
                    if debug_merge:
                        print(f"[MergeDebug] Phase2 cross-page-table: merged={merged} deleted_splits={len(to_delete_indices)}")

                # Cleanup: remove split entries that end up with no snapshot at all.
                # This can happen when we delete/move snapshots during stitching.
                try:
                    empty_splits: List[int] = []
                    for i, e in enumerate(entries):
                        eid = _safe_str(e.get("entryId")).strip()
                        jt = _safe_str(e.get("jobTitle")).strip()
                        is_split = ("__p3_split" in eid) or ("（图" in jt)
                        if not is_split:
                            continue
                        has_any_uri = False
                        blocks = e.get("blocks")
                        if isinstance(blocks, list):
                            for b in blocks:
                                if not isinstance(b, dict):
                                    continue
                                u = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                                if u:
                                    has_any_uri = True
                                    break
                        if not has_any_uri:
                            empty_splits.append(int(i))
                    if empty_splits:
                        for di in sorted(empty_splits, reverse=True):
                            try:
                                entries.pop(int(di))
                            except Exception:
                                pass
                        if debug_merge:
                            print(f"[MergeDebug] Phase2 cleanup: removed_empty_splits={len(empty_splits)}")
                except Exception:
                    pass

                # --- 后处理：同页附件多截图合并（例如 p30 idx1+idx2 同属“附件1”） ---
                # 目标：
                # - 当同一个附件模板(base)在同一物理页出现多个截图（idx1/idx2/...）时，把它们追加到 base 条目的 blocks 中
                # - 删除对应 split carrier，避免列表/搜索出现“附件1（图1）/附件1（图2）”等碎片

                def _parse_legend_page_idx(uri: str) -> Tuple[Optional[int], Optional[int]]:
                    s = _safe_str(uri).strip()
                    if not s:
                        return (None, None)
                    m = re.search(r"_p(\d+)_idx(\d+)", s)
                    if not m:
                        return (None, None)
                    try:
                        return (int(m.group(1)), int(m.group(2)))
                    except Exception:
                        return (None, None)

                def _collect_image_uris_from_entry(e: Dict[str, Any]) -> List[str]:
                    out: List[str] = []
                    blocks = e.get("blocks")
                    if not isinstance(blocks, list):
                        return out
                    for b in blocks:
                        if not isinstance(b, dict):
                            continue
                        t = _safe_str(b.get("type")).strip().lower()
                        if t == "image":
                            u = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                            if u:
                                out.append(u)
                        elif t == "table":
                            u = _safe_str(b.get("imageUri") or b.get("image_uri")).strip()
                            if u:
                                out.append(u)
                    return out

                def _append_image_block_to_end(e: Dict[str, Any], image_uri: str, caption: str) -> None:
                    blocks = e.get("blocks")
                    if not isinstance(blocks, list):
                        blocks = []
                    pn = int(_physical_page_from_asset_uri(image_uri) or 1)
                    blocks.append(_make_image_block(asset_uri=image_uri, page_number=pn, caption=caption))
                    e["blocks"] = blocks

                try:
                    to_delete_appendix_splits: Set[int] = set()
                    for t in base_template_indices:
                        if template_kind.get(int(t)) != "appendix":
                            continue
                        base_entry = entries[int(t)]
                        base_eid = _safe_str(base_entry.get("entryId")).strip()
                        if not base_eid:
                            continue

                        # Find all split entries for this appendix base.
                        group: List[Tuple[int, Dict[str, Any]]] = []
                        for gi, e2 in enumerate(entries):
                            if int(gi) == int(t):
                                continue
                            eid2 = _safe_str(e2.get("entryId")).strip()
                            if not eid2.startswith(base_eid + "__p3_split"):
                                continue
                            group.append((int(gi), e2))

                        if not group:
                            continue

                        existing_uris: Set[str] = set(_collect_image_uris_from_entry(base_entry))

                        # Also account for URIs already present on the base entry when deciding
                        # whether a physical page has multiple crops to merge.
                        base_page_to_has_image: Dict[int, bool] = {}
                        for u in list(existing_uris):
                            p, idx = _parse_legend_page_idx(u)
                            if p is None:
                                continue
                            base_page_to_has_image[int(p)] = True

                        # Group candidate URIs by physical page.
                        page_to_items: Dict[int, List[Tuple[int, int, str]]] = {}
                        # (split_index, idx, uri)
                        for gi, e2 in group:
                            for u in _collect_image_uris_from_entry(e2):
                                p, idx = _parse_legend_page_idx(u)
                                if p is None or idx is None:
                                    continue
                                page_to_items.setdefault(int(p), []).append((int(gi), int(idx), str(u)))

                        # Merge same-page multiple crops.
                        for p, triples in page_to_items.items():
                            # Merge when there are at least 2 crops on the same physical page,
                            # counting both base + split images.
                            has_base = bool(base_page_to_has_image.get(int(p)))
                            if (len(triples) + (1 if has_base else 0)) < 2:
                                continue
                            triples_sorted = sorted(triples, key=lambda t3: (int(t3[1]), _safe_str(t3[2])))
                            # Append in idx order, avoid duplicates.
                            for gi, idx, u in triples_sorted:
                                if u in existing_uris:
                                    continue
                                _append_image_block_to_end(base_entry, u, caption="附件原件（续）")
                                existing_uris.add(u)

                            # Delete split carriers for this page once merged.
                            for gi, idx, u in triples_sorted:
                                to_delete_appendix_splits.add(int(gi))

                    if to_delete_appendix_splits:
                        for di in sorted(to_delete_appendix_splits, reverse=True):
                            try:
                                entries.pop(int(di))
                            except Exception:
                                pass
                        if debug_merge:
                            print(f"[MergeDebug] Phase2 appendix-same-page: deleted_splits={len(to_delete_appendix_splits)}")
                except Exception:
                    pass

                # Final appendix compaction:
                # Collapse base appendix entry + all appendix __p3_split carriers back into the base entry,
                # keeping images near successive [[附件X]] markers instead of detached split items.
                try:
                    appendix_delete_indices: Set[int] = set()
                    compacted_count = 0
                    for t in base_template_indices:
                        if template_kind.get(int(t)) != "appendix":
                            continue

                        base_entry = entries[int(t)]
                        base_eid = _safe_str(base_entry.get("entryId")).strip()
                        base_title = _safe_str(base_entry.get("jobTitle")).strip()
                        if not base_eid or not base_title:
                            continue

                        image_items: List[Tuple[int, int, str]] = []
                        related_indices: List[int] = [int(t)]
                        for preserved_eid, preserved_items in preserved_phase1_images_by_eid.items():
                            if preserved_eid == base_eid or preserved_eid.startswith(base_eid + "__p3_split"):
                                image_items.extend(list(preserved_items))
                        for gi, e2 in enumerate(entries):
                            if not isinstance(e2, dict):
                                continue
                            eid2 = _safe_str(e2.get("entryId")).strip()
                            if eid2 != base_eid and not eid2.startswith(base_eid + "__p3_split"):
                                continue
                            related_indices.append(int(gi))
                            blocks2 = e2.get("blocks")
                            if not isinstance(blocks2, list):
                                continue
                            for b in blocks2:
                                if not isinstance(b, dict):
                                    continue
                                if _safe_str(b.get("type")).strip().lower() != "image":
                                    continue
                                uri = _safe_str(b.get("imageUri") or b.get("image_uri") or b.get("src")).strip()
                                if not uri:
                                    continue
                                m = re.search(r"_p(\d+)_idx(\d+)", uri)
                                if m:
                                    try:
                                        page = int(m.group(1))
                                        img_idx = int(m.group(2))
                                    except Exception:
                                        page = int(_parse_page_number(b.get("pageNumber")) or 1)
                                        img_idx = 1
                                else:
                                    page = int(_parse_page_number(b.get("pageNumber")) or 1)
                                    img_idx = 1
                                image_items.append((page, img_idx, uri))

                        dedup_seen: Set[str] = set()
                        dedup_items: List[Tuple[int, int, str]] = []
                        for page, img_idx, uri in sorted(image_items, key=lambda it: (int(it[0]), int(it[1]), _safe_str(it[2]))):
                            if uri in dedup_seen:
                                continue
                            dedup_seen.add(uri)
                            dedup_items.append((page, img_idx, uri))

                        if not dedup_items:
                            continue

                        clusters = _cluster_sorted_image_uris_by_page(dedup_items, gap_threshold=5)
                        rebuilt_blocks = _rebuild_appendix_blocks_from_content(
                            content_markdown=_safe_str(base_entry.get("contentMarkdown")),
                            page_number=int(_parse_page_number(base_entry.get("pageNumber")) or 1),
                            marker_label=base_title,
                            image_clusters=clusters,
                        )
                        if not rebuilt_blocks:
                            continue

                        base_entry["blocks"] = rebuilt_blocks
                        compacted_count += 1

                        for gi in related_indices:
                            if int(gi) == int(t):
                                continue
                            appendix_delete_indices.add(int(gi))

                    if appendix_delete_indices:
                        for di in sorted(appendix_delete_indices, reverse=True):
                            try:
                                entries.pop(int(di))
                            except Exception:
                                pass
                    if debug_merge and (compacted_count > 0 or appendix_delete_indices):
                        print(
                            f"[MergeDebug] Phase2 appendix-compaction: compacted={compacted_count} deleted_splits={len(appendix_delete_indices)}"
                        )
                except Exception:
                    pass
            except Exception:
                pass

    post_reroute = _reroute_existing_legend_images_into_tables(
        kb=kb,
        manifest=manifest,
        dry_run=dry_run,
        debug_merge=debug_merge,
    )
    post_reroute = int(post_reroute or 0)
    if debug_merge and post_reroute > 0:
        print(f"[MergeDebug] post_table_reroute={post_reroute}")

    bridged_prev_pages = _bridge_previous_legend_page_into_tables(
        kb=kb,
        dry_run=dry_run,
        debug_merge=debug_merge,
    )
    bridged_prev_pages = int(bridged_prev_pages or 0)
    if debug_merge and bridged_prev_pages > 0:
        print(f"[MergeDebug] bridged_prev_table_pages={bridged_prev_pages}")

    try:
        if isinstance(source_native_kb, dict):
            _overlay_structure_bboxes_from_source(kb, source_native_kb, debug=bool(debug_merge))
    except Exception:
        pass

    try:
        figure_rebinds = _rebind_figure_images_to_reference_entries(kb=kb, debug_merge=debug_merge)
        if debug_merge and figure_rebinds > 0:
            print(f"[MergeDebug] figure_reference_rebinds={figure_rebinds}")
    except Exception:
        pass

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Merge docx/pdf manifest items into a per-document knowledge_base.json by pageNumber. "
            "Recommended layout: app/src/main/assets/kb/<fileId>/knowledge_base.json"
        )
    )
    p.add_argument(
        "--file-id",
        default="",
        help="If provided, default --kb/--out to app/src/main/assets/kb/<fileId>/knowledge_base.json",
    )

    p.add_argument(
        "--overrides",
        default="",
        help=(
            "Optional overrides JSON path. JSON maps image file name (e.g., legend_p18_idx1.png) -> target entryId. "
            "If omitted and --file-id is set, defaults to app/src/main/assets/kb/<fileId>/overrides.json if it exists."
        ),
    )
    p.add_argument(
        "--assets-root",
        default="app/src/main/assets/kb",
        help="Assets KB root (default: app/src/main/assets/kb)",
    )
    p.add_argument(
        "--kb",
        default="app/src/main/assets/documents/knowledge_base.json",
        help="Path to knowledge_base.json (default: app/src/main/assets/documents/knowledge_base.json)",
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest JSON (e.g. docx_shapes_manifest.json)",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output KB path. If omitted, updates --kb in-place (atomic).",
    )
    p.add_argument(
        "--insert-mode",
        default="after-page",
        choices=["append", "after-page", "after-page-text"],
        help="Where to insert the new image block in the entry's blocks list.",
    )
    p.add_argument(
        "--page-match",
        default="best",
        choices=["best", "all"],
        help="If multiple entries share the same pageNumber, either pick best (default) or apply to all.",
    )
    p.add_argument(
        "--prefer-heading",
        action="store_true",
        help="When pageNumber matches multiple entries, prefer the entry whose title/contentNormalized contains manifest headingText.",
    )
    p.add_argument(
        "--no-prefer-heading",
        action="store_true",
        help="Disable headingText matching preference.",
    )
    p.add_argument(
        "--scope-unit-name",
        action="append",
        default=[],
        help="Only consider entries whose unitName contains this string. Can be provided multiple times.",
    )
    p.add_argument(
        "--scope-job-title",
        action="append",
        default=[],
        help="Only consider entries whose jobTitle contains this string. Can be provided multiple times.",
    )
    p.add_argument(
        "--scope-entry-id-prefix",
        action="append",
        default=[],
        help="Only consider entries whose entryId starts with this prefix. Can be provided multiple times.",
    )
    p.add_argument(
        "--gap-max-pages",
        type=int,
        default=3,
        help="P2 空标题降级：前后已命中页码的最大关联跨度（页数）（default: 3）",
    )
    p.add_argument(
        "--enable-appendix-fill",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="P3 空标题兜底：是否开启‘附件/附表槽位’顺序自动填充（default: True）",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="P1 语义匹配敏感度（Levenshtein ratio cutoff），范围[0,1]（default: 0.5）",
    )
    p.add_argument(
        "--debug-merge",
        action="store_true",
        help="打印合并阶段调试信息（Phase1/Phase2 计数、槽位与余料样例）",
    )
    p.add_argument("--dry-run", action="store_true", help="Compute changes but do not write output.")
    return p


def main(argv: List[str]) -> int:
    args = build_arg_parser().parse_args(argv)

    DEFAULT_KB = "app/src/main/assets/documents/knowledge_base.json"
    kb_path = (args.kb or "").strip() or DEFAULT_KB
    manifest_path = args.manifest

    if (args.file_id or "").strip() and kb_path == DEFAULT_KB:
        # PATH-ONLY: Use for directory/asset names.
        safe_file_id = _sanitize_folder_name(args.file_id)
        kb_path = os.path.join((args.assets_root or "app/src/main/assets/kb").strip(), safe_file_id, "knowledge_base.json")

    # Resolve overrides path.
    overrides_path = (args.overrides or "").strip()
    if not overrides_path and (args.file_id or "").strip():
        try:
            safe_file_id = _sanitize_folder_name(args.file_id)
            candidate = os.path.join((args.assets_root or "app/src/main/assets/kb").strip(), safe_file_id, "overrides.json")
            if os.path.exists(candidate):
                overrides_path = candidate
        except Exception:
            pass

    out_path = args.out.strip() or kb_path

    kb = _read_json(kb_path)
    manifest = _read_json(manifest_path)
    source_native_kb: Optional[Dict[str, Any]] = None
    try:
        source_kb_path = _safe_str(manifest.get("sourceKb") or manifest.get("source_pdf")).strip() if isinstance(manifest, dict) else ""
        if source_kb_path and os.path.exists(source_kb_path):
            source_native_kb = _read_json(source_kb_path)
    except Exception:
        source_native_kb = None

    # Auto-detect PDF-native manifest (produced by PDF crop pipeline).
    pdf_native_detected = False
    try:
        if isinstance(manifest, dict) and manifest.get('source_pdf'):
            pdf_native_detected = True
        else:
            items_check = manifest.get('items') if isinstance(manifest, dict) else None
            if isinstance(items_check, list):
                for it in items_check:
                    if not isinstance(it, dict):
                        continue
                    of = _safe_str(it.get('outFile') or it.get('assetUri') or it.get('imageUri') or it.get('src'))
                    if re.search(r'_p\d+_idx\d+', of) or re.search(r'legend_p\d+', of) or re.search(r'table_p\d+', of) or re.search(r'visual_p\d+_\d+', of):
                        pdf_native_detected = True
                        break
    except Exception:
        pdf_native_detected = False

    scope = MergeScope(
        unit_name_contains=[s for s in (args.scope_unit_name or []) if s],
        job_title_contains=[s for s in (args.scope_job_title or []) if s],
        entry_id_prefix=[s for s in (args.scope_entry_id_prefix or []) if s],
    )
    scope_obj: Optional[MergeScope] = scope if (scope.unit_name_contains or scope.job_title_contains or scope.entry_id_prefix) else None

    stats = merge_manifest_into_kb(
        kb=kb,
        manifest=manifest,
        insert_mode=args.insert_mode,
        page_match=args.page_match,
        prefer_heading=(False if args.no_prefer_heading else True),
        dry_run=args.dry_run,
        scope=scope_obj,
        gap_max_pages=int(args.gap_max_pages),
        enable_appendix_fill=bool(args.enable_appendix_fill),
        similarity_threshold=float(args.similarity_threshold),
        debug_merge=bool(args.debug_merge),
        assets_root=(args.assets_root or ""),
        pdf_native=bool(pdf_native_detected),
        source_native_kb=source_native_kb,
    )

    try:
        table_snapshot_rebinds = _rebind_table_snapshot_uris_by_content(
            kb=kb,
            manifest=manifest,
            debug_merge=bool(args.debug_merge),
        )
        if bool(args.debug_merge) and table_snapshot_rebinds > 0:
            print(f"[MergeDebug] table_snapshot_rebinds={table_snapshot_rebinds}")
        bridged_table_snapshot_pages = _bridge_table_snapshot_prev_page_figure_continuations(
            kb=kb,
            debug_merge=bool(args.debug_merge),
        )
        if bool(args.debug_merge) and bridged_table_snapshot_pages > 0:
            print(f"[MergeDebug] bridged_table_snapshot_pages={bridged_table_snapshot_pages}")
        _cleanup_empty_split_entries(kb, debug=bool(args.debug_merge))
        _sort_table_entry_images_after_merge(kb, debug=bool(args.debug_merge))
    except Exception:
        pass

    # Apply optional manual overrides after automatic merge.
    try:
        overrides = _load_overrides(overrides_path)
        if overrides:
            moved = _apply_overrides_to_kb(kb, overrides, debug=bool(args.debug_merge))
            if bool(args.debug_merge):
                print(f"[MergeOverride] applied_overrides={len(overrides)} moved_blocks={moved} path={overrides_path}")
            # Cleanup after moves (may leave empty split shells).
            _cleanup_empty_split_entries(kb, debug=bool(args.debug_merge))
            # Deterministic ordering for table entries after moving blocks.
            _sort_table_entry_images_after_merge(kb, debug=bool(args.debug_merge))
    except Exception:
        pass

    _normalize_visual_snapshot_uris(kb, debug=bool(args.debug_merge))
    _rebuild_entry_figure_nodes(kb, debug=bool(args.debug_merge))
    collapsed_figure_only_entries = _collapse_figure_only_entries(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and collapsed_figure_only_entries:
        print(f"[MergeFigureNode] collapsed_figure_only_entries={collapsed_figure_only_entries}")
    retagged_numbered_callouts = _retag_misclassified_numbered_figure_callouts(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and retagged_numbered_callouts:
        print(f"[MergeFigureNode] retagged_numbered_callouts={retagged_numbered_callouts}")
    recovered_placeholder_bodies = _extract_body_blocks_from_empty_figure_placeholder_entries(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and recovered_placeholder_bodies:
        print(f"[MergeRecover] recovered_placeholder_entries={recovered_placeholder_bodies}")
    stripped_stale_body_bindings = _strip_stale_figure_annotation_bindings_from_body_blocks(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and stripped_stale_body_bindings:
        print(f"[MergeRecover] stripped_stale_body_bindings={stripped_stale_body_bindings}")
    repaired_degraded_clauses = 0
    if isinstance(source_native_kb, dict):
        repaired_degraded_clauses = _repair_degraded_numbered_entries_from_source_sections(
            kb,
            source_native_kb,
            debug=bool(args.debug_merge),
        )
    if bool(args.debug_merge) and repaired_degraded_clauses:
        print(f"[MergeRecover] repaired_degraded_clauses={repaired_degraded_clauses}")
    synced_source_clauses = 0
    if isinstance(source_native_kb, dict):
        synced_source_clauses = _sync_top_level_entries_from_source_sections(
            kb,
            source_native_kb,
            debug=bool(args.debug_merge),
        )
    if bool(args.debug_merge) and synced_source_clauses:
        print(f"[MergeRecover] synced_source_clauses={synced_source_clauses}")
    repaired_page_boundary_entries = 0
    if isinstance(source_native_kb, dict):
        repaired_page_boundary_entries = _repair_orphaned_page_boundary_figure_entries(
            kb,
            source_native_kb,
            screenshot_dir=os.path.join(os.path.dirname(out_path), "截图"),
            debug=bool(args.debug_merge),
        )
    if bool(args.debug_merge) and repaired_page_boundary_entries:
        print(f"[MergeRecover] repaired_page_boundary_entries={repaired_page_boundary_entries}")
    split_multi_item_entries = _split_multi_item_text_entries(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and split_multi_item_entries:
        print(f"[MergeSplit] created_split_entries={split_multi_item_entries}")
    _rebuild_entry_figure_nodes(kb, debug=bool(args.debug_merge))
    pruned_unreferenced_figure_nodes = _prune_unreferenced_image_only_figure_nodes(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and pruned_unreferenced_figure_nodes:
        print(f"[MergeFigureNode] pruned_unreferenced_image_nodes={pruned_unreferenced_figure_nodes}")
    _sanitize_mixed_text_entry_artifact_tails(kb, debug=bool(args.debug_merge))
    _sanitize_figure_managed_entry_texts(kb, debug=bool(args.debug_merge))
    cleaned_text48_area_bindings = _cleanup_known_text48_area_bindings(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and cleaned_text48_area_bindings:
        print(f"[MergeFigureNode] cleaned_text48_area_bindings={cleaned_text48_area_bindings}")
    cleaned_wrong_image_anchors = _cleanup_known_wrong_image_anchors(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and cleaned_wrong_image_anchors:
        print(f"[MergeFigureNode] cleaned_wrong_image_anchors={cleaned_wrong_image_anchors}")
    cleaned_local_figure_nodes = _cleanup_known_orphaned_local_figure_nodes(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and cleaned_local_figure_nodes:
        print(f"[MergeFigureNode] cleaned_local_figure_nodes={cleaned_local_figure_nodes}")
    cleaned_text_only_figure_entries = _cleanup_known_text_only_figure_reference_entries(kb, debug=bool(args.debug_merge))
    if bool(args.debug_merge) and cleaned_text_only_figure_entries:
        print(f"[MergeFigureNode] cleaned_text_only_figure_entries={cleaned_text_only_figure_entries}")
    asset_alignment_metrics = _run_asset_alignment_check(
        kb,
        inserted_assets=stats.inserted_assets,
        debug=bool(args.debug_merge),
    )
    if asset_alignment_metrics.get("reattachedInsertedAssets", 0):
        _rebuild_entry_figure_nodes(kb, debug=bool(args.debug_merge))
    canonical_metrics = _canonicalize_figure_nodes_across_entries(kb, debug=bool(args.debug_merge))
    dedupe_metrics = _dedupe_figure_image_blocks(kb, debug=bool(args.debug_merge))
    removed_payloads = _strip_embedded_image_payloads(kb)
    if bool(args.debug_merge) and removed_payloads:
        print(f"[MergeCleanup] removed_embedded_image_payloads={removed_payloads}")
    metadata = kb.get("fileMetadata") if isinstance(kb.get("fileMetadata"), dict) else {}
    metadata["semanticAudit"] = _compute_semantic_audit(kb, dedupe_metrics=dedupe_metrics)
    metadata["assetAlignmentAudit"] = asset_alignment_metrics
    kb["fileMetadata"] = metadata
    _refresh_file_metadata(kb)

    if not args.dry_run:
        _atomic_write_json(out_path, kb)
        try:
            _prune_unreferenced_screenshot_files(kb, manifest, debug=bool(args.debug_merge))
        except Exception:
            pass

    msg = (
        "Merge complete:\n"
        f"  items_total={stats.items_total}\n"
        f"  items_with_page={stats.items_with_page}\n"
        f"  items_with_uri={stats.items_with_uri}\n"
        f"  inserted={stats.inserted}\n"
        f"  skipped_duplicate={stats.skipped_duplicate}\n"
        f"  skipped_no_entry_for_page={stats.skipped_no_entry_for_page}\n"
        f"  skipped_missing_fields={stats.skipped_missing_fields}\n"
        f"  skipped_out_of_scope={stats.skipped_out_of_scope}\n"
        f"  ambiguous_pages={stats.ambiguous_pages}\n"
        f"  asset_alignment_reattached={asset_alignment_metrics.get('reattachedInsertedAssets', 0)}\n"
        f"  asset_alignment_unresolved={asset_alignment_metrics.get('unresolvedInsertedAssets', 0)}\n"
        f"  canonical_figure_nodes={canonical_metrics.get('canonicalFigureNodes', 0)}\n"
        f"  canonical_merged_groups={canonical_metrics.get('canonicalFigureMergedGroups', 0)}\n"
        f"  duplicate_image_blocks_removed={dedupe_metrics.get('duplicateImageBlocksRemoved', 0)}\n"
        f"  duplicate_image_keys={dedupe_metrics.get('duplicateImageKeys', 0)}\n"
        f"  wrote={'NO (dry-run)' if args.dry_run else out_path}"
    )

    try:
        print(msg)
    except BrokenPipeError:
        try:
            sys.stdout.close()
        except Exception:
            pass
        return 0

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except BrokenPipeError:
        # Happens when output is piped and the consumer closes early
        # (e.g., PowerShell `| Select-Object -First N`). Treat as success.
        raise SystemExit(0)
