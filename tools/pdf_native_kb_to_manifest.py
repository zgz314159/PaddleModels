#!/usr/bin/env python3
"""Convert PDF-native KB output into a merge_manifest_to_kb-compatible manifest.

This utility is used by DOCX+PDF mode to reuse modern PDF-native cropping while
keeping the downstream legacy merge logic stable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

try:
    from utils.file_id_sanitizer import sanitize_asset_relative_path
except Exception:
    import sys

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from utils.file_id_sanitizer import sanitize_asset_relative_path


SHOTS_DIR_NAME = "\u622a\u56fe"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
_FIGURE_TABLE_LABEL_RE = re.compile(r"((?:附图|附表|图|表)\s*[0-9一二三四五六七八九十零〇O口]+(?:\s*[-—~〜\.．]\s*[0-9一二三四五六七八九十零〇O口]+)*)")
_REFERENCE_HINT_RE = re.compile(r"(见|如|参见|所示|示例图|示意图|接线图|布置图|明细表)")


def _safe_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _parse_positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        try:
            v = int(value)
            return v if v > 0 else None
        except Exception:
            return None
    text = _safe_str(value)
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        try:
            v = int(text)
            return v if v > 0 else None
        except Exception:
            return None
    return None


def _parse_non_negative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        try:
            v = int(value)
            return v if v >= 0 else None
        except Exception:
            return None
    text = _safe_str(value)
    if not text:
        return None
    if re.fullmatch(r"\d+", text):
        try:
            v = int(text)
            return v if v >= 0 else None
        except Exception:
            return None
    return None


def _basename_from_uri(uri: str) -> str:
    text = _safe_str(uri)
    if not text:
        return ""

    try:
        parsed = urlparse(text)
        if parsed.scheme:
            path = unquote(parsed.path or "")
            name = os.path.basename(path)
            return _safe_str(name)
    except Exception:
        pass

    text = text.split("?", 1)[0].split("#", 1)[0]
    text = text.replace("\\", "/")
    return _safe_str(os.path.basename(text))


def _parse_page_from_name(name: str) -> Optional[int]:
    m = re.search(r"_p(\d+)_", _safe_str(name), flags=re.IGNORECASE)
    if not m:
        return None
    try:
        v = int(m.group(1))
        return v if v > 0 else None
    except Exception:
        return None


def _normalize_kind(block_type: str) -> Optional[str]:
    t = _safe_str(block_type).lower()
    if t == "table":
        return "table"
    if t in {"figure", "equation", "image"}:
        return "legend"
    return None


def _looks_generic_caption(text: str) -> bool:
    t = _safe_str(text).lower()
    if not t:
        return True
    if re.fullmatch(r"table\s*\d+", t):
        return True
    if re.fullmatch(r"figure\s*\d+", t):
        return True
    if re.fullmatch(r"equation\s*\d+", t):
        return True
    if t in {"table", "figure", "equation", "image"}:
        return True
    return False


def _looks_generic_table_heading(text: str) -> bool:
    t = _safe_str(text)
    if not t:
        return True
    compact = re.sub(r"\s+", "", t)
    if compact in {"表", "明细表", "明细", "附表", "附图"}:
        return True
    if len(compact) <= 2 and compact.startswith("表"):
        return True
    return False


def _summarize_table_block(block: Dict[str, Any]) -> str:
    text = _safe_str(block.get("contentMarkdown"))
    if text:
        parts: List[str] = []
        for raw in text.splitlines():
            ln = _safe_str(raw)
            if not ln:
                continue
            if _looks_generic_table_heading(ln):
                continue
            parts.append(ln)
            if len(" ".join(parts)) >= 72 or len(parts) >= 6:
                break
        if parts:
            return " ".join(parts)[:120]

    rows = block.get("table_rows")
    if isinstance(rows, list):
        cells: List[str] = []
        for row in rows:
            if not isinstance(row, list):
                continue
            for cell in row:
                value = _safe_str(cell)
                if not value:
                    continue
                if _looks_generic_table_heading(value):
                    continue
                cells.append(value)
                if len(" ".join(cells)) >= 72 or len(cells) >= 10:
                    return " ".join(cells)[:120]
        if cells:
            return " ".join(cells)[:120]

    return ""


def _normalize_visual_label(text: str) -> str:
    label = _safe_str(text)
    if not label:
        return ""
    label = label.replace("附 图", "附图").replace("附 表", "附表")
    label = re.sub(r"[\s　]+", "", label)
    label = label.replace("—", "-").replace("－", "-").replace("～", "~").replace("〜", "~")
    label = label.replace("口", "0").replace("O", "0")
    return label


def _extract_visual_label(text: str) -> str:
    match = _FIGURE_TABLE_LABEL_RE.search(_safe_str(text))
    if not match:
        return ""
    return _normalize_visual_label(match.group(1))


def _bbox_vertical_gap(a: Dict[str, int], b: Dict[str, int]) -> int:
    if int(a.get("bottom", 0)) <= int(b.get("top", 0)):
        return int(b.get("top", 0)) - int(a.get("bottom", 0))
    if int(b.get("bottom", 0)) <= int(a.get("top", 0)):
        return int(a.get("top", 0)) - int(b.get("bottom", 0))
    return 0


def _bbox_horizontal_overlap_ratio(a: Dict[str, int], b: Dict[str, int]) -> float:
    left = max(int(a.get("left", 0)), int(b.get("left", 0)))
    right = min(int(a.get("right", 0)), int(b.get("right", 0)))
    overlap = max(0, right - left)
    base = min(int(a.get("width", 0)), int(b.get("width", 0)))
    if base <= 0:
        return 0.0
    return float(overlap) / float(base)


def _is_textual_block(block: Dict[str, Any]) -> bool:
    block_type = _safe_str(block.get("type")).lower()
    return block_type in {"code", "text", "markdown", "paragraph"}


def _collect_page_text_blocks(entry: Dict[str, Any], page_number: int) -> List[Dict[str, Any]]:
    blocks = entry.get("blocks")
    if not isinstance(blocks, list):
        return []
    out: List[Dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict) or not _is_textual_block(block):
            continue
        if _parse_positive_int(block.get("pageNumber")) != int(page_number):
            continue
        bbox = _extract_bbox(block)
        if bbox is None:
            continue
        text = _safe_str(block.get("code") or block.get("text") or block.get("contentMarkdown"))
        if not text:
            continue
        out.append({
            "text": text,
            "bbox": bbox,
            "readingOrder": _parse_positive_int(block.get("readingOrder")) or 10**6,
        })
    out.sort(key=lambda item: (int(item["readingOrder"]), int(item["bbox"]["top"]), int(item["bbox"]["left"])))
    return out


def _join_visual_text_context(text_blocks: List[Dict[str, Any]], index: int) -> str:
    if index < 0 or index >= len(text_blocks):
        return ""
    parts: List[str] = []
    current = text_blocks[index]
    current_text = _safe_str(current.get("text"))
    if not current_text:
        return ""
    current_has_label = bool(_extract_visual_label(current_text))

    if index > 0:
        prev = text_blocks[index - 1]
        prev_text = _safe_str(prev.get("text"))
        prev_gap = _bbox_vertical_gap(prev["bbox"], current["bbox"])
        if prev_text and prev_gap <= 60:
            prev_is_caption_fragment = len(prev_text) <= 24 and any(token in prev_text for token in ("示例图", "示意图", "接线图", "布置图", "明细表"))
            if prev_text.endswith(("见", "如", "参见")) or (current_has_label and prev_is_caption_fragment):
                parts.append(prev_text)

    parts.append(current_text)

    if index + 1 < len(text_blocks):
        nxt = text_blocks[index + 1]
        next_text = _safe_str(nxt.get("text"))
        next_gap = _bbox_vertical_gap(current["bbox"], nxt["bbox"])
        if next_text and next_gap <= 42 and len(next_text) <= 32:
            if _REFERENCE_HINT_RE.search(next_text) or _extract_visual_label(next_text) or len(current_text) <= 16:
                parts.append(next_text)

    merged = " ".join(part.strip() for part in parts if part and part.strip())
    return re.sub(r"\s+", " ", merged).strip()


def _score_text_block_for_visual_anchor(
    text_blocks: List[Dict[str, Any]],
    index: int,
    visual_bbox: Dict[str, int],
    *,
    kind: str,
    target_label: str,
    prefer_reference: bool,
) -> Tuple[int, str]:
    block = text_blocks[index]
    bbox = block["bbox"]
    text = _join_visual_text_context(text_blocks, index)
    if not text:
        return (-(10**9), "")

    score = 0
    gap = _bbox_vertical_gap(bbox, visual_bbox)
    overlap = _bbox_horizontal_overlap_ratio(bbox, visual_bbox)
    label = _extract_visual_label(text)
    normalized_text = _normalize_visual_label(text)

    if int(bbox.get("bottom", 0)) <= int(visual_bbox.get("top", 0)):
        score += 120
    elif int(bbox.get("top", 0)) >= int(visual_bbox.get("bottom", 0)):
        score += 85
    else:
        score += 70

    score -= min(260, int(gap))
    score += int(overlap * 90.0)

    if target_label and label:
        if label == target_label:
            score += 260
        elif target_label in normalized_text or label in target_label:
            score += 170
    elif label:
        score += 140

    if prefer_reference:
        if _REFERENCE_HINT_RE.search(text):
            score += 180
        if kind == "table" and "明细表" in text:
            score += 80
        if kind == "legend" and any(token in text for token in ("示例图", "示意图", "接线图", "布置图")):
            score += 70
    else:
        if label:
            score += 150
        if kind == "table" and any(token in text for token in ("明细表", "附表", "表")):
            score += 90
        if kind == "legend" and any(token in text for token in ("示例图", "示意图", "接线图", "布置图", "图")):
            score += 80
        if kind == "legend" and not label and not any(token in text for token in ("示例图", "示意图", "接线图", "布置图", "明细表")):
            score -= 220
        if kind == "table" and not label and "明细表" not in text and "表" not in text:
            score -= 180

    if len(text) <= 2:
        score -= 180
    return (score, text)


def _infer_visual_anchor_pair(entry: Dict[str, Any], kind: str, block: Dict[str, Any]) -> Tuple[str, str]:
    page_number = _parse_positive_int(block.get("pageNumber")) or _parse_positive_int(entry.get("pageNumber")) or 1
    visual_bbox = _extract_bbox(block)
    if visual_bbox is None:
        heading_text = _infer_heading_text(entry, kind, block)
        return (heading_text, heading_text)

    text_blocks = _collect_page_text_blocks(entry, int(page_number))
    if not text_blocks:
        heading_text = _infer_heading_text(entry, kind, block)
        return (heading_text, heading_text)

    target_label = _extract_visual_label(_safe_str(block.get("caption")))
    best_reference = ""
    best_reference_score = -(10**9)
    best_heading = ""
    best_heading_score = -(10**9)

    for index in range(len(text_blocks)):
        ref_score, ref_text = _score_text_block_for_visual_anchor(
            text_blocks,
            index,
            visual_bbox,
            kind=kind,
            target_label=target_label,
            prefer_reference=True,
        )
        if ref_score > best_reference_score:
            best_reference_score = ref_score
            best_reference = ref_text

        heading_score, heading_text = _score_text_block_for_visual_anchor(
            text_blocks,
            index,
            visual_bbox,
            kind=kind,
            target_label=target_label,
            prefer_reference=False,
        )
        if heading_score > best_heading_score:
            best_heading_score = heading_score
            best_heading = heading_text

    fallback_heading = _infer_heading_text(entry, kind, block)
    heading_text = best_heading or fallback_heading
    anchor_text = best_reference or heading_text or fallback_heading
    return (anchor_text[:160], heading_text[:160])


def _infer_heading_text(entry: Dict[str, Any], kind: str, block: Optional[Dict[str, Any]] = None) -> str:
    lines: List[str] = []
    markdown = _safe_str(entry.get("contentMarkdown"))
    if markdown:
        for raw in markdown.splitlines():
            ln = _safe_str(raw)
            if ln:
                lines.append(ln)

    if kind == "table":
        if isinstance(block, dict):
            table_summary = _summarize_table_block(block)
            if table_summary:
                return table_summary
        for ln in lines:
            if re.search(r"(^|\s)\u8868\s*\d", ln):
                return ln[:120]
            if ln.startswith("\u8868") and not _looks_generic_table_heading(ln):
                return ln[:120]
    if kind == "legend":
        for ln in lines:
            if re.search(r"(^|\s)\u56fe\s*\w*\d", ln):
                return ln[:120]
            if ln.startswith("\u56fe") or ln.startswith("\u9644\u56fe"):
                return ln[:120]

    title = _safe_str(entry.get("jobTitle"))
    if title:
        return title[:120]
    if lines:
        return lines[0][:120]
    return ""


def _cleanup_legacy_named_images(assets_dir: Path) -> int:
    removed = 0
    if not assets_dir.is_dir():
        return removed

    for p in assets_dir.iterdir():
        if not p.is_file():
            continue
        low = p.name.lower()
        if not low.endswith(IMAGE_EXTS):
            continue
        if not (low.startswith("table_p") or low.startswith("legend_p")):
            continue
        try:
            p.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def _resolve_source_image_path(assets_dir: Path, source_uri: str) -> Path:
    base = _basename_from_uri(source_uri)
    if base:
        candidate = assets_dir / base
        if candidate.is_file():
            return candidate

    parsed_path = ""
    try:
        parsed = urlparse(_safe_str(source_uri))
        parsed_path = unquote(parsed.path or "")
    except Exception:
        parsed_path = ""

    if parsed_path:
        name = os.path.basename(parsed_path)
        if name:
            candidate = assets_dir / name
            if candidate.is_file():
                return candidate

    return assets_dir / base if base else assets_dir / ""


def _extract_bbox(item: Dict[str, Any]) -> Optional[Dict[str, int]]:
    bbox = item.get("bbox")
    if isinstance(bbox, dict):
        left = _parse_non_negative_int(bbox.get("left"))
        top = _parse_non_negative_int(bbox.get("top"))
        width = _parse_positive_int(bbox.get("width"))
        height = _parse_positive_int(bbox.get("height"))
        if left is not None and top is not None and width is not None and height is not None:
            return {
                "left": int(left),
                "top": int(top),
                "right": int(left + width),
                "bottom": int(top + height),
                "width": int(width),
                "height": int(height),
            }

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            left = int(round(float(bbox[0])))
            top = int(round(float(bbox[1])))
            width = int(round(float(bbox[2])))
            height = int(round(float(bbox[3])))
            if width > 0 and height > 0:
                return {
                    "left": left,
                    "top": top,
                    "right": left + width,
                    "bottom": top + height,
                    "width": width,
                    "height": height,
                }
        except Exception:
            return None

    return None


def convert_pdf_native_kb_to_manifest(
    kb: Dict[str, Any],
    *,
    source_kb_path: Path,
    file_id: str,
    assets_root: Path,
    clean_legacy: bool,
) -> Dict[str, Any]:
    safe_file_id = sanitize_asset_relative_path(file_id)
    file_id_path = Path(*[seg for seg in safe_file_id.split("/") if seg])
    assets_dir = assets_root / file_id_path / SHOTS_DIR_NAME
    assets_dir.mkdir(parents=True, exist_ok=True)

    cleaned = _cleanup_legacy_named_images(assets_dir) if clean_legacy else 0

    entries = kb.get("entries")
    if not isinstance(entries, list):
        entries = []

    kind_page_counter: Dict[Tuple[str, int], int] = {}
    items: List[Dict[str, Any]] = []
    copied = 0
    missing_sources = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        entry_page = _parse_positive_int(entry.get("pageNumber"))
        blocks = entry.get("blocks")
        if not isinstance(blocks, list):
            continue

        for block in blocks:
            if not isinstance(block, dict):
                continue

            kind = _normalize_kind(_safe_str(block.get("type")))
            if not kind:
                continue

            source_uri = _safe_str(block.get("imageUri") or block.get("src") or block.get("uri"))
            if not source_uri:
                continue

            source_name = _basename_from_uri(source_uri)
            if not source_name:
                continue

            page_number = (
                _parse_positive_int(block.get("pageNumber"))
                or _parse_page_from_name(source_name)
                or entry_page
                or 1
            )

            key = (kind, int(page_number))
            next_idx = int(kind_page_counter.get(key, 0)) + 1
            kind_page_counter[key] = next_idx

            src_path = _resolve_source_image_path(assets_dir, source_uri)
            out_name = source_name
            out_path = assets_dir / out_name
            if src_path.is_file():
                try:
                    if not out_path.is_file():
                        shutil.copy2(src_path, out_path)
                        copied += 1
                except Exception:
                    missing_sources += 1
                    continue
            elif not out_path.is_file():
                missing_sources += 1
                continue

            asset_uri = f"file:///android_asset/kb/{safe_file_id}/{SHOTS_DIR_NAME}/{out_name}".replace("\\", "/")

            caption = _safe_str(block.get("caption"))
            anchor_text, heading_text = _infer_visual_anchor_pair(entry, kind, block)
            if not heading_text:
                heading_text = _infer_heading_text(entry, kind, block)
            if _looks_generic_caption(anchor_text):
                anchor_text = heading_text
            if not anchor_text:
                anchor_text = caption or _safe_str(entry.get("jobTitle"))

            item: Dict[str, Any] = {
                "kind": kind,
                "pageNumber": int(page_number),
                "sourceEntryId": _safe_str(entry.get("entryId")),
                "outFile": out_name,
                "assetUri": asset_uri,
                "method": "pdf_native_layout",
            }

            score = block.get("confidence")
            if isinstance(score, (int, float)):
                item["score"] = float(score)
            if anchor_text:
                item["anchorText"] = anchor_text
            if heading_text:
                item["headingText"] = heading_text

            bbox = _extract_bbox(block)
            if bbox is not None:
                item["bbox"] = bbox

            items.append(item)

    payload: Dict[str, Any] = {
        "version": 2,
        "fileId": safe_file_id,
        "generatedAt": int(time.time() * 1000),
        "root": str(assets_root.as_posix()),
        "pagesDir": str((assets_root / file_id_path / "pages").as_posix()),
        "outDir": str(assets_dir.as_posix()),
        "source_pdf": str(source_kb_path.as_posix()),
        "sourceKb": str(source_kb_path.as_posix()),
        "items": items,
        "written": int(len(items)),
        "stats": {
            "cleanedLegacy": int(cleaned),
            "copied": int(copied),
            "missingSources": int(missing_sources),
        },
    }
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert pdf_to_base64_kb output to legacy merge manifest.")
    ap.add_argument("--kb", required=True, help="Path to temporary PDF-native knowledge_base JSON")
    ap.add_argument("--file-id", required=True, help="KB fileId")
    ap.add_argument(
        "--assets-root",
        default=os.path.join("app", "src", "main", "assets", "kb"),
        help="Assets KB root folder",
    )
    ap.add_argument("--out", default=None, help="Output manifest path (default: <assets-root>/<file-id>/截图/manifest.json)")
    ap.add_argument(
        "--no-clean-legacy",
        action="store_true",
        help="Do not remove existing table_p*/legend_p* images before generating manifest",
    )
    return ap


def main(argv: List[str]) -> int:
    args = build_arg_parser().parse_args(argv)

    kb_path = Path(args.kb).expanduser().resolve()
    if not kb_path.is_file():
        raise SystemExit(f"KB JSON not found: {kb_path}")

    project_root = Path(__file__).resolve().parents[1]
    assets_root = Path(args.assets_root)
    if not assets_root.is_absolute():
        assets_root = (project_root / assets_root).resolve()

    safe_file_id = sanitize_asset_relative_path(_safe_str(args.file_id))
    file_id_path = Path(*[seg for seg in safe_file_id.split("/") if seg])
    default_out = assets_root / file_id_path / SHOTS_DIR_NAME / "manifest.json"
    out_path = Path(args.out).expanduser().resolve() if args.out else default_out

    kb = _read_json(kb_path)
    if not isinstance(kb, dict):
        raise SystemExit(f"Unexpected KB JSON structure: {kb_path}")

    payload = convert_pdf_native_kb_to_manifest(
        kb,
        source_kb_path=kb_path,
        file_id=safe_file_id,
        assets_root=assets_root,
        clean_legacy=not bool(args.no_clean_legacy),
    )
    _write_json(out_path, payload)

    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    print(
        f"Wrote manifest: {out_path} | items={payload.get('written', 0)} "
        f"copied={stats.get('copied', 0)} missing={stats.get('missingSources', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
