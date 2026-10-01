"""Figure/image asset adapter — Phase 2D trusted visual assets.

Owns embedded extraction, visual figure matching, page-crop fallback,
asset writing, and figure DocBlocks. NativeAdapter no longer creates images.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pipeline.canonical_ir import BBox, DocBlock, DocPage

try:
    import fitz
except ImportError:
    fitz = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

# --- Centralized thresholds (do not scatter into v2_runner) ---
# Page-area fraction at or above which a placed image is treated as full-page background.
BACKGROUND_AREA_FRACTION = 0.90
# Minimum absolute pixel dimensions for a non-fabricated bitmap asset.
MIN_PIXEL_DIMENSION = 8
# Minimum placed area (pt²) for an embedded image with no visual-figure support.
# Below this, decorations (rules, tiny icons) are dropped unless visual confirms.
MIN_DECORATION_AREA_PT2 = 120.0
# Edge zone as fraction of page height for header/footer icon heuristics.
EDGE_ZONE_FRACTION = 0.08
# Visual–embedded bbox overlap ratio (intersection / embedded area) to count as matched.
VISUAL_MATCH_OVERLAP = 0.35


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_hex(raw: Any) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.hex()
    if isinstance(raw, str):
        return raw
    return ""


def stable_figure_block_id(
    page_number: int,
    bbox: Sequence[float],
    content_digest: str,
) -> str:
    """Stable id from page + bbox + content digest (no wall-clock)."""
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    seed = f"{page_number}|{x0:.2f},{y0:.2f},{x1:.2f},{y1:.2f}|{content_digest}"
    return "fig_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def asset_filename_for_digest(content_digest: str, ext: str = "png") -> str:
    """Content-addressed asset filename (shared across locations)."""
    short = (content_digest or "empty")[:32]
    ext = (ext or "png").lower().lstrip(".")
    return f"img_{short}.{ext}"


def clamp_bbox_to_page(
    bbox: Sequence[float],
    page_width: float,
    page_height: float,
) -> Tuple[float, float, float, float]:
    x0 = max(0.0, float(bbox[0]))
    y0 = max(0.0, float(bbox[1]))
    x1 = min(float(page_width), float(bbox[2]))
    y1 = min(float(page_height), float(bbox[3]))
    if x1 <= x0:
        x1 = min(float(page_width), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(page_height), y0 + 1.0)
    return (x0, y0, x1, y1)


def should_exclude_embedded(
    bbox_xyxy: Sequence[float],
    page_width: float,
    page_height: float,
    *,
    pixel_width: int = 0,
    pixel_height: int = 0,
    visual_matched: bool = False,
) -> Optional[str]:
    """Return exclusion reason or None to keep the candidate.

    Pure policy — all thresholds live at module top.
    """
    if page_width <= 0 or page_height <= 0:
        return "invalid_page_size"
    if pixel_width is not None and pixel_height is not None:
        if pixel_width <= 0 or pixel_height <= 0:
            return "invalid_pixel_size"
        if pixel_width < MIN_PIXEL_DIMENSION or pixel_height < MIN_PIXEL_DIMENSION:
            # Tiny without visual support → decoration
            if not visual_matched:
                return "tiny_decoration"

    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy[:4])
    w = max(0.0, x1 - x0)
    h = max(0.0, y1 - y0)
    if w <= 0 or h <= 0:
        return "empty_bbox"

    area_frac = (w * h) / (page_width * page_height)
    if area_frac >= BACKGROUND_AREA_FRACTION and not visual_matched:
        # Full-page background; keep only if visual figure explicitly matched.
        return "full_page_background"

    area_pt2 = w * h
    if area_pt2 < MIN_DECORATION_AREA_PT2 and not visual_matched:
        return "tiny_decoration"

    # Header/footer edge icons: fully inside edge band and small.
    edge = EDGE_ZONE_FRACTION * page_height
    in_top = y1 <= edge
    in_bottom = y0 >= page_height - edge
    if (in_top or in_bottom) and not visual_matched and area_frac < 0.05:
        return "edge_icon"

    return None


def bbox_overlap_ratio(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Intersection area / area(a)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    return inter / area_a


def match_visual_to_embedded(
    visual_bbox_xywh: Sequence[float],
    embedded: Sequence[Dict[str, Any]],
    *,
    min_overlap: float = VISUAL_MATCH_OVERLAP,
) -> Optional[int]:
    """Return index of best embedded match for a visual bbox, or None."""
    vx0, vy0, vw, vh = visual_bbox_xywh
    v_xyxy = (vx0, vy0, vx0 + vw, vy0 + vh)
    best_i = None
    best_r = 0.0
    for i, emb in enumerate(embedded):
        eb = emb.get("bbox")
        if not eb:
            continue
        r = bbox_overlap_ratio(eb, v_xyxy)
        if r >= min_overlap and r > best_r:
            best_r = r
            best_i = i
    return best_i


def enumerate_embedded_images(page: Any, page_number: int) -> List[Dict[str, Any]]:
    """List embedded placed images on a page (PDF points bbox)."""
    if page is None:
        return []
    try:
        infos = page.get_image_info(hashes=True, xrefs=True)
    except Exception:
        try:
            infos = page.get_image_info(hashes=True)
        except Exception:
            return []
    out: List[Dict[str, Any]] = []
    for info in infos or []:
        bbox = info.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        out.append(
            {
                "page_number": page_number,
                "xref": info.get("xref"),
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                "pixel_width": int(info.get("width") or 0),
                "pixel_height": int(info.get("height") or 0),
                "digest": _digest_hex(info.get("digest")),
                "info": info,
            }
        )
    return out


def extract_embedded_bytes(doc: Any, xref: Optional[int]) -> Optional[Tuple[bytes, str]]:
    """Extract original image bytes + ext from PDF doc by xref."""
    if doc is None or not xref:
        return None
    try:
        base = doc.extract_image(int(xref))
        if base and base.get("image"):
            ext = str(base.get("ext") or "png").lower()
            return base["image"], ext
    except Exception:
        return None
    return None


def _mime_for_ext(ext: str) -> str:
    ext = (ext or "").lower().lstrip(".")
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "bmp": "image/bmp",
        "webp": "image/webp",
        "tif": "image/tiff",
        "tiff": "image/tiff",
    }.get(ext, f"image/{ext}" if ext else "application/octet-stream")


def write_shared_asset(
    asset_dir: Path,
    data: bytes,
    filename: str,
) -> Dict[str, Any]:
    """Write asset once; if exists with same content, reuse. Returns meta."""
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    path = asset_dir / filename
    sha = _sha256_hex(data)
    if path.exists():
        existing = path.read_bytes()
        if _sha256_hex(existing) == sha:
            data = existing  # already correct
        else:
            # Content-addressed name should match; rewrite if collision
            path.write_bytes(data)
    else:
        path.write_bytes(data)
    size = path.stat().st_size if path.exists() else 0
    meta = {
        "path": path,
        "exists": path.exists() and size > 0,
        "bytes": size,
        "contentSha256": sha,
    }
    try:
        if Image is not None and path.exists():
            with Image.open(path) as im:
                meta["pixelWidth"], meta["pixelHeight"] = im.size
    except Exception:
        meta.setdefault("pixelWidth", 0)
        meta.setdefault("pixelHeight", 0)
    return meta


def crop_page_region(
    page_image_bytes: bytes,
    bbox_xywh: Sequence[float],
    scale: float,
    page_width: float,
    page_height: float,
    dpi: int,
) -> Optional[bytes]:
    """Crop region from rendered page PNG; bbox in PDF points, scale=72/dpi."""
    if Image is None:
        return None
    try:
        img = Image.open(io.BytesIO(page_image_bytes))
        pw, ph = img.size
        x0 = float(bbox_xywh[0]) / scale
        y0 = float(bbox_xywh[1]) / scale
        x1 = float(bbox_xywh[0] + bbox_xywh[2]) / scale
        y1 = float(bbox_xywh[1] + bbox_xywh[3]) / scale
        # Clamp in pixel space to rendered image
        ix0 = max(0, int(x0))
        iy0 = max(0, int(y0))
        ix1 = min(pw, int(x1))
        iy1 = min(ph, int(y1))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        cropped = img.crop((ix0, iy0, ix1, iy1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _visual_figures(visual_blocks: Sequence[DocBlock]) -> List[DocBlock]:
    return [b for b in visual_blocks if b.type in ("figure", "image")]


def build_figure_blocks(
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    embedded: Sequence[Dict[str, Any]],
    visual_figures: Sequence[DocBlock],
    doc: Any,
    asset_dir: Path,
    page_image_bytes: Optional[bytes] = None,
    scale: float = 1.0,
    dpi: int = 144,
) -> Tuple[List[DocBlock], List[str], Dict[str, Any]]:
    """Build figure DocBlocks for one page.

    Returns (blocks, warnings, stats).
    Never emits both embedded and crop for the same placement.
    """
    warnings: List[str] = []
    stats = {
        "embedded_candidates": len(embedded),
        "visual_candidates": len(visual_figures),
        "matched": 0,
        "embedded_kept": 0,
        "cropped": 0,
        "excluded": 0,
        "unique_assets_written": 0,
        "duplicate_asset_hits": 0,
    }
    blocks: List[DocBlock] = []
    matched_visual_idx: set = set()

    # Pass 1: embedded images (priority over crop)
    embedded_used_digests: Dict[str, str] = {}  # digest -> imageUri

    for emb in embedded:
        bbox = emb["bbox"]
        # Match visual for decoration/background exceptions + layout evidence
        v_match: Optional[DocBlock] = None
        for i, vf in enumerate(visual_figures):
            if i in matched_visual_idx:
                continue
            vb = (vf.bbox.x, vf.bbox.y, vf.bbox.x + vf.bbox.w, vf.bbox.y + vf.bbox.h)
            r = bbox_overlap_ratio(bbox, vb)
            if r >= VISUAL_MATCH_OVERLAP:
                v_match = vf
                matched_visual_idx.add(visual_figures.index(vf) if False else i)
                break

        visual_matched = v_match is not None
        if visual_matched:
            stats["matched"] += 1

        reason = should_exclude_embedded(
            bbox,
            page_width,
            page_height,
            pixel_width=emb.get("pixel_width") or 0,
            pixel_height=emb.get("pixel_height") or 0,
            visual_matched=visual_matched,
        )
        if reason:
            # Spec: uncertain boundaries → keep candidate with warning instead of silent drop
            # for borderline; only hard-drop clear background/tiny/invalid.
            if reason in ("full_page_background", "tiny_decoration", "invalid_pixel_size", "empty_bbox", "invalid_page_size"):
                stats["excluded"] += 1
                warnings.append(
                    f"page{page_number}: excluded embedded image xref={emb.get('xref')} reason={reason}"
                )
                if reason == "full_page_background":
                    # Keep out of IR per spec (整页背景不计 figure)
                    continue
                if reason in ("tiny_decoration", "invalid_pixel_size", "empty_bbox", "invalid_page_size"):
                    continue
            elif reason == "edge_icon":
                stats["excluded"] += 1
                warnings.append(
                    f"page{page_number}: excluded edge icon xref={emb.get('xref')}"
                )
                continue

        # Extract embedded bytes (preferred over crop)
        data = None
        ext = "png"
        extracted = extract_embedded_bytes(doc, emb.get("xref"))
        if extracted:
            data, ext = extracted
        content_sha = emb.get("digest") or (_sha256_hex(data) if data else "")
        if data and not content_sha:
            content_sha = _sha256_hex(data)
        if not content_sha:
            content_sha = hashlib.sha1(
                f"{page_number}|{bbox}".encode("utf-8")
            ).hexdigest()

        if data is None:
            # Fall back to page crop for this placement
            if page_image_bytes is None:
                warnings.append(
                    f"page{page_number}: embedded bytes missing xref={emb.get('xref')} and no page render"
                )
                blocks.append(
                    _figure_block(
                        page_number=page_number,
                        bbox=bbox,
                        content_digest=content_sha,
                        asset_source="embedded",
                        asset_status="missing",
                        warnings=warnings,
                        visual_matched=visual_matched,
                        xref=emb.get("xref"),
                    )
                )
                continue
            data = crop_from_render(
                page_image_bytes,
                (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
                scale,
            )
            ext = "png"
            if data is None:
                warnings.append(
                    f"page{page_number}: crop fallback failed xref={emb.get('xref')}"
                )
                blocks.append(
                    _figure_block(
                        page_number=page_number,
                        bbox=bbox,
                        content_digest=content_sha,
                        asset_source="embedded",
                        asset_status="missing",
                        warnings=warnings,
                        visual_matched=visual_matched,
                        xref=emb.get("xref"),
                    )
                )
                continue
            content_sha = _sha256_hex(data)

        # Dedupe by content: one file, many blocks
        if content_sha in embedded_used_digests:
            uri = embedded_used_digests[content_sha]
            stats["duplicate_asset_hits"] += 1
            asset_status = "ready"
            fname = Path(uri).name
        else:
            fname = asset_filename_for_digest(content_sha, ext)
            pre_existed = (Path(asset_dir) / fname).exists()
            wmeta = write_shared_asset(asset_dir, data, fname)
            uri = f"shots/{fname}"
            if not wmeta["exists"]:
                warnings.append(f"page{page_number}: asset write failed {fname}")
                asset_status = "missing"
            else:
                asset_status = "ready"
                embedded_used_digests[content_sha] = uri
                if pre_existed:
                    # Same content already on disk from another location/page
                    stats["duplicate_asset_hits"] += 1
                else:
                    stats["unique_assets_written"] += 1
            content_sha = wmeta.get("contentSha256") or content_sha

        if asset_status == "ready":
            stats["embedded_kept"] += 1

        # Block id: page + bbox + digest (stable); visual bbox kept in metadata if present
        block_bbox = bbox
        if v_match is not None:
            # Prefer larger of embedded/visual for placement? Keep embedded as primary;
            # record visual bbox as evidence.
            pass

        bid = stable_figure_block_id(page_number, bbox, content_sha)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        block = DocBlock(
            id=bid,
            type="figure",
            text="",
            bbox=BBox(bbox[0], bbox[1], w, h),
            page_number=page_number,
            reading_order=0,
            source="native",
            metadata={
                "assetSource": "embedded",
                "assetStatus": asset_status,
                "imageUri": uri if asset_status == "ready" else "",
                "contentSha256": content_sha,
                "mimeType": _mime_for_ext(ext if asset_status == "ready" else "png"),
                "pixelWidth": 0,
                "pixelHeight": 0,
                "detectionSource": "pymupdf_embedded",
                "visualMatched": visual_matched,
                "xref": emb.get("xref"),
                "warnings": [w for w in warnings if w.startswith(f"page{page_number}:")],
            },
        )
        # Fill pixel size from write meta when ready
        if asset_status == "ready":
            apath = asset_dir / fname
            try:
                if Image is not None and apath.exists():
                    with Image.open(apath) as im:
                        block.metadata["pixelWidth"] = int(im.size[0])
                        block.metadata["pixelHeight"] = int(im.size[1])
            except Exception:
                pass
            if block.metadata["pixelWidth"] < 1 or block.metadata["pixelHeight"] < 1:
                block.metadata["pixelWidth"] = emb.get("pixel_width") or 0
                block.metadata["pixelHeight"] = emb.get("pixel_height") or 0
        else:
            # missing: do not claim ready fields complete
            block.metadata["imageUri"] = ""
            block.metadata["contentSha256"] = content_sha

        if v_match is not None:
            block.metadata["visualBbox"] = [
                v_match.bbox.x,
                v_match.bbox.y,
                v_match.bbox.x + v_match.bbox.w,
                v_match.bbox.y + v_match.bbox.h,
            ]
        blocks.append(block)

    # Pass 2: unmatched visual figures → page crop
    for i, vf in enumerate(visual_figures):
        if i in matched_visual_idx:
            continue
        bbox_xyxy = (vf.bbox.x, vf.bbox.y, vf.bbox.x + vf.bbox.w, vf.bbox.y + vf.bbox.h)
        # Exclude backgrounds via same policy (visual_matched=False here → may exclude)
        reason = should_exclude_embedded(
            bbox_xyxy,
            page_width,
            page_height,
            pixel_width=1000,
            pixel_height=1000,
            visual_matched=True,  # it IS a visual figure candidate
        )
        # visual figures are explicitly detected — only drop full-page bg
        if reason == "full_page_background":
            stats["excluded"] += 1
            warnings.append(f"page{page_number}: excluded visual full-page figure")
            continue

        clamped = clamp_bbox_to_page(bbox_xyxy, page_width, page_height)
        content_preview = ""
        if page_image_bytes is not None:
            data = crop_from_render(
                page_image_bytes,
                (clamped[0], clamped[1], clamped[2] - clamped[0], clamped[3] - clamped[1]),
                scale,
            )
            if data is None:
                warnings.append(f"page{page_number}: visual crop failed id={vf.id}")
                blocks.append(
                    _figure_block(
                        page_number=page_number,
                        bbox=list(clamped),
                        content_digest=hashlib.sha1(vf.id.encode()).hexdigest(),
                        asset_source="page_crop",
                        asset_status="missing",
                        warnings=warnings,
                        visual_matched=True,
                        xref=None,
                        detection="visual_crop",
                    )
                )
                continue
            content_sha = _sha256_hex(data)
            if content_sha in embedded_used_digests:
                uri = embedded_used_digests[content_sha]
                fname = Path(uri).name
                asset_status = "ready"
                stats["duplicate_asset_hits"] += 1
            else:
                fname = asset_filename_for_digest(content_sha, "png")
                wmeta = write_shared_asset(asset_dir, data, fname)
                uri = f"shots/{fname}"
                asset_status = "ready" if wmeta["exists"] else "missing"
                if asset_status == "ready":
                    embedded_used_digests[content_sha] = uri
                    stats["unique_assets_written"] += 1
                    stats["cropped"] += 1
                else:
                    warnings.append(f"page{page_number}: crop asset write failed {fname}")
        else:
            content_sha = hashlib.sha1(f"crop|{page_number}|{vf.id}".encode()).hexdigest()
            uri = ""
            asset_status = "missing"
            warnings.append(f"page{page_number}: no page render for visual crop")

        bid = stable_figure_block_id(page_number, clamped, content_sha)
        block = DocBlock(
            id=bid,
            type="figure",
            text="",
            bbox=BBox(clamped[0], clamped[1], clamped[2] - clamped[0], clamped[3] - clamped[1]),
            page_number=page_number,
            reading_order=0,
            source="visual",
            metadata={
                "assetSource": "page_crop",
                "assetStatus": asset_status,
                "imageUri": uri if asset_status == "ready" else "",
                "contentSha256": content_sha,
                "mimeType": "image/png",
                "pixelWidth": 0,
                "pixelHeight": 0,
                "detectionSource": "visual_crop",
                "visualMatched": True,
                "xref": None,
                "warnings": [w for w in warnings if w.startswith(f"page{page_number}:")],
            },
        )
        if asset_status == "ready":
            apath = asset_dir / Path(uri).name
            try:
                if Image is not None and apath.exists():
                    with Image.open(apath) as im:
                        block.metadata["pixelWidth"] = int(im.size[0])
                        block.metadata["pixelHeight"] = int(im.size[1])
            except Exception:
                pass
        blocks.append(block)

    return blocks, warnings, stats


def crop_from_render(
    page_image_bytes: bytes,
    bbox_xywh: Sequence[float],
    scale: float,
) -> Optional[bytes]:
    """Crop using scale = 72/dpi (points → pixels)."""
    if Image is None or not page_image_bytes:
        return None
    try:
        img = Image.open(io.BytesIO(page_image_bytes))
        pw, ph = img.size
        x0 = float(bbox_xywh[0]) / scale
        y0 = float(bbox_xywh[1]) / scale
        x1 = float(bbox_xywh[0] + bbox_xywh[2]) / scale
        y1 = float(bbox_xywh[1] + bbox_xywh[3]) / scale
        ix0, iy0 = max(0, int(x0)), max(0, int(y0))
        ix1, iy1 = min(pw, int(x1)), min(ph, int(y1))
        if ix1 <= ix0 or iy1 <= iy0:
            return None
        cropped = img.crop((ix0, iy0, ix1, iy1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _figure_block(
    *,
    page_number: int,
    bbox: Sequence[float],
    content_digest: str,
    asset_source: str,
    asset_status: str,
    warnings: List[str],
    visual_matched: bool,
    xref: Optional[int],
    detection: str = "pymupdf_embedded",
) -> DocBlock:
    bid = stable_figure_block_id(page_number, bbox, content_digest)
    x0, y0, x1, y1 = bbox[:4]
    return DocBlock(
        id=bid,
        type="figure",
        text="",
        bbox=BBox(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
        page_number=page_number,
        reading_order=0,
        source="native" if detection == "pymupdf_embedded" else "visual",
        metadata={
            "assetSource": asset_source,
            "assetStatus": asset_status,
            "imageUri": "",
            "contentSha256": content_digest,
            "mimeType": "",
            "pixelWidth": 0,
            "pixelHeight": 0,
            "detectionSource": detection,
            "visualMatched": visual_matched,
            "xref": xref,
            "warnings": list(warnings[-3:]),
        },
    )


def process_page_figures(
    *,
    page_number: int,
    page_width: float,
    page_height: float,
    doc: Any,
    visual_blocks: Sequence[DocBlock],
    asset_dir: Path,
    page_image_bytes: Optional[bytes] = None,
    scale: float = 1.0,
    dpi: int = 144,
) -> Tuple[List[DocBlock], List[str], Dict[str, Any]]:
    """Public entry: enumerate embedded + match visual → figure blocks."""
    embedded = enumerate_embedded_images(
        doc.load_page(page_number - 1) if doc is not None else None,
        page_number,
    ) if doc is not None else []
    visuals = _visual_figures(visual_blocks)
    return build_figure_blocks(
        page_number=page_number,
        page_width=page_width,
        page_height=page_height,
        embedded=embedded,
        visual_figures=visuals,
        doc=doc,
        asset_dir=asset_dir,
        page_image_bytes=page_image_bytes,
        scale=scale,
        dpi=dpi,
    )


# Keep extract_image_bytes available for adapters that still hold a doc handle.
def extract_image_bytes(doc: Any, xref: int) -> Optional[Tuple[bytes, str]]:
    return extract_embedded_bytes(doc, xref)
