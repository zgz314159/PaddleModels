"""Figure-internal caption OCR post-pass — Phase 2F.

Runs after native figure↔text association (Phase 2E) and before
SemanticProjector. Only processes ready embedded figure assets that still
lack a primary caption association. Creates deterministic OCR caption
DocBlocks and appends `ocr_inside_image` caption associations.

Correctness stage: unexpected pass-level failures propagate (fail-fast);
per-figure OCR dependency blocks are reported by the caller before entry.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pipeline.canonical_ir import BBox, DocBlock, DocPage

OCR_METHOD = "ocr_inside_image"
OCR_SOURCE = "ocr"
FIGURE_CAPTION_OCR_REPORT = "figure_caption_ocr.json"


def write_ocr_figure_labels(
    figure: DocBlock,
    labels: Sequence[str],
    *,
    confidence01: float,
) -> None:
    """Persist trusted OCR labels onto figure IR metadata (Phase 2G).

    Writes/merges metadata.figureLabels. Dedup by normalizedLabel;
    native evidence (if any already present) wins over OCR.
    """
    from pipeline.semantics.figure_reference import (
        _make_label_entry,
        merge_figure_labels,
    )

    md = figure.metadata if isinstance(figure.metadata, dict) else {}
    md = dict(md)
    existing = list(md.get("figureLabels") or [])
    incoming = [
        _make_label_entry(
            str(lab),
            str(lab),
            source="ocr",
            method=OCR_METHOD,
            confidence=confidence01,
        )
        for lab in labels
        if str(lab or "").strip()
    ]
    md["figureLabels"] = merge_figure_labels(existing, incoming)
    figure.metadata = md


def stable_ocr_caption_block_id(
    figure_id: str,
    content_sha256: str,
    engine: str,
    extractor_version: str,
) -> str:
    """Deterministic caption block id from figure + asset content + engine."""
    seed = f"{figure_id}|{content_sha256}|{engine}|{extractor_version}"
    return "capocr_" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def is_safe_run_relative_uri(image_uri: str, run_dir: Path) -> bool:
    """True iff image_uri is a relative path resolving inside run_dir."""
    if not image_uri or not isinstance(image_uri, str):
        return False
    raw = image_uri.strip()
    if not raw:
        return False
    # Reject absolute / drive-letter / UNC / parent traversal outright.
    p = Path(raw)
    if p.is_absolute() or raw.startswith(("/", "\\")) or raw.startswith("~"):
        return False
    if ".." in p.parts:
        return False
    if p.drive or (len(raw) >= 2 and raw[1] == ":"):
        return False
    try:
        base = Path(run_dir).resolve()
        target = (base / raw).resolve()
    except Exception:
        return False
    try:
        target.relative_to(base)
    except ValueError:
        return False
    return True


def figure_has_primary_caption(figure: DocBlock) -> bool:
    assocs = (figure.metadata or {}).get("figureTextAssociations") or []
    return any(
        isinstance(a, dict) and a.get("kind") == "caption" for a in assocs
    )


def is_ocr_eligible_figure(figure: DocBlock, run_dir: Path) -> Tuple[bool, str]:
    """Eligibility: ready + embedded + safe imageUri + no primary caption yet."""
    md = figure.metadata or {}
    if figure.type not in ("figure", "image"):
        return False, "not_figure"
    if md.get("assetStatus") != "ready":
        return False, f"asset_status:{md.get('assetStatus') or 'missing'}"
    if md.get("assetSource") != "embedded":
        return False, f"asset_source:{md.get('assetSource') or 'unknown'}"
    uri = md.get("imageUri") or ""
    if not is_safe_run_relative_uri(uri, run_dir):
        return False, "unsafe_image_uri"
    if figure_has_primary_caption(figure):
        return False, "has_primary_caption"
    return True, ""


def map_norm_bbox_to_figure_pdf(
    text_bbox_norm: Optional[Sequence[float]],
    figure: DocBlock,
    band_fallback_norm: Optional[Sequence[float]] = None,
) -> BBox:
    """Map normalized (0..1 within asset) OCR bbox → figure PDF bbox (points).

    Falls back to the bottom-band region of the figure when word boxes are
    unavailable, so the caption block never collapses to a zero-area rect.
    """
    src = list(text_bbox_norm) if text_bbox_norm else None
    if src is None or len(src) != 4:
        src = list(band_fallback_norm or [0.0, 0.7, 1.0, 1.0])
    nx0, ny0, nx1, ny1 = (max(0.0, min(1.0, float(v))) for v in src)
    if nx1 <= nx0:
        nx0, nx1 = 0.0, 1.0
    if ny1 <= ny0:
        ny0, ny1 = 0.7, 1.0
    fx, fy, fw, fh = figure.bbox.x, figure.bbox.y, figure.bbox.w, figure.bbox.h
    x = fx + nx0 * fw
    y = fy + ny0 * fh
    w = max(1.0, (nx1 - nx0) * fw)
    h = max(1.0, (ny1 - ny0) * fh)
    return BBox(float(x), float(y), float(w), float(h))


def append_ocr_caption_association(figure: DocBlock, entry: Dict[str, Any]) -> None:
    """Append OCR caption association; at most one primary caption per figure."""
    md = figure.metadata if isinstance(figure.metadata, dict) else {}
    md = dict(md)
    assocs = list(md.get("figureTextAssociations") or [])
    if any(
        isinstance(a, dict) and a.get("kind") == "caption" for a in assocs
    ):
        # Race: a caption appeared after eligibility — keep first, never upgrade.
        return
    assocs.append(entry)
    # Stable order: kind rank then blockId (caption before legend).
    kind_rank = {"caption": 0, "legend": 1}
    assocs.sort(
        key=lambda a: (
            kind_rank.get(a.get("kind"), 9) if isinstance(a, dict) else 9,
            a.get("blockId", "") if isinstance(a, dict) else "",
        )
    )
    md["figureTextAssociations"] = assocs
    figure.metadata = md


def _band_fallback_norm(top_ratio: Optional[float]) -> List[float]:
    tr = float(top_ratio) if top_ratio is not None else 0.68
    tr = max(0.0, min(0.95, tr))
    return [0.0, tr, 1.0, 1.0]


def run_figure_caption_ocr_pass(
    doc_ir: Any,
    *,
    run_dir: Path,
    engine: str,
    language: str,
    cache_root: Path,
    label_hits_fn: Callable[[str], List[str]],
    extract_candidate_fn: Callable[..., str],
    ocr_fn: Optional[Callable[[bytes, int, str], Dict[str, Any]]] = None,
    extractor_version: str = "v2",
) -> Dict[str, Any]:
    """Post-pass: OCR bottom-band captions for eligible embedded figures.

    Returns diagnostics dict (written to v2/figure_caption_ocr.json by caller)
    plus aggregate stats. Same contentSha256 is OCR'd once per engine/version;
    repeats reuse the in-memory result (disk cache handles cross-run reuse).
    """
    from imaging.figure_caption_ocr import (  # lazy: keep --help light
        EXTRACTOR_VERSION as _DEFAULT_EXTRACTOR_VERSION,
        cache_key,
        cache_path,
        is_cacheable_result,
        load_cache,
        ocr_embedded_figure_caption,
        save_cache,
    )

    started = time.time()
    run_dir = Path(run_dir)
    items: List[Dict[str, Any]] = []
    report_warnings: List[str] = []
    if extractor_version == "v1":
        # Phase 2F.1: default upgraded — callers should pass EXTRACTOR_VERSION.
        extractor_version = _DEFAULT_EXTRACTOR_VERSION

    stats = {
        "figure_caption_ocr_attempted": 0,
        "figure_caption_ocr_cache_hits": 0,
        "figures_with_ocr_caption": 0,
        "ocr_captions_linked": 0,
        "ocr_captions_rejected": 0,
        "ocr_captions_ambiguous": 0,
        "ocr_caption_skipped": 0,
        "ocr_caption_errors": 0,
    }

    # contentSha256 → structured OCR result (in-run dedupe).
    seen: Dict[str, Dict[str, Any]] = {}
    # contentSha256 → cache-hit flag (first observation wins for metrics).
    cache_hit_seen: Dict[str, bool] = {}

    pages = getattr(doc_ir, "pages", None) or []
    for page in pages:
        figures = [
            b for b in page.blocks if b.type in ("figure", "image")
        ]
        for fig in figures:
            eligible, reason = is_ocr_eligible_figure(fig, run_dir)
            if not eligible:
                # Only track skip reasons for embedded candidates (diagnostics);
                # missing/crop/table figures are out of scope, not failures.
                md = fig.metadata or {}
                if md.get("assetSource") == "embedded" or reason == "has_primary_caption":
                    stats["ocr_caption_skipped"] += 1
                    items.append(
                        {
                            "page": fig.page_number,
                            "figureId": fig.id,
                            "contentSha256": md.get("contentSha256"),
                            "status": "skipped",
                            "skipReason": reason,
                            "cacheHit": False,
                            "engine": engine,
                            "language": language,
                        }
                    )
                continue

            md = fig.metadata or {}
            content_sha = str(md.get("contentSha256") or "").strip().lower()
            image_uri = md.get("imageUri") or ""

            stats["figure_caption_ocr_attempted"] += 1

            item: Dict[str, Any] = {
                "page": fig.page_number,
                "figureId": fig.id,
                "contentSha256": content_sha,
                "imageUri": image_uri,
                "engine": engine,
                "language": language,
                "extractorVersion": extractor_version,
                "cacheKey": cache_key(engine, language),
                "cacheHit": False,
            }

            # --- Result resolution: in-run → disk cache → fresh OCR ---
            result: Optional[Dict[str, Any]] = None
            cache_hit = False

            if content_sha and content_sha in seen:
                result = dict(seen[content_sha])
                cache_hit = True
                stats["figure_caption_ocr_cache_hits"] += 1
                item["cacheHit"] = True
                item["cacheSource"] = "in_run"
            else:
                cpath: Optional[Path] = None
                if content_sha:
                    try:
                        cpath = cache_path(cache_root, content_sha, engine)
                        cached = load_cache(cpath, engine, language)
                    except ValueError as ve:
                        cached = None
                        item["cacheError"] = str(ve)
                        cpath = None
                    if cached is not None:
                        result = cached
                        cache_hit = True
                        stats["figure_caption_ocr_cache_hits"] += 1
                        item["cacheHit"] = True
                        item["cacheSource"] = "disk"

            if result is None:
                # Fresh OCR — load asset bytes only from safe run-relative path.
                asset_path = run_dir / image_uri
                try:
                    image_bytes = asset_path.read_bytes()
                except Exception as exc:
                    stats["ocr_caption_errors"] += 1
                    result = {
                        "status": "rejected",
                        "labels": [],
                        "candidate": "",
                        "rejection_reason": f"asset_read_error:{type(exc).__name__}",
                        "mean_confidence": 0.0,
                        "roi": None,
                        "psm": None,
                        "language": language,
                        "text_bbox_norm": None,
                        "elapsed_ms": 0,
                        "ocr_error": str(exc),
                        "attempts": [],
                    }
                    image_bytes = b""

                if image_bytes:
                    try:
                        result = ocr_embedded_figure_caption(
                            image_bytes,
                            label_hits_fn=label_hits_fn,
                            extract_candidate_fn=extract_candidate_fn,
                            ocr_fn=ocr_fn,
                            language=language,
                        )
                    except Exception as exc:
                        # Internal OCR exception must never masquerade as linked.
                        stats["ocr_caption_errors"] += 1
                        result = {
                            "status": "rejected",
                            "labels": [],
                            "candidate": "",
                            "rejection_reason": f"ocr_exception:{type(exc).__name__}",
                            "mean_confidence": 0.0,
                            "roi": None,
                            "psm": None,
                            "language": language,
                            "text_bbox_norm": None,
                            "elapsed_ms": 0,
                            "ocr_error": str(exc),
                            "attempts": [],
                        }

                if content_sha and result is not None and cpath is not None:
                    if is_cacheable_result(result):
                        payload = dict(result)
                        payload["cacheKey"] = cache_key(engine, language)
                        payload["engine"] = engine
                        payload["language"] = language
                        payload["contentSha256"] = content_sha
                        payload["extractorVersion"] = extractor_version
                        if not save_cache(cpath, payload):
                            warn = f"cache_write_failed:{content_sha[:12]}"
                            report_warnings.append(warn)
                            item.setdefault("warnings", []).append(warn)
                    # Non-cacheable (transient/OCR error): do NOT write cache.
                if content_sha and result is not None and is_cacheable_result(result):
                    seen[content_sha] = dict(result)
                elif content_sha and result is not None and not is_cacheable_result(result):
                    # Do not reuse transient failures within the same run either
                    # (except we still record for diagnostics only).
                    pass

            if content_sha:
                cache_hit_seen.setdefault(content_sha, cache_hit)

            assert result is not None
            status = result.get("status") or "rejected"
            item["status"] = status
            item["labels"] = result.get("labels") or []
            item["candidate"] = result.get("candidate") or ""
            item["rejectionReason"] = result.get("rejection_reason")
            item["meanConfidence"] = result.get("mean_confidence")
            item["roi"] = result.get("roi")
            item["psm"] = result.get("psm")
            item["attempts"] = result.get("attempts") or []
            item["ocrError"] = result.get("ocr_error")
            item["elapsedMs"] = result.get("elapsed_ms")
            item["captionLineKeys"] = result.get("caption_line_keys") or []
            item["captionWordCount"] = int(result.get("caption_word_count") or 0)
            item["textBBoxNorm"] = result.get("text_bbox_norm")
            if "cacheSource" not in item:
                item["cacheSource"] = None

            if status == "linked":
                caption_text = (result.get("candidate") or "").strip()
                labels = result.get("labels") or []
                if not caption_text or len(labels) != 1:
                    # Defense: classifier said linked but gates inconsistent.
                    stats["ocr_captions_rejected"] += 1
                    item["status"] = "rejected"
                    item["rejectionReason"] = "linked_but_inconsistent"
                else:
                    block_id = stable_ocr_caption_block_id(
                        fig.id, content_sha, engine, extractor_version
                    )
                    top_ratio = (result.get("roi") or {}).get("top_ratio")
                    bbox = map_norm_bbox_to_figure_pdf(
                        result.get("text_bbox_norm"),
                        fig,
                        band_fallback_norm=_band_fallback_norm(top_ratio),
                    )
                    cap_block = DocBlock(
                        id=block_id,
                        type="caption",
                        text=caption_text,
                        bbox=bbox,
                        page_number=fig.page_number,
                        reading_order=0,
                        confidence=float(result.get("mean_confidence") or 0.0) / 100.0,
                        source=OCR_SOURCE,
                        metadata={
                            "semanticRole": "caption",
                            "figureId": fig.id,
                            "ocrEngine": engine,
                            "insideFigure": True,
                            "ocrLanguage": language,
                            "contentSha256": content_sha,
                        },
                    )
                    # Insert caption right after its figure for reading order.
                    try:
                        idx = page.blocks.index(fig)
                        page.blocks.insert(idx + 1, cap_block)
                    except ValueError:
                        page.blocks.append(cap_block)

                    append_ocr_caption_association(
                        fig,
                        {
                            "kind": "caption",
                            "blockId": block_id,
                            "text": caption_text,
                            "source": OCR_SOURCE,
                            "method": OCR_METHOD,
                            "ocrEngine": engine,
                        },
                    )
                    stats["ocr_captions_linked"] += 1
                    stats["figures_with_ocr_caption"] += 1
                    item["captionBlockId"] = block_id
                    item["captionText"] = caption_text
                    # Phase 2G: trusted label from linked OCR caption.
                    conf01 = max(
                        0.0,
                        min(1.0, float(result.get("mean_confidence") or 0.0) / 100.0),
                    )
                    write_ocr_figure_labels(fig, labels, confidence01=conf01)
            elif status == "ambiguous_multiple_labels":
                stats["ocr_captions_ambiguous"] += 1
                # Ambiguous cannot pick one caption, but labels remain trusted.
                amb_labels = result.get("labels") or []
                if amb_labels:
                    conf01 = max(
                        0.0,
                        min(
                            1.0, float(result.get("mean_confidence") or 0.0) / 100.0
                        ),
                    )
                    write_ocr_figure_labels(
                        fig, amb_labels, confidence01=conf01
                    )
            elif status == "label_only":
                stats["ocr_captions_rejected"] += 1
                # Reliable label_only: no caption text, but the label is trusted.
                lo_labels = result.get("labels") or []
                if lo_labels:
                    conf01 = max(
                        0.0,
                        min(
                            1.0, float(result.get("mean_confidence") or 0.0) / 100.0
                        ),
                    )
                    write_ocr_figure_labels(fig, lo_labels, confidence01=conf01)
            elif status in ("rejected", "empty"):
                stats["ocr_captions_rejected"] += 1
            else:
                stats["ocr_captions_rejected"] += 1

            items.append(item)

    elapsed_ms = int((time.time() - started) * 1000)
    stats["ocr_caption_elapsed_ms"] = elapsed_ms

    report = {
        "engine": engine,
        "language": language,
        "extractorVersion": extractor_version,
        "method": OCR_METHOD,
        "runDir": str(run_dir),
        "stats": dict(stats),
        "elapsedMs": elapsed_ms,
        "items": items,
        "warnings": report_warnings,
    }
    return report
