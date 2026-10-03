"""PaddleModels unified CLI — Phase 1.1.

Flat flags: --mode legacy|v2|shadow, --input, --out, --profile, --max-pages.
Heavy PDF deps are loaded only when a mode actually runs.
--help never imports PDF/OCR libraries.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

# Source-checkout root (…/<repo>/src/paddle_models/cli/main.py → parents[3]).
# Installed wheels have no such root: runtime code and contract schemas are
# resolved from the installed packages instead (see _repo_root/_contracts_dir).
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def _repo_root() -> Optional[Path]:
    """Return the source-checkout root when running from a git checkout, else None."""
    if (_SOURCE_ROOT / "pipeline").is_dir() and (_SOURCE_ROOT / "contracts").is_dir():
        return _SOURCE_ROOT
    return None


def _contracts_dir() -> Optional[Path]:
    """Directory holding the packaged contract schemas (installed package or checkout)."""
    try:
        import importlib.resources as resources

        candidate = Path(str(resources.files("contracts")))
        if candidate.is_dir():
            return candidate
    except Exception:
        pass
    root = _repo_root()
    if root is not None:
        candidate = root / "contracts"
        if candidate.is_dir():
            return candidate
    return None


def _resolve_contract(rel: Path) -> Optional[Path]:
    """Resolve a contract file by name against the packaged contracts directory."""
    directory = _contracts_dir()
    if directory is None:
        return None
    candidate = directory / Path(rel).name
    return candidate if candidate.is_file() else None


def _tool_script(name: str) -> Optional[Path]:
    """Locate a bundled tools/ script (installed package or source checkout)."""
    try:
        import importlib.resources as resources

        candidate = Path(str(resources.files("tools").joinpath(name)))
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    root = _repo_root()
    if root is not None:
        candidate = root / "tools" / name
        if candidate.is_file():
            return candidate
    return None

# Hard deps required to actually execute a PDF mode end-to-end.
HARD_DEPS: Dict[str, List[str]] = {
    "legacy": ["fitz", "PIL"],
    "v2": ["fitz", "PIL"],
    "shadow": ["fitz", "PIL"],
}

PKG_NAMES = {
    "fitz": "PyMuPDF",
    "PIL": "Pillow",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "jsonschema": "jsonschema",
}

KB_SCHEMA_REL = Path("contracts") / "knowledge_base_schema_v2.json"
IR_SCHEMA_REL = Path("contracts") / "knowledge-base.v2.schema.json"

# Metrics comparable only when both sides expose the same key with a real value.
SHADOW_COMPARABLE_KEYS = (
    "pages",
    "entries",
    "entries_nonempty",
    "entries_empty",
    "tables",
    "tables_structured",
    "tables_image_only",
    "table_cells",
    "table_assets_missing",
    "images",
    "images_embedded",
    "images_cropped",
    "images_missing",
    "image_assets_unique",
    "image_assets_duplicate",
    "figures_with_native_caption",
    "figures_with_native_legend",
    "native_captions_linked",
    "native_legends_linked",
    "unmatched_caption_candidates",
    "text_chars",
    "normalized_text_chars",
    # Phase 2F figure-caption OCR — legacy omits → shadow lists as skipped.
    "figure_caption_ocr_attempted",
    "figure_caption_ocr_cache_hits",
    "figures_with_ocr_caption",
    "ocr_captions_linked",
    "ocr_captions_rejected",
    "ocr_captions_ambiguous",
    "ocr_caption_elapsed_ms",
    # Phase 2G figure label index / references — legacy omits → skipped.
    "figure_labels_indexed",
    "figure_references_found",
    "figure_references_resolved",
    "figure_references_unresolved",
    "figure_references_ambiguous",
    "referenced_figures",
)

# v2-only figure-caption OCR metric keys (legacy never emits them).
FIGURE_CAPTION_OCR_METRIC_KEYS = (
    "figure_caption_ocr_attempted",
    "figure_caption_ocr_cache_hits",
    "figures_with_ocr_caption",
    "ocr_captions_linked",
    "ocr_captions_rejected",
    "ocr_captions_ambiguous",
    "ocr_caption_elapsed_ms",
)

# v2-only figure reference metric keys (legacy never emits them).
FIGURE_REFERENCE_METRIC_KEYS = (
    "figure_labels_indexed",
    "figure_references_found",
    "figure_references_resolved",
    "figure_references_unresolved",
    "figure_references_ambiguous",
    "referenced_figures",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(text: str) -> str:
    """Stable whitespace-only normalization; no content rewriting."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


def make_run_id(stem: str) -> str:
    """Unique run id: second-resolution stem + microsecond + short random suffix."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stem}_{ts}_{uuid.uuid4().hex[:6]}"


def probe_capabilities(mode: str = "shadow") -> Dict[str, Any]:
    """Discovery-only capability probe (find_spec; never imports heavy deps).

    `discovered=true` means the module can be found on sys.path — NOT that it
    imports cleanly or that OCR/runtime works. Real import failures surface
    later as side status `blocked_by_dependency`.
    """
    required = set(HARD_DEPS.get(mode, []))
    modules: Dict[str, Any] = {}
    missing_packages: List[str] = []
    for mod in ("fitz", "PIL", "numpy", "cv2", "jsonschema"):
        try:
            discovered = importlib.util.find_spec(mod) is not None
        except Exception:
            discovered = False
        modules[mod] = {
            "discovered": discovered,
            "required_for_mode": mod in required,
            "package": PKG_NAMES[mod],
        }
        if not discovered and (mod in required or mod == "jsonschema"):
            missing_packages.append(PKG_NAMES[mod])
    return {
        "probe_semantics": (
            "discovered=find_spec only; not an import/runtime guarantee. "
            "Actual import failures are recorded as blocked_by_dependency on each side."
        ),
        "mode": mode,
        "modules": modules,
        "missing_packages": missing_packages,
        "python": sys.version.split()[0],
        "platform": os.name,
    }


def missing_hard_deps(mode: str, caps: Dict[str, Any]) -> List[str]:
    modules = caps.get("modules", {})
    return [
        mod
        for mod in HARD_DEPS.get(mode, [])
        if not modules.get(mod, {}).get("discovered", False)
    ]


def _ensure_repo_on_path() -> None:
    """Put a source checkout on sys.path; a no-op for installed wheels."""
    root = _repo_root()
    if root is None:
        return
    for p in (str(root), str(root / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _collect_page_numbers(kb: Dict[str, Any]) -> Set[int]:
    """Union of page numbers from pageSizes keys, entry.pageNumber, block.pageNumber."""
    pages: Set[int] = set()
    page_sizes = (kb.get("fileMetadata") or {}).get("pageSizes") or {}
    for key in page_sizes.keys():
        try:
            pages.add(int(key))
        except (TypeError, ValueError):
            continue
    for entry in kb.get("entries") or []:
        pn = entry.get("pageNumber")
        if pn is not None:
            try:
                pages.add(int(pn))
            except (TypeError, ValueError):
                pass
        for block in entry.get("blocks") or []:
            bpn = block.get("pageNumber")
            if bpn is not None:
                try:
                    pages.add(int(bpn))
                except (TypeError, ValueError):
                    pass
    return pages


def _entry_normalized_text(entry: Dict[str, Any]) -> str:
    return str(entry.get("contentNormalized") or entry.get("contentMarkdown") or "")


def _is_nonempty_entry(entry: Dict[str, Any]) -> bool:
    """Nonempty = non-whitespace searchable content (contentNormalized/Markdown).

    Blocks alone do not make an entry nonempty — legacy page entries can keep
    visual/text blocks while content fields are stripped to ''.
    """
    return bool(normalize_text(_entry_normalized_text(entry)))


def _canonical_table_grid(block: Dict[str, Any]) -> List[List[Any]]:
    """Canonical 2D grid for a table block.

    Prefer a list-valued `rows` (the v2 contract, including an empty list) over the
    legacy `table_rows` alias; a list-valued `rows` is a grid, never a row count.
    """
    rows = block.get("rows")
    if isinstance(rows, list):
        return rows
    legacy = block.get("table_rows")
    if isinstance(legacy, list):
        return legacy
    return []


def _classify_table_status(block: Dict[str, Any]) -> str:
    """Legacy-compatible structureStatus inference (no fabricated cells)."""
    status = block.get("structureStatus")
    if status in ("structured", "image_only"):
        return status
    if _canonical_table_grid(block):
        return "structured"
    # No rows: image_only only if a visual asset is present.
    if block.get("imageUri") or block.get("src"):
        return "image_only"
    # No rows, no image — treat as image_only (unstructured visual candidate).
    return "image_only"


def kb_metrics_from_obj(
    kb: Dict[str, Any],
    *,
    include_figure_association: bool = True,
) -> Dict[str, Any]:
    """Extract real observed metrics from a KB object (no max_pages fabrication).

    include_figure_association=False → omit the five v2-only association keys
    (legacy does not support them; must not report fabricated 0).
    """
    entries = kb.get("entries") or []
    num_tables = 0
    num_tables_structured = 0
    num_tables_image_only = 0
    table_cells = 0
    table_assets_missing = 0
    num_images = 0
    images_embedded = 0
    images_cropped = 0
    images_missing = 0
    image_uris: List[str] = []
    figures_with_native_caption = 0
    figures_with_native_legend = 0
    native_captions_linked = 0
    native_legends_linked = 0
    unmatched_caption_candidates = 0
    figures_with_ocr_caption = 0
    ocr_captions_linked = 0
    figure_labels_indexed = 0
    figure_references_found = 0
    figure_references_resolved = 0
    figure_references_unresolved = 0
    figure_references_ambiguous = 0
    referenced_figures = 0
    text_chars = 0
    normalized_parts: List[str] = []
    entries_nonempty = 0
    entries_empty = 0

    for entry in entries:
        content = _entry_normalized_text(entry)
        if _is_nonempty_entry(entry):
            entries_nonempty += 1
        else:
            entries_empty += 1
        # Text metrics come from content fields only (searchable text).
        text_chars += len(content)
        norm = normalize_text(content)
        if norm:
            normalized_parts.append(norm)
        for block in entry.get("blocks") or []:
            btype = block.get("type")
            if btype == "table":
                num_tables += 1
                st = _classify_table_status(block)
                if st == "structured":
                    num_tables_structured += 1
                    grid = _canonical_table_grid(block)
                    if grid:
                        # Count non-empty cells from the canonical dense grid.
                        for row in grid:
                            if isinstance(row, list):
                                table_cells += sum(1 for cell in row if str(cell or "").strip())
                    else:
                        # No grid: fall back to the explicit physical cells list.
                        cells = block.get("cells")
                        if isinstance(cells, list) and cells:
                            table_cells += len(cells)
                else:
                    num_tables_image_only += 1
                uri = block.get("imageUri") or block.get("src") or ""
                if st == "structured":
                    # crop optional for structured; missing only counted when expected visual
                    # For structured without imageUri we still expect crop in Phase 2A.
                    if not uri:
                        table_assets_missing += 1
                else:
                    if not uri:
                        table_assets_missing += 1
            elif btype in ("image", "figure"):
                num_images += 1
                asset_status = block.get("assetStatus") or ""
                asset_source = block.get("assetSource") or ""
                uri = block.get("imageUri") or block.get("src") or ""
                if asset_status == "missing" or not uri:
                    images_missing += 1
                else:
                    image_uris.append(uri)
                    if asset_source == "embedded":
                        images_embedded += 1
                    elif asset_source == "page_crop":
                        images_cropped += 1
                    else:
                        # Legacy conservative inference — do not fabricate embedded/cropped.
                        pass
                # Phase 2E associations (only if present — legacy has none)
                assocs = block.get("figureTextAssociations")
                if isinstance(assocs, list) and assocs:
                    # Native = non-OCR caption/legend links (Phase 2F adds source=ocr).
                    native_caps = [
                        a for a in assocs
                        if isinstance(a, dict)
                        and a.get("kind") == "caption"
                        and a.get("source") != "ocr"
                    ]
                    native_legs = [
                        a for a in assocs
                        if isinstance(a, dict)
                        and a.get("kind") == "legend"
                    ]
                    if native_caps:
                        figures_with_native_caption += 1
                        native_captions_linked += len(native_caps)
                    if native_legs:
                        figures_with_native_legend += 1
                        native_legends_linked += len(native_legs)
                    # Phase 2F: OCR-sourced caption associations (source=ocr).
                    ocr_caps = [
                        a
                        for a in assocs
                        if isinstance(a, dict)
                        and a.get("kind") == "caption"
                        and a.get("source") == "ocr"
                    ]
                    if ocr_caps:
                        figures_with_ocr_caption += 1
                        ocr_captions_linked += len(ocr_caps)
                # Phase 2G: trusted figure labels on image/figure blocks.
                fig_labels = block.get("figureLabels")
                if isinstance(fig_labels, list):
                    figure_labels_indexed += sum(
                        1 for lab in fig_labels if isinstance(lab, dict)
                    )
                ref_by = block.get("referencedBy")
                if isinstance(ref_by, list) and any(
                    isinstance(b, dict) for b in ref_by
                ):
                    referenced_figures += 1
            # Phase 2G: body figure references (KB projects body as type=code).
            fig_refs = block.get("figureReferences")
            if isinstance(fig_refs, list):
                for ref in fig_refs:
                    if not isinstance(ref, dict):
                        continue
                    figure_references_found += 1
                    st = ref.get("status")
                    if st == "resolved":
                        figure_references_resolved += 1
                    elif st == "ambiguous":
                        figure_references_ambiguous += 1
                    else:
                        figure_references_unresolved += 1
            if btype in ("text", "heading", "caption"):
                # unmatched caption-role text blocks (legacy often has no figureTextAssociations)
                role = block.get("semanticRole") or ""
                if role == "caption":
                    # Not linked if no figure association references this id — approximate:
                    # count only when block id not found in any figure assoc (set later)
                    unmatched_caption_candidates += 0  # refined after full pass below

    page_sizes = (kb.get("fileMetadata") or {}).get("pageSizes") or {}
    pages_declared = len(page_sizes)
    page_numbers = sorted(_collect_page_numbers(kb))
    # Stable hash over whitespace-normalized concatenation of entry content.
    hash_source = normalize_text("\n".join(_entry_normalized_text(e) for e in entries))

    unique_uris = sorted(set(u for u in image_uris if u))
    image_assets_unique = len(unique_uris)
    image_assets_duplicate = max(0, len(image_uris) - image_assets_unique)

    # Unmatched caption candidates: caption-role text blocks whose id is not in any figure assoc
    linked_ids = set()
    for entry in entries:
        for block in entry.get("blocks") or []:
            assocs = block.get("figureTextAssociations")
            if isinstance(assocs, list):
                for a in assocs:
                    if isinstance(a, dict) and a.get("blockId"):
                        linked_ids.add(a["blockId"])
    unmatched_caption_candidates = 0
    for entry in entries:
        for block in entry.get("blocks") or []:
            if block.get("type") in ("text", "heading", "caption") and (
                block.get("semanticRole") == "caption"
            ):
                if block.get("id") not in linked_ids:
                    unmatched_caption_candidates += 1

    metrics: Dict[str, Any] = {
        "pages": len(page_numbers),
        "page_numbers": page_numbers,
        "pages_declared": pages_declared,
        "entries": len(entries),
        "entries_nonempty": entries_nonempty,
        "entries_empty": entries_empty,
        "tables": num_tables,
        "tables_structured": num_tables_structured,
        "tables_image_only": num_tables_image_only,
        "table_cells": table_cells,
        "table_assets_missing": table_assets_missing,
        "images": num_images,
        "images_embedded": images_embedded,
        "images_cropped": images_cropped,
        "images_missing": images_missing,
        "image_assets_unique": image_assets_unique,
        "image_assets_duplicate": image_assets_duplicate,
        "text_chars": text_chars,
        "normalized_text_chars": len(hash_source),
        "normalized_text_sha256": hashlib.sha256(hash_source.encode("utf-8")).hexdigest(),
    }
    # v2-only figure-association metrics: omit entirely when unsupported (legacy).
    # Never emit fabricated 0 for legacy — shadow treats missing keys as skipped.
    if include_figure_association:
        metrics["figures_with_native_caption"] = figures_with_native_caption
        metrics["figures_with_native_legend"] = figures_with_native_legend
        metrics["native_captions_linked"] = native_captions_linked
        metrics["native_legends_linked"] = native_legends_linked
        metrics["unmatched_caption_candidates"] = unmatched_caption_candidates
        # Phase 2F: OCR-derived caption links observable in KB associations.
        # attempted/cache/rejected/ambiguous/elapsed come from the run report
        # (merged by run_v2 after the pipeline) — default 0 only for v2.
        metrics["figures_with_ocr_caption"] = figures_with_ocr_caption
        metrics["ocr_captions_linked"] = ocr_captions_linked
        metrics["figure_caption_ocr_attempted"] = 0
        metrics["figure_caption_ocr_cache_hits"] = 0
        metrics["ocr_captions_rejected"] = 0
        metrics["ocr_captions_ambiguous"] = 0
        metrics["ocr_caption_elapsed_ms"] = 0
        # Phase 2G: real KB-observed label/reference counts (v2 only).
        metrics["figure_labels_indexed"] = figure_labels_indexed
        metrics["figure_references_found"] = figure_references_found
        metrics["figure_references_resolved"] = figure_references_resolved
        metrics["figure_references_unresolved"] = figure_references_unresolved
        metrics["figure_references_ambiguous"] = figure_references_ambiguous
        metrics["referenced_figures"] = referenced_figures
    return metrics


def kb_metrics(
    kb_path: Optional[Path],
    *,
    include_figure_association: bool = True,
) -> Optional[Dict[str, Any]]:
    """Read real metrics from a KB JSON file. None if unreadable/missing."""
    if not kb_path:
        return None
    p = Path(kb_path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            kb = json.load(f)
    except Exception:
        return None
    return kb_metrics_from_obj(
        kb, include_figure_association=include_figure_association
    )


def _brief_validation_errors(exc: Exception) -> List[str]:
    """Keep field path + short reason only."""
    briefs: List[str] = []
    # jsonschema.ValidationError: use absolute_path / message directly.
    abs_path = getattr(exc, "absolute_path", None)
    if abs_path is not None:
        parts: List[str] = []
        for p in abs_path:
            parts.append(str(p))
        path = ".".join(parts) if parts else "$"
        msg = str(getattr(exc, "message", exc)).splitlines()[0][:200]
        briefs.append(f"{path}: {msg}")
        # Nested best-match errors (if any) — keep a few with paths.
        nested = getattr(exc, "context", None) or []
        for item in list(nested)[:10]:
            npath = getattr(item, "absolute_path", None)
            if npath is None:
                continue
            nparts = [str(x) for x in npath]
            npath_s = ".".join(nparts) if nparts else "$"
            nmsg = str(getattr(item, "message", "")).splitlines()[0][:200]
            briefs.append(f"{npath_s}: {nmsg}")
        return briefs[:12]

    raw_errors: Sequence[Any] = []
    if hasattr(exc, "errors"):
        try:
            raw_errors = list(exc.errors())  # type: ignore[call-arg]
        except Exception:
            raw_errors = []
    if not raw_errors:
        msg = str(exc).strip().splitlines()
        briefs.append(msg[0][:300] if msg else type(exc).__name__)
        return briefs
    for item in list(raw_errors)[:20]:
        if isinstance(item, dict):
            path = ".".join(str(p) for p in item.get("path", []) or []) or "$"
            msg = str(item.get("message") or "invalid").splitlines()[0][:200]
            briefs.append(f"{path}: {msg}")
        else:
            briefs.append(str(item)[:200])
    if len(raw_errors) > 20:
        briefs.append(f"... {len(raw_errors) - 20} more")
    return briefs


def validate_against_schema(
    instance_path: Optional[Path],
    schema_rel: Path,
    label: str,
) -> Dict[str, Any]:
    """Validate a JSON file against a contract schema.

    Returns contract_validation entry with status:
    valid | invalid | not_applicable | skipped_dependency | missing_schema | missing_file
    """
    result: Dict[str, Any] = {
        "status": "missing_schema",
        "schema": str(schema_rel),
        "errors": [],
    }
    schema_path = _resolve_contract(schema_rel)
    if schema_path is None:
        result["status"] = "missing_schema"
        result["errors"] = [f"schema not found: {schema_rel}"]
        return result

    try:
        import jsonschema  # lazy
    except ImportError:
        result["status"] = "skipped_dependency"
        result["errors"] = ["jsonschema not installed; validation skipped"]
        return result

    if instance_path is None:
        result["status"] = "not_applicable"
        result["errors"] = [f"{label} not produced by this side"]
        return result
    path = Path(instance_path)
    if not path.exists():
        result["status"] = "missing_file"
        result["errors"] = [f"file not found: {path}"]
        return result

    try:
        with open(path, "r", encoding="utf-8") as f:
            instance = json.load(f)
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as e:
        result["status"] = "invalid"
        result["errors"] = [f"load error: {type(e).__name__}: {e}"]
        return result

    try:
        jsonschema.validate(instance=instance, schema=schema)
        result["status"] = "valid"
        result["errors"] = []
    except jsonschema.ValidationError as e:
        result["status"] = "invalid"
        result["errors"] = _brief_validation_errors(e)
    except jsonschema.SchemaError as e:
        result["status"] = "missing_schema"
        result["errors"] = _brief_validation_errors(e)
    except Exception as e:
        result["status"] = "invalid"
        result["errors"] = [f"{type(e).__name__}: {e}"]
    return result


def empty_contract_validation() -> Dict[str, Any]:
    return {
        "knowledge_base": {
            "status": "not_applicable",
            "schema": str(KB_SCHEMA_REL),
            "errors": [],
        },
        "canonical_ir": {
            "status": "not_applicable",
            "schema": str(IR_SCHEMA_REL),
            "errors": [],
        },
    }


def apply_contract_validation(result: Dict[str, Any], side: str) -> None:
    """Attach contract_validation and demote ok → failed if schema invalid.

    For legacy: validates both raw and compat copies. Status demotion uses the
    **compat** knowledge_base (and IR for v2). Raw invalid alone does not fail
    the side when compat is valid — a warning is recorded instead.
    """
    cv = empty_contract_validation()
    outputs = result.get("outputs") or {}
    kb_path = outputs.get("knowledge_base")
    raw_kb_path = outputs.get("knowledge_base_raw")
    ir_path = outputs.get("canonical_ir")

    cv["knowledge_base"] = validate_against_schema(kb_path, KB_SCHEMA_REL, "knowledge_base")
    if raw_kb_path:
        cv["raw_knowledge_base"] = validate_against_schema(
            raw_kb_path, KB_SCHEMA_REL, "raw_knowledge_base"
        )
    if side == "v2":
        cv["canonical_ir"] = validate_against_schema(ir_path, IR_SCHEMA_REL, "canonical_ir")
    else:
        cv["canonical_ir"] = {
            "status": "not_applicable",
            "schema": str(IR_SCHEMA_REL),
            "errors": ["legacy side does not emit Canonical IR"],
        }

    # Compatibility adapter report (legacy only)
    compat = result.get("compatibility")
    if compat is not None:
        cv["compatibility"] = compat

    result["contract_validation"] = cv

    # Raw invalid + compat valid → warn, keep side ok (do not hide raw).
    raw_cv = cv.get("raw_knowledge_base")
    compat_cv = cv.get("knowledge_base")
    if (
        side == "legacy"
        and raw_cv
        and raw_cv.get("status") == "invalid"
        and compat_cv
        and compat_cv.get("status") == "valid"
    ):
        warn = "legacy raw KB schema invalid; used compatibility adapter for compat copy"
        if warn not in result.get("warnings", []):
            result.setdefault("warnings", []).append(warn)

    if result.get("status") == "ok":
        # Only demote on COMPAT (or v2 IR) invalid — not on raw-only invalid.
        parts = [compat_cv]
        if side == "v2":
            parts.append(cv.get("canonical_ir"))
        for part in parts:
            if part and part.get("status") == "invalid":
                result["status"] = "failed"
                result.setdefault("errors", []).append(
                    f"contract_validation invalid: {part.get('schema')}"
                )
                for err in (part.get("errors") or [])[:5]:
                    result.setdefault("errors", []).append(f"schema: {err}")
                break


def run_legacy(
    run_dir: Path,
    input_path: Path,
    caps: Dict[str, Any],
    max_pages: Optional[int] = None,
    start_page: int = 1,
) -> Dict[str, Any]:
    """Thin subprocess adapter to tools/pdf_to_base64_kb.py (isolated out dir)."""
    legacy_dir = run_dir / "legacy"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "status": "failed",
        "outputs": {"dir": str(legacy_dir)},
        "metrics": None,
        "elapsed_ms": 0,
        "errors": [],
        "warnings": [],
        "contract_validation": empty_contract_validation(),
    }

    missing = missing_hard_deps("legacy", caps)
    if missing:
        result["status"] = "blocked_by_dependency"
        pkgs = ", ".join(PKG_NAMES[m] for m in missing)
        result["errors"].append(f"missing dependencies: {pkgs}")
        result["warnings"].append(
            "suggested install: pip install " + " ".join(PKG_NAMES[m] for m in missing)
        )
        return result

    kb_raw = legacy_dir / "knowledge_base.raw.json"
    kb_out = legacy_dir / "knowledge_base.json"
    assets_root = legacy_dir / "assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    legacy_script = _tool_script("pdf_to_base64_kb.py")
    if legacy_script is None:
        result["status"] = "blocked_by_dependency"
        result["errors"].append(
            "legacy pipeline script tools/pdf_to_base64_kb.py is not available in this installation"
        )
        result["warnings"].append(
            "legacy/shadow require a source checkout or a wheel that bundles tools/"
        )
        return result
    cmd = [
        sys.executable,
        str(legacy_script),
        "--pdf",
        str(input_path),
        "--out",
        str(kb_raw),
        "--assets-root",
        str(assets_root),
    ]
    if start_page and start_page > 1:
        cmd.extend(["--start-page", str(start_page)])
    if max_pages:
        cmd.extend(["--max-pages", str(max_pages)])
    env = os.environ.copy()
    root = _repo_root()
    if root is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(root), str(root / "src")]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
    started = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root) if root is not None else str(legacy_dir),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["outputs"]["knowledge_base_raw"] = str(kb_raw)
        result["outputs"]["knowledge_base"] = str(kb_out)
        result["outputs"]["command"] = cmd
        if proc.returncode == 0 and kb_raw.exists():
            # Legacy contract adapter: raw → compat (id fill only)
            try:
                with open(kb_raw, "r", encoding="utf-8") as f:
                    raw_kb = json.load(f)
                _ensure_repo_on_path()
                from pipeline.compatibility.legacy_kb import (
                    ADAPTER_NAME,
                    normalize_legacy_kb_contract,
                )

                compat_kb, compat_report = normalize_legacy_kb_contract(raw_kb)
                _write_json(kb_out, compat_kb)
                result["compatibility"] = {
                    "adapter": ADAPTER_NAME,
                    "applied": compat_report.get("applied", False),
                    "transforms": compat_report.get("transforms", {}),
                    "changed_paths": compat_report.get("changed_paths", []),
                }
                if compat_report.get("applied"):
                    n = (compat_report.get("transforms") or {}).get("missing_block_ids", 0)
                    result.setdefault("warnings", []).append(
                        f"legacy contract adapter filled {n} missing block id(s)"
                    )
                result["metrics"] = kb_metrics(
                    kb_out, include_figure_association=False
                )
                result["status"] = "ok"
            except Exception as adj_e:
                result["metrics"] = kb_metrics(
                    kb_raw, include_figure_association=False
                )
                result["status"] = "failed"
                result["errors"].append(
                    f"legacy contract adapter failed: {type(adj_e).__name__}: {adj_e}"
                )
        else:
            combined = ((proc.stderr or "") + "\n" + (proc.stdout or "")).strip()
            tail = combined[-800:] if combined else f"exit code {proc.returncode}"
            if "ModuleNotFoundError" in combined or "ImportError" in combined:
                result["status"] = "blocked_by_dependency"
            else:
                result["status"] = "failed"
            result["errors"].append(tail)
            if proc.returncode != 0 and "Traceback" in combined:
                lines = [ln for ln in combined.splitlines() if ln.strip()]
                if lines:
                    result["errors"] = [lines[-1]]
            if kb_raw.exists():
                result["metrics"] = kb_metrics(
                    kb_raw, include_figure_association=False
                )
    except subprocess.TimeoutExpired:
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["errors"].append("legacy pipeline timed out after 3600s")
    except FileNotFoundError as e:
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["status"] = "blocked_by_dependency"
        result["errors"].append(str(e))
    except Exception as e:
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["errors"].append(f"{type(e).__name__}: {e}")

    apply_contract_validation(result, side="legacy")
    return result


def check_figure_reference_metrics(
    kb_metrics: Optional[Dict[str, Any]],
    fr_stats: Optional[Dict[str, Any]],
    *,
    report_exists: bool,
) -> List[str]:
    """KB/report consistency gate for Phase 2G figure reference metrics.

    KB is product truth for labels/references observed in the final document;
    figure_reference_index.json stats must match exactly (no max/min merge).
    """
    errors: List[str] = []
    if kb_metrics is None:
        errors.append("figure_reference metrics unavailable (no KB)")
        return errors
    if not report_exists:
        errors.append("figure_reference_index report missing")
        return errors
    if fr_stats is None:
        errors.append("figure_reference_index report unreadable (no stats)")
        return errors
    for key in FIGURE_REFERENCE_METRIC_KEYS:
        kb_val = int(kb_metrics.get(key) or 0)
        rep_val = int(fr_stats.get(key) or 0)
        if kb_val != rep_val:
            errors.append(f"{key} mismatch: kb={kb_val} report={rep_val}")
    return errors


def check_figure_caption_ocr_metrics(
    kb_metrics: Optional[Dict[str, Any]],
    fc_stats: Optional[Dict[str, Any]],
    *,
    report_exists: bool,
) -> List[str]:
    """KB/report consistency gate for OCR caption metrics (Phase 2F.1).

    KB is product truth for linked/figures counts; OCR report must match
    exactly. Returns a list of field-level error strings (empty = consistent).
    """
    errors: List[str] = []
    if kb_metrics is None:
        errors.append("figure_caption_ocr metrics unavailable (no KB)")
        return errors
    if not report_exists:
        errors.append("figure_caption_ocr report missing")
        return errors
    if fc_stats is None:
        errors.append("figure_caption_ocr report unreadable (no stats)")
        return errors
    for key in ("ocr_captions_linked", "figures_with_ocr_caption"):
        kb_val = int(kb_metrics.get(key) or 0)
        rep_val = int(fc_stats.get(key) or 0)
        if kb_val != rep_val:
            errors.append(f"{key} mismatch: kb={kb_val} report={rep_val}")
    return errors


def check_v2_referenced_assets(
    kb: Optional[Dict[str, Any]],
    output_dir: Path,
) -> List[str]:
    """Contract-required asset presence for a v2 KB (degradation stays allowed).

    Only a block that *claims* a deliverable local asset — a non-empty relative
    ``imageUri``/``src`` whose file is absent under the run directory — is an
    inconsistency. Blocks that mark an asset missing/degraded (empty uri,
    assetStatus="missing", cropWarnings) and non-local URIs (absolute paths,
    URLs) are the sanctioned degradation and are never turned into a failure.
    """
    if not isinstance(kb, dict):
        return ["knowledge_base unavailable for referenced-asset check"]
    base = Path(output_dir)
    errors: List[str] = []
    seen: Set[str] = set()
    for entry in kb.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        for block in entry.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            uri = block.get("imageUri") or block.get("src") or ""
            if not isinstance(uri, str) or not uri.strip():
                continue
            uri = uri.strip()
            if "://" in uri or Path(uri).is_absolute():
                continue
            if uri in seen:
                continue
            seen.add(uri)
            if not (base / uri).is_file():
                errors.append(
                    f"missing referenced asset: {uri} (block {block.get('id')})"
                )
    return errors


def run_v2(
    run_dir: Path,
    input_path: Path,
    caps: Dict[str, Any],
    profile: str,
    max_pages: Optional[int],
    start_page: int = 1,
    figure_caption_ocr: str = "off",
) -> Dict[str, Any]:
    """Run pipeline.v2_runner.run_v2 into outputs/runs/<id>/v2/."""
    v2_dir = run_dir / "v2"
    v2_dir.mkdir(parents=True, exist_ok=True)
    # Bound here so the publication boundary stays safe if the (lazy) pipeline
    # import itself fails; the pipeline import block rebinds them.
    v2_staged_paths = None
    publish_staged_outputs = None
    discard_staged_outputs = None
    result: Dict[str, Any] = {
        "status": "failed",
        "outputs": {"dir": str(v2_dir), "published": False},
        "metrics": None,
        "elapsed_ms": 0,
        "errors": [],
        "warnings": [],
        "contract_validation": empty_contract_validation(),
        "figure_caption_ocr": {
            "requested_engine": figure_caption_ocr,
            "selected_engine": None,
            "language": None,
            "capability": None,
        },
    }

    # Refuse to reuse a run directory that already holds a published deliverable.
    # Checked before anything is written, this protects the whole previous set
    # (KB, IR and shots). A first failure publishes no KB, so a same-directory
    # retry is still allowed.
    try:
        from pipeline.v2_runner import published_kb_path

        published_kb = published_kb_path(v2_dir)
    except Exception:
        published_kb = v2_dir / "knowledge_base.json"
    if published_kb.exists():
        result["errors"].append(
            "refusing to run: run directory already contains a published "
            f"{published_kb.name}; move or remove it before rerunning"
        )
        return result

    missing = missing_hard_deps("v2", caps)
    if missing:
        result["status"] = "blocked_by_dependency"
        pkgs = ", ".join(PKG_NAMES[m] for m in missing)
        result["errors"].append(f"missing dependencies: {pkgs}")
        result["warnings"].append(
            "suggested install: pip install " + " ".join(PKG_NAMES[m] for m in missing)
        )
        return result

    # Phase 2F capability gate — probe BEFORE running so failures are a clean
    # blocked_by_dependency (full run report, no deep traceback). Never fall
    # back to another OCR engine when tesseract was requested.
    fc_language: Optional[str] = None
    if figure_caption_ocr and figure_caption_ocr != "off":
        _ensure_repo_on_path()
        from imaging.figure_caption_ocr import (
            DEFAULT_LANGUAGE,
            probe_tesseract_capability,
        )

        fc_language = DEFAULT_LANGUAGE
        fc_cap = probe_tesseract_capability(DEFAULT_LANGUAGE)
        result["figure_caption_ocr"]["capability"] = fc_cap
        result["figure_caption_ocr"]["language"] = DEFAULT_LANGUAGE
        if not fc_cap.get("ready"):
            result["status"] = "blocked_by_dependency"
            miss = ", ".join(fc_cap.get("missing") or ["unknown"]) or "unknown"
            result["errors"].append(
                f"figure-caption-ocr capability missing: {miss}"
            )
            result["warnings"].append(
                "requested figure-caption-ocr engine stays blocked; "
                "no silent fallback to another engine"
            )
            result["elapsed_ms"] = 0
            apply_contract_validation(result, side="v2")
            return result
        result["figure_caption_ocr"]["selected_engine"] = figure_caption_ocr
    else:
        result["figure_caption_ocr"]["capability"] = {"ready": True, "engine": "off"}

    _ensure_repo_on_path()
    started = time.time()
    try:
        from models.run_context import BuildProfile, RunContext
        from pipeline.v2_runner import (
            discard_staged_outputs,
            publish_staged_outputs,
            run_v2 as pipeline_run_v2,
            v2_staged_paths,
        )

        build_profile = BuildProfile()
        build_profile.name = profile
        if profile == "smoke":
            build_profile.ocr_enabled = False
        build_profile.figure_caption_ocr = figure_caption_ocr or "off"

        ctx = RunContext(
            run_id=run_dir.name,
            input_path=input_path,
            output_dir=v2_dir,
            profile=build_profile,
        )
        start = max(1, int(start_page or 1))
        if max_pages:
            end = start + int(max_pages) - 1
            page_range = f"{start}-{end}"
        else:
            page_range = str(start) if start > 1 else None
        # Defer publication: the pipeline stages IR + KB; the CLI validates and
        # only then publishes (see the publication boundary at the end).
        _, kb_final = pipeline_run_v2(ctx, page_range, publish=False)

        staged_kb, staged_ir = v2_staged_paths(v2_dir)
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        # Validation and metrics read the staged artifacts; nothing is published
        # until every gate below passes.
        result["outputs"]["knowledge_base"] = str(staged_kb)
        if staged_ir.exists():
            result["outputs"]["canonical_ir"] = str(staged_ir)
        fc_report_path = v2_dir / "figure_caption_ocr.json"
        if fc_report_path.exists():
            result["outputs"]["figure_caption_ocr"] = str(fc_report_path)
        fr_report_path = v2_dir / "figure_reference_index.json"
        if fr_report_path.exists():
            result["outputs"]["figure_reference_index"] = str(fr_report_path)
        result["metrics"] = kb_metrics(staged_kb)
        if result["metrics"] is None and kb_final is not None:
            result["metrics"] = kb_metrics_from_obj(kb_final)

        # Phase 2F.1: OCR report merge + KB/report consistency gate.
        # - attempted/cache/rejected/ambiguous/elapsed come from the OCR report.
        # - ocr_captions_linked / figures_with_ocr_caption: KB is product truth;
        #   report must match exactly — no max()/min()/defaults to hide drift.
        ocr_metric_gate_failed = False
        if figure_caption_ocr and figure_caption_ocr != "off":
            fc_stats: Optional[Dict[str, Any]] = None
            if fc_report_path.exists():
                try:
                    with open(fc_report_path, "r", encoding="utf-8") as f:
                        fc_report = json.load(f)
                    fc_stats = fc_report.get("stats") or {}
                    if result["metrics"] is not None:
                        for key in (
                            "figure_caption_ocr_attempted",
                            "figure_caption_ocr_cache_hits",
                            "ocr_captions_rejected",
                            "ocr_captions_ambiguous",
                            "ocr_caption_elapsed_ms",
                        ):
                            if key in fc_stats:
                                result["metrics"][key] = fc_stats[key]
                except Exception as fc_e:
                    result["warnings"].append(
                        f"figure_caption_ocr report unreadable: {fc_e}"
                    )
                    fc_stats = None
            gate_errors = check_figure_caption_ocr_metrics(
                result["metrics"],
                fc_stats,
                report_exists=fc_report_path.exists(),
            )
            if gate_errors:
                ocr_metric_gate_failed = True
                for ge in gate_errors:
                    result["errors"].append(f"figure_caption_ocr: {ge}")

        # Phase 2G: figure reference report + KB/report consistency gate.
        fr_metric_gate_failed = False
        fr_stats: Optional[Dict[str, Any]] = None
        if fr_report_path.exists():
            try:
                with open(fr_report_path, "r", encoding="utf-8") as f:
                    fr_report = json.load(f)
                fr_stats = fr_report.get("stats") or {}
            except Exception as fr_e:
                result["warnings"].append(
                    f"figure_reference_index report unreadable: {fr_e}"
                )
                fr_stats = None
        fr_gate_errors = check_figure_reference_metrics(
            result["metrics"],
            fr_stats,
            report_exists=fr_report_path.exists(),
        )
        if fr_gate_errors:
            fr_metric_gate_failed = True
            for ge in fr_gate_errors:
                result["errors"].append(f"figure_reference: {ge}")

        # Referenced-asset gate: a block that claims a deliverable local asset
        # must have that file present. Marked-missing/degraded assets stay valid.
        asset_gate_failed = False
        if kb_final is not None:
            asset_errors = check_v2_referenced_assets(kb_final, v2_dir)
            if asset_errors:
                asset_gate_failed = True
                for ae in asset_errors:
                    result["errors"].append(f"v2 assets: {ae}")

        if ocr_metric_gate_failed or fr_metric_gate_failed or asset_gate_failed:
            result["status"] = "failed"
        else:
            result["status"] = "ok" if result["metrics"] is not None else "failed"
            if result["status"] == "failed":
                result["errors"].append(
                    "v2 produced no readable knowledge_base.json"
                )
    except ModuleNotFoundError as e:
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["status"] = "blocked_by_dependency"
        name = getattr(e, "name", None) or str(e)
        pkg = PKG_NAMES.get(str(name).split(".")[0], str(name))
        result["errors"].append(f"missing dependency: {pkg}")
        result["warnings"].append(f"suggested install: pip install {pkg}")
    except ImportError as e:
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["status"] = "blocked_by_dependency"
        result["errors"].append(str(e))
    except Exception as e:
        result["elapsed_ms"] = int((time.time() - started) * 1000)
        result["errors"].append(f"{type(e).__name__}: {e}")
        tb = traceback.format_exc().strip().splitlines()
        if tb:
            result["warnings"].append("traceback_tail: " + " | ".join(tb[-5:]))

    apply_contract_validation(result, side="v2")

    # Publication boundary. Only a run whose schema/report gates all passed may
    # expose the deliverable KB + IR. A failed run discards its staged copies
    # and never reports a knowledge_base output; a previously published pair is
    # left untouched.
    if publish_staged_outputs is None or discard_staged_outputs is None:
        # The pipeline never imported, so nothing was staged; there is no
        # deliverable to publish.
        result["outputs"].pop("knowledge_base", None)
        result["outputs"].pop("canonical_ir", None)
        result["outputs"]["published"] = False
    elif result.get("status") == "ok":
        try:
            final_kb, final_ir = publish_staged_outputs(v2_dir)
        except Exception as pub_e:
            # A rename failure must never be reported as success.
            discard_staged_outputs(v2_dir)
            result["status"] = "failed"
            result["errors"].append(
                f"publication failed: {type(pub_e).__name__}: {pub_e}"
            )
            result["outputs"].pop("knowledge_base", None)
            result["outputs"].pop("canonical_ir", None)
            result["outputs"]["published"] = False
        else:
            result["outputs"]["knowledge_base"] = str(final_kb)
            if final_ir.exists():
                result["outputs"]["canonical_ir"] = str(final_ir)
            else:
                result["outputs"].pop("canonical_ir", None)
            result["outputs"]["published"] = True
    else:
        discard_staged_outputs(v2_dir)
        result["outputs"].pop("knowledge_base", None)
        result["outputs"].pop("canonical_ir", None)
        result["outputs"]["published"] = False

    return result


def build_shadow_diff(
    run_id: str,
    legacy_result: Dict[str, Any],
    v2_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare only keys present with real values on BOTH sides; no fabricated deltas."""
    leg_m = legacy_result.get("metrics")
    v2_m = v2_result.get("metrics")
    diff: Dict[str, Any] = {}
    skipped: List[str] = []

    if leg_m and v2_m:
        for key in SHADOW_COMPARABLE_KEYS:
            if key in leg_m and key in v2_m and leg_m[key] is not None and v2_m[key] is not None:
                diff[key] = {
                    "legacy": leg_m[key],
                    "v2": v2_m[key],
                    "delta": v2_m[key] - leg_m[key],
                }
            else:
                skipped.append(key)
        if skipped:
            diff["skipped_keys"] = skipped
            diff["note"] = "keys missing on at least one side; no delta emitted"
    else:
        diff["note"] = "metrics unavailable on at least one side; delta not computed"

    return {
        "run_id": run_id,
        "generated_at": _utc_now(),
        "legacy": {
            "status": legacy_result.get("status"),
            "elapsed_ms": legacy_result.get("elapsed_ms", 0),
            "metrics": leg_m,
            "errors": legacy_result.get("errors", []),
            "contract_validation": legacy_result.get("contract_validation"),
        },
        "v2": {
            "status": v2_result.get("status"),
            "elapsed_ms": v2_result.get("elapsed_ms", 0),
            "metrics": v2_m,
            "errors": v2_result.get("errors", []),
            "contract_validation": v2_result.get("contract_validation"),
        },
        "elapsed_ms": {
            "legacy": legacy_result.get("elapsed_ms", 0),
            "v2": v2_result.get("elapsed_ms", 0),
        },
        "diff": diff,
    }


def _overall_status(sides: List[Dict[str, Any]]) -> str:
    statuses = [s.get("status") for s in sides]
    if all(st == "ok" for st in statuses):
        return "ok"
    if any(st == "blocked_by_dependency" for st in statuses):
        # Prefer failed over blocked if any side failed hard after deps present.
        if any(st == "failed" for st in statuses):
            return "failed"
        return "blocked_by_dependency"
    return "failed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m paddle_models.cli.main",
        description=(
            "PaddleModels unified CLI (Phase 1). "
            "Runs legacy / v2 / shadow with isolated run dirs and JSON reports."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["legacy", "v2", "shadow"],
        default="v2",
        help="Execution mode (default: v2)",
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="<pdf>",
        help="Path to input PDF",
    )
    parser.add_argument(
        "--out",
        default="outputs/runs",
        metavar="<directory>",
        help="Base output directory for runs (default: outputs/runs)",
    )
    parser.add_argument(
        "--profile",
        choices=["smoke", "default"],
        default="default",
        help="Build profile (smoke limits work; default: default)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="<N>",
        help="Process at most N pages (smoke defaults to 2 if omitted)",
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=1,
        metavar="<N>",
        help="1-based start page (default: 1); must be a positive integer",
    )
    parser.add_argument(
        "--figure-caption-ocr",
        choices=["off", "tesseract"],
        default="off",
        help=(
            "Figure-internal caption OCR engine (default: off). "
            "Only bottom-band ROI of ready embedded figures is OCRed; "
            "never full-page OCR. Explicit tesseract never falls back silently."
        ),
    )
    return parser


def _positive_page(value: int, name: str) -> int:
    if value is None:
        return 1
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise SystemExit(f"[ERROR] {name} must be a positive integer, got {value!r}")
    if iv < 1:
        raise SystemExit(f"[ERROR] {name} must be a positive integer, got {value!r}")
    return iv


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    command = list(argv) if argv is not None else list(sys.argv)
    started_at = _utc_now()
    t0 = time.time()

    input_path = Path(args.input).expanduser().resolve()
    out_base = Path(args.out).expanduser()
    if not out_base.is_absolute():
        out_base = (Path.cwd() / out_base).resolve()

    max_pages = args.max_pages
    if args.profile == "smoke" and max_pages is None:
        max_pages = 2
    start_page = _positive_page(args.start_page, "--start-page")
    if max_pages is not None:
        max_pages = _positive_page(max_pages, "--max-pages")

    caps = probe_capabilities(args.mode)

    run_id = make_run_id(input_path.stem)
    run_dir = out_base / run_id
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    errors: List[str] = []
    warnings: List[str] = []
    outputs: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "reports_dir": str(reports_dir),
    }
    metrics: Dict[str, Any] = {}
    side_results: Dict[str, Dict[str, Any]] = {}

    input_ok = input_path.exists() and input_path.is_file()
    input_sha = _sha256_file(input_path) if input_ok else None
    if not input_ok:
        errors.append(f"input not found: {args.input}")

    modes = ["legacy", "v2"] if args.mode == "shadow" else [args.mode]

    if input_ok:
        if args.mode in ("legacy", "shadow"):
            side = run_legacy(run_dir, input_path, caps, max_pages, start_page)
            side_results["legacy"] = side
            outputs["legacy"] = side.get("outputs", {})
            if side.get("errors"):
                errors.extend(f"legacy: {e}" for e in side["errors"])
            if side.get("warnings"):
                warnings.extend(f"legacy: {w}" for w in side["warnings"])

        if args.mode in ("v2", "shadow"):
            side = run_v2(
                run_dir,
                input_path,
                caps,
                args.profile,
                max_pages,
                start_page,
                figure_caption_ocr=args.figure_caption_ocr,
            )
            side_results["v2"] = side
            outputs["v2"] = side.get("outputs", {})
            if side.get("errors"):
                errors.extend(f"v2: {e}" for e in side["errors"])
            if side.get("warnings"):
                warnings.extend(f"v2: {w}" for w in side["warnings"])

        if args.mode == "shadow":
            shadow_diff = build_shadow_diff(
                run_id,
                side_results.get(
                    "legacy",
                    {
                        "status": "not_run",
                        "metrics": None,
                        "elapsed_ms": 0,
                        "errors": ["not run"],
                    },
                ),
                side_results.get(
                    "v2",
                    {
                        "status": "not_run",
                        "metrics": None,
                        "elapsed_ms": 0,
                        "errors": ["not run"],
                    },
                ),
            )
            diff_path = reports_dir / "shadow_diff.json"
            _write_json(diff_path, shadow_diff)
            outputs["shadow_diff"] = str(diff_path)

            # Table structure diagnosis (page-focused when start_page set)
            try:
                _ensure_repo_on_path()
                from pipeline.compatibility.table_diff import build_table_diff_report

                leg_out = (side_results.get("legacy") or {}).get("outputs") or {}
                v2_out = (side_results.get("v2") or {}).get("outputs") or {}
                leg_kb_path = leg_out.get("knowledge_base")
                v2_kb_path = v2_out.get("knowledge_base")
                if leg_kb_path and v2_kb_path and Path(leg_kb_path).exists() and Path(v2_kb_path).exists():
                    with open(leg_kb_path, "r", encoding="utf-8") as f:
                        leg_kb_obj = json.load(f)
                    with open(v2_kb_path, "r", encoding="utf-8") as f:
                        v2_kb_obj = json.load(f)
                    # Prefer primary start_page when it has tables, else page 29 default filter=None scans all
                    page_filter = start_page if start_page and start_page > 1 else None
                    table_diff = build_table_diff_report(
                        leg_kb_obj, v2_kb_obj, page_number=page_filter
                    )
                    # If filter produced nothing useful, rebuild without filter for diagnosis
                    if not table_diff.get("comparisons"):
                        table_diff = build_table_diff_report(leg_kb_obj, v2_kb_obj, page_number=None)
                    td_path = reports_dir / "table_diff.json"
                    _write_json(td_path, table_diff)
                    outputs["table_diff"] = str(td_path)
            except Exception as td_e:
                warnings.append(f"table_diff generation failed: {td_e}")

            metrics = {
                "legacy": side_results.get("legacy", {}).get("metrics"),
                "v2": side_results.get("v2", {}).get("metrics"),
            }
        else:
            only = side_results.get(args.mode, {})
            metrics = only.get("metrics") or {}
            if args.mode == "legacy":
                outputs["legacy"] = only.get("outputs", {})
            else:
                outputs["v2"] = only.get("outputs", {})

        overall = _overall_status(
            [side_results[m] for m in modes if m in side_results]
        )
    else:
        overall = "failed"
        metrics = {}

    finished_at = _utc_now()
    elapsed_ms = int((time.time() - t0) * 1000)

    run_report = {
        "run_id": run_id,
        "mode": args.mode,
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": elapsed_ms,
        "profile": args.profile,
        "max_pages": max_pages,
        "start_page": start_page,
        "figure_caption_ocr": {
            "requested_engine": args.figure_caption_ocr,
            "selected_engine": (
                (side_results.get("v2") or {})
                .get("figure_caption_ocr", {})
                .get("selected_engine")
            ),
            "language": (
                (side_results.get("v2") or {})
                .get("figure_caption_ocr", {})
                .get("language")
            ),
            "capability": (
                (side_results.get("v2") or {})
                .get("figure_caption_ocr", {})
                .get("capability")
            ),
        },
        "capabilities": caps,
        "outputs": outputs,
        "metrics": metrics,
        "warnings": warnings,
        "errors": errors,
        "command": command,
        "status": overall if input_ok else "failed",
        "contract_validation": {
            name: side.get("contract_validation")
            for name, side in side_results.items()
        },
        "compatibility": {
            name: side.get("compatibility")
            for name, side in side_results.items()
            if side.get("compatibility") is not None
        },
    }
    report_path = reports_dir / "run_report.json"
    _write_json(report_path, run_report)

    print(f"run_id={run_id}")
    print(f"mode={args.mode} profile={args.profile} start_page={start_page} max_pages={max_pages}")
    fc_info = run_report.get("figure_caption_ocr") or {}
    print(
        f"figure_caption_ocr: requested={fc_info.get('requested_engine')} "
        f"selected={fc_info.get('selected_engine')} "
        f"language={fc_info.get('language')} "
        f"capability_ready={(fc_info.get('capability') or {}).get('ready')}"
    )
    print(f"capabilities: {caps['modules']}")
    if caps["missing_packages"]:
        print(f"missing packages: {', '.join(caps['missing_packages'])}")
    print(f"run_dir={run_dir}")
    print(f"run_report={report_path}")
    if outputs.get("shadow_diff"):
        print(f"shadow_diff={outputs['shadow_diff']}")
    if outputs.get("table_diff"):
        print(f"table_diff={outputs['table_diff']}")
    for name, side in side_results.items():
        cv = side.get("contract_validation") or {}
        kb_cv = (cv.get("knowledge_base") or {}).get("status")
        raw_cv = (cv.get("raw_knowledge_base") or {}).get("status")
        ir_cv = (cv.get("canonical_ir") or {}).get("status")
        compat = side.get("compatibility") or {}
        compat_s = ""
        if compat:
            compat_s = (
                f" compat_applied={compat.get('applied')} "
                f"transforms={compat.get('transforms')}"
            )
        raw_s = f" contract_raw={raw_cv}" if raw_cv else ""
        print(
            f"side {name}: status={side.get('status')} "
            f"elapsed_ms={side.get('elapsed_ms', 0)} metrics={side.get('metrics')} "
            f"contract_kb={kb_cv}{raw_s} contract_ir={ir_cv}{compat_s}"
        )
    for w in warnings:
        print(f"[WARN] {w}")
    for e in errors:
        print(f"[ERROR] {e}")
    print(f"status={overall} elapsed_ms={elapsed_ms}")

    return 0 if overall == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
