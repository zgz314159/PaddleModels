import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pipeline.canonical_ir import CanonicalIR, BBox

# Projection classes for text-like semantic roles.
_SEARCHABLE_ROLES = frozenset({"heading", "body", "caption", "table_note"})
_SUPPORTING_ROLES = frozenset({
    "figure_callout", "legend", "annotation", "ocr_annotation",
})
_EXCLUDED_ROLES = frozenset({"artifact"})


def classify_text_projection(semantic_role: Optional[str]) -> str:
    """Map a semantic role to searchable | supporting | excluded.

    Unknown or missing roles default to searchable so content is not silently lost.
    """
    if semantic_role is None:
        return "searchable"
    role = str(semantic_role).strip()
    if not role:
        return "searchable"
    # Preserve exact known tokens; also accept lower-case variants.
    if role in _EXCLUDED_ROLES or role.lower() in _EXCLUDED_ROLES:
        return "excluded"
    if role in _SUPPORTING_ROLES or role.lower() in _SUPPORTING_ROLES:
        return "supporting"
    if role in _SEARCHABLE_ROLES or role.lower() in _SEARCHABLE_ROLES:
        return "searchable"
    return "searchable"


class SemanticProjector:
    """Projects CanonicalIR into Android-compatible Knowledge Base entries."""

    def __init__(self, document_id: str, strategy: str = "heading"):
        self.document_id = document_id
        self.strategy = strategy

    def project(self, ir: CanonicalIR) -> Dict[str, Any]:
        entries: List[Dict[str, Any]] = []
        page_sizes: Dict[str, List[float]] = {}
        current_entry: Optional[Dict[str, Any]] = None
        # Page-scoped supporting text waiting for a searchable entry host.
        pending_supporting: List[Dict[str, Any]] = []
        # Page-scoped figure blocks waiting for a host (Phase 1.3 pending pattern).
        pending_figures: List[Dict[str, Any]] = []
        page_has_entry = False

        for page in ir.pages:
            p_num = page['page_number'] if isinstance(page, dict) else page.page_number
            p_width = page['width'] if isinstance(page, dict) else page.width
            p_height = page['height'] if isinstance(page, dict) else page.height
            page_sizes[str(p_num)] = [p_width, p_height]

            blocks_data = page['blocks'] if isinstance(page, dict) else page.blocks
            pending_supporting = []
            pending_figures = []
            page_has_entry = False
            first_block_on_page = True

            for block_data in blocks_data:
                if not isinstance(block_data, dict):
                    b_id = block_data.id
                    b_type = block_data.type
                    b_text = block_data.text
                    b_bbox = block_data.bbox
                    b_meta = block_data.metadata
                else:
                    b_id = block_data['id']
                    b_type = block_data['type']
                    b_text = block_data['text']
                    bb = block_data['bbox']
                    b_bbox = BBox(bb['x'], bb['y'], bb['w'], bb['h'])
                    b_meta = block_data['metadata']

                role = b_meta.get("semanticRole") if isinstance(b_meta, dict) else None

                # --- Text-like blocks: three-way projection ---
                if b_type in ("text", "heading", "caption"):
                    projection = classify_text_projection(role)
                    if projection == "excluded":
                        continue  # stays in Canonical IR only

                    is_heading = role == "heading" or b_type == "heading"
                    block_dict = self._text_block_dict(
                        b_id, b_text, p_num, role, b_bbox, searchable=(projection == "searchable")
                    )
                    # Phase 2G: copy body figureReferences (metadata only — never contentMarkdown).
                    fig_refs = b_meta.get("figureReferences") if isinstance(b_meta, dict) else None
                    if isinstance(fig_refs, list) and fig_refs:
                        block_dict["figureReferences"] = list(fig_refs)

                    if projection == "supporting":
                        # Do not create/trigger an entry solely from supporting text.
                        if page_has_entry and current_entry is not None and current_entry["pageNumber"] == p_num:
                            current_entry["blocks"].append(block_dict)
                        else:
                            pending_supporting.append(block_dict)
                        continue

                    # searchable
                    should_start_new = False
                    if self.strategy == "heading":
                        should_start_new = is_heading or current_entry is None
                    elif self.strategy == "page":
                        should_start_new = (
                            current_entry is None
                            or (current_entry["blocks"] and current_entry["pageNumber"] != p_num)
                        )

                    if should_start_new:
                        if current_entry:
                            entries.append(self._finalize_entry(current_entry))
                        current_entry = self._new_entry(
                            b_id, b_text if is_heading else f"Page {p_num}", p_num, entries
                        )
                        current_entry["blocks"].extend(pending_supporting)
                        current_entry["blocks"].extend(pending_figures)
                        pending_supporting = []
                        pending_figures = []
                        page_has_entry = True
                    else:
                        # Select existing entry as host for this page's pending blocks.
                        if pending_supporting and current_entry is not None:
                            current_entry["blocks"].extend(pending_supporting)
                            pending_supporting = []
                        if pending_figures and current_entry is not None:
                            current_entry["blocks"].extend(pending_figures)
                            pending_figures = []
                        page_has_entry = True

                    if current_entry["contentMarkdown"]:
                        current_entry["contentMarkdown"] += "\n\n"
                    current_entry["contentMarkdown"] += b_text
                    current_entry["blocks"].append(block_dict)
                    continue

                # --- Non-text: figure/image handled with pending pattern; tables need host ---
                if b_type in ("image", "figure"):
                    uri = (b_meta.get("imageUri", "") if isinstance(b_meta, dict) else "")
                    asset_status = (
                        b_meta.get("assetStatus", "") if isinstance(b_meta, dict) else ""
                    ) or ("ready" if uri else "missing")
                    figure_dict = {
                        "id": b_id,
                        "type": "image",
                        "imageUri": uri,
                        "src": uri,
                        "pageNumber": p_num,
                        "semanticRole": "figure",
                        "searchable": False,  # figures never enter contentMarkdown
                        "assetSource": (b_meta.get("assetSource", "") if isinstance(b_meta, dict) else ""),
                        "assetStatus": asset_status,
                        "contentSha256": (b_meta.get("contentSha256", "") if isinstance(b_meta, dict) else ""),
                        "mimeType": (b_meta.get("mimeType", "") if isinstance(b_meta, dict) else ""),
                        "pixelWidth": int((b_meta.get("pixelWidth", 0) if isinstance(b_meta, dict) else 0) or 0),
                        "pixelHeight": int((b_meta.get("pixelHeight", 0) if isinstance(b_meta, dict) else 0) or 0),
                        "bbox": {
                            "left": int(b_bbox.x),
                            "top": int(b_bbox.y),
                            "right": int(b_bbox.x + b_bbox.w),
                            "bottom": int(b_bbox.y + b_bbox.h),
                            "width": int(b_bbox.w),
                            "height": int(b_bbox.h)
                        },
                    }
                    # Optional Phase 2E associations (caption/legend) — copy from IR
                    assocs = b_meta.get("figureTextAssociations") if isinstance(b_meta, dict) else None
                    if isinstance(assocs, list) and assocs:
                        figure_dict["figureTextAssociations"] = list(assocs)
                    # Phase 2G: trusted figure labels + reverse reference edges.
                    fig_labels = b_meta.get("figureLabels") if isinstance(b_meta, dict) else None
                    if isinstance(fig_labels, list) and fig_labels:
                        figure_dict["figureLabels"] = list(fig_labels)
                    ref_by = b_meta.get("referencedBy") if isinstance(b_meta, dict) else None
                    if isinstance(ref_by, list) and ref_by:
                        figure_dict["referencedBy"] = list(ref_by)
                    if page_has_entry and current_entry is not None and current_entry["pageNumber"] == p_num:
                        current_entry["blocks"].append(figure_dict)
                    else:
                        pending_figures.append(figure_dict)
                    continue

                # --- Tables / other visual need an entry host ---
                if current_entry is None or current_entry["pageNumber"] != p_num:
                    if current_entry:
                        entries.append(self._finalize_entry(current_entry))
                    current_entry = self._new_entry(b_id, f"Page {p_num}", p_num, entries)
                    current_entry["blocks"].extend(pending_supporting)
                    current_entry["blocks"].extend(pending_figures)
                    pending_supporting = []
                    pending_figures = []
                    page_has_entry = True
                elif not page_has_entry:
                    current_entry["blocks"].extend(pending_supporting)
                    current_entry["blocks"].extend(pending_figures)
                    pending_supporting = []
                    pending_figures = []
                    page_has_entry = True

                if b_type == "table":
                    cells = b_meta.get("cells", []) if isinstance(b_meta, dict) else []
                    structure_status = (
                        b_meta.get("structureStatus") if isinstance(b_meta, dict) else None
                    ) or ("structured" if cells else "image_only")
                    is_structured = structure_status == "structured" and bool(cells)

                    # Build rectangular grid + stable sorted physical cells
                    table_rows: List[List[str]] = []
                    cells_out: List[Dict[str, Any]] = []
                    meta_rows = 0
                    meta_cols = 0
                    if is_structured:
                        sorted_cells = sorted(
                            cells,
                            key=lambda c: (
                                int(c.get("row") or 0),
                                int(c.get("col") or 0),
                                int(c.get("colSpan") or 1),
                            ),
                        )
                        max_r = 0
                        max_c = 0
                        for c in sorted_cells:
                            r = int(c.get("row") or 0)
                            col = int(c.get("col") or 0)
                            rs = max(1, int(c.get("rowSpan") or 1))
                            cs = max(1, int(c.get("colSpan") or 1))
                            max_r = max(max_r, r + rs)
                            max_c = max(max_c, col + cs)
                        meta_rows = max_r
                        meta_cols = max_c
                        table_rows = [["" for _ in range(meta_cols)] for _ in range(meta_rows)]
                        for c in sorted_cells:
                            r = int(c.get("row") or 0)
                            col = int(c.get("col") or 0)
                            rs = max(1, int(c.get("rowSpan") or 1))
                            cs = max(1, int(c.get("colSpan") or 1))
                            text = str(c.get("text") or "")
                            if 0 <= r < meta_rows and 0 <= col < meta_cols:
                                # Anchor text only — spanned slots stay "" (no duplicate).
                                table_rows[r][col] = text
                            cell_out: Dict[str, Any] = {
                                "row": r,
                                "col": col,
                                "rowSpan": rs,
                                "colSpan": cs,
                                "text": text,
                                "bbox": c.get("bbox"),
                            }
                            cells_out.append(cell_out)

                    searchable_tbl = bool(is_structured)
                    md_table = ""
                    if is_structured and table_rows:
                        try:
                            from pipeline.extraction_adapters.table_adapter import (
                                table_rows_to_markdown,
                            )
                            md_table = table_rows_to_markdown(table_rows)
                        except Exception:
                            md_table = ""
                        if md_table:
                            if current_entry["contentMarkdown"]:
                                current_entry["contentMarkdown"] += "\n\n"
                            current_entry["contentMarkdown"] += md_table

                    block_out: Dict[str, Any] = {
                        "id": b_id,
                        "type": "table",
                        "table_rows": table_rows if is_structured else [],
                        "pageNumber": p_num,
                        "semanticRole": "table",
                        "structureStatus": structure_status if structure_status in (
                            "structured", "image_only"
                        ) else ("structured" if is_structured else "image_only"),
                        "searchable": searchable_tbl,
                        "bbox": {
                            "left": int(b_bbox.x),
                            "top": int(b_bbox.y),
                            "right": int(b_bbox.x + b_bbox.w),
                            "bottom": int(b_bbox.y + b_bbox.h),
                            "width": int(b_bbox.w),
                            "height": int(b_bbox.h)
                        },
                        "imageUri": (b_meta.get("imageUri", "") if isinstance(b_meta, dict) else ""),
                        "src": (b_meta.get("imageUri", "") if isinstance(b_meta, dict) else "")
                    }
                    if is_structured:
                        block_out["rows"] = int(meta_rows)
                        block_out["cols"] = int(meta_cols)
                        block_out["cells"] = cells_out
                    current_entry["blocks"].append(block_out)
                    continue

                # other non-text already handled above; fall through unreachable for known types

            # End of page:
            # - drop unused pending supporting text (no empty entry from callouts alone)
            # - figures: if NO host entry on this page and all pending figures are ready,
            #   emit a kind=figure entry (real ready assets only).
            if pending_figures and not page_has_entry:
                ready_figs = [
                    f for f in pending_figures
                    if f.get("assetStatus") == "ready" and f.get("imageUri")
                ]
                if ready_figs:
                    fig_entry = self._new_entry(
                        ready_figs[0]["id"], f"Page {p_num}", p_num, entries
                    )
                    fig_entry["kind"] = "figure"
                    fig_entry["blocks"] = list(ready_figs)
                    # contentMarkdown stays empty — figures are not searchable text
                    entries.append(self._finalize_entry(fig_entry))
                # missing-only pending figures: do not fabricate figure entry
            pending_supporting = []
            pending_figures = []

        if current_entry:
            entries.append(self._finalize_entry(current_entry))

        # Drop empty entries that somehow remain (defensive).
        entries = [e for e in entries if e.get("blocks") or (e.get("contentMarkdown") or "").strip()]

        # Build final metadata
        unique_images = set()
        for entry in entries:
            for b in entry["blocks"]:
                uri = b.get("imageUri")
                if uri:
                    unique_images.add(uri)

        metadata = {
            "schemaVersion": "2.0",
            "fileId": self.document_id,
            "fileName": "knowledge_base.json",
            "source": f"assets/kb/{self.document_id}",
            "importTimestamp": None,
            "entriesCount": len(entries),
            "imagesCount": len(unique_images),
            "docSha256": ir.sha256,
            "pageSizes": page_sizes,
            "coordinateUnit": "pt",
            "buildTool": "v2_runner.py"
        }

        return {
            "fileMetadata": metadata,
            "entries": entries
        }

    def _new_entry(
        self,
        block_id: str,
        job_title: str,
        page_number: int,
        existing: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "entryId": f"{self.document_id}_{block_id}",
            "unitName": "",
            "jobTitle": job_title,
            "contentMarkdown": "",
            "contentNormalized": "",
            "pageNumber": page_number,
            "position": len(existing) + 1,
            "kind": "text",
            "blocks": []
        }

    @staticmethod
    def _text_block_dict(
        b_id: str,
        b_text: str,
        p_num: int,
        role: Optional[str],
        b_bbox: BBox,
        *,
        searchable: bool,
    ) -> Dict[str, Any]:
        return {
            "id": b_id,
            "type": "code",
            "language": "markdown",
            "code": b_text,
            "pageNumber": p_num,
            "semanticRole": role if role is not None else "body",
            "searchable": searchable,
            "bbox": {
                "left": int(b_bbox.x),
                "top": int(b_bbox.y),
                "right": int(b_bbox.x + b_bbox.w),
                "bottom": int(b_bbox.y + b_bbox.h),
                "width": int(b_bbox.w),
                "height": int(b_bbox.h)
            }
        }

    def _finalize_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize text (searchable content only — supporting never enters contentMarkdown).
        entry["contentNormalized"] = entry["contentMarkdown"].strip()
        # In legacy, kind='table' if it has a table block
        has_table = any(b["type"] == "table" for b in entry["blocks"])
        if has_table:
            entry["kind"] = "table"
        return entry
