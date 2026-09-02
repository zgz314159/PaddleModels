"""Entry 数据模型与序列化"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from classification.text_classifier import infer_docx_text_semantic_role


@dataclass
class Entry:
    entry_id: str
    unit_name: str
    job_title: str
    content_markdown: str
    content_normalized: str
    page_number: int
    position: int
    kind: str  # 'text' | 'table'
    table_position: int
    table_rows: List[List[str]]
    image_uris: List[str]
    table_image_uri: Optional[str] = None


def stable_block_id(seed: str) -> str:
    h = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"b_{h}"


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(path))


def should_emit_markdown_code_block(e: Entry) -> bool:
    return (e.kind or "").strip().lower() != "table"


def entry_to_payload_dict(*, file_id: str, e: Entry) -> Dict[str, object]:
    blocks: List[Dict[str, object]] = []

    if should_emit_markdown_code_block(e):
        text_role = infer_docx_text_semantic_role(e.content_markdown)
        blocks.append(
            {
                "id": stable_block_id(f"{file_id}|{e.entry_id}|code|markdown"),
                "type": "code",
                "language": "markdown",
                "code": e.content_markdown,
                "pageNumber": e.page_number,
                "semanticRole": text_role,
            }
        )

    if (e.kind or "").strip().lower() == "table":
        blocks.append(
            {
                "id": stable_block_id(f"{file_id}|{e.entry_id}|table|{e.job_title}"),
                "type": "table",
                "rows": e.table_rows,
                "pageNumber": e.page_number,
                "semanticRole": "table",
                **({"imageUri": e.table_image_uri} if e.table_image_uri else {}),
                **({"tablePosition": int(e.table_position or 0)} if int(e.table_position or 0) > 0 else {}),
            }
        )

    blocks.extend(
        [
            {
                "id": stable_block_id(f"{file_id}|{e.entry_id}|image|{uri}|{idx}"),
                "type": "image",
                "src": uri,
                "imageUri": uri,
                "pageNumber": e.page_number,
                "semanticRole": "figure",
            }
            for idx, uri in enumerate(e.image_uris)
        ]
    )

    return {
        "entryId": e.entry_id,
        "unitName": e.unit_name,
        "jobTitle": e.job_title,
        "contentMarkdown": e.content_markdown,
        "contentNormalized": e.content_normalized,
        "pageNumber": e.page_number,
        "position": e.position,
        "kind": e.kind,
        "blocks": blocks,
    }


def build_output_payload(
    *,
    file_id: str,
    file_name: str,
    source: str,
    doc_sha256: str,
    images_total: int,
    entries: List[Entry],
) -> Dict[str, object]:
    return {
        "fileMetadata": {
            "fileId": file_id,
            "fileName": file_name,
            "source": source,
            "importTimestamp": None,
            "entriesCount": len(entries),
            "imagesCount": images_total,
            "docSha256": doc_sha256,
        },
        "entries": [entry_to_payload_dict(file_id=file_id, e=e) for e in entries],
    }


def load_entries_from_partial_payload(partial_payload: Dict[str, object]) -> Tuple[List[Entry], int, str]:
    try:
        fm = partial_payload.get("fileMetadata") or {}
        file_id = str((fm or {}).get("fileId") or "").strip() or "knowledge_base"
        images_total = int((fm or {}).get("imagesCount") or 0)
        doc_sha256 = str((fm or {}).get("docSha256") or "")
        items = partial_payload.get("entries") or []
        if not isinstance(items, list):
            return [], 0, doc_sha256

        loaded: List[Entry] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            position = int(item.get("position") or 0)
            entry_id = str(item.get("entryId") or "").strip() or f"{file_id}::table#{position}"
            unit_name = str(item.get("unitName") or "")
            job_title = str(item.get("jobTitle") or "")
            content_markdown = str(item.get("contentMarkdown") or "")
            content_normalized = str(item.get("contentNormalized") or "")

            kind = str(item.get("kind") or "table").strip() or "table"
            table_position = int(item.get("tablePosition") or 0)

            table_rows: List[List[str]] = []
            image_uris: List[str] = []
            page_number = int(item.get("pageNumber") or 1)
            table_image_uri: Optional[str] = None

            blocks = item.get("blocks") or []
            if isinstance(blocks, list):
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    btype = b.get("type")
                    if btype == "table":
                        rows = b.get("rows")
                        if isinstance(rows, list):
                            # Best-effort; tolerate mixed types.
                            table_rows = [
                                [str(c or "") for c in (r if isinstance(r, list) else [])]
                                for r in rows
                                if isinstance(r, list)
                            ]
                        if b.get("imageUri"):
                            table_image_uri = str(b.get("imageUri"))
                        if b.get("pageNumber"):
                            try:
                                page_number = int(b.get("pageNumber") or 1)
                            except Exception:
                                pass
                    elif btype == "image":
                        uri = b.get("imageUri") or b.get("src")
                        if uri:
                            image_uris.append(str(uri))
                    elif btype == "code":
                        if b.get("pageNumber"):
                            try:
                                page_number = int(b.get("pageNumber") or 1)
                            except Exception:
                                pass

            loaded.append(
                Entry(
                    entry_id=entry_id,
                    unit_name=unit_name,
                    job_title=job_title,
                    content_markdown=content_markdown,
                    content_normalized=content_normalized,
                    page_number=page_number,
                    position=position,
                    kind=kind,
                    table_position=table_position,
                    table_rows=table_rows,
                    image_uris=image_uris,
                    table_image_uri=table_image_uri,
                )
            )

        # Keep deterministic order.
        loaded.sort(key=lambda e: e.position)
        return loaded, images_total, doc_sha256
    except Exception:
        return [], 0, ""
