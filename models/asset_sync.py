"""资源同步：增量写入 Android assets KB（知识库 JSON）"""

import json
from pathlib import Path
from typing import Dict

from models.entry_model import Entry, atomic_write_json, entry_to_payload_dict


class AssetsKnowledgeBaseSync:
    def __init__(
        self,
        *,
        assets_kb_path: Path,
        file_id: str,
        file_name: str,
        source: str,
    ) -> None:
        self._path = assets_kb_path
        self._file_id = file_id
        self._file_name = file_name
        self._source = source
        self._payload: Dict[str, object] = {
            "fileMetadata": {
                "fileId": file_id,
                "fileName": file_name,
                "source": source,
                "importTimestamp": None,
                "entriesCount": 0,
                "imagesCount": 0,
                "docSha256": "",
            },
            "entries": [],
        }
        self._entries_by_entry_id: Dict[str, Dict[str, object]] = {}
        self._entries_by_position: Dict[int, Dict[str, object]] = {}
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            return
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(loaded, dict):
                self._payload = loaded
        except Exception as ex:
            print(f"[AssetsSync] WARNING: failed to read assets KB; will overwrite with new payload: {ex}")
            return

        items = self._payload.get("entries")
        if not isinstance(items, list):
            self._payload["entries"] = []
            items = self._payload["entries"]

        self._entries_by_position = {}
        self._entries_by_entry_id = {}
        for item in items:
            if not isinstance(item, dict):
                continue

            entry_id = str(item.get("entryId") or "").strip()
            if entry_id:
                self._entries_by_entry_id[entry_id] = item
            try:
                pos = int(item.get("position") or 0)
            except Exception:
                pos = 0
            if pos > 0:
                self._entries_by_position[pos] = item

    def sync_entry(self, *, entry: Entry, images_total: int, doc_sha256: str) -> None:
        pos = int(entry.position or 0)
        if pos <= 0:
            return

        entry_id = str(entry.entry_id or "").strip()
        existing = self._entries_by_entry_id.get(entry_id) if entry_id else None
        if existing is None:
            existing = self._entries_by_position.get(pos)
        if existing is None:
            new_item = entry_to_payload_dict(file_id=self._file_id, e=entry)
            items = self._payload.get("entries")
            if not isinstance(items, list):
                items = []
                self._payload["entries"] = items
            items.append(new_item)
            if entry_id:
                self._entries_by_entry_id[entry_id] = new_item
            self._entries_by_position[pos] = new_item
            existing = new_item

        if entry_id:
            existing["entryId"] = entry_id
        existing["contentMarkdown"] = entry.content_markdown
        existing["contentNormalized"] = entry.content_normalized
        existing["unitName"] = entry.unit_name
        existing["jobTitle"] = entry.job_title
        existing["pageNumber"] = int(entry.page_number or 1)
        existing["kind"] = entry.kind

        rebuilt = entry_to_payload_dict(file_id=self._file_id, e=entry)
        existing["blocks"] = rebuilt.get("blocks") or []

        fm = self._payload.get("fileMetadata")
        if not isinstance(fm, dict):
            fm = {}
            self._payload["fileMetadata"] = fm
        fm["fileId"] = self._file_id
        fm["fileName"] = self._file_name
        fm["source"] = self._source
        if doc_sha256:
            fm["docSha256"] = doc_sha256
        items2 = self._payload.get("entries")
        if isinstance(items2, list):
            fm["entriesCount"] = len(items2)
        try:
            fm["imagesCount"] = max(int(fm.get("imagesCount") or 0), int(images_total or 0))
        except Exception:
            pass

        atomic_write_json(self._path, self._payload)
        print(f"[AssetsSync] Updated assets KB entry position={pos} -> {self._path}")
