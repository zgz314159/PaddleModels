import json
import hashlib
from pathlib import Path
from typing import Any, Optional

class PageCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, input_sha256: str, page_number: int, method: str) -> str:
        key_content = f"{input_sha256}_{page_number}_{method}"
        return hashlib.sha256(key_content.encode()).hexdigest()

    def get(self, input_sha256: str, page_number: int, method: str) -> Optional[Any]:
        key = self._get_key(input_sha256, page_number, method)
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, input_sha256: str, page_number: int, method: str, data: Any):
        key = self._get_key(input_sha256, page_number, method)
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
