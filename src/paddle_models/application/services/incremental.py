import hashlib
from pathlib import Path
from typing import Optional, Any
from paddle_models.core.cache import PageCache

class IncrementalService:
    """
    Manages incremental processing by checking page-level cache and file hashes.
    """
    def __init__(self, cache: PageCache):
        self.cache = cache

    def should_process_page(self, file_sha256: str, page_number: int, method: str) -> bool:
        """
        Returns True if the page needs processing, False if it can be skipped.
        """
        cached_data = self.cache.get(file_sha256, page_number, method)
        return cached_data is None

    def get_cached_page_data(self, file_sha256: str, page_number: int, method: str) -> Optional[Any]:
        return self.cache.get(file_sha256, page_number, method)
