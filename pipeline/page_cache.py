"""Page-level cache for the active v2 runner.

Single responsibility: turn a page's Canonical IR into a self-describing,
fingerprinted envelope on disk, and only return it as a hit when every
identity component matches. The legacy bare ``page_N.json`` format is never
read as a hit, and no file is ever deleted or overwritten by a miss.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.canonical_ir import BBox, DocBlock, DocPage

# Explicit on-disk cache contract version. Bump when the envelope layout
# changes so older files can never be mistaken for current ones.
PAGE_CACHE_SCHEMA_VERSION = 2

# Versioned namespace inside the run cache dir. The legacy flat ``page_N.json``
# files live one level up and are therefore naturally isolated, never read.
CACHE_NAMESPACE = "pages_v2"

# Keys that must be present and equal for a cache entry to be a hit. ``pageIR``
# is validated separately because it is the payload, not an identity field.
_IDENTITY_KEYS = (
    "cacheSchemaVersion",
    "fingerprint",
    "inputSha256",
    "pageNumber",
    "strategy",
)


def _canonical_json(payload: Any) -> str:
    """Deterministic JSON (sorted keys, no incidental whitespace, no hash())."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def compute_page_fingerprint(
    *,
    schema_version: int,
    profile_config: Dict[str, Any],
    router_config: Dict[str, Any],
    strategy: str,
) -> str:
    """SHA-256 over the stable inputs that determine a page's IR.

    Deliberately excludes run_id, timestamps, output/temp paths and the page
    number so identical input + identical effective config reuse a hit, while
    any change to schema, profile IR config, router threshold or strategy
    produces a different identity.
    """
    payload = {
        "cacheSchemaVersion": schema_version,
        "profileConfig": profile_config,
        "routerConfig": router_config,
        "strategy": strategy,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def page_ir_to_dict(page: DocPage) -> Dict[str, Any]:
    """Serialize a DocPage to its plain-dict Canonical IR form."""
    return page.to_dict()


def page_ir_from_dict(page_ir_dict: Dict[str, Any]) -> DocPage:
    """Reconstruct a DocPage from its plain-dict Canonical IR form."""
    blocks = []
    for b in page_ir_dict["blocks"]:
        bb = b["bbox"]
        blocks.append(
            DocBlock(
                id=b["id"],
                type=b["type"],
                text=b["text"],
                bbox=BBox(bb["x"], bb["y"], bb["w"], bb["h"]),
                page_number=b["page_number"],
                reading_order=b["reading_order"],
                confidence=b.get("confidence", 1.0),
                source=b.get("source", "unknown"),
                metadata=b.get("metadata", {}),
            )
        )
    return DocPage(
        page_number=page_ir_dict["page_number"],
        width=page_ir_dict["width"],
        height=page_ir_dict["height"],
        blocks=blocks,
        rotation=page_ir_dict.get("rotation", 0),
        method=page_ir_dict.get("method", "unknown"),
    )


def _warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


class PageCache:
    """Read/write the fingerprinted page cache for one run configuration."""

    def __init__(
        self,
        cache_dir: Any,
        *,
        input_sha256: str,
        profile_config: Dict[str, Any],
        router_config: Dict[str, Any],
        schema_version: int = PAGE_CACHE_SCHEMA_VERSION,
        namespace: str = CACHE_NAMESPACE,
    ):
        self.cache_dir = Path(cache_dir)
        self.input_sha256 = str(input_sha256)
        self.profile_config = dict(profile_config or {})
        self.router_config = dict(router_config or {})
        self.schema_version = schema_version
        self.namespace = namespace
        self._fingerprints: Dict[str, str] = {}

    @property
    def namespace_dir(self) -> Path:
        return self.cache_dir / self.namespace

    def fingerprint_for(self, strategy: str) -> str:
        fp = self._fingerprints.get(strategy)
        if fp is None:
            fp = compute_page_fingerprint(
                schema_version=self.schema_version,
                profile_config=self.profile_config,
                router_config=self.router_config,
                strategy=strategy,
            )
            self._fingerprints[strategy] = fp
        return fp

    def path_for(self, page_number: int, strategy: str) -> Path:
        return self.namespace_dir / f"page_{int(page_number)}.{strategy}.json"

    def load(self, page_number: int, strategy: str) -> Optional[DocPage]:
        """Return the cached DocPage, or None on any mismatch/corruption."""
        path = self.path_for(page_number, strategy)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                envelope = json.load(f)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            _warn(f"page {page_number}: unreadable page cache {path.name}: {exc}")
            return None

        reason = self._identity_miss_reason(envelope, page_number, strategy)
        if reason is not None:
            _warn(f"page {page_number}: ignoring page cache {path.name}: {reason}")
            return None

        try:
            page = page_ir_from_dict(envelope["pageIR"])
        except Exception as exc:
            _warn(f"page {page_number}: malformed page IR in {path.name}: {exc}")
            return None
        if page.page_number != int(page_number):
            _warn(f"page {page_number}: page IR page number mismatch in {path.name}")
            return None
        return page

    def _identity_miss_reason(
        self, envelope: Any, page_number: int, strategy: str
    ) -> Optional[str]:
        if not isinstance(envelope, dict):
            return "not a metadata envelope"
        for key in _IDENTITY_KEYS:
            if key not in envelope:
                return f"missing {key}"
        if "pageIR" not in envelope:
            return "missing pageIR"
        if envelope["cacheSchemaVersion"] != self.schema_version:
            return "schema version mismatch"
        if envelope["fingerprint"] != self.fingerprint_for(strategy):
            return "fingerprint mismatch"
        if envelope["inputSha256"] != self.input_sha256:
            return "input sha mismatch"
        if envelope["pageNumber"] != int(page_number):
            return "page number mismatch"
        if envelope["strategy"] != strategy:
            return "strategy mismatch"
        return None

    def store(self, page_number: int, strategy: str, page_ir: DocPage) -> Path:
        """Atomically write the envelope so no half-written file is ever read."""
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(page_number, strategy)
        envelope = {
            "cacheSchemaVersion": self.schema_version,
            "fingerprint": self.fingerprint_for(strategy),
            "inputSha256": self.input_sha256,
            "pageNumber": int(page_number),
            "strategy": strategy,
            "pageIR": page_ir_to_dict(page_ir),
        }
        payload = (
            json.dumps(envelope, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.namespace_dir),
            prefix=".pctmp-",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, str(path))
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        return path
