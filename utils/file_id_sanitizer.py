import unicodedata
import warnings
from typing import Optional


_DEPRECATION_WARNED = False


def sanitize_file_id_for_path_only(name: Optional[str], default: str = "doc") -> str:
    """Sanitize a file-id to a safe/stable folder name (PATH-ONLY).

    Unified rule (project-wide):
    - Keep: underscore `_`, Unicode letters, Unicode numbers (includes Chinese).
    - Drop: whitespace and all other special characters.

    Notes:
    - Output is lowercased for stable paths.
    - Result is safe for Windows folder names and Android assets paths.
    """

    s = (name or "").strip()
    out = []
    for ch in s:
        if ch.isspace():
            continue
        if ch == "_":
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat and cat[0] in ("L", "N"):
            out.append(ch)

    safe = "".join(out).lower()
    return safe or default


def sanitize_asset_relative_path(path: Optional[str], default: str = "doc") -> str:
    """Sanitize a slash-separated relative path for Android assets (PATH-ONLY).

    Use this when you want a hierarchical taxonomy layout like:
      "铁路/规章制度/电力/高速铁路电力管理规则2015"

    Rules:
    - Split by both '/' and '\\'.
    - Sanitize each segment with sanitize_file_id_for_path_only().
    - Drop empty segments and '.'/'..'.
    - Join segments with '/'.

    Returns:
    - A safe relative path (no leading/trailing '/'), or `default` if empty.
    """

    s = (path or "").strip().replace("\\", "/")
    if not s:
        return default

    segments = []
    for raw in s.split("/"):
        seg = (raw or "").strip()
        if not seg:
            continue
        if seg in (".", ".."):
            continue
        safe_seg = sanitize_file_id_for_path_only(seg, default="")
        if safe_seg:
            segments.append(safe_seg)

    return "/".join(segments) or default


def sanitize_file_id_for_folder(name: Optional[str], default: str = "doc") -> str:
    """Deprecated alias for sanitize_file_id_for_path_only()."""

    global _DEPRECATION_WARNED
    if not _DEPRECATION_WARNED:
        _DEPRECATION_WARNED = True
        warnings.warn(
            "sanitize_file_id_for_folder() is deprecated; use sanitize_file_id_for_path_only() for PATH-ONLY usage.",
            DeprecationWarning,
            stacklevel=2,
        )
    return sanitize_file_id_for_path_only(name=name, default=default)
