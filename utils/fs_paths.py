"""Filesystem path helpers shared by the cache code.

On Windows the legacy ``MAX_PATH`` limit (260 chars) makes long paths raise
``FileNotFoundError`` / ``OSError`` (WinError 3 / 206) for otherwise valid
locations. Prefixing an absolute path with the extended-length marker
``\\\\?\\`` lifts that limit for the Win32 APIs Python uses (``open``,
``os.mkdir``/``makedirs``, ``os.replace``, ``os.stat`` ...).

``fs_path`` is a no-op on non-Windows platforms and is deliberately applied to
every filesystem call that touches the page cache, so directory creation, reads,
temp-file writes and the atomic replace all address the same path shape.
"""
from __future__ import annotations

import os

_EXTENDED_PREFIX = "\\\\?\\"


def fs_path(path) -> str:
    """Return an OS path string that is safe for long paths.

    On Windows an absolute path is returned with the ``\\\\?\\`` extended-length
    prefix (UNC shares use ``\\\\?\\UNC\\``); already-prefixed paths are returned
    unchanged. On other platforms the path is returned as-is.
    """
    raw = os.fspath(path)
    if os.name != "nt":
        return raw
    if raw.startswith(_EXTENDED_PREFIX):
        return raw
    absolute = os.path.abspath(raw)
    if absolute.startswith(_EXTENDED_PREFIX):
        return absolute
    if absolute.startswith("\\\\"):
        return _EXTENDED_PREFIX + "UNC\\" + absolute[2:]
    return _EXTENDED_PREFIX + absolute
