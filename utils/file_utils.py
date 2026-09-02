"""文件 / 校验 / 属性工具"""

import hashlib
import zipfile
from pathlib import Path
from typing import Dict

from utils.file_id_sanitizer import sanitize_asset_relative_path


def validate_docx_container(docx_path: Path) -> None:
    suffix = docx_path.suffix.lower()
    if suffix == ".doc":
        raise SystemExit(
            f"DOCX pipeline does not support legacy .doc files: {docx_path}. "
            "Please convert the document to .docx and rerun."
        )
    if not zipfile.is_zipfile(docx_path):
        raise SystemExit(
            f"DOCX pipeline requires an OOXML .docx/.docm package, but the input is not a zip-based Word file: {docx_path}"
        )
    try:
        with zipfile.ZipFile(docx_path, "r") as docx_zip:
            if "word/document.xml" not in docx_zip.namelist():
                raise SystemExit(
                    f"Invalid DOCX package: missing word/document.xml in {docx_path}. "
                    "Please confirm the file is a normal .docx/.docm exported by Word or convert it again before rerunning."
                )
    except zipfile.BadZipFile as exc:
        raise SystemExit(
            f"DOCX pipeline requires an OOXML .docx/.docm package, but the input archive is unreadable: {docx_path}"
        ) from exc


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_local_properties(project_root: Path) -> Dict[str, str]:
    """Best-effort loader for local.properties (Android Studio).

    This is only to reduce friction when running this script locally. We still
    prefer environment variables when present.
    """

    props_path = project_root / "local.properties"
    if not props_path.exists():
        return {}
    out: Dict[str, str] = {}
    try:
        for line in props_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip()
            if k and v:
                out[k] = v
    except Exception:
        return {}
    return out


def sanitize_folder_name(name: str) -> str:
    # Stable docKey for assets paths.
    # PATH-ONLY: Use for directory/asset names.
    # Allow hierarchical taxonomy paths like "铁路/规章制度/电力/...".
    return sanitize_asset_relative_path(name)
