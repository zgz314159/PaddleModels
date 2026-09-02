"""DOCX 表格图片导出：Word COM PowerShell 导出、manifest 加载、截图查找"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.file_utils import sanitize_folder_name


def resolve_docx_table_export_script(script_dir: Path) -> Optional[Path]:
    """Locate the Word table export PowerShell script from common repo layouts."""

    direct_candidates = [
        script_dir / "export_docx_tables_to_pdfs.ps1",
        script_dir / "tools" / "kb_builder" / "export_docx_tables_to_pdfs.ps1",
        Path.cwd() / "export_docx_tables_to_pdfs.ps1",
        Path.cwd() / "tools" / "kb_builder" / "export_docx_tables_to_pdfs.ps1",
    ]
    for candidate in direct_candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except Exception:
            continue

    seen: set[Path] = set()
    for root in (script_dir, Path.cwd()):
        try:
            root_resolved = root.resolve()
        except Exception:
            continue
        for parent in [root_resolved, *list(root_resolved.parents)[:5]]:
            if parent in seen:
                continue
            seen.add(parent)
            candidate = parent / "tools" / "kb_builder" / "export_docx_tables_to_pdfs.ps1"
            try:
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()
            except Exception:
                continue

    return None


def try_export_docx_tables_via_word_powershell(
    docx_path: Path,
    out_dir: Path,
    dpi: int = 180,
) -> Dict[int, Path]:
    """Export each Word table as an image using Word via PowerShell (COM).

    Implementation:
      - PowerShell uses Word COM to grab `Table.Range.EnhMetaFileBits` (no clipboard)
        and renders that EMF into a PNG via .NET `System.Drawing`.

    This avoids any page screenshot cropping and preserves a faithful rendering.
    Requires Windows + Microsoft Word installed.
    """

    script_dir = Path(__file__).resolve().parent
    script_path_pdfs = resolve_docx_table_export_script(script_dir)
    if script_path_pdfs is None:
        print("[Tables] Word exporter script not found; DOCX table screenshots export is unavailable.")
        return {}
    if script_path_pdfs.parent != script_dir:
        print(f"[Tables] Using fallback Word exporter script: {script_path_pdfs}")

    # Word COM is very sensitive to relative paths (often resolving from system32).
    # Always pass absolute paths to the PowerShell exporter.
    docx_path = docx_path.resolve()
    out_dir = out_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    exported: Dict[int, Path] = {}

    with tempfile.TemporaryDirectory(prefix="powerai_word_tables_") as td:
        tmp_dir = Path(td)

        def _kill_pid_from_file(pid_file: Path) -> None:
            try:
                if not pid_file.exists():
                    return
                pid_text = pid_file.read_text(encoding="utf-8", errors="ignore").strip()
                pid = int(pid_text)
                subprocess.run(
                    ["cmd.exe", "/c", "taskkill", "/F", "/PID", str(pid)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                print(f"[Tables] Killed hung WINWORD pid={pid}.")
            except Exception:
                return

        def _tail_text_file(path: Path, max_chars: int = 4000) -> str:
            try:
                if not path.exists():
                    return ""
                s = path.read_text(encoding="utf-8", errors="ignore")
                if len(s) <= max_chars:
                    return s
                return s[-max_chars:]
            except Exception:
                return ""

        def _run_ps(
            args: List[str],
            timeout_seconds: int,
            tag: str,
        ) -> Optional[subprocess.CompletedProcess[str]]:
            pid_file = tmp_dir / f"word_{tag}.pid"
            log_file = tmp_dir / f"word_{tag}.log"
            cmd = [
                "powershell",
                "-STA",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path_pdfs),
                "-DocxPath",
                str(docx_path),
                "-OutDir",
                str(tmp_dir),
                "-PidFile",
                str(pid_file),
                "-LogPath",
                str(log_file),
                *args,
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
                if res.returncode != 0:
                    tail = _tail_text_file(log_file)
                    if tail:
                        print(f"[Tables] Word exporter log tail ({tag}):\n{tail}")
                return res
            except subprocess.TimeoutExpired:
                print(f"[Tables] Word exporter timed out ({tag}); attempting self-heal.")
                _kill_pid_from_file(pid_file)
                tail = _tail_text_file(log_file)
                if tail:
                    print(f"[Tables] Word exporter log tail ({tag}):\n{tail}")
                return None

        # 1) List tables count (separate short-lived Word instance).
        res = _run_ps(["-ListOnly"], timeout_seconds=60, tag="list")
        if res is None:
            print("[Tables] Word exporter timed out while listing tables.")
            return {}
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        if res.returncode != 0:
            return {}

        m_count = re.search(r"Tables\.Count=(\d+)", res.stdout or "")
        if not m_count:
            print("[Tables] Could not parse Tables.Count from Word exporter output.")
            return {}
        table_count = int(m_count.group(1))
        if table_count <= 0:
            return {}

        def _export_table_png(table_idx: int) -> Optional[Path]:
            png_path = tmp_dir / f"table_pos{table_idx}.png"
            if png_path.exists():
                try:
                    png_path.unlink()
                except Exception:
                    pass

            tag = f"t{table_idx}"
            res = _run_ps(
                [
                    "-TableIndex",
                    str(table_idx),
                    "-OutFormat",
                    "Png",
                    "-Dpi",
                    str(dpi),
                ],
                timeout_seconds=180,
                tag=tag,
            )

            if res is None:
                print(f"[Tables] Word exporter timed out on table {table_idx} (png).")
                return None

            if res.stdout:
                print(res.stdout.strip())
            if res.stderr:
                print(res.stderr.strip())

            if res.returncode != 0 or not png_path.exists():
                tail = _tail_text_file(tmp_dir / f"word_{tag}.log")
                if tail:
                    print(f"[Tables] Word exporter log tail ({tag}):\n{tail}")
                return None

            return png_path

        # 2) Export each table in its own process (reduces hang impact).
        for idx in range(1, table_count + 1):
            try:
                png_tmp = _export_table_png(idx)
                if png_tmp is None:
                    continue
                out_path = out_dir / png_tmp.name
                shutil.copy2(png_tmp, out_path)
                exported[idx] = out_path
            except Exception:
                continue
    return exported


def load_existing_exported_table_images(out_dir: Path) -> Dict[int, Path]:
    """Load already-exported table images from assets folder.

    Expected filenames:
      - table_posN.png
    """

    existing: Dict[int, Path] = {}
    try:
        if not out_dir.exists():
            return {}
        for p in out_dir.iterdir():
            if not p.is_file():
                continue
            m = re.match(r"^table_pos(\d+)\.(png|jpg|jpeg)$", p.name, flags=re.IGNORECASE)
            if not m:
                continue
            pos = int(m.group(1))
            existing[pos] = p
    except Exception:
        return {}
    return existing


def load_manifest_table_images(out_dir: Path) -> Tuple[Dict[int, List[Path]], List[Path]]:
    """Load page-keyed table screenshots from a sibling manifest.json if present.

    This supports reusing PDF-native exports such as table_p31_idx1.png when we
    skip Word export but still want table blocks to point at existing screenshots.
    """

    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return {}, []

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}, []

    page_map: Dict[int, List[Path]] = {}
    ordered: List[Path] = []
    for item in payload.get("items") or []:
        if not isinstance(item, dict):
            continue
        if (item.get("kind") or "").strip().lower() != "table":
            continue
        try:
            page_number = int(item.get("pageNumber") or 0)
        except Exception:
            continue
        out_file = (item.get("outFile") or "").strip()
        if page_number <= 0 or not out_file:
            continue
        candidate = out_dir / out_file
        if not candidate.exists() or not candidate.is_file():
            continue
        page_map.setdefault(page_number, []).append(candidate)
        ordered.append(candidate)

    return page_map, ordered


def find_original_table_screenshot_uri(
    original_screenshots_root: Path,
    file_id: str,
    page_number: int,
    position: int,
) -> Optional[str]:
    """Resolve user-provided original screenshots for a table.

    Expected layout:
            app/src/main/assets/kb/<file_id>/截图/table_p{page}_pos{position}.png

    Returns Android asset uri: file:///android_asset/kb/<file_id>/截图/<filename>
    """

    # PATH-ONLY: Use for directory/asset names.
    safe_file_id = sanitize_folder_name(file_id)
    base_dir = original_screenshots_root / safe_file_id / "截图"
    candidates = [
        f"table_p{page_number}_pos{position}.png",
        f"table_p{page_number}_pos{position}.jpg",
        f"table_p{page_number}_pos{position}.jpeg",
        # Optional simpler naming for manual workflow.
        f"table_pos{position}.png",
        f"table_pos{position}.jpg",
        f"table_pos{position}.jpeg",
    ]

    for name in candidates:
        p = base_dir / name
        if p.exists() and p.is_file():
            # Note: use forward slashes for Android assets URIs.
            return f"file:///android_asset/kb/{safe_file_id}/截图/{name}"
    return None
