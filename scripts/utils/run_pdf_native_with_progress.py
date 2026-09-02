#!/usr/bin/env python3
"""Run pdf_to_base64_kb.py with real-time page progress and log tee.

This wrapper keeps detailed extractor output in a log file while printing only
page-level progress to stdout for CMD users.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PAGE_PROGRESS_RE = re.compile(
    r"\[PAGE_PROGRESS\]\s*current=(\d+)\s+total=(\d+)(?:\s+status=([A-Za-z_]+))?",
    re.IGNORECASE,
)


def _detect_total_pages(pdf_path: Path) -> int:
    try:
        import fitz  # type: ignore

        doc = fitz.open(str(pdf_path))
        count = int(doc.page_count)
        doc.close()
        return count
    except Exception:
        pass

    try:
        import pypdfium2 as pdfium  # type: ignore

        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            return int(len(doc))
        finally:
            try:
                doc.close()
            except Exception:
                pass
    except Exception:
        return 0


def _progress_line(current: int, total: int, status: str) -> str:
    width = 30
    if total <= 0:
        total = 1
    pct = int((current * 100) / total)
    if pct < 0:
        pct = 0
    if pct > 100:
        pct = 100
    filled = int((current * width) / total)
    if filled < 0:
        filled = 0
    if filled > width:
        filled = width
    bar = ("#" * filled) + ("-" * (width - filled))
    suffix = f" status={status}" if status else ""
    return f"[PAGE_PROGRESS] [{bar}] {pct}% ({current}/{total}){suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run native PDF extractor with page progress")
    parser.add_argument("--python-exe", required=True, help="Python executable for target script")
    parser.add_argument("--script", required=True, help="Target script path (pdf_to_base64_kb.py)")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--out", required=True, help="Output KB JSON path")
    parser.add_argument("--file-id", required=True, help="fileId for KB")
    parser.add_argument("--assets-root", required=True, help="Assets root path")
    parser.add_argument("--log", required=True, help="Log file path to append detailed output")
    parser.add_argument("--target-bytes", type=int, default=1363148, help="Target bytes for image payload")
    args, passthrough = parser.parse_known_args()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_pages = _detect_total_pages(Path(args.pdf))
    current_page = 0
    last_status = ""

    if total_pages > 0:
        print(_progress_line(0, total_pages, "start"), flush=True)

    cmd = [
        args.python_exe,
        args.script,
        "--pdf",
        args.pdf,
        "--out",
        args.out,
        "--file-id",
        args.file_id,
        "--assets-root",
        args.assets_root,
        "--target-bytes",
        str(args.target_bytes),
    ]
    if passthrough:
        cmd.extend(passthrough)

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[WRAP] run_pdf_native_with_progress cmd={' '.join(cmd)}\n")
        lf.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()

            m = PAGE_PROGRESS_RE.search(line)
            if m:
                cur = int(m.group(1))
                tot = int(m.group(2))
                status = (m.group(3) or "").strip()
                if tot > 0:
                    total_pages = tot
                if cur != current_page or status != last_status:
                    current_page = cur
                    last_status = status
                    print(_progress_line(current_page, total_pages if total_pages > 0 else cur, status), flush=True)
                continue

        rc = proc.wait()

    if rc != 0:
        print(f"[PAGE_PROGRESS] extractor failed, exitCode={rc}", flush=True)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())