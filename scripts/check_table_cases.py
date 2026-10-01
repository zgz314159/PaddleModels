"""Lightweight table fixture detect-only regression (no algorithm tuning)."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import fitz  # noqa: E402
from imaging.table_processor import extract_native_table_cells  # noqa: E402


def main() -> int:
    cases = json.loads((REPO / "tests" / "fixtures" / "table_cases.json").read_text(encoding="utf-8"))
    doc = fitz.open(str(REPO / cases["pdf"]))
    results = []
    all_ok = True
    for c in cases["cases"]:
        page = doc.load_page(c["page"] - 1)
        tabs = list(page.find_tables().tables)
        structured = 0
        detail = []
        for t in tabs:
            try:
                cells = extract_native_table_cells(t) or []
            except Exception:
                cells = []
            if cells:
                structured += 1
            detail.append(
                {
                    "rows": getattr(t, "row_count", None),
                    "cols": getattr(t, "col_count", None),
                    "cells": len(cells),
                }
            )
        ok = len(tabs) >= c["expected_min_tables"]
        struct_ok = (not c.get("require_structured")) or structured >= 1
        page_ok = ok and struct_ok
        all_ok = all_ok and page_ok
        results.append(
            {
                "page": c["page"],
                "role": c["role"],
                "found": len(tabs),
                "min": c["expected_min_tables"],
                "structured": structured,
                "ok": page_ok,
                "detail": detail,
            }
        )
        print(
            f"page {c['page']} {c['role']}: found={len(tabs)} "
            f"structured={structured} ok={page_ok}"
        )
    doc.close()
    # Do not write into tests/fixtures (avoid runtime artifacts in test tree).
    out_dir = REPO / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "table_cases_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("ALL_OK" if all_ok else "SOME_FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
