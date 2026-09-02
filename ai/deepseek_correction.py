"""DeepSeek 纠错：可用性检查、语义修复、审计 CSV"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import deepseek_client
except Exception:
    deepseek_client = None

DOCX_CORRECTION_FIELDNAMES = [
    "page_number",
    "context_kind",
    "style_name",
    "entry_id",
    "unit_name",
    "job_title",
    "suspicion_score",
    "suspicion_reasons",
    "status",
    "rejected_reason",
    "original_text",
    "fixed_text",
    "llm_reason",
    "confidence",
]
_DOCX_IMAGE_REF_RE = re.compile(r"\[IMAGE_REF:[^\]]+\]")


def deepseek_correction_available() -> bool:
    if deepseek_client is None:
        return False
    return bool(getattr(deepseek_client, "DEEPSEEK_API_KEY", "")) and bool(
        getattr(deepseek_client, "DEEPSEEK_API_URL", "")
    )


def normalize_docx_semantic_fix(*, original: str, fixed: str, context_kind: str) -> Tuple[str, str]:
    candidate = (fixed or "").replace("```", "").strip()
    if not candidate:
        return original, "empty_result"
    if abs(len(candidate) - len(original)) > max(18, int(len(original) * 0.6)):
        return original, "length_delta_too_large"
    if context_kind != "paragraph" and candidate.count("\n") > original.count("\n") + 2:
        return original, "unexpected_multiline"
    if _DOCX_IMAGE_REF_RE.findall(original) != _DOCX_IMAGE_REF_RE.findall(candidate):
        return original, "image_placeholder_changed"
    from classification.text_classifier import detect_anchor
    original_anchor = detect_anchor(original)
    if original_anchor is not None and detect_anchor(candidate) is None:
        return original, "anchor_lost"
    return candidate, ""


def write_docx_correction_audit(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOCX_CORRECTION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in DOCX_CORRECTION_FIELDNAMES})
