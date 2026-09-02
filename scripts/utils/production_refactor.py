"""通用型 RAG 知识库修复工具

此文件为 production_refactor.py 的副本，已移到通用脚本目录以便复用。
"""

#!/usr/bin/env python3
import os
import sys
import json
import csv
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

# Ensure repo root is on sys.path so imports like `pdf_table_structure` work
# when this script is executed from scripts/utils directory.
try:
    _here = Path(__file__).resolve()
    # scripts/utils/production_refactor.py -> repo root is parents[2]
    repo_root = _here.parents[2]
    sys.path.insert(0, str(repo_root))
except Exception:
    pass

from pdf_table_structure import process_image
import paddle_ocr_util
import deepseek_client
from classification.text_quality_gate import analyze_text_suspicion, should_call_llm_correction

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None


_WARNED_NO_SKLEARN = False
_KB_INDEX_LOCK = threading.Lock()
_KB_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
_CORRECTION_CACHE_LOCK = threading.Lock()
_CORRECTION_CACHE: Dict[str, Dict[str, Any]] = {}
_SEMANTIC_ROLE_WEIGHTS: Dict[str, int] = {
    "heading": 3,
    "body": 2,
    "table": 2,
    "table_note": 1,
    "caption": 1,
    "artifact": 0,
    "figure": 0,
}


def _normalize_text_value(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(x) for x in value if str(x or "").strip())
    return str(value or "")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _flatten_table_rows(rows: Any, max_rows: int = 8, max_cells: int = 8) -> str:
    if not isinstance(rows, list):
        return ""
    parts: List[str] = []
    for row in rows[:max_rows]:
        if isinstance(row, list):
            cell_texts = [_normalize_ws(cell) for cell in row[:max_cells]]
            row_text = " | ".join([cell for cell in cell_texts if cell])
        else:
            row_text = _normalize_ws(row)
        if row_text:
            parts.append(row_text)
    return "\n".join(parts)


def _extract_block_text(block: Dict[str, Any]) -> str:
    block_type = str(block.get("type") or "").strip().lower()
    if block_type == "code":
        return _normalize_text_value(block.get("code") or block.get("text"))
    if block_type == "table":
        table_text = _normalize_text_value(block.get("markdown") or block.get("text"))
        if table_text.strip():
            return table_text
        return _flatten_table_rows(block.get("rows"))
    if block_type == "image":
        return _normalize_text_value(block.get("caption") or block.get("text"))
    return _normalize_text_value(block.get("text") or block.get("caption") or block.get("code"))


def _looks_like_body_fallback(text: str) -> bool:
    s = _normalize_ws(text)
    if not s:
        return False
    if len(s) >= 40:
        return True
    if re.search(r"[（(]\d+[)）]", s):
        return True
    if len(re.findall(r"[。；：:;]", s)) >= 2:
        return True
    if len(re.findall(r"[\u4e00-\u9fff]", s)) >= 20:
        return True
    return False


def _extract_entry_semantic_texts(entry: Dict[str, Any]) -> Dict[str, List[str]]:
    role_texts: Dict[str, List[str]] = {}

    def add_text(role: str, text: Any) -> None:
        clean = _normalize_ws(_normalize_text_value(text))
        if not clean:
            return
        bucket = role_texts.setdefault(role or "body", [])
        if clean not in bucket:
            bucket.append(clean)

    title = _normalize_ws(entry.get("jobTitle") or entry.get("title") or entry.get("heading") or "")
    if title:
        add_text("heading", title)
    unit_name = _normalize_ws(entry.get("unitName") or "")
    if unit_name and unit_name != title:
        add_text("heading", unit_name)

    blocks = entry.get("blocks")
    if isinstance(blocks, list) and blocks:
        for block in blocks:
            if not isinstance(block, dict):
                continue
            role = str(block.get("semanticRole") or "").strip().lower()
            if not role:
                block_type = str(block.get("type") or "").strip().lower()
                if block_type == "table":
                    role = "table"
                elif block_type == "image":
                    role = "figure"
                else:
                    role = "body"
            add_text(role, _extract_block_text(block))

    if not role_texts:
        fallback = entry.get("contentNormalized") or entry.get("contentMarkdown") or entry.get("content") or ""
        add_text("body", fallback)
    else:
        fallback = entry.get("contentNormalized") or entry.get("contentMarkdown") or entry.get("content") or ""
        low_priority_only = all(role in {"caption", "artifact", "figure"} for role in role_texts)
        if low_priority_only and _looks_like_body_fallback(_normalize_text_value(fallback)):
            add_text("body", fallback)

    return role_texts


def _build_entry_index_text(entry: Dict[str, Any]) -> Tuple[str, str]:
    role_texts = _extract_entry_semantic_texts(entry)
    weighted_parts: List[str] = []
    context_parts: List[str] = []
    positive_roles: List[str] = []

    for role in ("heading", "body", "table", "table_note", "caption", "artifact", "figure"):
        texts = role_texts.get(role) or []
        if not texts:
            continue
        joined = "\n".join(texts)
        if role != "artifact":
            context_parts.append(joined)
        weight = _SEMANTIC_ROLE_WEIGHTS.get(role, 1)
        if weight > 0:
            positive_roles.append(role)
            weighted_parts.extend([joined] * weight)

    if not weighted_parts:
        raw_fallback = _normalize_ws(
            entry.get("contentNormalized") or entry.get("contentMarkdown") or entry.get("content") or ""
        )
        if raw_fallback:
            weighted_parts.append(raw_fallback)
            if raw_fallback not in context_parts:
                context_parts.append(raw_fallback)

    if not positive_roles and role_texts.get("caption"):
        caption_text = "\n".join(role_texts["caption"])
        weighted_parts.append(caption_text)
        if caption_text not in context_parts:
            context_parts.append(caption_text)

    ranking_text = "\n\n".join(part for part in weighted_parts if part).strip()
    context_text = "\n\n".join(part for part in context_parts if part).strip()
    if not context_text:
        context_text = ranking_text
    return ranking_text, context_text


def _is_effective_candidate_text(text: str) -> bool:
    """Return True when the cell text is worth semantic correction.

    Filters out placeholders/noise so DeepSeek budget focuses on useful OCR text.
    """
    s = (text or "").strip()
    if not s:
        return False
    if s in ("[]", "{}", "null", "None", "nan", "N/A"):
        return False
    # Extremely short one-char fragments are usually grid noise.
    if len(s) <= 1:
        return False
    # Keep CJK-rich content.
    if re.search(r"[\u4e00-\u9fff]", s):
        return True
    # Keep alpha-numeric tokens only if they are long enough and not punctuation-heavy.
    alnum = re.sub(r"[^0-9A-Za-z]", "", s)
    if len(alnum) < 3:
        return False
    punct_count = len(re.findall(r"[^\w\u4e00-\u9fff]", s, flags=re.UNICODE))
    if punct_count > max(2, len(s) // 2):
        return False
    return True


def _should_send_to_deepseek(text: str) -> Tuple[bool, str, float]:
    report = analyze_text_suspicion(text)
    if should_call_llm_correction(text, relaxed_for_short_text=True):
        reasons = ",".join(report.reasons) or "suspicious_pattern"
        return True, reasons, float(report.score)
    return False, ",".join(report.reasons), float(report.score)


def _load_kb_texts(kb_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    p = Path(kb_path)
    if not p.exists():
        return [], []
    with p.open("r", encoding="utf-8") as f:
        kb = json.load(f)
    entries = kb if isinstance(kb, list) else kb.get("entries") or kb.get("items") or []
    texts = []
    for e in entries:
        txt = e.get("contentNormalized") or e.get("contentMarkdown") or e.get("content") or ""
        texts.append(_normalize_text_value(txt))
    return texts, entries


def _load_kb_index_texts(kb_path: str) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
    p = Path(kb_path)
    if not p.exists():
        return [], [], []
    with p.open("r", encoding="utf-8") as f:
        kb = json.load(f)
    entries = kb if isinstance(kb, list) else kb.get("entries") or kb.get("items") or []
    ranking_texts: List[str] = []
    context_texts: List[str] = []
    for e in entries:
        rank_text, context_text = _build_entry_index_text(e)
        ranking_texts.append(rank_text)
        context_texts.append(context_text)
    return ranking_texts, entries, context_texts


def _get_kb_index(kb_path: str) -> Dict[str, Any]:
    """Build and cache KB retrieval index once per kb_path."""
    with _KB_INDEX_LOCK:
        cached = _KB_INDEX_CACHE.get(kb_path)
        if cached is not None:
            return cached

        texts, entries, context_texts = _load_kb_index_texts(kb_path)
        idx: Dict[str, Any] = {
            "texts": texts,
            "entries": entries,
            "context_texts": context_texts,
            "vectorizer": None,
            "matrix": None,
            "query_cache": {},
        }

        if texts and TfidfVectorizer is not None:
            try:
                vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
                mat = vec.fit_transform(texts)
                idx["vectorizer"] = vec
                idx["matrix"] = mat
            except Exception:
                idx["vectorizer"] = None
                idx["matrix"] = None

        _KB_INDEX_CACHE[kb_path] = idx
        return idx


class DeepSeekWithMock:
    """Wrapper around deepseek_client.correct_text that enforces JSON shape
    and falls back to a small mock correction mode when the real API fails.
    """

    def __init__(self):
        self._client_available = bool(getattr(deepseek_client, "DEEPSEEK_API_KEY", "")) and bool(
            getattr(deepseek_client, "DEEPSEEK_API_URL", "")
        )
        self._warned_unavailable = False

    def correct(self, text: str, context: Optional[str] = None, role_prompt: Optional[str] = None) -> Dict[str, Any]:
        out = deepseek_client.correct_text(text, context=context, role_prompt=role_prompt)
        original = out.get("original", text)
        fixed = out.get("fixed", original)
        reason = out.get("reason", "")
        diff_markup = out.get("diff_markup")
        if (not self._client_available) and (not self._warned_unavailable):
            self._warned_unavailable = True
            print(
                "[WARN] DeepSeek client not configured (missing DEEPSEEK_API_KEY/DEEPSEEK_API_URL). "
                "Using local mock correction only.",
                file=sys.stderr,
            )
        if (not self._client_available) or reason in ("client_not_configured", "request_exception", "sync_run_error") or (original == fixed and (out.get("confidence", 0.0) <= 0.0)):
            replacements = {
                "可压器": "变压器",
                "开关拒合": "开关合闸",
                "电心设务": "电力设备",
                "电心设备": "电力设备",
            }
            fixed_mock = original
            for k, v in replacements.items():
                if k in fixed_mock:
                    fixed_mock = fixed_mock.replace(k, v)
            if fixed_mock != original:
                diff_markup = _make_diff_markup(original, fixed_mock)
                return {"original": original, "fixed": fixed_mock, "reason": "mock_correction", "diff_markup": diff_markup}
        if diff_markup is None:
            if original != fixed:
                diff_markup = _make_diff_markup(original, fixed)
            else:
                diff_markup = ""
        return {"original": original, "fixed": fixed, "reason": reason or "ok", "diff_markup": diff_markup}


def _make_diff_markup(original: str, fixed: str) -> str:
    if original == fixed:
        return ""
    if original in fixed:
        return fixed.replace(original, f"~~{original}~~ -> **{original}**")
    return f"~~{original}~~ -> **{fixed}**"


def _warn_no_sklearn_once() -> None:
    global _WARNED_NO_SKLEARN
    if _WARNED_NO_SKLEARN:
        return
    _WARNED_NO_SKLEARN = True
    print("[WARN] scikit-learn unavailable, falling back to difflib context ranking.", file=sys.stderr)


def _get_top_k_contexts(kb_path: str, query: str, k: int = 20) -> Tuple[str, List[int]]:
    idx = _get_kb_index(kb_path)
    texts = idx.get("texts") or []
    entries = idx.get("entries") or []
    context_texts = idx.get("context_texts") or texts
    if not texts:
        return "", []

    cache_key = f"{k}|{query[:512]}"
    query_cache = idx.get("query_cache")
    if isinstance(query_cache, dict) and cache_key in query_cache:
        return query_cache[cache_key]

    try:
        vec = idx.get("vectorizer")
        mat = idx.get("matrix")
        if vec is not None and mat is not None:
            qv = vec.transform([query])
            import numpy as np

            sims = (mat @ qv.T).toarray().ravel()
            idxs = list(reversed(np.argsort(sims)))[:k]
        else:
            _warn_no_sklearn_once()
            sims = [SequenceMatcher(None, t[:4000], query[:4000]).ratio() for t in texts]
            idxs = sorted(range(len(texts)), key=lambda i: sims[i], reverse=True)[:k]

        contexts = []
        ids = []
        for i in idxs:
            if i < len(entries):
                e = entries[i]
                txt = context_texts[i] if i < len(context_texts) else texts[i]
                entry_id = e.get("id") or e.get("entryId") or i
                contexts.append(f"[{entry_id}] {txt}")
                ids.append(entry_id)
        result = ("\n\n".join(contexts), ids)
        if isinstance(query_cache, dict):
            # Keep cache bounded to avoid unlimited growth.
            if len(query_cache) > 2048:
                query_cache.clear()
            query_cache[cache_key] = result
        return result
    except Exception:
        result = ("\n\n".join(context_texts[:k]), list(range(min(k, len(texts)))))
        if isinstance(query_cache, dict):
            if len(query_cache) > 2048:
                query_cache.clear()
            query_cache[cache_key] = result
        return result


def process_images_with_corrections(image_paths: List[str], kb_path: str, out_dir: Optional[str] = None, max_workers: int = 4) -> List[Dict[str, Any]]:
    out_dir_p = Path(out_dir or Path.cwd())
    out_dir_p.mkdir(parents=True, exist_ok=True)
    # Warm up retriever index once to avoid per-cell expensive setup.
    _get_kb_index(kb_path)
    ds = DeepSeekWithMock()
    results: List[Dict[str, Any]] = []
    change_log_rows: List[Dict[str, Any]] = []

    def _process_one(image_path: str) -> Dict[str, Any]:
        res = process_image(image_path, out_dir=str(out_dir_p))
        payload = res.get("payload", {})
        img_name = Path(image_path).name
        effective_candidates = 0
        skipped_noise = 0
        skipped_clean = 0
        for c in payload.get("cells", []):
            orig = c.get("text", "")
            if not orig:
                continue
            if not _is_effective_candidate_text(orig):
                skipped_noise += 1
                # Preserve original text and avoid low-value API calls.
                c["marked_text"] = orig
                continue

            should_correct, suspicion_reasons, suspicion_score = _should_send_to_deepseek(orig)
            if not should_correct:
                skipped_clean += 1
                c["marked_text"] = orig
                continue

            effective_candidates += 1
            context_text, entry_ids = _get_top_k_contexts(kb_path, orig, k=20)
            role = (
                "你是铁路电力OCR纠错助手。"
                " 仅修正明确的OCR识别错误，不要改写原意，不要扩写。"
                " 若不确定请保持原文。"
            )
            corr_key = orig.strip()
            with _CORRECTION_CACHE_LOCK:
                corrected = _CORRECTION_CACHE.get(corr_key)
            if corrected is None:
                corrected = ds.correct(orig, context=context_text, role_prompt=role)
                with _CORRECTION_CACHE_LOCK:
                    if len(_CORRECTION_CACHE) > 4096:
                        _CORRECTION_CACHE.clear()
                    _CORRECTION_CACHE[corr_key] = corrected
            original = corrected.get("original", orig)
            fixed = corrected.get("fixed", original)
            diff = corrected.get("diff_markup", _make_diff_markup(original, fixed))
            if original != fixed:
                marked = orig.replace(original, f"~~{original}~~ -> **{fixed}**", 1)
            else:
                marked = orig
            c["text"] = fixed
            c["marked_text"] = marked
            change_log_rows.append({
                "image": img_name,
                "page": payload.get("meta", {}).get("page", ""),
                "original_recognition": original,
                "ai_correction": fixed,
                "kb_entry_ids": ";".join([str(x) for x in entry_ids]) if entry_ids else "",
                "diff_markup": diff,
                "suspicion_reasons": suspicion_reasons,
                "suspicion_score": f"{suspicion_score:.3f}",
            })

        json_out = Path(res.get("json"))
        try:
            with json_out.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        content_lines = [c.get("marked_text", c.get("text", "")) for c in payload.get("cells", [])]
        content_md = "\n\n".join([ln for ln in content_lines if ln])
        kb_like = {
            "image": str(image_path),
            "contentMarkdown": content_md,
            "contentNormalized": [c.get("text", "") for c in payload.get("cells", [])],
        }
        kb_out = out_dir_p / f"{Path(image_path).stem}.kb.json"
        try:
            with kb_out.open("w", encoding="utf-8") as f:
                json.dump(kb_like, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return {
            "image": img_name,
            "json": str(json_out),
            "kb": str(kb_out),
            "effective_candidates": effective_candidates,
            "skipped_noise": skipped_noise,
            "skipped_clean": skipped_clean,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_one, p): p for p in image_paths}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                results.append({"error": str(e), "image": futures[fut]})

    csv_path = out_dir_p / "change_log.csv"
    fieldnames = [
        "image",
        "page",
        "original_recognition",
        "ai_correction",
        "kb_entry_ids",
        "diff_markup",
        "suspicion_reasons",
        "suspicion_score",
    ]
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in change_log_rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception:
        pass

    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Production refactor: batch OCR+RAG+DeepSeek mock-safe correction pipeline")
    ap.add_argument("--images", nargs="+", required=False, help="Image paths to process")
    ap.add_argument("--images-dir", default=None, help="Directory (or glob) containing images (eg. --images-dir C:\\...\\截图)")
    ap.add_argument("--kb", required=True, help="Path to knowledge_base.json")
    ap.add_argument("--out-dir", default=None, help="Output directory")
    ap.add_argument("--workers", type=int, default=4, help="Max parallel workers")
    args = ap.parse_args()

    # support --images-dir which will expand PNG/JPG files inside that directory
    images = []
    if args.images:
        images.extend(args.images)
    if args.images_dir:
        from pathlib import Path
        p = Path(args.images_dir)
        if p.is_dir():
            images.extend([str(x) for x in sorted(p.glob('*.png'))])
            images.extend([str(x) for x in sorted(p.glob('*.jpg'))])
            images.extend([str(x) for x in sorted(p.glob('*.jpeg'))])
        else:
            # allow glob pattern
            import glob
            images.extend(glob.glob(args.images_dir))

    if not images:
        raise SystemExit('No images provided. Use --images or --images-dir to specify inputs.')

    res = process_images_with_corrections(images, args.kb, out_dir=args.out_dir, max_workers=args.workers)
    print("Processed:", res)
