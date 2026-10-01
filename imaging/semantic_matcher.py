import re
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from imaging.text_utils import (
    normalize_for_search_like_app,
    get_text_fingerprint,
    levenshtein_similarity,
    safe_str,
    normalize_anchor_key,
    ocr_match_tokens,
    figure_table_label_variants
)
from imaging.pdf_analyzer import (
    entry_page_hint,
    collect_entry_pages
)

def normalize_for_similarity(text: str) -> str:
    """Normalize text for fuzzy matching by removing all whitespace."""
    norm = normalize_for_search_like_app(safe_str(text)).replace("\n", " ").strip()
    return re.sub(r"\s+", "", norm)

def fingerprint_similarity_ratio(a_text: str, b_text: str, *, cutoff_ratio: float) -> float:
    """Compare text fingerprints using Levenshtein similarity."""
    a = get_text_fingerprint(a_text)
    b = get_text_fingerprint(b_text)
    if not a or not b:
        return 0.0
    return levenshtein_similarity(a, b, cutoff_ratio=cutoff_ratio)

def has_meaningful_anchor_text(anchor_text: str) -> bool:
    """Check if anchor text is long enough to be useful for matching."""
    fp = get_text_fingerprint(anchor_text)
    if len(fp) >= 2:
        return True
    raw = normalize_for_similarity(anchor_text)
    return len(raw) >= 2

def semantic_similarity_ratio(a_text: str, b_text: str, *, cutoff_ratio: float) -> float:
    """Compute semantic similarity using fingerprint first, then normalized text."""
    fp_ratio = fingerprint_similarity_ratio(a_text, b_text, cutoff_ratio=cutoff_ratio)
    if fp_ratio > 0.0:
        return fp_ratio
    a = normalize_for_similarity(a_text)
    b = normalize_for_similarity(b_text)
    if not a or not b:
        return 0.0
    return levenshtein_similarity(a, b, cutoff_ratio=cutoff_ratio)

def order_score(item_index: int, item_total: int, entry_index: int, entry_total: int) -> float:
    """Order consistency score in [0,1] based on relative rank distance."""
    if item_total <= 1 or entry_total <= 1:
        return 0.0
    i_rank = float(item_index) / float(item_total - 1)
    e_rank = float(entry_index) / float(entry_total - 1)
    return max(0.0, 1.0 - abs(e_rank - i_rank))

def find_target_entry_index(
    entries: List[Dict[str, Any]],
    page_number: int,
    strategy: str,
    heading_text: str = "",
    prefer_heading: bool = True,
    candidate_indices: Any = None,
    allowed_pages: Optional[Set[int]] = None,
) -> Optional[int]:
    """Find the best KB entry to merge a manifest item into, based on page and text."""
    allowed = allowed_pages if allowed_pages is not None else {int(page_number)}
    candidates: List[Tuple[int, Dict[str, Any]]] = []
    
    indices_to_check = candidate_indices if candidate_indices is not None else range(len(entries))
    for idx in indices_to_check:
        if idx < 0 or idx >= len(entries): continue
        entry = entries[idx]
        pages = collect_entry_pages(entry)
        if any(p in allowed for p in pages):
            candidates.append((idx, entry))

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][0]

    heading_text = (heading_text or "").strip()
    heading_key = ""
    if prefer_heading and heading_text:
        heading_key = normalize_for_search_like_app(heading_text).replace("\n", " ").strip()

    if heading_key:
        def score_entry(e_obj: Dict[str, Any]) -> int:
            score = 0
            for field in ("jobTitle", "unitName", "title", "entryId"):
                v = safe_str(e_obj.get(field)).strip()
                if not v: continue
                v_norm = normalize_for_search_like_app(v).replace("\n", " ").strip()
                if v_norm and heading_key in v_norm:
                    score += 120

            cn = safe_str(e_obj.get("contentNormalized")).strip()
            if cn and heading_key in cn:
                score += 60

            if len(heading_key) < 3:
                score = int(score * 0.5)
            return score

        scored = [(score_entry(e_cand), idx_cand) for (idx_cand, e_cand) in candidates]
        scored.sort(key=lambda t: (-t[0], t[1]))
        best_score, best_idx = scored[0]
        if best_score > 0:
            return best_idx

    # Prefer entries whose dominant page matches
    dom_matching = [idx for idx, e in candidates if entry_page_hint(e) == page_number]
    if dom_matching:
        return dom_matching[0]

    # Prefer smaller position if present.
    candidates.sort(key=lambda item: (item[1].get("position") if isinstance(item[1].get("position"), int) else 10**9, item[0]))
    return candidates[0][0]

def page_window(page_number: int, radius: int = 2) -> List[int]:
    p = int(page_number or 0)
    r = max(0, int(radius or 0))
    return [x for x in range(p - r, p + r + 1) if x > 0]

def candidate_indices_for_page_window(page_to_entries: Dict[int, List[int]], page_number: int, radius: int = 2) -> List[int]:
    out: List[int] = []
    seen: Set[int] = set()
    for p in page_window(page_number, radius=radius):
        for idx in page_to_entries.get(p, []):
            if idx in seen: continue
            seen.add(idx)
            out.append(idx)
    return out

def entry_contains_anchor(entry: Dict[str, Any], anchor_key: str) -> int:
    """Return a score (>0 means match) for anchor_key in entry signals."""
    key = (anchor_key or "").strip()
    if not key: return 0

    score = 0
    for field in ("jobTitle", "unitName", "title", "entryId"):
        v = safe_str(entry.get(field)).strip()
        if not v: continue
        v_norm = normalize_for_search_like_app(v).replace("\n", " ").strip()
        if v_norm and key in v_norm:
            score += 300

    cn = safe_str(entry.get("contentNormalized")).strip()
    if cn and key in cn:
        score += 200

    return score

def entry_has_image_ref_markers(entry: Dict[str, Any]) -> bool:
    """Check if the entry or its blocks contains the [IMAGE_REF: marker."""
    for field in ("contentMarkdown", "contentNormalized"):
        if "[IMAGE_REF:" in safe_str(entry.get(field)):
            return True

    blocks = entry.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict): continue
            for key in ("code", "text", "contentMarkdown", "contentNormalized"):
                if "[IMAGE_REF:" in safe_str(block.get(key)):
                    return True
    return False

def manifest_item_semantic_kind_hint(item: Dict[str, Any]) -> str:
    from imaging.pdf_analyzer import basename_from_uri
    kind = safe_str(item.get("kind")).strip().lower()
    out_file = safe_str(item.get("outFile")).strip().lower()
    
    # Try multiple URI keys
    uri = ""
    for k in ("assetUri", "asset_uri", "imageUri", "image_uri", "src"):
        val = item.get(k)
        if val:
            uri = str(val).strip()
            break
            
    basename = basename_from_uri(uri).lower()
    if kind == "table" or basename.startswith("table_") or out_file.startswith("table_"):
        return "table"
    if kind == "legend" or basename.startswith("legend_") or out_file.startswith("legend_") or basename.startswith("visual_p") or out_file.startswith("visual_p"):
        return "legend"
    return "unknown"

def iter_entry_semantic_texts(entry: Dict[str, Any]) -> List[str]:
    """Iterate over all strings in an entry that contribute to its semantic meaning."""
    texts: List[str] = []
    seen: Set[str] = set()

    def _push(value: Any, *, max_len: int = 4000) -> None:
        text = safe_str(value).strip()
        if not text: return
        text = text[: max(1, int(max_len))]
        if text in seen: return
        seen.add(text)
        texts.append(text)

    for field in ("jobTitle", "unitName", "title", "entryId", "contentNormalized", "contentMarkdown"):
        _push(entry.get(field), max_len=5000 if field in ("contentNormalized", "contentMarkdown") else 240)

    blocks = entry.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict): continue
            block_type = safe_str(block.get("type")).strip().lower()
            if block_type not in {"code", "text", "markdown", "table"}: continue
            for key in ("code", "text", "contentMarkdown", "contentNormalized", "caption"):
                _push(block.get(key), max_len=3000)

    return texts

def build_entry_semantic_profile(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Precompute normalized forms and fingerprints for an entry for faster matching."""
    short_texts: List[str] = []
    long_parts: List[str] = []

    for field in ("jobTitle", "unitName", "title", "entryId"):
        text = safe_str(entry.get(field)).strip()
        if text: short_texts.append(text[:240])

    for text in iter_entry_semantic_texts(entry):
        if text not in short_texts:
            long_parts.append(text)

    short_norms = [normalize_for_search_like_app(text).replace("\n", " ").strip() for text in short_texts]
    short_fps = [get_text_fingerprint(text) for text in short_texts]
    long_blob = "\n".join(long_parts)
    long_norm = normalize_for_search_like_app(long_blob).replace("\n", " ").strip()
    long_fp = get_text_fingerprint(long_blob)

    return {
        "entryPage": entry_page_hint(entry),
        "shortTexts": short_texts,
        "shortNorms": [v for v in short_norms if v],
        "shortFingerprints": [v for v in short_fps if v],
        "longNorm": long_norm,
        "longCompact": re.sub(r"\s+", "", long_norm),
        "longFingerprint": long_fp,
    }

def debug_print_nearest_entries(entries: List[Dict[str, Any]], target_page: int, limit: int = 6) -> None:
    """Print a list of entries close to the target page for debugging mismatch issues."""
    try:
        items = []
        for eid_idx, entry in enumerate(entries):
            p = entry_page_hint(entry)
            if p is None: continue
            eid = safe_str(entry.get("entryId")).strip()
            jt = safe_str(entry.get("jobTitle")).strip()
            items.append((abs(p - int(target_page)), p, eid, jt))
            
        items.sort(key=lambda t: (t[0], t[1]))
        items = items[: max(1, int(limit))]
        print(f"[MergeDebug] Matching failed: target_page={target_page} nearest entry pages (top {len(items)}):")
        for diff, p, eid, jt in items:
            print(f"  - page={p} (Δ={diff}) entryId={eid} jobTitle={jt[:80]}")
    except Exception:
        pass

def score_manifest_signal_against_entry(
    anchor_signal: str,
    entry: Dict[str, Any],
    profile: Dict[str, Any],
    *,
    item_kind: str,
    cutoff_ratio: float,
    page_number: Optional[int],
) -> Tuple[int, int, float]:
    """Score how well a manifest signal text matches a KB entry's semantic profile."""
    signal_text = safe_str(anchor_signal).strip()
    if not signal_text: return (0, 0, 0.0)

    signal_norm = normalize_for_search_like_app(signal_text).replace("\n", " ").strip()
    signal_fp = get_text_fingerprint(signal_text)
    signal_tokens = ocr_match_tokens(signal_text)

    short_texts = profile.get("shortTexts", [])
    short_norms = profile.get("shortNorms", [])
    short_fps = profile.get("shortFingerprints", [])
    long_norm = safe_str(profile.get("longNorm")).strip()
    long_compact = safe_str(profile.get("longCompact")).strip()
    long_fp = safe_str(profile.get("longFingerprint")).strip()

    contain_score = 0
    if signal_norm:
        for text in short_norms:
            if signal_norm in text: contain_score = max(contain_score, 900)
        if long_norm and signal_norm in long_norm: contain_score = max(contain_score, 620)

    if signal_fp:
        for fp in short_fps:
            if signal_fp in fp or fp in signal_fp: contain_score = max(contain_score, 980)
        if long_fp and signal_fp in long_fp: contain_score = max(contain_score, 760)

    token_hits = 0
    if signal_tokens:
        haystacks = list(short_norms)
        if long_compact: haystacks.append(long_compact)
        for token in signal_tokens[:12]:
            compact = re.sub(r"\s+", "", token)
            if len(compact) < 2: continue
            if any(compact in hay for hay in haystacks if hay): token_hits += 1

    similarity_best = 0.0
    for text in short_texts:
        similarity_best = max(similarity_best, semantic_similarity_ratio(signal_text, text, cutoff_ratio=cutoff_ratio))
    if long_norm:
        similarity_best = max(similarity_best, semantic_similarity_ratio(signal_text, long_norm[:2400], cutoff_ratio=cutoff_ratio))

    reference_bonus = 0
    label_variants = figure_table_label_variants(signal_text)
    if label_variants:
        haystacks = list(short_norms)
        if long_compact: haystacks.append(long_compact)
        for hay in haystacks:
            hay_compact = normalize_anchor_key(hay)
            if not hay_compact: continue
            for label in label_variants:
                if label in hay_compact: reference_bonus = max(reference_bonus, 120)
                if re.search(rf"(见|如|按|所示|示例图|示意图|接线图|布置图).*{re.escape(label)}", hay_compact): reference_bonus = max(reference_bonus, 280)
                if re.search(rf"{re.escape(label)}.*(所示|示例图|示意图|接线图|布置图|明细表)", hay_compact): reference_bonus = max(reference_bonus, 240)

    entry_page = profile.get("entryPage")
    page_bonus = 0
    if isinstance(page_number, int) and page_number > 0 and isinstance(entry_page, int) and entry_page > 0:
        diff = abs(page_number - entry_page)
        if diff == 0: page_bonus = 90
        elif diff <= 2: page_bonus = 40
        elif diff <= 6: page_bonus = 15

    from imaging.table_processor import entry_has_table_block
    entry_has_table = entry_has_table_block(entry)
    entry_has_image_refs = entry_has_image_ref_markers(entry)

    kind_bonus = 0
    if item_kind == "table":
        kind_bonus += 220 if entry_has_table else -80
    elif item_kind == "legend":
        kind_bonus += -90 if entry_has_table else 40
        if entry_has_image_refs: kind_bonus += 95

    score = int(contain_score + token_hits * 55 + int(similarity_best * 220.0) + page_bonus + kind_bonus + reference_bonus)
    evidence = 0
    if contain_score > 0: evidence += 1
    if token_hits > 0: evidence += 1
    if similarity_best > 0.0: evidence += 1
    if reference_bonus >= 240: evidence += 1
    return (score, evidence, float(similarity_best))
