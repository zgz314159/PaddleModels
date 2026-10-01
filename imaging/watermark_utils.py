import re
import logging
from typing import List, Tuple, Dict, Any, Optional, Set
try:
    import fitz
except ImportError:
    fitz = None

logger = logging.getLogger(__name__)

# Constants for watermark detection
REPEATED_WATERMARK_MIN_PAGES = 3
REPEATED_WATERMARK_MAX_THRESHOLD = 12

def normalize_repeated_watermark_text(text: str) -> str:
    raw = str(text or '').strip()
    if not raw:
        return ''
    compact = re.sub(r'[\s\-_—–~〜·•,，.。:：;；|/\\]+', '', raw)
    compact = compact.replace('（', '').replace('）', '').replace('(', '').replace(')', '')
    return compact.strip()

def is_plausible_repeated_watermark_text(text: str) -> bool:
    compact = normalize_repeated_watermark_text(text)
    if not compact:
        return False
    if len(compact) < 5 or len(compact) > 24:
        return False
    if re.search(r'[。！？!?；;]', str(text or '')):
        return False
    if re.match(r'^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+', compact, flags=re.IGNORECASE):
        return False
    if re.match(r'^第[一二三四五六七八九十百零〇0-9]+[章节]', compact):
        return False
    if re.match(r'^[一二三四五六七八九十]+、', compact):
        return False
    digit_count = len(re.findall(r'\d', compact))
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', compact))
    alpha_count = len(re.findall(r'[A-Za-z]', compact))
    if digit_count < 3:
        return False
    if cjk_count + alpha_count < 1:
        return False
    if cjk_count > 10:
        return False
    return True

def looks_like_repeated_watermark_text(text: str, repeated_candidates: Optional[Set] = None) -> bool:
    compact = normalize_repeated_watermark_text(text)
    if not compact or not repeated_candidates:
        return False
    text_digits = ''.join(re.findall(r'\d', compact))
    text_is_alpha_cjk = bool(re.fullmatch(r'[\u4e00-\u9fffA-Za-z]+', compact))
    for candidate in repeated_candidates:
        cand = normalize_repeated_watermark_text(candidate)
        if not cand:
            continue
        cand_digits = ''.join(re.findall(r'\d', cand))
        if compact == cand:
            return True
        if compact in cand and len(compact) >= max(5, int(len(cand) * 0.45)):
            return True
        if text_is_alpha_cjk and cand.startswith(compact) and len(compact) >= 3:
            return True
        if cand in compact and len(cand) >= 5 and (len(compact) - len(cand)) <= 2:
            return True
        if (
            text_digits
            and len(text_digits) >= 3
            and text_digits in cand_digits
            and len(compact) <= 8
            and re.search(r'[\u4e00-\u9fffA-Za-z]', compact)
        ):
            return True
    return False

def filter_repeated_watermark_lines(lines: List[str], repeated_candidates: Optional[Set] = None) -> List[str]:
    if not repeated_candidates:
        return [str(line or '').strip() for line in (lines or []) if str(line or '').strip()]
    filtered: List[str] = []
    for line in lines or []:
        line_text = str(line or '').strip()
        if not line_text:
            continue
        if looks_like_repeated_watermark_text(line_text, repeated_candidates):
            continue
        filtered.append(line_text)
    return filtered

def filter_repeated_watermark_text(text: str, repeated_candidates: Optional[Set] = None) -> str:
    raw = str(text or '')
    if not raw:
        return ''
    lines = raw.splitlines()
    filtered = filter_repeated_watermark_lines(lines, repeated_candidates)
    return '\n'.join(filtered)

def filter_repeated_watermark_dicts(items: List[Dict], repeated_candidates: Optional[Set] = None) -> List[Dict]:
    if not repeated_candidates:
        return list(items or [])
    filtered: List[Dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get('text') or item.get('code') or item.get('contentMarkdown') or '').strip()
        if text and looks_like_repeated_watermark_text(text, repeated_candidates):
            continue
        filtered.append(item)
    return filtered

def collect_repeated_watermark_candidates(doc) -> Set[str]:
    page_hits: Dict[str, Set[int]] = {}
    try:
        page_count = int(getattr(doc, 'page_count', 0) or 0)
    except Exception:
        page_count = 0
    if page_count <= 0:
        return set()

    threshold = max(
        REPEATED_WATERMARK_MIN_PAGES,
        min(REPEATED_WATERMARK_MAX_THRESHOLD, max(REPEATED_WATERMARK_MIN_PAGES, (page_count + 9) // 10)),
    )

    for page_index in range(page_count):
        try:
            page = doc.load_page(page_index)
            data = page.get_text('dict') or {}
        except Exception:
            continue
        blocks = data.get('blocks') if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            continue
        page_seen: Set[str] = set()
        for blk in blocks:
            if not isinstance(blk, dict) or int(blk.get('type', 0)) != 0:
                continue
            lines = blk.get('lines') if isinstance(blk.get('lines'), list) else []
            for line in lines:
                if not isinstance(line, dict):
                    continue
                spans = line.get('spans') if isinstance(line.get('spans'), list) else []
                span_texts: List[str] = []
                for span in spans:
                    if not isinstance(span, dict):
                        continue
                    txt = str(span.get('text') or '').strip()
                    if not txt:
                        continue
                    span_texts.append(txt)
                    compact = normalize_repeated_watermark_text(txt)
                    if is_plausible_repeated_watermark_text(compact):
                        page_seen.add(compact)
                line_text = ' '.join(span_texts).strip()
                compact_line = normalize_repeated_watermark_text(line_text)
                if is_plausible_repeated_watermark_text(compact_line):
                    page_seen.add(compact_line)
        for text in page_seen:
            page_hits.setdefault(text, set()).add(page_index + 1)

    repeated = {
        text
        for text, pages in page_hits.items()
        if len(pages) >= threshold
    }
    return repeated
