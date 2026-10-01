import re
import logging
import unicodedata
from typing import List, Dict, Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

def normalize_for_search_like_app(text: str) -> str:
    """Port of TextSanitizer.normalizeForSearch (Kotlin).
    Keep only letters/digits and whitespace; everything else becomes space.
    Newlines are preserved.
    """
    if not text or text.strip() == "":
        return ""
    normalized = unicodedata.normalize("NFC", text)
    out_chars: List[str] = []
    for ch in normalized:
        if ch == "\n":
            out_chars.append("\n")
        elif ch.isalnum():
            out_chars.append(ch)
        elif ch.isspace():
            out_chars.append(" ")
        else:
            out_chars.append(" ")

    out = "".join(out_chars)
    out = re.sub(r"[\t\r\u00A0 ]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    return out.strip()

def normalize_simple_compact(text: Any) -> str:
    t = str(text) if text is not None else ""
    norm = normalize_for_search_like_app(t).replace("\n", " ").strip()
    norm = re.sub(r"\s+", " ", norm)
    return norm

def normalize_anchor_key(text: str) -> str:
    """Specialized normalization for figure/table keys and anchors."""
    if not text: return ""
    # Use standard app search normalization as base
    normalized = normalize_for_search_like_app(str(text)).replace("\n", " ").strip()
    # Standardize common dash-like characters
    normalized = normalized.replace("—", "-").replace("－", "-").replace("～", "~").replace("〜", "~")
    # Remove all whitespace for exact key matching
    normalized = re.sub(r"\s+", "", normalized)
    return normalized

def get_text_fingerprint(text: str) -> str:
    """Get a robust fingerprint string for cross-source alignment.
    Filters boilerplate prefixes, removes whitespace and brackets.
    """
    if not text:
        return ""

    s = normalize_for_search_like_app(str(text)).replace("\n", " ").strip()
    if not s:
        return ""

    # If OCR captured extra noise before the actual label/title, cut from the first marker.
    m = re.search(r"(附件|附表|表|图|第\s*[0-9一二三四五六七八九十百千]+\s*条)", s)
    if m and m.start() > 0:
        s = s[m.start():].strip()

    # Remove bracket characters
    s = s.translate(str.maketrans({
        "(": "", ")": "", "（": "", "）": "", "[": "", "]": "",
        "【": "", "】": "", "{": "", "}": "", "<": "", ">": "",
        "《": "", "》": "", "“": "", "”": "", "\"": "",
    }))

    # Strip common leading labels/prefixes
    for _ in range(3):
        before = s
        s = s.strip()
        s = re.sub(r"^(?:附件|附表|表|图)\s*[0-9一二三四五六七八九十百千]+(?:[-._][0-9]+)?\s*", "", s)
        s = re.sub(r"^第\s*[0-9一二三四五六七八九十百千]+\s*条\s*", "", s)
        s = re.sub(r"^[：:、.\-]+\s*", "", s)
        if s == before:
            break

    # Remove all whitespace
    s = re.sub(r"\s+", "", s)
    return s

def levenshtein_distance(a: str, b: str, cutoff: Optional[int] = None) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if len(a) < len(b): a, b = b, a
    if cutoff is not None and abs(len(a) - len(b)) > cutoff:
        return cutoff + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
        if cutoff is not None and min(prev) > cutoff:
            return cutoff + 1
    return prev[-1]

def levenshtein_similarity(a: str, b: str, cutoff_ratio: float = 0.0) -> float:
    a = a or ""
    b = b or ""
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0: return 1.0
    cutoff = int(max_len * (1.0 - cutoff_ratio)) + 1
    dist = levenshtein_distance(a, b, cutoff=cutoff)
    if dist > cutoff: return 0.0
    return max(0.0, 1.0 - (float(dist) / float(max_len)))

def safe_str(v: Any) -> str:
    """Safe conversion to string, handling None as empty string."""
    return "" if v is None else str(v)

def collapse_match_text(text: str) -> str:
    """Remove all whitespace from normalized OCR text for matching."""
    return re.sub(r"\s+", "", normalize_simple_compact(text))

def ocr_match_tokens(text: str) -> List[str]:
    """Tokenize normalized OCR text, filtering out common generic Chinese terms."""
    norm = normalize_simple_compact(text)
    if not norm:
        return []

    raw_tokens = [token.strip() for token in re.split(r"\s+", norm) if token.strip()]
    raw_tokens = [token for token in raw_tokens if len(token) >= 2]
    generic = {
        "检查", "要求", "内容", "项目", "备注", "规定", "单位", "名称", "标准", "周期", "范围",
        "应当", "必须", "可以", "进行", "作业", "设备", "附件", "附表", "表格", "图例",
    }

    out: List[str] = []
    for token in raw_tokens:
        if token in generic:
            continue
        out.append(token)
    return out

def extract_figure_table_labels(text: str) -> List[str]:
    """Extract labels like '图1-1' or '表2' from text."""
    out: List[str] = []
    seen: Set[str] = set()
    # Matches patterns like 图1-1, 表2, 附图3.4 etc.
    # Supported numbers: digits and common Chinese numerals
    pattern = r"((?:附图|附表|图|表)\s*[0-9一二三四五六七八九十零〇O口]+(?:\s*[-—~〜\.．]\s*[0-9一二三四五六七八九十零〇O口]+)*)"
    for match in re.finditer(pattern, safe_str(text)):
        label = normalize_anchor_key(match.group(1))
        if not label or label in seen:
            continue
        seen.add(label)
        out.append(label)
    return out

def figure_table_label_variants(text: str) -> List[str]:
    """Get all variants of figure/table labels found in text for better matching."""
    out: List[str] = []
    seen: Set[str] = set()
    for label in extract_figure_table_labels(text):
        variants = {label, label.replace("-", ""), label.replace("~", "")}
        # Handle cases where hyphen might be missing between prefix and number
        match = re.match(r"^(附图|附表|图|表)(\d{2,})$", label)
        if match:
            head = match.group(1)
            digits = match.group(2)
            if len(digits) >= 2:
                variants.add(f"{head}{digits[0]}-{digits[1:]}")
        for variant in variants:
            value = normalize_anchor_key(variant)
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
    return out

def extract_figure_label_pairs(text: str) -> List[Tuple[str, str]]:
    """Extract pairs of (raw_label, normalized_label) from text."""
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    raw_text = safe_str(text)
    pattern = r"((?:附图|图)\s*[0-9一二三四五六七八九十零〇O口]+(?:\s*[-—~〜\.．]\s*[0-9一二三四五六七八九十零〇O口]+)*)"
    for match in re.finditer(pattern, raw_text):
        raw_label = safe_str(match.group(1)).strip()
        normalized_label = normalize_anchor_key(raw_label)
        if not normalized_label or normalized_label in seen:
            continue
        seen.add(normalized_label)
        out.append((raw_label, normalized_label))
    return out
