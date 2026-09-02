"""文本语义分类：噪音过滤、标题判定、图注/标注识别、结构检测"""

import re
import unicodedata
from typing import Optional, Tuple

from classification.text_quality_gate import analyze_text_suspicion

# ===== 正则模式库 =====

_ARTICLE_RE = re.compile(r"^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*条\b(.*)$")
_APPENDIX_RE = re.compile(r"^(附录|附件)\s*([0-9一二三四五六七八九十百千〇零两]+)\b[：:]?(.*)$")
_APPENDIX_TABLE_RE = re.compile(r"^附表\s*([0-9一二三四五六七八九十百千〇零两]+)\b[：:]?(.*)$")
_CHAPTER_HEADING_RE = re.compile(r"^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*章(?:\s+|[：:])?(.*)$")
_CN_SECTION_HEADING_RE = re.compile(r"^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*节(?:\s+|[：:])?(.*)$")
_CN_SUBSECTION_HEADING_RE = re.compile(r"^([一二三四五六七八九十百千]+)\s*[、,，.]\s*(.+)$")
_TOP_LEVEL_SECTION_RE = re.compile(r"^([0-9]{1,2})(?![）\)])(?:[、，,．.]\s*|\s+)?([^\s0-9）\)].+)$")
_NUMBERED_SECTION_RE = re.compile(r"^([0-9]{1,2}(?:\.[0-9]{1,3}){1})(?:[、，,．.]\s*|\s+)?([^\s0-9）\)].+)$")
_NUMBERED_CLAUSE_RE = re.compile(r"^([0-9]{1,2}(?:\.[0-9]{1,3}){2,})(?:[、，,．.]?\s*(.*))?$")
_ABBREVIATED_NUMBER_CHAIN_RE = re.compile(r"^[.．]\s*([0-9]{1,3}(?:\s*[.．]\s*[0-9]{1,3}){0,})(?:[、，,．.]?\s*(.*))?$")
_CONTEXTUAL_NUMBER_CHAIN_RE = re.compile(r"^([0-9]{1,3}(?:\s*[.．]\s*[0-9]{1,3}){1,})(?:[、，,．.]?\s*(.*))?$")

_CH_NUM_MAP = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}

# ===== 噪音过滤 =====

_HEADING_NOISE_EXACT = {
    "3m0",
    "h500",
    "1.7mo",
    "1.7m0",
}
_HEADING_NOISE_PATTERNS = (
    re.compile(r"^[hH]\s*/?\s*\d{2,4}$"),
    re.compile(r"^\d+(?:\.\d+)?\s*m\s*[o0]$", re.IGNORECASE),
)
_NONSEMANTIC_HEADING_PATTERNS = (
    re.compile(r"^零件\s*[0-9A-Za-z一二三四五六七八九十]+$", re.IGNORECASE),
    re.compile(r"^[0-9一二三四五六七八九十]+\s*[（(]\s*[0-9Oo口]{2,}\s*[）)]"),
)


def parse_chinese_or_arabic_int(s: str) -> Optional[int]:
    raw = (s or "").strip()
    if not raw:
        return None
    if re.fullmatch(r"\d+", raw):
        try:
            return int(raw)
        except Exception:
            return None

    total = 0
    num = 0
    unit_map = {"十": 10, "百": 100, "千": 1000}
    seen_any = False
    for ch in raw:
        if ch in _CH_NUM_MAP:
            num = _CH_NUM_MAP[ch]
            seen_any = True
            continue
        if ch in unit_map:
            seen_any = True
            unit = unit_map[ch]
            if num == 0:
                num = 1
            total += num * unit
            num = 0
            continue
        return None

    if not seen_any:
        return None
    return total + num


def normalize_number_chain(text: str) -> str:
    return re.sub(r"\s*[.．]\s*", ".", (text or "").strip())


def replace_leading_number_chain(text: str, full_number: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    m = re.match(r"^(?:[.．]\s*)?[0-9]{1,3}(?:\s*[.．]\s*[0-9]{1,3})*", t)
    if not m:
        return t
    return f"{full_number}{t[m.end():]}".strip()


def infer_implicit_section_number(chapter_number: str, current_section_number: str) -> str:
    chapter = (chapter_number or "").strip()
    if not chapter:
        return ""

    current = (current_section_number or "").strip()
    prefix = f"{chapter}."
    if current.startswith(prefix):
        last = current.split(".")[-1].strip()
        if last.isdigit():
            return f"{chapter}.{int(last) + 1}"
    return f"{chapter}.1"


def normalize_heading_candidate_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", (text or "").strip()).lower()
    normalized = re.sub(r"[\s_/\\-]+", "", normalized)
    return normalized


def is_heading_noise_blacklisted(text: str) -> bool:
    raw = unicodedata.normalize("NFKC", (text or "").strip())
    compact = normalize_heading_candidate_key(raw)
    if not compact:
        return False
    if compact in _HEADING_NOISE_EXACT:
        return True
    return any(pattern.match(raw) for pattern in _HEADING_NOISE_PATTERNS)


def is_nonsemantic_heading_artifact(text: str) -> bool:
    raw = unicodedata.normalize("NFKC", (text or "").strip())
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return False
    if any(pattern.match(compact) for pattern in _NONSEMANTIC_HEADING_PATTERNS):
        return True
    if compact.startswith("零件") and len(compact) <= 8:
        return True
    if ("最小布置尺寸" in compact or "布置尺寸" in compact) and re.match(r"^[0-9一二三四五六七八九十（(]", compact):
        return True
    digit_count = len(re.findall(r"\d", compact))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
    latin_count = len(re.findall(r"[A-Za-z]", compact))
    if digit_count >= 3 and cjk_count <= 1 and re.search(r"[~〜\-—()（）_/\\]", compact):
        return True
    if digit_count >= 2 and latin_count == 0 and cjk_count <= 1 and re.fullmatch(r"[0-9Oo口~〜\-—()（）_/\\.]+", compact):
        return True
    return False


def is_low_confidence_heading_candidate(text: str) -> bool:
    raw = unicodedata.normalize("NFKC", (text or "").strip())
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return True
    if is_heading_noise_blacklisted(raw):
        return True
    if is_nonsemantic_heading_artifact(raw):
        return True

    report = analyze_text_suspicion(raw)
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
    asciiish = bool(re.fullmatch(r"[A-Za-z0-9./\\_+=:;,%()\[\]×x-]+", compact))

    if asciiish and cjk_count < 2 and report.compact_length <= 20:
        return True
    if cjk_count == 0 and re.search(r"[A-Za-z]", compact) and re.search(r"\d", compact):
        return True
    if report.score >= 2.5 and report.compact_length <= 20 and cjk_count < 3:
        return True
    return False


def is_likely_heading_title(title: str, *, max_len: int = 36) -> bool:
    t = re.sub(r"\s+", "", (title or "").strip())
    if not t or len(t) > max_len:
        return False
    if is_nonsemantic_heading_artifact(title):
        return False
    if is_low_confidence_heading_candidate(title):
        return False
    if t[0] in "-—,.，;；:：":
        return False
    # Reject OCR/code-like fragments that are rarely true headings.
    if any(ch in t for ch in "/\\=|~"):
        return False
    if re.search(r"\d\s*[xX×]\s*\d", t):
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", t))
    if re.search(r"[A-Za-z]{3,}|[A-Za-z]{2,}\d|\d[A-Za-z]{2,}", t) and cjk_count < 2:
        return False
    if any(ch in t for ch in "。；;！？!？：:"):
        return False
    return True


def is_likely_plain_section_heading(text: str) -> bool:
    t = re.sub(r"\s+", "", (text or "").strip())
    if not t or len(t) < 2 or len(t) > 24:
        return False
    if is_nonsemantic_heading_artifact(text):
        return False
    if is_low_confidence_heading_candidate(text):
        return False
    # Prefer natural-language headings and drop OCR noise like "QP6Z", "H/500", "40X40mm".
    if any(ch in t for ch in "/\\=|~"):
        return False
    if re.search(r"\d\s*[xX×]\s*\d", t):
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", t))
    if cjk_count == 0:
        return False
    if re.search(r"[A-Za-z]{3,}|[A-Za-z]{2,}\d|\d[A-Za-z]{2,}", t) and cjk_count < 2:
        return False
    if re.match(r"^[0-9.．]", t):
        return False
    # Figure/table captions are often short and can be misdetected as section headings,
    # e.g. "图2-4一路电源供电主结线示例图".
    if re.match(r"^[图表][0-9一二三四五六七八九十零〇\-—_.．]+", t):
        return False
    if any(token in t for token in ("示例图", "示意图", "安装图", "布置图", "接线图", "明细表")):
        return False
    if any(ch in t for ch in "：:。；;，,（）()[]【】"):
        return False
    return True


def is_likely_figure_or_table_caption(text: str) -> bool:
    t = re.sub(r"\s+", "", (text or "").strip())
    if not t:
        return False
    if re.match(r"^[图表]\s*[0-9一二三四五六七八九十零〇\-—_.．]+", t):
        return True
    if any(token in t for token in ("示例图", "示意图", "安装图", "布置图", "接线图", "明细表")) and ("图" in t or "表" in t):
        return True
    return False


def is_likely_caption_fragment(text: str, previous_text: str) -> bool:
    current = (text or "").strip()
    previous = (previous_text or "").strip()
    current_compact = re.sub(r"\s+", "", current)
    previous_compact = re.sub(r"\s+", "", previous)
    if not current_compact or not previous_compact:
        return False
    if len(current_compact) < 2 or len(current_compact) > 24:
        return False
    if detect_structured_heading(current) is not None:
        return False
    if detect_anchor(current) is not None:
        return False
    if any(ch in current_compact for ch in "。；;！？!?：:，,（）()[]【】"):
        return False
    previous_is_label_only = bool(
        re.match(r"^[图表]\s*[0-9一二三四五六七八九十零〇\-—_.．]+$", previous_compact)
    )
    previous_is_caption_like = is_likely_figure_or_table_caption(previous)
    if not previous_is_label_only and not previous_is_caption_like:
        return False
    if not is_likely_heading_title(current, max_len=24):
        return False
    if any(token in current_compact for token in ("第", "章", "节", "条")):
        return False
    return True


def is_likely_figure_callout_or_annotation(text: str) -> bool:
    raw = (text or "").strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return False
    if len(compact) > 24:
        return False
    if detect_structured_heading(raw) is not None:
        return False
    if detect_anchor(raw) is not None:
        return False
    if is_likely_figure_or_table_caption(raw):
        return False
    if any(ch in compact for ch in "。；;！？!?：:"):
        return False

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
    digit_count = sum(ch.isdigit() for ch in compact)
    symbol_count = len(re.findall(r"[A-Za-z()（）/\\_+=:;,%~〜×xX.-]", compact))

    if re.fullmatch(r"[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?", compact):
        return True
    if re.search(r"\d+(?:mm|MM|cm|CM|kv|kV|V|A|m)", compact):
        return True
    if re.search(r"(?:×|x|X|~|〜|－|-)", compact) and cjk_count <= 10:
        return True
    if any(token in compact for token in ("中心线", "尺寸", "推荐", "最小布置", "侧视图", "平面图", "屏前通道")):
        return True
    if len(compact) <= 12 and cjk_count <= 10 and (digit_count > 0 or symbol_count >= 2):
        return True
    return False


def infer_docx_text_semantic_role(text: str) -> str:
    raw = (text or "").strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return "artifact"
    if is_likely_figure_or_table_caption(raw):
        return "caption"
    if is_likely_figure_callout_or_annotation(raw):
        return "figure_callout"
    if any(token in compact for token in ("明细表", "附注", "注：", "注:")):
        return "table_note"
    structured = detect_structured_heading(raw)
    if structured is not None and structured[0] in ("chapter", "section", "subsection"):
        return "heading"
    if is_heading_noise_blacklisted(raw) or is_nonsemantic_heading_artifact(raw):
        return "artifact"
    if is_low_confidence_heading_candidate(raw) and len(compact) <= 24:
        return "artifact"
    return "body"


def is_plausible_styled_heading(text: str, *, max_len: int = 32) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if is_likely_figure_or_table_caption(raw):
        return False
    if is_likely_figure_callout_or_annotation(raw):
        return False
    if is_nonsemantic_heading_artifact(raw):
        return False
    if detect_structured_heading(raw) is not None:
        return True
    if is_likely_plain_section_heading(raw):
        return True
    return is_likely_heading_title(raw, max_len=max_len)


def make_number_slug(prefix: str, number: str) -> str:
    normalized = re.sub(r"[^0-9]+", "_", (number or "").strip()).strip("_")
    if not normalized:
        normalized = "0"
    return f"{prefix}_{normalized}"


# ===== 结构检测 =====

def detect_structured_heading(text: str) -> Optional[Tuple[str, str, str, str]]:
    """Detect standard-style numbered structures.

    Returns (kind, number, display, slug), where kind is one of:
      - chapter: top-level section like "第1章 总则" or "1 总则"
      - section: mid-level section like "4.1 导线"
      - clause: granular clause like "4.1.1 ..."
    """

    t = (text or "").strip()
    if not t:
        return None

    m = _CHAPTER_HEADING_RE.match(t)
    if m:
        n = parse_chinese_or_arabic_int(m.group(1))
        if n is not None and n > 0:
            return ("chapter", str(n), t, f"chapter_{n}")

    m = _CN_SECTION_HEADING_RE.match(t)
    if m:
        n = parse_chinese_or_arabic_int(m.group(1))
        title = (m.group(2) or "").strip()
        if n is not None and n > 0 and is_likely_heading_title(title or t):
            return ("section", str(n), t, f"section_{n}")

    m = _CN_SUBSECTION_HEADING_RE.match(t)
    if m:
        n = parse_chinese_or_arabic_int(m.group(1))
        title = (m.group(2) or "").strip()
        if n is not None and n > 0 and is_likely_heading_title(title):
            return ("subsection", str(n), t, f"subsection_{n}")

    m = _NUMBERED_CLAUSE_RE.match(t)
    if m:
        number = (m.group(1) or "").strip()
        if number:
            return ("clause", number, t, make_number_slug("clause", number))

    m = _NUMBERED_SECTION_RE.match(t)
    if m:
        number = (m.group(1) or "").strip()
        title = (m.group(2) or "").strip()
        if number and title and is_likely_heading_title(title):
            return ("section", number, t, make_number_slug("section", number))

    m = _TOP_LEVEL_SECTION_RE.match(t)
    if m:
        number = (m.group(1) or "").strip()
        title = (m.group(2) or "").strip()
        if number and title and is_likely_heading_title(title):
            return ("chapter", number, t, f"chapter_{number}")

    return None


def detect_abbreviated_clause(
    text: str,
    *,
    chapter_number: str = "",
    section_number: str = "",
) -> Optional[Tuple[str, str, str, str]]:
    """Detect OCR-truncated clauses like '.0.1 ...' under a known chapter.

    Returns the same tuple format as detect_structured_heading.
    """

    t = (text or "").strip()
    chapter = (chapter_number or "").strip()
    section = (section_number or "").strip()
    if not t or not (chapter or section):
        return None

    dotted = t.startswith(".") or t.startswith("．")
    m = _ABBREVIATED_NUMBER_CHAIN_RE.match(t) if dotted else _CONTEXTUAL_NUMBER_CHAIN_RE.match(t)
    if not m:
        return None

    chain = normalize_number_chain(m.group(1) or "")
    if not chain:
        return None

    parts = [part for part in chain.split(".") if part]
    if not parts:
        return None

    full_number = ""
    if dotted:
        if len(parts) >= 2 and chapter:
            full_number = f"{chapter}.{chain}"
        elif section:
            full_number = f"{section}.{chain}"
    else:
        if section:
            section_parts = [part for part in section.split(".") if part]
            if len(parts) >= 2 and section_parts and parts[0] == section_parts[-1]:
                if len(section_parts) >= 2:
                    full_number = ".".join(section_parts[:-1] + parts)
                elif chapter:
                    full_number = f"{chapter}.{chain}"
            elif chapter and parts[0] != chapter:
                full_number = f"{chapter}.{chain}"
        elif chapter and len(parts) >= 2:
            full_number = f"{chapter}.{chain}"

    if not full_number:
        return None

    display = replace_leading_number_chain(t, full_number)
    return ("clause", full_number, display, make_number_slug("clause", full_number))


def detect_sequential_clause_start(text: str) -> Optional[Tuple[Optional[int], str]]:
    """Detect clause-start paragraphs in older standards documents.

    Returns (explicit_number, normalized_text).
    - explicit_number is an int when the source still exposes a leading ordinal like `8.` or `1-`.
    - explicit_number is None for OCR-truncated starts like `.配电装置应...`.
    """

    t = (text or "").strip()
    if not t or t.startswith("[IMAGE_REF:"):
        return None
    if re.match(r"^[（(]?\d+[）)]", t):
        return None
    if re.match(r"^[一二三四五六七八九十百千]+\s*[、,，.]", t):
        return None

    m = re.match(r"^(\d{1,3})\s*[-—.．、]\s*(.+)$", t)
    if m:
        number = int(m.group(1))
        rest = (m.group(2) or "").strip()
        if rest and not is_likely_heading_title(rest, max_len=20):
            return number, f"{number}. {rest}"

    m = re.match(r"^[.．]\s*(.+)$", t)
    if m:
        rest = (m.group(1) or "").strip()
        if rest:
            return None, rest

    return None


def detect_anchor(text: str) -> Optional[Tuple[str, int, str, str]]:
    """Detect granular anchors and return (kind, number, display, slug).

    kind:
      - article
      - appendix
      - appendix_table
    """

    t = (text or "").strip()
    if not t:
        return None

    m = _ARTICLE_RE.match(t)
    if m:
        n = parse_chinese_or_arabic_int(m.group(1))
        if n is not None and n > 0:
            return ("article", n, f"第{n}条", f"article_{n}")

    m = _APPENDIX_RE.match(t)
    if m:
        n = parse_chinese_or_arabic_int(m.group(2))
        if n is not None and n > 0:
            return ("appendix", n, f"{(m.group(1) or '附录').strip()}{n}", f"appendix_{n}")

    m = _APPENDIX_TABLE_RE.match(t)
    if m:
        n = parse_chinese_or_arabic_int(m.group(1))
        if n is not None and n > 0:
            return ("appendix_table", n, f"附表{n}", f"appendix_table_{n}")

    return None
