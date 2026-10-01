import re
import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from imaging.text_utils import (
    safe_str, 
    normalize_anchor_key, 
    figure_table_label_variants
)

logger = logging.getLogger(__name__)

# Constants and RE patterns
CN_NUM_MAP = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
}

PDF_CHAPTER_RE = re.compile(r'^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*章(?:\s+|[：:]|$)(.*)$')
PDF_PART_RE = re.compile(r'^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*篇(?:\s+|[：:]|$)(.*)$')
PDF_SECTION_RE = re.compile(r'^第\s*([0-9一二三四五六七八九十百千〇零两]+)\s*节\s*(.*)$')
PDF_SUBSECTION_RE = re.compile(r'^([一二三四五六七八九十百千]+)\s*[、,，.]\s*(.+)$')
PDF_APPENDIX_RE = re.compile(r'^附录\s*([0-9A-Za-z一二三四五六七八九十百千〇零两]+)\s*(.*)$')
PDF_TOP_LEVEL_RE = re.compile(r'^([0-9]{1,2})(?![）\)\-—])\s+([\u4e00-\u9fff][^。；;！？!?：:|]{0,20})$')

FIGURE_DIAGRAM_LEGEND_ONLY_RE = re.compile(
    r'^(?:[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?)\s*[-—一.:：]\s*[\u4e00-\u9fffA-Za-z]{1,16}(?:[；;，,、\s]+(?:[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?)\s*[-—一.:：]\s*[\u4e00-\u9fffA-Za-z]{1,16})*$'
)
CABLE_DIAGRAM_LEGEND_ONLY_RE = re.compile(
    r'^(?:[a-zA-Z0-9.\-]+)\s*[-—一.:：]\s*[\u4e00-\u9fffA-Za-z0-9]{1,24}$'
)

PDF_TOC_DOT_LEADER_RE = re.compile(r'[.．…]{6,}')
PDF_HEADING_MARKER_RE = re.compile(r'(第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节]|附录\s*[0-9A-Za-z一二三四五六七八九十百千〇零两]+)')
PDF_INLINE_HEADING_START_RE = re.compile(
    r'(第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节]|'
    r'附录\s*[0-9A-Za-z一二三四五六七八九十百千〇零两]+|'
    r'[一二三四五六七八九十百0-9]+\s*、)'
)

def parse_cn_or_digit_int(text: str) -> Optional[int]:
    raw = (text or '').strip()
    if not raw: return None
    if re.fullmatch(r'\d+', raw):
        try: return int(raw)
        except Exception: return None
    
    total, num = 0, 0
    unit_map = {'十': 10, '百': 100, '千': 1000}
    seen = False
    for ch in raw:
        if ch in CN_NUM_MAP:
            num = CN_NUM_MAP[ch]
            seen = True
            continue
        if ch in unit_map:
            seen = True
            unit = unit_map[ch]
            total += (num if num != 0 else 1) * unit
            num = 0
            continue
        return None
    return (total + num) if seen else None

def is_short_heading_title(text: str, max_len: int = 40) -> bool:
    t = re.sub(r'\s+', '', str(text or '').strip())
    if not t or len(t) > max_len:
        return False
    if t[0] in '-—,.，;；:：':
        return False
    if any(ch in t for ch in '。；;！？!?：:'):
        return False
    return True

def strip_pdf_page_prefix(text: str) -> str:
    raw = str(text or '').strip()
    if not raw: return ''
    compact = re.sub(r'\s+', '', raw)
    if re.match(r'^\d{1,4}第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节]', compact):
        return re.sub(r'^\s*\d{1,4}\s*', '', raw)
    if re.match(r'^\d{1,4}附录', compact):
        return re.sub(r'^\s*\d{1,4}\s*', '', raw)
    return raw

def split_inline_pdf_heading_segments(text: str) -> List[str]:
    normalized = re.sub(r'\s+', ' ', str(text or '').strip())
    if not normalized: return []
    matches = list(PDF_INLINE_HEADING_START_RE.finditer(normalized))
    if len(matches) < 2: return [normalized]

    segments: List[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        segment = normalized[start:end].strip(' |/，,；;')
        if segment: segments.append(segment)
    return segments or [normalized]

def detect_pdf_heading(line: str) -> Optional[Tuple]:
    t = strip_pdf_page_prefix(line)
    if not t: return None
    
    m = PDF_PART_RE.match(t)
    if m:
        n = parse_cn_or_digit_int(m.group(1))
        if n is not None and n > 0: return ('part', str(n), t)
        
    m = PDF_CHAPTER_RE.match(t)
    if m:
        n = parse_cn_or_digit_int(m.group(1))
        if n is not None and n > 0: return ('chapter', str(n), t)
        
    m = PDF_SECTION_RE.match(t)
    if m:
        n = parse_cn_or_digit_int(m.group(1))
        title = str(m.group(2) or '').strip()
        if n is not None and n > 0 and is_short_heading_title(title or t):
            return ('section', str(n), t)
            
    m = PDF_SUBSECTION_RE.match(t)
    if m:
        n = parse_cn_or_digit_int(m.group(1))
        title = str(m.group(2) or '').strip()
        if n is not None and n > 0 and is_short_heading_title(title):
            return ('subsection', str(n), t)
            
    m = PDF_APPENDIX_RE.match(t)
    if m:
        title = str(m.group(2) or '').strip()
        if is_short_heading_title(title or t, max_len=24):
            return ('appendix', str(m.group(1) or '').strip(), t)
            
    m = PDF_TOP_LEVEL_RE.match(t)
    if m:
        title = str(m.group(2) or '').strip()
        compact_title = re.sub(r'\s+', '', title)
        if (is_short_heading_title(title, max_len=18) and '图' not in title and '表' not in title
            and not re.search(r'[A-Za-z0-9|]', compact_title)):
            return ('chapter', str(m.group(1) or '').strip(), t)
            
    return None

def looks_like_pdf_table_of_contents(text: str, lines: List[str]) -> bool:
    raw = str(text or '').strip()
    if not raw: return False
    compact = re.sub(r'\s+', '', raw)
    if '目录' in compact or '目次' in compact: return True

    dot_leaders = len(PDF_TOC_DOT_LEADER_RE.findall(raw))
    heading_markers = len(PDF_HEADING_MARKER_RE.findall(raw))
    page_number_tokens = len(re.findall(r'(?<!\d)\d{1,4}(?!\d)', raw))
    if heading_markers >= 4 and page_number_tokens >= 4 and not re.search(r'[。；;！？!?]', raw):
        return True
    return dot_leaders >= 2 and heading_markers >= 3

def looks_like_pdf_chapter_opener(lines: List[str]) -> bool:
    meaningful_lines = [str(line or '').strip() for line in lines if str(line or '').strip()]
    if not meaningful_lines or len(meaningful_lines) > 4: return False
    heading_like_count = 0
    for line in meaningful_lines:
        stripped = strip_pdf_page_prefix(line)
        compact = re.sub(r'\s+', '', stripped)
        if re.fullmatch(r'\d{1,4}', compact): continue
        if detect_pdf_heading(compact) or detect_pdf_heading(stripped):
            heading_like_count += 1
            continue
        return False
    return heading_like_count >= 1

def parse_top_level_item_number(text: str) -> Optional[int]:
    """Parse leading digit from a numbered list item (e.g. '1. ' -> 1)."""
    match = re.match(r"^\s*(\d+)\s*[\.．、•·]", str(text or '').strip())
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None

def physical_page_from_asset_uri(uri: str) -> Optional[int]:
    """Extract page number from URI pattern like '_p12_'."""
    s = str(uri or "").strip()
    if not s:
        return None
    match = re.search(r"_p(\d+)_", s, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None

def parse_page_number(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if not s:
        return None
    # match pure digits
    if re.fullmatch(r"\d+", s):
        return int(s)
    # match 'p12'
    match = re.match(r"^p(\d+)$", s, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def collect_entry_pages(entry: Dict[str, Any]) -> List[int]:
    """Collect all unique page numbers mentioned in an entry's blocks or metadata."""
    out: Set[int] = set()
    
    # Root level
    pn = parse_page_number(entry.get("pageNumber"))
    if pn: out.add(pn)
    
    # Blocks
    blocks = entry.get("blocks")
    if isinstance(blocks, list):
        for b in blocks:
            if not isinstance(b, dict): continue
            bpn = parse_page_number(b.get("pageNumber"))
            if bpn: out.add(bpn)
            
            uri = b.get("imageUri") or b.get("image_uri") or b.get("src")
            if uri:
                upn = physical_page_from_asset_uri(uri)
                if upn: out.add(upn)
                
    # Figure Nodes
    nodes = entry.get("figureNodes")
    if isinstance(nodes, list):
        for n in nodes:
            if not isinstance(n, dict): continue
            for p in (n.get("pageNumbers") or []):
                ppn = parse_page_number(p)
                if ppn: out.add(ppn)
    
    return sorted(list(out))

def entry_dominant_page(entry: Dict[str, Any]) -> Optional[int]:
    """Find the most frequently occurring page number in the entry."""
    pages = collect_entry_pages(entry)
    if not pages:
        return None
    if len(pages) == 1:
        return pages[0]
        
    counts: Dict[int, int] = {}
    blocks = entry.get("blocks") or []
    for b in blocks:
        if not isinstance(b, dict): continue
        pn = parse_page_number(b.get("pageNumber"))
        if pn: counts[pn] = counts.get(pn, 0) + 1
        
        uri = b.get("imageUri") or b.get("image_uri") or b.get("src")
        if uri:
            upn = physical_page_from_asset_uri(uri)
            if upn: counts[upn] = counts.get(upn, 0) + 1
            
    if not counts:
        return pages[0]
        
    # sort by frequency desc, then page number asc
    sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return sorted_counts[0][0]

def entry_page_hint(entry: Dict[str, Any]) -> Optional[int]:
    """Best-effort guess for the primary page number of an entry."""
    pn = parse_page_number(entry.get("pageNumber"))
    if pn and pn > 0:
        return pn
    dom = entry_dominant_page(entry)
    if dom and dom > 0:
        return dom
    pages = collect_entry_pages(entry)
    return min(pages) if pages else None

def parse_visual_snapshot_page_idx(uri: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse (page, index) from a visual snapshot filename like 'visual_p12_3.png'."""
    name = basename_from_uri(uri)
    match = re.search(r"visual_p(\d+)_(\d+)", name, flags=re.IGNORECASE)
    if not match:
        return (None, None)
    try:
        return (int(match.group(1)), int(match.group(2)))
    except Exception:
        return (None, None)

def basename_from_uri(uri: str) -> str:
    """Extract file basename from a URI or path, stripping query parameters."""
    s = str(uri or "").strip()
    if not s:
        return ""
    base = os.path.basename(s)
    if "?" in base:
        base = base.split("?")[0]
    return base

def resolve_manifest_local_path(manifest: Dict[str, Any], item: Dict[str, Any]) -> str:
    """Resolve the local file path for an item described in the manifest."""
    out_dir = str(manifest.get("outDir") or "").strip()
    out_file = str(item.get("outFile") or "").strip()
    if out_dir and out_file:
        candidate = os.path.join(out_dir, out_file)
        if os.path.exists(candidate):
            return candidate

    # Fallback to assetUri basename
    uri = ""
    for key in ("assetUri", "asset_uri", "imageUri", "image_uri", "src"):
        val = item.get(key)
        if val:
            uri = str(val).strip()
            break
    
    base = basename_from_uri(uri)
    if out_dir and base:
        candidate = os.path.join(out_dir, base)
        if os.path.exists(candidate):
            return candidate
    return ""

def entry_text_lines_for_style_detection(entry: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    seen: Set[str] = set()
    preserved_lines = entry.get('_styleDetectionLines')
    if isinstance(preserved_lines, list):
        for raw_line in preserved_lines:
            ln = str(raw_line or '').strip()
            if ln and ln not in seen:
                lines.append(ln)
                seen.add(ln)
        return lines
    
    text = str(entry.get('contentNormalized') or entry.get('contentMarkdown') or '').strip()
    if text:
        for ln in text.splitlines():
            ln = ln.strip()
            if ln and ln not in seen:
                lines.append(ln)
                seen.add(ln)
    return lines

def is_figure_scope_title_fragment(text: str) -> bool:
    """Detect if a text fragment looks like a figure/table scope title (e.g. '图1-1' part)."""
    raw = str(text or '').strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact or len(compact) > 24:
        return False
    if re.match(r"^(第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[章节]|[一二三四五六七八九十]+\s*[、,，.])", raw):
        return False
    if re.match(r"^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+", compact):
        return True
    if any(ch in compact for ch in "。；;！？!?：:"):
        return False
    if re.fullmatch(r"[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?", compact):
        return True
    if re.search(r"\d+(?:mm|MM|cm|CM|kv|kV|V|A|m)", compact):
        return True
    if any(token in compact for token in ("中心线", "尺寸", "推荐", "最小布置", "侧视图", "平面图", "屏前通道")):
        return True
    if re.search(r"(?:×|x|X|~|〜|－|-)", compact) and len(compact) <= 18:
        return True
    return False

def reference_label_score_in_text(text: str, label: str) -> int:
    """Score how likely a label is a cross-reference in the given text."""
    normalized_text = normalize_anchor_key(text)
    normalized_label = normalize_anchor_key(label)
    if not normalized_text or not normalized_label or normalized_label not in normalized_text:
        return 0

    if re.search(rf"(见|如|按).*{re.escape(normalized_label)}", normalized_text):
        return 3
    if re.search(rf"{re.escape(normalized_label)}.*(所示|规定|要求|安装|采用)", normalized_text):
        return 3
    if re.search(rf"{re.escape(normalized_label)}.*(示例图|示意图|接线图|布置图|安装图|尺寸图)", normalized_text):
        return 1
    return 0

def entry_reference_labels(entry: Dict[str, Any]) -> List[str]:
    """Extract all likely cross-reference labels (e.g. '图1-1') from an entry's text content."""
    labels: List[str] = []
    seen: Set[str] = set()
    texts: List[str] = []
    for field in ("contentMarkdown", "contentNormalized"):
        text = safe_str(entry.get(field)).strip()
        if text:
            texts.append(text)
    blocks = entry.get("blocks")
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            # Also check text/code blocks for labels
            if str(block.get("type") or "").strip().lower() in ("code", "text", "markdown"):
                text = safe_str(block.get("code") or block.get("text") or block.get("contentMarkdown")).strip()
                if text:
                    texts.append(text)
                    
    for text in texts:
        for label in figure_table_label_variants(text):
            if reference_label_score_in_text(text, label) < 2:
                continue
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
    return labels

def image_block_label_variants(block: Dict[str, Any], entry: Dict[str, Any]) -> List[str]:
    """Extract label variants from an image block's caption."""
    texts = [safe_str(block.get("caption")).strip()]
    out: List[str] = []
    seen: Set[str] = set()
    for text in texts:
        if not text: continue
        for label in figure_table_label_variants(text):
            if label in seen:
                continue
            seen.add(label)
            out.append(label)
    return out

def looks_like_figure_legend_text(text: str, top_level_item_fn = None) -> bool:
    """Detect if text looks like a figure legend (key-value mapping of parts)."""
    raw = str(text or '').strip()
    if not raw:
        return False
    # Simple normalization for length/token check
    compact = re.sub(r"[^\w\u4e00-\u9fa5]", "", raw).lower()
    if not compact or len(compact) > 120: # Slightly relaxed limit
        return False
    
    # If it starts with a top-level item number but is long, check for action verbs that suggest it's a sentence
    if top_level_item_fn and top_level_item_fn(raw) is not None and len(raw) >= 18:
        if any(token in raw for token in ("应", "不应", "应为", "采用", "装设", "考虑", "布置", "安装要求", "有关规定", "房间", "控制室", "变压器")):
            return False
            
    pair_matches = re.findall(
        r"(?:^|[；;，,、\s])(?:[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?)\s*[-—一.:：]\s*[\u4e00-\u9fffA-Za-z]{1,16}",
        raw,
    )
    if len(pair_matches) >= 2:
        return True
    if len(pair_matches) == 1 and any(sep in raw for sep in ("；", ";", "、", "，", ",")):
        return True
    
    has_leading_numbered_term = bool(re.match(r"^\s*\d+\s*[\.．、•·:：-]?\s*[\u4e00-\u9fffA-Za-z]{1,8}", raw))
    has_following_numbered_term = bool(re.search(r"[；;，,、:：]\s*\d+\s*[\.．、•·:：-]?\s*[\u4e00-\u9fffA-Za-z]{1,8}", raw))
    if has_leading_numbered_term and has_following_numbered_term and len(raw) <= 60 and not any(ch in raw for ch in ("见", "应", "不应")):
        return True
    return False

def is_pdf_page_marker_text(text: str) -> bool:
    """Detect if text is a simple page number (e.g. '1', '- 1 -', '第1页')."""
    raw = str(text or "").strip()
    if not raw:
        return False
    compact = re.sub(r"\s+", "", raw)
    # Pure digits
    if re.fullmatch(r"\d{1,4}", compact):
        return True
    # - 12 -
    if re.fullmatch(r"[-—一]\s*\d{1,4}\s*[-—一]", raw):
        return True
    # 第 12 页
    if re.fullmatch(r"第\s*\d{1,4}\s*页", compact):
        return True
    # p. 12 or Page 12
    if re.match(r"^(?:p\.|page)\s*\d{1,4}$", compact, flags=re.IGNORECASE):
        return True
    return False

def looks_like_pdf_numbered_heading(compact: str) -> bool:
    value = re.sub(r'\s+', '', str(compact or ''))
    if not value or len(value) > 24:
        return False
    if re.search(r'[。；;！？!?：:]|如图|如表', value):
        return False
    # Chapter/section headings in standards often appear as "5变、配电所" or "5.1一般规定".
    if re.match(r'^\d{1,2}(?:\.\d{1,2})?[\u4e00-\u9fff]', value):
        return True
    return False

def looks_like_pdf_numbered_body_item(compact: str) -> bool:
    value = re.sub(r'\s+', '', str(compact or ''))
    if not value:
        return False
    if looks_like_pdf_numbered_heading(value):
        return False
    # Standards commonly omit the space after list numbers: "1变、配电所..." / "2用于...".
    return bool(re.match(r'^\d{1,2}[\u4e00-\u9fff]', value))

def looks_like_pdf_prose_continuation_fragment(text: str) -> bool:
    raw = str(text or '').strip()
    compact = re.sub(r'\s+', '', raw)
    if not compact or len(compact) > 36:
        return False
    if is_pdf_page_marker_text(raw):
        return False
    if re.match(r'^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+', compact, flags=re.IGNORECASE):
        return False
    if FIGURE_DIAGRAM_LEGEND_ONLY_RE.match(compact) or CABLE_DIAGRAM_LEGEND_ONLY_RE.match(compact):
        return False
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', compact))
    symbol_count = len(re.findall(r'[A-Za-z0-9()（）/\\_+=:;,%~〜×xX.-]', compact))
    return cjk_count >= 6 and symbol_count == 0

def is_likely_pdf_callout_or_annotation(
    text: str,
    *,
    line_count: int = 0,
    avg_line_len: float = 0.0,
    bbox: Any = None,
) -> bool:
    raw = str(text or '').strip()
    compact = re.sub(r'\s+', '', raw)
    if not compact:
        return False
    if is_pdf_page_marker_text(raw):
        return True
    if len(compact) > 24:
        return False
    if re.match(r'^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+', compact, flags=re.IGNORECASE):
        return False
    if re.match(r'^第[一二三四五六七八九十百零〇0-9]+[章节]', compact):
        return False
    if re.match(r'^(?:第?[一二三四五六七八九十百零〇0-9]*)?条[）)]?$', compact):
        return False
    if re.search(r'第[一二三四五六七八九十百零〇0-9]+条[）)]?$', compact):
        return False
    if re.match(r'^[一二三四五六七八九十]+、', compact):
        return False
    if FIGURE_DIAGRAM_LEGEND_ONLY_RE.match(compact) or CABLE_DIAGRAM_LEGEND_ONLY_RE.match(compact):
        return True
    if any(ch in compact for ch in '。；;！？!?：:'):
        return False
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', compact))
    if cjk_count >= 4 and re.search(r'[》）)]$', compact):
        return False
    if re.fullmatch(r'[（(]?[A-Za-z0-9一二三四五六七八九十]+[)）]?', compact):
        return True
    if re.search(r'\d+(?:mm|MM|cm|CM|kv|kV|V|A|m)', compact):
        return True
    if any(token in compact for token in ('中心线', '尺寸', '推荐', '最小布置', '侧视图', '平面图', '屏前通道')):
        return True
    if re.search(r'(?:×|x|X|~|〜|－|-)', compact) and len(compact) <= 18:
        return True
    symbol_count = len(re.findall(r'[A-Za-z0-9()（）/\\_+=:;,%~〜×xX.-]', compact))
    
    width = 0
    if isinstance(bbox, (tuple, list)) and len(bbox) >= 3:
        width = int(bbox[2])
    elif isinstance(bbox, dict):
        width = int(bbox.get('width', bbox.get('w', 0)))

    if line_count <= 2 and avg_line_len <= 10.0 and cjk_count <= 10 and symbol_count >= 1:
        return True
    if width and width <= 240 and len(compact) <= 14 and cjk_count <= 10:
        return True
    return False

def infer_pdf_text_role(tb: Dict) -> str:
    try:
        text = str(tb.get('text') or '').strip()
        compact = re.sub(r'\s+', '', text)
        line_count = int(tb.get('line_count') or 0)
        avg_line_len = float(tb.get('avg_line_len') or 0.0)
        bbox = tb.get('bbox')
        
        if not compact:
            return 'artifact'
        
        if looks_like_pdf_table_of_contents(text, [text]):
            return 'heading'

        if re.match(r'^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+', compact, flags=re.IGNORECASE):
            if len(compact) >= 48 and re.search(
                r'(如图所示|一般应用|其特点|特点是|传统的|需要强调|当电力|系指)',
                compact,
            ):
                return 'body'
            return 'caption'
            
        if any(token in compact for token in ('示例图', '示意图', '安装图', '布置图', '接线图')):
            return 'caption'
            
        if looks_like_pdf_numbered_heading(compact):
            return 'heading'
            
        if looks_like_pdf_numbered_body_item(compact):
            return 'body'

        if re.match(r'^\s*[（(]?\d+[）)]', text) or re.match(r'^\s*•', text) or re.match(r'^\s*\d+\.\s', text):
            return 'body'

        if looks_like_pdf_prose_continuation_fragment(text):
            return 'body'
            
        if is_likely_pdf_callout_or_annotation(
            text,
            line_count=line_count,
            avg_line_len=avg_line_len,
            bbox=bbox,
        ):
            return 'figure_callout'
            
        if '明细表' in compact or compact in {'附注', '注', '注：', '注:'}:
            return 'table_note'
            
        if re.match(r'^第[一二三四五六七八九十百零〇0-9]+[章节]', compact):
            return 'heading'
            
        if re.match(r'^[一二三四五六七八九十]+、', compact):
            return 'heading'

        # Bold title detection
        if tb.get('isBold') and len(compact) <= 32 and not re.search(r'[。；;！？!?]', compact):
            if not re.match(r'^\d+\s*$', compact) and not is_likely_pdf_callout_or_annotation(
                text, 
                line_count=line_count, 
                avg_line_len=avg_line_len, 
                bbox=bbox
            ):
                return 'heading'

        cjk_count = len(re.findall(r'[\u4e00-\u9fff]', compact))
        if cjk_count == 0 and re.search(r'[A-Za-z0-9]', compact):
            return 'artifact'
        if re.search(r'^[A-Za-z0-9./\\_+=:;,%()\[\]×x-]+$', compact) and cjk_count < 2:
            return 'artifact'
    except Exception:
        return 'body'
    return 'body'
