import re
import hashlib
from typing import Dict, Optional, Tuple, List

# --- Constants moved from pdf_to_base64_kb.py ---
_FIGURE_DIAGRAM_LEGEND_ONLY_RE = re.compile(
    r'^(?:不接地系统\s*){1,2}'
    r'(?:经消弧线圈接地系统|经小电阻接地系统|直接接地系统)\s*$'
)
_CABLE_DIAGRAM_LEGEND_ONLY_RE = re.compile(
    r'^(?:单端接地|双端接地)(?:\s+(?:单端接地|双端接地))*\s*$'
)

def is_pdf_page_marker_text(text: str) -> bool:
    """Detects page numbers or footer artifacts."""
    raw = str(text or '').strip()
    compact = re.sub(r'\s+', '', raw)
    if not compact:
        return False
        
    # Page markers like "·7·" / "·8·".
    if re.fullmatch(r'[·•∙⋅・]\d{1,4}[·•∙⋅・]', compact):
        return True
        
    if re.fullmatch(r'[-—–_]*\d{1,4}[-—–_]*', compact):
        # Avoid classifying short list markers like "1" / "2" / "48" as page markers.
        if re.fullmatch(r'\d{1,2}', compact):
            return False
        return True
        
    # "Page 1", "第1页", "1/10"
    if re.fullmatch(r'(?:Page|第)?\d{1,4}(?:页|/\d{1,4})?', compact, flags=re.IGNORECASE):
         return True
         
    # Fallback to loose matches for "Page 1 of 10", etc.
    if re.match(r'^(?:Page|第)\s*\d+\s*(?:of|/|页)\s*\d*$', raw, re.IGNORECASE):
        return True
        
    return False

# Document-title suffixes for unnumbered display headings (non-exhaustive).
_DISPLAY_HEADING_SUFFIX_RE = re.compile(
    r'(规则|规程|办法|条例|标准|规范|导则|细则|总则|规定|制度|职责|预案|方案)$'
)
_DISPLAY_HEADING_EXACT = frozenset({
    '总则', '目录', '附录', '前言', '引言', '范围',
})


def looks_like_unnumbered_display_heading(
    tb: Dict,
    page_context: Optional[Dict] = None,
) -> bool:
    """True for short unnumbered document titles in the upper page with display type.

    Multi-evidence: short, no body ending punctuation, upper half, display size
    or bold, and a title-like ending (规则/规程/…). Never hard-codes a full title.
    """
    if not isinstance(tb, dict) or page_context is None:
        return False
    try:
        text = str(tb.get('text') or '').strip()
        compact = re.sub(r'\s+', '', text)
        if not compact or len(compact) > 24:
            return False
        if re.match(r'^(图|表)\s*[0-9一二三四五六七八九十零〇\-—_.．]+', compact, flags=re.IGNORECASE):
            return False
        if re.fullmatch(r'\d{1,4}', compact):
            return False
        # "2用于……" style body items — not headings.
        if re.match(r'^\d{1,2}[一-鿿]', compact) and not _DISPLAY_HEADING_SUFFIX_RE.search(compact):
            return False
        if re.match(r'^\s*[（(]?\d+[）)]', text) or re.match(r'^\s*[•·]', text):
            return False
        if re.search(r'[。；;！？!?]', compact):
            return False
        if looks_like_pdf_numbered_heading(compact):
            return False

        page_height = float(page_context.get('page_height') or 0)
        if page_height <= 0:
            return False
        bbox = tb.get('bbox') if isinstance(tb.get('bbox'), (tuple, list)) else None
        if not bbox or len(bbox) < 4:
            return False
        # tb bbox is (x, y, w, h) — same as NativeAdapter tb_info.
        y = float(bbox[1])
        h_box = float(bbox[3])
        y_center = y + h_box / 2.0
        if y_center > page_height * 0.5:
            return False

        is_bold = bool(tb.get('is_bold') or tb.get('isBold'))
        try:
            font_size = float(tb.get('font_size') or tb.get('fontSize') or 0)
        except (TypeError, ValueError):
            font_size = 0.0
        if not (font_size >= 14.0 or is_bold):
            return False

        if compact in _DISPLAY_HEADING_EXACT:
            return True
        if _DISPLAY_HEADING_SUFFIX_RE.search(compact):
            return True
        return False
    except Exception:
        return False


def is_positional_page_marker(
    text: str,
    bbox: Optional[Tuple[float, float, float, float]],
    page_width: float,
    page_height: float,
) -> bool:
    """True only for decorative/edge page numbers; body-area digits are kept.

    `bbox` is PDF-style (x0, y0, x1, y1) as passed from NativeAdapter/fitz.
    Pure digits count as page markers only in top/bottom edge bands.
    Decorated markers (·2·, -2-, 第2页, Page 2) remain markers anywhere.
    """
    raw = str(text or '').strip()
    compact = re.sub(r'\s+', '', raw)
    if not compact or len(compact) > 24:
        return False
    if not page_height or page_height <= 0:
        return False

    if re.fullmatch(r'[·•∙⋅・]\d{1,4}[·•∙⋅・]', compact):
        return True
    if re.fullmatch(r'[-—–_]+\d{1,4}[-—–_]+', compact):
        return True
    if re.fullmatch(r'第\s*\d{1,4}\s*页', compact):
        return True
    if re.match(r'^(?:p\.|page)\s*\d{1,4}$', compact, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r'\d{1,4}/\d{1,4}', compact):
        return True

    # Pure short digits: require edge position (top or bottom band).
    if re.fullmatch(r'\d{1,4}', compact):
        if not isinstance(bbox, (tuple, list)) or len(bbox) < 4:
            return False
        y0, y1 = float(bbox[1]), float(bbox[3])
        if y1 < y0:
            y0, y1 = y1, y0
        # Sample footer "2" sits near y≈320/360 ≈ 0.89; body digits sit mid-page.
        y_center = (y0 + y1) / 2.0
        if y_center <= page_height * 0.12:
            return True
        if y_center >= page_height * 0.85:
            return True
        return False

    return False


def infer_text_structure_semantic_role(
    tb: Dict,
    page_context: Optional[Dict] = None,
) -> str:
    """
    Analyzes a text block dictionary and infers its semantic role.

    Expected 'tb' keys (snake_case preferred; camelCase accepted):
      text, line_count, avg_line_len, bbox, is_bold / isBold, font_size / fontSize

    page_context:
      None → legacy-stable behavior (tools/pdf_to_base64_kb.py).
      {'page_number','page_width','page_height'} → enables unnumbered display
      heading detection before prose-continuation (v2 NativeAdapter path).
    """
    try:
        text = str(tb.get('text') or '').strip()
        compact = re.sub(r'\s+', '', text)
        line_count = int(tb.get('line_count') or 0)
        avg_line_len = float(tb.get('avg_line_len') or 0.0)
        bbox = tb.get('bbox') if isinstance(tb.get('bbox'), (tuple, list)) else None
        
        if not compact:
            return 'artifact'
            
        # Figures and Tables captions
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

        # Enhanced list item recognition
        if re.match(r'^\s*[（(]?\d+[）)]', text) or re.match(r'^\s*•', text) or re.match(r'^\s*\d+\.\s', text):
            return 'body'

        # v2-only (page_context present): unnumbered display headings BEFORE prose.
        if page_context is not None and looks_like_unnumbered_display_heading(tb, page_context):
            return 'heading'

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

        # Preservation logic for bold titles in standards
        is_bold_flag = bool(tb.get('is_bold') or tb.get('isBold'))
        if is_bold_flag and len(compact) <= 32 and not re.search(r'[。；;！？!?]', compact):
            # Exclude things that look like list items or figure callouts
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

def looks_like_pdf_numbered_heading(compact: str) -> bool:
    value = re.sub(r'\s+', '', str(compact or ''))
    if not value or len(value) > 24:
        return False
    if re.search(r'[。；;！？!?：:]|如图|如表', value):
        return False
    # Decimal section headings: "5.1一般规定"
    if re.match(r'^\d{1,2}\.\d{1,2}[一-鿿]', value):
        return True
    # Single-level: "5变、配电所" — but not clause body starters like "2用于…"
    if re.match(r'^\d{1,2}[一-鿿]', value):
        if re.match(
            r'^\d{1,2}(用于|使用|应|必须|须|要|可|的|在|是|有|包括|按照|根据|遇|当|对|由|为|和|与|或|不|未|已)',
            value,
        ):
            return False
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

def looks_like_pdf_attached_numbered_body_item(text: str) -> bool:
    value = str(text or '').strip()
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
    if _FIGURE_DIAGRAM_LEGEND_ONLY_RE.match(compact) or _CABLE_DIAGRAM_LEGEND_ONLY_RE.match(compact):
        return False
    cjk_count = len(re.findall(r'[\u4e00-\u9fff]', compact))
    symbol_count = len(re.findall(r'[A-Za-z0-9()（）/\\_+=:;,%~〜×xX.-]', compact))
    return cjk_count >= 6 and symbol_count == 0

def is_likely_pdf_callout_or_annotation(
    text: str,
    *,
    line_count: int = 0,
    avg_line_len: float = 0.0,
    bbox: Optional[tuple] = None,
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
    if _FIGURE_DIAGRAM_LEGEND_ONLY_RE.match(compact) or _CABLE_DIAGRAM_LEGEND_ONLY_RE.match(compact):
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
    width = int(bbox[2]) if isinstance(bbox, (tuple, list)) and len(bbox) >= 3 else 0
    if line_count <= 2 and avg_line_len <= 10.0 and cjk_count <= 10 and symbol_count >= 1:
        return True
    if width and width <= 240 and len(compact) <= 14 and cjk_count <= 10:
        return True
    return False
