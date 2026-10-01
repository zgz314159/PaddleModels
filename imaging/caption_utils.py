import re
import logging
import io
from typing import List, Dict, Any, Optional, Set, Tuple
from PIL import Image
from imaging.ocr_engine import ocr_image_bytes

logger = logging.getLogger(__name__)

VISUAL_CAPTION_BODY_CUES = (
    '截止', '其中', '由于', '为保证', '通过', '沿线', '共有', '说明', '检查',
    '普速', '高铁', '当', '传统', '一般应用', '其特点', '特点是', '需要强调',
    '当电力', '系指',
)

VISUAL_CAPTION_SPLIT_RE = re.compile(
    r'([①②③④⑤⑥⑦⑧⑨⑩]\s*[\u4e00-\u9fff]|（[一二三四五六七八九十]+）\s*[\u4e00-\u9fff]|[一二三四五六七八九十]+\s*[、,，.．]\s*[\u4e00-\u9fff]|(?<![\d-])[0-9]{1,2}\.\s*[\u4e00-\u9fff]|第\s*[0-9一二三四五六七八九十百千〇零两]+\s*[篇章节])'
)

FIGURE_CALLOUT_CAPTION_SUFFIXES = (
    '放大图', '结构图', '实物图', '示意图', '接线图', '原理图', '剖面图',
)

def visual_label_matches_type(label: str, visual_type: str) -> bool:
    raw = str(label or '').strip()
    vt = str(visual_type or '').strip().lower()
    if vt == 'table':
        return raw.startswith(('表', '附表'))
    return raw.startswith(('图', '附图'))

def find_visual_caption_split_index(tail: str) -> Optional[int]:
    if not tail: return None
    split_index: Optional[int] = None
    for cue in VISUAL_CAPTION_BODY_CUES:
        cue_index = tail.find(cue)
        if cue_index <= 0: continue
        if split_index is None or cue_index < split_index:
            split_index = cue_index

    marker_match = VISUAL_CAPTION_SPLIT_RE.search(tail)
    if marker_match is not None and marker_match.start() > 0:
        marker_index = marker_match.start()
        if split_index is None or marker_index < split_index:
            split_index = marker_index
    return split_index

def find_label_span(raw: str, label: str) -> Optional[Tuple[int, int]]:
    """Locate normalized label inside raw OCR text, tolerating inter-char spaces.

    extract_figure_table_labels returns whitespace-stripped keys (图12), while
    raw OCR often reads '图 12'. Returns (start, end) in raw or None.
    """
    if not raw or not label:
        return None
    idx = raw.find(label)
    if idx >= 0:
        return idx, idx + len(label)
    # Whitespace-tolerant regex built from each label character.
    pattern = r"\s*".join(re.escape(ch) for ch in label)
    try:
        m = re.search(pattern, raw)
    except re.error:
        m = None
    if m:
        return m.start(), m.end()
    # Compact fallback with index mapping.
    compact = re.sub(r"\s+", "", raw)
    cidx = compact.find(label)
    if cidx < 0:
        return None
    ci = 0
    for i, ch in enumerate(raw):
        if ch.isspace():
            continue
        if ci == cidx:
            seen = 0
            j = i
            while j < len(raw) and seen < len(label):
                if not raw[j].isspace():
                    seen += 1
                j += 1
            return i, j
        ci += 1
    return None


def extract_visual_caption_candidate(text: str, visual_type: str, label_hits_fn) -> str:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw: return ''

    hits = [hit for hit in label_hits_fn(raw) if visual_label_matches_type(hit, visual_type)]
    if not hits: return ''

    label = hits[0]
    span = find_label_span(raw, label)
    if span is None: return ''
    label_index, label_end = span
    if label_index > 4: return ''

    tail = raw[label_end:].strip()
    if not tail: return label

    split_index = find_visual_caption_split_index(tail)
    if split_index is not None:
        title = tail[:split_index].strip(' ，,；;：:')
        if title:
            return f'{label} {title}'.strip()
        return label

    if len(re.sub(r'\s+', '', raw)) <= 36 and not re.search(r'[。；;！？!?]', raw):
        return raw
    return label

def extract_visual_caption_candidates_from_text(text: str, visual_type: str, label_hits_fn, primary_label_fn) -> Dict[str, str]:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw: return {}

    hits = [hit for hit in label_hits_fn(raw) if visual_label_matches_type(hit, visual_type)]
    if not hits: return {}

    candidates: Dict[str, str] = {}
    for index, label in enumerate(hits):
        label_index = raw.find(label)
        if label_index < 0: continue
        next_index = len(raw)
        if index + 1 < len(hits):
            next_label = hits[index + 1]
            found_next = raw.find(next_label, label_index + len(label))
            if found_next > label_index:
                next_index = found_next
        segment = raw[label_index:next_index].strip()
        candidate = extract_visual_caption_candidate(segment, visual_type, label_hits_fn)
        if not candidate: continue
        primary = primary_label_fn(candidate, visual_type)
        if primary: candidates[primary] = candidate
    return candidates

def extract_figure_callout_caption_candidate(text: str, label_hits_fn) -> str:
    raw = re.sub(r'\s+', ' ', str(text or '').strip())
    if not raw or len(raw) > 40: return ''
    if label_hits_fn(raw):
        return extract_visual_caption_candidate(raw, 'figure', label_hits_fn)

    compact = re.sub(r'\s+', '', raw)
    if not compact: return ''
    if any(compact.endswith(suffix) for suffix in FIGURE_CALLOUT_CAPTION_SUFFIXES):
        return raw
    return ''

def extract_embedded_visual_caption_from_image_bytes(
    image_bytes: bytes,
    visual_type: str,
    ocr_engine: str = 'auto',
    repeated_watermark_candidates: Optional[Set] = None,
    label_hits_fn = None
) -> str:
    if not image_bytes: return ''
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception: return ''

    width, height = img.size
    if width <= 0 or height <= 0: return ''

    for top_ratio in (0.68, 0.58):
        top = int(max(0, min(height - 1, round(height * top_ratio))))
        if top >= height - 8: continue
        try:
            band = img.crop((0, top, width, height))
            if band.width <= 4 or band.height <= 4: continue
            if band.width < 960:
                band = band.resize((band.width * 2, band.height * 2), Image.LANCZOS)
            buf = io.BytesIO()
            band.save(buf, format='PNG', optimize=True)
            ocr_text = ocr_image_bytes(buf.getvalue(), ocr_engine, repeated_watermark_candidates)
            if not ocr_text: continue
        except Exception: continue
        
        if label_hits_fn:
            candidate = extract_visual_caption_candidate(ocr_text, visual_type, label_hits_fn)
            if candidate: return candidate
    return ''
