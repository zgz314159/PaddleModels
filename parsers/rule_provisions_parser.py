"""条文条款型解析器（"第X条/章/节"层级结构）。

从 build_kb_from_docx.py 中迁移的全部硬编码逻辑：
  - 正则模式库（第X条/章/节/附件/附表）
  - 中文数字转换、编号链标准化
  - 标题分类（_detect_structured_heading、_detect_anchor 等）
  - build_entries() 状态机
"""

import re
import unicodedata
from typing import Optional, Tuple, List, Set, Dict

from parsers.base_parser import BaseDocxParser, ParserContext

from models.entry_model import Entry
from parsing.docx_xml_parser import (
    iter_block_elements, collect_text_from_element, table_to_markdown,
    cell_to_markdown, paragraph_has_page_break, paragraph_page_break_flags,
)
from ai.gemini_client import (
    GEMINI_FIXED_MODEL, normalize_gemini_model_name, fix_table_with_gemini,
    gemini_table_fix,
)
from imaging.table_render import (
    render_table_image, parse_markdown_table_rows,
    extract_first_markdown_table_block,
)
from ai.deepseek_correction import (
    normalize_docx_semantic_fix, deepseek_correction_available,
    write_docx_correction_audit, _DOCX_IMAGE_REF_RE,
)
from classification.text_quality_gate import analyze_text_suspicion

# Backward compat aliases used in parser body
_iter_block_elements = iter_block_elements
_collect_text_from_element = collect_text_from_element
_table_to_markdown = table_to_markdown
_cell_to_markdown = cell_to_markdown
_paragraph_has_page_break = paragraph_has_page_break
_paragraph_page_break_flags = paragraph_page_break_flags
_render_table_image = render_table_image
_parse_markdown_table_rows = parse_markdown_table_rows
_extract_first_markdown_table_block = extract_first_markdown_table_block
_normalize_docx_semantic_fix = normalize_docx_semantic_fix
_deepseek_correction_available = deepseek_correction_available
_write_docx_correction_audit = write_docx_correction_audit
_normalize_gemini_model_name = normalize_gemini_model_name

# ===== 正则模式库（从 text_classifier 导入） =====
from classification.text_classifier import (
    _CH_NUM_MAP,
    _ARTICLE_RE, _APPENDIX_RE, _APPENDIX_TABLE_RE,
    _CHAPTER_HEADING_RE, _CN_SECTION_HEADING_RE, _CN_SUBSECTION_HEADING_RE,
    _TOP_LEVEL_SECTION_RE, _NUMBERED_SECTION_RE, _NUMBERED_CLAUSE_RE,
    _ABBREVIATED_NUMBER_CHAIN_RE, _CONTEXTUAL_NUMBER_CHAIN_RE,
    _HEADING_NOISE_EXACT, _HEADING_NOISE_PATTERNS, _NONSEMANTIC_HEADING_PATTERNS,
    parse_chinese_or_arabic_int, normalize_number_chain, replace_leading_number_chain,
    infer_implicit_section_number, normalize_heading_candidate_key,
    is_heading_noise_blacklisted, is_nonsemantic_heading_artifact,
    make_number_slug,
)
_parse_chinese_or_arabic_int = parse_chinese_or_arabic_int
_normalize_number_chain = normalize_number_chain
_replace_leading_number_chain = replace_leading_number_chain
_infer_implicit_section_number = infer_implicit_section_number
_normalize_heading_candidate_key = normalize_heading_candidate_key
_is_heading_noise_blacklisted = is_heading_noise_blacklisted
_is_nonsemantic_heading_artifact = is_nonsemantic_heading_artifact
from classification.text_classifier import is_low_confidence_heading_candidate
_is_low_confidence_heading_candidate = is_low_confidence_heading_candidate

# ===== 解析器专用的分类函数变体（与 text_classifier.py 行为不同） =====

def _is_likely_heading_title(title: str, *, max_len: int = 36) -> bool:
    t = re.sub(r"\s+", "", (title or "").strip())
    if not t or len(t) > max_len:
        return False
    if _is_nonsemantic_heading_artifact(title):
        return False
    if re.fullmatch(r"[0-9Oo口~〜\-—()（）_/\\.]+", t):
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", t))
    ascii_count = len(re.findall(r"[A-Za-z]", t))
    digit_count = len(re.findall(r"\d", t))
    if cjk_count == 0 and ascii_count + digit_count > 0 and cjk_count + ascii_count <= 6:
        return True
    if cjk_count < 2 and ascii_count <= 2 and digit_count > ascii_count and digit_count >= 2:
        return False
    return True

def _is_likely_plain_section_heading(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    if _is_nonsemantic_heading_artifact(text):
        return False
    compact = re.sub(r"\s+", "", raw)
    if len(compact) < 2 or len(compact) > 28:
        return False
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", compact))
    if cjk_count == 0 or cjk_count > 14:
        return False
    prefix = "第X节（" if re.match(r"^第.节（", compact) else None
    return True

def _is_likely_figure_or_table_caption(text: str) -> bool:
    raw = (text or "").strip()
    compact = re.sub(r"\s+", "", raw)
    if re.match(r"^[图表]\s*[0-9一二三四五六七八九十零〇\-—_.．]+$", compact):
        return True
    if re.match(r"^附表\s*[0-9一二三四五六七八九十零〇]", compact):
        return True
    return False

def _is_likely_caption_fragment(text: str, previous_text: str) -> bool:
    current = (text or "").strip()
    previous = (previous_text or "").strip()
    if not current or not previous:
        return False
    current_compact = re.sub(r"\s+", "", current)
    previous_compact = re.sub(r"\s+", "", previous)
    if len(current_compact) < 2 or len(current_compact) > 24:
        return False
    if any(ch in current_compact for ch in "。；;！？!?：:，,（）()[]【】"):
        return False
    if _detect_structured_heading(current) is not None:
        return False
    if _detect_anchor(current) is not None:
        return False
    previous_is_label_only = bool(re.match(r"^[图表]\s*[0-9一二三四五六七八九十零〇\-—_.．]+$", previous_compact))
    previous_is_caption_like = _is_likely_figure_or_table_caption(previous)
    if not previous_is_label_only and not previous_is_caption_like:
        return False
    if not _is_likely_heading_title(current, max_len=24):
        return False
    if any(token in current_compact for token in ("第", "章", "节", "条")):
        return False
    return True

def _is_likely_figure_callout_or_annotation(text: str) -> bool:
    raw = (text or "").strip()
    compact = re.sub(r"\s+", "", raw)
    if not compact:
        return False
    if len(compact) < 2 or len(compact) > 60:
        return False
    if _detect_structured_heading(raw) is not None:
        return False
    if _detect_anchor(raw) is not None:
        return False
    if _is_likely_figure_or_table_caption(raw):
        return False
    if any(kw in compact for kw in ("尺寸", "mm", "cm", "m㎡", "㎡", "kV", "单位", "平面图", "剖面图", "示意图")):
        return True
    if re.search(r"\d{2,}\s*[xX×]\s*\d{2,}", compact):
        return True
    if re.search(r"[\d\u00d7\u00d8\u0276\u0277][xX\u00d7\u0276\u0277\u0278\u02d3]", compact):
        return True
    return False

def _make_number_slug(prefix: str, number: str) -> str:
    normalized = re.sub(r"[^0-9]+", "_", (number or "").strip()).strip("_")
    if not normalized:
        normalized = "0"
    return f"{prefix}_{normalized}"

def _detect_structured_heading(text: str) -> Optional[Tuple[str, str, str, str]]:
    t = (text or "").strip()
    if not t:
        return None
    m = _CHAPTER_HEADING_RE.match(t)
    if m:
        n = _parse_chinese_or_arabic_int(m.group(1))
        if n is not None and n > 0:
            return ("chapter", str(n), t, f"chapter_{n}")
    m = _CN_SECTION_HEADING_RE.match(t)
    if m:
        n = _parse_chinese_or_arabic_int(m.group(1))
        title = (m.group(2) or "").strip()
        if n is not None and n > 0 and _is_likely_heading_title(title or t):
            return ("section", str(n), t, f"section_{n}")
    m = _CN_SUBSECTION_HEADING_RE.match(t)
    if m:
        n = _parse_chinese_or_arabic_int(m.group(1))
        title = (m.group(2) or "").strip()
        if n is not None and n > 0 and _is_likely_heading_title(title):
            return ("subsection", str(n), t, f"subsection_{n}")
    m = _NUMBERED_CLAUSE_RE.match(t)
    if m:
        number = (m.group(1) or "").strip()
        if number:
            return ("clause", number, t, _make_number_slug("clause", number))
    m = _NUMBERED_SECTION_RE.match(t)
    if m:
        number = (m.group(1) or "").strip()
        title = (m.group(2) or "").strip()
        if number and title and _is_likely_heading_title(title):
            return ("section", number, t, _make_number_slug("section", number))
    m = _TOP_LEVEL_SECTION_RE.match(t)
    if m:
        number = (m.group(1) or "").strip()
        title = (m.group(2) or "").strip()
        if number and title and _is_likely_heading_title(title):
            return ("chapter", number, t, f"chapter_{number}")
    return None

def _detect_anchor(text: str) -> Optional[Tuple[str, int, str, str]]:
    t = (text or "").strip()
    if not t:
        return None
    m = _ARTICLE_RE.match(t)
    if m:
        n = _parse_chinese_or_arabic_int(m.group(1))
        if n is not None and n > 0:
            return ("article", n, f"第{n}条", f"article_{n}")
    m = _APPENDIX_RE.match(t)
    if m:
        n = _parse_chinese_or_arabic_int(m.group(2))
        if n is not None and n > 0:
            return ("appendix", n, f"{(m.group(1) or '附录').strip()}{n}", f"appendix_{n}")
    m = _APPENDIX_TABLE_RE.match(t)
    if m:
        n = _parse_chinese_or_arabic_int(m.group(1))
        if n is not None and n > 0:
            return ("appendix_table", n, f"附表{n}", f"appendix_table_{n}")
    return None

def _detect_abbreviated_clause(text: str, ctx_flags) -> Optional[Tuple[str, str, str, str]]:
    t = (text or "").strip()
    if not t:
        return None
    dotted = bool(ctx_flags.get("preceding_is_numbered", False))
    m = _ABBREVIATED_NUMBER_CHAIN_RE.match(t) if dotted else _CONTEXTUAL_NUMBER_CHAIN_RE.match(t)
    if not m:
        return None
    chain = _normalize_number_chain(m.group(1) or "")
    parts = (chain or "").split(".")
    if len([p for p in parts if p.strip().isdigit()]) < 2:
        return None
    full_number = ".".join(p.strip() for p in parts if p.strip())
    display = _replace_leading_number_chain(t, full_number)
    return ("clause", full_number, display, _make_number_slug("clause", full_number))

def _detect_sequential_clause_start(text: str) -> Optional[Tuple[Optional[int], str]]:
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
        if rest and not _is_likely_heading_title(rest, max_len=20):
            return number, f"{number}. {rest}"
    m = re.match(r"^[.．]\s*(.+)$", t)
    if m:
        rest = (m.group(1) or "").strip()
        if rest:
            return None, rest
    return None


# ===== 解析器类 =====

class RuleProvisionsParser(BaseDocxParser):

    @property
    def parser_name(self) -> str:
        return "条文条款型解析器"

    def parse(self, ctx: ParserContext) -> List[Entry]:
        import time
        import json

        root = ctx.doc_root
        rid_to_file = ctx.rid_to_file
        images_dir = ctx.images_dir
        style_id_to_name = ctx.style_id_to_name

        unit_name = ""
        job_title = ""
        current_section_title = ""
        current_subsection_title = ""

        entries: List[Entry] = list(ctx.initial_entries)
        images_total = int(ctx.images_total_initial or 0)

        entry_position = max((int(e.position or 0) for e in entries), default=0)
        table_position = max((int(e.table_position or 0) for e in entries), default=0)

        current_anchor: Optional[Tuple[str, int, str, str]] = None
        current_chapter_number = ""
        current_section_number = ""
        current_sequential_clause_number = 0
        current_entry_id = ""
        current_entry_unit = ""
        current_entry_title = ""
        current_entry_page = 1
        current_entry_lines: List[str] = []
        current_entry_image_uris: List[str] = []
        current_entry_is_synthetic = False
        docx_correction_cache: Dict[str, Dict[str, object]] = {}
        docx_correction_budget = max(0, int(ctx.docx_correction_max_candidates or 0))
        docx_correction_enabled = bool(
            ctx.docx_semantic_correction and docx_correction_budget > 0 and _deepseek_correction_available()
        )
        synthetic_text_index = 0

        def _dedupe_keep_order(xs: List[str]) -> List[str]:
            seen_local: Set[str] = set()
            out_local: List[str] = []
            for x in xs:
                s = (x or "").strip()
                if not s:
                    continue
                if s in seen_local:
                    continue
                seen_local.add(s)
                out_local.append(s)
            return out_local

        def _flush_text_entry():
            nonlocal entry_position, table_position, entries, images_total
            nonlocal current_entry_id, current_entry_unit, current_entry_title, current_entry_page
            nonlocal current_entry_lines, current_entry_image_uris, current_entry_is_synthetic

            lines = [l for l in current_entry_lines if (l or "").strip()]
            if not lines and not current_entry_image_uris:
                current_entry_lines = []
                current_entry_image_uris = []
                current_entry_is_synthetic = False
                return

            content_markdown = "\n\n".join(lines).strip()
            if not current_entry_id:
                current_entry_id = f"{ctx.file_id}_line_{current_entry_page}_{entry_position}"

            current_entry_image_uris = _dedupe_keep_order(current_entry_image_uris)

            entry = Entry(
                entry_id=current_entry_id,
                unit_name=current_entry_unit or "",
                job_title=current_entry_title or "",
                content_markdown=content_markdown,
                content_normalized=content_markdown,
                page_number=current_entry_page,
                position=entry_position,
                kind="text",
                table_position=0,
                table_rows=[],
                image_uris=list(current_entry_image_uris),
            )

            if current_entry_is_synthetic:
                entry_position += 1
            else:
                entry_position += 1

            entries.append(entry)
            if ctx.assets_sync_callback:
                try:
                    ctx.assets_sync_callback(entry, images_total, ctx.file_id)
                except Exception:
                    pass

            current_entry_lines = []
            current_entry_image_uris = []
            current_entry_is_synthetic = False

        current_page = 1
        last_nonempty_paragraph_text = ""

        exported_table_images = ctx.exported_table_images
        manifest_table_images = ctx.manifest_table_images
        manifest_table_images_ordered = ctx.manifest_table_images_ordered

        body = root.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body")
        if body is None:
            return entries

        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        for el in _iter_block_elements(root):
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag

            # ---- 段落 ----
            if tag == "p":
                inc_before, inc_after = _paragraph_page_break_flags(el)
                if inc_before:
                    _flush_text_entry()
                    current_page += 1
                elif inc_after:
                    _flush_text_entry()
                    current_page += 1
                    pass

                text = _collect_text_from_element(el)

                # 跳过非语义标题
                if _is_low_confidence_heading_candidate(text) and len(re.sub(r"\s+", "", text)) <= 24:
                    last_nonempty_paragraph_text = text
                    continue

                is_caption_like = _is_likely_figure_or_table_caption(text)
                is_callout_like = _is_likely_figure_callout_or_annotation(text)

                if is_caption_like or is_callout_like:
                    if current_entry_lines:
                        current_entry_lines.append(text)
                    last_nonempty_paragraph_text = text
                    continue

                structured_heading = _detect_structured_heading(text)

                # 样式标题
                pPr = el.find("w:pPr", ns)
                style_id = None
                if pPr is not None:
                    pStyle = pPr.find("w:pStyle", ns)
                    if pStyle is not None:
                        style_id = pStyle.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val")
                if style_id:
                    style_name = style_id_to_name.get(style_id, "")
                    if "Heading" in style_id or any(kw in (style_name or "") for kw in ("标题", "目 录", "目录", "标题")):
                        if _is_likely_heading_title(text):
                            _flush_text_entry()
                            current_entry_lines.append(text)
                            current_entry_title = text
                            current_entry_id = f"{ctx.file_id}_heading_{current_page}_{entry_position}"
                            _flush_text_entry()
                            last_nonempty_paragraph_text = text
                            continue

                # 纯文本节标题
                if _is_likely_plain_section_heading(text) and not structured_heading:
                    _flush_text_entry()
                    current_entry_lines.append(text)
                    current_entry_title = text
                    current_entry_id = f"{ctx.file_id}_section_{current_page}_{entry_position}"
                    _flush_text_entry()
                    last_nonempty_paragraph_text = text
                    continue

                # 锚点
                anchor = _detect_anchor(text)
                if anchor is not None or (structured_heading is not None and structured_heading[0] == "clause"):
                    _flush_text_entry()
                    if anchor:
                        current_anchor = anchor
                    current_entry_lines.append(text)
                    current_entry_id = f"{ctx.file_id}_{current_anchor[3] if current_anchor else 'clause'}_{current_page}"
                    last_nonempty_paragraph_text = text
                    continue

                # 结构化标题
                if structured_heading:
                    kind = structured_heading[0]
                    _flush_text_entry()
                    if kind == "chapter":
                        current_chapter_number = structured_heading[1]
                        current_section_number = ""
                        current_entry_title = f"第{current_chapter_number}章"
                    elif kind == "section":
                        current_section_number = structured_heading[1]
                        if not current_section_number.startswith(f"{current_chapter_number}."):
                            current_section_number = _infer_implicit_section_number(current_chapter_number, current_section_number)
                        current_entry_title = f"{current_section_number}"
                    current_entry_lines.append(text)
                    current_entry_id = f"{ctx.file_id}_{kind}_{structured_heading[1]}_{current_page}"
                    _flush_text_entry()
                    last_nonempty_paragraph_text = text
                    continue

                # 普通段落
                current_entry_lines.append(text)
                last_nonempty_paragraph_text = text

            # ---- 表格 ----
            elif tag == "tbl":
                _flush_text_entry()
                table_position += 1
                md, rendered_rows, img_count = _table_to_markdown(el, rid_to_file)

                if not md.strip():
                    continue

                # 表格图片
                table_image_uri = None
                if table_position in exported_table_images:
                    table_image_uri = str(exported_table_images[table_position])
                elif manifest_table_images:
                    for p, paths in manifest_table_images.items():
                        if p == current_page and paths:
                            table_image_uri = str(paths[0])
                            break

                entry = Entry(
                    entry_id=f"{ctx.file_id}_table_{table_position}_{current_page}",
                    unit_name="",
                    job_title="",
                    content_markdown=md,
                    content_normalized=md,
                    page_number=current_page,
                    position=entry_position,
                    kind="table",
                    table_position=table_position,
                    table_rows=_parse_markdown_table_rows(md) or [],
                    image_uris=[],
                    table_image_uri=table_image_uri,
                )
                entry_position += 1
                entries.append(entry)

        _flush_text_entry()
        return entries
