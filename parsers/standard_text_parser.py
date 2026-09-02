"""非条文条款型解析器（纯文本节标题 + 图注/标注）。"""

from typing import List

from models.entry_model import Entry
from parsing.docx_xml_parser import (
    iter_block_elements, collect_text_from_element, table_to_markdown,
    cell_to_markdown, paragraph_has_page_break, paragraph_page_break_flags,
)
from imaging.table_render import parse_markdown_table_rows

_iter_block_elements = iter_block_elements
_collect_text_from_element = collect_text_from_element
_table_to_markdown = table_to_markdown
_cell_to_markdown = cell_to_markdown
_paragraph_has_page_break = paragraph_has_page_break
_paragraph_page_break_flags = paragraph_page_break_flags
_parse_markdown_table_rows = parse_markdown_table_rows

from parsers.base_parser import BaseDocxParser, ParserContext
from parsers.rule_provisions_parser import (
    _detect_structured_heading,
)
from classification.text_classifier import (
    is_likely_figure_or_table_caption,
    is_likely_figure_callout_or_annotation,
    is_likely_heading_title,
    is_likely_plain_section_heading,
    is_low_confidence_heading_candidate,
    is_nonsemantic_heading_artifact,
)
_is_likely_figure_or_table_caption = is_likely_figure_or_table_caption
_is_likely_figure_callout_or_annotation = is_likely_figure_callout_or_annotation
_is_likely_heading_title = is_likely_heading_title
_is_likely_plain_section_heading = is_likely_plain_section_heading
_is_low_confidence_heading_candidate = is_low_confidence_heading_candidate
_is_nonsemantic_heading_artifact = is_nonsemantic_heading_artifact


class StandardTextParser(BaseDocxParser):

    @property
    def parser_name(self) -> str:
        return "非条文条款型解析器"

    def parse(self, ctx: ParserContext) -> List[Entry]:
        import re

        root = ctx.doc_root
        rid_to_file = ctx.rid_to_file
        images_dir = ctx.images_dir

        entries: List[Entry] = list(ctx.initial_entries)
        images_total = int(ctx.images_total_initial or 0)

        entry_position = max((int(e.position or 0) for e in entries), default=0)
        table_position = max((int(e.table_position or 0) for e in entries), default=0)

        current_entry_id = ""
        current_entry_unit = ""
        current_entry_title = ""
        current_entry_page = 1
        current_entry_lines: List[str] = []
        current_entry_image_uris: List[str] = []

        def _dedupe_keep_order(xs):
            seen_local = set()
            out_local = []
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
            nonlocal current_entry_lines, current_entry_image_uris

            lines = [l for l in current_entry_lines if (l or "").strip()]
            if not lines and not current_entry_image_uris:
                current_entry_lines = []
                current_entry_image_uris = []
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
            entry_position += 1
            entries.append(entry)

            current_entry_lines = []
            current_entry_image_uris = []

        current_page = 1

        body = root.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body")
        if body is None:
            return entries

        for el in _iter_block_elements(root):
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag

            if tag == "p":
                inc_before, inc_after = _paragraph_page_break_flags(el)
                if inc_before or inc_after:
                    _flush_text_entry()
                    if inc_before or inc_after:
                        current_page += 1

                text = _collect_text_from_element(el, rid_to_file=rid_to_file, images_dir=images_dir)
                stripped = (text or "").strip()
                if not stripped:
                    continue

                # 跳过非语义内容
                if _is_low_confidence_heading_candidate(stripped) and len(re.sub(r"\s+", "", stripped)) <= 24:
                    continue
                if _is_nonsemantic_heading_artifact(stripped):
                    continue

                # 结构化标题
                heading = _detect_structured_heading(stripped)
                if heading and heading[0] in ("chapter", "section"):
                    _flush_text_entry()
                    current_entry_lines.append(stripped)
                    current_entry_title = heading[2]
                    current_entry_id = f"{ctx.file_id}_{heading[0]}_{heading[1]}_{current_page}"
                    _flush_text_entry()
                    continue

                # 纯文本节标题
                if _is_likely_plain_section_heading(stripped):
                    _flush_text_entry()
                    current_entry_lines.append(stripped)
                    current_entry_title = stripped
                    current_entry_id = f"{ctx.file_id}_section_{current_page}_{entry_position}"
                    _flush_text_entry()
                    continue

                # 图注/标注：追加到当次 entry
                if _is_likely_figure_or_table_caption(stripped) or _is_likely_figure_callout_or_annotation(stripped):
                    current_entry_lines.append(stripped)
                    continue

                # 普通段落
                current_entry_lines.append(stripped)

            elif tag == "tbl":
                _flush_text_entry()
                table_position += 1
                md, rendered_rows, img_count = _table_to_markdown(el, rid_to_file)
                if not md.strip():
                    continue

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
                )
                entry_position += 1
                entries.append(entry)

        _flush_text_entry()
        return entries
