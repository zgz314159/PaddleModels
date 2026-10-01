"""Phase 1.2 unit tests: unnumbered display heading + positional page marker.

Pure functions and hand-built Canonical IR only — no real PDF required.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.semantics.role_inference import (  # noqa: E402
    infer_text_structure_semantic_role,
    is_positional_page_marker,
    looks_like_unnumbered_display_heading,
    looks_like_pdf_numbered_heading,
)
from pipeline.semantic_projector import SemanticProjector  # noqa: E402
from pipeline.canonical_ir import CanonicalIR, DocPage, DocBlock, BBox  # noqa: E402

# Sample geometry from samples/103号(2).pdf page 2 (Phase 1.1 IR).
PAGE_W = 246.65
PAGE_H = 360.05
# tb bbox form (x, y, w, h) for role_inference display-heading checks
TITLE_BBOX_XYWH = (34.32, 63.98, 175.55, 21.95)
# fitz/xyxy form (x0, y0, x1, y1) for is_positional_page_marker / _is_artifact
FOOTER_BBOX_XYXY = (216.60, 320.34, 221.10, 332.45)
BODY_MID_BBOX_XYXY = (40.0, 180.0, 120.0, 194.0)

PAGE_CONTEXT = {
    "page_number": 2,
    "page_width": PAGE_W,
    "page_height": PAGE_H,
}


def _tb(text, bbox, *, font_size=10.0, is_bold=False, line_count=1):
    return {
        "text": text,
        "bbox": bbox,
        "line_count": line_count,
        "avg_line_len": float(len(text)),
        "is_bold": is_bold,
        "font_size": font_size,
    }


class TestUnnumberedDisplayHeading(unittest.TestCase):
    def test_target_title_is_heading_with_context(self):
        tb = _tb("铁路电力管理规则", TITLE_BBOX_XYWH, font_size=21.95, is_bold=False)
        self.assertTrue(looks_like_unnumbered_display_heading(tb, PAGE_CONTEXT))
        role = infer_text_structure_semantic_role(tb, PAGE_CONTEXT)
        self.assertEqual(role, "heading")

    def test_short_chinese_body_stays_body(self):
        tb = _tb(
            "规程另发单行本。",
            (40.0, 100.0, 120.0, 12.0),
            font_size=11.0,
        )
        self.assertFalse(looks_like_unnumbered_display_heading(tb, PAGE_CONTEXT))
        role = infer_text_structure_semantic_role(tb, PAGE_CONTEXT)
        self.assertEqual(role, "body")

    def test_org_header_not_heading(self):
        # 铁道部文件 — short/large but no title suffix / exact match.
        tb = _tb("铁道部文件", (80.0, 40.0, 80.0, 18.0), font_size=18.0, is_bold=True)
        self.assertFalse(looks_like_unnumbered_display_heading(tb, PAGE_CONTEXT))
        role = infer_text_structure_semantic_role(tb, PAGE_CONTEXT)
        self.assertNotEqual(role, "heading")

    def test_numbered_body_item_stays_body(self):
        tb = _tb("2用于变电所倒闸作业", (40.0, 100.0, 150.0, 12.0), font_size=11.0)
        role = infer_text_structure_semantic_role(tb, PAGE_CONTEXT)
        self.assertEqual(role, "body")
        self.assertFalse(looks_like_unnumbered_display_heading(tb, PAGE_CONTEXT))

    def test_existing_numbered_heading_still_heading(self):
        tb = _tb("5.1一般规定", (40.0, 50.0, 80.0, 12.0), font_size=12.0)
        self.assertTrue(looks_like_pdf_numbered_heading("5.1一般规定"))
        role = infer_text_structure_semantic_role(tb, PAGE_CONTEXT)
        self.assertEqual(role, "heading")
        role_legacy = infer_text_structure_semantic_role(tb)  # no context
        self.assertEqual(role_legacy, "heading")

    def test_no_page_context_keeps_legacy_body_for_title(self):
        # Without page_context, prose-continuation still wins (Phase 1.1 behavior).
        tb = _tb("铁路电力管理规则", TITLE_BBOX_XYWH, font_size=21.95)
        self.assertFalse(looks_like_unnumbered_display_heading(tb, None))
        role = infer_text_structure_semantic_role(tb)
        self.assertEqual(role, "body")

    def test_title_below_half_page_not_heading(self):
        tb = _tb("铁路电力管理规则", (34.0, 300.0, 175.0, 21.0), font_size=21.95)
        self.assertFalse(looks_like_unnumbered_display_heading(tb, PAGE_CONTEXT))


class TestPositionalPageMarker(unittest.TestCase):
    def test_footer_digit_is_page_marker(self):
        self.assertTrue(
            is_positional_page_marker("2", FOOTER_BBOX_XYXY, PAGE_W, PAGE_H)
        )

    def test_body_digit_not_page_marker(self):
        self.assertFalse(
            is_positional_page_marker("2", BODY_MID_BBOX_XYXY, PAGE_W, PAGE_H)
        )

    def test_decorated_marker_anywhere(self):
        self.assertTrue(
            is_positional_page_marker("·2·", BODY_MID_BBOX_XYXY, PAGE_W, PAGE_H)
        )
        self.assertTrue(
            is_positional_page_marker("第2页", BODY_MID_BBOX_XYXY, PAGE_W, PAGE_H)
        )
        self.assertTrue(
            is_positional_page_marker("Page 2", BODY_MID_BBOX_XYXY, PAGE_W, PAGE_H)
        )
        self.assertTrue(
            is_positional_page_marker("-2-", BODY_MID_BBOX_XYXY, PAGE_W, PAGE_H)
        )

    def test_top_edge_digit_is_page_marker(self):
        top_bbox = (100.0, 5.0, 110.0, 15.0)
        self.assertTrue(is_positional_page_marker("3", top_bbox, PAGE_W, PAGE_H))


class TestProjectorAndIR(unittest.TestCase):
    def test_heading_triggers_new_entry(self):
        ir = CanonicalIR(document_id="doc", sha256="0" * 64)
        page = DocPage(page_number=1, width=PAGE_W, height=PAGE_H, method="native")
        page.blocks.append(
            DocBlock(
                id="p1_b1",
                type="text",
                text="正文段落开始这里是较长的一段说明文字内容",
                bbox=BBox(40, 100, 200, 40),
                page_number=1,
                reading_order=1,
                source="native",
                metadata={"semanticRole": "body"},
            )
        )
        page.blocks.append(
            DocBlock(
                id="p1_b2",
                type="heading",
                text="铁路电力管理规则",
                bbox=BBox(34, 64, 175, 22),
                page_number=1,
                reading_order=2,
                source="native",
                metadata={"semanticRole": "heading"},
            )
        )
        ir.pages.append(page)
        kb = SemanticProjector("doc", strategy="heading").project(ir)
        self.assertEqual(len(kb["entries"]), 2)
        self.assertEqual(kb["entries"][1]["jobTitle"], "铁路电力管理规则")
        self.assertEqual(kb["entries"][1]["pageNumber"], 1)

    def test_filtered_page_marker_absent_from_ir(self):
        # Simulate NativeAdapter after positional filter: no "2" block on page.
        ir = CanonicalIR(document_id="doc", sha256="0" * 64)
        page = DocPage(page_number=2, width=PAGE_W, height=PAGE_H, method="native")
        page.blocks.append(
            DocBlock(
                id="p2_b1",
                type="heading",
                text="铁路电力管理规则",
                bbox=BBox(*TITLE_BBOX_XYWH),
                page_number=2,
                reading_order=1,
                source="native",
                metadata={"semanticRole": "heading"},
            )
        )
        ir.pages.append(page)
        texts = [b.text for b in ir.pages[0].blocks]
        self.assertNotIn("2", texts)
        self.assertTrue(
            is_positional_page_marker("2", FOOTER_BBOX_XYXY, PAGE_W, PAGE_H),
            "footer 2 must be filtered before DocBlock creation",
        )
        kb = SemanticProjector("doc", strategy="heading").project(ir)
        self.assertEqual(len(kb["entries"]), 1)
        content = kb["entries"][0].get("contentMarkdown") or ""
        self.assertNotIn("\n2", content)
        self.assertFalse(content.strip().endswith("2") and content.strip() == "2")


if __name__ == "__main__":
    unittest.main()
