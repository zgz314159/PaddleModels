"""Phase 1.3 unit tests: Semantic Projection searchability policy.

Hand-built Canonical IR only — no PDF dependency.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.semantic_projector import (  # noqa: E402
    SemanticProjector,
    classify_text_projection,
)
from pipeline.canonical_ir import CanonicalIR, DocPage, DocBlock, BBox  # noqa: E402

KB_SCHEMA = REPO_ROOT / "contracts" / "knowledge_base_schema_v2.json"


def _block(bid, text, role, *, page=1, ro=1, btype="text"):
    return DocBlock(
        id=bid,
        type=btype,
        text=text,
        bbox=BBox(10, 20, 100, 15),
        page_number=page,
        reading_order=ro,
        source="native",
        metadata={"semanticRole": role},
    )


def _ir(*pages_blocks, sha="a" * 64):
    """pages_blocks: sequence of (page_number, [DocBlock, ...])"""
    ir = CanonicalIR(document_id="doc", sha256=sha)
    for p_num, blocks in pages_blocks:
        ir.pages.append(
            DocPage(page_number=p_num, width=200.0, height=300.0, method="native", blocks=list(blocks))
        )
    return ir


def _project(blocks_by_page, strategy="heading"):
    ir = _ir(*blocks_by_page)
    return SemanticProjector("doc", strategy=strategy).project(kb_ir := ir) if False else SemanticProjector("doc", strategy=strategy).project(ir)


class TestClassifyTextProjection(unittest.TestCase):
    def test_searchable_roles(self):
        for role in ("heading", "body", "caption", "table_note"):
            self.assertEqual(classify_text_projection(role), "searchable", role)

    def test_supporting_roles(self):
        for role in ("figure_callout", "legend", "annotation", "ocr_annotation"):
            self.assertEqual(classify_text_projection(role), "supporting", role)

    def test_excluded_role(self):
        self.assertEqual(classify_text_projection("artifact"), "excluded")

    def test_unknown_and_missing_default_searchable(self):
        self.assertEqual(classify_text_projection(None), "searchable")
        self.assertEqual(classify_text_projection(""), "searchable")
        self.assertEqual(classify_text_projection("mystery_role"), "searchable")


class TestProjectionBehaviors(unittest.TestCase):
    def test_body_writes_content_and_searchable_true(self):
        kb = _project([(1, [
            _block("b1", "正文内容甲", "body", ro=1),
        ])])
        e = kb["entries"][0]
        self.assertIn("正文内容甲", e["contentMarkdown"])
        self.assertIn("正文内容甲", e["contentNormalized"])
        self.assertTrue(e["blocks"][0]["searchable"])
        self.assertEqual(e["blocks"][0]["semanticRole"], "body")

    def test_heading_creates_entry_and_content(self):
        kb = _project([(1, [
            _block("b1", "前言正文一段较长的内容说明", "body", ro=1),
            _block("b2", "铁路电力管理规则", "heading", ro=2, btype="heading"),
        ])])
        self.assertEqual(len(kb["entries"]), 2)
        e1 = kb["entries"][1]
        self.assertEqual(e1["jobTitle"], "铁路电力管理规则")
        self.assertIn("铁路电力管理规则", e1["contentMarkdown"])
        self.assertTrue(e1["blocks"][0]["searchable"])

    def test_figure_callout_kept_block_not_content(self):
        kb = _project([(1, [
            _block("c1", "铁道部文件", "figure_callout", ro=1),
            _block("b1", "关于发布的通知正文", "body", ro=2),
        ])])
        e = kb["entries"][0]
        self.assertNotIn("铁道部文件", e["contentMarkdown"])
        self.assertNotIn("铁道部文件", e["contentNormalized"])
        callouts = [b for b in e["blocks"] if b.get("semanticRole") == "figure_callout"]
        self.assertEqual(len(callouts), 1)
        self.assertEqual(callouts[0]["code"], "铁道部文件")
        self.assertFalse(callouts[0]["searchable"])
        self.assertIn("关于发布的通知正文", e["contentMarkdown"])

    def test_artifact_not_in_kb(self):
        kb = _project([(1, [
            _block("a1", "noise", "artifact", ro=1),
            _block("b1", "正文", "body", ro=2),
        ])])
        e = kb["entries"][0]
        roles = [b.get("semanticRole") for b in e["blocks"]]
        self.assertNotIn("artifact", roles)
        self.assertNotIn("noise", e["contentMarkdown"])
        # only body block
        self.assertEqual(len(e["blocks"]), 1)

    def test_supporting_alone_does_not_create_entry(self):
        kb = _project([(1, [
            _block("c1", "铁道部文件", "figure_callout", ro=1),
            _block("c2", "铁运[1999]103 号", "figure_callout", ro=2),
        ])])
        self.assertEqual(len(kb["entries"]), 0)

    def test_supporting_before_heading_attaches_to_new_entry(self):
        kb = _project([(1, [
            _block("c1", "铁道部文件", "figure_callout", ro=1),
            _block("c2", "铁运[1999]103 号", "figure_callout", ro=2),
            _block("h1", "铁路电力管理规则", "heading", ro=3, btype="heading"),
        ])])
        self.assertEqual(len(kb["entries"]), 1)
        e = kb["entries"][0]
        self.assertEqual(e["jobTitle"], "铁路电力管理规则")
        roles = [b.get("semanticRole") for b in e["blocks"]]
        self.assertIn("figure_callout", roles)
        self.assertIn("heading", roles)
        self.assertNotIn("铁道部文件", e["contentMarkdown"])
        self.assertIn("铁路电力管理规则", e["contentMarkdown"])

    def test_supporting_before_body_attaches_to_body_entry(self):
        kb = _project([(1, [
            _block("c1", "铁道部文件", "figure_callout", ro=1),
            _block("b1", "正文开始", "body", ro=2),
        ])], strategy="heading")
        self.assertEqual(len(kb["entries"]), 1)
        e = kb["entries"][0]
        self.assertIn("正文开始", e["contentMarkdown"])
        self.assertNotIn("铁道部文件", e["contentMarkdown"])
        self.assertTrue(any(b.get("semanticRole") == "figure_callout" for b in e["blocks"]))

    def test_supporting_only_page_no_empty_entry(self):
        kb = _project([
            (1, [
                _block("b1", "第一页正文内容足够长", "body", ro=1),
                _block("c1", "仅辅助", "figure_callout", ro=2, page=1),
            ]),
            (2, [
                _block("c2", "第二页只有callout", "figure_callout", ro=1, page=2),
            ]),
        ], strategy="heading")
        # Page 1 body entry exists; page 2 supporting-only does not add an entry.
        self.assertEqual(len(kb["entries"]), 1)
        self.assertEqual(kb["entries"][0]["pageNumber"], 1)

    def test_caption_is_searchable(self):
        kb = _project([(1, [
            _block("cap1", "图1 系统结构示意", "caption", ro=1),
        ])])
        e = kb["entries"][0]
        self.assertIn("图1", e["contentMarkdown"])
        self.assertTrue(e["blocks"][0]["searchable"])

    def test_unknown_role_defaults_searchable(self):
        kb = _project([(1, [
            _block("u1", "未知角色文本", "brand_new_role", ro=1),
        ])])
        e = kb["entries"][0]
        self.assertIn("未知角色文本", e["contentMarkdown"])
        self.assertTrue(e["blocks"][0]["searchable"])
        self.assertEqual(classify_text_projection("brand_new_role"), "searchable")

    def test_page_strategy_no_extra_entry_from_supporting(self):
        kb = _project([(1, [
            _block("c1", "callout先", "figure_callout", ro=1),
            _block("b1", "正文", "body", ro=2),
        ])], strategy="page")
        self.assertEqual(len(kb["entries"]), 1)
        e = kb["entries"][0]
        self.assertIn("正文", e["contentMarkdown"])
        self.assertNotIn("callout先", e["contentMarkdown"])
        self.assertTrue(any(b.get("semanticRole") == "figure_callout" for b in e["blocks"]))


class TestSchemaSearchable(unittest.TestCase):
    def test_searchable_non_boolean_fails_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        with open(KB_SCHEMA, "r", encoding="utf-8") as f:
            schema = json.load(f)
        bad = {
            "fileMetadata": {
                "schemaVersion": "2.0",
                "fileId": "t",
                "docSha256": "a" * 64,
            },
            "entries": [{
                "entryId": "e",
                "jobTitle": "t",
                "blocks": [{
                    "id": "b",
                    "type": "code",
                    "searchable": "yes",  # wrong type
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                }],
            }],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)

    def test_searchable_boolean_passes_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        with open(KB_SCHEMA, "r", encoding="utf-8") as f:
            schema = json.load(f)
        good = {
            "fileMetadata": {
                "schemaVersion": "2.0",
                "fileId": "t",
                "docSha256": "a" * 64,
            },
            "entries": [{
                "entryId": "e",
                "jobTitle": "t",
                "blocks": [{
                    "id": "b",
                    "type": "code",
                    "searchable": True,
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                }],
            }],
        }
        jsonschema.validate(instance=good, schema=schema)


if __name__ == "__main__":
    unittest.main()
