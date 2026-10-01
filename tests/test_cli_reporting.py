"""Unit tests for Phase 1.1 CLI reporting helpers (no PDF deps required)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from paddle_models.cli.main import (  # noqa: E402
    SHADOW_COMPARABLE_KEYS,
    build_shadow_diff,
    kb_metrics_from_obj,
    make_run_id,
    normalize_text,
    validate_against_schema,
    KB_SCHEMA_REL,
)


def _minimal_kb(
    *,
    page_sizes=None,
    entries=None,
) -> dict:
    return {
        "fileMetadata": {
            "schemaVersion": "2.0",
            "fileId": "t",
            "docSha256": "0" * 64,
            "pageSizes": page_sizes if page_sizes is not None else {},
        },
        "entries": entries if entries is not None else [],
    }


class TestPageMetrics(unittest.TestCase):
    def test_pages_from_entries_when_page_sizes_empty(self):
        kb = _minimal_kb(
            page_sizes={},
            entries=[
                {
                    "entryId": "a",
                    "jobTitle": "p1",
                    "pageNumber": 1,
                    "contentNormalized": "hello",
                    "blocks": [
                        {"id": "b1", "type": "code", "pageNumber": 1, "bbox": {}}
                    ],
                },
                {
                    "entryId": "b",
                    "jobTitle": "p2",
                    "pageNumber": 2,
                    "contentNormalized": "world",
                    "blocks": [
                        {"id": "b2", "type": "code", "pageNumber": 2, "bbox": {}}
                    ],
                },
            ],
        )
        m = kb_metrics_from_obj(kb)
        self.assertEqual(m["pages"], 2)
        self.assertEqual(m["page_numbers"], [1, 2])
        self.assertEqual(m["pages_declared"], 0)

    def test_pages_union_with_page_sizes(self):
        kb = _minimal_kb(
            page_sizes={"1": [100, 200], "3": [100, 200]},
            entries=[
                {
                    "entryId": "a",
                    "pageNumber": 2,
                    "contentNormalized": "x",
                    "blocks": [],
                }
            ],
        )
        m = kb_metrics_from_obj(kb)
        self.assertEqual(m["pages"], 3)
        self.assertEqual(m["page_numbers"], [1, 2, 3])
        self.assertEqual(m["pages_declared"], 2)

    def test_block_page_numbers_counted(self):
        kb = _minimal_kb(
            page_sizes={},
            entries=[
                {
                    "entryId": "a",
                    "pageNumber": None,
                    "contentNormalized": "x",
                    "blocks": [{"id": "b", "type": "text", "pageNumber": 5, "bbox": {}}],
                }
            ],
        )
        m = kb_metrics_from_obj(kb)
        self.assertEqual(m["page_numbers"], [5])
        self.assertEqual(m["pages"], 1)


class TestEmptyEntries(unittest.TestCase):
    def test_empty_and_nonempty_split(self):
        kb = _minimal_kb(
            page_sizes={"1": [1, 1], "2": [1, 1]},
            entries=[
                {
                    "entryId": "e1",
                    "pageNumber": 1,
                    "contentNormalized": "非空内容",
                    "blocks": [{"id": "b", "type": "code", "pageNumber": 1, "bbox": {}}],
                },
                {
                    "entryId": "e2",
                    "pageNumber": 2,
                    "contentNormalized": "",
                    "contentMarkdown": "   \n  ",
                    "blocks": [],
                },
            ],
        )
        m = kb_metrics_from_obj(kb)
        self.assertEqual(m["entries"], 2)
        self.assertEqual(m["entries_nonempty"], 1)
        self.assertEqual(m["entries_empty"], 1)

    def test_visual_only_entry_counts_as_empty_content(self):
        # content fields empty → entries_empty (blocks alone do not fill content).
        kb = _minimal_kb(
            entries=[
                {
                    "entryId": "e",
                    "pageNumber": 1,
                    "contentNormalized": "",
                    "blocks": [
                        {
                            "id": "b",
                            "type": "image",
                            "pageNumber": 1,
                            "bbox": {},
                            "imageUri": "shots/x.png",
                        }
                    ],
                }
            ],
        )
        m = kb_metrics_from_obj(kb)
        self.assertEqual(m["entries_nonempty"], 0)
        self.assertEqual(m["entries_empty"], 1)
        self.assertEqual(m["images"], 1)


class TestNormalizedTextHash(unittest.TestCase):
    def test_whitespace_difference_same_hash(self):
        a = _minimal_kb(
            entries=[
                {
                    "entryId": "1",
                    "pageNumber": 1,
                    "contentNormalized": "关于发布 规则\n\n和规程",
                    "blocks": [],
                }
            ]
        )
        b = _minimal_kb(
            entries=[
                {
                    "entryId": "1",
                    "pageNumber": 1,
                    "contentNormalized": "关于发布 规则 和规程",
                    "blocks": [],
                }
            ]
        )
        ma, mb = kb_metrics_from_obj(a), kb_metrics_from_obj(b)
        self.assertEqual(ma["normalized_text_sha256"], mb["normalized_text_sha256"])
        self.assertEqual(normalize_text("  a \n b\t c "), normalize_text("a b c"))

    def test_different_content_different_hash(self):
        a = _minimal_kb(entries=[{"entryId": "1", "contentNormalized": "abc", "blocks": []}])
        b = _minimal_kb(entries=[{"entryId": "1", "contentNormalized": "abd", "blocks": []}])
        self.assertNotEqual(
            kb_metrics_from_obj(a)["normalized_text_sha256"],
            kb_metrics_from_obj(b)["normalized_text_sha256"],
        )


class TestShadowDiff(unittest.TestCase):
    def test_only_shared_metrics_compared(self):
        leg = {
            "status": "ok",
            "metrics": {
                "pages": 2,
                "entries": 2,
                "tables": 0,
                "text_chars": 100,
            },
            "elapsed_ms": 10,
            "errors": [],
        }
        v2 = {
            "status": "ok",
            "metrics": {
                "pages": 2,
                "entries": 1,
                # tables missing on purpose
                "text_chars": 120,
            },
            "elapsed_ms": 5,
            "errors": [],
        }
        diff = build_shadow_diff("r1", leg, v2)
        self.assertIn("pages", diff["diff"])
        self.assertEqual(diff["diff"]["pages"]["delta"], 0)
        self.assertIn("entries", diff["diff"])
        self.assertEqual(diff["diff"]["entries"]["delta"], -1)
        self.assertNotIn("tables", diff["diff"])
        self.assertIn("tables", diff["diff"].get("skipped_keys", []))

    def test_missing_side_metrics_no_fake_delta(self):
        leg = {"status": "blocked_by_dependency", "metrics": None, "elapsed_ms": 0, "errors": ["x"]}
        v2 = {"status": "ok", "metrics": {"pages": 2}, "elapsed_ms": 1, "errors": []}
        diff = build_shadow_diff("r2", leg, v2)
        # No numeric deltas — only an explanatory note.
        self.assertNotIn("pages", diff["diff"])
        self.assertNotIn("entries", diff["diff"])
        self.assertIn("note", diff["diff"])
        self.assertIsNone(diff["legacy"]["metrics"])


class TestRunIdUniqueness(unittest.TestCase):
    def test_two_consecutive_run_ids_differ(self):
        a = make_run_id("doc")
        b = make_run_id("doc")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("doc_"))
        self.assertTrue(b.startswith("doc_"))


class TestSchemaValidation(unittest.TestCase):
    def _valid_kb(self) -> dict:
        return {
            "fileMetadata": {
                "schemaVersion": "2.0",
                "fileId": "t",
                "docSha256": "a" * 64,
            },
            "entries": [
                {
                    "entryId": "e1",
                    "jobTitle": "title",
                    "blocks": [
                        {
                            "id": "b1",
                            "type": "code",
                            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                        }
                    ],
                }
            ],
        }

    def test_valid_kb_passes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "kb.json"
            p.write_text(json.dumps(self._valid_kb()), encoding="utf-8")
            res = validate_against_schema(p, KB_SCHEMA_REL, "knowledge_base")
            if res["status"] == "skipped_dependency":
                self.skipTest("jsonschema not installed")
            self.assertEqual(res["status"], "valid")
            self.assertEqual(res["errors"], [])

    def test_invalid_kb_returns_field_path(self):
        bad = self._valid_kb()
        bad["entries"][0]["jobTitle"] = 123  # must be string
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "kb.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            res = validate_against_schema(p, KB_SCHEMA_REL, "knowledge_base")
            if res["status"] == "skipped_dependency":
                self.skipTest("jsonschema not installed")
            self.assertEqual(res["status"], "invalid")
            self.assertTrue(res["errors"], "expected brief errors")
            joined = " ".join(res["errors"])
            self.assertIn("jobTitle", joined)

    def test_missing_schema_file(self):
        res = validate_against_schema(None, Path("contracts/nope.json"), "knowledge_base")
        self.assertEqual(res["status"], "missing_schema")


class TestFigureAssociationMetrics(unittest.TestCase):
    """2E.1: legacy omits v2-only association keys; v2 may report true 0."""

    ASSOC_KEYS = (
        "figures_with_native_caption",
        "figures_with_native_legend",
        "native_captions_linked",
        "native_legends_linked",
        "unmatched_caption_candidates",
    )

    def _kb_with_image(self, with_assocs_field: bool) -> dict:
        img = {
            "id": "im1",
            "type": "image",
            "imageUri": "shots/a.png",
            "assetSource": "embedded",
            "assetStatus": "ready",
            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        }
        if with_assocs_field:
            # v2 post-association always attaches the field (may be empty list)
            img["figureTextAssociations"] = []
        return {
            "fileMetadata": {"pageSizes": {"1": [1, 1]}},
            "entries": [
                {
                    "entryId": "e",
                    "contentNormalized": "x",
                    "blocks": [img],
                }
            ],
        }

    def test_legacy_metrics_omit_association_keys(self):
        kb = self._kb_with_image(with_assocs_field=False)
        m = kb_metrics_from_obj(kb, include_figure_association=False)
        for k in self.ASSOC_KEYS:
            self.assertNotIn(k, m)
        # common keys still present
        self.assertIn("images", m)
        self.assertIn("pages", m)

    def test_v2_metrics_include_zero_when_supported(self):
        kb = self._kb_with_image(with_assocs_field=True)
        m = kb_metrics_from_obj(kb, include_figure_association=True)
        for k in self.ASSOC_KEYS:
            self.assertIn(k, m)
            # true observation of zero is allowed for v2
            self.assertEqual(m[k], 0)
        self.assertEqual(m["figures_with_native_legend"], 0)

    def test_shadow_skips_association_keys_for_legacy(self):
        leg_m = kb_metrics_from_obj(
            self._kb_with_image(False), include_figure_association=False
        )
        v2_m = kb_metrics_from_obj(
            self._kb_with_image(True), include_figure_association=True
        )
        self.assertIn("figures_with_native_legend", self.SHADOW_KEYS_HELPER())
        diff = build_shadow_diff(
            "r",
            {"status": "ok", "metrics": leg_m, "elapsed_ms": 1, "errors": []},
            {"status": "ok", "metrics": v2_m, "elapsed_ms": 1, "errors": []},
        )
        skipped = diff["diff"].get("skipped_keys") or []
        for k in self.ASSOC_KEYS:
            self.assertIn(k, skipped, f"{k} must be skipped, not deltaed")
            self.assertNotIn(k, diff["diff"], f"{k} must not emit delta")
        # pages still compared
        self.assertIn("pages", diff["diff"])

    def SHADOW_KEYS_HELPER(self):
        return SHADOW_COMPARABLE_KEYS

    def test_both_null_association_skipped(self):
        leg_m = {"pages": 1, "figures_with_native_legend": None}
        v2_m = {"pages": 1, "figures_with_native_legend": 0}
        diff = build_shadow_diff(
            "r",
            {"status": "ok", "metrics": leg_m, "elapsed_ms": 0, "errors": []},
            {"status": "ok", "metrics": v2_m, "elapsed_ms": 0, "errors": []},
        )
        self.assertIn("figures_with_native_legend", diff["diff"].get("skipped_keys") or [])
        self.assertNotIn("figures_with_native_legend", diff["diff"])


if __name__ == "__main__":
    unittest.main()
