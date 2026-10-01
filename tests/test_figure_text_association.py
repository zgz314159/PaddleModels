"""Phase 2E tests: figure ↔ native caption/legend association (no PDF required)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.canonical_ir import BBox, CanonicalIR, DocBlock, DocPage  # noqa: E402
from pipeline.semantics.figure_text_association import (  # noqa: E402
    associate_ir_figures,
    associate_page_texts,
    has_strong_legend_markers,
    is_reference_only,
    looks_like_legend,
    looks_like_primary_caption,
)
from pipeline.semantic_projector import SemanticProjector  # noqa: E402

KB_SCHEMA = REPO_ROOT / "contracts" / "knowledge_base_schema_v2.json"
IR_SCHEMA = REPO_ROOT / "contracts" / "knowledge-base.v2.schema.json"


def _fig(fid="fig1", x=20, y=40, w=160, h=180, page=1):
    return DocBlock(
        id=fid,
        type="figure",
        text="",
        bbox=BBox(x, y, w, h),
        page_number=page,
        reading_order=1,
        metadata={
            "assetSource": "embedded",
            "assetStatus": "ready",
            "imageUri": f"shots/{fid}.png",
        },
    )


def _text(bid, text, x, y, w, h, role="body", page=1, btype="text", source="native"):
    return DocBlock(
        id=bid,
        type=btype,
        text=text,
        bbox=BBox(x, y, w, h),
        page_number=page,
        reading_order=2,
        metadata={"semanticRole": role},
        source=source,
    )


class TestReferenceReject(unittest.TestCase):
    def test_jian_tu_not_caption(self):
        self.assertTrue(is_reference_only("2.5.3口对口(鼻)人工呼吸(见图4)："))
        self.assertTrue(is_reference_only("固定(图11),也可利用"))

    def test_legend_pattern(self):
        self.assertTrue(looks_like_legend("(a)气道通畅(b) 气道阻塞"))
        self.assertTrue(looks_like_legend("（ａ）气道通畅（ｂ）气道阻塞"))
        self.assertFalse(looks_like_legend("见图4"))

    def test_primary_caption_pattern(self):
        self.assertTrue(looks_like_primary_caption("图4 人工呼吸示意"))
        self.assertFalse(looks_like_primary_caption("2.5.3口对口人工呼吸(见图4)"))


class TestAssociationRules(unittest.TestCase):
    def test_caption_below_figure_associates(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("f1", x=30, y=40, w=180, h=160, page=1))
        # caption-like with role=caption below figure
        page.blocks.append(
            _text("b_cap", "图4 人工呼吸示意", 40, 210, 160, 16, role="caption", page=1, btype="caption")
        )
        st = associate_page_texts(page)
        fig = page.blocks[0]
        assocs = fig.metadata.get("figureTextAssociations") or []
        self.assertEqual(st["captions_linked"], 1)
        self.assertEqual(len(assocs), 1)
        self.assertEqual(assocs[0]["kind"], "caption")
        self.assertEqual(assocs[0]["blockId"], "b_cap")
        self.assertEqual(assocs[0]["method"], "same_page_geometry")
        self.assertEqual(assocs[0]["source"], "native")
        # text block still present
        self.assertTrue(any(b.id == "b_cap" for b in page.blocks))

    def test_page137_legend_not_caption(self):
        page = DocPage(page_number=137, width=246, height=360, method="native")
        # figure like real p137
        page.blocks.append(_fig("f137", x=54, y=81, w=163, h=191, page=137))
        # legend just below figure
        page.blocks.append(
            _text(
                "b_leg",
                "(a)气道通畅(b) 气道阻塞",
                82,
                278,
                99,
                10,
                role="figure_callout",
                page=137,
            )
        )
        # body with 见图4 further below
        page.blocks.append(
            _text(
                "b_body",
                "2.5.3口对口(鼻)人工呼吸(见图4)：",
                50,
                296,
                180,
                10,
                role="body",
                page=137,
            )
        )
        associate_page_texts(page)
        fig = page.blocks[0]
        assocs = fig.metadata.get("figureTextAssociations") or []
        self.assertEqual(len(assocs), 1)
        self.assertEqual(assocs[0]["kind"], "legend")
        self.assertEqual(assocs[0]["blockId"], "b_leg")
        # 见图 body not linked
        self.assertNotIn("b_body", [a["blockId"] for a in assocs])
        # legend role upgraded
        self.assertEqual(page.blocks[1].metadata.get("semanticRole"), "legend")

    def test_page151_ref_not_caption(self):
        page = DocPage(page_number=151, width=246, height=360, method="native")
        page.blocks.append(_fig("f151", x=23, y=35, w=213, h=80, page=151))
        page.blocks.append(
            _text(
                "b_ref",
                "竿等将断骨上、下方两个关节固定(图11),也可利用",
                23,
                121,
                200,
                40,
                role="body",
                page=151,
            )
        )
        st = associate_page_texts(page)
        assocs = page.blocks[0].metadata.get("figureTextAssociations") or []
        self.assertEqual(assocs, [])
        self.assertEqual(st["captions_linked"], 0)

    def test_page135_plain_body_not_caption(self):
        page = DocPage(page_number=135, width=246, height=360, method="native")
        page.blocks.append(_fig("f135", x=44, y=37, w=151, h=208, page=135))
        page.blocks.append(
            _text(
                "b_body",
                "2.4.1.3试——试测口鼻有无呼气的气流。",
                23,
                252,
                200,
                40,
                role="body",
                page=135,
            )
        )
        associate_page_texts(page)
        assocs = page.blocks[0].metadata.get("figureTextAssociations") or []
        self.assertEqual(assocs, [])

    def test_too_far_no_associate(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("f1", x=30, y=40, w=180, h=100, page=1))
        page.blocks.append(
            _text("b_far", "图4 示意", 40, 300, 100, 16, role="caption", page=1, btype="caption")
        )
        associate_page_texts(page)
        self.assertEqual(page.blocks[0].metadata.get("figureTextAssociations"), [])

    def test_no_horizontal_overlap_no_associate(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("f1", x=20, y=40, w=60, h=80, page=1))
        page.blocks.append(
            _text("b_side", "图4 示意", 200, 100, 40, 16, role="caption", page=1, btype="caption")
        )
        associate_page_texts(page)
        self.assertEqual(page.blocks[0].metadata.get("figureTextAssociations"), [])

    def test_one_caption_not_two_figures(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("f1", x=20, y=40, w=100, h=80, page=1))
        page.blocks.append(_fig("f2", x=130, y=40, w=100, h=80, page=1))
        # one caption centered under both-ish but closer to f1 — only one link
        page.blocks.append(
            _text("b_cap", "图4 示意", 40, 130, 80, 16, role="caption", page=1, btype="caption")
        )
        st = associate_page_texts(page)
        total_caption_links = 0
        for b in page.blocks:
            if b.type == "figure":
                for a in (b.metadata.get("figureTextAssociations") or []):
                    if a["kind"] == "caption":
                        total_caption_links += 1
                        self.assertEqual(a["blockId"], "b_cap")
        self.assertEqual(total_caption_links, 1)
        self.assertEqual(st["captions_linked"], 1)

    def test_two_figs_two_legends_stable_match(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("fL", x=20, y=40, w=90, h=80, page=1))
        page.blocks.append(_fig("fR", x=130, y=40, w=90, h=80, page=1))
        # role=legend is a first-class candidate (no strong-marker requirement)
        page.blocks.append(
            _text("legL", "(a)左图说明", 25, 130, 80, 12, role="legend", page=1)
        )
        page.blocks.append(
            _text("legR", "(b)右图说明", 135, 130, 80, 12, role="legend", page=1)
        )
        associate_page_texts(page)
        fL = next(b for b in page.blocks if b.id == "fL")
        fR = next(b for b in page.blocks if b.id == "fR")
        aL = [a["blockId"] for a in fL.metadata.get("figureTextAssociations") or []]
        aR = [a["blockId"] for a in fR.metadata.get("figureTextAssociations") or []]
        self.assertIn("legL", aL)
        self.assertIn("legR", aR)
        self.assertNotIn("legR", aL)

    def test_input_order_stable(self):
        def build(order):
            page = DocPage(page_number=1, width=246, height=360, method="native")
            blocks = {
                "f": _fig("f1", x=30, y=40, w=180, h=160, page=1),
                "c": _text("b1", "图4 示意", 40, 210, 160, 16, role="caption", page=1, btype="caption"),
                "x": _text("b2", "正文", 40, 250, 160, 16, role="body", page=1),
            }
            for k in order:
                page.blocks.append(blocks[k])
            associate_page_texts(page)
            fig = blocks["f"]
            return fig.metadata.get("figureTextAssociations") or []

        a = build(["f", "c", "x"])
        b = build(["x", "c", "f"])
        c = build(["c", "f", "x"])
        self.assertEqual(a, b)
        self.assertEqual(a, c)
        self.assertEqual(len(a), 1)


class TestProjectionAndContent(unittest.TestCase):
    def test_caption_once_legend_not_in_content(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=246, height=360, method="native")
        # body first so entry exists
        page.blocks.append(
            DocBlock(
                id="body1",
                type="text",
                text="正文段落说明内容",
                bbox=BBox(20, 10, 200, 40),
                page_number=1,
                reading_order=1,
                metadata={"semanticRole": "body"},
            )
        )
        fig = _fig("f1", x=30, y=60, w=180, h=120, page=1)
        fig.metadata["figureTextAssociations"] = [
            {
                "kind": "caption",
                "blockId": "cap1",
                "text": "图4 人工呼吸示意",
                "source": "native",
                "method": "same_page_geometry",
            },
            {
                "kind": "legend",
                "blockId": "leg1",
                "text": "(a)通畅 (b)阻塞",
                "source": "native",
                "method": "same_page_geometry",
            },
        ]
        page.blocks.append(fig)
        # caption text block (searchable) appears once in body path
        page.blocks.append(
            DocBlock(
                id="cap1",
                type="caption",
                text="图4 人工呼吸示意",
                bbox=BBox(40, 190, 160, 16),
                page_number=1,
                reading_order=3,
                metadata={"semanticRole": "caption"},
            )
        )
        page.blocks.append(
            DocBlock(
                id="leg1",
                type="text",
                text="(a)通畅 (b)阻塞",
                bbox=BBox(40, 210, 160, 12),
                page_number=1,
                reading_order=4,
                metadata={"semanticRole": "legend"},
            )
        )
        ir.pages.append(page)
        kb = SemanticProjector("d", strategy="heading").project(ir)
        content = ""
        for e in kb["entries"]:
            content += e.get("contentMarkdown") or ""
        # caption appears once in content (from text/caption block)
        self.assertEqual(content.count("图4 人工呼吸示意"), 1)
        # legend is supporting — not in content
        self.assertNotIn("(a)通畅", content)
        # image block has associations
        img_blocks = [
            b
            for e in kb["entries"]
            for b in e["blocks"]
            if b.get("type") == "image"
        ]
        self.assertEqual(len(img_blocks), 1)
        assocs = img_blocks[0].get("figureTextAssociations") or []
        self.assertEqual({a["kind"] for a in assocs}, {"caption", "legend"})

    def test_ir_postpass_totals(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=137, width=246, height=360, method="native")
        # Match real p137 geometry so legend sits just under figure
        page.blocks.append(_fig("f", x=54, y=81, w=163, h=191, page=137))
        page.blocks.append(
            _text("leg", "(a)通畅(b)阻塞", 82, 278, 99, 10, role="figure_callout", page=137)
        )
        ir.pages.append(page)
        totals = associate_ir_figures(ir)
        self.assertEqual(totals["figures_with_native_legend"], 1)
        self.assertEqual(totals["native_legends_linked"], 1)
        self.assertEqual(totals["figures_with_native_caption"], 0)


class TestSchemaAssociation(unittest.TestCase):
    def _kb(self, assocs=None, **extra):
        img = {
            "id": "im1",
            "type": "image",
            "imageUri": "shots/a.png",
            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        }
        if assocs is not None:
            img["figureTextAssociations"] = assocs
        img.update(extra)
        return {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [{"entryId": "e", "jobTitle": "t", "blocks": [img]}],
        }

    def test_valid_associations(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        good = self._kb(
            [
                {
                    "kind": "caption",
                    "blockId": "b1",
                    "text": "图4",
                    "source": "native",
                    "method": "same_page_geometry",
                },
                {
                    "kind": "legend",
                    "blockId": "b2",
                    "text": "(a)",
                    "source": "native",
                    "method": "same_page_geometry",
                },
            ]
        )
        jsonschema.validate(instance=good, schema=schema)

    def test_reject_bad_kind(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        bad = self._kb(
            [
                {
                    "kind": "ref",
                    "blockId": "b1",
                    "text": "x",
                    "source": "native",
                    "method": "same_page_geometry",
                }
            ]
        )
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)


class TestPhase2E1Gates(unittest.TestCase):
    """2E.1: global determinism, legend tightening, IR schema, fail-fast."""

    def test_figure_order_reversed_same_result(self):
        def build(fig_order):
            page = DocPage(page_number=1, width=246, height=360, method="native")
            fA = _fig("figA", x=20, y=40, w=90, h=80, page=1)
            fB = _fig("figB", x=130, y=40, w=90, h=80, page=1)
            legL = _text("legL", "(a)左", 25, 130, 80, 12, role="legend", page=1)
            legR = _text("legR", "(b)右", 135, 130, 80, 12, role="legend", page=1)
            for fid in fig_order:
                page.blocks.append(fA if fid == "figA" else fB)
            page.blocks.extend([legL, legR])
            associate_page_texts(page)
            return {
                fid: (next(b for b in page.blocks if b.id == fid).metadata.get(
                    "figureTextAssociations"
                ) or [])
                for fid in ("figA", "figB")
            }

        a = build(["figA", "figB"])
        b = build(["figB", "figA"])
        self.assertEqual(a, b)

    def test_caption_binds_closer_figure_not_list_first(self):
        # figFar listed first but caption is much closer to figNear
        page = DocPage(page_number=1, width=300, height=400, method="native")
        fig_far = _fig("figFar", x=10, y=40, w=80, h=80, page=1)
        fig_near = _fig("figNear", x=100, y=100, w=80, h=80, page=1)
        # caption just under figNear (gap ~5), far from figFar
        cap = _text(
            "cap1", "图4 示意", 110, 190, 60, 16, role="caption", page=1, btype="caption"
        )
        page.blocks.extend([fig_far, fig_near, cap])
        associate_page_texts(page)
        far_assocs = fig_far.metadata.get("figureTextAssociations") or []
        near_assocs = fig_near.metadata.get("figureTextAssociations") or []
        self.assertEqual(far_assocs, [])
        self.assertEqual(len(near_assocs), 1)
        self.assertEqual(near_assocs[0]["blockId"], "cap1")

    def test_shuffled_figs_and_texts_stable(self):
        def build(order_ids):
            page = DocPage(page_number=1, width=300, height=400, method="native")
            blocks = {
                "fA": _fig("fA", x=20, y=40, w=90, h=80, page=1),
                "fB": _fig("fB", x=140, y=40, w=90, h=80, page=1),
                "lA": _text("lA", "(a)甲说明", 25, 130, 80, 12, role="legend", page=1),
                "lB": _text("lB", "(b)乙说明", 145, 130, 80, 12, role="legend", page=1),
            }
            for oid in order_ids:
                page.blocks.append(blocks[oid])
            associate_page_texts(page)
            return {
                k: (blocks[k].metadata.get("figureTextAssociations") or [])
                for k in ("fA", "fB")
            }

        base = build(["fA", "fB", "lA", "lB"])
        shuffled = build(["lB", "fB", "lA", "fA"])
        self.assertEqual(base, shuffled)
        self.assertEqual(
            [a["blockId"] for a in base["fA"]], ["lA"]
        )
        self.assertEqual(
            [a["blockId"] for a in base["fB"]], ["lB"]
        )

    def test_identical_geometry_tiebreak_by_id(self):
        # Two identical figures overlapping same legend region — deterministic by fig id
        page = DocPage(page_number=1, width=300, height=400, method="native")
        f1 = _fig("figZ", x=50, y=40, w=100, h=80, page=1)
        f2 = _fig("figA", x=50, y=40, w=100, h=80, page=1)  # same bbox, different id
        leg = _text("leg1", "(a)说明", 55, 130, 90, 12, role="legend", page=1)
        page.blocks.extend([f1, f2, leg])
        associate_page_texts(page)
        a1 = f1.metadata.get("figureTextAssociations") or []
        a2 = f2.metadata.get("figureTextAssociations") or []
        # Only one assignment; winner is stable (figA < figZ in sort after equal geometry)
        winners = [x for x in (a1, a2) if x]
        self.assertEqual(len(winners), 1)
        # Same run twice with reversed input → same winner id
        page2 = DocPage(page_number=1, width=300, height=400, method="native")
        f2b = _fig("figA", x=50, y=40, w=100, h=80, page=1)
        f1b = _fig("figZ", x=50, y=40, w=100, h=80, page=1)
        legb = _text("leg1", "(a)说明", 55, 130, 90, 12, role="legend", page=1)
        page2.blocks.extend([f2b, f1b, legb])
        associate_page_texts(page2)
        w1 = [b.id for b in (f1, f2) if (b.metadata.get("figureTextAssociations") or [])]
        w2 = [b.id for b in (f2b, f1b) if (b.metadata.get("figureTextAssociations") or [])]
        self.assertEqual(w1, w2)

    def test_body_list_not_legend(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("f1", x=30, y=40, w=180, h=160, page=1))
        page.blocks.append(
            _text("b1", "(1)操作步骤", 40, 210, 160, 16, role="body", page=1)
        )
        page.blocks.append(
            _text("b2", "(a)注意事项", 40, 240, 160, 16, role="body", page=1)
        )
        st = associate_page_texts(page)
        assocs = page.blocks[0].metadata.get("figureTextAssociations") or []
        self.assertEqual(assocs, [])
        self.assertEqual(st["legends_linked"], 0)

    def test_figure_callout_single_marker_not_legend(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("f1", x=30, y=40, w=180, h=160, page=1))
        page.blocks.append(
            _text("c1", "(1)步骤", 40, 210, 100, 16, role="figure_callout", page=1)
        )
        associate_page_texts(page)
        self.assertEqual(
            page.blocks[0].metadata.get("figureTextAssociations") or [], []
        )

    def test_ocr_source_not_native_candidate(self):
        page = DocPage(page_number=1, width=246, height=360, method="native")
        page.blocks.append(_fig("f1", x=30, y=40, w=180, h=160, page=1))
        page.blocks.append(
            _text(
                "ocr1",
                "图4 OCR caption",
                40,
                210,
                160,
                16,
                role="caption",
                page=1,
                btype="caption",
                source="ocr",
            )
        )
        st = associate_page_texts(page)
        assocs = page.blocks[0].metadata.get("figureTextAssociations") or []
        self.assertEqual(assocs, [])
        self.assertEqual(st["captions_linked"], 0)

    def test_page137_fullwidth_legend_still_works(self):
        page = DocPage(page_number=137, width=246, height=360, method="native")
        page.blocks.append(_fig("f137", x=54, y=81, w=163, h=191, page=137))
        page.blocks.append(
            _text(
                "b_leg",
                "（ａ）气道通畅 （ｂ）气道阻塞",
                82,
                278,
                99,
                10,
                role="figure_callout",
                page=137,
            )
        )
        associate_page_texts(page)
        assocs = page.blocks[0].metadata.get("figureTextAssociations") or []
        self.assertEqual(len(assocs), 1)
        self.assertEqual(assocs[0]["kind"], "legend")

    def test_association_arrays_sorted_stable(self):
        page = DocPage(page_number=1, width=300, height=400, method="native")
        page.blocks.append(_fig("f1", x=20, y=40, w=200, h=80, page=1))
        page.blocks.append(
            _text("cap9", "图9 题", 40, 130, 80, 14, role="caption", page=1, btype="caption")
        )
        page.blocks.append(
            _text("leg2", "(a)甲 (b)乙", 40, 150, 100, 12, role="legend", page=1)
        )
        associate_page_texts(page)
        assocs = page.blocks[0].metadata.get("figureTextAssociations") or []
        kinds = [a["kind"] for a in assocs]
        self.assertEqual(kinds, sorted(kinds, key=lambda k: {"caption": 0, "legend": 1}[k]))
        # caption before legend; within kind by blockId
        if len(assocs) >= 2 and assocs[0]["kind"] == assocs[1]["kind"]:
            ids = [a["blockId"] for a in assocs if a["kind"] == assocs[0]["kind"]]
            self.assertEqual(ids, sorted(ids))


class TestIRSchemaAssociation(unittest.TestCase):
    def _ir(self, assocs=None):
        meta: dict = {}
        if assocs is not None:
            meta["figureTextAssociations"] = assocs
        return {
            "document_id": "d",
            "schema_version": "2.0",
            "pages": [
                {
                    "page_number": 1,
                    "width": 100.0,
                    "height": 100.0,
                    "blocks": [
                        {
                            "id": "fig1",
                            "type": "figure",
                            "text": "",
                            "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                            "page_number": 1,
                            "reading_order": 1,
                            "metadata": meta,
                        }
                    ],
                }
            ],
        }

    def test_ir_valid_association(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(IR_SCHEMA.read_text(encoding="utf-8"))
        good = self._ir(
            [
                {
                    "kind": "legend",
                    "blockId": "b1",
                    "text": "(a)x(b)y",
                    "source": "native",
                    "method": "same_page_geometry",
                }
            ]
        )
        jsonschema.validate(instance=good, schema=schema)

    def test_ir_reject_bad_kind(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(IR_SCHEMA.read_text(encoding="utf-8"))
        bad = self._ir(
            [
                {
                    "kind": "ref",
                    "blockId": "b1",
                    "text": "x",
                    "source": "native",
                    "method": "same_page_geometry",
                }
            ]
        )
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)

    def test_kb_and_ir_reject_kind_ref(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        kb_schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        ir_schema = json.loads(IR_SCHEMA.read_text(encoding="utf-8"))
        entry = [
            {
                "kind": "ref",
                "blockId": "b1",
                "text": "x",
                "source": "native",
                "method": "same_page_geometry",
            }
        ]
        kb = {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [
                {
                    "entryId": "e",
                    "jobTitle": "t",
                    "blocks": [
                        {
                            "id": "im",
                            "type": "image",
                            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                            "figureTextAssociations": entry,
                        }
                    ],
                }
            ],
        }
        ir = self._ir(entry)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=kb, schema=kb_schema)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=ir, schema=ir_schema)


class TestAssociationFailFast(unittest.TestCase):
    def test_exception_not_swallowed_by_associate(self):
        """associate_ir_figures must raise — no silent success path inside module."""
        ir = CanonicalIR(document_id="d", sha256="0" * 64)

        class BoomPage:
            blocks = property(lambda self: (_ for _ in ()).throw(RuntimeError("inject")))

        # Direct call with broken page object
        bad_ir = type("I", (), {"pages": [type("P", (), {"blocks": property(
            lambda self: (_ for _ in ()).throw(RuntimeError("inject-association"))
        )})()]})()
        with self.assertRaises(RuntimeError):
            associate_ir_figures(bad_ir)

    def test_run_v2_association_failure_sets_failed(self):
        """Fault injection: association error → v2 status failed, no fake ok report."""
        import importlib.util

        if importlib.util.find_spec("fitz") is None:
            self.skipTest("fitz required for run_v2 integration")
        from unittest.mock import patch
        from paddle_models.cli.main import run_v2, probe_capabilities

        pdf = REPO_ROOT / "samples" / "103号(2).pdf"
        if not pdf.exists():
            self.skipTest("fixture PDF missing")
        caps = probe_capabilities("v2")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            with patch(
                "pipeline.v2_runner.associate_ir_figures",
                side_effect=RuntimeError("inject-association-boom"),
            ):
                result = run_v2(run_dir, pdf, caps, "smoke", 1, 1)
        self.assertEqual(result["status"], "failed")
        errs = " ".join(result.get("errors") or [])
        self.assertIn("inject-association-boom", errs)
        # Must not claim ok
        self.assertNotEqual(result.get("status"), "ok")
        # Fake success KB must not exist (association runs before write)
        kb = run_dir / "v2" / "knowledge_base.json"
        self.assertFalse(kb.exists())


if __name__ == "__main__":
    unittest.main()
