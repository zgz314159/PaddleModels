"""Phase 2G tests: figure label index + cross-page reference resolver."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.canonical_ir import BBox, CanonicalIR, DocBlock, DocPage  # noqa: E402
from pipeline.semantics.figure_reference import (  # noqa: E402
    build_label_index,
    collect_trusted_labels,
    merge_figure_labels,
    resolve_figure_references,
    resolve_targets,
    stable_reference_id,
)
from pipeline.semantic_projector import SemanticProjector  # noqa: E402
from paddle_models.cli.main import (  # noqa: E402
    FIGURE_REFERENCE_METRIC_KEYS,
    SHADOW_COMPARABLE_KEYS,
    build_shadow_diff,
    check_figure_reference_metrics,
    kb_metrics_from_obj,
)

KB_SCHEMA = REPO_ROOT / "contracts" / "knowledge_base_schema_v2.json"
IR_SCHEMA = REPO_ROOT / "contracts" / "knowledge-base.v2.schema.json"


def _fig(
    fid,
    page,
    x=20.0,
    y=40.0,
    w=100.0,
    h=80.0,
    assocs=None,
    labels=None,
):
    md = {
        "assetStatus": "ready",
        "assetSource": "embedded",
        "imageUri": f"shots/{fid}.png",
        "contentSha256": "a" * 64,
    }
    if assocs is not None:
        md["figureTextAssociations"] = assocs
    if labels is not None:
        md["figureLabels"] = labels
    return DocBlock(
        id=fid,
        type="figure",
        text="",
        bbox=BBox(x, y, w, h),
        page_number=page,
        reading_order=1,
        metadata=md,
    )


def _text(bid, text, page, x=20.0, y=200.0, w=180.0, h=20.0, role="body", source="native", btype="text"):
    return DocBlock(
        id=bid,
        type=btype,
        text=text,
        bbox=BBox(x, y, w, h),
        page_number=page,
        reading_order=2,
        source=source,
        metadata={"semanticRole": role},
    )


def _native_caption_assoc(text, block_id="cap1"):
    return {
        "kind": "caption",
        "blockId": block_id,
        "text": text,
        "source": "native",
        "method": "same_page_geometry",
    }


def _ocr_label(norm, conf=0.91):
    return {
        "rawLabel": norm,
        "normalizedLabel": norm,
        "source": "ocr",
        "method": "ocr_inside_image",
        "confidence": conf,
    }


def _ir_with_pages(page_specs):
    """page_specs: list of (page_number, [blocks])."""
    ir = CanonicalIR(document_id="d", sha256="0" * 64)
    for pnum, blocks in page_specs:
        page = DocPage(page_number=pnum, width=300.0, height=400.0, method="native")
        page.blocks.extend(blocks)
        ir.pages.append(page)
    return ir


class TestLabelCollection(unittest.TestCase):
    def test_native_caption_labels_indexed(self):
        fig = _fig(
            "f1",
            1,
            assocs=[_native_caption_assoc("图4 人工呼吸示意")],
        )
        ir = _ir_with_pages([(1, [fig])])
        total = collect_trusted_labels(ir)
        self.assertEqual(total, 1)
        labs = fig.metadata["figureLabels"]
        self.assertEqual(labs[0]["normalizedLabel"], "图4")
        self.assertEqual(labs[0]["source"], "native")
        self.assertEqual(labs[0]["confidence"], 1.0)

    def test_ocr_linked_labels_already_on_figure(self):
        fig = _fig("f1", 1, labels=[_ocr_label("图12", 0.88)])
        ir = _ir_with_pages([(1, [fig])])
        collect_trusted_labels(ir)
        labs = fig.metadata["figureLabels"]
        self.assertEqual(len(labs), 1)
        self.assertEqual(labs[0]["source"], "ocr")
        self.assertAlmostEqual(labs[0]["confidence"], 0.88)

    def test_label_only_and_ambiguous_labels_kept(self):
        # label_only style + ambiguous multi-label (page 137 composite).
        f135 = _fig("f135", 135, labels=[_ocr_label("图1", 0.72)])
        f137 = _fig(
            "f137",
            137,
            labels=[_ocr_label("图2", 0.79), _ocr_label("图3", 0.79)],
        )
        ir = _ir_with_pages([(135, [f135]), (137, [f137])])
        collect_trusted_labels(ir)
        self.assertEqual(
            [l["normalizedLabel"] for l in f135.metadata["figureLabels"]],
            ["图1"],
        )
        self.assertEqual(
            sorted(l["normalizedLabel"] for l in f137.metadata["figureLabels"]),
            ["图2", "图3"],
        )

    def test_native_wins_over_ocr_on_same_label(self):
        existing = [_ocr_label("图2", 0.5)]
        incoming = [
            {
                "rawLabel": "图2",
                "normalizedLabel": "图2",
                "source": "native",
                "method": "same_page_geometry",
                "confidence": 1.0,
            }
        ]
        merged = merge_figure_labels(existing, incoming)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["source"], "native")

    def test_dedup_by_normalized_label(self):
        merged = merge_figure_labels(
            [_ocr_label("图2"), _ocr_label("图2", 0.5)],
            [_ocr_label("图2", 0.9)],
        )
        self.assertEqual(len(merged), 1)


class TestReferenceExtraction(unittest.TestCase):
    def test_body_text_extracted_caption_legend_not(self):
        fig = _fig(
            "f1",
            1,
            assocs=[_native_caption_assoc("图4 题")],
            labels=[_ocr_label("图4")],
        )
        body = _text("b1", "见图4说明操作步骤", 1)
        cap = _text(
            "c1", "图4 题", 1, role="caption", btype="caption"
        )
        leg = _text("l1", "(a)说明 见图9", 1, role="legend")
        callout = _text("k1", "(a) 见图7", 1, role="figure_callout")
        ocr_cap = _text(
            "o1", "图12 颈椎", 1, role="caption", source="ocr", btype="caption"
        )
        ir = _ir_with_pages(
            [(1, [fig, body, cap, leg, callout, ocr_cap])]
        )
        report = resolve_figure_references(ir)
        # Only body b1 contributes references.
        self.assertEqual(report["stats"]["figure_references_found"], 1)
        refs = body.metadata.get("figureReferences")
        self.assertIsNotNone(refs)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["normalizedLabel"], "图4")
        # caption/legend/ocr caption blocks must not have figureReferences
        for b in (cap, leg, callout, ocr_cap):
            self.assertIsNone(b.metadata.get("figureReferences"))

    def test_same_block_two_labels_deduped(self):
        fig2 = _fig("f2", 137, labels=[_ocr_label("图2"), _ocr_label("图3")])
        # Same normalized label twice in one block → one ref; 图2 and 图3 both.
        body = _text("b1", "见图2和参见图3以及图2", 136, y=100.0)
        ir = _ir_with_pages([(136, [body]), (137, [fig2])])
        report = resolve_figure_references(ir)
        refs = body.metadata["figureReferences"]
        norms = [r["normalizedLabel"] for r in refs]
        self.assertEqual(norms, ["图2", "图3"])  # deduped 图2
        self.assertEqual(report["stats"]["figure_references_found"], 2)

    def test_both_labels_resolve_to_same_composite_figure(self):
        fig137 = _fig(
            "p137_fig1",
            137,
            labels=[_ocr_label("图2"), _ocr_label("图3")],
        )
        body = _text("p136_b4", "见图2，参见图3", 136, y=100.0)
        ir = _ir_with_pages([(136, [body]), (137, [fig137])])
        report = resolve_figure_references(ir)
        refs = body.metadata["figureReferences"]
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r["status"], "resolved")
            self.assertEqual(r["targetFigureId"], "p137_fig1")
            self.assertEqual(r["targetPageNumber"], 137)
            self.assertEqual(r["pageDelta"], 1)
        back = fig137.metadata["referencedBy"]
        self.assertEqual(len(back), 2)
        self.assertEqual(
            {b["referenceId"] for b in back},
            {r["referenceId"] for r in refs},
        )
        self.assertEqual(report["stats"]["figure_references_resolved"], 2)
        self.assertEqual(report["stats"]["referenced_figures"], 1)

    def test_reference_id_stable_not_order_dependent(self):
        a = stable_reference_id("b1", "图2")
        b = stable_reference_id("b1", "图2")
        c = stable_reference_id("b1", "图3")
        d = stable_reference_id("b2", "图2")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(a, d)

    def test_same_page_forward_backward(self):
        # Same page
        f1 = _fig("fa", 1, labels=[_ocr_label("图1")])
        b1 = _text("b1", "见图1", 1, y=300.0)
        # Forward: page 151 body → 152 figure
        f152 = _fig("fb", 152, labels=[_ocr_label("图2")])
        b151 = _text("b151", "见图2", 151, y=300.0)
        # Backward: page 153 body → 152 figure
        b153 = _text("b153", "见图2", 153, y=300.0)
        ir = _ir_with_pages(
            [
                (1, [f1, b1]),
                (151, [b151]),
                (152, [f152]),
                (153, [b153]),
            ]
        )
        report = resolve_figure_references(ir)
        self.assertEqual(b1.metadata["figureReferences"][0]["status"], "resolved")
        self.assertEqual(b1.metadata["figureReferences"][0]["pageDelta"], 0)
        self.assertEqual(b151.metadata["figureReferences"][0]["status"], "resolved")
        self.assertEqual(b151.metadata["figureReferences"][0]["pageDelta"], 1)
        self.assertEqual(b153.metadata["figureReferences"][0]["status"], "resolved")
        self.assertEqual(b153.metadata["figureReferences"][0]["pageDelta"], -1)

    def test_unique_nearest_same_page_wins(self):
        # Two figures with same label on same page — nearest by geometry wins.
        # Note: both have 图10 (shouldn't happen in prod for unique labels, but
        # tests multi-target same-page nearest rule).
        fA = _fig("fa", 151, x=20.0, y=40.0, labels=[_ocr_label("图10")])
        fB = _fig("fb", 151, x=20.0, y=200.0, w=200.0, h=80.0, labels=[_ocr_label("图10")])
        # Source near fB (y=300 below fB)
        body = _text("b1", "见图10", 151, y=300.0, x=40.0)
        ir = _ir_with_pages([(151, [fA, fB, body])])
        report = resolve_figure_references(ir)
        ref = body.metadata["figureReferences"][0]
        self.assertEqual(ref["status"], "resolved")
        # fB center y=240, fA center y=80; source center y=310 → fB closer
        self.assertEqual(ref["targetFigureId"], "fb")

    def test_tied_distance_ambiguous_not_first(self):
        # Two figures equidistant from source on another page (same |delta|).
        fL = _fig("fl", 10, x=10.0, labels=[_ocr_label("图5")])
        fR = _fig("fr", 12, x=10.0, labels=[_ocr_label("图5")])
        body = _text("b1", "见图5", 11, y=200.0)
        ir = _ir_with_pages([(10, [fL]), (11, [body]), (12, [fR])])
        report = resolve_figure_references(ir)
        ref = body.metadata["figureReferences"][0]
        self.assertEqual(ref["status"], "ambiguous")
        self.assertIn("tied_page_distance", ref.get("reason", ""))
        self.assertNotIn("targetFigureId", ref)
        self.assertEqual(report["stats"]["figure_references_ambiguous"], 1)
        # No referencedBy written for ambiguous
        self.assertIsNone(fL.metadata.get("referencedBy"))
        self.assertIsNone(fR.metadata.get("referencedBy"))

    def test_truncated_range_unresolved(self):
        # Only page 136 loaded; 图2 target is on 137 (not in range).
        body = _text("b1", "见图2 参见图3", 136)
        ir = _ir_with_pages([(136, [body])])
        report = resolve_figure_references(ir)
        refs = body.metadata["figureReferences"]
        self.assertEqual(len(refs), 2)
        for r in refs:
            self.assertEqual(r["status"], "unresolved")
            self.assertEqual(r["reason"], "no_label_in_index")
        self.assertEqual(report["stats"]["figure_references_unresolved"], 2)
        self.assertEqual(report["stats"]["figure_references_resolved"], 0)

    def test_order_independence(self):
        def build(page_order, block_order):
            f = _fig("f137", 137, labels=[_ocr_label("图2")])
            b = _text("b136", "见图2", 136)
            pages = {
                136: [b],
                137: [f],
            }
            specs = [(p, list(pages[p])) for p in page_order]
            # Also shuffle blocks within page via block_order marker
            if block_order == "rev" and len(specs[0][1]) > 1:
                specs[0] = (specs[0][0], list(reversed(specs[0][1])))
            ir = _ir_with_pages(specs)
            resolve_figure_references(ir)
            refs = b.metadata["figureReferences"]
            return [(r["referenceId"], r["status"], r.get("targetFigureId")) for r in refs]

        a = build([136, 137], "fwd")
        c = build([137, 136], "fwd")
        self.assertEqual(a, c)

    def test_resolver_fail_fast(self):
        class BoomIR:
            pages = property(lambda self: (_ for _ in ()).throw(RuntimeError("inject-refs")))

        with self.assertRaises(RuntimeError):
            resolve_figure_references(BoomIR())


class TestProjectorCopy(unittest.TestCase):
    def test_copies_fields_without_changing_content(self):
        fig = _fig(
            "f1",
            1,
            labels=[_ocr_label("图4")],
            assocs=[_native_caption_assoc("图4 题")],
        )
        body = _text("b1", "正文见图4", 1)
        ir = _ir_with_pages([(1, [body, fig])])
        resolve_figure_references(ir)
        kb = SemanticProjector("d", strategy="page").project(ir)
        content = "\n\n".join(e.get("contentMarkdown") or "" for e in kb["entries"])
        self.assertIn("正文见图4", content)
        # Reference metadata must not leak into contentMarkdown as extra text
        self.assertEqual(content.count("正文见图4"), 1)

        text_block = None
        img_block = None
        for e in kb["entries"]:
            for b in e["blocks"]:
                if b.get("id") == "b1" or (
                    b.get("type") == "code" and "见图4" in (b.get("code") or "")
                ):
                    text_block = b
                if b.get("type") == "image":
                    img_block = b
        self.assertIsNotNone(text_block)
        self.assertIsNotNone(img_block)
        self.assertIn("figureReferences", text_block)
        self.assertEqual(text_block["figureReferences"][0]["status"], "resolved")
        self.assertIn("figureLabels", img_block)
        self.assertIn("referencedBy", img_block)
        self.assertFalse(img_block.get("searchable"))

    def test_caption_text_not_duplicated(self):
        # OCR/native caption body already in content; references are metadata only.
        fig = _fig(
            "f1",
            1,
            labels=[_ocr_label("图4")],
            assocs=[_native_caption_assoc("图4 人工呼吸示意")],
        )
        body = _text("b1", "见图4", 1)
        cap = _text(
            "cap1",
            "图4 人工呼吸示意",
            1,
            y=250.0,
            role="caption",
            btype="caption",
        )
        ir = _ir_with_pages([(1, [body, fig, cap])])
        resolve_figure_references(ir)
        kb = SemanticProjector("d", strategy="page").project(ir)
        content = "\n\n".join(
            (e.get("contentMarkdown") or "") for e in kb["entries"]
        )
        self.assertEqual(content.count("图4 人工呼吸示意"), 1)


class TestSchema(unittest.TestCase):
    def _kb(self, **block_extra):
        block = {
            "id": "b",
            "type": "code",
            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        }
        block.update(block_extra)
        return {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [{"entryId": "e", "jobTitle": "t", "blocks": [block]}],
        }

    def test_valid_labels_and_references(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        kb = self._kb(
            figureReferences=[
                {
                    "referenceId": "ref_abc",
                    "rawLabel": "图2",
                    "normalizedLabel": "图2",
                    "status": "resolved",
                    "method": "exact_label_index",
                    "targetFigureId": "f1",
                    "targetPageNumber": 137,
                    "pageDelta": 1,
                },
                {
                    "referenceId": "ref_def",
                    "rawLabel": "图4",
                    "normalizedLabel": "图4",
                    "status": "unresolved",
                    "method": "exact_label_index",
                    "reason": "no_label_in_index",
                },
            ],
            figureLabels=[
                {
                    "rawLabel": "图2",
                    "normalizedLabel": "图2",
                    "source": "ocr",
                    "method": "ocr_inside_image",
                    "confidence": 0.91,
                }
            ],
            referencedBy=[
                {
                    "referenceId": "ref_abc",
                    "sourceBlockId": "b136",
                    "sourcePageNumber": 136,
                    "rawLabel": "图2",
                    "normalizedLabel": "图2",
                }
            ],
        )
        jsonschema.validate(instance=kb, schema=schema)

    def test_reject_bad_status(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        kb = self._kb(
            figureReferences=[
                {
                    "referenceId": "r",
                    "rawLabel": "图2",
                    "normalizedLabel": "图2",
                    "status": "maybe",
                    "method": "exact_label_index",
                }
            ]
        )
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=kb, schema=schema)

    def test_resolved_requires_target(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        kb = self._kb(
            figureReferences=[
                {
                    "referenceId": "r",
                    "rawLabel": "图2",
                    "normalizedLabel": "图2",
                    "status": "resolved",
                    "method": "exact_label_index",
                }
            ]
        )
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=kb, schema=schema)

    def test_reject_bad_label_source(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        kb = self._kb(
            figureLabels=[
                {
                    "rawLabel": "图2",
                    "normalizedLabel": "图2",
                    "source": "guess",
                    "method": "x",
                    "confidence": 0.5,
                }
            ]
        )
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=kb, schema=schema)

    def test_ir_valid_association_and_labels(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(IR_SCHEMA.read_text(encoding="utf-8"))
        ir = {
            "document_id": "d",
            "schema_version": "2.0",
            "pages": [
                {
                    "page_number": 1,
                    "width": 100.0,
                    "height": 100.0,
                    "blocks": [
                        {
                            "id": "f1",
                            "type": "figure",
                            "text": "",
                            "bbox": {"x": 1, "y": 2, "w": 3, "h": 4},
                            "page_number": 1,
                            "reading_order": 1,
                            "metadata": {
                                "figureLabels": [
                                    {
                                        "rawLabel": "图2",
                                        "normalizedLabel": "图2",
                                        "source": "ocr",
                                        "method": "ocr_inside_image",
                                        "confidence": 0.9,
                                    }
                                ],
                                "referencedBy": [
                                    {
                                        "referenceId": "ref_x",
                                        "sourceBlockId": "b1",
                                        "sourcePageNumber": 1,
                                        "rawLabel": "图2",
                                        "normalizedLabel": "图2",
                                    }
                                ],
                            },
                        },
                        {
                            "id": "b1",
                            "type": "text",
                            "text": "见图2",
                            "bbox": {"x": 1, "y": 10, "w": 5, "h": 2},
                            "page_number": 1,
                            "reading_order": 2,
                            "metadata": {
                                "semanticRole": "body",
                                "figureReferences": [
                                    {
                                        "referenceId": "ref_x",
                                        "rawLabel": "图2",
                                        "normalizedLabel": "图2",
                                        "status": "resolved",
                                        "method": "exact_label_index",
                                        "targetFigureId": "f1",
                                        "targetPageNumber": 1,
                                        "pageDelta": 0,
                                    }
                                ],
                            },
                        },
                    ],
                }
            ],
        }
        jsonschema.validate(instance=ir, schema=schema)

    def test_ir_reject_bad_status(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(IR_SCHEMA.read_text(encoding="utf-8"))
        ir = {
            "document_id": "d",
            "schema_version": "2.0",
            "pages": [
                {
                    "page_number": 1,
                    "width": 1.0,
                    "height": 1.0,
                    "blocks": [
                        {
                            "id": "b1",
                            "type": "text",
                            "text": "见图2",
                            "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
                            "page_number": 1,
                            "reading_order": 1,
                            "metadata": {
                                "figureReferences": [
                                    {
                                        "referenceId": "r",
                                        "rawLabel": "图2",
                                        "normalizedLabel": "图2",
                                        "status": "nope",
                                        "method": "exact_label_index",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=ir, schema=schema)


class TestMetricsAndGate(unittest.TestCase):
    def _kb(self, labels=0, refs=None, referenced=0):
        refs = refs or []
        blocks = []
        if labels:
            blocks.append(
                {
                    "id": "im1",
                    "type": "image",
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                    "figureLabels": [
                        {
                            "rawLabel": f"图{i}",
                            "normalizedLabel": f"图{i}",
                            "source": "ocr",
                            "method": "ocr_inside_image",
                            "confidence": 0.9,
                        }
                        for i in range(labels)
                    ],
                    **(
                        {
                            "referencedBy": [
                                {
                                    "referenceId": "r1",
                                    "sourceBlockId": "b",
                                    "sourcePageNumber": 1,
                                    "rawLabel": "图1",
                                    "normalizedLabel": "图1",
                                }
                            ]
                        }
                        if referenced
                        else {}
                    ),
                }
            )
        if refs:
            blocks.append(
                {
                    "id": "b1",
                    "type": "code",
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                    "figureReferences": refs,
                }
            )
        return {
            "fileMetadata": {"pageSizes": {"1": [1, 1]}},
            "entries": [{"entryId": "e", "contentNormalized": "x", "blocks": blocks}],
        }

    def test_v2_metrics_include_reference_keys(self):
        kb = self._kb(
            labels=2,
            refs=[
                {
                    "referenceId": "r1",
                    "rawLabel": "图2",
                    "normalizedLabel": "图2",
                    "status": "resolved",
                    "method": "exact_label_index",
                },
                {
                    "referenceId": "r2",
                    "rawLabel": "图4",
                    "normalizedLabel": "图4",
                    "status": "unresolved",
                    "method": "exact_label_index",
                    "reason": "no_label_in_index",
                },
            ],
            referenced=1,
        )
        m = kb_metrics_from_obj(kb, include_figure_association=True)
        for k in FIGURE_REFERENCE_METRIC_KEYS:
            self.assertIn(k, m, k)
        self.assertEqual(m["figure_labels_indexed"], 2)
        self.assertEqual(m["figure_references_found"], 2)
        self.assertEqual(m["figure_references_resolved"], 1)
        self.assertEqual(m["figure_references_unresolved"], 1)
        self.assertEqual(m["figure_references_ambiguous"], 0)
        self.assertEqual(m["referenced_figures"], 1)

    def test_legacy_metrics_omit_reference_keys(self):
        m = kb_metrics_from_obj(self._kb(), include_figure_association=False)
        for k in FIGURE_REFERENCE_METRIC_KEYS:
            self.assertNotIn(k, m, k)

    def test_shadow_skips_reference_keys(self):
        for k in FIGURE_REFERENCE_METRIC_KEYS:
            self.assertIn(k, SHADOW_COMPARABLE_KEYS, k)
        leg_m = kb_metrics_from_obj(
            self._kb(), include_figure_association=False
        )
        v2_m = kb_metrics_from_obj(
            self._kb(labels=1, refs=[], referenced=0),
            include_figure_association=True,
        )
        diff = build_shadow_diff(
            "r",
            {"status": "ok", "metrics": leg_m, "elapsed_ms": 1, "errors": []},
            {"status": "ok", "metrics": v2_m, "elapsed_ms": 1, "errors": []},
        )
        skipped = diff["diff"].get("skipped_keys") or []
        for k in FIGURE_REFERENCE_METRIC_KEYS:
            self.assertIn(k, skipped, k)
            self.assertNotIn(k, diff["diff"], k)

    def test_gate_consistent(self):
        errs = check_figure_reference_metrics(
            {
                "figure_labels_indexed": 4,
                "figure_references_found": 6,
                "figure_references_resolved": 5,
                "figure_references_unresolved": 1,
                "figure_references_ambiguous": 0,
                "referenced_figures": 3,
            },
            {
                "figure_labels_indexed": 4,
                "figure_references_found": 6,
                "figure_references_resolved": 5,
                "figure_references_unresolved": 1,
                "figure_references_ambiguous": 0,
                "referenced_figures": 3,
            },
            report_exists=True,
        )
        self.assertEqual(errs, [])

    def test_gate_mismatch_report_6_kb_5(self):
        kb = {
            "figure_labels_indexed": 4,
            "figure_references_found": 5,
            "figure_references_resolved": 5,
            "figure_references_unresolved": 0,
            "figure_references_ambiguous": 0,
            "referenced_figures": 3,
        }
        rep = dict(kb)
        rep["figure_references_found"] = 6
        errs = check_figure_reference_metrics(kb, rep, report_exists=True)
        self.assertTrue(errs)
        self.assertIn("figure_references_found mismatch: kb=5 report=6", errs[0])

    def test_gate_missing_report(self):
        errs = check_figure_reference_metrics(
            {"figure_labels_indexed": 0}, None, report_exists=False
        )
        self.assertTrue(errs)
        self.assertIn("report missing", errs[0])


class TestPage151NotConfused(unittest.TestCase):
    def test_fig11_ref_points_to_second_figure(self):
        """页151正文图11必须指向第二张 figure，不能误指图10."""
        f10 = _fig(
            "fig_151_a",
            151,
            x=22.0,
            y=35.0,
            w=213.0,
            h=80.0,
            labels=[_ocr_label("图10", 0.57)],
        )
        f11 = _fig(
            "fig_151_b",
            151,
            x=22.0,
            y=167.0,
            w=185.0,
            h=108.0,
            labels=[_ocr_label("图11", 0.73)],
        )
        # Body between/below figures referencing 图11
        body = _text(
            "b151",
            "骨折固定方法见图11，也可参照图10",
            151,
            y=300.0,
            x=30.0,
        )
        ir = _ir_with_pages([(151, [f10, f11, body])])
        report = resolve_figure_references(ir)
        refs = {r["normalizedLabel"]: r for r in body.metadata["figureReferences"]}
        self.assertEqual(refs["图11"]["status"], "resolved")
        self.assertEqual(refs["图11"]["targetFigureId"], "fig_151_b")
        self.assertEqual(refs["图10"]["status"], "resolved")
        self.assertEqual(refs["图10"]["targetFigureId"], "fig_151_a")
        # f11 has referencedBy for 图11 only
        back = f11.metadata.get("referencedBy") or []
        self.assertEqual(len(back), 1)
        self.assertEqual(back[0]["normalizedLabel"], "图11")


if __name__ == "__main__":
    unittest.main()
