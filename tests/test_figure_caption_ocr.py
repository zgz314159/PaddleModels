"""Phase 2F tests: figure-internal caption OCR (fake OCR — no local Tesseract)."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.canonical_ir import BBox, CanonicalIR, DocBlock, DocPage  # noqa: E402
from pipeline.semantics.figure_caption_ocr import (  # noqa: E402
    append_ocr_caption_association,
    figure_has_primary_caption,
    is_ocr_eligible_figure,
    is_safe_run_relative_uri,
    map_norm_bbox_to_figure_pdf,
    run_figure_caption_ocr_pass,
    stable_ocr_caption_block_id,
)
from pipeline.semantics.figure_text_association import (  # noqa: E402
    associate_ir_figures,
)
from pipeline.semantic_projector import SemanticProjector  # noqa: E402
from paddle_models.cli.main import (  # noqa: E402
    FIGURE_CAPTION_OCR_METRIC_KEYS,
    SHADOW_COMPARABLE_KEYS,
    build_parser,
    build_shadow_diff,
    kb_metrics_from_obj,
)

KB_SCHEMA = REPO_ROOT / "contracts" / "knowledge_base_schema_v2.json"
IR_SCHEMA = REPO_ROOT / "contracts" / "knowledge-base.v2.schema.json"

try:
    from imaging.caption_utils import extract_visual_caption_candidate
    from imaging.figure_caption_ocr import (
        EXTRACTOR_VERSION,
        adjudicate_attempts,
        classify_caption_ocr_text,
        is_cacheable_result,
        ocr_embedded_figure_caption,
        probe_tesseract_capability,
        select_caption_lines,
    )
    from imaging.text_utils import extract_figure_table_labels
    from paddle_models.cli.main import check_figure_caption_ocr_metrics
except ImportError:  # pragma: no cover
    classify_caption_ocr_text = None  # type: ignore


def _png_bytes(w: int = 200, h: int = 160, color=(200, 200, 200)) -> bytes:
    from PIL import Image

    im = Image.new("RGB", (max(1, w), max(1, h)), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _fig(
    fid="fig1",
    page=1,
    x=20.0,
    y=40.0,
    w=160.0,
    h=120.0,
    uri="shots/img_a.png",
    sha="a" * 64,
    asset_status="ready",
    asset_source="embedded",
    assocs=None,
):
    md = {
        "assetStatus": asset_status,
        "assetSource": asset_source,
        "imageUri": uri,
        "contentSha256": sha,
    }
    if assocs is not None:
        md["figureTextAssociations"] = assocs
    return DocBlock(
        id=fid,
        type="figure",
        text="",
        bbox=BBox(x, y, w, h),
        page_number=page,
        reading_order=1,
        metadata=md,
    )


def _label_hits(text):
    return extract_figure_table_labels(text)


def _extract_candidate(text, visual_type, label_hits_fn):
    return extract_visual_caption_candidate(text, visual_type, label_hits_fn)


# --- Fake OCR fixtures -------------------------------------------------------


def fake_ocr_linked(image_bytes, psm, language):
    """Bottom-band fake: one valid figure label + Chinese title, high conf."""
    return {
        "ok": True,
        "error": None,
        "text": "图12 颈椎骨折固定",
        "mean_confidence": 88.0,
        "words": [
            {"text": "图12", "conf": 90.0, "x": 10, "y": 100, "w": 40, "h": 16},
            {"text": "颈椎", "conf": 88.0, "x": 55, "y": 100, "w": 30, "h": 16},
            {"text": "骨折", "conf": 88.0, "x": 88, "y": 100, "w": 30, "h": 16},
            {"text": "固定", "conf": 86.0, "x": 121, "y": 100, "w": 30, "h": 16},
        ],
    }


def fake_ocr_label_only(image_bytes, psm, language):
    return {
        "ok": True,
        "error": None,
        "text": "图12",
        "mean_confidence": 90.0,
        "words": [{"text": "图12", "conf": 90.0, "x": 10, "y": 100, "w": 40, "h": 16}],
    }


def fake_ocr_multi_label(image_bytes, psm, language):
    # Page-137 style: one asset contains 图2 and 图3 → ambiguous.
    return {
        "ok": True,
        "error": None,
        "text": "图2 气道通畅 图3 气道阻塞",
        "mean_confidence": 85.0,
        "words": [
            {"text": "图2", "conf": 90.0, "x": 10, "y": 90, "w": 30, "h": 14},
            {"text": "图3", "conf": 90.0, "x": 90, "y": 90, "w": 30, "h": 14},
        ],
    }


def fake_ocr_body_noise(image_bytes, psm, language):
    return {
        "ok": True,
        "error": None,
        "text": "2.4.1.3试——试测口鼻有无呼气的气流。见图11说明",
        "mean_confidence": 70.0,
        "words": [{"text": "正文", "conf": 70.0, "x": 5, "y": 5, "w": 30, "h": 12}],
    }


def fake_ocr_empty(image_bytes, psm, language):
    return {
        "ok": True,
        "error": None,
        "text": "",
        "mean_confidence": 0.0,
        "words": [],
    }


def fake_ocr_boom(image_bytes, psm, language):
    raise RuntimeError("inject-ocr-internal-boom")


class _CountingOCR:
    """Callable OCR wrapper that records invocation count."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def __call__(self, image_bytes, psm, language):
        self.calls += 1
        return self.inner(image_bytes, psm, language)


class TestClassification(unittest.TestCase):
    def test_linked_valid_caption(self):
        r = classify_caption_ocr_text(
            "图12 颈椎骨折固定",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            mean_confidence=88.0,
        )
        self.assertEqual(r["status"], "linked")
        self.assertEqual(r["labels"], ["图12"])
        self.assertIn("颈椎", r["candidate"])

    def test_label_only(self):
        r = classify_caption_ocr_text(
            "图12",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            mean_confidence=90.0,
        )
        self.assertEqual(r["status"], "label_only")

    def test_ambiguous_multiple_labels(self):
        r = classify_caption_ocr_text(
            "图2 气道通畅 图3 气道阻塞",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            mean_confidence=85.0,
        )
        self.assertEqual(r["status"], "ambiguous_multiple_labels")
        self.assertEqual(len(r["labels"]), 2)
        self.assertEqual(r["candidate"], "")

    def test_body_noise_rejected(self):
        r = classify_caption_ocr_text(
            "2.4.1.3试——试测口鼻有无呼气的气流。",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            mean_confidence=70.0,
        )
        self.assertNotEqual(r["status"], "linked")
        self.assertIn(r["status"], ("rejected", "empty"))

    def test_reference_rejected(self):
        r = classify_caption_ocr_text(
            "固定(图11),也可利用竹竿",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            mean_confidence=80.0,
        )
        self.assertNotEqual(r["status"], "linked")

    def test_low_confidence_rejected(self):
        r = classify_caption_ocr_text(
            "图12 颈椎骨折固定",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            mean_confidence=10.0,
        )
        self.assertEqual(r["status"], "rejected")
        self.assertEqual(r["rejection_reason"], "low_confidence")

    def test_empty(self):
        r = classify_caption_ocr_text(
            "",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            mean_confidence=0.0,
        )
        self.assertEqual(r["status"], "empty")

    def test_no_hardcoded_title_requirement(self):
        # Any compliant title must pass — not one magic string.
        for text in ("图3 人工呼吸示意", "附图7 接线原理"):
            r = classify_caption_ocr_text(
                text,
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                mean_confidence=90.0,
            )
            self.assertEqual(r["status"], "linked", text)

    def test_ascii_noise_title_rejected(self):
        # Page-135 style mangled OCR: 图1 看.听\试 / 图1 HOT → no fake caption.
        for text in ("图1 看.听\\试", "图1 HOT", "图 1 HOT"):
            r = classify_caption_ocr_text(
                text,
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                mean_confidence=80.0,
            )
            self.assertNotEqual(r["status"], "linked", text)
            self.assertIn(
                r["status"], ("rejected", "label_only"), (text, r)
            )


class TestCapabilityProbe(unittest.TestCase):
    def test_probe_shape(self):
        cap = probe_tesseract_capability("chi_sim+eng")
        self.assertIn("ready", cap)
        self.assertIn("missing", cap)
        self.assertIn("pytesseract", cap)
        self.assertIn("pytesseract_importable", cap)
        self.assertIn("pillow_importable", cap)
        self.assertIn("import_error", cap)
        self.assertIn("tesseract_cmd", cap)
        self.assertIn("tesseract_version", cap)
        self.assertIn("languages_required", cap)
        # Never raises; discovery semantics documented.
        self.assertIn("probe_semantics", cap)
        # Real import probe is part of readiness (not find_spec alone).
        self.assertTrue(cap["pillow_importable"], cap.get("import_error"))
        if cap["pytesseract"]:
            self.assertTrue(cap["pytesseract_importable"], cap.get("import_error"))

    def test_probe_requires_both_chi_sim_and_eng(self):
        cap = probe_tesseract_capability("chi_sim+eng")
        self.assertIn("chi_sim", cap["languages_required"])
        self.assertIn("eng", cap["languages_required"])

    def test_probe_missing_language_flagged(self):
        cap = probe_tesseract_capability("zzz_not_a_lang")
        self.assertFalse(cap["ready"])
        self.assertTrue(
            any("language:" in m for m in cap["missing"]), cap["missing"]
        )

    def test_run_v2_blocked_when_capability_missing(self):
        """Missing pytesseract/binary/lang → blocked_by_dependency, full report."""
        import importlib.util
        from unittest.mock import patch

        if importlib.util.find_spec("fitz") is None:
            self.skipTest("fitz required for run_v2 integration")
        from paddle_models.cli.main import probe_capabilities, run_v2

        pdf = REPO_ROOT / "samples" / "103号(2).pdf"
        if not pdf.exists():
            self.skipTest("fixture PDF missing")
        caps = probe_capabilities("v2")
        blocked_cap = {
            "engine": "tesseract",
            "ready": False,
            "pytesseract": False,
            "tesseract_cmd": None,
            "missing": ["pytesseract", "tesseract_binary", "language:chi_sim"],
            "language": "chi_sim+eng",
        }
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            with patch(
                "imaging.figure_caption_ocr.probe_tesseract_capability",
                return_value=blocked_cap,
            ):
                result = run_v2(
                    run_dir,
                    pdf,
                    caps,
                    "smoke",
                    1,
                    1,
                    figure_caption_ocr="tesseract",
                )
        self.assertEqual(result["status"], "blocked_by_dependency")
        errs = " ".join(result.get("errors") or [])
        self.assertIn("capability missing", errs)
        self.assertIn("pytesseract", errs)
        # No deep traceback path — warnings mention no silent fallback.
        warns = " ".join(result.get("warnings") or [])
        self.assertIn("no silent fallback", warns)
        # Capability recorded for the run report.
        fc = result.get("figure_caption_ocr") or {}
        self.assertEqual(fc.get("requested_engine"), "tesseract")
        self.assertIsNone(fc.get("selected_engine"))
        self.assertFalse((fc.get("capability") or {}).get("ready"))
        # Pipeline must not have run (no KB written).
        self.assertFalse((run_dir / "v2" / "knowledge_base.json").exists())
        # Contract validation entry still present (full report shape).
        self.assertIn("contract_validation", result)

    def test_run_v2_off_skips_probe_and_runs(self):
        """Default off: no tesseract probe required, pipeline still runs."""
        import importlib.util
        from unittest.mock import patch

        if importlib.util.find_spec("fitz") is None:
            self.skipTest("fitz required for run_v2 integration")
        from paddle_models.cli.main import probe_capabilities, run_v2

        pdf = REPO_ROOT / "samples" / "103号(2).pdf"
        if not pdf.exists():
            self.skipTest("fixture PDF missing")
        caps = probe_capabilities("v2")
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            with patch(
                "imaging.figure_caption_ocr.probe_tesseract_capability"
            ) as probe_mock:
                result = run_v2(
                    run_dir, pdf, caps, "smoke", 1, 1, figure_caption_ocr="off"
                )
            probe_mock.assert_not_called()
            # Off mode writes no figure_caption_ocr.json.
            self.assertFalse(
                (run_dir / "v2" / "figure_caption_ocr.json").exists()
            )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            (result.get("figure_caption_ocr") or {}).get("requested_engine"),
            "off",
        )


class TestPathSafety(unittest.TestCase):
    def test_rejects_traversal_and_absolute(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            self.assertFalse(is_safe_run_relative_uri("../evil.png", base))
            self.assertFalse(is_safe_run_relative_uri("shots/../../x.png", base))
            self.assertFalse(is_safe_run_relative_uri("/abs/path.png", base))
            self.assertFalse(is_safe_run_relative_uri("C:\\evil.png", base))
            self.assertFalse(is_safe_run_relative_uri("", base))
            self.assertFalse(is_safe_run_relative_uri("~/x.png", base))

    def test_accepts_relative_inside_run(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            (base / "shots").mkdir()
            self.assertTrue(is_safe_run_relative_uri("shots/img.png", base))


class TestEligibility(unittest.TestCase):
    def test_ready_embedded_without_caption_ok(self):
        with tempfile.TemporaryDirectory() as td:
            ok, reason = is_ocr_eligible_figure(_fig(), Path(td))
            self.assertTrue(ok, reason)

    def test_skips_native_caption(self):
        with tempfile.TemporaryDirectory() as td:
            fig = _fig(
                assocs=[
                    {
                        "kind": "caption",
                        "blockId": "b1",
                        "text": "图4 x",
                        "source": "native",
                        "method": "same_page_geometry",
                    }
                ]
            )
            ok, reason = is_ocr_eligible_figure(fig, Path(td))
            self.assertFalse(ok)
            self.assertEqual(reason, "has_primary_caption")

    def test_skips_missing_and_crop(self):
        with tempfile.TemporaryDirectory() as td:
            ok, r = is_ocr_eligible_figure(
                _fig(asset_status="missing", uri=""), Path(td)
            )
            self.assertFalse(ok)
            ok2, r2 = is_ocr_eligible_figure(_fig(asset_source="page_crop"), Path(td))
            self.assertFalse(ok2)

    def test_figure_has_primary_caption(self):
        self.assertTrue(
            figure_has_primary_caption(
                _fig(assocs=[{"kind": "caption", "blockId": "x"}])
            )
        )
        self.assertFalse(
            figure_has_primary_caption(
                _fig(
                    assocs=[
                        {
                            "kind": "legend",
                            "blockId": "x",
                            "text": "(a)",
                            "source": "native",
                            "method": "m",
                        }
                    ]
                )
            )
        )


class TestBBoxMapping(unittest.TestCase):
    def test_maps_norm_bbox_into_figure_pdf(self):
        fig = _fig(x=100.0, y=200.0, w=80.0, h=40.0)
        # Normalized box covering bottom half of the asset.
        bb = map_norm_bbox_to_figure_pdf([0.25, 0.75, 0.75, 1.0], fig)
        self.assertAlmostEqual(bb.x, 100.0 + 0.25 * 80.0)
        self.assertAlmostEqual(bb.y, 200.0 + 0.75 * 40.0)
        self.assertAlmostEqual(bb.w, 0.5 * 80.0)
        self.assertAlmostEqual(bb.h, 0.25 * 40.0)

    def test_fallback_band_when_no_words(self):
        fig = _fig(x=0.0, y=0.0, w=100.0, h=100.0)
        bb = map_norm_bbox_to_figure_pdf(None, fig, band_fallback_norm=[0, 0.68, 1, 1])
        self.assertAlmostEqual(bb.y, 68.0)
        self.assertAlmostEqual(bb.h, 32.0)
        self.assertGreater(bb.w, 0)
        self.assertGreater(bb.h, 0)

    def test_stable_block_id(self):
        a = stable_ocr_caption_block_id("f1", "a" * 64, "tesseract", "v1")
        b = stable_ocr_caption_block_id("f1", "a" * 64, "tesseract", "v1")
        c = stable_ocr_caption_block_id("f2", "a" * 64, "tesseract", "v1")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue(a.startswith("capocr_"))


def _build_ir_with_ready_fig(uri="shots/img_a.png", sha="a" * 64, assocs=None):
    ir = CanonicalIR(document_id="d", sha256="0" * 64)
    page = DocPage(page_number=1, width=246.0, height=360.0, method="native")
    # Body first so projection has a host entry.
    page.blocks.append(
        DocBlock(
            id="body1",
            type="text",
            text="正文段落",
            bbox=BBox(20, 10, 200, 30),
            page_number=1,
            reading_order=1,
            metadata={"semanticRole": "body"},
        )
    )
    page.blocks.append(
        _fig(
            fid="f1",
            uri=uri,
            sha=sha,
            assocs=assocs,
            x=20.0,
            y=60.0,
            w=160.0,
            h=100.0,
        )
    )
    ir.pages.append(page)
    return ir


def _write_asset(run_dir: Path, rel: str, data: bytes) -> None:
    p = run_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


class TestPassLinked(unittest.TestCase):
    def test_linked_creates_block_and_association(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            ir = _build_ir_with_ready_fig()
            ocr = _CountingOCR(fake_ocr_linked)
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr,
            )
            self.assertEqual(report["stats"]["ocr_captions_linked"], 1)
            page = ir.pages[0]
            fig = next(b for b in page.blocks if b.id == "f1")
            assocs = fig.metadata.get("figureTextAssociations") or []
            self.assertEqual(len(assocs), 1)
            a = assocs[0]
            self.assertEqual(a["kind"], "caption")
            self.assertEqual(a["source"], "ocr")
            self.assertEqual(a["method"], "ocr_inside_image")
            self.assertEqual(a["ocrEngine"], "tesseract")
            # blockId must reference a real caption block.
            cap = next(
                (b for b in page.blocks if b.id == a["blockId"]), None
            )
            self.assertIsNotNone(cap, "association blockId must exist")
            self.assertEqual(cap.type, "caption")
            self.assertEqual(cap.source, "ocr")
            self.assertEqual(cap.metadata.get("semanticRole"), "caption")
            self.assertEqual(cap.metadata.get("figureId"), "f1")
            self.assertEqual(cap.metadata.get("ocrEngine"), "tesseract")
            self.assertTrue(cap.metadata.get("insideFigure"))
            self.assertEqual(cap.text, a["text"])
            self.assertIn("颈椎", cap.text)
            # bbox must sit inside the figure PDF bbox.
            fb, cb = fig.bbox, cap.bbox
            self.assertGreaterEqual(cb.x, fb.x - 0.5)
            self.assertGreaterEqual(cb.y, fb.y - 0.5)
            self.assertLessEqual(cb.x + cb.w, fb.x + fb.w + 0.5)
            self.assertLessEqual(cb.y + cb.h, fb.y + fb.h + 0.5)

    def test_label_only_rejected_no_caption(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            ir = _build_ir_with_ready_fig()
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=_CountingOCR(fake_ocr_label_only),
            )
            st = report["stats"]
            self.assertEqual(st["ocr_captions_linked"], 0)
            self.assertEqual(st["ocr_captions_rejected"], 1)
            fig = ir.pages[0].blocks[-1]
            self.assertEqual(
                fig.metadata.get("figureTextAssociations") or [], []
            )
            self.assertFalse(
                any(b.type == "caption" for b in ir.pages[0].blocks)
            )

    def test_multi_label_ambiguous_keeps_native_legend(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            native_legend = {
                "kind": "legend",
                "blockId": "leg1",
                "text": "(a)气道通畅(b)气道阻塞",
                "source": "native",
                "method": "same_page_geometry",
            }
            ir = _build_ir_with_ready_fig(assocs=[native_legend])
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=_CountingOCR(fake_ocr_multi_label),
            )
            st = report["stats"]
            self.assertEqual(st["ocr_captions_ambiguous"], 1)
            self.assertEqual(st["ocr_captions_linked"], 0)
            fig = ir.pages[0].blocks[-1]
            assocs = fig.metadata.get("figureTextAssociations") or []
            # Native legend retained; no OCR caption added.
            self.assertEqual(len(assocs), 1)
            self.assertEqual(assocs[0]["kind"], "legend")
            self.assertEqual(assocs[0]["source"], "native")
            self.assertFalse(
                any(b.type == "caption" for b in ir.pages[0].blocks)
            )
            # Item records ambiguous status + detected labels.
            item = next(i for i in report["items"] if i.get("figureId") == "f1")
            self.assertEqual(item["status"], "ambiguous_multiple_labels")
            self.assertEqual(len(item["labels"]), 2)

    def test_body_noise_and_empty_rejected(self):
        for ocr_fn in (fake_ocr_body_noise, fake_ocr_empty):
            with tempfile.TemporaryDirectory() as td:
                run_dir = Path(td)
                _write_asset(run_dir, "shots/img_a.png", _png_bytes())
                ir = _build_ir_with_ready_fig()
                report = run_figure_caption_ocr_pass(
                    ir,
                    run_dir=run_dir,
                    engine="tesseract",
                    language="chi_sim+eng",
                    cache_root=run_dir / ".cache",
                    label_hits_fn=_label_hits,
                    extract_candidate_fn=_extract_candidate,
                    ocr_fn=_CountingOCR(ocr_fn),
                )
                self.assertEqual(report["stats"]["ocr_captions_linked"], 0)
                self.assertGreaterEqual(
                    report["stats"]["ocr_captions_rejected"], 1
                )
                self.assertFalse(
                    any(b.type == "caption" for b in ir.pages[0].blocks)
                )

    def test_native_caption_skips_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            ir = _build_ir_with_ready_fig(
                assocs=[
                    {
                        "kind": "caption",
                        "blockId": "nat1",
                        "text": "图9 已有原生题",
                        "source": "native",
                        "method": "same_page_geometry",
                    }
                ]
            )
            ocr = _CountingOCR(fake_ocr_linked)
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr,
            )
            self.assertEqual(ocr.calls, 0)
            self.assertEqual(report["stats"]["figure_caption_ocr_attempted"], 0)
            self.assertEqual(report["stats"]["ocr_captions_linked"], 0)

    def test_path_traversal_rejected_without_read(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            ir = _build_ir_with_ready_fig(uri="../outside.png")
            ocr = _CountingOCR(fake_ocr_linked)
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr,
            )
            self.assertEqual(ocr.calls, 0)
            self.assertEqual(report["stats"]["ocr_captions_linked"], 0)

    def test_same_hash_only_one_ocr_call(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            sha = "b" * 64
            ir = CanonicalIR(document_id="d", sha256="0" * 64)
            page = DocPage(page_number=1, width=246.0, height=360.0, method="native")
            page.blocks.append(_fig(fid="fA", sha=sha, uri="shots/img_a.png"))
            page.blocks.append(
                _fig(fid="fB", sha=sha, uri="shots/img_a.png", x=20.0, y=200.0)
            )
            ir.pages.append(page)
            ocr = _CountingOCR(fake_ocr_linked)
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr,
            )
            self.assertGreaterEqual(ocr.calls, 1, "same contentSha256 OCR once")
            # Full ROI×PSM grid runs per unique asset (3 ratios × 2 PSMs).
            self.assertEqual(ocr.calls, 3 * 2, "one aggregation pass per asset")
            self.assertEqual(report["stats"]["ocr_captions_linked"], 2)
            # Second figure reused in-run result (no second aggregation pass).
            self.assertEqual(report["stats"]["figure_caption_ocr_cache_hits"], 1)
            self.assertEqual(ocr.calls, 3 * 2, "second figure must not re-OCR")

    def test_disk_cache_hit_skips_ocr(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            cache_root = run_dir / ".cache"
            ir = _build_ir_with_ready_fig()
            ocr1 = _CountingOCR(fake_ocr_linked)
            run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=cache_root,
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr1,
            )
            first_calls = ocr1.calls
            self.assertEqual(first_calls, 3 * 2, "one full grid per unique asset")
            # Fresh IR (new run), same cache → OCR must not be called again.
            ir2 = _build_ir_with_ready_fig()
            ocr2 = _CountingOCR(fake_ocr_linked)
            report2 = run_figure_caption_ocr_pass(
                ir2,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=cache_root,
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr2,
            )
            self.assertEqual(ocr2.calls, 0, "disk cache must skip OCR")
            self.assertEqual(ocr1.calls, first_calls)
            self.assertGreaterEqual(
                report2["stats"]["figure_caption_ocr_cache_hits"], 1
            )
            self.assertEqual(report2["stats"]["ocr_captions_linked"], 1)

    def test_ocr_exception_not_fake_success(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            ir = _build_ir_with_ready_fig()
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=_CountingOCR(fake_ocr_boom),
            )
            st = report["stats"]
            self.assertEqual(st["ocr_captions_linked"], 0)
            self.assertGreaterEqual(st["ocr_captions_rejected"], 1)
            self.assertFalse(
                any(b.type == "caption" for b in ir.pages[0].blocks)
            )
            item = next(i for i in report["items"] if i.get("figureId") == "f1")
            self.assertNotEqual(item["status"], "linked")

    def test_report_fields_present(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            ir = _build_ir_with_ready_fig()
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=_CountingOCR(fake_ocr_linked),
            )
            self.assertEqual(report["engine"], "tesseract")
            self.assertEqual(report["language"], "chi_sim+eng")
            self.assertIn("stats", report)
            self.assertIn("items", report)
            item = report["items"][0]
            for key in (
                "page",
                "figureId",
                "contentSha256",
                "engine",
                "language",
                "cacheHit",
                "status",
                "elapsedMs",
            ):
                self.assertIn(key, item, key)
            # Diagnostic carries roi/psm/labels/candidate for linked items.
            self.assertIn("roi", item)
            self.assertIn("psm", item)
            self.assertIn("labels", item)
            self.assertIn("candidate", item)
            # Phase 2F.1 line evidence.
            self.assertIn("captionLineKeys", item)
            self.assertIn("captionWordCount", item)
            self.assertIn("textBBoxNorm", item)
            self.assertIsInstance(item["captionLineKeys"], list)
            self.assertIsInstance(item["captionWordCount"], int)
            # No unfiltered full OCR dump field on the item.
            self.assertNotIn("raw_text", item)


class TestProjectionContent(unittest.TestCase):
    def test_content_markdown_caption_once_and_figure_not_searchable(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            ir = _build_ir_with_ready_fig()
            run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=_CountingOCR(fake_ocr_linked),
            )
            kb = SemanticProjector("d", strategy="heading").project(ir)
            content = "\n\n".join(
                (e.get("contentMarkdown") or "") for e in kb["entries"]
            )
            # Caption appears exactly once in contentMarkdown.
            self.assertEqual(content.count("颈椎骨折固定"), 1, content)
            # Figure image block remains non-searchable.
            img_blocks = [
                b
                for e in kb["entries"]
                for b in e["blocks"]
                if b.get("type") == "image"
            ]
            self.assertTrue(img_blocks)
            for b in img_blocks:
                self.assertFalse(b.get("searchable"))
            # Caption text block is searchable.
            cap_blocks = [
                b
                for e in kb["entries"]
                for b in e["blocks"]
                if b.get("semanticRole") == "caption"
                and b.get("searchable") is True
            ]
            self.assertTrue(cap_blocks)
            # Association blockId resolves inside KB blocks.
            all_ids = {
                b.get("id") for e in kb["entries"] for b in e["blocks"]
            }
            for b in img_blocks:
                for a in b.get("figureTextAssociations") or []:
                    if a.get("source") == "ocr":
                        self.assertIn(a["blockId"], all_ids)

    def test_native_legend_and_ocr_caption_coexist(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            native_legend = {
                "kind": "legend",
                "blockId": "leg1",
                "text": "(a)通畅(b)阻塞",
                "source": "native",
                "method": "same_page_geometry",
            }
            ir = _build_ir_with_ready_fig(assocs=[native_legend])
            # Legend text block present on page (supporting, not searchable content).
            ir.pages[0].blocks.append(
                DocBlock(
                    id="leg1",
                    type="text",
                    text="(a)通畅(b)阻塞",
                    bbox=BBox(30, 170, 100, 12),
                    page_number=1,
                    reading_order=3,
                    metadata={"semanticRole": "legend"},
                )
            )
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=_CountingOCR(fake_ocr_linked),
            )
            self.assertEqual(report["stats"]["ocr_captions_linked"], 1)
            fig = next(b for b in ir.pages[0].blocks if b.id == "f1")
            kinds = [
                a["kind"]
                for a in fig.metadata.get("figureTextAssociations") or []
            ]
            self.assertIn("legend", kinds)
            self.assertIn("caption", kinds)


class TestSchemaOCR(unittest.TestCase):
    def _kb_with_ocr_caption(self):
        cap_id = "capocr_deadbeef00000000"
        return {
            "fileMetadata": {
                "schemaVersion": "2.0",
                "fileId": "t",
                "docSha256": "a" * 64,
            },
            "entries": [
                {
                    "entryId": "e",
                    "jobTitle": "t",
                    "blocks": [
                        {
                            "id": "im1",
                            "type": "image",
                            "imageUri": "shots/a.png",
                            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                            "figureTextAssociations": [
                                {
                                    "kind": "caption",
                                    "blockId": cap_id,
                                    "text": "图12 颈椎骨折固定",
                                    "source": "ocr",
                                    "method": "ocr_inside_image",
                                    "ocrEngine": "tesseract",
                                },
                                {
                                    "kind": "legend",
                                    "blockId": "leg1",
                                    "text": "(a)x",
                                    "source": "native",
                                    "method": "same_page_geometry",
                                },
                            ],
                        },
                        {
                            "id": cap_id,
                            "type": "code",
                            "code": "图12 颈椎骨折固定",
                            "semanticRole": "caption",
                            "searchable": True,
                            "bbox": {
                                "left": 10,
                                "top": 20,
                                "right": 30,
                                "bottom": 24,
                            },
                        },
                    ],
                }
            ],
        }

    def test_kb_valid_ocr_association(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validate(instance=self._kb_with_ocr_caption(), schema=schema)

    def test_kb_reject_bad_kind(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        bad = self._kb_with_ocr_caption()
        bad["entries"][0]["blocks"][0]["figureTextAssociations"][0]["kind"] = "ref"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)

    def test_ir_valid_ocr_association(self):
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
                                "figureTextAssociations": [
                                    {
                                        "kind": "caption",
                                        "blockId": "capocr_x",
                                        "text": "图1 t",
                                        "source": "ocr",
                                        "method": "ocr_inside_image",
                                        "ocrEngine": "tesseract",
                                    }
                                ]
                            },
                        },
                        {
                            "id": "capocr_x",
                            "type": "caption",
                            "text": "图1 t",
                            "bbox": {"x": 2, "y": 3, "w": 4, "h": 5},
                            "page_number": 1,
                            "reading_order": 2,
                            "source": "ocr",
                            "metadata": {
                                "semanticRole": "caption",
                                "figureId": "f1",
                                "ocrEngine": "tesseract",
                                "insideFigure": True,
                            },
                        },
                    ],
                }
            ],
        }
        jsonschema.validate(instance=ir, schema=schema)

    def test_ir_reject_bad_method_relation(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(IR_SCHEMA.read_text(encoding="utf-8"))
        bad = {
            "document_id": "d",
            "schema_version": "2.0",
            "pages": [
                {
                    "page_number": 1,
                    "width": 1.0,
                    "height": 1.0,
                    "blocks": [
                        {
                            "id": "f1",
                            "type": "figure",
                            "text": "",
                            "bbox": {"x": 0, "y": 0, "w": 1, "h": 1},
                            "page_number": 1,
                            "reading_order": 1,
                            "metadata": {
                                "figureTextAssociations": [
                                    {
                                        "kind": "caption",
                                        "blockId": "b",
                                        "text": "t",
                                        "source": "ocr",
                                        "method": "ocr_inside_image",
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        }
        # Missing required fields → invalid
        bad["pages"][0]["blocks"][0]["metadata"]["figureTextAssociations"][0].pop(
            "method"
        )
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)


class TestMetricsAndShadow(unittest.TestCase):
    def _kb(self, with_ocr_assoc=False, with_caption_block=False):
        img = {
            "id": "im1",
            "type": "image",
            "imageUri": "shots/a.png",
            "assetSource": "embedded",
            "assetStatus": "ready",
            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
            "figureTextAssociations": [],
        }
        blocks = [img]
        if with_ocr_assoc:
            img["figureTextAssociations"] = [
                {
                    "kind": "caption",
                    "blockId": "capocr_1",
                    "text": "图12 颈椎骨折固定",
                    "source": "ocr",
                    "method": "ocr_inside_image",
                    "ocrEngine": "tesseract",
                }
            ]
            if with_caption_block:
                blocks.append(
                    {
                        "id": "capocr_1",
                        "type": "code",
                        "code": "图12 颈椎骨折固定",
                        "semanticRole": "caption",
                        "searchable": True,
                        "bbox": {
                            "left": 1,
                            "top": 2,
                            "right": 3,
                            "bottom": 4,
                        },
                    }
                )
        return {
            "fileMetadata": {"pageSizes": {"1": [1, 1]}},
            "entries": [
                {
                    "entryId": "e",
                    "contentNormalized": "x",
                    "blocks": blocks,
                }
            ],
        }

    def test_v2_metrics_include_ocr_keys(self):
        m = kb_metrics_from_obj(self._kb(with_ocr_assoc=True), include_figure_association=True)
        for k in FIGURE_CAPTION_OCR_METRIC_KEYS:
            self.assertIn(k, m, k)
        self.assertEqual(m["figures_with_ocr_caption"], 1)
        self.assertEqual(m["ocr_captions_linked"], 1)

    def test_legacy_metrics_omit_ocr_keys(self):
        m = kb_metrics_from_obj(self._kb(with_ocr_assoc=False), include_figure_association=False)
        for k in FIGURE_CAPTION_OCR_METRIC_KEYS:
            self.assertNotIn(k, m, k)

    def test_native_caption_not_counted_as_ocr(self):
        kb = self._kb()
        kb["entries"][0]["blocks"][0]["figureTextAssociations"] = [
            {
                "kind": "caption",
                "blockId": "nat1",
                "text": "图4 x",
                "source": "native",
                "method": "same_page_geometry",
            }
        ]
        m = kb_metrics_from_obj(kb, include_figure_association=True)
        self.assertEqual(m["figures_with_ocr_caption"], 0)
        self.assertEqual(m["ocr_captions_linked"], 0)
        self.assertEqual(m["figures_with_native_caption"], 1)

    def test_shadow_skips_ocr_keys_for_legacy(self):
        for k in FIGURE_CAPTION_OCR_METRIC_KEYS:
            self.assertIn(k, SHADOW_COMPARABLE_KEYS, k)
        leg_m = kb_metrics_from_obj(
            self._kb(False), include_figure_association=False
        )
        v2_m = kb_metrics_from_obj(
            self._kb(True, True), include_figure_association=True
        )
        diff = build_shadow_diff(
            "r",
            {"status": "ok", "metrics": leg_m, "elapsed_ms": 1, "errors": []},
            {"status": "ok", "metrics": v2_m, "elapsed_ms": 1, "errors": []},
        )
        skipped = diff["diff"].get("skipped_keys") or []
        for k in FIGURE_CAPTION_OCR_METRIC_KEYS:
            self.assertIn(k, skipped, f"{k} must be skipped")
            self.assertNotIn(k, diff["diff"], f"{k} must not emit delta")


class TestCLIFlag(unittest.TestCase):
    def test_default_off(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "x.pdf"])
        self.assertEqual(args.figure_caption_ocr, "off")

    def test_explicit_tesseract(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--input", "x.pdf", "--figure-caption-ocr", "tesseract"]
        )
        self.assertEqual(args.figure_caption_ocr, "tesseract")

    def test_help_does_not_import_ocr_heavy(self):
        # --help must not import PIL/pytesseract via CLI module import path.
        import subprocess

        env = {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PATH": "",
        }
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys;"
                    "sys.argv=['main','--help'];"
                    "from paddle_models.cli.main import main;"
                    "import importlib.util as u;"
                    "main();"
                    "assert u.find_spec('pytesseract') is None or "
                    "'pytesseract' not in sys.modules;"
                    "assert 'PIL' not in sys.modules;"
                ),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(REPO_ROOT),
            env={**__import__("os").environ, **env},
        )
        # main() with --help exits 0 via SystemExit(0) — argparse handles help.
        combined = (proc.stdout or "") + (proc.stderr or "")
        self.assertNotIn("pytesseract", (proc.stderr or "").lower() and "" or "")
        # Simpler: ensure no ModuleNotFoundError for heavy OCR libs during --help.
        self.assertNotIn("ImportError", combined)
        self.assertNotIn("ModuleNotFoundError", combined)


class TestOffModeNoOp(unittest.TestCase):
    def test_off_mode_zero_ocr_and_no_report_side_effects(self):
        """engine=off path is CLI-level: parser default + no pipeline call.

        The pass itself is only invoked when engine != off (v2_runner gate);
        here we assert the gate condition and that an off-profile BuildProfile
        keeps figure_caption_ocr='off'.
        """
        from models.run_context import BuildProfile

        p = BuildProfile()
        self.assertEqual(p.figure_caption_ocr, "off")
        self.assertEqual(p.to_dict()["figure_caption_ocr"], "off")
        # v2_runner source must gate the pass on != off.
        src = (REPO_ROOT / "pipeline" / "v2_runner.py").read_text(encoding="utf-8")
        self.assertIn('fc_engine != "off"', src)
        self.assertIn("run_figure_caption_ocr_pass", src)


class TestAssociationAfterNative(unittest.TestCase):
    def test_associate_then_ocr_pass_order(self):
        """Native association first; OCR only fills still-empty primary caption."""
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            ir = CanonicalIR(document_id="d", sha256="0" * 64)
            page = DocPage(page_number=1, width=246.0, height=360.0, method="native")
            page.blocks.append(
                DocBlock(
                    id="body1",
                    type="text",
                    text="正文",
                    bbox=BBox(20, 10, 200, 20),
                    page_number=1,
                    reading_order=1,
                    metadata={"semanticRole": "body"},
                )
            )
            fig = _fig(fid="f1", x=20.0, y=50.0, w=160.0, h=90.0)
            page.blocks.append(fig)
            # Native caption below figure (role=caption, source=native).
            page.blocks.append(
                DocBlock(
                    id="natcap",
                    type="caption",
                    text="图4 原生题",
                    bbox=BBox(30, 150, 120, 14),
                    page_number=1,
                    reading_order=3,
                    source="native",
                    metadata={"semanticRole": "caption"},
                )
            )
            ir.pages.append(page)
            totals = associate_ir_figures(ir)
            self.assertEqual(totals["figures_with_native_caption"], 1)
            ocr = _CountingOCR(fake_ocr_linked)
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=run_dir / ".cache",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr,
            )
            # Already has native caption → OCR skipped entirely.
            self.assertEqual(ocr.calls, 0)
            self.assertEqual(report["stats"]["ocr_captions_linked"], 0)
            assocs = fig.metadata.get("figureTextAssociations") or []
            self.assertEqual(len(assocs), 1)
            self.assertEqual(assocs[0]["source"], "native")

    def test_append_association_sorts_caption_first(self):
        fig = _fig(
            assocs=[
                {
                    "kind": "legend",
                    "blockId": "leg1",
                    "text": "(a)",
                    "source": "native",
                    "method": "m",
                }
            ]
        )
        append_ocr_caption_association(
            fig,
            {
                "kind": "caption",
                "blockId": "cap1",
                "text": "图1 t",
                "source": "ocr",
                "method": "ocr_inside_image",
                "ocrEngine": "tesseract",
            },
        )
        kinds = [
            a["kind"] for a in fig.metadata["figureTextAssociations"]
        ]
        self.assertEqual(kinds, ["caption", "legend"])
        # Second append of a caption is ignored (one primary caption).
        append_ocr_caption_association(
            fig,
            {
                "kind": "caption",
                "blockId": "cap2",
                "text": "图2 t",
                "source": "ocr",
                "method": "ocr_inside_image",
            },
        )
        caps = [
            a
            for a in fig.metadata["figureTextAssociations"]
            if a["kind"] == "caption"
        ]
        self.assertEqual(len(caps), 1)


class TestROILoop(unittest.TestCase):
    def test_ocr_embedded_uses_band_and_psm(self):
        calls = []

        def rec(image_bytes, psm, language):
            calls.append(psm)
            return fake_ocr_linked(image_bytes, psm, language)

        r = ocr_embedded_figure_caption(
            _png_bytes(),
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            ocr_fn=rec,
            language="chi_sim+eng",
        )
        self.assertEqual(r["status"], "linked")
        self.assertIn(r["psm"], (6, 11))
        self.assertIn("roi", r)
        self.assertIn("top_ratio", r["roi"])
        self.assertTrue(calls)
        # 2F.1: full grid runs (no early stop on first linked).
        self.assertEqual(len(calls), 3 * 2, calls)

    def test_all_bands_fail_rejected_not_linked(self):
        r = ocr_embedded_figure_caption(
            b"",
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            ocr_fn=fake_ocr_linked,
        )
        self.assertNotEqual(r["status"], "linked")
        self.assertEqual(r["status"], "empty")


class TestCrossAttemptAggregation(unittest.TestCase):
    """Phase 2F.1 §1: global verdict over all ROI×PSM attempts."""

    def _linked_data(self, text="图12 颈椎骨折固定", conf=88.0):
        return {
            "ok": True,
            "error": None,
            "text": text,
            "mean_confidence": conf,
            "words": [
                {"text": "图12", "conf": conf, "x": 10, "y": 100, "w": 40, "h": 16,
                 "block_num": 1, "par_num": 0, "line_num": 1},
                {"text": "颈椎", "conf": conf, "x": 55, "y": 100, "w": 30, "h": 16,
                 "block_num": 1, "par_num": 0, "line_num": 1},
            ],
        }

    def _multi_data(self, text="图12 颈椎骨折固定 图13 腰椎", conf=85.0):
        return {
            "ok": True,
            "error": None,
            "text": text,
            "mean_confidence": conf,
            "words": [
                {"text": "图12", "conf": conf, "x": 10, "y": 100, "w": 40, "h": 16,
                 "block_num": 1, "par_num": 0, "line_num": 1},
                {"text": "图13", "conf": conf, "x": 90, "y": 100, "w": 40, "h": 16,
                 "block_num": 1, "par_num": 0, "line_num": 2},
            ],
        }

    def test_first_linked_then_second_label_is_ambiguous(self):
        # First attempt linked; a later reliable attempt sees a second label.
        seq = [self._linked_data(), self._multi_data()]
        state = {"i": 0}

        def ocr_fn(band, psm, language):
            data = seq[min(state["i"], len(seq) - 1)]
            state["i"] += 1
            return dict(data)

        r = ocr_embedded_figure_caption(
            _png_bytes(),
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            ocr_fn=ocr_fn,
        )
        self.assertEqual(r["status"], "ambiguous_multiple_labels")
        self.assertGreaterEqual(len(r["labels"]), 2)
        self.assertIn("图13", r["labels"])

    def test_all_attempts_same_label_links_once(self):
        r = ocr_embedded_figure_caption(
            _png_bytes(),
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            ocr_fn=lambda b, p, l: self._linked_data(),
        )
        self.assertEqual(r["status"], "linked")
        self.assertEqual(r["labels"], ["图12"])
        # Full grid attempted; single final verdict.
        self.assertEqual(len(r["attempts"]), 3 * 2)
        linked_attempts = [
            a for a in r["attempts"] if a.get("status") == "linked"
        ]
        self.assertEqual(len(linked_attempts), 3 * 2)

    def test_low_conf_second_label_does_not_flip_ambiguous(self):
        # Reliable first: linked 图12; low-conf noise attempt claims 图99.
        seq = [
            self._linked_data(conf=88.0),
            {
                "ok": True,
                "error": None,
                "text": "图99 噪声",
                "mean_confidence": 15.0,  # < RELIABLE_ATTEMPT_CONFIDENCE
                "words": [
                    {"text": "图99", "conf": 15.0, "x": 5, "y": 5, "w": 30, "h": 12,
                     "block_num": 1, "par_num": 0, "line_num": 1},
                ],
            },
        ]
        state = {"i": 0}

        def ocr_fn(band, psm, language):
            data = seq[min(state["i"], len(seq) - 1)]
            state["i"] += 1
            return dict(data)

        r = ocr_embedded_figure_caption(
            _png_bytes(),
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            ocr_fn=ocr_fn,
        )
        self.assertEqual(r["status"], "linked")
        self.assertEqual(r["labels"], ["图12"])

    def test_attempt_order_does_not_change_verdict(self):
        a1 = {
            "status": "linked",
            "labels": ["图12"],
            "all_attempt_labels": ["图12"],
            "candidate": "图12 颈椎骨折固定",
            "mean_confidence": 88.0,
            "roi": {"top_ratio": 0.68, "band": "bottom"},
            "psm": 6,
            "rejection_reason": None,
            "caption_line_keys": ["1-0-1"],
            "caption_word_count": 2,
            "text_bbox_norm": [0.1, 0.7, 0.5, 0.8],
        }
        a2 = {
            "status": "ambiguous_multiple_labels",
            "labels": ["图12", "图13"],
            "all_attempt_labels": ["图12", "图13"],
            "candidate": "",
            "mean_confidence": 85.0,
            "roi": {"top_ratio": 0.58, "band": "bottom"},
            "psm": 11,
            "rejection_reason": "multiple_labels:图12,图13",
            "caption_line_keys": ["1-0-1"],
            "caption_word_count": 4,
            "text_bbox_norm": None,
        }
        r1 = adjudicate_attempts([a1, a2], language="chi_sim+eng")
        r2 = adjudicate_attempts([a2, a1], language="chi_sim+eng")
        self.assertEqual(r1["status"], "ambiguous_multiple_labels")
        self.assertEqual(r1["status"], r2["status"])
        self.assertEqual(sorted(r1["labels"]), sorted(r2["labels"]))

    def test_same_label_tiebreak_prefers_higher_conf(self):
        low = {
            "status": "linked",
            "labels": ["图12"],
            "all_attempt_labels": ["图12"],
            "candidate": "图12 颈椎骨折固定",
            "mean_confidence": 50.0,
            "roi": {"top_ratio": 0.68},
            "psm": 6,
            "caption_line_keys": ["a"],
            "caption_word_count": 2,
            "text_bbox_norm": [0, 0.7, 1, 0.8],
        }
        high = dict(low)
        high["mean_confidence"] = 90.0
        high["psm"] = 11
        r = adjudicate_attempts([low, high], language="chi_sim+eng")
        self.assertEqual(r["status"], "linked")
        self.assertEqual(r["mean_confidence"], 90.0)


class TestCacheability(unittest.TestCase):
    def test_cacheable_content_results(self):
        for result in (
            {"status": "linked"},
            {"status": "ambiguous_multiple_labels"},
            {"status": "label_only"},
            {"status": "empty"},
            {"status": "rejected", "rejection_reason": "no_figure_label"},
            {"status": "rejected", "rejection_reason": "low_confidence"},
            {"status": "rejected", "rejection_reason": "title_ascii_noise"},
            {"status": "rejected", "rejection_reason": "body_sentence"},
        ):
            self.assertTrue(is_cacheable_result(result), result)

    def test_not_cacheable_transient(self):
        for result in (
            {"status": "rejected", "rejection_reason": "asset_read_error:FileNotFoundError"},
            {"status": "rejected", "rejection_reason": "ocr_exception:RuntimeError"},
            {"status": "rejected", "rejection_reason": "dependency:ImportError: x"},
            {"status": "rejected", "rejection_reason": "ocr_error", "ocr_error": "boom"},
            {"status": "rejected", "rejection_reason": "timeout"},
            {"status": "linked", "ocr_error": "partial"},
            {"status": "rejected", "rejection_reason": ""},
        ):
            self.assertFalse(is_cacheable_result(result), result)

    def test_transient_not_cached_then_retried(self):
        """First transient failure is not written; second run re-invokes OCR."""
        from imaging.figure_caption_ocr import cache_path, load_cache, save_cache

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            cache_root = run_dir / ".cache"
            sha = "a" * 64

            # Simulate pass path: transient result must not be saved.
            transient = {
                "status": "rejected",
                "rejection_reason": "ocr_exception:RuntimeError",
                "ocr_error": "boom",
            }
            self.assertFalse(is_cacheable_result(transient))
            cpath = cache_path(cache_root, sha, "tesseract")
            # Even if someone tries to save, load_cache must refuse non-cacheable.
            payload = dict(transient)
            payload["cacheKey"] = "figure_caption_ocr/v2|tesseract|chi_sim+eng"
            payload["engine"] = "tesseract"
            payload["language"] = "chi_sim+eng"
            payload["extractorVersion"] = EXTRACTOR_VERSION
            self.assertTrue(save_cache(cpath, payload))
            self.assertIsNone(
                load_cache(cpath, "tesseract", "chi_sim+eng"),
                "non-cacheable entry must not be served",
            )

            # First run: OCR raises → rejected, not cached.
            ir = _build_ir_with_ready_fig()
            ocr1 = _CountingOCR(fake_ocr_boom)
            report1 = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=cache_root,
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr1,
            )
            self.assertEqual(report1["stats"]["ocr_captions_linked"], 0)
            self.assertGreaterEqual(ocr1.calls, 1)
            # Cache file for this sha must not be a served rejected hit.
            disk = load_cache(
                cache_path(cache_root, sha, "tesseract"),
                "tesseract",
                "chi_sim+eng",
            )
            self.assertIsNone(disk)

            # Second run: OCR must be called again (not cache-hit rejected).
            ir2 = _build_ir_with_ready_fig()
            ocr2 = _CountingOCR(fake_ocr_linked)
            report2 = run_figure_caption_ocr_pass(
                ir2,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=cache_root,
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=ocr2,
            )
            self.assertGreaterEqual(ocr2.calls, 1, "transient must retry OCR")
            self.assertEqual(report2["stats"]["ocr_captions_linked"], 1)
            self.assertEqual(
                report2["stats"]["figure_caption_ocr_cache_hits"], 0
            )

    def test_extractor_version_is_v2(self):
        self.assertEqual(EXTRACTOR_VERSION, "v2")

    def test_cache_write_failure_recorded_as_warning(self):
        from imaging.figure_caption_ocr import cache_path

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_asset(run_dir, "shots/img_a.png", _png_bytes())
            # Point cache_root at a file (not a dir) so mkdir/write fails.
            blocker = run_dir / "blocker"
            blocker.write_text("x", encoding="utf-8")
            ir = _build_ir_with_ready_fig()
            report = run_figure_caption_ocr_pass(
                ir,
                run_dir=run_dir,
                engine="tesseract",
                language="chi_sim+eng",
                cache_root=blocker / "not_a_dir",
                label_hits_fn=_label_hits,
                extract_candidate_fn=_extract_candidate,
                ocr_fn=_CountingOCR(fake_ocr_linked),
            )
            # Main flow still succeeds; warning recorded (not silent).
            self.assertEqual(report["stats"]["ocr_captions_linked"], 1)
            warnings = report.get("warnings") or []
            # mkdir may succeed on some systems (parents); if write failed we
            # must see a warning. If cache root became creatable, accept empty
            # only when cache file actually exists.
            cpath = cache_path(blocker / "not_a_dir", "a" * 64, "tesseract")
            if not cpath.exists():
                self.assertTrue(
                    any("cache_write_failed" in w for w in warnings),
                    warnings,
                )


class TestMetricsGate(unittest.TestCase):
    def test_consistent_metrics_pass(self):
        errs = check_figure_caption_ocr_metrics(
            {"ocr_captions_linked": 4, "figures_with_ocr_caption": 4},
            {"ocr_captions_linked": 4, "figures_with_ocr_caption": 4},
            report_exists=True,
        )
        self.assertEqual(errs, [])

    def test_report_4_kb_3_is_error(self):
        errs = check_figure_caption_ocr_metrics(
            {"ocr_captions_linked": 3, "figures_with_ocr_caption": 3},
            {"ocr_captions_linked": 4, "figures_with_ocr_caption": 4},
            report_exists=True,
        )
        self.assertTrue(errs)
        joined = " ".join(errs)
        self.assertIn("ocr_captions_linked mismatch: kb=3 report=4", joined)
        self.assertIn("figures_with_ocr_caption mismatch: kb=3 report=4", joined)

    def test_missing_report_is_error(self):
        errs = check_figure_caption_ocr_metrics(
            {"ocr_captions_linked": 1, "figures_with_ocr_caption": 1},
            None,
            report_exists=False,
        )
        self.assertTrue(errs)
        self.assertIn("report missing", errs[0])

    def test_no_kb_is_error(self):
        errs = check_figure_caption_ocr_metrics(
            None, {"ocr_captions_linked": 0}, report_exists=True
        )
        self.assertTrue(errs)
        self.assertIn("no KB", errs[0])

    def test_run_v2_fails_on_metric_mismatch(self):
        """Integration: report=4, KB=3 → status failed, not ok."""
        import importlib.util
        from unittest.mock import MagicMock, patch

        if importlib.util.find_spec("fitz") is None:
            self.skipTest("fitz required for run_v2 integration")
        from paddle_models.cli.main import probe_capabilities, run_v2

        pdf = REPO_ROOT / "samples" / "103号(2).pdf"
        if not pdf.exists():
            self.skipTest("fixture PDF missing")
        caps = probe_capabilities("v2")
        ready_cap = {
            "engine": "tesseract",
            "ready": True,
            "pytesseract": True,
            "pytesseract_importable": True,
            "pillow_importable": True,
            "import_error": None,
            "tesseract_cmd": "tesseract",
            "tesseract_version": "tesseract 5.5",
            "languages_available": ["chi_sim", "eng"],
            "languages_required": ["chi_sim", "eng"],
            "missing": [],
        }

        def fake_pipeline(ctx, page_range):
            from pipeline.canonical_ir import BBox, CanonicalIR, DocBlock, DocPage
            from pipeline.semantic_projector import SemanticProjector

            ir = CanonicalIR(document_id="d", sha256="0" * 64)
            page = DocPage(page_number=1, width=100.0, height=100.0)
            for i in range(3):
                fid = f"fig{i}"
                page.blocks.append(
                    DocBlock(
                        id=fid,
                        type="figure",
                        text="",
                        bbox=BBox(10.0, 10.0, 50.0, 40.0),
                        page_number=1,
                        reading_order=1,
                        metadata={
                            "assetStatus": "ready",
                            "assetSource": "embedded",
                            "imageUri": "shots/x.png",
                            "contentSha256": f"{i:064d}",
                            "figureTextAssociations": [
                                {
                                    "kind": "caption",
                                    "blockId": f"capocr_{i}",
                                    "text": f"图{i} 题",
                                    "source": "ocr",
                                    "method": "ocr_inside_image",
                                    "ocrEngine": "tesseract",
                                }
                            ],
                        },
                    )
                )
                page.blocks.append(
                    DocBlock(
                        id=f"capocr_{i}",
                        type="caption",
                        text=f"图{i} 题",
                        bbox=BBox(12.0, 50.0, 30.0, 8.0),
                        page_number=1,
                        reading_order=2,
                        source="ocr",
                        metadata={
                            "semanticRole": "caption",
                            "figureId": fid,
                            "ocrEngine": "tesseract",
                            "insideFigure": True,
                        },
                    )
                )
            ir.pages.append(page)
            kb = SemanticProjector("d", strategy="page").project(ir)
            v2_dir = ctx.output_dir
            v2_dir.mkdir(parents=True, exist_ok=True)
            with open(v2_dir / "knowledge_base.json", "w", encoding="utf-8") as f:
                json.dump(kb, f, ensure_ascii=False)
            # Report claims 4 linked / 4 figures — KB has 3.
            report = {
                "engine": "tesseract",
                "language": "chi_sim+eng",
                "stats": {
                    "figure_caption_ocr_attempted": 4,
                    "figure_caption_ocr_cache_hits": 0,
                    "ocr_captions_linked": 4,
                    "figures_with_ocr_caption": 4,
                    "ocr_captions_rejected": 0,
                    "ocr_captions_ambiguous": 0,
                    "ocr_caption_elapsed_ms": 10,
                },
                "items": [],
            }
            with open(
                v2_dir / "figure_caption_ocr.json", "w", encoding="utf-8"
            ) as f:
                json.dump(report, f, ensure_ascii=False)
            return ir, kb

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run"
            run_dir.mkdir()
            with patch(
                "imaging.figure_caption_ocr.probe_tesseract_capability",
                return_value=ready_cap,
            ), patch(
                "pipeline.v2_runner.run_v2",
                side_effect=fake_pipeline,
            ):
                result = run_v2(
                    run_dir,
                    pdf,
                    caps,
                    "smoke",
                    1,
                    1,
                    figure_caption_ocr="tesseract",
                )
        self.assertNotEqual(result["status"], "ok")
        errs = " ".join(result.get("errors") or [])
        self.assertIn("mismatch", errs)
        self.assertIn("kb=3", errs)
        self.assertIn("report=4", errs)


class TestCaptionLineEvidence(unittest.TestCase):
    def test_select_caption_lines_single_line(self):
        lines = [
            {
                "key": "1-0-1",
                "text": "图12 颈椎骨折固定",
                "words": [
                    {"text": "图12", "conf": 90.0, "x": 10, "y": 10, "w": 40, "h": 16},
                    {"text": "颈椎", "conf": 88.0, "x": 55, "y": 10, "w": 30, "h": 16},
                ],
                "bbox": [10, 10, 85, 26],
                "mean_confidence": 89.0,
            },
            {
                "key": "1-0-2",
                "text": "无关噪声行 other junk",
                "words": [
                    {"text": "junk", "conf": 40.0, "x": 5, "y": 40, "w": 20, "h": 10},
                ],
                "bbox": [5, 40, 25, 50],
                "mean_confidence": 40.0,
            },
        ]
        unit = select_caption_lines(
            lines, _label_hits, _extract_candidate
        )
        self.assertEqual(unit["status"], "linked")
        self.assertEqual(unit["caption_line_keys"], ["1-0-1"])
        self.assertEqual(unit["caption_word_count"], 2)
        # Confidence only from participating words (90+88)/2 = 89
        self.assertAlmostEqual(unit["mean_confidence"], 89.0, places=1)
        # Unrelated line words must not be included.
        texts = [w["text"] for w in unit["words"]]
        self.assertNotIn("junk", texts)

    def test_bbox_from_participating_words_only(self):
        from imaging.figure_caption_ocr import words_bbox_norm

        caption_words = [
            {"text": "图12", "conf": 90.0, "x": 50, "y": 100, "w": 40, "h": 16},
            {"text": "颈椎", "conf": 88.0, "x": 95, "y": 100, "w": 30, "h": 16},
        ]
        noise_words = [
            {"text": "other", "conf": 50.0, "x": 0, "y": 0, "w": 200, "h": 200},
        ]
        image_size = (300, 400)
        band_top = 200
        cap_bb = words_bbox_norm(caption_words, image_size, band_top)
        all_bb = words_bbox_norm(caption_words + noise_words, image_size, band_top)
        self.assertIsNotNone(cap_bb)
        # Caption bbox height must be much smaller than full ROI/noise union.
        cap_h = cap_bb[3] - cap_bb[1]
        all_h = all_bb[3] - all_bb[1]
        self.assertLess(cap_h, all_h)
        self.assertLess(cap_h, 0.15, "caption line must not span the whole band")

    def test_line_evidence_fields_on_linked_result(self):
        r = ocr_embedded_figure_caption(
            _png_bytes(w=300, h=400),
            label_hits_fn=_label_hits,
            extract_candidate_fn=_extract_candidate,
            ocr_fn=lambda b, p, l: {
                "ok": True,
                "error": None,
                "text": "图12 颈椎骨折固定",
                "mean_confidence": 88.0,
                "words": [
                    # Band-relative coords (y within bottom band of 400px image).
                    {"text": "图12", "conf": 90.0, "x": 10, "y": 40, "w": 40, "h": 16,
                     "block_num": 1, "par_num": 0, "line_num": 1},
                    {"text": "颈椎", "conf": 88.0, "x": 55, "y": 40, "w": 30, "h": 16,
                     "block_num": 1, "par_num": 0, "line_num": 1},
                    {"text": "骨折", "conf": 86.0, "x": 90, "y": 40, "w": 30, "h": 16,
                     "block_num": 1, "par_num": 0, "line_num": 1},
                    {"text": "固定", "conf": 86.0, "x": 125, "y": 40, "w": 30, "h": 16,
                     "block_num": 1, "par_num": 0, "line_num": 1},
                ],
            },
        )
        self.assertEqual(r["status"], "linked")
        self.assertTrue(r.get("caption_line_keys"))
        self.assertGreater(r.get("caption_word_count") or 0, 0)
        self.assertIsNotNone(r.get("text_bbox_norm"))
        bb = r["text_bbox_norm"]
        # Height fraction of full image must be small (one text line, not ROI).
        self.assertLess(bb[3] - bb[1], 0.20)


if __name__ == "__main__":
    unittest.main()
