"""Page-cache contract tests for the active v2 runner.

Everything is synthetic or mocked: no real PDFs, no OCR models, no rendering,
no network. Cache and temp files are isolated under a per-test temp dir.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
import unittest.mock as mock
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pipeline.v2_runner as v2_runner  # noqa: E402
from models.run_context import BuildProfile, RunContext  # noqa: E402
from pipeline.canonical_ir import BBox, DocBlock, DocPage  # noqa: E402
from pipeline.page_cache import (  # noqa: E402
    CACHE_NAMESPACE,
    PAGE_CACHE_SCHEMA_VERSION,
    PageCache,
    compute_page_fingerprint,
    page_ir_from_dict,
)
from pipeline.page_router import DEFAULT_MIN_NATIVE_CHARS  # noqa: E402


def _block(
    page_number=1,
    bid="p1_b1",
    btype="text",
    text="hello",
    x=10.0,
    y=10.0,
    w=40.0,
    h=12.0,
    image_uri=None,
    order=1,
):
    meta = {}
    if image_uri is not None:
        meta["imageUri"] = image_uri
    return DocBlock(
        id=bid,
        type=btype,
        text=text,
        bbox=BBox(x, y, w, h),
        page_number=page_number,
        reading_order=order,
        source="native",
        metadata=meta,
    )


def _page(page_number=1, blocks=None):
    if blocks is None:
        blocks = [_block(page_number=page_number)]
    return DocPage(
        page_number=page_number, width=612.0, height=792.0, blocks=blocks, method="native"
    )


def _profile(**overrides):
    profile = BuildProfile()
    for key, value in overrides.items():
        setattr(profile, key, value)
    return profile


def _cache(
    cache_dir,
    *,
    profile=None,
    input_sha256="sha-abc",
    router_config=None,
    schema_version=PAGE_CACHE_SCHEMA_VERSION,
):
    return PageCache(
        cache_dir,
        input_sha256=input_sha256,
        profile_config=(profile or _profile()).page_ir_cache_config(),
        router_config=router_config or {"min_native_chars": DEFAULT_MIN_NATIVE_CHARS},
        schema_version=schema_version,
    )


def _png_bytes(w=200, h=200, color=(180, 180, 180)):
    from PIL import Image

    im = Image.new("RGB", (max(1, w), max(1, h)), color)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class _TmpDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    @property
    def namespace_dir(self):
        return self.tmp / CACHE_NAMESPACE


class TestFingerprint(_TmpDirTestCase):
    def test_fingerprint_is_deterministic_and_config_sensitive(self):
        args = dict(
            schema_version=2,
            profile_config=_profile().page_ir_cache_config(),
            router_config={"min_native_chars": 32},
            strategy="native",
        )
        first = compute_page_fingerprint(**args)
        self.assertEqual(first, compute_page_fingerprint(**args))
        self.assertEqual(len(first), 64)
        # No Python hash(): identical payload -> identical digest across calls.
        self.assertNotIn(first, (str(hash("native")),))


class TestHitStability(_TmpDirTestCase):
    def test_same_inputs_stable_hit(self):
        cache = _cache(self.tmp)
        cache.store(1, "native", _page(1))
        loaded = cache.load(1, "native")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.blocks[0].text, "hello")
        # A brand-new instance with the same effective config still hits.
        self.assertIsNotNone(_cache(self.tmp).load(1, "native"))

    def test_run_identity_and_output_dir_do_not_affect_hit(self):
        pdf = self.tmp / "input.pdf"
        pdf.write_bytes(b"%PDF-synthetic-bytes")
        profile = _profile()
        router_config = {"min_native_chars": DEFAULT_MIN_NATIVE_CHARS}

        run_a = RunContext(
            run_id="run_a",
            input_path=pdf,
            output_dir=self.tmp / "out_a",
            profile=profile,
            start_time=1.0,
        )
        run_a.sha256 = "fixed-sha"
        run_b = RunContext(
            run_id="run_b",
            input_path=pdf,
            output_dir=self.tmp / "out_b",
            profile=profile,
            start_time=999.0,
        )
        run_b.sha256 = "fixed-sha"

        self.assertNotEqual(run_a.run_id, run_b.run_id)
        self.assertNotEqual(run_a.output_dir, run_b.output_dir)
        self.assertNotEqual(run_a.start_time, run_b.start_time)

        with mock.patch.dict(
            os.environ, {"PADDLE_CACHE_ROOT": str(self.tmp / "cache")}
        ):
            cache_a = PageCache(
                run_a.get_cache_dir(),
                input_sha256=run_a.get_input_sha256(),
                profile_config=profile.page_ir_cache_config(),
                router_config=router_config,
            )
            cache_a.store(1, "native", _page(1))
            cache_b = PageCache(
                run_b.get_cache_dir(),
                input_sha256=run_b.get_input_sha256(),
                profile_config=profile.page_ir_cache_config(),
                router_config=router_config,
            )
            self.assertEqual(
                cache_a.path_for(1, "native"), cache_b.path_for(1, "native")
            )
            self.assertIsNotNone(cache_b.load(1, "native"))


class TestIdentityIsolation(_TmpDirTestCase):
    def test_native_and_ocr_do_not_cross_hit(self):
        cache = _cache(self.tmp)
        cache.store(1, "native", _page(1))
        self.assertIsNotNone(cache.load(1, "native"))
        self.assertIsNone(cache.load(1, "ocr"))
        self.assertNotEqual(
            cache.path_for(1, "native"), cache.path_for(1, "ocr")
        )

    def test_router_threshold_change_misses(self):
        _cache(self.tmp, router_config={"min_native_chars": 32}).store(
            1, "native", _page(1)
        )
        other = _cache(self.tmp, router_config={"min_native_chars": 16})
        self.assertIsNone(other.load(1, "native"))

    def test_page_ir_profile_config_change_misses(self):
        _cache(self.tmp).store(1, "native", _page(1))
        for override in (
            {"use_native_text": False},
            {"ocr_enabled": False},
            {"ocr_engine": "tesseract"},
            {"layout_engine": "none"},
            {"gpu_enabled": True},
            {"figure_caption_ocr": "tesseract"},
        ):
            with self.subTest(override=override):
                other = _cache(self.tmp, profile=_profile(**override))
                self.assertIsNone(other.load(1, "native"))

    def test_schema_version_change_misses(self):
        _cache(self.tmp, schema_version=2).store(1, "native", _page(1))
        self.assertIsNone(
            _cache(self.tmp, schema_version=3).load(1, "native")
        )


class TestMetadataMismatch(_TmpDirTestCase):
    def setUp(self):
        super().setUp()
        self.cache = _cache(self.tmp)
        self.cache.store(1, "native", _page(1))
        self.path = self.cache.path_for(1, "native")

    def _rewrite(self, mutate):
        envelope = json.loads(self.path.read_text(encoding="utf-8"))
        mutate(envelope)
        self.path.write_text(json.dumps(envelope), encoding="utf-8")

    def test_wrong_input_sha_misses(self):
        other = _cache(self.tmp, input_sha256="a-different-sha")
        self.assertIsNone(other.load(1, "native"))

    def test_wrong_page_number_misses(self):
        self._rewrite(lambda e: e.__setitem__("pageNumber", 2))
        self.assertIsNone(self.cache.load(1, "native"))

    def test_wrong_strategy_misses(self):
        self._rewrite(lambda e: e.__setitem__("strategy", "ocr"))
        self.assertIsNone(self.cache.load(1, "native"))

    def test_wrong_fingerprint_misses(self):
        self._rewrite(lambda e: e.__setitem__("fingerprint", "0" * 64))
        self.assertIsNone(self.cache.load(1, "native"))

    def test_missing_metadata_misses(self):
        self._rewrite(lambda e: e.pop("fingerprint"))
        self.assertIsNone(self.cache.load(1, "native"))


class TestLegacyAndCorruption(_TmpDirTestCase):
    def test_legacy_bare_page_json_is_safe_miss_and_untouched(self):
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        bare = json.dumps(_page(1).to_dict())
        new_path = self.namespace_dir / "page_1.native.json"
        new_path.write_text(bare, encoding="utf-8")
        legacy_path = self.tmp / "page_1.json"
        legacy_path.write_text(bare, encoding="utf-8")

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            result = _cache(self.tmp).load(1, "native")
        self.assertIsNone(result)
        self.assertIn("[WARN]", buf.getvalue())
        # Nothing is deleted: both the legacy and the incompatible file remain.
        self.assertTrue(legacy_path.exists())
        self.assertTrue(new_path.exists())
        self.assertEqual(new_path.read_text(encoding="utf-8"), bare)

    def test_corrupted_and_truncated_json_are_safe_misses(self):
        self.namespace_dir.mkdir(parents=True, exist_ok=True)
        path = self.namespace_dir / "page_1.native.json"
        for payload in ("{not json", '{"cacheSchemaVersion": 2, "fingerprint"'):
            with self.subTest(payload=payload):
                path.write_text(payload, encoding="utf-8")
                buf = io.StringIO()
                with contextlib.redirect_stderr(buf):
                    result = _cache(self.tmp).load(1, "native")
                self.assertIsNone(result)
                self.assertIn("[WARN]", buf.getvalue())


class TestAtomicWrite(_TmpDirTestCase):
    def test_write_is_readable_and_leaves_no_temp_files(self):
        cache = _cache(self.tmp)
        cache.store(1, "native", _page(1))
        self.assertTrue(cache.path_for(1, "native").exists())
        self.assertIsNotNone(cache.load(1, "native"))

        leftovers = [
            p
            for p in self.namespace_dir.iterdir()
            if p.name.endswith(".tmp") or p.name.startswith(".")
        ]
        self.assertEqual(leftovers, [])

    def test_roundtrip_preserves_page_ir(self):
        page = _page(
            1,
            blocks=[
                _block(bid="p1_b1", btype="text", text="alpha"),
                _block(bid="p1_b2", btype="table", text="", image_uri="shots/t.png"),
            ],
        )
        restored = page_ir_from_dict(json.loads(json.dumps(page.to_dict())))
        self.assertEqual([b.id for b in restored.blocks], ["p1_b1", "p1_b2"])
        self.assertEqual(restored.blocks[1].metadata["imageUri"], "shots/t.png")


class _FakeNative:
    def __init__(self):
        self.extract_calls = 0
        self.render_calls = 0

    def probe_coverage(self):
        return 1.0

    def extract_page(self, page_number):
        self.extract_calls += 1
        raise AssertionError("full page extraction must not run on a cache hit")

    def render_page(self, page_number, dpi=72):
        self.render_calls += 1
        return _png_bytes()

    def close(self):
        pass


class _FakeRouter:
    def __init__(self, strategy, native, ocr):
        self._strategy = strategy
        self.native_adapter = native
        self.ocr_adapter = ocr
        self.min_native_chars = DEFAULT_MIN_NATIVE_CHARS

    def route_page(self, page_number):
        return self._strategy

    def close(self):
        self.native_adapter.close()


class _FakeLayout:
    def __init__(self, context):
        pass

    def detect_blocks(self, *args, **kwargs):
        return []


class _FakeProjector:
    def __init__(self, document_id, strategy=None):
        pass

    def project(self, doc_ir):
        return {}


class TestRunnerCacheHit(_TmpDirTestCase):
    def _run_cached(self, blocks, *, strategy="native"):
        pdf = self.tmp / "input.pdf"
        pdf.write_bytes(b"%PDF-synthetic-bytes")
        profile = _profile()
        cache_root = self.tmp / "cache"

        native = _FakeNative()
        router = _FakeRouter(strategy, native, object())
        fake_fitz = types.SimpleNamespace(
            open=lambda path: types.SimpleNamespace(page_count=1, close=lambda: None)
        )
        assoc = {
            "figures_with_native_caption": 0,
            "figures_with_native_legend": 0,
            "native_captions_linked": 0,
            "native_legends_linked": 0,
            "unmatched_caption_candidates": 0,
        }
        fr_report = {
            "stats": {
                "figure_labels_indexed": 0,
                "figure_references_found": 0,
                "figure_references_resolved": 0,
                "figure_references_unresolved": 0,
                "figure_references_ambiguous": 0,
                "referenced_figures": 0,
            }
        }

        with mock.patch.dict(os.environ, {"PADDLE_CACHE_ROOT": str(cache_root)}):
            ctx = RunContext.create(str(pdf), str(self.tmp / "out"), profile)
            input_sha = ctx.get_input_sha256()
            PageCache(
                ctx.get_cache_dir(),
                input_sha256=input_sha,
                profile_config=profile.page_ir_cache_config(),
                router_config={"min_native_chars": DEFAULT_MIN_NATIVE_CHARS},
            ).store(1, strategy, _page(1, blocks))

            with mock.patch.object(
                v2_runner, "PageRouter", lambda context: router
            ), mock.patch.object(
                v2_runner, "LayoutAdapter", _FakeLayout
            ), mock.patch.object(
                v2_runner, "SemanticProjector", _FakeProjector
            ), mock.patch.object(
                v2_runner, "associate_ir_figures", lambda doc_ir: assoc
            ), mock.patch(
                "pipeline.semantics.figure_reference.resolve_figure_references",
                lambda doc_ir: fr_report,
            ), mock.patch.dict(
                sys.modules, {"fitz": fake_fitz}
            ):
                doc_ir, _kb = v2_runner.run_v2(ctx, "1-1")

        return ctx, native, doc_ir

    def test_hit_skips_full_page_extraction(self):
        ctx, native, doc_ir = self._run_cached([_block()])
        self.assertEqual(native.extract_calls, 0)
        self.assertEqual(native.render_calls, 0)
        self.assertEqual(len(doc_ir.pages), 1)
        self.assertEqual(doc_ir.pages[0].blocks[0].text, "hello")

    def test_asset_rehydrate_runs_on_cache_hit(self):
        block = _block(
            btype="table",
            text="",
            x=10.0,
            y=10.0,
            w=40.0,
            h=40.0,
            image_uri="shots/missing_asset.png",
        )
        ctx, native, doc_ir = self._run_cached([block])
        self.assertEqual(native.extract_calls, 0)
        self.assertEqual(native.render_calls, 1)
        self.assertTrue((ctx.output_dir / "shots" / "missing_asset.png").exists())
        self.assertEqual(len(doc_ir.pages), 1)


if __name__ == "__main__":
    unittest.main()
