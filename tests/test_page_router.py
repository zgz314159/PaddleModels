"""Page-router slice tests: per-page deterministic native/ocr routing.

All PDF, OCR, and model work is faked / injected — no real PDFs, no OCR
models, no rendering, no network.
"""
from __future__ import annotations

import io
import contextlib
import types
import unittest
import unittest.mock
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.extraction_adapters import native_adapter as native_adapter_module  # noqa: E402
from pipeline.extraction_adapters.native_adapter import NativeAdapter  # noqa: E402
from pipeline.page_router import DEFAULT_MIN_NATIVE_CHARS, PageRouter  # noqa: E402


class _FakePage:
    """Minimal stand-in for fitz.Page (plain text read only)."""

    def __init__(self, text):
        self._text = text
        self.get_text_modes = []

    def get_text(self, mode="text"):
        self.get_text_modes.append(mode)
        if isinstance(self._text, BaseException):
            raise self._text
        return self._text


class _FakeDoc:
    """Minimal stand-in for a fitz.Document."""

    def __init__(self, texts):
        self._pages = [_FakePage(t) for t in texts]
        self.page_count = len(self._pages)
        self.load_calls = []
        self.closed = False

    def load_page(self, index):
        if index < 0 or index >= self.page_count:
            raise IndexError(index)
        self.load_calls.append(index)
        return self._pages[index]

    def close(self):
        self.closed = True


def _bare_native_adapter(doc=None):
    """NativeAdapter without fitz/PDF: bypass __init__ and set state directly."""
    adapter = NativeAdapter.__new__(NativeAdapter)
    adapter.context = types.SimpleNamespace(input_path="fake.pdf")
    adapter.doc = doc
    adapter._watermarks = set()
    adapter._dynamic_artifacts = set()
    adapter._initialized = False
    return adapter


class _FakeNativeAdapter:
    """Injectable native adapter returning canned probe results / errors."""

    def __init__(self, results):
        self.results = results  # page_number -> int | BaseException
        self.probe_calls = []
        self.close_calls = 0

    def probe_page_native_chars(self, page_number):
        self.probe_calls.append(page_number)
        result = self.results.get(page_number, 0)
        if isinstance(result, BaseException):
            raise result
        return result

    def close(self):
        self.close_calls += 1


def _router(results, **kwargs):
    native = _FakeNativeAdapter(results)
    return PageRouter(object(), native_adapter=native, ocr_adapter=object(), **kwargs), native


class TestPageRouter(unittest.TestCase):
    def test_strong_native_text_routes_native(self):
        router, _ = _router({1: 1200})
        self.assertEqual(router.route_page(1), "native")

    def test_blank_or_scanned_page_routes_ocr(self):
        router, _ = _router({1: 0})
        self.assertEqual(router.route_page(1), "ocr")

    def test_threshold_boundary_exact_is_native(self):
        router, _ = _router({1: DEFAULT_MIN_NATIVE_CHARS})
        self.assertEqual(router.route_page(1), "native")

    def test_threshold_boundary_below_is_ocr(self):
        router, _ = _router({1: DEFAULT_MIN_NATIVE_CHARS - 1})
        self.assertEqual(router.route_page(1), "ocr")

    def test_custom_threshold_is_honored(self):
        router, _ = _router({1: 5, 2: 4}, min_native_chars=5)
        self.assertEqual(router.route_page(1), "native")
        self.assertEqual(router.route_page(2), "ocr")

    def test_multi_page_results_differ_and_never_hybrid(self):
        router, _ = _router({1: 500, 2: 0, 3: 12})
        decisions = [router.route_page(p) for p in (1, 2, 3)]
        self.assertEqual(decisions, ["native", "ocr", "ocr"])
        self.assertNotIn("hybrid", decisions)

    def test_page_is_probed_individually(self):
        router, native = _router({1: 500, 2: 0})
        router.route_page(1)
        router.route_page(2)
        self.assertEqual(native.probe_calls, [1, 2])

    def test_out_of_range_page_degrades_to_ocr(self):
        router, _ = _router({7: ValueError("page 7 out of range (1..3)")})
        self.assertEqual(router.route_page(7), "ocr")

    def test_probe_exception_degrades_to_ocr_without_crashing(self):
        router, _ = _router({4: RuntimeError("boom")})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            decision = router.route_page(4)
        self.assertEqual(decision, "ocr")
        self.assertIn("[WARN]", buf.getvalue())

    def test_close_closes_native_adapter(self):
        router, native = _router({1: 10})
        router.close()
        self.assertEqual(native.close_calls, 1)


class TestNativeAdapterProbe(unittest.TestCase):
    def test_counts_non_whitespace_characters(self):
        adapter = _bare_native_adapter(_FakeDoc(["a b\nc\t d"]))
        self.assertEqual(adapter.probe_page_native_chars(1), 4)

    def test_whitespace_only_page_is_zero(self):
        adapter = _bare_native_adapter(_FakeDoc(["   \n\t  "]))
        self.assertEqual(adapter.probe_page_native_chars(1), 0)

    def test_cjk_text_is_counted(self):
        adapter = _bare_native_adapter(_FakeDoc(["铁路电力管理规则"]))
        self.assertEqual(adapter.probe_page_native_chars(1), 8)

    def test_out_of_range_page_raises_value_error(self):
        adapter = _bare_native_adapter(_FakeDoc(["text", "text"]))
        with self.assertRaises(ValueError):
            adapter.probe_page_native_chars(0)
        with self.assertRaises(ValueError):
            adapter.probe_page_native_chars(3)

    def test_get_text_error_propagates(self):
        adapter = _bare_native_adapter(_FakeDoc([RuntimeError("read failed")]))
        with self.assertRaises(RuntimeError):
            adapter.probe_page_native_chars(1)

    def test_doc_not_open_is_opened_lazily(self):
        fake_doc = _FakeDoc(["hello world"])
        opened = []

        fake_fitz = types.SimpleNamespace(
            open=lambda path: opened.append(path) or fake_doc
        )
        adapter = _bare_native_adapter(doc=None)
        with unittest.mock.patch.object(native_adapter_module, "fitz", fake_fitz):
            self.assertEqual(adapter.probe_page_native_chars(1), 10)
        self.assertEqual(opened, ["fake.pdf"])
        self.assertIs(adapter.doc, fake_doc)

    def test_probe_is_read_only_and_skips_full_extraction(self):
        fake_doc = _FakeDoc(["some native text here"])
        adapter = _bare_native_adapter(fake_doc)
        adapter.probe_page_native_chars(1)
        self.assertEqual(fake_doc._pages[0].get_text_modes, ["text"])
        self.assertEqual(fake_doc.load_calls, [0])
        self.assertFalse(adapter._initialized)  # no watermark/artifact scan

    def test_close_closes_doc_and_is_idempotent(self):
        fake_doc = _FakeDoc(["text"])
        adapter = _bare_native_adapter(fake_doc)
        adapter.close()
        self.assertTrue(fake_doc.closed)
        self.assertIsNone(adapter.doc)
        adapter.close()  # must not raise on an already-closed adapter


if __name__ == "__main__":
    unittest.main()
