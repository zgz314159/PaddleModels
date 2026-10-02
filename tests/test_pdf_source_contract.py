"""Contract: PDF-generated v2 KBs declare a PowerAi-parseable PDF source.

PowerAi links an entry to its original PDF via a `pdf:{sha256}::{fileName}`
source string (see PowerAi util/PdfSourceRef). The active v2 generator must emit
that form for PDF input, with the sha256 equal to `docSha256`, and must keep the
historical `assets/kb/{id}` form for non-PDF / legacy / sha-less projections.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.canonical_ir import CanonicalIR  # noqa: E402
from pipeline.semantic_projector import SemanticProjector  # noqa: E402

SAMPLE_PDF = REPO_ROOT / "samples" / "103号(2).pdf"
PDF_SOURCE_RE = re.compile(r"^pdf:([0-9a-f]{64})::(.+)$")


class TestProjectorSourceContract(unittest.TestCase):
    def test_pdf_source_uses_doc_sha_and_original_name(self):
        sha = "a" * 64
        kb = SemanticProjector("doc", pdf_source_name="orig.pdf").project(
            CanonicalIR(document_id="doc", sha256=sha)
        )
        meta = kb["fileMetadata"]
        self.assertEqual(f"pdf:{sha}::orig.pdf", meta["source"])
        self.assertEqual("orig.pdf", meta["fileName"])
        match = PDF_SOURCE_RE.match(meta["source"])
        self.assertIsNotNone(match)
        self.assertEqual(sha, match.group(1))
        self.assertEqual("orig.pdf", match.group(2))

    def test_non_pdf_keeps_assets_kb_source(self):
        kb = SemanticProjector("doc").project(
            CanonicalIR(document_id="doc", sha256="b" * 64)
        )
        meta = kb["fileMetadata"]
        self.assertEqual("assets/kb/doc", meta["source"])
        self.assertEqual("knowledge_base.json", meta["fileName"])

    def test_pdf_without_sha_falls_back_to_assets_kb(self):
        kb = SemanticProjector("doc", pdf_source_name="orig.pdf").project(
            CanonicalIR(document_id="doc", sha256=None)
        )
        self.assertEqual("assets/kb/doc", kb["fileMetadata"]["source"])


class TestRealPdfKbSourceContract(unittest.TestCase):
    """Generate a KB from a few real PDF pages; no post-edit of the JSON."""

    def test_real_pdf_kb_source_matches_doc_sha(self):
        try:
            import fitz  # noqa: F401  (PyMuPDF required by the v2 runner)
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        if not SAMPLE_PDF.exists():
            self.skipTest(f"sample PDF missing: {SAMPLE_PDF}")

        from models.run_context import BuildProfile, RunContext
        from pipeline.v2_runner import run_v2

        expected_sha = hashlib.sha256(SAMPLE_PDF.read_bytes()).hexdigest()

        prev_cache = os.environ.get("PADDLE_CACHE_ROOT")
        with tempfile.TemporaryDirectory() as td:
            os.environ["PADDLE_CACHE_ROOT"] = str(Path(td) / "cache")
            try:
                ctx = RunContext.create(
                    str(SAMPLE_PDF), str(Path(td) / "out"), BuildProfile(name="contract")
                )
                _doc_ir, kb = run_v2(ctx, "1-2")
            finally:
                if prev_cache is None:
                    os.environ.pop("PADDLE_CACHE_ROOT", None)
                else:
                    os.environ["PADDLE_CACHE_ROOT"] = prev_cache

        meta = kb["fileMetadata"]
        self.assertEqual(expected_sha, meta["docSha256"])
        self.assertEqual(f"pdf:{expected_sha}::{SAMPLE_PDF.name}", meta["source"])
        self.assertEqual(SAMPLE_PDF.name, meta["fileName"])
        match = PDF_SOURCE_RE.match(meta["source"])
        self.assertIsNotNone(match)
        self.assertEqual(expected_sha, match.group(1))
        self.assertEqual(SAMPLE_PDF.name, match.group(2))


if __name__ == "__main__":
    unittest.main()
