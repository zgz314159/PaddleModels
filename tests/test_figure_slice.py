"""Phase 2D unit tests: figure adapter, projection, metrics, schema."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.canonical_ir import BBox, CanonicalIR, DocBlock, DocPage  # noqa: E402
from pipeline.extraction_adapters.figure_adapter import (  # noqa: E402
    BACKGROUND_AREA_FRACTION,
    MIN_DECORATION_AREA_PT2,
    asset_filename_for_digest,
    build_figure_blocks,
    clamp_bbox_to_page,
    should_exclude_embedded,
    stable_figure_block_id,
    write_shared_asset,
)
from pipeline.semantic_projector import SemanticProjector  # noqa: E402
from paddle_models.cli.main import kb_metrics_from_obj  # noqa: E402

KB_SCHEMA = REPO_ROOT / "contracts" / "knowledge_base_schema_v2.json"


# Minimal valid 1x1 PNG (works without PIL for byte fixtures)
_MIN_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
    "53de0000000c4944415408d763f8cf00000101010018dd8db0000000004945"
    "4e44ae426082"
)


def _png_bytes(w: int = 32, h: int = 32, color=(10, 20, 30)) -> bytes:
    try:
        from PIL import Image
        import io

        im = Image.new("RGB", (max(1, w), max(1, h)), color)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return _MIN_PNG


def _embed(page=1, bbox=None, digest=b"", xref=1, pw=100, ph=100):
    return {
        "page_number": page,
        "xref": xref,
        "bbox": bbox or [20.0, 40.0, 80.0, 100.0],
        "pixel_width": pw,
        "pixel_height": ph,
        "digest": digest if isinstance(digest, str) else digest.hex() if digest else "",
    }


class TestExcludePolicies(unittest.TestCase):
    def test_full_page_background_filtered(self):
        reason = should_exclude_embedded(
            (0, 0, 200, 300), 200, 300, pixel_width=100, pixel_height=100, visual_matched=False
        )
        self.assertEqual(reason, "full_page_background")

    def test_tiny_decoration_filtered(self):
        reason = should_exclude_embedded(
            (10, 10, 14, 14), 200, 300, pixel_width=4, pixel_height=4, visual_matched=False
        )
        self.assertIn(reason, ("tiny_decoration", "invalid_pixel_size"))

    def test_normal_image_kept(self):
        reason = should_exclude_embedded(
            (20, 40, 80, 100), 200, 300, pixel_width=100, pixel_height=100, visual_matched=False
        )
        self.assertIsNone(reason)

    def test_edge_icon_filtered(self):
        reason = should_exclude_embedded(
            (10, 2, 30, 14), 200, 300, pixel_width=50, pixel_height=30, visual_matched=False
        )
        self.assertEqual(reason, "edge_icon")


class TestStableIdsAndAssets(unittest.TestCase):
    def test_stable_block_id_no_time(self):
        a = stable_figure_block_id(5, (1, 2, 3, 4), "abc")
        b = stable_figure_block_id(5, (1, 2, 3, 4), "abc")
        self.assertEqual(a, b)
        c = stable_figure_block_id(6, (1, 2, 3, 4), "abc")
        self.assertNotEqual(a, c)

    def test_stable_asset_filename(self):
        d = hashlib.sha256(b"x").hexdigest()
        self.assertEqual(asset_filename_for_digest(d, "png"), f"img_{d[:32]}.png")

    def test_digest_same_asset_written_once(self):
        data = _png_bytes()
        with tempfile.TemporaryDirectory() as td:
            m1 = write_shared_asset(Path(td), data, asset_filename_for_digest(hashlib.sha256(data).hexdigest()))
            m2 = write_shared_asset(Path(td), data, asset_filename_for_digest(hashlib.sha256(data).hexdigest()))
            self.assertTrue(m1["exists"] and m2["exists"])
            self.assertEqual(m1["contentSha256"], m2["contentSha256"])
            files = list(Path(td).iterdir())
            self.assertEqual(len(files), 1)


class TestBuildFigureBlocks(unittest.TestCase):
    def test_embedded_priority_over_crop(self):
        data = _png_bytes()
        digest = hashlib.sha256(data).hexdigest()

        class FakeDoc:
            def extract_image(self, xref):
                return {"image": data, "ext": "png"}

        emb = _embed(digest=digest, xref=1)
        visual = DocBlock(
            id="v1",
            type="figure",
            text="",
            bbox=BBox(20, 40, 60, 60),
            page_number=1,
            reading_order=1,
            metadata={"visual": True},
        )
        with tempfile.TemporaryDirectory() as td:
            blocks, warns, stats = build_figure_blocks(
                page_number=1,
                page_width=200,
                page_height=300,
                embedded=[emb],
                visual_figures=[visual],
                doc=FakeDoc(),
                asset_dir=Path(td),
                page_image_bytes=None,
                scale=0.5,
                dpi=144,
            )
            self.assertEqual(len(blocks), 1)
            b = blocks[0]
            self.assertEqual(b.metadata["assetSource"], "embedded")
            self.assertEqual(b.metadata["assetStatus"], "ready")
            self.assertTrue(b.metadata["visualMatched"])
            self.assertEqual(stats["matched"], 1)
            self.assertEqual(stats["cropped"], 0)

    def test_same_content_different_positions_keeps_blocks(self):
        data = _png_bytes()
        digest = hashlib.sha256(data).hexdigest()

        class FakeDoc:
            def extract_image(self, xref):
                return {"image": data, "ext": "png", "width": 32, "height": 32}

        emb1 = _embed(page=1, bbox=[10, 10, 50, 50], digest=digest, xref=1)
        emb2 = _embed(page=2, bbox=[60, 60, 100, 100], digest=digest, xref=1)
        with tempfile.TemporaryDirectory() as td:
            blocks, _, stats = build_figure_blocks(
                page_number=1,
                page_width=200,
                page_height=300,
                embedded=[emb1],
                visual_figures=[],
                doc=FakeDoc(),
                asset_dir=Path(td),
                page_image_bytes=None,
            )
            # second position on "page 2" simulated as second block same digest
            blocks2, _, stats2 = build_figure_blocks(
                page_number=2,
                page_width=200,
                page_height=300,
                embedded=[emb2],
                visual_figures=[],
                doc=FakeDoc(),
                asset_dir=Path(td),
                page_image_bytes=None,
            )
            self.assertEqual(len(blocks), 1)
            self.assertEqual(len(blocks2), 1)
            self.assertNotEqual(blocks[0].id, blocks2[0].id)  # different page+bbox
            self.assertEqual(
                blocks[0].metadata["imageUri"], blocks2[0].metadata["imageUri"]
            )  # shared URI
            self.assertEqual(stats["unique_assets_written"], 1)
            self.assertEqual(stats2["unique_assets_written"], 0)  # dedup hit
            self.assertEqual(stats2["duplicate_asset_hits"], 1)
            files = list(Path(td).iterdir())
            self.assertEqual(len(files), 1)

    def test_different_bbox_not_merged_blocks(self):
        data = _png_bytes()
        digest = hashlib.sha256(data).hexdigest()

        class FakeDoc:
            def extract_image(self, xref):
                return {"image": data, "ext": "png"}

        e1 = _embed(bbox=[10, 10, 50, 50], digest=digest, xref=1)
        e2 = _embed(bbox=[80, 10, 120, 50], digest=digest, xref=2)
        with tempfile.TemporaryDirectory() as td:
            blocks, _, _ = build_figure_blocks(
                page_number=1,
                page_width=200,
                page_height=300,
                embedded=[e1, e2],
                visual_figures=[],
                doc=FakeDoc(),
                asset_dir=Path(td),
            )
            self.assertEqual(len(blocks), 2)
            self.assertNotEqual(blocks[0].id, blocks[1].id)

    def test_same_position_embedded_visual_merged_once(self):
        data = _png_bytes()
        digest = hashlib.sha256(data).hexdigest()

        class FakeDoc:
            def extract_image(self, xref):
                return {"image": data, "ext": "png"}

        emb = _embed(bbox=[20, 40, 80, 100], digest=digest, xref=1)
        visual = DocBlock(
            id="v1",
            type="figure",
            text="",
            bbox=BBox(20, 40, 60, 60),
            page_number=1,
            reading_order=1,
        )
        with tempfile.TemporaryDirectory() as td:
            blocks, _, stats = build_figure_blocks(
                page_number=1,
                page_width=200,
                page_height=300,
                embedded=[emb],
                visual_figures=[visual],
                doc=FakeDoc(),
                asset_dir=Path(td),
            )
            self.assertEqual(len(blocks), 1)  # not two blocks same placement
            self.assertEqual(stats["matched"], 1)
            self.assertEqual(stats["cropped"], 0)  # embedded preferred

    def test_missing_asset_warning(self):
        emb = _embed(digest="deadbeef", xref=999)
        with tempfile.TemporaryDirectory() as td:
            blocks, warns, _ = build_figure_blocks(
                page_number=1,
                page_width=200,
                page_height=300,
                embedded=[emb],
                visual_figures=[],
                doc=None,  # no extract
                asset_dir=Path(td),
                page_image_bytes=None,  # no crop either
            )
            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].metadata["assetStatus"], "missing")
            self.assertEqual(blocks[0].metadata["imageUri"], "")
            self.assertTrue(any("missing" in w or "no page render" in w or "and no page render" in w for w in warns))

    def test_clamp_bbox(self):
        x0, y0, x1, y1 = clamp_bbox_to_page((-5, -5, 500, 500), 200, 300)
        self.assertEqual((x0, y0), (0.0, 0.0))
        self.assertEqual((x1, y1), (200.0, 300.0))


class TestProjectionFigures(unittest.TestCase):
    def _ready_fig(self, fid="f1", uri="shots/img_a.png"):
        return DocBlock(
            id=fid,
            type="figure",
            text="",
            bbox=BBox(10, 20, 40, 50),
            page_number=1,
            reading_order=1,
            metadata={
                "assetSource": "embedded",
                "assetStatus": "ready",
                "imageUri": uri,
                "contentSha256": "a" * 64,
                "mimeType": "image/png",
                "pixelWidth": 32,
                "pixelHeight": 32,
            },
        )

    def test_figure_not_in_searchable_content(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=200, height=300, method="native")
        page.blocks.append(self._ready_fig())
        page.blocks.append(
            DocBlock(
                id="b1",
                type="text",
                text="正文内容",
                bbox=BBox(10, 100, 80, 20),
                page_number=1,
                reading_order=2,
                metadata={"semanticRole": "body"},
            )
        )
        ir.pages.append(page)
        kb = SemanticProjector("d", strategy="heading").project(ir)
        e = kb["entries"][0]
        self.assertNotIn("shots/", e["contentMarkdown"])
        self.assertIn("正文内容", e["contentMarkdown"])
        figs = [b for b in e["blocks"] if b["type"] == "image"]
        self.assertEqual(len(figs), 1)
        self.assertFalse(figs[0]["searchable"])
        self.assertEqual(figs[0]["assetStatus"], "ready")
        self.assertEqual(figs[0]["imageUri"], "shots/img_a.png")

    def test_figure_attaches_existing_entry(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=200, height=300, method="native")
        page.blocks.append(
            DocBlock(
                id="b1",
                type="text",
                text="先有正文",
                bbox=BBox(10, 10, 80, 20),
                page_number=1,
                reading_order=1,
                metadata={"semanticRole": "body"},
            )
        )
        page.blocks.append(self._ready_fig("f1", "shots/img_b.png"))
        ir.pages.append(page)
        kb = SemanticProjector("d").project(ir)
        self.assertEqual(len(kb["entries"]), 1)
        types = [b["type"] for b in kb["entries"][0]["blocks"]]
        self.assertIn("image", types)

    def test_figure_only_page_creates_kind_figure_entry(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=3, width=200, height=300, method="native")
        page.blocks.append(self._ready_fig("fonly", "shots/img_c.png"))
        # fix pageNumber on block
        page.blocks[0].page_number = 3
        ir.pages.append(page)
        kb = SemanticProjector("d").project(ir)
        self.assertEqual(len(kb["entries"]), 1)
        e = kb["entries"][0]
        self.assertEqual(e["kind"], "figure")
        self.assertEqual(len(e["blocks"]), 1)
        self.assertEqual((e.get("contentMarkdown") or ""), "")

    def test_missing_figure_only_no_entry(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=200, height=300, method="native")
        fig = self._ready_fig("fmiss", "")
        fig.metadata["assetStatus"] = "missing"
        fig.page_number = 1
        page.blocks.append(fig)
        ir.pages.append(page)
        kb = SemanticProjector("d").project(ir)
        # no ready asset → no figure-only entry
        self.assertEqual(len(kb["entries"]), 0)

    def test_images_count_unique_uri(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=200, height=300, method="native")
        page.blocks.append(self._ready_fig("a", "shots/img_shared.png"))
        page.blocks.append(self._ready_fig("b", "shots/img_shared.png"))
        page.blocks[1].page_number = 1
        # need a host: add body
        page.blocks.append(
            DocBlock(
                id="b",
                type="text",
                text="x",
                bbox=BBox(1, 1, 10, 10),
                page_number=1,
                reading_order=3,
                metadata={"semanticRole": "body"},
            )
        )
        # reorder: body first so figures attach
        ordered = [page.blocks[2], page.blocks[0], page.blocks[1]]
        page.blocks = ordered
        ir.pages.append(page)
        kb = SemanticProjector("d").project(ir)
        self.assertEqual(kb["fileMetadata"]["imagesCount"], 1)


class TestImageMetrics(unittest.TestCase):
    def test_metrics_classification(self):
        kb = {
            "fileMetadata": {"pageSizes": {"1": [1, 1]}},
            "entries": [{
                "entryId": "e",
                "contentNormalized": "x",
                "blocks": [
                    {
                        "type": "image",
                        "assetSource": "embedded",
                        "assetStatus": "ready",
                        "imageUri": "shots/img_a.png",
                        "pageNumber": 1,
                    },
                    {
                        "type": "image",
                        "assetSource": "embedded",
                        "assetStatus": "ready",
                        "imageUri": "shots/img_a.png",  # duplicate URI
                        "pageNumber": 2,
                    },
                    {
                        "type": "image",
                        "assetSource": "page_crop",
                        "assetStatus": "ready",
                        "imageUri": "shots/img_b.png",
                        "pageNumber": 3,
                    },
                    {
                        "type": "image",
                        "assetSource": "embedded",
                        "assetStatus": "missing",
                        "imageUri": "",
                        "pageNumber": 4,
                    },
                ],
            }],
        }
        m = kb_metrics_from_obj(kb)
        self.assertEqual(m["images"], 4)
        self.assertEqual(m["images_embedded"], 2)
        self.assertEqual(m["images_cropped"], 1)
        self.assertEqual(m["images_missing"], 1)
        self.assertEqual(m["image_assets_unique"], 2)
        self.assertEqual(m["image_assets_duplicate"], 1)


class TestSchemaAssets(unittest.TestCase):
    def _kb(self, **block_extra):
        block = {
            "id": "b",
            "type": "image",
            "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        }
        block.update(block_extra)
        return {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [{"entryId": "e", "jobTitle": "t", "blocks": [block]}],
        }

    def test_valid_asset_fields(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        good = self._kb(
            imageUri="shots/img_x.png",
            assetSource="embedded",
            assetStatus="ready",
            contentSha256="a" * 64,
            mimeType="image/png",
            pixelWidth=10,
            pixelHeight=10,
        )
        jsonschema.validate(instance=good, schema=schema)

    def test_reject_bad_enum(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        for key, val in (("assetSource", "remote"), ("assetStatus", "ok")):
            bad = self._kb(**{key: val})
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(instance=bad, schema=schema)


class TestNativeAdapterOwnership(unittest.TestCase):
    def test_native_adapter_no_image_blocks(self):
        src = (
            REPO_ROOT / "pipeline" / "extraction_adapters" / "native_adapter.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("get_image_info", src)
        self.assertNotIn('type="image"', src)
        self.assertNotIn("extract_image_bytes", src)
        self.assertIn("figure_adapter", src)


if __name__ == "__main__":
    unittest.main()
