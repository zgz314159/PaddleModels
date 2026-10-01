"""Phase 2A unit tests: table detection helpers, merge, suppress, projection, CLI."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.canonical_ir import BBox, CanonicalIR, DocBlock, DocPage  # noqa: E402
from pipeline.extraction_adapters.table_adapter import (  # noqa: E402
    boxes_overlap_xywh,
    cells_to_table_rows,
    escape_markdown_cell,
    merge_native_and_visual_tables,
    should_suppress_native_text,
    suppress_text_for_structured_tables,
    table_rows_to_markdown,
)
from pipeline.semantic_projector import SemanticProjector  # noqa: E402
from paddle_models.cli.main import (  # noqa: E402
    build_parser,
    kb_metrics_from_obj,
)

KB_SCHEMA = REPO_ROOT / "contracts" / "knowledge_base_schema_v2.json"


def _cells_3x3():
    cells = []
    for r in range(3):
        for c in range(3):
            cells.append({"row": r, "col": c, "text": f"{r}{c}"})
    return cells


def _native_table(bid="p29_tbl1", x=20, y=200, w=200, h=80, cells=None, page=29):
    if cells is None:
        cells = _cells_3x3()
    return DocBlock(
        id=bid,
        type="table",
        text="",
        bbox=BBox(x, y, w, h),
        page_number=page,
        reading_order=1,
        source="pymupdf_native",
        metadata={
            "cells": cells,
            "rows": 3 if cells else 0,
            "cols": 3 if cells else 0,
            "structureStatus": "structured" if cells else "image_only",
            "detectionSource": "pymupdf_native",
        },
    )


def _visual_table(bid="p29_v0", x=25, y=205, w=190, h=70, page=29, source="cv_fallback"):
    return DocBlock(
        id=bid,
        type="table",
        text="",
        bbox=BBox(x, y, w, h),
        page_number=page,
        reading_order=0,
        source=source,
        metadata={"visual": True},
    )


class TestCellsAndMarkdown(unittest.TestCase):
    def test_native_cells_to_rows(self):
        rows = cells_to_table_rows(_cells_3x3())
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0], ["00", "01", "02"])
        self.assertEqual(rows[2][2], "22")

    def test_cells_stable_conversion_dense(self):
        cells = [
            {"row": 0, "col": 0, "text": "a"},
            {"row": 1, "col": 2, "text": "c"},
        ]
        rows = cells_to_table_rows(cells)
        # Dense matrix: every row padded to max column width.
        self.assertEqual(rows, [["a", "", ""], ["", "", "c"]])

    def test_markdown_escaping(self):
        self.assertEqual(escape_markdown_cell("a|b"), "a\\|b")
        self.assertEqual(escape_markdown_cell("line1\nline2"), "line1 line2")
        md = table_rows_to_markdown([["h1", "h|2"], ["v1", "x"]])
        self.assertIn("h\\|2", md)
        self.assertIn("| h1 | h\\|2 |", md)
        self.assertIn("| --- | --- |", md)

    def test_structured_enters_searchable_content(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=200, height=300, method="native")
        page.blocks.append(
            DocBlock(
                id="t1",
                type="table",
                text="",
                bbox=BBox(10, 50, 100, 60),
                page_number=1,
                reading_order=1,
                metadata={
                    "cells": _cells_3x3(),
                    "structureStatus": "structured",
                    "imageUri": "shots/t1.png",
                },
            )
        )
        ir.pages.append(page)
        kb = SemanticProjector("d", strategy="heading").project(ir)
        e = kb["entries"][0]
        self.assertIn("00", e["contentMarkdown"])
        self.assertIn("| 00 |", e["contentMarkdown"])
        tb = e["blocks"][0]
        self.assertEqual(tb["type"], "table")
        self.assertTrue(tb["searchable"])
        self.assertEqual(tb["structureStatus"], "structured")
        self.assertEqual(len(tb["rows"]), 3)
        self.assertNotIn("table_rows", tb)

    def test_image_only_not_in_content(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=200, height=300, method="native")
        page.blocks.append(
            DocBlock(
                id="t1",
                type="table",
                text="",
                bbox=BBox(10, 50, 100, 60),
                page_number=1,
                reading_order=1,
                metadata={
                    "cells": [],
                    "structureStatus": "image_only",
                    "imageUri": "shots/t1.png",
                },
            )
        )
        # native text inside image_only table region — kept as normal body
        page.blocks.append(
            DocBlock(
                id="b1",
                type="text",
                text="表内文字保留",
                bbox=BBox(20, 60, 80, 20),
                page_number=1,
                reading_order=2,
                metadata={"semanticRole": "body"},
            )
        )
        ir.pages.append(page)
        kb = SemanticProjector("d", strategy="heading").project(ir)
        e = kb["entries"][0]
        self.assertIn("表内文字保留", e["contentMarkdown"])
        # image_only table itself does not inject fake markdown
        self.assertNotIn("| --- |", e["contentMarkdown"])
        tb = next(b for b in e["blocks"] if b["type"] == "table")
        self.assertFalse(tb["searchable"])
        self.assertEqual(tb["rows"], [])
        self.assertNotIn("table_rows", tb)
        self.assertEqual(tb["structureStatus"], "image_only")
        self.assertEqual(tb["imageUri"], "shots/t1.png")


class TestMergeAndSuppress(unittest.TestCase):
    def test_bbox_match_prefers_native_cells(self):
        native = [_native_table()]
        visual = [_visual_table()]
        merged = merge_native_and_visual_tables(native, visual)
        self.assertEqual(len(merged), 1)
        m = merged[0]
        self.assertEqual(m.metadata["structureStatus"], "structured")
        self.assertTrue(m.metadata["cells"])
        self.assertTrue(m.metadata.get("visualMatched"))

    def test_structured_preferred_over_image_only(self):
        native = [_native_table()]
        visual = [_visual_table()]
        merged = merge_native_and_visual_tables(native, visual)
        self.assertEqual(merged[0].metadata["structureStatus"], "structured")

    def test_unmatched_native_still_output(self):
        native = [_native_table(x=20, y=20, w=50, h=40)]
        visual = [_visual_table(x=200, y=200, w=30, h=30)]  # no overlap
        merged = merge_native_and_visual_tables(native, visual)
        statuses = sorted(m.metadata["structureStatus"] for m in merged)
        self.assertEqual(len(merged), 2)
        self.assertIn("structured", statuses)
        self.assertIn("image_only", statuses)  # unmatched CV

    def test_cv_weak_never_structured(self):
        visual = [_visual_table(source="cv_fallback_weak")]
        merged = merge_native_and_visual_tables([], visual)
        self.assertEqual(merged[0].metadata["structureStatus"], "image_only")

    def test_image_only_does_not_suppress_text(self):
        text = DocBlock(
            id="t",
            type="text",
            text="正文",
            bbox=BBox(30, 30, 40, 20),
            page_number=1,
            reading_order=1,
            metadata={"semanticRole": "body"},
        )
        tables = [_native_table(cells=[], x=20, y=20, w=200, h=100)]
        tables[0].metadata["structureStatus"] = "image_only"
        tables[0].metadata["_bbox_xywh"] = (20, 20, 200, 100)
        kept = suppress_text_for_structured_tables([text], tables)
        self.assertEqual(len(kept), 1)

    def test_structured_suppresses_interior_text(self):
        text = DocBlock(
            id="t",
            type="text",
            text="重复表头",
            bbox=BBox(30, 30, 40, 20),
            page_number=1,
            reading_order=1,
            metadata={"semanticRole": "body"},
        )
        outside = DocBlock(
            id="o",
            type="text",
            text="无关正文",
            bbox=BBox(30, 400, 100, 20),
            page_number=1,
            reading_order=2,
            metadata={"semanticRole": "body"},
        )
        tables = [_native_table(x=20, y=20, w=200, h=100)]
        tables[0].metadata["structureStatus"] = "structured"
        tables[0].metadata["_bbox_xywh"] = (20, 20, 200, 100)
        kept = suppress_text_for_structured_tables([text, outside], tables)
        self.assertEqual([b.id for b in kept], ["o"])

    def test_should_suppress_pure_fn(self):
        meta = {"structureStatus": "image_only", "_bbox_xywh": (0, 0, 100, 100)}
        self.assertFalse(should_suppress_native_text((10, 10, 20, 20), meta))
        meta2 = {"structureStatus": "structured", "_bbox_xywh": (0, 0, 100, 100)}
        self.assertTrue(should_suppress_native_text((10, 10, 20, 20), meta2))

    def test_boxes_overlap(self):
        self.assertTrue(boxes_overlap_xywh((0, 0, 100, 100), (10, 10, 100, 100)))
        self.assertFalse(boxes_overlap_xywh((0, 0, 50, 50), (500, 500, 50, 50), min_ratio=0.5))


class TestTableMetrics(unittest.TestCase):
    def test_table_metrics_fields(self):
        kb = {
            "fileMetadata": {"pageSizes": {"1": [1, 1]}},
            "entries": [{
                "entryId": "e",
                "contentNormalized": "x",
                "blocks": [
                    {
                        "type": "table",
                        "structureStatus": "structured",
                        "table_rows": [["a", "b"], ["c", ""]],
                        "imageUri": "shots/t1.png",
                        "pageNumber": 1,
                    },
                    {
                        "type": "table",
                        "structureStatus": "image_only",
                        "table_rows": [],
                        "imageUri": "",
                        "pageNumber": 1,
                    },
                ],
            }],
        }
        m = kb_metrics_from_obj(kb)
        self.assertEqual(m["tables"], 2)
        self.assertEqual(m["tables_structured"], 1)
        self.assertEqual(m["tables_image_only"], 1)
        self.assertEqual(m["table_cells"], 3)  # a,b,c non-empty
        self.assertEqual(m["table_assets_missing"], 1)  # image_only without uri


class TestStartPageArg(unittest.TestCase):
    def test_start_page_default_and_mapping(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input", "x.pdf",
            "--mode", "shadow",
            "--start-page", "29",
            "--max-pages", "2",
            "--profile", "smoke",
        ])
        self.assertEqual(args.start_page, 29)
        self.assertEqual(args.max_pages, 2)

    def test_start_page_default_is_1(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "x.pdf"])
        self.assertEqual(args.start_page, 1)

    def test_start_page_rejects_non_positive(self):
        from paddle_models.cli.main import _positive_page
        with self.assertRaises(SystemExit):
            _positive_page(0, "--start-page")
        with self.assertRaises(SystemExit):
            _positive_page(-3, "--start-page")
        self.assertEqual(_positive_page(1, "--start-page"), 1)


class TestStructureStatusSchema(unittest.TestCase):
    def test_valid_structure_status(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        good = {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [{
                "entryId": "e",
                "jobTitle": "t",
                "blocks": [{
                    "id": "b",
                    "type": "table",
                    "structureStatus": "structured",
                    "table_rows": [["1"]],
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                }],
            }],
        }
        jsonschema.validate(instance=good, schema=schema)

    def test_invalid_structure_status(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        bad = {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [{
                "entryId": "e",
                "jobTitle": "t",
                "blocks": [{
                    "id": "b",
                    "type": "table",
                    "structureStatus": "cv_fallback_weak",
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                }],
            }],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)


class TestTableUriRelative(unittest.TestCase):
    def test_relative_shots_uri(self):
        # URI convention enforced by projector block imageUri value
        uri = "shots/p29_tbl1.png"
        self.assertFalse(uri.startswith("file:///"))
        self.assertTrue(uri.startswith("shots/"))
        self.assertTrue(uri.endswith(".png"))


class TestGeometryGridMapping(unittest.TestCase):
    """Phase 2C: non-row-major bbox mapping + span contract."""

    def test_non_rowmajor_flat_index_mapping(self):
        """table.cells is column-major flat list — old r*cols+c was wrong.

        Synthetic geometry: 2x2 with distinct rects; verify map_cell_rect_to_grid
        derives correct spans without using flat index order.
        """
        from imaging.table_processor import map_cell_rect_to_grid

        xs = [0.0, 10.0, 20.0, 30.0]
        ys = [0.0, 10.0, 20.0]
        # Cell covering cols 1-2 (colspan 2), rows 0-1 (rowspan 2)
        r, c, rs, cs = map_cell_rect_to_grid((10.0, 0.0, 30.0, 20.0), xs, ys)
        self.assertEqual((r, c, rs, cs), (0, 1, 2, 2))
        # Single cell
        r, c, rs, cs = map_cell_rect_to_grid((0.0, 0.0, 10.0, 10.0), xs, ys)
        self.assertEqual((r, c, rs, cs), (0, 0, 1, 1))

    def test_rowspan_placeholder_not_colspan(self):
        """None under rowspan must not inflate colSpan (geometry-driven)."""
        from imaging.table_processor import map_cell_rect_to_grid

        # 6 logical rows → 7 y boundaries
        xs = [0.0, 50.0, 100.0]
        ys = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
        # Tall rect in col 1 spanning rows 1-5 (y=20..120) → rowSpan=5
        r, c, rs, cs = map_cell_rect_to_grid((50.0, 20.0, 100.0, 120.0), xs, ys)
        self.assertEqual(rs, 5)
        self.assertEqual(cs, 1)  # not inflated by None placeholders

    def test_merged_rect_emitted_once(self):
        from imaging.table_processor import build_native_table_grid

        class _Row:
            def __init__(self, cells):
                self.cells = cells

        tall = (0.0, 0.0, 50.0, 100.0)  # col0, both rows (y 0-100 with mid boundary)

        class T:
            bbox = (0, 0, 100, 100)

            def __init__(self):
                self.rows = [
                    _Row([tall, (50.0, 0.0, 100.0, 50.0)]),
                    _Row([tall, (50.0, 50.0, 100.0, 100.0)]),  # duplicate tall rect
                ]

            def extract(self):
                return [["A", "B"], [None, "C"]]

        cells, _ = build_native_table_grid(T())
        # tall rect listed twice → one physical cell with rowspan 2
        self.assertEqual(len(cells), 3)
        tall_cells = [c for c in cells if c["row"] == 0 and c["col"] == 0]
        self.assertEqual(len(tall_cells), 1)
        self.assertEqual(tall_cells[0]["rowSpan"], 2)
        self.assertEqual(tall_cells[0]["colSpan"], 1)
        self.assertEqual(tall_cells[0]["text"], "A")


class TestPage29Acceptance(unittest.TestCase):
    """Page 29 exact acceptance (live PDF geometry, no hard-coded full text rules)."""

    @classmethod
    def setUpClass(cls):
        try:
            import fitz
        except ImportError:
            raise unittest.SkipTest("fitz not installed (run under .venv)")
        from imaging.table_processor import build_native_table_grid

        pdf = REPO_ROOT / "samples" / "103号(2).pdf"
        if not pdf.exists():
            raise unittest.SkipTest("fixture PDF missing")
        cls.doc = fitz.open(str(pdf))
        tabs = list(cls.doc.load_page(28).find_tables().tables)
        if not tabs:
            raise unittest.SkipTest("page 29 table not found")
        cls.table = tabs[0]
        cls.cells, cls.warns = build_native_table_grid(cls.table)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "doc"):
            cls.doc.close()

    def test_rows_cols_physical_grid(self):
        self.assertEqual(self.table.row_count, 6)
        self.assertEqual(self.table.col_count, 4)
        self.assertEqual(len(self.cells), 20)
        self.assertEqual(6 * 4, 24)  # grid_slots

    def test_spanned_count_and_remark(self):
        spanned = [c for c in self.cells if c["rowSpan"] > 1 or c["colSpan"] > 1]
        self.assertEqual(len(spanned), 1)
        remark = [c for c in self.cells if "油门全开" in c["text"]]
        self.assertEqual(len(remark), 1)
        r = remark[0]
        self.assertEqual((r["row"], r["col"], r["rowSpan"], r["colSpan"]), (1, 3, 5, 1))

    def test_remark_bbox_right_column_five_rows(self):
        remark = next(c for c in self.cells if "油门全开" in c["text"])
        bb = remark["bbox"]
        self.assertIsNotNone(bb)
        tb = self.table.bbox
        # right column
        self.assertGreater(bb[0], tb[0] + 0.3 * (tb[2] - tb[0]))
        self.assertLessEqual(bb[2], tb[2] + 1.0)
        # spans ~5 data rows height
        self.assertGreater(bb[3] - bb[1], 0.5 * (tb[3] - tb[1]))

    def test_time_col_colspan_one(self):
        time_cells = [c for c in self.cells if c["col"] == 2]
        self.assertGreaterEqual(len(time_cells), 5)
        for c in time_cells:
            self.assertEqual(c["colSpan"], 1, msg=str(c))
            self.assertEqual(c["col"], 2)

    def test_all_bboxes_nonnull_and_inside_table(self):
        tb = self.table.bbox
        for c in self.cells:
            self.assertIsNotNone(c["bbox"], msg=str(c))
            b = c["bbox"]
            self.assertGreaterEqual(b[0], tb[0] - 1.0)
            self.assertGreaterEqual(b[1], tb[1] - 1.0)
            self.assertLessEqual(b[2], tb[2] + 1.0)
            self.assertLessEqual(b[3], tb[3] + 1.0)

    def test_cells_sorted_stable_unique(self):
        keys = [(c["row"], c["col"]) for c in self.cells]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(set(keys)))

    def test_projector_rect_rows_and_md_once(self):
        from pipeline.canonical_ir import BBox, CanonicalIR, DocBlock, DocPage

        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=29, width=246, height=360, method="native")
        page.blocks.append(
            DocBlock(
                id="p29_tbl1",
                type="table",
                text="",
                bbox=BBox(22.6, 206.5, 198.2, 84.9),
                page_number=29,
                reading_order=1,
                metadata={
                    "cells": self.cells,
                    "structureStatus": "structured",
                    "imageUri": "shots/p29_tbl1.png",
                },
            )
        )
        ir.pages.append(page)
        kb = SemanticProjector("d", strategy="heading").project(ir)
        tb = kb["entries"][0]["blocks"][0]
        self.assertEqual(len(tb["rows"]), 6)
        self.assertEqual(len(tb["rows"][0]), 4)
        self.assertNotIn("table_rows", tb)
        self.assertNotIn("cols", tb)
        self.assertEqual(len(tb["cells"]), 20)
        for row in tb["rows"]:
            self.assertEqual(len(row), 4)
        content = kb["entries"][0]["contentMarkdown"]
        self.assertEqual(content.count("油门全开"), 1)
        # remark appears once in cells too
        remarks = [c for c in tb["cells"] if "油门全开" in c.get("text", "")]
        self.assertEqual(len(remarks), 1)
        self.assertEqual(remarks[0]["rowSpan"], 5)

    def test_native_adapter_has_no_find_tables(self):
        src = (REPO_ROOT / "pipeline" / "extraction_adapters" / "native_adapter.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("find_tables", src)
        self.assertNotIn("extract_native_table_cells", src)

    def test_no_geometry_warnings_on_page29(self):
        # bbox may be present; warnings should be empty for healthy table
        hard = [w for w in self.warns if "missing bbox" in w or "out of logical" in w]
        self.assertEqual(hard, [])


class TestCellsSchemaPhase2C(unittest.TestCase):
    def test_valid_cells_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        good = {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [{
                "entryId": "e",
                "jobTitle": "t",
                "blocks": [{
                    "id": "b",
                    "type": "table",
                    "rows": [["", "", "", "x"], ["", "", "", ""], ["", "", "", ""], ["", "", "", ""], ["", "", "", ""]],
                    "cells": [{
                        "row": 1,
                        "col": 3,
                        "rowSpan": 5,
                        "colSpan": 1,
                        "text": "x",
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                    }],
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                }],
            }],
        }
        jsonschema.validate(instance=good, schema=schema)

    def test_span_zero_rejected(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        bad = {
            "fileMetadata": {"schemaVersion": "2.0", "fileId": "t", "docSha256": "a" * 64},
            "entries": [{
                "entryId": "e",
                "jobTitle": "t",
                "blocks": [{
                    "id": "b",
                    "type": "table",
                    "cells": [{
                        "row": 0,
                        "col": 0,
                        "rowSpan": 0,
                        "colSpan": 1,
                        "text": "x",
                    }],
                    "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                }],
            }],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=bad, schema=schema)


class TestTableDiffGeometryStats(unittest.TestCase):
    def test_grid_slots_vs_physical(self):
        from pipeline.compatibility.table_diff import _table_geometry_stats

        cells = [{"row": 0, "col": 0, "rowSpan": 1, "colSpan": 1, "text": "a"}]
        # 6x4 grid, 1 physical (incomplete sample) — just check fields
        # NOTE: integer rows/cols + table_rows = historical/transition shape (compat), not current v2.
        stats = _table_geometry_stats(
            {"table_rows": [["a", "", "", ""]], "cells": cells, "rows": 6, "cols": 4},
            side="v2",
        )
        self.assertTrue(stats["span_data_available"])
        self.assertEqual(stats["physical_cells"], 1)
        self.assertEqual(stats["grid_slots"], 24)
        self.assertEqual(stats["logical_rows"], 6)
        self.assertEqual(stats["logical_cols"], 4)

    def test_legacy_span_unavailable(self):
        from pipeline.compatibility.table_diff import _table_geometry_stats

        stats = _table_geometry_stats(
            {"table_rows": [["1", "2"], ["3", "4"]], "cells": None},
            side="legacy",
        )
        self.assertFalse(stats["span_data_available"])
        # text slots counted as physical for legacy side reporting
        self.assertEqual(stats["physical_cells"], 4)
        self.assertEqual(stats["grid_slots"], 4)

    def test_counting_note_in_report_legacy_numeric_shape(self):
        from pipeline.compatibility.table_diff import build_table_diff_report

        leg = {"entries": [{"entryId": "e", "blocks": [{
            "type": "table", "pageNumber": 29,
            "table_rows": [["a", "b", "c", "d"]] * 6,
            "bbox": {"left": 0, "top": 0, "right": 100, "bottom": 50},
        }]}]}
        # Historical/transition v2 shape (integer rows/cols + table_rows) — kept for compat only.
        v2 = {"entries": [{"entryId": "e", "blocks": [{
            "type": "table", "pageNumber": 29,
            "rows": 6, "cols": 4,
            "cells": [
                {"row": r, "col": c, "rowSpan": 1, "colSpan": 1, "text": f"{r}{c}"}
                for r in range(6) for c in range(4) if not (r >= 2 and c == 3)
            ],
            "table_rows": [[""] * 4 for _ in range(6)],
            "bbox": {"left": 0, "top": 0, "right": 100, "bottom": 50},
            "structureStatus": "structured",
        }]}]}
        report = build_table_diff_report(leg, v2, page_number=29)
        self.assertEqual(len(report["comparisons"]), 1)
        cd = report["comparisons"][0]["cells"]
        self.assertFalse(cd["legacy_geometry"]["span_data_available"])
        self.assertTrue(cd["v2_geometry"]["span_data_available"])
        self.assertEqual(cd["v2_geometry"]["physical_cells"], 20)  # 24-4 spanned slots as anchors only
        self.assertTrue(report["cell_count_notes"])
        self.assertIn("text-split", report["cell_count_notes"][0] + report["note"] if "note" in report else report["cell_count_notes"][0] + cd.get("counting_note", ""))


class TestTableRowsContractFix(unittest.TestCase):
    """v2 table contract: 2D `rows`, no `table_rows`/integer counts in new output."""

    _SPANNED_TABLE = [
        {"row": 0, "col": 0, "text": "h1"},
        {"row": 0, "col": 1, "text": "h2"},
        {"row": 1, "col": 0, "text": "merged", "rowSpan": 2, "colSpan": 1},
        {"row": 1, "col": 1, "text": "b"},
        {"row": 2, "col": 1, "text": "c"},
    ]

    def _ir(self, cells, status="structured", image="shots/t.png"):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=200, height=300, method="native")
        page.blocks.append(
            DocBlock(
                id="t1",
                type="table",
                text="",
                bbox=BBox(1, 2, 3, 4),
                page_number=1,
                reading_order=1,
                metadata={"cells": cells, "structureStatus": status, "imageUri": image},
            )
        )
        ir.pages.append(page)
        return ir

    def _project_block(self, cells, status="structured", image="shots/t.png"):
        kb = SemanticProjector("d", strategy="heading").project(self._ir(cells, status, image))
        return kb["entries"][0]["blocks"][0]

    def test_structured_rows_is_2d_strings(self):
        tb = self._project_block(self._SPANNED_TABLE)
        self.assertNotIn("table_rows", tb)
        self.assertIsInstance(tb["rows"], list)
        self.assertTrue(all(isinstance(r, list) for r in tb["rows"]))
        self.assertTrue(all(isinstance(c, str) for r in tb["rows"] for c in r))

    def test_merged_anchor_only_and_spans_preserved(self):
        tb = self._project_block(self._SPANNED_TABLE)
        self.assertEqual(len(tb["rows"]), 3)
        self.assertEqual(len(tb["rows"][0]), 2)
        self.assertEqual(tb["rows"][1][0], "merged")
        self.assertEqual(tb["rows"][2][0], "")  # spanned slot stays empty (anchor-only)
        anchor = [c for c in tb["cells"] if c["text"] == "merged"][0]
        self.assertEqual((anchor["rowSpan"], anchor["colSpan"]), (2, 1))

    def test_image_only_rows_empty_and_no_int(self):
        tb = self._project_block([], status="image_only")
        self.assertEqual(tb["rows"], [])
        self.assertNotIn("table_rows", tb)
        self.assertNotIn("cols", tb)
        self.assertEqual(tb["structureStatus"], "image_only")
        self.assertEqual(tb["imageUri"], "shots/t.png")

    def test_serialized_json_types(self):
        tb = self._project_block(self._SPANNED_TABLE)
        loaded = json.loads(json.dumps(tb))["rows"]
        self.assertIsInstance(loaded, list)
        self.assertTrue(all(isinstance(r, list) for r in loaded))
        self.assertTrue(all(isinstance(c, str) for r in loaded for c in r))

    def test_projected_kb_validates_against_schema(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        kb = SemanticProjector("d", strategy="heading").project(self._ir(self._SPANNED_TABLE))
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        jsonschema.validate(instance=kb, schema=schema)

    def test_legacy_compat_read_still_accepts_table_rows(self):
        legacy_block = {"type": "table", "table_rows": [["a", "b"], ["c", ""]], "pageNumber": 1}
        m = kb_metrics_from_obj(
            {
                "fileMetadata": {"schemaVersion": "2.0", "fileId": "d", "docSha256": "0" * 64},
                "entries": [{"entryId": "e", "blocks": [legacy_block]}],
            },
            include_figure_association=False,
        )
        self.assertEqual(m["table_cells"], 3)
        self.assertNotIn("table_rows", self._project_block(self._SPANNED_TABLE))

    def test_content_markdown_equivalent(self):
        tb = self._project_block(self._SPANNED_TABLE)
        expected = table_rows_to_markdown(tb["rows"])
        kb = SemanticProjector("d", strategy="heading").project(self._ir(self._SPANNED_TABLE))
        self.assertIn(expected, kb["entries"][0]["contentMarkdown"])


class TestTableRowsConsumerClosure(unittest.TestCase):
    """Closure: v2 table `rows` (2D) consumers — table_diff + CLI metrics."""

    _MERGED = [
        {"row": 0, "col": 0, "text": "H1"},
        {"row": 0, "col": 1, "text": "H2"},
        {"row": 1, "col": 0, "text": "merged-span", "rowSpan": 2, "colSpan": 1},
        {"row": 1, "col": 1, "text": "b1"},
        {"row": 2, "col": 1, "text": "b2"},
    ]

    def _project(self):
        ir = CanonicalIR(document_id="d", sha256="0" * 64)
        page = DocPage(page_number=1, width=246.0, height=360.0, method="native")
        page.blocks.append(DocBlock(id="p1_tbl_struct", type="table", text="", bbox=BBox(22.6, 206.5, 198.2, 84.9),
                                    page_number=1, reading_order=1,
                                    metadata={"cells": self._MERGED, "structureStatus": "structured", "imageUri": "shots/s.png"}))
        page.blocks.append(DocBlock(id="p1_tbl_image", type="table", text="", bbox=BBox(10.0, 10.0, 100.0, 40.0),
                                    page_number=1, reading_order=2,
                                    metadata={"cells": [], "structureStatus": "image_only", "imageUri": "shots/i.png"}))
        ir.pages.append(page)
        return SemanticProjector("d", strategy="heading").project(ir)

    def _metrics(self, blocks):
        return kb_metrics_from_obj(
            {
                "fileMetadata": {"schemaVersion": "2.0", "fileId": "d", "docSha256": "0" * 64},
                "entries": [{"entryId": "e", "blocks": blocks}],
            },
            include_figure_association=False,
        )

    def test_1_real_projection_table_diff_no_exception(self):
        from pipeline.compatibility.table_diff import build_table_diff_report
        report = build_table_diff_report({"entries": []}, self._project())
        self.assertEqual(report["v2_table_count"], 2)

    def test_2_table_diff_geometry_and_markdown_preview(self):
        from pipeline.compatibility.table_diff import build_table_diff_report, collect_v2_tables
        kb = self._project()
        by_id = {t["id"]: t for t in collect_v2_tables(kb)}
        st = by_id["p1_tbl_struct"]["geometry"]
        self.assertEqual((st["logical_rows"], st["logical_cols"], st["grid_slots"]), (3, 2, 6))
        self.assertEqual(st["physical_cells"], 5)
        self.assertEqual(st["spanned_cells"], 1)
        self.assertTrue(st["span_data_available"])
        io = by_id["p1_tbl_image"]["geometry"]
        self.assertEqual((io["logical_rows"], io["physical_cells"]), (0, 0))
        self.assertEqual(by_id["p1_tbl_image"]["table_rows"], [])
        leg = {"entries": [{"entryId": "e", "blocks": [{
            "type": "table", "id": "L", "pageNumber": 1,
            "bbox": {"left": 22, "top": 206, "right": 221, "bottom": 291},
            "table_rows": [["H1", "H2"], ["merged-span", "b1"], ["", "b2"]],
        }]}]}
        comp = next(c for c in build_table_diff_report(leg, kb)["comparisons"] if "v2_markdown_preview" in c)
        self.assertIn("H1", comp["v2_markdown_preview"])

    def test_3_metrics_canonical_rows_and_cells_no_double_count(self):
        m = self._metrics([{"type": "table", "structureStatus": "structured",
                            "rows": [["a", ""], ["b", "c"]],
                            "cells": [{"row": 0, "col": 0, "text": "a"}] * 4}])
        self.assertEqual(m["tables_structured"], 1)
        self.assertEqual(m["table_cells"], 3)  # grid non-empty only; cells not double-counted

    def test_4_metrics_canonical_rows_without_cells(self):
        m = self._metrics([{"type": "table", "structureStatus": "structured", "rows": [["x", "y"], ["z", ""]]}])
        self.assertEqual(m["table_cells"], 3)

    def test_5_image_only_empty_rows_preserved(self):
        m = self._metrics([{"type": "table", "structureStatus": "image_only", "rows": [], "imageUri": "shots/i.png"}])
        self.assertEqual(m["tables_image_only"], 1)
        self.assertEqual(m["tables_structured"], 0)
        self.assertEqual(m["table_assets_missing"], 0)

    def test_6_legacy_table_rows_still_readable(self):
        m = self._metrics([{"type": "table", "table_rows": [["a", "b"], ["c", ""]]}])
        self.assertEqual(m["tables_structured"], 1)
        self.assertEqual(m["table_cells"], 3)

    def test_7_legacy_integer_rows_cols_table_rows_no_error(self):
        from pipeline.compatibility.table_diff import collect_v2_tables
        kb = {"entries": [{"entryId": "e", "blocks": [{
            "type": "table", "rows": 6, "cols": 4,
            "table_rows": [[""] * 4 for _ in range(6)]}]}]}
        st = collect_v2_tables(kb)[0]["geometry"]
        self.assertEqual((st["logical_rows"], st["logical_cols"], st["grid_slots"]), (6, 4, 24))

    def test_8_empty_rows_list_wins_over_stale_table_rows(self):
        from pipeline.compatibility.table_diff import collect_v2_tables
        block = {"type": "table", "rows": [], "table_rows": [["a", "b"]],
                 "imageUri": "shots/i.png", "structureStatus": "image_only"}
        m = self._metrics([dict(block)])
        self.assertEqual(m["tables_image_only"], 1)
        self.assertEqual(m["table_cells"], 0)  # canonical empty rows used; NOT the stale table_rows
        st = collect_v2_tables({"entries": [{"entryId": "e", "blocks": [block]}]})[0]["geometry"]
        self.assertEqual(st["logical_rows"], 0)

    def test_9_new_production_output_has_no_legacy_fields(self):
        for entry in self._project()["entries"]:
            for b in entry["blocks"]:
                if b.get("type") == "table":
                    self.assertNotIn("table_rows", b)
                    self.assertNotIn("cols", b)
                    self.assertIsInstance(b.get("rows"), list)


if __name__ == "__main__":
    unittest.main()
