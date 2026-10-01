"""Phase 2B tests: legacy KB contract adapter + table structure diff."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipeline.compatibility.legacy_kb import (  # noqa: E402
    ADAPTER_NAME,
    normalize_legacy_kb_contract,
)
from pipeline.compatibility.table_diff import (  # noqa: E402
    build_table_diff_report,
    cells_diff,
    collect_legacy_tables,
    match_tables,
)
from paddle_models.cli.main import (  # noqa: E402
    apply_contract_validation,
    kb_metrics_from_obj,
    normalize_text,
)

KB_SCHEMA = REPO_ROOT / "contracts" / "knowledge_base_schema_v2.json"


def _sample_kb(*, with_ids: bool = True) -> dict:
    blocks = []
    for i in range(3):
        b = {
            "type": "code",
            "code": f"text-{i}",
            "pageNumber": 1,
            "bbox": {"left": 0, "top": i * 10, "right": 100, "bottom": i * 10 + 8},
        }
        if with_ids:
            b["id"] = f"existing_{i}"
        blocks.append(b)
    # two identical content blocks — need unique ids via blockIndex
    identical = [
        {"type": "image", "imageUri": "shots/a.png", "pageNumber": 1, "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1}},
        {"type": "image", "imageUri": "shots/a.png", "pageNumber": 1, "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1}},
    ]
    if with_ids:
        identical[0]["id"] = "img_0"
        identical[1]["id"] = "img_1"
    else:
        for b in identical:
            b["id"] = ""  # blank
    return {
        "fileMetadata": {
            "schemaVersion": "2.0",
            "fileId": "doc1",
            "docSha256": "a" * 64,
        },
        "entries": [
            {
                "entryId": "e1",
                "jobTitle": "t",
                "contentNormalized": "hello world",
                "contentMarkdown": "hello world",
                "pageNumber": 1,
                "blocks": blocks + identical,
            }
        ],
    }


class TestNormalizeLegacyKb(unittest.TestCase):
    def test_missing_id_filled(self):
        kb = _sample_kb(with_ids=False)
        # remove all ids
        for b in kb["entries"][0]["blocks"]:
            b.pop("id", None)
        new_kb, report = normalize_legacy_kb_contract(kb)
        self.assertTrue(report["applied"])
        # 3 text blocks (no id key) + 2 identical blank-id blocks = 5
        self.assertEqual(report["transforms"]["missing_block_ids"], 5)
        self.assertEqual(len(report["changed_paths"]), 5)
        self.assertEqual(report["adapter"], ADAPTER_NAME)
        for b in new_kb["entries"][0]["blocks"]:
            self.assertTrue(b.get("id"))

    def test_blank_id_filled(self):
        kb = _sample_kb(with_ids=True)
        kb["entries"][0]["blocks"][0]["id"] = "   "
        new_kb, report = normalize_legacy_kb_contract(kb)
        self.assertTrue(report["applied"])
        self.assertEqual(report["transforms"]["missing_block_ids"], 1)
        self.assertEqual(report["changed_paths"], ["entries.0.blocks.0.id"])
        self.assertTrue(new_kb["entries"][0]["blocks"][0]["id"].strip())

    def test_existing_ids_preserved(self):
        kb = _sample_kb(with_ids=True)
        orig_ids = [b.get("id") for b in kb["entries"][0]["blocks"]]
        new_kb, report = normalize_legacy_kb_contract(kb)
        self.assertFalse(report["applied"])
        self.assertEqual(report["transforms"]["missing_block_ids"], 0)
        new_ids = [b.get("id") for b in new_kb["entries"][0]["blocks"]]
        self.assertEqual(orig_ids, new_ids)

    def test_identical_blocks_unique_ids(self):
        kb = _sample_kb(with_ids=False)
        for b in kb["entries"][0]["blocks"]:
            b.pop("id", None)
        new_kb, _ = normalize_legacy_kb_contract(kb)
        ids = [b["id"] for b in new_kb["entries"][0]["blocks"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_repeatable_stable(self):
        kb = _sample_kb(with_ids=False)
        for b in kb["entries"][0]["blocks"]:
            b.pop("id", None)
        a, _ = normalize_legacy_kb_contract(copy.deepcopy(kb))
        b2, _ = normalize_legacy_kb_contract(copy.deepcopy(kb))
        self.assertEqual(
            [x["id"] for x in a["entries"][0]["blocks"]],
            [x["id"] for x in b2["entries"][0]["blocks"]],
        )

    def test_input_not_mutated(self):
        kb = _sample_kb(with_ids=False)
        for b in kb["entries"][0]["blocks"]:
            b.pop("id", None)
        snapshot = copy.deepcopy(kb)
        normalize_legacy_kb_contract(kb)
        self.assertEqual(kb, snapshot)

    def test_only_id_field_changes(self):
        kb = _sample_kb(with_ids=False)
        for b in kb["entries"][0]["blocks"]:
            b.pop("id", None)
        new_kb, _ = normalize_legacy_kb_contract(kb)
        # Compare after stripping id from new
        stripped = copy.deepcopy(new_kb)
        for b in stripped["entries"][0]["blocks"]:
            b.pop("id", None)
        self.assertEqual(stripped, kb)
        # metrics/hash unchanged
        m0 = kb_metrics_from_obj(kb)
        m1 = kb_metrics_from_obj(new_kb)
        self.assertEqual(m0["normalized_text_sha256"], m1["normalized_text_sha256"])
        self.assertEqual(m0["entries"], m1["entries"])
        self.assertEqual(m0["tables"], m1["tables"])
        self.assertEqual(m0["table_cells"], m1["table_cells"])


class TestSchemaRawVsCompat(unittest.TestCase):
    def test_raw_invalid_compat_valid(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        schema = json.loads(KB_SCHEMA.read_text(encoding="utf-8"))
        raw = _sample_kb(with_ids=False)
        for b in raw["entries"][0]["blocks"]:
            b.pop("id", None)
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=raw, schema=schema)
        compat, report = normalize_legacy_kb_contract(raw)
        self.assertTrue(report["applied"])
        jsonschema.validate(instance=compat, schema=schema)

    def test_applied_false_when_no_change(self):
        kb = _sample_kb(with_ids=True)
        _, report = normalize_legacy_kb_contract(kb)
        self.assertFalse(report["applied"])
        self.assertEqual(report["transforms"]["missing_block_ids"], 0)

    def test_apply_contract_validation_raw_invalid_compat_ok(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            self.skipTest("jsonschema not installed")
        import tempfile

        raw = _sample_kb(with_ids=False)
        for b in raw["entries"][0]["blocks"]:
            b.pop("id", None)
        compat, report = normalize_legacy_kb_contract(raw)
        with tempfile.TemporaryDirectory() as td:
            raw_path = Path(td) / "knowledge_base.raw.json"
            compat_path = Path(td) / "knowledge_base.json"
            raw_path.write_text(json.dumps(raw), encoding="utf-8")
            compat_path.write_text(json.dumps(compat), encoding="utf-8")
            result = {
                "status": "ok",
                "outputs": {
                    "knowledge_base_raw": str(raw_path),
                    "knowledge_base": str(compat_path),
                },
                "errors": [],
                "warnings": [],
                "compatibility": {
                    "adapter": ADAPTER_NAME,
                    "applied": report["applied"],
                    "transforms": report["transforms"],
                },
            }
            apply_contract_validation(result, side="legacy")
            cv = result["contract_validation"]
            self.assertEqual(cv["raw_knowledge_base"]["status"], "invalid")
            self.assertEqual(cv["knowledge_base"]["status"], "valid")
            self.assertEqual(result["status"], "ok")  # not demoted
            self.assertTrue(
                any("compatibility adapter" in w for w in result["warnings"])
            )


class TestTableDiff(unittest.TestCase):
    def test_row_col_span_diffs(self):
        leg = {
            "rows": 2,
            "cols": 2,
            "table_rows": [["a", "b"], ["c", "d"]],
            "cells_raw": None,
        }
        v2 = {
            "rows": 2,
            "cols": 2,
            "table_rows": [["a", "b"], ["c", "X"]],
            "cells_raw": None,
        }
        d = cells_diff(leg, v2)
        self.assertEqual(d["legacy_nonempty_cells"], 4)
        self.assertEqual(d["v2_nonempty_cells"], 4)
        # text mismatch on (1,1)
        texts = [x for x in d["span_or_position_diffs"] if "text" in x["diff_fields"]]
        self.assertTrue(texts)

    def test_legacy_without_bbox_uses_page_order(self):
        leg = [
            {"page_number": 29, "bbox": None, "rows": 1, "cols": 1, "table_rows": [["x"]], "nonempty_cells": 1, "id": "L1", "entry_index": 0, "block_index": 0},
            {"page_number": 29, "bbox": None, "rows": 1, "cols": 1, "table_rows": [["y"]], "nonempty_cells": 1, "id": "L2", "entry_index": 0, "block_index": 1},
        ]
        v2 = [
            {"page_number": 29, "bbox": None, "rows": 1, "cols": 1, "table_rows": [["x"]], "nonempty_cells": 1, "id": "V1", "entry_index": 0, "block_index": 0},
            {"page_number": 29, "bbox": None, "rows": 1, "cols": 1, "table_rows": [["y"]], "nonempty_cells": 1, "id": "V2", "entry_index": 0, "block_index": 1},
        ]
        pairs = match_tables(leg, v2)
        self.assertEqual(len(pairs), 2)
        methods = [m for _, _, m in pairs]
        self.assertIn("page+order", methods)
        # no false cell diffs for identical tables
        for lt, vt, _ in pairs:
            if lt and vt:
                d = cells_diff(lt, vt)
                self.assertEqual(len(d["only_legacy"]), 0)
                self.assertEqual(len(d["only_v2"]), 0)
                self.assertEqual(len(d["span_or_position_diffs"]), 0)

    def test_identical_tables_no_false_diff(self):
        leg = {"rows": 2, "cols": 2, "table_rows": [["h1", "h2"], ["1", "2"]], "cells_raw": None}
        v2 = {"rows": 2, "cols": 2, "table_rows": [["h1", "h2"], ["1", "2"]], "cells_raw": None}
        d = cells_diff(leg, v2)
        self.assertEqual(d["only_legacy"], [])
        self.assertEqual(d["only_v2"], [])
        self.assertEqual(d["span_or_position_diffs"], [])

    def test_bbox_overlap_match(self):
        leg = [{
            "page_number": 29, "bbox": (20.0, 200.0, 200.0, 80.0),
            "rows": 1, "cols": 1, "table_rows": [["x"]], "nonempty_cells": 1,
            "id": "L", "entry_index": 0, "block_index": 0,
        }]
        v2 = [{
            "page_number": 29, "bbox": (22.0, 206.0, 199.0, 86.0),
            "rows": 1, "cols": 1, "table_rows": [["x"]], "nonempty_cells": 1,
            "id": "V", "entry_index": 0, "block_index": 0, "structureStatus": "structured",
        }]
        pairs = match_tables(leg, v2)
        self.assertEqual(pairs[0][2], "page+bbox")

    def test_report_structure(self):
        leg_kb = {
            "entries": [{
                "entryId": "e",
                "blocks": [{
                    "type": "table",
                    "pageNumber": 29,
                    "table_rows": [["a", "b"], ["c", "d"], ["e", ""]],
                    "bbox": {"left": 10, "top": 10, "right": 100, "bottom": 50},
                }],
            }],
        }
        v2_kb = {
            "entries": [{
                "entryId": "e",
                "blocks": [{
                    "type": "table",
                    "pageNumber": 29,
                    "table_rows": [["a", "b"], ["c", "d"], ["e", "f"]],
                    "bbox": {"left": 10, "top": 10, "right": 100, "bottom": 50},
                    "structureStatus": "structured",
                }],
            }],
        }
        report = build_table_diff_report(leg_kb, v2_kb, page_number=29)
        self.assertEqual(report["legacy_table_count"], 1)
        self.assertEqual(report["v2_table_count"], 1)
        self.assertEqual(len(report["comparisons"]), 1)
        cmp0 = report["comparisons"][0]
        self.assertEqual(cmp0["match_method"], "page+bbox")
        self.assertIn("legacy_markdown_preview", cmp0)
        self.assertIn("v2_markdown_preview", cmp0)
        # nonempty: legacy 5, v2 6
        self.assertEqual(cmp0["cells"]["legacy_nonempty_cells"], 5)
        self.assertEqual(cmp0["cells"]["v2_nonempty_cells"], 6)
        self.assertTrue(report["cell_count_notes"])


class TestFixtureManifest(unittest.TestCase):
    def test_table_cases_fixture_exists(self):
        path = REPO_ROOT / "tests" / "fixtures" / "table_cases.json"
        self.assertTrue(path.exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("pdf", data)
        pages = {c["page"] for c in data["cases"]}
        self.assertIn(29, pages)
        self.assertIn(28, pages)
        self.assertIn(30, pages)
        # no embedded binary
        self.assertFalse(str(data["pdf"]).endswith(".pdf") and "://" in str(data["pdf"]))


if __name__ == "__main__":
    unittest.main()
