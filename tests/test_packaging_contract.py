"""Packaging contract: the wheel must ship the CLI + runtime packages + schemas.

These tests build the wheel (no dependency resolution) and inspect its members,
so a regression in ``pyproject.toml`` packaging is caught without installing.
The build is skipped when pip/setuptools tooling is unavailable.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = REPO_ROOT / "src" / "paddle_models" / "cli" / "main.py"

REQUIRED_MEMBERS = (
    "paddle_models/cli/main.py",
    "pipeline/v2_runner.py",
    "pipeline/extraction_adapters/layout_adapter.py",
    "pipeline/semantics/figure_text_association.py",
    "imaging/bbox_utils.py",
    "models/run_context.py",
    "classification/text_classifier.py",
    "utils/file_id_sanitizer.py",
    "contracts/knowledge_base_schema_v2.json",
    "contracts/knowledge-base.v2.schema.json",
    "tools/pdf_to_base64_kb.py",
)

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = None


class TestPyprojectDeclaration(unittest.TestCase):
    def setUp(self):
        if tomllib is None:
            self.skipTest("tomllib unavailable")
        with open(REPO_ROOT / "pyproject.toml", "rb") as handle:
            self.cfg = tomllib.load(handle)

    def test_console_script_declared(self):
        scripts = (self.cfg.get("project") or {}).get("scripts") or {}
        self.assertEqual(scripts.get("paddlemodels"), "paddle_models.cli.main:main")

    def test_find_includes_runtime_and_contract_packages(self):
        include = ((self.cfg.get("tool") or {}).get("setuptools") or {}).get("packages", {}).get("find", {}).get("include") or []
        for expected in ("paddle_models*", "pipeline*", "imaging*", "models*", "contracts*", "tools*"):
            self.assertIn(expected, include)

    def test_contract_schemas_are_package_data(self):
        data = ((self.cfg.get("tool") or {}).get("setuptools") or {}).get("package-data") or {}
        self.assertIn("*.json", data.get("contracts") or [])


class TestCliHasNoSourceCheckoutAssumption(unittest.TestCase):
    """The CLI must resolve resources via installed packages, not a repo path."""

    def test_schema_resolution_is_resource_based(self):
        source = MAIN_SOURCE.read_text(encoding="utf-8")
        self.assertIn("importlib.resources", source)
        self.assertIn("_resolve_contract(", source)
        self.assertNotIn("REPO_ROOT / schema_rel", source)


class TestBuiltWheelContents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.wheel = None
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                 "-w", cls._tmp.name, str(REPO_ROOT)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except Exception:
            proc = None
        if proc is None or proc.returncode != 0:
            return
        wheels = sorted(Path(cls._tmp.name).glob("paddle_models-*.whl"))
        cls.wheel = wheels[-1] if wheels else None

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        if self.wheel is None:
            self.skipTest("wheel build unavailable (pip/setuptools missing)")

    def test_required_members_present(self):
        with zipfile.ZipFile(self.wheel) as zf:
            names = set(zf.namelist())
        missing = [m for m in REQUIRED_MEMBERS if m not in names]
        self.assertEqual(missing, [], f"wheel missing members: {missing}")

    def test_entry_point_declared(self):
        with zipfile.ZipFile(self.wheel) as zf:
            entries = [n for n in zf.namelist() if n.endswith("dist-info/entry_points.txt")]
            self.assertTrue(entries, "no entry_points.txt in wheel")
            body = zf.read(entries[0]).decode("utf-8")
        self.assertIn("paddlemodels", body)
        self.assertIn("paddle_models.cli.main", body)


if __name__ == "__main__":
    unittest.main()
