"""v2 output publication-boundary regression tests.

Every assertion here is made against final files, the run_report status/outputs
and the process exit code — not against internal helpers alone.

Failure modes are injected deterministically:
  * post-write schema validation failure
  * serializer interruption during the KB write
  * a KB block that claims a deliverable asset whose file is absent

The CLI is driven with a fake pipeline that stages artifacts through the real
publication helpers, so most tests need no PDF/OCR models. A separate test
builds and runs the installed wheel from outside the repository.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from paddle_models.cli import main as cli  # noqa: E402
from paddle_models.cli.main import check_v2_referenced_assets  # noqa: E402

try:
    from pipeline import v2_runner as vr  # noqa: E402

    _VR_ERROR = None
except Exception as exc:  # pragma: no cover - environment without pipeline deps
    vr = None
    _VR_ERROR = exc


class _DocIRStub:
    """Minimal Canonical IR object exposing ``to_dict`` (schema-valid)."""

    _payload = {"document_id": "t", "schema_version": "2.0", "pages": []}

    def to_dict(self):
        return dict(self._payload)


def _img_block(bid="b1", uri="shots/ok.png", status="ready", page=1):
    return {
        "id": bid,
        "type": "image",
        "bbox": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        "pageNumber": page,
        "imageUri": uri,
        "src": uri,
        "assetStatus": status,
    }


def _minimal_kb(*, blocks=None, entries=None):
    if entries is None:
        entries = []
        if blocks is not None:
            entries = [
                {
                    "entryId": "e1",
                    "jobTitle": "t",
                    "contentNormalized": "x",
                    "contentMarkdown": "x",
                    "pageNumber": 1,
                    "blocks": blocks,
                }
            ]
    return {
        "fileMetadata": {
            "schemaVersion": "2.0",
            "fileId": "t",
            "docSha256": "a" * 64,
            "pageSizes": {"1": [1.0, 1.0]},
        },
        "entries": entries,
    }


class _CliV2TestCase(unittest.TestCase):
    """Base: fixed run dir, faked dependency probe, fake staging pipeline."""

    def setUp(self):
        if vr is None:
            self.skipTest(f"pipeline.v2_runner unavailable: {_VR_ERROR}")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.input = self.tmp / "input.pdf"
        self.input.write_bytes(b"%PDF-1.4 dummy")
        self.out = self.tmp / "out"
        self._run_id = "run_fixed"
        self._orig = {
            "run_v2": vr.run_v2,
            "json": vr.json,
            "missing": cli.missing_hard_deps,
            "run_id": cli.make_run_id,
            "validate": cli.validate_against_schema,
        }
        # Faked dependency probe: the injected pipeline never imports fitz.
        cli.missing_hard_deps = lambda mode, caps: []
        cli.make_run_id = lambda stem: self._run_id

    def tearDown(self):
        vr.run_v2 = self._orig["run_v2"]
        vr.json = self._orig["json"]
        cli.missing_hard_deps = self._orig["missing"]
        cli.make_run_id = self._orig["run_id"]
        cli.validate_against_schema = self._orig["validate"]
        self._tmp.cleanup()

    # -- helpers -----------------------------------------------------------
    def install_pipeline(self, kb, *, shots=()):
        """Fake pipeline: write shots + fr report, then stage IR + KB."""
        def _fake(ctx, page_range, *, publish=True):
            out = Path(ctx.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            for rel in shots:
                p = out / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
            (out / "figure_reference_index.json").write_text(
                json.dumps({"stats": {}}), encoding="utf-8"
            )
            vr.write_staged_outputs(out, _DocIRStub(), kb)
            return None, kb

        vr.run_v2 = _fake

    def interrupt_kb_write(self):
        """Make the 2nd serialization (the KB) raise mid-write."""
        real_dump = self._orig["json"].dump
        state = {"calls": 0}

        def _bad_dump(obj, fp, *a, **k):
            state["calls"] += 1
            if state["calls"] == 2:
                fp.write('{"entries": [')
                fp.flush()
                raise OSError("simulated interruption during KB write")
            return real_dump(obj, fp, *a, **k)

        vr.json = types.SimpleNamespace(dump=_bad_dump)

    def run_cli(self):
        return cli.main(
            [
                "--mode", "v2",
                "--input", str(self.input),
                "--out", str(self.out),
                "--profile", "smoke",
                "--max-pages", "1",
            ]
        )

    @property
    def v2_dir(self):
        return self.out / self._run_id / "v2"

    def final_kb(self):
        return self.v2_dir / "knowledge_base.json"

    def final_ir(self):
        return self.v2_dir / "knowledge_base.v2.json"

    def staging_files(self):
        staging = self.v2_dir / ".staging"
        return sorted(p.name for p in staging.glob("*")) if staging.is_dir() else []

    def run_report(self):
        path = self.out / self._run_id / "reports" / "run_report.json"
        return json.loads(path.read_text(encoding="utf-8"))

    # -- tests -------------------------------------------------------------
    def test_success_publishes_final_outputs_and_consistent_report(self):
        self.install_pipeline(_minimal_kb(blocks=[_img_block()]), shots=["shots/ok.png"])
        rc = self.run_cli()
        self.assertEqual(rc, 0)
        self.assertTrue(self.final_kb().is_file())
        self.assertTrue(self.final_ir().is_file())
        self.assertEqual(self.staging_files(), [])
        rep = self.run_report()
        self.assertEqual(rep["status"], "ok")
        v2_out = rep["outputs"]["v2"]
        self.assertTrue(v2_out["published"])
        self.assertEqual(Path(v2_out["knowledge_base"]), self.final_kb())
        self.assertEqual(
            Path(v2_out["canonical_ir"]), self.final_ir()
        )
        # Both authoritative schemas validate the published artifacts.
        self.assertEqual(
            self._orig["validate"](self.final_kb(), cli.KB_SCHEMA_REL, "kb")["status"],
            "valid",
        )
        self.assertEqual(
            self._orig["validate"](self.final_ir(), cli.IR_SCHEMA_REL, "ir")["status"],
            "valid",
        )

    def test_post_validation_failure_leaves_no_final_kb(self):
        self.install_pipeline(_minimal_kb(blocks=[_img_block()]), shots=["shots/ok.png"])
        cli.validate_against_schema = lambda *a, **k: {
            "status": "invalid",
            "schema": "injected",
            "errors": ["injected schema failure"],
        }
        rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertFalse(self.final_kb().exists())
        self.assertFalse(self.final_ir().exists())
        self.assertEqual(self.staging_files(), [])
        rep = self.run_report()
        self.assertEqual(rep["status"], "failed")
        v2_out = rep["outputs"]["v2"]
        self.assertFalse(v2_out["published"])
        self.assertNotIn("knowledge_base", v2_out)
        self.assertNotIn("canonical_ir", v2_out)

    def test_write_interruption_leaves_no_final_kb(self):
        self.install_pipeline(_minimal_kb(blocks=[_img_block()]), shots=["shots/ok.png"])
        self.interrupt_kb_write()
        rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertFalse(self.final_kb().exists())
        self.assertFalse(self.final_ir().exists())
        self.assertEqual(self.staging_files(), [])
        self.assertEqual(self.run_report()["status"], "failed")

    def test_referenced_asset_anomaly_fails_without_publishing(self):
        kb = _minimal_kb(blocks=[_img_block(uri="shots/missing.png", status="ready")])
        self.install_pipeline(kb, shots=[])  # referenced file is absent
        rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertFalse(self.final_kb().exists())
        rep = self.run_report()
        self.assertEqual(rep["status"], "failed")
        self.assertFalse(rep["outputs"]["v2"]["published"])
        joined = " ".join(rep.get("errors") or [])
        self.assertIn("missing referenced asset", joined)

    def test_marked_missing_asset_is_still_a_success(self):
        # Degradation is sanctioned: an empty uri / assetStatus=missing must not
        # be converted into a failure just to make every reference resolve.
        kb = _minimal_kb(blocks=[_img_block(uri="", status="missing")])
        self.install_pipeline(kb, shots=[])
        rc = self.run_cli()
        self.assertEqual(rc, 0)
        self.assertTrue(self.final_kb().is_file())
        self.assertEqual(self.run_report()["status"], "ok")

    def test_failed_rerun_preserves_existing_valid_outputs(self):
        self.install_pipeline(_minimal_kb(blocks=[_img_block()]), shots=["shots/ok.png"])
        self.v2_dir.mkdir(parents=True, exist_ok=True)
        self.final_kb().write_bytes(b"SENTINEL-KB")
        self.final_ir().write_bytes(b"SENTINEL-IR")
        cli.validate_against_schema = lambda *a, **k: {
            "status": "invalid",
            "schema": "injected",
            "errors": ["injected schema failure"],
        }
        rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertEqual(self.final_kb().read_bytes(), b"SENTINEL-KB")
        self.assertEqual(self.final_ir().read_bytes(), b"SENTINEL-IR")
        self.assertEqual(self.staging_files(), [])

    def test_success_after_failure_publishes_cleanly(self):
        self.install_pipeline(_minimal_kb(blocks=[_img_block()]), shots=["shots/ok.png"])
        self.interrupt_kb_write()
        self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.final_kb().exists())
        # Second attempt, healthy serializer, same run directory.
        vr.json = self._orig["json"]
        self.install_pipeline(_minimal_kb(blocks=[_img_block()]), shots=["shots/ok.png"])
        rc = self.run_cli()
        self.assertEqual(rc, 0)
        self.assertTrue(self.final_kb().is_file())
        self.assertEqual(self.run_report()["status"], "ok")

    # -- published-output protection --------------------------------------
    def _seed_published_outputs(self, shot="shots/ok.png"):
        """Create a complete old deliverable set: KB + IR + the same-named shot."""
        shot_path = self.v2_dir / shot
        shot_path.parent.mkdir(parents=True, exist_ok=True)
        shot_path.write_bytes(b"OLD-SHOT")
        self.final_kb().write_bytes(b"OLD-KB")
        self.final_ir().write_bytes(b"OLD-IR")
        return shot_path

    def _arm_pipeline_must_not_run(self):
        ran = {"pipeline": False}

        def _boom(ctx, page_range=None, *, publish=True):
            ran["pipeline"] = True
            raise AssertionError("pipeline must not run for a published run dir")

        vr.run_v2 = _boom
        return ran

    def _assert_old_set_intact(self, shot_path, ran):
        self.assertEqual(self.final_kb().read_bytes(), b"OLD-KB")
        self.assertEqual(self.final_ir().read_bytes(), b"OLD-IR")
        self.assertEqual(shot_path.read_bytes(), b"OLD-SHOT")
        self.assertFalse(ran["pipeline"], "pipeline must not have run")
        self.assertEqual(self.staging_files(), [])
        rep = self.run_report()
        self.assertEqual(rep["status"], "failed")
        v2_out = rep["outputs"]["v2"]
        self.assertFalse(v2_out["published"])
        self.assertNotIn("knowledge_base", v2_out)
        self.assertNotIn("canonical_ir", v2_out)
        self.assertIn(
            "already contains a published", " ".join(rep.get("errors") or [])
        )

    def test_published_rerun_publish_rename_failure_preserves_old_set(self):
        shot_path = self._seed_published_outputs()
        ran = self._arm_pipeline_must_not_run()
        real_replace = os.replace
        state = {"replace": 0}

        def _failing_replace(src, dst, *a, **k):
            state["replace"] += 1
            if str(dst).replace("\\\\?\\", "").endswith("knowledge_base.json"):
                raise OSError("injected publish KB rename failure")
            return real_replace(src, dst, *a, **k)

        with mock.patch.object(vr.os, "replace", _failing_replace):
            rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertEqual(state["replace"], 0, "guard must trip before any rename")
        self._assert_old_set_intact(shot_path, ran)

    def test_published_rerun_schema_failure_preserves_old_set(self):
        shot_path = self._seed_published_outputs()
        ran = self._arm_pipeline_must_not_run()
        state = {"validate": 0}

        def _spy_validate(*a, **k):
            state["validate"] += 1
            return {"status": "invalid", "schema": "injected", "errors": ["injected"]}

        cli.validate_against_schema = _spy_validate
        rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertEqual(state["validate"], 0, "guard must trip before validation")
        self._assert_old_set_intact(shot_path, ran)

    def _install_partial_pipeline(self, *, drop):
        kb = _minimal_kb()

        def _fake(ctx, page_range=None, *, publish=True):
            out = Path(ctx.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "figure_reference_index.json").write_text(
                json.dumps({"stats": {}}), encoding="utf-8"
            )
            vr.write_staged_outputs(out, _DocIRStub(), kb)
            staged_kb, staged_ir = vr.v2_staged_paths(out)
            (staged_kb if drop == "knowledge_base.json" else staged_ir).unlink()
            return None, kb

        vr.run_v2 = _fake

    def test_missing_staged_kb_is_not_reported_success(self):
        self._install_partial_pipeline(drop="knowledge_base.json")
        rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertFalse(self.final_kb().exists())
        rep = self.run_report()
        self.assertEqual(rep["status"], "failed")
        self.assertFalse(rep["outputs"]["v2"]["published"])
        self.assertNotIn("knowledge_base", rep["outputs"]["v2"])

    def test_missing_staged_ir_is_not_reported_success(self):
        self._install_partial_pipeline(drop="knowledge_base.v2.json")
        rc = self.run_cli()
        self.assertEqual(rc, 1)
        self.assertFalse(self.final_kb().exists())
        self.assertFalse(self.final_ir().exists())
        rep = self.run_report()
        self.assertEqual(rep["status"], "failed")
        self.assertFalse(rep["outputs"]["v2"]["published"])
        self.assertIn("staged outputs incomplete", " ".join(rep.get("errors") or []))


class TestReferencedAssetCheck(unittest.TestCase):
    """Pure unit coverage of the contract-required asset gate."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="v2_assets_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_present_relative_asset_ok(self):
        (self.tmp / "shots").mkdir()
        (self.tmp / "shots" / "a.png").write_bytes(b"x")
        kb = _minimal_kb(blocks=[_img_block(uri="shots/a.png")])
        self.assertEqual(check_v2_referenced_assets(kb, self.tmp), [])

    def test_absent_relative_asset_reported(self):
        kb = _minimal_kb(blocks=[_img_block(uri="shots/a.png")])
        errors = check_v2_referenced_assets(kb, self.tmp)
        self.assertEqual(len(errors), 1)
        self.assertIn("shots/a.png", errors[0])

    def test_marked_missing_and_empty_uri_ignored(self):
        kb = _minimal_kb(
            blocks=[_img_block(uri="", status="missing"), _img_block("b2", "", "missing")]
        )
        self.assertEqual(check_v2_referenced_assets(kb, self.tmp), [])

    def test_non_local_uri_ignored(self):
        kb = _minimal_kb(
            blocks=[
                _img_block(uri="https://example.com/a.png"),
                _img_block("b2", str(self.tmp / "abs.png")),
            ]
        )
        self.assertEqual(check_v2_referenced_assets(kb, self.tmp), [])


class TestStagedHelpers(unittest.TestCase):
    """The staging helpers must never leave a file at the final path."""

    def setUp(self):
        if vr is None:
            self.skipTest(f"pipeline.v2_runner unavailable: {_VR_ERROR}")
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "v2"
        self.dir.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_discard_preserves_existing_final(self):
        final_kb, final_ir = vr.v2_output_paths(self.dir)
        final_kb.write_bytes(b"OLD-KB")
        final_ir.write_bytes(b"OLD-IR")
        staged_kb, staged_ir = vr.write_staged_outputs(self.dir, _DocIRStub(), _minimal_kb())
        self.assertTrue(staged_kb.is_file() and staged_ir.is_file())
        self.assertEqual(final_kb.read_bytes(), b"OLD-KB")
        self.assertEqual(final_ir.read_bytes(), b"OLD-IR")
        vr.discard_staged_outputs(self.dir)
        self.assertFalse(staged_kb.exists())
        self.assertFalse(staged_ir.exists())
        self.assertEqual(final_kb.read_bytes(), b"OLD-KB")

    def test_interrupted_write_leaves_nothing_staged_or_final(self):
        orig = vr.json
        real_dump = orig.dump
        state = {"calls": 0}

        def _bad_dump(obj, fp, *a, **k):
            state["calls"] += 1
            if state["calls"] == 2:
                fp.write("{partial")
                fp.flush()
                raise OSError("simulated interruption")
            return real_dump(obj, fp, *a, **k)

        vr.json = types.SimpleNamespace(dump=_bad_dump)
        try:
            with self.assertRaises(OSError):
                vr.write_staged_outputs(self.dir, _DocIRStub(), _minimal_kb())
        finally:
            vr.json = orig
        final_kb, final_ir = vr.v2_output_paths(self.dir)
        self.assertFalse(final_kb.exists())
        self.assertFalse(final_ir.exists())
        self.assertEqual(sorted(p.name for p in (self.dir / ".staging").glob("*")), [])

    def test_publish_moves_staged_to_final(self):
        staged_kb, staged_ir = vr.write_staged_outputs(self.dir, _DocIRStub(), _minimal_kb())
        final_kb, final_ir = vr.publish_staged_outputs(self.dir)
        self.assertTrue(final_kb.is_file())
        self.assertTrue(final_ir.is_file())
        self.assertFalse(staged_kb.exists())
        self.assertFalse(staged_ir.exists())

    def test_publish_refuses_to_overwrite_published_kb(self):
        final_kb, final_ir = vr.v2_output_paths(self.dir)
        final_kb.write_bytes(b"OLD-KB")
        final_ir.write_bytes(b"OLD-IR")
        vr.write_staged_outputs(self.dir, _DocIRStub(), _minimal_kb())
        with self.assertRaises(FileExistsError):
            vr.publish_staged_outputs(self.dir)
        self.assertEqual(final_kb.read_bytes(), b"OLD-KB")
        self.assertEqual(final_ir.read_bytes(), b"OLD-IR")

    def test_publish_requires_both_staged_files(self):
        staged_kb, staged_ir = vr.v2_staged_paths(self.dir)
        staged_kb.parent.mkdir(parents=True, exist_ok=True)
        vr._atomic_write_json(staged_ir, _DocIRStub().to_dict())  # IR only, no KB
        with self.assertRaises(FileNotFoundError):
            vr.publish_staged_outputs(self.dir)
        final_kb, final_ir = vr.v2_output_paths(self.dir)
        self.assertFalse(final_kb.exists())
        self.assertFalse(final_ir.exists())

    def test_publish_rolls_back_on_rename_failure(self):
        vr.write_staged_outputs(self.dir, _DocIRStub(), _minimal_kb())
        real_replace = os.replace
        state = {"calls": 0}

        def _failing_replace(src, dst, *a, **k):
            state["calls"] += 1
            if state["calls"] == 2:  # KB rename (the commit point) fails
                raise OSError("injected KB rename failure")
            return real_replace(src, dst, *a, **k)

        with mock.patch.object(vr.os, "replace", _failing_replace):
            with self.assertRaises(OSError):
                vr.publish_staged_outputs(self.dir)
        final_kb, final_ir = vr.v2_output_paths(self.dir)
        self.assertFalse(final_kb.exists())
        self.assertFalse(final_ir.exists(), "half-published IR must be rolled back")


class TestInstalledWheelExternalRun(unittest.TestCase):
    """Build the wheel, install it to a target, run from outside the repo.

    This is an *install/entry* smoke (plus a tiny synthetic-PDF v2 run when
    PyMuPDF is available). It is deliberately NOT the 158-page real acceptance
    run, which stays local and is never committed.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.target = self.tmp / "site"
        self.wheel_dir = self.tmp / "wheel"
        self.wheel_dir.mkdir()
        proc = subprocess.run(
            [
                sys.executable, "-m", "pip", "wheel", "--no-deps",
                "--no-build-isolation", "-w", str(self.wheel_dir), str(REPO_ROOT),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        wheels = sorted(self.wheel_dir.glob("paddle_models-*.whl"))
        if proc.returncode != 0 or not wheels:
            self.skipTest("wheel build unavailable (pip/setuptools missing)")
        self.wheel = wheels[-1]
        proc = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--no-deps",
                "--target", str(self.target), str(self.wheel),
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            self.skipTest(f"wheel install to --target failed: {proc.stderr[-400:]}")

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, args, cwd):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.target)
        env["PADDLE_CACHE_ROOT"] = str(self.tmp / "cache")
        return subprocess.run(
            [sys.executable, "-m", "paddle_models.cli.main", *args],
            cwd=str(cwd), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=900,
        )

    def test_wheel_contains_entrypoint_and_schemas(self):
        with zipfile.ZipFile(self.wheel) as zf:
            names = set(zf.namelist())
        for member in (
            "paddle_models/cli/main.py",
            "pipeline/v2_runner.py",
            "contracts/knowledge_base_schema_v2.json",
            "contracts/knowledge-base.v2.schema.json",
        ):
            self.assertIn(member, names)

    def test_help_runs_from_outside_repo(self):
        workdir = self.tmp / "outside"
        workdir.mkdir()
        proc = self._run(["--help"], workdir)
        self.assertEqual(proc.returncode, 0, proc.stderr[-600:])
        self.assertIn("--mode", proc.stdout)

    def test_real_tiny_pdf_run_from_outside_repo(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed in this interpreter")
        workdir = self.tmp / "outside"
        workdir.mkdir()
        pdf = workdir / "tiny.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "PaddleModels publication smoke")
        doc.save(pdf)
        doc.close()
        out = workdir / "out"
        proc = self._run(
            [
                "--mode", "v2", "--input", str(pdf), "--out", str(out),
                "--profile", "smoke", "--max-pages", "1",
            ],
            workdir,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-1500:] + proc.stderr[-1500:])
        runs = [p for p in out.glob("*") if p.is_dir()]
        self.assertEqual(len(runs), 1)
        v2_dir = runs[0] / "v2"
        self.assertTrue((v2_dir / "knowledge_base.json").is_file())
        self.assertTrue((v2_dir / "knowledge_base.v2.json").is_file())


if __name__ == "__main__":
    unittest.main()
