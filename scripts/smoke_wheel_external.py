#!/usr/bin/env python3
"""External-directory wheel smoke test for the ``paddlemodels`` CLI.

Builds the wheel, creates a throwaway virtualenv OUTSIDE the repository,
installs the wheel with ``--no-deps`` and runs the installed console script
with an empty ``PYTHONPATH`` and a cwd outside the repo. This proves the CLI is
independent of the working directory, the source checkout and any manual
``PYTHONPATH``.

Stdlib only. Dependency reuse is offline: by default the venv is created with
``--system-site-packages``; pass ``--deps-from <site-packages>`` to instead add
that directory via a ``.pth`` file.

Examples:
    python scripts/smoke_wheel_external.py
    python scripts/smoke_wheel_external.py --sample "samples/103号(2).pdf"
    python scripts/smoke_wheel_external.py --deps-from .venv/Lib/site-packages

Exit code 0 = pass, 1 = failure (message printed).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd, *, cwd=None, env=None, capture=True):
    print("+ " + " ".join(str(c) for c in cmd))
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _fail(message: str, proc=None) -> "NoReturn":  # type: ignore[name-defined]
    print("\n[SMOKE][FAIL] " + message)
    if proc is not None:
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if tail:
            print(tail[-2000:])
    raise SystemExit(1)


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _venv_launcher(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "paddlemodels.exe"
    return venv_dir / "bin" / "paddlemodels"


def _clean_env() -> dict:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    return env


def main(argv=None) -> int:
    # Keep console output readable when the sample path contains non-ASCII text
    # on a non-UTF-8 (e.g. GBK) Windows console.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Base interpreter for building/venv")
    parser.add_argument("--sample", default=None, help="Optional PDF for a full v2 run")
    parser.add_argument("--out", default=None, help="Scratch root (default: a temp dir)")
    parser.add_argument("--deps-from", default=None, help="Reuse this site-packages offline (adds a .pth)")
    parser.add_argument("--keep", action="store_true", help="Keep the scratch directory")
    args = parser.parse_args(argv)

    scratch = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="pmsmoke_"))
    scratch.mkdir(parents=True, exist_ok=True)
    wheel_dir = scratch / "wheel"
    venv_dir = scratch / "venv"
    work_dir = scratch / "work"
    for directory in (wheel_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[SMOKE] repo={REPO_ROOT}")
        print(f"[SMOKE] scratch={scratch}")

        # 1. Build the wheel.
        build = _run(
            [args.python, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
             "-w", wheel_dir, REPO_ROOT]
        )
        wheels = sorted(wheel_dir.glob("paddle_models-*.whl"))
        if build.returncode != 0 or not wheels:
            _fail("wheel build failed", build)
        wheel = wheels[-1]
        print(f"[SMOKE] built {wheel.name}")

        # 2. Create an external virtualenv.
        venv_cmd = [args.python, "-m", "venv"]
        if not args.deps_from:
            venv_cmd.append("--system-site-packages")
        venv_cmd.append(venv_dir)
        created = _run(venv_cmd)
        if created.returncode != 0:
            _fail("venv creation failed", created)
        if args.deps_from:
            pth = _venv_python(venv_dir).parent.parent / "Lib" / "site-packages" / "deps.pth"
            if os.name != "nt":
                pth = _venv_python(venv_dir).parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "deps.pth"
            pth.parent.mkdir(parents=True, exist_ok=True)
            pth.write_text(str(Path(args.deps_from).resolve()), encoding="utf-8")
            print(f"[SMOKE] reused deps from {args.deps_from}")

        # 3. Install the wheel (no dependency resolution: offline).
        installed = _run([_venv_python(venv_dir), "-m", "pip", "install", "--no-deps", "--force-reinstall", wheel])
        if installed.returncode != 0:
            _fail("wheel install failed", installed)

        launcher = _venv_launcher(venv_dir)
        if not launcher.exists():
            _fail(f"console script not installed: {launcher}")

        # 4. `--help` from outside the repo with a clean environment.
        helped = _run([launcher, "--help"], cwd=work_dir, env=_clean_env())
        if helped.returncode != 0 or "usage:" not in (helped.stdout or ""):
            _fail("`paddlemodels --help` failed", helped)
        print("[SMOKE] paddlemodels --help OK")

        # 5. Optional full v2 run.
        if args.sample:
            sample = Path(args.sample)
            if not sample.is_absolute():
                sample = (REPO_ROOT / sample).resolve()
            if not sample.is_file():
                _fail(f"sample not found: {sample}")
            out_dir = work_dir / "v2_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            # The page cache nests a 64-char sha under the cache root; Windows'
            # MAX_PATH would overflow when the scratch path is deep, so point the
            # cache at a short temp dir (documented PADDLE_CACHE_ROOT override).
            run_env = _clean_env()
            run_env["PADDLE_CACHE_ROOT"] = str(Path(tempfile.gettempdir()) / "paddlemodels_smoke_cache")
            run = _run(
                [launcher, "--mode", "v2", "--input", sample, "--out", out_dir],
                cwd=out_dir, env=run_env,
            )
            if run.returncode != 0:
                _fail("`paddlemodels --mode v2` failed", run)
            reports = sorted(out_dir.glob("*/reports/run_report.json"), key=lambda p: p.stat().st_mtime)
            if not reports:
                _fail("no run_report.json produced")
            report = json.loads(reports[-1].read_text(encoding="utf-8"))
            if report.get("status") != "ok":
                _fail(f"v2 status={report.get('status')} errors={report.get('errors')}")
            metrics = report.get("metrics") or {}
            cv = (report.get("contract_validation") or {}).get("v2") or {}
            print(
                "[SMOKE] v2 OK: pages={pages} entries={entries} "
                "kb_schema={kb} ir_schema={ir}".format(
                    pages=metrics.get("pages"),
                    entries=metrics.get("entries"),
                    kb=(cv.get("knowledge_base") or {}).get("status"),
                    ir=(cv.get("canonical_ir") or {}).get("status"),
                )
            )

        print("\n[SMOKE][PASS] installed wheel runs outside the repo with a clean environment")
        return 0
    finally:
        if not args.keep:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
