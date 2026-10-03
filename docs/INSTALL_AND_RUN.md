# Install and Run (wheel)

The `paddlemodels` command ships as a wheel that bundles the runtime packages
(`paddle_models`, `pipeline`, `imaging`, `models`, `utils`, `classification`,
`tools`) together with the contract schemas (`contracts/*.json`). The installed
command needs **no** source checkout on `sys.path` and **no** manually set
`PYTHONPATH`; it can be run from any working directory.

## Install

```bash
python -m pip install .
# or build then install a wheel
python -m pip wheel --no-deps -w dist .
python -m pip install --no-deps dist/paddle_models-*.whl
```

Runtime dependencies are declared in `pyproject.toml`
(`PyMuPDF`, `Pillow`, `jsonschema`, `numpy`, `opencv-python`, ...).

## Run

```bash
paddlemodels --help
paddlemodels --mode v2 --input path/to/doc.pdf --out /some/external/out
```

All outputs (KB/IR JSON, reports, cropped assets, page cache) are written under
`--out`; nothing is written back into the installed package.

## Modes and installed-environment support

| Mode | Installed wheel | Notes |
|---|---|---|
| `v2` | Fully supported | Packages + schemas are bundled; results validate against the packaged schema. |
| `legacy` | Supported | Uses the bundled `tools/pdf_to_base64_kb.py`. If an install omits it, the legacy side returns `blocked_by_dependency` with an explicit message — it never fake-succeeds. |
| `shadow` | Supported | Runs legacy + v2 and writes `reports/shadow_diff.json`; requires both modes above. |

## Contract schemas

The KB and Canonical IR are validated at runtime against
`contracts/knowledge_base_schema_v2.json` and
`contracts/knowledge-base.v2.schema.json`. Schemas are located through the
installed `contracts` package (`importlib.resources`), with a source-checkout
fallback — never via a hard-coded repository path.

## External wheel smoke test

```bash
python scripts/smoke_wheel_external.py --sample "samples/103号(2).pdf"
```

The script builds the wheel, creates a throwaway virtualenv **outside** the
repository, installs the wheel with `--no-deps`, and runs `paddlemodels --help`
(and optionally a full `v2` run on `--sample`) with an empty `PYTHONPATH` and a
cwd outside the repo. Use `--deps-from <site-packages>` to reuse an existing
environment's dependencies offline instead of the default `--system-site-packages`.
