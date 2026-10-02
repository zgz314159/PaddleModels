"""Architecture boundary guard: the active pipeline must not import retired code.

Read-only static analysis over source text. Standard library only (`ast`,
`pathlib`, `unittest`) — no dependencies, no product imports, no cwd/installed-package
assumptions. Files are located from this file's absolute position.

The guard scans the active production scope and the test tree; it never scans the
retired packages' own internal imports. Violations are reported with file, line and
module. Counts/LOC/absolute-import totals are intentionally NOT asserted so unrelated
new files cannot make this guard flaky.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD_TEST_PATH = Path(__file__).resolve()

# Retired experiment zone — these modules must not be re-imported.
FORBIDDEN_PREFIXES = (
    "paddle_models.application",
    "paddle_models.core",
    "paddle_models.domain",
    "paddle_models.infrastructure",
    "paddle_models.contracts",
    "paddle_models.utils",
    "paddle_models.validation",
)

# Active production scope (directories + single files), relative to REPO_ROOT.
ACTIVE_DIRS = (
    "src/paddle_models/cli",
    "pipeline",
    "imaging",
    "models",
)
ACTIVE_FILES = ("tools/pdf_to_base64_kb.py",)
TESTS_DIR = "tests"

_SKIP_DIR_NAMES = {
    "__pycache__",
    ".venv",
    ".venv_ocr",
    ".venv3.11",
    ".git",
    ".cache",
    "build",
    "dist",
}


def _iter_py(root: Path):
    for path in sorted(root.rglob("*.py")):
        if any(part in _SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def _active_files():
    files = []
    for rel in ACTIVE_DIRS:
        directory = REPO_ROOT / rel
        if directory.is_dir():
            files.extend(_iter_py(directory))
    for rel in ACTIVE_FILES:
        single = REPO_ROOT / rel
        if single.is_file():
            files.append(single)
    return sorted(set(files))


def _test_files():
    directory = REPO_ROOT / TESTS_DIR
    if not directory.is_dir():
        return []
    return sorted(_iter_py(directory))


def _display(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in FORBIDDEN_PREFIXES
    )


def _literal_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dynamic_import_target(node: ast.Call):
    """Return the literal module string of an import_module()/__import__() call."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "__import__":
        arg = node.args[0] if node.args else None
        return _literal_str(arg) if arg is not None else None
    if isinstance(func, ast.Attribute) and func.attr == "import_module":
        arg = node.args[0] if node.args else None
        return _literal_str(arg) if arg is not None else None
    if isinstance(func, ast.Name) and func.id == "import_module":
        arg = node.args[0] if node.args else None
        return _literal_str(arg) if arg is not None else None
    return None


def _collect_violations(tree: ast.AST, display: str):
    """Violations are (display_path, lineno, module) for forbidden imports.

    Only real import statements and literal dynamic imports count. Plain string
    constants (e.g. this file's own FORBIDDEN_PREFIXES tuple or synthetic source
    strings in the self-tests) are never treated as violations.
    """
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden(alias.name):
                    violations.append((display, node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                if _is_forbidden(node.module):
                    violations.append((display, node.lineno, node.module))
                for alias in node.names:
                    candidate = f"{node.module}.{alias.name}"
                    if _is_forbidden(candidate):
                        violations.append((display, node.lineno, candidate))
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target is not None and _is_forbidden(target):
                violations.append((display, node.lineno, target))
    return violations


def _scan_file(path: Path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    tree = ast.parse(text, filename=str(path))
    return _collect_violations(tree, _display(path))


def _scan_source(source: str, display: str = "<memory>"):
    return _collect_violations(ast.parse(source), display)


def _format(violations):
    if not violations:
        return ""
    lines = [
        f"  {path}:{lineno}: forbidden import of '{module}'"
        for path, lineno, module in sorted(violations)
    ]
    return "Retired architecture import(s) detected:\n" + "\n".join(lines)


def _imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="replace"), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
            for alias in node.names:
                modules.add(f"{node.module}.{alias.name}")
    return modules


class TestActivePathHasNoRetiredImports(unittest.TestCase):
    def test_active_scope_is_nonempty(self):
        # Functional presence guard (not a fixed file count): the CLI entry must be scanned.
        self.assertIn(REPO_ROOT / "src/paddle_models/cli/main.py", _active_files())

    def test_active_path_imports_no_retired_package(self):
        violations = []
        for path in _active_files():
            violations.extend(_scan_file(path))
        self.assertEqual(violations, [], _format(violations))


class TestTestsHaveNoRetiredImports(unittest.TestCase):
    def test_test_tree_is_nonempty(self):
        self.assertTrue(_test_files())

    def test_test_tree_imports_no_retired_package(self):
        violations = []
        for path in _test_files():
            violations.extend(_scan_file(path))
        self.assertEqual(violations, [], _format(violations))

    def test_guard_file_itself_is_clean(self):
        # Guards against the guard's own blocklist strings being mistaken for imports.
        self.assertEqual(_scan_file(GUARD_TEST_PATH), [])


class TestCliEntrypointWiring(unittest.TestCase):
    def setUp(self):
        self.modules = _imported_modules(REPO_ROOT / "src/paddle_models/cli/main.py")

    def test_cli_still_references_active_v2_runner(self):
        self.assertTrue(
            any(m == "pipeline.v2_runner" or m.startswith("pipeline.v2_runner.") for m in self.modules),
            f"src/paddle_models/cli/main.py no longer imports pipeline.v2_runner "
            f"(imports: {sorted(self.modules)})",
        )

    def test_cli_does_not_reference_retired_build_document(self):
        offenders = [m for m in self.modules if "build_document" in m]
        self.assertEqual(offenders, [], f"retired build_document referenced: {offenders}")
        self.assertFalse(any(m.startswith("paddle_models.application") for m in self.modules))


class TestDetectorSemantics(unittest.TestCase):
    """Self-tests proving the scanner catches real imports and ignores plain strings."""

    def test_detects_static_from_import(self):
        found = _scan_source("from paddle_models.core.cache import PageCache\n")
        self.assertTrue(any(m.startswith("paddle_models.core") for _, _, m in found), found)

    def test_detects_plain_import(self):
        found = _scan_source("import paddle_models.validation.comparison\n")
        self.assertTrue(found)

    def test_detects_from_paddle_models_alias(self):
        found = _scan_source("from paddle_models import domain\n")
        self.assertTrue(any(m == "paddle_models.domain" for _, _, m in found), found)

    def test_detects_importlib_import_module_literal(self):
        found = _scan_source(
            'import importlib\nimportlib.import_module("paddle_models.domain.models")\n'
        )
        self.assertTrue(any(m == "paddle_models.domain.models" for _, _, m in found), found)

    def test_detects_dunder_import_literal(self):
        found = _scan_source('__import__("paddle_models.infrastructure.pdf.renderer")\n')
        self.assertTrue(found)

    def test_plain_string_literal_is_not_a_violation(self):
        found = _scan_source('X = "paddle_models.core.cache"\n')
        self.assertEqual(found, [])

    def test_active_imports_are_allowed(self):
        found = _scan_source(
            "from pipeline.v2_runner import run_v2\n"
            "from paddle_models.cli.main import main\n"
            "from imaging.ocr_engine import ocr_image_bytes\n"
        )
        self.assertEqual(found, [])


if __name__ == "__main__":
    unittest.main()
