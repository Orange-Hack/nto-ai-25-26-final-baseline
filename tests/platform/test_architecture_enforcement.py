"""Architectural enforcement tests to ensure Two-Zone boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "src"
TESTS_DIR = ROOT_DIR / "tests"


def get_all_python_files(directory: Path) -> list[Path]:
    """Recursively find all .py files in a directory."""
    return list(directory.rglob("*.py"))


def check_imports(file_path: Path) -> list[str]:
    """Analyze imports in a file and return violations."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except SyntaxError:
            return []

    violations = []
    rel_path = file_path.relative_to(ROOT_DIR)

    # Forbidden legacy namespaces
    LEGACY_NAMESPACES = {
        "src.pipeline",
        "src.core",
        "src.participants",
        "src.io",
        "src.utils",
        "src.candidates",
        "src.ranking",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for legacy in LEGACY_NAMESPACES:
                    if alias.name.startswith(legacy):
                        violations.append(f"Forbidden legacy import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for legacy in LEGACY_NAMESPACES:
                    if node.module.startswith(legacy):
                        violations.append(f"Forbidden legacy import from: {node.module}")

                # Boundary rule: competition -> platform/pipeline (forbidden)
                if str(rel_path).startswith("src/competition"):
                    if node.module.startswith("src.platform.pipeline"):
                        violations.append(
                            f"Boundary violation: competition logic depends on platform pipeline: {node.module}"
                        )

    return violations


@pytest.mark.parametrize("file_path", get_all_python_files(SRC_DIR) + get_all_python_files(TESTS_DIR))
def test_architectural_boundaries(file_path: Path) -> None:
    """Verify that file follows Two-Zone import rules."""
    violations = check_imports(file_path)
    if violations:
        pytest.fail(f"Architectural violations in {file_path}:\n" + "\n".join(violations))


def test_no_solution_logic_in_platform() -> None:
    """Verify that platform/ does not contain solution-specific logic."""
    PLATFORM_DIR = SRC_DIR / "platform"
    forbidden_keywords = ["features", "generators", "ranking"]

    for file_path in get_all_python_files(PLATFORM_DIR):
        # Skip stages/workflows which are allowed to CALL solution logic
        if "stages" in str(file_path) or "workflows" in str(file_path):
            continue

        rel_path = file_path.relative_to(SRC_DIR)
        for keyword in forbidden_keywords:
            if keyword in rel_path.name.lower():
                pytest.fail(
                    f"Platform module '{rel_path}' seems to contain solution logic (keyword: {keyword})"
                )
