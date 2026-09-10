"""Regression tests for release-version consistency."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess
import sys
import tomllib

import powernap


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+"


def _project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def _first_match(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, f"Version is missing from {path.relative_to(ROOT)}"
    return match.group(1)


def test_release_version_is_consistent_across_project_files() -> None:
    package_version = powernap.__version__
    project_version = _project_version()
    readme_version = _first_match(
        ROOT / "README.md",
        rf"^> \*\*Project status:\*\* PowerNap ({VERSION_PATTERN})\b",
    )
    changelog_version = _first_match(
        ROOT / "CHANGELOG.md",
        rf"^## ({VERSION_PATTERN})$",
    )

    assert package_version == project_version == readme_version == changelog_version


def test_cli_version_matches_package_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "powernap.cli", "--version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == powernap.__version__

