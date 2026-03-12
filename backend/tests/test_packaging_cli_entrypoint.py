from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_uses_setuptools_build_backend() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    build_system = data["build-system"]
    assert build_system["build-backend"] == "setuptools.build_meta"
    assert "setuptools" in " ".join(build_system["requires"])


def test_pyproject_includes_src_package_for_cli_script() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    packages_find = data["tool"]["setuptools"]["packages"]["find"]
    assert "src*" in packages_find["include"]

    scripts = data["project"]["scripts"]
    assert scripts["deerflow"] == "src.cli:main"
