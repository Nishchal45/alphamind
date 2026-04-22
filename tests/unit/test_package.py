"""Smoke tests for the alphamind package."""

from __future__ import annotations

import re

import alphamind


def test_package_exposes_version() -> None:
    assert isinstance(alphamind.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", alphamind.__version__)
