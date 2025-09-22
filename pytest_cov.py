"""Lightweight pytest-cov stub accepting coverage options."""

from __future__ import annotations

from typing import Any


def pytest_addoption(parser: Any) -> None:  # pragma: no cover - integration hook
    group = parser.getgroup("cov")
    group.addoption("--cov", action="append", default=[], help="ignored stub option")
    group.addoption(
        "--cov-report", action="append", default=[], help="ignored stub option"
    )


def pytest_configure(config: Any) -> None:  # pragma: no cover - integration hook
    # No-op: coverage reporting is unavailable in this offline environment.
    pass
