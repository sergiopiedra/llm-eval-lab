"""Helpers shared by the eval tests."""

from __future__ import annotations

import pytest

from llm_eval_lab.dataset import load_golden

GOLDEN = load_golden()


def ids(category: str | None = None, answerable: bool | None = None) -> list[str]:
    """Case ids for parametrizing per-case tests at collection time."""
    return [
        c.id
        for c in GOLDEN
        if (category is None or c.category == category)
        and (answerable is None or c.answerable == answerable)
    ]


def assert_gate(run: dict, metric: str) -> None:
    gate = next((g for g in run["gates"] if g["metric"] == metric), None)
    if gate is None:
        pytest.skip(f"no quality gate configured for {metric!r}")
    m = run["metrics"][metric]
    value = "n/a" if m["value"] is None else f"{m['value']:.1%}"
    rule = f"{'>=' if gate['op'] == 'min' else '<='} {gate['threshold']:.0%}"
    assert gate["passed"], (
        f"quality gate failed: {metric} = {value} (required {rule}, n={m['n']}). "
        f"Offending cases: {', '.join(m['failures']) or '-'}"
    )
