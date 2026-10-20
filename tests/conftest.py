"""Shared pytest options and fixtures.

`pytest` defaults to the deterministic mock backend so it is safe to run anywhere (and in CI).
Evaluate the real model with:  pytest tests/evals --backend ollama [--model qwen3:8b]
"""

from __future__ import annotations

import os

import pytest

from llm_eval_lab.config import load_config
from llm_eval_lab.dataset import load_golden
from llm_eval_lab.llm import LLMError
from llm_eval_lab.runner import run_eval, save_run


def pytest_addoption(parser):
    group = parser.getgroup("llm-eval")
    group.addoption("--backend", choices=["mock", "ollama"], default=None,
                    help="model backend (default: $LLM_EVAL_BACKEND or 'mock')")
    group.addoption("--model", default=None, help="model under test, e.g. qwen3:8b")
    group.addoption("--judge-model", default=None, help="judge model (default: from config)")
    group.addoption("--save-run", action="store_true", help="save the eval run to runs/ for reports")
    group.addoption("--tag", default="pytest", help="tag stored with a saved run")


@pytest.fixture(scope="session")
def eval_config(pytestconfig):
    backend = pytestconfig.getoption("--backend") or os.environ.get("LLM_EVAL_BACKEND") or "mock"
    return load_config(
        backend=backend,
        model=pytestconfig.getoption("--model"),
        judge_model=pytestconfig.getoption("--judge-model"),
    )


@pytest.fixture(scope="session")
def golden():
    return load_golden()


@pytest.fixture(scope="session")
def eval_run(eval_config, golden, pytestconfig):
    """Run the whole golden dataset once per session; every eval test reads from it."""
    try:
        run = run_eval(eval_config, cases=golden, tag=pytestconfig.getoption("--tag"))
    except LLMError as exc:
        pytest.exit(f"LLM backend unavailable: {exc}", returncode=3)
    if pytestconfig.getoption("--save-run"):
        path = save_run(run)
        print(f"\nSaved eval run to {path}")
    return run


@pytest.fixture(scope="session")
def cases_by_id(eval_run):
    return {c["id"]: c for c in eval_run["cases"]}
