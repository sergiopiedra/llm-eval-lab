"""Configuration loading.

Precedence (highest first): explicit overrides (CLI flags / pytest options) > environment
variables > eval_config.toml > built-in defaults.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "eval_config.toml"
DOCS_DIR = REPO_ROOT / "data" / "docs"
GOLDEN_PATH = REPO_ROOT / "data" / "golden.jsonl"
CALIBRATION_PATH = REPO_ROOT / "data" / "judge_calibration.jsonl"
RUNS_DIR = REPO_ROOT / "runs"
REPORTS_DIR = REPO_ROOT / "reports"

ENV_VARS = {
    "backend": "LLM_EVAL_BACKEND",
    "model": "LLM_EVAL_MODEL",
    "judge_model": "LLM_EVAL_JUDGE_MODEL",
    "host": "OLLAMA_HOST",
}


@dataclass(frozen=True)
class Gate:
    metric: str
    op: str  # "min" or "max"
    threshold: float

    def check(self, value: float | None) -> bool:
        if value is None:
            return False
        return value >= self.threshold if self.op == "min" else value <= self.threshold


@dataclass(frozen=True)
class EvalConfig:
    backend: str = "ollama"
    model: str = "qwen3:8b"
    judge_model: str = "qwen3:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 8192
    timeout_s: float = 180.0
    top_k: int = 4
    judge_pass_score: int = 4
    min_calibration_agreement: float = 0.85
    gates: tuple[Gate, ...] = field(default_factory=tuple)

    def with_overrides(self, **overrides) -> EvalConfig:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})


def _normalize_host(host: str) -> str:
    # OLLAMA_HOST is often set without a scheme (e.g. "0.0.0.0:11434").
    return host if host.startswith(("http://", "https://")) else f"http://{host}"


def load_config(path: Path | str | None = None, **overrides) -> EvalConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = tomllib.loads(path.read_text()) if path.exists() else {}
    model = raw.get("model", {})

    values = {
        "backend": model.get("backend"),
        "model": model.get("name"),
        "judge_model": model.get("judge"),
        "host": model.get("host"),
        "temperature": model.get("temperature"),
        "seed": model.get("seed"),
        "num_ctx": model.get("num_ctx"),
        "timeout_s": model.get("timeout_s"),
        "top_k": raw.get("retrieval", {}).get("top_k"),
        "judge_pass_score": raw.get("judge", {}).get("pass_score"),
        "min_calibration_agreement": raw.get("judge", {}).get("min_calibration_agreement"),
    }
    for key, env in ENV_VARS.items():
        if os.environ.get(env):
            values[key] = os.environ[env]
    values.update({k: v for k, v in overrides.items() if v is not None})

    gates = []
    for metric, spec in raw.get("gates", {}).items():
        (op, threshold), = spec.items()
        if op not in ("min", "max"):
            raise ValueError(f"gate {metric!r}: expected 'min' or 'max', got {op!r}")
        gates.append(Gate(metric, op, float(threshold)))
    values["gates"] = tuple(gates)

    config = EvalConfig().with_overrides(**values)
    return replace(config, host=_normalize_host(config.host))
