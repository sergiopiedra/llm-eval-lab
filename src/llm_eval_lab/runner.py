"""Runs the golden dataset through the RAG pipeline, checks, judge, metrics and gates."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from llm_eval_lab.checks import run_checks
from llm_eval_lab.config import GOLDEN_PATH, RUNS_DIR, EvalConfig
from llm_eval_lab.dataset import GoldenCase, dataset_fingerprint, load_golden
from llm_eval_lab.judge import Judge
from llm_eval_lab.llm import LLMClient, make_client
from llm_eval_lab.metrics import (
    case_failures,
    case_verdicts,
    compute_metrics,
    evaluate_gates,
    summary_stats,
)
from llm_eval_lab.rag import RAGPipeline
from llm_eval_lab.retriever import BM25Retriever


def evaluate_case(case: GoldenCase, pipeline: RAGPipeline, judge: Judge, pass_score: int) -> dict:
    resp = pipeline.answer(case.question)
    checks = run_checks(case, resp, pipeline.retriever.doc_ids)
    verdict = judge.grade(case.question, case.expected_answer, resp.context, resp.answer)
    record = case.to_dict()
    response = resp.to_dict()
    response.pop("context")  # large and reproducible from `retrieved`; keep run files small
    record.update(response=response, checks=checks, judge=verdict.to_dict())
    record["verdicts"] = case_verdicts(record, record["judge"], checks, pass_score)
    record["failure_reasons"] = case_failures(record["verdicts"])
    record["passed"] = not record["failure_reasons"]
    return record


def run_eval(
    config: EvalConfig,
    *,
    cases: list[GoldenCase] | None = None,
    llm: LLMClient | None = None,
    judge_llm: LLMClient | None = None,
    tag: str = "run",
    mock_behavior: str = "good",
    progress: Callable[[int, int, dict], None] | None = None,
) -> dict:
    cases = cases if cases is not None else load_golden()
    llm = llm or make_client(config.backend, config.model, config, cases, mock_behavior)
    judge_llm = judge_llm or make_client(
        config.backend, config.judge_model, config, cases, mock_behavior
    )
    if hasattr(llm, "check_ready"):
        llm.check_ready()
    if judge_llm is not llm and hasattr(judge_llm, "check_ready"):
        judge_llm.check_ready()

    pipeline = RAGPipeline(llm, BM25Retriever.from_dir(), top_k=config.top_k)
    judge = Judge(judge_llm)

    results = []
    for i, case in enumerate(cases, start=1):
        record = evaluate_case(case, pipeline, judge, config.judge_pass_score)
        results.append(record)
        if progress:
            progress(i, len(cases), record)

    metrics = compute_metrics(results)
    gates = evaluate_gates(metrics, config.gates)
    created = datetime.now(UTC)
    return {
        "run_id": f"{created:%Y%m%dT%H%M%SZ}_{_slug(tag)}",
        "tag": tag,
        "created_at": created.isoformat(timespec="seconds"),
        "config": {
            "backend": config.backend,
            "model": llm.model,
            "judge_model": judge_llm.model,
            "top_k": config.top_k,
            "temperature": config.temperature,
            "seed": config.seed,
            "judge_pass_score": config.judge_pass_score,
            "dataset": GOLDEN_PATH.name,
            "dataset_sha": dataset_fingerprint(),
        },
        "summary": summary_stats(results),
        "metrics": metrics,
        "gates": gates,
        "passed": all(g["passed"] for g in gates),
        "cases": results,
    }


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", text).strip("-") or "run"


def save_run(run: dict, runs_dir: Path | str = RUNS_DIR) -> Path:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run['run_id']}.json"
    path.write_text(json.dumps(run, indent=2))
    return path


def load_run(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
