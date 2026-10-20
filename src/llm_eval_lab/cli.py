"""Command line interface: `llm-eval run | report | ask`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_eval_lab.config import REPORTS_DIR, RUNS_DIR, load_config
from llm_eval_lab.dataset import load_golden
from llm_eval_lab.llm import LLMError, make_client
from llm_eval_lab.metrics import METRICS
from llm_eval_lab.rag import RAGPipeline
from llm_eval_lab.report import write_reports
from llm_eval_lab.retriever import BM25Retriever
from llm_eval_lab.runner import load_run, run_eval, save_run


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="path to eval_config.toml")
    p.add_argument("--backend", choices=["ollama", "mock"])
    p.add_argument("--model", help="model under test (default qwen3:8b)")
    p.add_argument("--host", help="Ollama base URL")
    p.add_argument("--top-k", type=int)


def _config(args) -> object:
    return load_config(
        args.config,
        backend=args.backend,
        model=args.model,
        host=args.host,
        top_k=args.top_k,
        judge_model=getattr(args, "judge_model", None),
    )


def cmd_run(args) -> int:
    config = _config(args)
    cases = load_golden()
    if args.category:
        cases = [c for c in cases if c.category == args.category]
    if args.limit:
        cases = cases[: args.limit]

    def progress(i: int, total: int, record: dict) -> None:
        mark = "PASS" if record["passed"] else "FAIL " + ",".join(record["failure_reasons"])
        print(f"[{i:>2}/{total}] {record['id']:<8} {mark}", flush=True)

    print(f"Evaluating {len(cases)} cases · backend={config.backend} model={config.model} "
          f"judge={config.judge_model}")
    try:
        run = run_eval(config, cases=cases, tag=args.tag, mock_behavior=args.mock_behavior, progress=progress)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    path = save_run(run, args.runs_dir)

    print("\nMetric                       Value    Gate        Status")
    gates = {g["metric"]: g for g in run["gates"]}
    for name in ["retrieval_recall", *METRICS]:
        value = run["metrics"][name]["value"]
        gate = gates.get(name)
        shown = "n/a" if value is None else f"{value:.1%}"
        rule = f"{'>=' if gate['op'] == 'min' else '<='} {gate['threshold']:.0%}" if gate else "-"
        status = "-" if not gate else ("pass" if gate["passed"] else "FAIL")
        print(f"{name:<28} {shown:>7}  {rule:<10}  {status}")
    print(f"\nSaved {path}")
    print("QUALITY GATES PASSED" if run["passed"] else "QUALITY GATES FAILED")
    return 0 if run["passed"] or args.no_fail else 1


def cmd_report(args) -> int:
    latest = sorted(Path(args.runs_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)[-2:]
    paths = [Path(p) for p in args.runs] or latest
    if not paths:
        print("error: no run files found; run `llm-eval run` first", file=sys.stderr)
        return 2
    runs = [load_run(p) for p in paths]
    md, html_path = write_reports(runs, args.out_dir)
    print(f"Compared {len(runs)} run(s): {', '.join(r['run_id'] for r in runs)}")
    print(f"Wrote {md}\nWrote {html_path}")
    return 0


def cmd_ask(args) -> int:
    config = _config(args)
    llm = make_client(config.backend, config.model, config, load_golden())
    pipeline = RAGPipeline(llm, BM25Retriever.from_dir(), top_k=config.top_k)
    try:
        resp = pipeline.answer(args.question)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("Retrieved:")
    for r in resp.retrieved:
        print(f"  {r['score']:>6.2f}  {r['doc_id']} › {r['section']}")
    print(f"\nAnswer: {resp.answer}")
    print(f"Citations: {', '.join(resp.citations) or '-'} · answerable={resp.answerable} "
          f"· {resp.latency_s:.2f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate the golden dataset and save a run file")
    _add_model_args(run)
    run.add_argument("--judge-model", help="judge model (default: same as config)")
    run.add_argument("--tag", default="run", help="label for this run, e.g. 'baseline'")
    run.add_argument("--category", choices=["answerable", "unanswerable", "injection"])
    run.add_argument("--limit", type=int, help="only evaluate the first N cases")
    run.add_argument("--runs-dir", default=RUNS_DIR)
    run.add_argument("--mock-behavior", choices=["good", "faulty"], default="good")
    run.add_argument("--no-fail", action="store_true", help="exit 0 even if gates fail")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="compare run files in Markdown and HTML")
    report.add_argument("runs", nargs="*", help="run JSON files, baseline first (default: last 2)")
    report.add_argument("--runs-dir", default=RUNS_DIR)
    report.add_argument("--out-dir", default=REPORTS_DIR)
    report.set_defaults(func=cmd_report)

    ask = sub.add_parser("ask", help="ask the RAG demo a single question")
    _add_model_args(ask)
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
