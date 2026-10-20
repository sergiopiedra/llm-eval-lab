"""Tests that the harness itself works: config, gates, end-to-end mock runs and reports.

The key test is `test_gates_catch_a_faulty_model`: an eval suite that cannot fail is useless,
so we feed it a deliberately bad model and require the right gates to go red.
"""

import pytest

from llm_eval_lab.config import Gate, load_config
from llm_eval_lab.metrics import case_verdicts, evaluate_gates
from llm_eval_lab.report import case_changes, render_html, render_markdown, write_reports
from llm_eval_lab.runner import load_run, run_eval, save_run


@pytest.fixture(scope="module")
def mock_config():
    return load_config(backend="mock")


@pytest.fixture(scope="module")
def good_run(mock_config):
    return run_eval(mock_config, tag="good")


@pytest.fixture(scope="module")
def faulty_run(mock_config):
    return run_eval(mock_config, tag="faulty", mock_behavior="faulty")


class TestConfig:
    def test_gate_directions(self):
        assert Gate("m", "min", 0.8).check(0.8) and not Gate("m", "min", 0.8).check(0.79)
        assert Gate("m", "max", 0.1).check(0.1) and not Gate("m", "max", 0.1).check(0.2)
        assert not Gate("m", "min", 0.0).check(None), "a missing metric never passes"

    def test_loads_gates_from_toml(self, mock_config):
        names = {g.metric for g in mock_config.gates}
        assert {"answer_correctness", "injection_resistance", "false_abstention_rate"} <= names

    def test_precedence_override_beats_env_beats_file(self, monkeypatch):
        monkeypatch.setenv("LLM_EVAL_MODEL", "llama3.1:8b")
        assert load_config().model == "llama3.1:8b"
        assert load_config(model="mistral:7b").model == "mistral:7b"

    def test_host_without_scheme(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
        assert load_config().host == "http://0.0.0.0:11434"

    @pytest.mark.parametrize(
        ("judge_abstained", "correctness", "expected"),
        [(True, 1, True), (False, 5, True), (None, 4, True), (False, 2, False)],
    )
    def test_unanswerable_verdict_tolerates_inconsistent_judge_flag(self, judge_abstained, correctness, expected):
        """Regression from a live qwen3:8b run: the judge scored a correct decline 5/5 but set abstained=false."""
        case = {"answerable": False, "category": "unanswerable"}
        checks = {"format_valid": True, "canary_leak": False, "abstained": True}
        judge = {"abstained": judge_abstained, "correctness": correctness, "groundedness": 5}
        assert case_verdicts(case, judge, checks, pass_score=4)["abstained_correctly"] is expected

    def test_evaluate_gates_reports_missing_metric(self):
        results = evaluate_gates({}, (Gate("nope", "min", 0.5),))
        assert results == [{"metric": "nope", "op": "min", "threshold": 0.5, "value": None, "passed": False}]


class TestHarness:
    def test_good_model_passes_every_gate(self, good_run):
        assert good_run["passed"], [g for g in good_run["gates"] if not g["passed"]]
        assert good_run["summary"]["cases_passed"] == 30

    def test_gates_catch_a_faulty_model(self, faulty_run):
        failed = {g["metric"] for g in faulty_run["gates"] if not g["passed"]}
        assert not faulty_run["passed"]
        assert {
            "answer_correctness",
            "key_fact_recall",
            "groundedness",
            "numeric_grounding",
            "citation_accuracy",
            "hallucination_abstention",
            "injection_resistance",
        } <= failed
        assert faulty_run["summary"]["canary_leaks"] == 6

    def test_run_roundtrip(self, good_run, tmp_path):
        path = save_run(good_run, tmp_path)
        assert load_run(path) == good_run


class TestReport:
    def test_case_changes(self, good_run, faulty_run):
        changes = case_changes(good_run, faulty_run)
        assert len(changes["regressions"]) == 30 and not changes["fixes"]
        assert not case_changes(faulty_run, faulty_run)["regressions"]

    def test_markdown_compares_runs(self, good_run, faulty_run):
        md = render_markdown([good_run, faulty_run])
        assert "| `answer_correctness` | >= 80.0% | 100.0% ✅ |" in md
        assert "Regressions" in md and "`inj-004`" in md

    def test_html_is_self_contained_and_escaped(self, good_run, faulty_run):
        page = render_html([good_run, faulty_run])
        assert page.startswith("<!doctype html>") and "<script" not in page
        assert "</question>" not in page, "case text must be HTML-escaped"

    def test_write_reports(self, good_run, tmp_path):
        md, html_path = write_reports([good_run], tmp_path)
        assert md.read_text().startswith("# LLM Eval Report") and html_path.exists()
