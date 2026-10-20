import json

import pytest

from llm_eval_lab.judge import parse_verdict
from llm_eval_lab.llm import LLMError, OllamaClient
from llm_eval_lab.rag import (
    RESPONSE_SCHEMA,
    SYSTEM_CANARY,
    SYSTEM_PROMPT,
    build_messages,
    normalize_citation,
    parse_response,
)


class TestRagParsing:
    def test_valid_json(self):
        data, ok = parse_response('{"answer": "x", "citations": ["pricing"], "answerable": true}')
        assert ok and data["citations"] == ["pricing"]

    def test_json_wrapped_in_prose(self):
        _, ok = parse_response('Sure! {"answer": "x", "citations": [], "answerable": false} Done.')
        assert ok

    @pytest.mark.parametrize(
        "raw", ["not json", '{"answer": "x"}', '{"answer": 1, "citations": [], "answerable": true}']
    )
    def test_invalid_output_degrades_gracefully(self, raw):
        data, ok = parse_response(raw)
        assert not ok and data["answer"] == raw.strip() and data["citations"] == []

    @pytest.mark.parametrize("raw", ["pricing", "[pricing]", "Pricing.md", " pricing "])
    def test_normalize_citation(self, raw):
        assert normalize_citation(raw) == "pricing"

    def test_prompt_treats_question_as_data(self):
        messages = build_messages("What is X?", '<document id="a">ctx</document>')
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert SYSTEM_CANARY in SYSTEM_PROMPT
        assert "untrusted data" in SYSTEM_PROMPT
        assert "<question>\nWhat is X?\n</question>" in messages[1]["content"]


class TestJudgeParsing:
    def test_parses_and_clamps(self):
        v = parse_verdict(json.dumps({
            "correctness": 9, "groundedness": "4", "abstained": False,
            "followed_injection": False, "rationale": "ok",
        }))
        assert (v.correctness, v.groundedness, v.error) == (5, 4, None)

    def test_garbage_is_an_error_not_a_pass(self):
        v = parse_verdict("I think it's fine")
        assert v.correctness is None and v.error

    def test_missing_scores_flagged(self):
        v = parse_verdict('{"abstained": true, "rationale": "x"}')
        assert v.error == "judge output missing scores"


class TestOllamaClient:
    def test_payload_and_thinking_stripped(self, monkeypatch):
        sent = {}

        def stub_post(self, path, payload):
            sent.update(path=path, payload=payload)
            return {"message": {"content": '<think>reasoning</think>{"ok": true}'}}

        monkeypatch.setattr(OllamaClient, "_post", stub_post)
        client = OllamaClient("qwen3:8b", temperature=0.0, seed=7)
        out = client.chat([{"role": "user", "content": "hi"}], schema=RESPONSE_SCHEMA)

        assert out == '{"ok": true}'
        assert sent["path"] == "/api/chat"
        p = sent["payload"]
        assert p["model"] == "qwen3:8b" and p["stream"] is False and p["think"] is False
        assert p["format"] == RESPONSE_SCHEMA
        assert p["options"]["temperature"] == 0.0 and p["options"]["seed"] == 7

    def test_retries_then_raises(self, monkeypatch):
        calls = []

        def unreachable_post(self, path, payload):
            calls.append(1)
            raise OSError("connection refused")

        monkeypatch.setattr(OllamaClient, "_post", unreachable_post)
        monkeypatch.setattr("llm_eval_lab.llm.time.sleep", lambda s: None)
        with pytest.raises(LLMError):
            OllamaClient("qwen3:8b", retries=2).chat([])
        assert len(calls) == 3
