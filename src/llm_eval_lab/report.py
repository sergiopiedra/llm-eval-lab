"""Markdown and HTML reports comparing one or more eval runs.

The first run is treated as the baseline and the last run as the candidate; deltas and
per-case regressions are computed between those two.
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from pathlib import Path

from llm_eval_lab.metrics import METRICS

METRIC_ORDER = ["retrieval_recall", *METRICS]
LOWER_IS_BETTER = {"false_abstention_rate"}


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _gate_lookup(run: dict) -> dict[str, dict]:
    return {g["metric"]: g for g in run["gates"]}


def _gate_label(gate: dict | None) -> str:
    if gate is None:
        return "-"
    return f"{'>=' if gate['op'] == 'min' else '<='} {_pct(gate['threshold'])}"


def _delta(metric: str, base: float | None, cand: float | None) -> tuple[str, str]:
    """Return (text, tone) where tone is 'good', 'bad' or 'flat'."""
    if base is None or cand is None:
        return "n/a", "flat"
    diff = cand - base
    if abs(diff) < 1e-9:
        return "0.0 pp", "flat"
    better = diff < 0 if metric in LOWER_IS_BETTER else diff > 0
    return f"{diff * 100:+.1f} pp", "good" if better else "bad"


def case_changes(baseline: dict, candidate: dict) -> dict[str, list[dict]]:
    base = {c["id"]: c for c in baseline["cases"]}
    changes: dict[str, list[dict]] = {"regressions": [], "fixes": [], "still_failing": []}
    for case in candidate["cases"]:
        prev = base.get(case["id"])
        if prev is None:
            continue
        if prev["passed"] and not case["passed"]:
            changes["regressions"].append(case)
        elif not prev["passed"] and case["passed"]:
            changes["fixes"].append(case)
        elif not case["passed"]:
            changes["still_failing"].append(case)
    return changes


def _run_name(i: int, run: dict) -> str:
    return f"#{i + 1} {run['tag']}"


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# -- Markdown -----------------------------------------------------------------------------


def render_markdown(runs: list[dict]) -> str:
    baseline, candidate = runs[0], runs[-1]
    lines = [
        "# LLM Eval Report",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC}. "
        f"Baseline: **{baseline['tag']}** · Candidate: **{candidate['tag']}**.",
        "",
        "## Runs",
        "",
        "| Run | Model | Judge | Backend | Dataset | Cases passed | Gates |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, run in enumerate(runs):
        cfg, s = run["config"], run["summary"]
        status = "✅ PASS" if run["passed"] else "❌ FAIL"
        lines.append(
            f"| {_run_name(i, run)} (`{run['run_id']}`) | {cfg['model']} | {cfg['judge_model']} "
            f"| {cfg['backend']} | `{cfg['dataset_sha']}` | {s['cases_passed']}/{s['cases']} | {status} |"
        )
    if len({r["config"]["dataset_sha"] for r in runs}) > 1:
        lines += ["", "> ⚠️ Runs used different versions of the golden dataset; compare with care."]

    header = " | ".join(_run_name(i, r) for i, r in enumerate(runs))
    lines += [
        "",
        "## Quality gates",
        "",
        f"| Metric | Gate | {header} | Δ (candidate vs baseline) |",
        "|---|---|" + "---|" * len(runs) + "---|",
    ]
    for metric in METRIC_ORDER:
        gate = _gate_lookup(candidate).get(metric)
        cells = []
        for run in runs:
            g = _gate_lookup(run).get(metric)
            value = run["metrics"].get(metric, {}).get("value")
            mark = "" if g is None else (" ✅" if g["passed"] else " ❌")
            cells.append(f"{_pct(value)}{mark}")
        delta, _ = _delta(
            metric,
            baseline["metrics"].get(metric, {}).get("value"),
            candidate["metrics"].get(metric, {}).get("value"),
        )
        lines.append(f"| `{metric}` | {_gate_label(gate)} | {' | '.join(cells)} | {delta} |")

    if len(runs) > 1:
        changes = case_changes(baseline, candidate)
        lines += ["", "## Case changes (baseline → candidate)", ""]
        for key, title in (("regressions", "🔻 Regressions"), ("fixes", "🔺 Fixes")):
            lines.append(f"**{title}:** " + (", ".join(f"`{c['id']}`" for c in changes[key]) or "none"))
            lines.append("")

    failing = [c for c in candidate["cases"] if not c["passed"]]
    lines += ["", f"## Failing cases in candidate ({len(failing)})", ""]
    if failing:
        lines += ["| Case | Category | Failed checks | Answer | Judge rationale |", "|---|---|---|---|---|"]
        for c in failing:
            answer = _short(c["response"]["answer"]).replace("|", "\\|")
            rationale = _short(c["judge"]["rationale"] or c["judge"]["error"] or "").replace("|", "\\|")
            lines.append(
                f"| `{c['id']}` | {c['category']} | {', '.join(c['failure_reasons'])} | {answer} | {rationale} |"
            )
    else:
        lines.append("None. 🎉")
    return "\n".join(lines) + "\n"


# -- HTML ---------------------------------------------------------------------------------

_CSS = """
:root { --bg:#f7f7f8; --card:#fff; --fg:#1d1d1f; --muted:#6b6b76; --line:#e4e4e9;
        --good:#1a7f37; --good-bg:#dcf5e3; --bad:#c62828; --bad-bg:#fde2e2; --accent:#4f46e5; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#111114; --card:#1b1b20; --fg:#ececf1; --muted:#9a9aa6; --line:#2c2c34;
          --good:#4ade80; --good-bg:#12321f; --bad:#f87171; --bad-bg:#3a1414; --accent:#818cf8; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:32px 16px 64px; }
h1 { margin:0 0 4px; font-size:28px; } h2 { margin:36px 0 12px; font-size:19px; }
.sub { color:var(--muted); margin:0 0 20px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.card .k { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.card .v { font-size:22px; font-weight:650; margin-top:2px; }
.scroll { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:12px; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
tr:last-child td { border-bottom:0; }
code { font:12.5px ui-monospace, SFMono-Regular, Menlo, monospace; }
.pill { display:inline-block; padding:1px 8px; border-radius:999px; font-size:12px; font-weight:600; }
.pass { color:var(--good); background:var(--good-bg); } .fail { color:var(--bad); background:var(--bad-bg); }
.good { color:var(--good); } .bad { color:var(--bad); } .flat { color:var(--muted); }
.bar { height:4px; background:var(--line); border-radius:2px; margin-top:4px; position:relative; min-width:80px; }
.bar i { position:absolute; inset:0 auto 0 0; background:var(--accent); border-radius:2px; }
.bar b { position:absolute; top:-3px; width:2px; height:10px; background:var(--fg); opacity:.6; }
.desc { color:var(--muted); font-size:12px; }
.warn { background:var(--bad-bg); color:var(--bad); padding:10px 14px; border-radius:10px; margin-top:12px; }
"""


def _e(text) -> str:
    return html.escape(str(text))


def _metric_cell(run: dict, metric: str) -> str:
    value = run["metrics"].get(metric, {}).get("value")
    gate = _gate_lookup(run).get(metric)
    pill = ""
    if gate is not None:
        pill = f' <span class="pill {"pass" if gate["passed"] else "fail"}">{"pass" if gate["passed"] else "fail"}</span>'
    width = 0 if value is None else value * 100
    marker = f'<b style="left:{gate["threshold"] * 100:.1f}%"></b>' if gate else ""
    return f'<td>{_pct(value)}{pill}<div class="bar"><i style="width:{width:.1f}%"></i>{marker}</div></td>'


def render_html(runs: list[dict]) -> str:
    baseline, candidate = runs[0], runs[-1]
    s = candidate["summary"]
    gates_passed = sum(g["passed"] for g in candidate["gates"])
    status = "pass" if candidate["passed"] else "fail"
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>LLM Eval Report</title><style>{_CSS}</style></head><body><main>",
        "<h1>LLM Eval Report</h1>",
        f"<p class='sub'>Generated {datetime.now(UTC):%Y-%m-%d %H:%M UTC} · baseline "
        f"<b>{_e(baseline['tag'])}</b> → candidate <b>{_e(candidate['tag'])}</b></p>",
        "<div class='cards'>",
        f"<div class='card'><div class='k'>Candidate verdict</div><div class='v'>"
        f"<span class='pill {status}' style='font-size:16px'>{status.upper()}</span></div></div>",
        f"<div class='card'><div class='k'>Quality gates</div><div class='v'>{gates_passed}/{len(candidate['gates'])}</div></div>",
        f"<div class='card'><div class='k'>Cases passed</div><div class='v'>{s['cases_passed']}/{s['cases']}</div></div>",
        f"<div class='card'><div class='k'>Model · judge</div><div class='v' style='font-size:16px'>"
        f"{_e(candidate['config']['model'])} · {_e(candidate['config']['judge_model'])}</div></div>",
        "</div>",
    ]

    out += ["<h2>Runs</h2><div class='scroll'><table><tr><th>Run</th><th>Created</th><th>Model</th>"
            "<th>Judge</th><th>Backend</th><th>Dataset</th><th>Cases passed</th><th>p50 latency</th><th>Gates</th></tr>"]
    for i, run in enumerate(runs):
        cfg, rs = run["config"], run["summary"]
        latency = "n/a" if rs["latency_p50_s"] is None else f"{rs['latency_p50_s']:.2f}s"
        out.append(
            f"<tr><td><b>{_e(_run_name(i, run))}</b><br><code>{_e(run['run_id'])}</code></td>"
            f"<td>{_e(run['created_at'])}</td><td>{_e(cfg['model'])}</td><td>{_e(cfg['judge_model'])}</td>"
            f"<td>{_e(cfg['backend'])}</td><td><code>{_e(cfg['dataset_sha'])}</code></td>"
            f"<td>{rs['cases_passed']}/{rs['cases']}</td><td>{latency}</td>"
            f"<td><span class='pill {'pass' if run['passed'] else 'fail'}'>{'PASS' if run['passed'] else 'FAIL'}</span></td></tr>"
        )
    out.append("</table></div>")
    if len({r["config"]["dataset_sha"] for r in runs}) > 1:
        out.append("<div class='warn'>Runs used different versions of the golden dataset; compare with care.</div>")

    head = "".join(f"<th>{_e(_run_name(i, r))}</th>" for i, r in enumerate(runs))
    out += [f"<h2>Quality gates</h2><div class='scroll'><table><tr><th>Metric</th><th>Gate</th>{head}<th>Δ</th></tr>"]
    for metric in METRIC_ORDER:
        desc = candidate["metrics"].get(metric, {}).get("description", "")
        delta, tone = _delta(
            metric,
            baseline["metrics"].get(metric, {}).get("value"),
            candidate["metrics"].get(metric, {}).get("value"),
        )
        cells = "".join(_metric_cell(run, metric) for run in runs)
        out.append(
            f"<tr><td><code>{_e(metric)}</code><div class='desc'>{_e(desc)}</div></td>"
            f"<td>{_e(_gate_label(_gate_lookup(candidate).get(metric)))}</td>{cells}"
            f"<td class='{tone}'>{_e(delta)}</td></tr>"
        )
    out.append("</table></div>")

    if len(runs) > 1:
        changes = case_changes(baseline, candidate)
        out.append("<h2>Case changes (baseline → candidate)</h2><div class='cards'>")
        for key, title, tone in (("regressions", "Regressions", "bad"), ("fixes", "Fixes", "good")):
            ids = ", ".join(f"<code>{_e(c['id'])}</code>" for c in changes[key]) or "none"
            out.append(f"<div class='card'><div class='k {tone}'>{title} ({len(changes[key])})</div><div>{ids}</div></div>")
        out.append("</div>")

    failing = [c for c in candidate["cases"] if not c["passed"]]
    out.append(f"<h2>Failing cases in candidate ({len(failing)})</h2>")
    if failing:
        out.append("<div class='scroll'><table><tr><th>Case</th><th>Question</th><th>Answer</th>"
                   "<th>Failed checks</th><th>Judge</th></tr>")
        for c in failing:
            j = c["judge"]
            scores = f"C{j['correctness']} · G{j['groundedness']}" if j["correctness"] else "error"
            reasons = " ".join(f"<span class='pill fail'>{_e(r)}</span>" for r in c["failure_reasons"])
            out.append(
                f"<tr><td><code>{_e(c['id'])}</code><br><span class='desc'>{_e(c['category'])}</span></td>"
                f"<td>{_e(_short(c['question'], 200))}</td><td>{_e(_short(c['response']['answer'], 240))}</td>"
                f"<td>{reasons}</td>"
                f"<td><b>{_e(scores)}</b><div class='desc'>{_e(_short(j['rationale'] or j['error'] or '', 220))}</div></td></tr>"
            )
        out.append("</table></div>")
    else:
        out.append("<p>None. Every case passed.</p>")
    out.append("</main></body></html>")
    return "\n".join(out)


def write_reports(runs: list[dict], out_dir: Path | str) -> tuple[Path, Path]:
    if not runs:
        raise ValueError("need at least one run to report on")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, html_path = out_dir / "report.md", out_dir / "report.html"
    md_path.write_text(render_markdown(runs))
    html_path.write_text(render_html(runs))
    return md_path, html_path
