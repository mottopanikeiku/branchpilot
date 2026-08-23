from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_FAMILY_COLOR = {"offline-rl": "#a78bfa", "fixed": "#22d3ee", "heuristic": "#fb923c"}


def _unique_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (row["policy"], float(row["accuracy"]), float(row["average_samples"]))
        points[key] = row
    return list(points.values())


def render_svg(payload: dict[str, Any], title: str = "Accuracy × inference compute") -> str:
    points = _unique_points(payload["rows"])
    width, height = 1000, 560
    left, right, top, bottom = 92, 34, 72, 76
    plot_width, plot_height = width - left - right, height - top - bottom
    max_samples = max(float(point["average_samples"]) for point in points)
    min_accuracy = min(float(point["accuracy"]) for point in points)
    max_accuracy = max(float(point["accuracy"]) for point in points)
    y_floor = max(0.0, min_accuracy - 0.025)
    y_ceiling = min(1.0, max_accuracy + 0.025)
    if y_ceiling - y_floor < 0.05:
        y_floor = max(0.0, y_floor - 0.025)
        y_ceiling = min(1.0, y_ceiling + 0.025)

    def x(value: float) -> float:
        return left + (value - 1.0) / max(1e-6, max_samples - 1.0) * plot_width

    def y(value: float) -> float:
        return top + (y_ceiling - value) / max(1e-6, y_ceiling - y_floor) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<defs><filter id=\"glow\"><feGaussianBlur stdDeviation=\"3\" result=\"blur\"/><feMerge><feMergeNode in=\"blur\"/><feMergeNode in=\"SourceGraphic\"/></feMerge></filter></defs>",
        '<rect width="100%" height="100%" rx="20" fill="#080b14"/>',
        f'<text x="{left}" y="38" fill="#f8fafc" font-size="24" font-family="ui-monospace, monospace" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left}" y="58" fill="#94a3b8" font-size="12" font-family="ui-monospace, monospace">left and up is better · held-out rollouts</text>',
    ]
    for tick in range(6):
        value = y_floor + (y_ceiling - y_floor) * tick / 5
        py = y(value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#1e293b" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{py+4:.1f}" text-anchor="end" fill="#64748b" font-size="11" font-family="ui-monospace, monospace">{value:.1%}</text>'
        )
    for tick in range(1, int(max_samples) + 1):
        px = x(float(tick))
        parts.append(
            f'<text x="{px:.1f}" y="{height-bottom+28}" text-anchor="middle" fill="#64748b" font-size="11" font-family="ui-monospace, monospace">{tick}</text>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_width/2:.1f}" y="{height-20}" text-anchor="middle" fill="#94a3b8" font-size="13" font-family="ui-monospace, monospace">average samples / prompt</text>',
            f'<text x="22" y="{top + plot_height/2:.1f}" transform="rotate(-90 22 {top + plot_height/2:.1f})" text-anchor="middle" fill="#94a3b8" font-size="13" font-family="ui-monospace, monospace">exact-match accuracy</text>',
        ]
    )

    for point in sorted(points, key=lambda row: row["family"] != "offline-rl"):
        family = str(point["family"])
        color = _FAMILY_COLOR.get(family, "#e2e8f0")
        px = x(float(point["average_samples"]))
        py = y(float(point["accuracy"]))
        radius = 7 if family == "offline-rl" else 4
        opacity = 1.0 if family == "offline-rl" else 0.65
        glow = ' filter="url(#glow)"' if family == "offline-rl" else ""
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" opacity="{opacity}"{glow}><title>{html.escape(str(point["policy"]))}: {float(point["accuracy"]):.1%} at {float(point["average_samples"]):.2f} samples</title></circle>'
        )

    legend_x = width - 330
    for index, (family, color) in enumerate(_FAMILY_COLOR.items()):
        px = legend_x + index * 105
        parts.append(f'<circle cx="{px}" cy="39" r="5" fill="{color}"/>')
        parts.append(
            f'<text x="{px+10}" y="43" fill="#cbd5e1" font-size="11" font-family="ui-monospace, monospace">{family}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def write_report(
    benchmark_path: str | Path,
    svg_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> None:
    payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    svg = render_svg(payload)
    if svg_path is not None:
        destination = Path(svg_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(svg, encoding="utf-8")
    if html_path is not None:
        best_rl = max(
            (row for row in payload["rows"] if row["family"] == "offline-rl"),
            key=lambda row: (row["accuracy"], -row["average_samples"]),
        )
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>BranchPilot benchmark</title><style>
:root{{color-scheme:dark;font-family:Inter,ui-sans-serif,system-ui;background:#05070d;color:#e2e8f0}}
body{{margin:0}} main{{max-width:1100px;margin:auto;padding:64px 24px}} .eyebrow{{color:#a78bfa;font:600 13px ui-monospace;letter-spacing:.16em;text-transform:uppercase}}
h1{{font-size:clamp(44px,8vw,88px);line-height:.94;letter-spacing:-.06em;margin:18px 0}} .lede{{font-size:20px;color:#94a3b8;max-width:760px;line-height:1.55}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:42px 0}} .card{{padding:22px;border:1px solid #1e293b;border-radius:16px;background:#0b1020}} .value{{font:700 30px ui-monospace;color:#f8fafc}} .label{{color:#64748b;margin-top:7px}} img{{width:100%;height:auto;border:1px solid #1e293b;border-radius:20px}}
code{{color:#22d3ee}} @media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">offline reinforcement learning × llm systems</div>
<h1>Know when to<br>stop thinking.</h1><p class="lede">BranchPilot learns whether another reasoning sample is worth its GPU time. One cost-conditioned Q-network controls the entire accuracy–compute frontier.</p>
<div class="grid"><div class="card"><div class="value">{float(best_rl['accuracy']):.1%}</div><div class="label">best held-out accuracy</div></div><div class="card"><div class="value">{float(best_rl['average_samples']):.2f}</div><div class="label">samples per prompt</div></div><div class="card"><div class="value">{int(payload['records'])}</div><div class="label">held-out trajectories</div></div></div>
{svg}<p class="lede">Reproduce with <code>modal run modal_app.py</code>, then train and evaluate locally with <code>branchpilot</code>.</p></main></body></html>"""
        destination = Path(html_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8")
