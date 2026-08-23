from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_PALETTE = {
    "canvas": "#080a0f",
    "surface": "#0d1119",
    "surface-raised": "#121824",
    "rule": "#273142",
    "rule-strong": "#3a465a",
    "text": "#edf1f7",
    "muted": "#a7b1c2",
    "subtle": "#778397",
    "learned": "#c4b5fd",
    "learned-strong": "#e2dcff",
    "fixed": "#74aef2",
    "heuristic": "#e8b15a",
}
_FAMILY_META = {
    "offline-rl": ("BranchPilot · learned", _PALETTE["learned"]),
    "fixed": ("Fixed-sample baseline", _PALETTE["fixed"]),
    "heuristic": ("Heuristic baseline", _PALETTE["heuristic"]),
}
_FAMILY_COLOR = {family: meta[1] for family, meta in _FAMILY_META.items()}


def _unique_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: dict[tuple[str, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (row["policy"], float(row["accuracy"]), float(row["average_samples"]))
        points[key] = row
    return list(points.values())


def render_svg(payload: dict[str, Any], title: str = "Accuracy × inference compute") -> str:
    points = _unique_points(payload["rows"])
    width, height = 1120, 680
    left, right, top, bottom = 94, 54, 132, 82
    plot_width, plot_height = width - left - right, height - top - bottom

    min_samples = min(float(point["average_samples"]) for point in points)
    max_samples = max(float(point["average_samples"]) for point in points)
    x_floor = min(1.0, min_samples)
    x_ceiling = float(max(1, int(max_samples) + (max_samples % 1 > 1e-9)))
    if x_ceiling - x_floor < 1e-6:
        x_floor = max(0.0, x_floor - 0.5)
        x_ceiling += 0.5

    min_accuracy = min(float(point["accuracy"]) for point in points)
    max_accuracy = max(float(point["accuracy"]) for point in points)
    accuracy_span = max(0.06, max_accuracy - min_accuracy)
    y_floor = max(0.0, min_accuracy - accuracy_span * 0.14)
    y_ceiling = min(1.0, max_accuracy + accuracy_span * 0.14)
    if y_ceiling - y_floor < 0.06:
        midpoint = (y_ceiling + y_floor) / 2
        y_floor = max(0.0, midpoint - 0.03)
        y_ceiling = min(1.0, midpoint + 0.03)

    def x(value: float) -> float:
        return left + (value - x_floor) / (x_ceiling - x_floor) * plot_width

    def y(value: float) -> float:
        return top + (y_ceiling - value) / (y_ceiling - y_floor) * plot_height

    max_sample_tick = int(x_ceiling)
    if max_sample_tick <= 9:
        sample_ticks = list(range(max(1, int(x_floor)), max_sample_tick + 1))
    else:
        tick_step = max(1, (max_sample_tick - 1 + 5) // 6)
        sample_ticks = list(range(1, max_sample_tick + 1, tick_step))
        if sample_ticks[-1] != max_sample_tick:
            sample_ticks.append(max_sample_tick)

    escaped_title = html.escape(title)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
            'role="img" aria-labelledby="branchpilot-chart-title branchpilot-chart-desc">'
        ),
        f'<title id="branchpilot-chart-title">{escaped_title}</title>',
        (
            '<desc id="branchpilot-chart-desc">'
            f'Scatter plot of {len(points)} unique policies. Large violet diamonds show '
            'BranchPilot learned policies; blue squares and amber circles show baselines.'
            "</desc>"
        ),
        (
            '<defs><clipPath id="branchpilot-plot-clip">'
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>'
            "</clipPath><style>"
            'text{font-family:"Aptos","Helvetica Neue",sans-serif}'
            '.chart-title{font-family:"Charter","Georgia",serif;font-size:28px;font-weight:700;'
            f'fill:{_PALETTE["text"]};letter-spacing:-.3px}}'
            '.chart-kicker,.tick,.policy-label,.learned-label{'
            'font-family:"IBM Plex Mono","SFMono-Regular","Cascadia Mono",monospace}'
            f'.chart-kicker{{font-size:11px;font-weight:700;fill:{_PALETTE["learned"]};'
            'letter-spacing:1.7px}'
            f'.subtitle{{font-size:13px;fill:{_PALETTE["muted"]}}}'
            f'.legend{{font-size:12px;font-weight:600;fill:{_PALETTE["muted"]}}}'
            f'.tick{{font-size:11px;fill:{_PALETTE["subtle"]}}}'
            f'.axis-label{{font-size:13px;font-weight:600;fill:{_PALETTE["muted"]}}}'
            f'.plot-note{{font-size:11px;fill:{_PALETTE["subtle"]}}}'
            f'.grid{{stroke:{_PALETTE["rule"]};stroke-width:1}}'
            f'.grid-vertical{{stroke:{_PALETTE["rule"]};stroke-width:1;stroke-dasharray:3 7}}'
            f'.learned-line{{fill:none;stroke:{_PALETTE["learned"]};stroke-width:2.5;'
            'stroke-opacity:.55;stroke-linejoin:round;stroke-linecap:round}'
            f'.learned-halo{{fill:{_PALETTE["learned"]};fill-opacity:.12}}'
            f'.learned-point{{fill:{_PALETTE["learned"]};stroke:{_PALETTE["canvas"]};'
            'stroke-width:3}'
            '.policy-label,.learned-label{paint-order:stroke;'
            f'stroke:{_PALETTE["surface"]};stroke-width:4px;stroke-linejoin:round}}'
            f'.policy-label{{font-size:10px;fill:{_PALETTE["muted"]}}}'
            f'.learned-label{{font-size:10px;font-weight:700;fill:{_PALETTE["learned-strong"]}}}'
            "</style></defs>"
        ),
        f'<rect width="{width}" height="{height}" rx="12" fill="{_PALETTE["canvas"]}"/>',
        (
            f'<rect x="1" y="1" width="{width-2}" height="{height-2}" rx="11" '
            f'fill="none" stroke="{_PALETTE["rule"]}"/>'
        ),
        f'<text x="{left}" y="35" class="chart-kicker">HELD-OUT POLICY EVALUATION</text>',
        f'<text x="{left}" y="67" class="chart-title">{escaped_title}</text>',
        (
            f'<text x="{left}" y="91" class="subtitle">Unique operating points · '
            f'{int(payload["records"]):,} held-out trajectories</text>'
        ),
        (
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
            f'rx="6" fill="{_PALETTE["surface"]}" stroke="{_PALETTE["rule"]}"/>'
        ),
    ]

    legend_y = 112
    legend_positions = (left, left + 245, left + 480)
    for index, (family, (label, color)) in enumerate(_FAMILY_META.items()):
        marker_x = legend_positions[index]
        if family == "offline-rl":
            parts.append(
                f'<path d="M {marker_x} {legend_y-7} L {marker_x+7} {legend_y} '
                f'L {marker_x} {legend_y+7} L {marker_x-7} {legend_y} Z" fill="{color}"/>'
            )
        elif family == "fixed":
            parts.append(
                f'<rect x="{marker_x-5}" y="{legend_y-5}" width="10" height="10" '
                f'rx="2" fill="{color}"/>'
            )
        else:
            parts.append(f'<circle cx="{marker_x}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{marker_x+13}" y="{legend_y+4}" class="legend">{label}</text>')

    for tick in range(6):
        value = y_floor + (y_ceiling - y_floor) * tick / 5
        py = y(value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left-13}" y="{py+4:.1f}" text-anchor="end" class="tick">'
            f"{value:.1%}</text>"
        )

    for tick in sample_ticks:
        px = x(float(tick))
        parts.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height-bottom}" '
            'class="grid-vertical"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{height-bottom+27}" text-anchor="middle" class="tick">'
            f"{tick}</text>"
        )

    parts.extend(
        [
            (
                f'<text x="{left + plot_width/2:.1f}" y="{height-24}" text-anchor="middle" '
                'class="axis-label">Average samples per prompt</text>'
            ),
            (
                f'<text x="25" y="{top + plot_height/2:.1f}" '
                f'transform="rotate(-90 25 {top + plot_height/2:.1f})" '
                'text-anchor="middle" class="axis-label">Exact-match accuracy</text>'
            ),
            (
                f'<text x="{width-right-14}" y="{top+22}" text-anchor="end" class="plot-note">'
                "higher accuracy ↑ · lower compute ←</text>"
            ),
        ]
    )

    baselines = [point for point in points if point["family"] != "offline-rl"]
    fixed_points = [point for point in baselines if point["family"] == "fixed"]
    heuristic_points = [point for point in baselines if point["family"] == "heuristic"]
    other_points = [
        point for point in baselines if point["family"] not in {"fixed", "heuristic"}
    ]

    for point in fixed_points + heuristic_points + other_points:
        family = str(point["family"])
        color = _FAMILY_COLOR.get(family, _PALETTE["text"])
        px = x(float(point["average_samples"]))
        py = y(float(point["accuracy"]))
        tooltip = html.escape(
            f'{point["policy"]}: {float(point["accuracy"]):.1%} accuracy at '
            f'{float(point["average_samples"]):.2f} samples'
        )
        parts.append(f'<g><title>{tooltip}</title>')
        if family == "fixed":
            parts.append(
                f'<rect x="{px-5:.1f}" y="{py-5:.1f}" width="10" height="10" rx="2" '
                f'fill="{color}" stroke="{_PALETTE["surface"]}" stroke-width="2"/>'
            )
        else:
            parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" '
                f'stroke="{_PALETTE["surface"]}" stroke-width="2"/>'
            )
        parts.append("</g>")

    for index, point in enumerate(sorted(fixed_points, key=lambda item: item["average_samples"])):
        px = x(float(point["average_samples"]))
        py = y(float(point["accuracy"]))
        near_right_edge = px > width - right - 62
        label_x = px - 7 if near_right_edge else px
        anchor = "end" if near_right_edge else "middle"
        label_y = py - 12 if index % 2 == 0 else py + 19
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" '
            f'class="policy-label">{html.escape(str(point["policy"]))}</text>'
        )

    heuristic_geometry = [
        (point, x(float(point["average_samples"])), y(float(point["accuracy"])))
        for point in heuristic_points
    ]
    heuristic_geometry.sort(key=lambda item: item[2])
    label_positions: list[float] = []
    for _, _, py in heuristic_geometry:
        label_positions.append(max(py + 3, label_positions[-1] + 17 if label_positions else py + 3))
    if label_positions and label_positions[-1] > height - bottom - 8:
        shift = label_positions[-1] - (height - bottom - 8)
        label_positions = [position - shift for position in label_positions]
    for (point, px, py), label_y in zip(heuristic_geometry, label_positions, strict=True):
        place_left = px > left + plot_width * 0.68
        label_x = px - 15 if place_left else px + 15
        anchor = "end" if place_left else "start"
        line_end = label_x + 4 if place_left else label_x - 4
        parts.append(
            f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{line_end:.1f}" y2="{label_y-3:.1f}" '
            f'stroke="{_PALETTE["heuristic"]}" stroke-opacity=".45"/>'
        )
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" '
            f'class="policy-label">{html.escape(str(point["policy"]))}</text>'
        )

    learned_points = sorted(
        (point for point in points if point["family"] == "offline-rl"),
        key=lambda item: item["average_samples"],
    )
    if len(learned_points) > 1:
        path = " ".join(
            f'{"M" if index == 0 else "L"} {x(float(point["average_samples"])):.1f} '
            f'{y(float(point["accuracy"])):.1f}'
            for index, point in enumerate(learned_points)
        )
        parts.append(
            f'<path d="{path}" class="learned-line" clip-path="url(#branchpilot-plot-clip)"/>'
        )

    for index, point in enumerate(learned_points):
        px = x(float(point["average_samples"]))
        py = y(float(point["accuracy"]))
        tooltip = html.escape(
            f'{point["policy"]}: {float(point["accuracy"]):.1%} accuracy at '
            f'{float(point["average_samples"]):.2f} samples'
        )
        parts.append(
            f'<g><title>{tooltip}</title><circle cx="{px:.1f}" cy="{py:.1f}" r="13" '
            'class="learned-halo"/>'
        )
        parts.append(
            f'<path d="M {px:.1f} {py-8:.1f} L {px+8:.1f} {py:.1f} '
            f'L {px:.1f} {py+8:.1f} L {px-8:.1f} {py:.1f} Z" class="learned-point"/></g>'
        )
        policy_label = str(point["policy"]).replace("BranchPilot ", "")
        label_y = py - 17 if index % 2 == 0 else py + 27
        label_y = min(height - bottom - 8, max(top + 16, label_y))
        parts.append(
            f'<text x="{px:.1f}" y="{label_y:.1f}" text-anchor="middle" '
            f'class="learned-label">{html.escape(policy_label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _family_label(family: str) -> str:
    return {
        "offline-rl": "Learned",
        "fixed": "Fixed",
        "heuristic": "Heuristic",
    }.get(family, family.replace("-", " ").title())


def _render_table_rows(rows: list[dict[str, Any]]) -> str:
    rendered = []
    for row in rows:
        family = str(row["family"])
        family_class = family if family in _FAMILY_META else "other"
        rendered.append(
            "<tr>"
            f'<td><span class="family-key family-{family_class}">'
            f'<span aria-hidden="true"></span>{html.escape(_family_label(family))}</span></td>'
            f'<th scope="row">{html.escape(str(row["policy"]))}</th>'
            f'<td class="numeric">{float(row["scoring_cost"]):.3g}</td>'
            f'<td class="numeric">{float(row["accuracy"]):.1%}</td>'
            f'<td class="numeric">{float(row["average_samples"]):.2f}</td>'
            f'<td class="numeric">{float(row["utility"]):.3f}</td>'
            "</tr>"
        )
    return "".join(rendered)


def _report_styles() -> str:
    return f"""
:root {{
  color-scheme: dark;
  --canvas: {_PALETTE["canvas"]};
  --surface: {_PALETTE["surface"]};
  --surface-raised: {_PALETTE["surface-raised"]};
  --rule: {_PALETTE["rule"]};
  --rule-strong: {_PALETTE["rule-strong"]};
  --text: {_PALETTE["text"]};
  --muted: {_PALETTE["muted"]};
  --subtle: {_PALETTE["subtle"]};
  --learned: {_PALETTE["learned"]};
  --learned-strong: {_PALETTE["learned-strong"]};
  --fixed: {_PALETTE["fixed"]};
  --heuristic: {_PALETTE["heuristic"]};
  --space-1: .25rem;
  --space-2: .5rem;
  --space-3: .75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-7: 3rem;
  --space-8: 4rem;
  --space-9: 6rem;
  --radius-sm: .375rem;
  --radius-md: .75rem;
  --shadow: 0 2rem 5rem rgb(2 5 11 / .42);
  font-family: "Aptos", "Helvetica Neue", sans-serif;
  background: var(--canvas);
  color: var(--text);
}}
* {{ box-sizing: border-box; }}
html {{ background: var(--canvas); }}
body {{
  margin: 0;
  min-width: 20rem;
  background: var(--canvas);
  color: var(--text);
  line-height: 1.5;
  text-rendering: optimizeLegibility;
}}
.skip-link {{
  position: fixed;
  z-index: 10;
  top: var(--space-3);
  left: var(--space-3);
  padding: var(--space-2) var(--space-4);
  transform: translateY(-200%);
  border-radius: var(--radius-sm);
  background: var(--text);
  color: var(--canvas);
  font-weight: 700;
}}
.skip-link:focus {{ transform: translateY(0); }}
.page-shell {{
  width: min(calc(100% - 2rem), 76rem);
  margin-inline: auto;
}}
.hero {{
  padding-block: var(--space-7) var(--space-8);
  border-bottom: 1px solid var(--rule);
}}
.topline {{
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
  padding-bottom: var(--space-6);
  border-bottom: 1px solid var(--rule);
  color: var(--subtle);
  font: 700 .6875rem/1.3 "IBM Plex Mono", "SFMono-Regular", "Cascadia Mono", monospace;
  letter-spacing: .12em;
  text-transform: uppercase;
}}
.topline span:last-child {{ text-align: right; }}
.kicker {{
  margin: var(--space-8) 0 var(--space-4);
  color: var(--learned);
  font: 700 .75rem/1.3 "IBM Plex Mono", "SFMono-Regular", "Cascadia Mono", monospace;
  letter-spacing: .16em;
  text-transform: uppercase;
}}
h1, h2, p {{ margin-top: 0; }}
h1 {{
  max-width: 58rem;
  margin-bottom: var(--space-5);
  font-family: "Charter", "Georgia", serif;
  font-size: clamp(3.5rem, 8vw, 6rem);
  font-weight: 700;
  letter-spacing: -.05em;
  line-height: .94;
}}
h1 span {{ color: var(--learned-strong); }}
.lede {{
  max-width: 48rem;
  margin-bottom: 0;
  color: var(--muted);
  font-size: clamp(1.0625rem, 2vw, 1.25rem);
  line-height: 1.65;
}}
main {{ display: block; }}
.summary-panel {{
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(17rem, .65fr);
  gap: var(--space-7);
  padding-block: var(--space-7);
  border-bottom: 1px solid var(--rule);
}}
.measure-label {{
  display: block;
  margin-bottom: var(--space-3);
  color: var(--muted);
  font-size: .8125rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}}
.measure {{
  display: block;
  margin-bottom: var(--space-4);
  font: 700 clamp(3.5rem, 8vw, 5rem)/.92 "IBM Plex Mono", "SFMono-Regular", monospace;
  letter-spacing: -.06em;
}}
.measure-note {{
  max-width: 39rem;
  margin-bottom: 0;
  color: var(--muted);
  font-size: .9375rem;
}}
.measure-note strong {{ color: var(--learned-strong); font-weight: 700; }}
.summary-list {{
  margin: 0;
  border-left: 1px solid var(--rule);
}}
.summary-list div {{
  display: grid;
  grid-template-columns: 1fr auto;
  gap: var(--space-4);
  padding: var(--space-4) 0 var(--space-4) var(--space-6);
  border-bottom: 1px solid var(--rule);
}}
.summary-list div:first-child {{ padding-top: 0; }}
.summary-list div:last-child {{ padding-bottom: 0; border-bottom: 0; }}
.summary-list dt {{ color: var(--subtle); font-size: .8125rem; }}
.summary-list dd {{
  margin: 0;
  color: var(--text);
  font: 700 .875rem/1.5 "IBM Plex Mono", "SFMono-Regular", monospace;
  text-align: right;
}}
.report-section {{ padding-block: var(--space-9); border-bottom: 1px solid var(--rule); }}
.section-heading {{
  display: grid;
  grid-template-columns: minmax(15rem, .72fr) minmax(18rem, 1fr);
  gap: var(--space-7);
  align-items: end;
  margin-bottom: var(--space-7);
}}
.section-index {{
  display: block;
  margin-bottom: var(--space-3);
  color: var(--learned);
  font: 700 .6875rem/1.3 "IBM Plex Mono", "SFMono-Regular", monospace;
  letter-spacing: .14em;
}}
h2 {{
  margin-bottom: 0;
  font-family: "Charter", "Georgia", serif;
  font-size: clamp(2rem, 4vw, 3rem);
  letter-spacing: -.035em;
  line-height: 1.05;
}}
.section-heading p {{
  margin-bottom: 0;
  color: var(--muted);
  font-size: .9375rem;
  line-height: 1.7;
}}
figure {{ margin: 0; }}
.figure-shell {{
  overflow: hidden;
  border: 1px solid var(--rule);
  border-radius: var(--radius-md);
  background: var(--surface);
  box-shadow: var(--shadow);
}}
.chart-viewport {{ overflow-x: auto; }}
.chart-viewport svg {{ display: block; width: 100%; height: auto; }}
figcaption {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-6);
  padding: var(--space-5) var(--space-6);
  border-top: 1px solid var(--rule);
  color: var(--subtle);
  font-size: .75rem;
  line-height: 1.65;
}}
figcaption strong {{ color: var(--muted); font-weight: 700; }}
.method-strip {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  margin-top: var(--space-7);
  border-block: 1px solid var(--rule);
}}
.method-strip article {{
  padding: var(--space-5) var(--space-6);
  border-right: 1px solid var(--rule);
}}
.method-strip article:first-child {{ padding-left: 0; }}
.method-strip article:last-child {{ padding-right: 0; border-right: 0; }}
.method-strip h3 {{
  margin: 0 0 var(--space-2);
  color: var(--text);
  font-size: .8125rem;
}}
.method-strip p {{
  margin-bottom: 0;
  color: var(--subtle);
  font-size: .75rem;
  line-height: 1.6;
}}
.formula {{
  color: var(--learned-strong);
  font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
}}
.table-shell {{
  overflow-x: auto;
  border-top: 1px solid var(--rule-strong);
  border-bottom: 1px solid var(--rule);
}}
table {{
  width: 100%;
  min-width: 47rem;
  border-collapse: collapse;
  font-size: .8125rem;
}}
th, td {{
  padding: var(--space-4) var(--space-3);
  border-bottom: 1px solid var(--rule);
  text-align: left;
  white-space: nowrap;
}}
thead th {{
  color: var(--subtle);
  font: 700 .6875rem/1.3 "IBM Plex Mono", "SFMono-Regular", monospace;
  letter-spacing: .08em;
  text-transform: uppercase;
}}
tbody th {{ color: var(--text); font-weight: 600; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
tbody tr:hover {{ background: var(--surface-raised); }}
.numeric {{
  font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
  font-variant-numeric: tabular-nums;
  text-align: right;
}}
.family-key {{
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  color: var(--muted);
}}
.family-key span {{ width: .5rem; height: .5rem; border-radius: 50%; background: var(--muted); }}
.family-offline-rl span {{ background: var(--learned); transform: rotate(45deg); border-radius: .0625rem; }}
.family-fixed span {{ background: var(--fixed); border-radius: .125rem; }}
.family-heuristic span {{ background: var(--heuristic); }}
details {{ margin-top: var(--space-7); border-block: 1px solid var(--rule); }}
summary {{
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
  padding-block: var(--space-5);
  color: var(--muted);
  cursor: pointer;
  font-size: .875rem;
  font-weight: 700;
  list-style: none;
}}
summary::-webkit-details-marker {{ display: none; }}
summary::after {{
  content: "+";
  color: var(--learned);
  font: 700 1rem/1 "IBM Plex Mono", "SFMono-Regular", monospace;
}}
details[open] summary::after {{ content: "−"; }}
summary:focus-visible {{ outline: 2px solid var(--learned); outline-offset: var(--space-1); }}
details .table-shell {{ margin-bottom: var(--space-5); }}
.report-footer {{
  display: flex;
  justify-content: space-between;
  gap: var(--space-5);
  padding-block: var(--space-6) var(--space-8);
  color: var(--subtle);
  font: 600 .6875rem/1.6 "IBM Plex Mono", "SFMono-Regular", monospace;
  letter-spacing: .04em;
}}
@media (max-width: 48rem) {{
  .hero {{ padding-top: var(--space-5); }}
  .topline {{ flex-direction: column; }}
  .topline span:last-child {{ text-align: left; }}
  .kicker {{ margin-top: var(--space-7); }}
  .summary-panel, .section-heading {{ grid-template-columns: 1fr; }}
  .summary-list {{ border-left: 0; border-top: 1px solid var(--rule); padding-top: var(--space-5); }}
  .summary-list div {{ padding-left: 0; }}
  .report-section {{ padding-block: var(--space-8); }}
  .chart-viewport svg {{ width: 48rem; max-width: none; }}
  figcaption, .method-strip {{ grid-template-columns: 1fr; }}
  figcaption {{ gap: var(--space-3); }}
  .method-strip article {{
    padding: var(--space-5) 0;
    border-right: 0;
    border-bottom: 1px solid var(--rule);
  }}
  .method-strip article:last-child {{ border-bottom: 0; }}
  .report-footer {{ flex-direction: column; }}
}}
@media (prefers-reduced-motion: no-preference) {{
  .skip-link {{ transition: transform 160ms ease-out; }}
}}
"""


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
        points = _unique_points(payload["rows"])
        learned_points = sorted(
            (row for row in points if row["family"] == "offline-rl"),
            key=lambda row: row["average_samples"],
        )
        best_rl = max(
            learned_points,
            key=lambda row: (row["accuracy"], -row["average_samples"]),
        )
        best_baseline = max(
            (row for row in points if row["family"] != "offline-rl"),
            key=lambda row: (row["accuracy"], -row["average_samples"]),
        )
        accuracy_delta = (
            float(best_rl["accuracy"]) - float(best_baseline["accuracy"])
        ) * 100
        sample_delta = 1 - (
            float(best_rl["average_samples"]) / float(best_baseline["average_samples"])
        )
        if sample_delta >= 0:
            sample_comparison = f"{sample_delta:.1%} fewer samples"
        else:
            sample_comparison = f"{-sample_delta:.1%} more samples"

        learned_table_rows = _render_table_rows(learned_points)
        all_rows = sorted(
            payload["rows"],
            key=lambda row: (
                float(row["scoring_cost"]),
                {"offline-rl": 0, "fixed": 1, "heuristic": 2}.get(row["family"], 3),
                float(row["average_samples"]),
                str(row["policy"]),
            ),
        )
        all_table_rows = _render_table_rows(all_rows)
        schema_version = html.escape(str(payload.get("schema_version", "—")))
        document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>BranchPilot · held-out policy evaluation</title>
<style>{_report_styles()}</style>
</head>
<body>
<a class="skip-link" href="#report-content">Skip to results</a>
<div class="page-shell">
<header class="hero">
  <div class="topline">
    <span>BranchPilot / Research evaluation</span>
    <span>Standalone offline artifact</span>
  </div>
  <p class="kicker">Cost-conditioned stopping policy</p>
  <h1>Adaptive inference,<br><span>measured.</span></h1>
  <p class="lede">BranchPilot learns when another language-model sample is worth its inference cost. This report maps the learned policy sweep against fixed-budget and rule-based baselines on held-out trajectories.</p>
</header>
<main id="report-content">
  <section class="summary-panel" aria-labelledby="summary-title">
    <div>
      <span class="measure-label" id="summary-title">Peak learned accuracy</span>
      <strong class="measure">{float(best_rl["accuracy"]):.1%}</strong>
      <p class="measure-note"><strong>{accuracy_delta:+.1f} percentage points</strong> against the strongest baseline, with {sample_comparison} at the respective highest-accuracy operating points.</p>
    </div>
    <dl class="summary-list">
      <div><dt>Compute at peak</dt><dd>{float(best_rl["average_samples"]):.2f} samples</dd></div>
      <div><dt>Strongest baseline</dt><dd>{float(best_baseline["accuracy"]):.1%} / {float(best_baseline["average_samples"]):.2f}</dd></div>
      <div><dt>Held-out set</dt><dd>{int(payload["records"]):,} trajectories</dd></div>
    </dl>
  </section>

  <section class="report-section" aria-labelledby="frontier-title">
    <div class="section-heading">
      <div>
        <span class="section-index">01 / FRONTIER</span>
        <h2 id="frontier-title">Accuracy–compute plane</h2>
      </div>
      <p>Each marker is one unique operating policy. Learned policies are large violet diamonds connected by their cost sweep; every fixed and heuristic baseline remains directly identified in the plot.</p>
    </div>
    <figure>
      <div class="figure-shell">
        <div class="chart-viewport">{svg}</div>
        <figcaption>
          <span><strong>Figure 1.</strong> Exact-match accuracy against average samples per prompt. Points toward the upper-left dominate.</span>
          <span><strong>Reading the sweep.</strong> Diamond labels report λ, the scoring-cost condition. The connecting line shows policy progression, not interpolation.</span>
        </figcaption>
      </div>
    </figure>
    <div class="method-strip" aria-label="Evaluation method">
      <article>
        <h3>Objective</h3>
        <p><span class="formula">utility = accuracy − λ × samples</span>. Increasing λ rewards earlier stopping.</p>
      </article>
      <article>
        <h3>Learned policy</h3>
        <p>A cost-conditioned stop/continue Q-policy evaluated over pre-generated self-consistency trajectories.</p>
      </article>
      <article>
        <h3>Reference policies</h3>
        <p>Fixed sample budgets and confidence/agreement heuristics establish the non-learned envelope.</p>
      </article>
    </div>
  </section>

  <section class="report-section" aria-labelledby="sweep-title">
    <div class="section-heading">
      <div>
        <span class="section-index">02 / POLICY SWEEP</span>
        <h2 id="sweep-title">Learned operating points</h2>
      </div>
      <p>The learned family is ordered by average inference compute. Utility is reported at the policy’s scoring-cost condition, preserving the benchmark’s cost-sensitive comparison.</p>
    </div>
    <div class="table-shell">
      <table aria-label="Learned policy operating points">
        <thead><tr><th>Family</th><th>Policy</th><th class="numeric">λ</th><th class="numeric">Accuracy</th><th class="numeric">Avg. samples</th><th class="numeric">Utility</th></tr></thead>
        <tbody>{learned_table_rows}</tbody>
      </table>
    </div>
    <details>
      <summary>Complete evaluation matrix <span>{len(payload["rows"])} scored rows</span></summary>
      <div class="table-shell">
        <table aria-label="Complete benchmark evaluation matrix">
          <thead><tr><th>Family</th><th>Policy</th><th class="numeric">Scoring cost</th><th class="numeric">Accuracy</th><th class="numeric">Avg. samples</th><th class="numeric">Utility</th></tr></thead>
          <tbody>{all_table_rows}</tbody>
        </table>
      </div>
    </details>
  </section>
</main>
<footer class="report-footer">
  <span>BranchPilot benchmark artifact</span>
  <span>Schema v{schema_version} · {len(points)} unique policies · no external assets</span>
</footer>
</div>
</body>
</html>
"""
        destination = Path(html_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(document, encoding="utf-8")
