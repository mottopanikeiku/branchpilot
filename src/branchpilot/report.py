from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from branchpilot.artifacts import atomic_write_text

_PALETTE = {
    "canvas": "#090b10",
    "surface": "#10141c",
    "surface-raised": "#171c26",
    "rule": "#303846",
    "rule-strong": "#4a5668",
    "text": "#edf0f4",
    "muted": "#b3bdca",
    "subtle": "#8d99a9",
    "learned": "#c7b9f6",
    "learned-strong": "#e4ddfb",
    "fixed": "#78afe8",
    "heuristic": "#e2b364",
    "other": "#a7b0ba",
    "positive": "#8bc9ae",
    "negative": "#e3a098",
}
_FAMILY_META = {
    "offline-rl": ("Learned policy", _PALETTE["learned"]),
    "fixed": ("Fixed-sample", _PALETTE["fixed"]),
    "heuristic": ("Heuristic", _PALETTE["heuristic"]),
}
_REQUIRED_ROW_FIELDS = (
    "policy",
    "family",
    "scoring_cost",
    "accuracy",
    "accuracy_interval",
    "average_samples",
    "average_samples_interval",
    "average_tokens",
    "average_tokens_interval",
    "p50_samples",
    "p90_samples",
    "utility",
    "utility_interval",
    "stop_histogram",
)
_REQUIRED_COMPARISON_FIELDS = (
    "scoring_cost",
    "learned_policy",
    "baseline_policy",
    "selection",
    "accuracy_delta",
    "accuracy_delta_interval",
    "average_samples_delta",
    "average_samples_delta_interval",
    "average_tokens_delta",
    "average_tokens_delta_interval",
    "utility_delta",
    "utility_delta_interval",
)


def _invalid(path: str, message: str) -> None:
    raise ValueError(f"invalid BranchPilot benchmark schema v2: {path} {message}")


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _invalid(path, "must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _invalid(path, "must be an array")
    return value


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _invalid(path, "must be a non-empty string")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _invalid(path, "must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        _invalid(path, "must be a finite number")
    return number


def _integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _invalid(path, "must be an integer")
    if minimum is not None and value < minimum:
        _invalid(path, f"must be at least {minimum}")
    return value


def _required(mapping: dict[str, Any], fields: tuple[str, ...], path: str) -> None:
    for field in fields:
        if field not in mapping:
            _invalid(f"{path}.{field}", "is required")


def _interval(value: Any, path: str) -> tuple[float, float]:
    interval = _mapping(value, path)
    _required(interval, ("lower", "upper"), path)
    lower = _number(interval["lower"], f"{path}.lower")
    upper = _number(interval["upper"], f"{path}.upper")
    if lower > upper:
        _invalid(path, "must have lower <= upper")
    return lower, upper


def _sha256(value: Any, path: str) -> str:
    digest = _string(value, path)
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        _invalid(path, "must be a 64-character hexadecimal SHA-256 digest")
    return digest


def _validate_payload(payload: Any) -> dict[str, Any]:
    root = _mapping(payload, "payload")
    required = (
        "schema_version",
        "records",
        "max_samples",
        "costs",
        "rows",
        "comparisons",
        "bootstrap",
        "objective",
        "data",
        "policy",
        "pareto_frontier",
    )
    _required(root, required, "payload")
    if root["schema_version"] != 2:
        _invalid("payload.schema_version", "must equal 2")

    records = _integer(root["records"], "payload.records", minimum=1)
    max_samples = _integer(root["max_samples"], "payload.max_samples", minimum=1)
    raw_costs = _list(root["costs"], "payload.costs")
    if not raw_costs:
        _invalid("payload.costs", "must contain at least one evaluated cost")
    costs = [_number(value, f"payload.costs[{index}]") for index, value in enumerate(raw_costs)]
    if len(set(costs)) != len(costs):
        _invalid("payload.costs", "must not contain duplicate costs")

    rows = _list(root["rows"], "payload.rows")
    if not rows:
        _invalid("payload.rows", "must contain measured policies")
    learned_by_cost: dict[float, list[str]] = {cost: [] for cost in costs}
    policies_by_cost: dict[float, dict[str, str]] = {cost: {} for cost in costs}
    full_rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        path = f"payload.rows[{index}]"
        row = _mapping(raw_row, path)
        _required(row, _REQUIRED_ROW_FIELDS, path)
        policy = _string(row["policy"], f"{path}.policy")
        family = _string(row["family"], f"{path}.family")
        cost = _number(row["scoring_cost"], f"{path}.scoring_cost")
        if cost not in learned_by_cost:
            _invalid(f"{path}.scoring_cost", "must be one of payload.costs")
        accuracy = _number(row["accuracy"], f"{path}.accuracy")
        if not 0.0 <= accuracy <= 1.0:
            _invalid(f"{path}.accuracy", "must be in [0, 1]")
        accuracy_lower, accuracy_upper = _interval(
            row["accuracy_interval"], f"{path}.accuracy_interval"
        )
        if accuracy_lower < 0.0 or accuracy_upper > 1.0:
            _invalid(f"{path}.accuracy_interval", "must stay within [0, 1]")
        average_samples = _number(row["average_samples"], f"{path}.average_samples")
        if average_samples < 1.0 or average_samples > max_samples:
            _invalid(f"{path}.average_samples", f"must be in [1, {max_samples}]")
        sample_lower, sample_upper = _interval(
            row["average_samples_interval"], f"{path}.average_samples_interval"
        )
        if sample_lower < 1.0 or sample_upper > max_samples:
            _invalid(
                f"{path}.average_samples_interval",
                f"must stay within [1, {max_samples}]",
            )
        average_tokens = _number(row["average_tokens"], f"{path}.average_tokens")
        if average_tokens < 0.0:
            _invalid(f"{path}.average_tokens", "must be non-negative")
        token_lower, _ = _interval(
            row["average_tokens_interval"], f"{path}.average_tokens_interval"
        )
        if token_lower < 0.0:
            _invalid(f"{path}.average_tokens_interval", "must be non-negative")
        _number(row["p50_samples"], f"{path}.p50_samples")
        _number(row["p90_samples"], f"{path}.p90_samples")
        _number(row["utility"], f"{path}.utility")
        _interval(row["utility_interval"], f"{path}.utility_interval")
        histogram = _list(row["stop_histogram"], f"{path}.stop_histogram")
        if len(histogram) != max_samples:
            _invalid(
                f"{path}.stop_histogram",
                f"must contain exactly {max_samples} sample-count bins",
            )
        counts = [
            _integer(count, f"{path}.stop_histogram[{bin_index}]", minimum=0)
            for bin_index, count in enumerate(histogram)
        ]
        if sum(counts) != records:
            _invalid(f"{path}.stop_histogram", f"must sum to payload.records ({records})")
        if policy in policies_by_cost[cost] and policies_by_cost[cost][policy] != family:
            _invalid(f"{path}.policy", "cannot identify more than one family at the same cost")
        policies_by_cost[cost][policy] = family
        if family == "offline-rl":
            learned_by_cost[cost].append(policy)
        full_rows.append(row)

    for cost, policies in learned_by_cost.items():
        if len(policies) != 1:
            _invalid(
                "payload.rows",
                f"must contain exactly one offline-rl policy at evaluated cost {cost:g}",
            )

    comparisons = _list(root["comparisons"], "payload.comparisons")
    comparison_costs: set[float] = set()
    for index, raw_comparison in enumerate(comparisons):
        path = f"payload.comparisons[{index}]"
        comparison = _mapping(raw_comparison, path)
        _required(comparison, _REQUIRED_COMPARISON_FIELDS, path)
        cost = _number(comparison["scoring_cost"], f"{path}.scoring_cost")
        if cost not in policies_by_cost:
            _invalid(f"{path}.scoring_cost", "must be one of payload.costs")
        if cost in comparison_costs:
            _invalid(f"{path}.scoring_cost", "must have exactly one paired comparison per cost")
        comparison_costs.add(cost)
        learned_policy = _string(comparison["learned_policy"], f"{path}.learned_policy")
        baseline_policy = _string(comparison["baseline_policy"], f"{path}.baseline_policy")
        _string(comparison["selection"], f"{path}.selection")
        if policies_by_cost[cost].get(learned_policy) != "offline-rl":
            _invalid(f"{path}.learned_policy", "must name the offline-rl row at the same cost")
        baseline_family = policies_by_cost[cost].get(baseline_policy)
        if baseline_family is None or baseline_family == "offline-rl":
            _invalid(f"{path}.baseline_policy", "must name a non-learned row at the same cost")
        for field in (
            "accuracy_delta",
            "average_samples_delta",
            "average_tokens_delta",
            "utility_delta",
        ):
            _number(comparison[field], f"{path}.{field}")
            _interval(comparison[f"{field}_interval"], f"{path}.{field}_interval")
    missing_comparisons = [cost for cost in costs if cost not in comparison_costs]
    if missing_comparisons:
        formatted = ", ".join(f"{cost:g}" for cost in missing_comparisons)
        _invalid("payload.comparisons", f"is missing evaluated costs: {formatted}")

    bootstrap = _mapping(root["bootstrap"], "payload.bootstrap")
    _required(bootstrap, ("resamples", "seed", "confidence"), "payload.bootstrap")
    _integer(bootstrap["resamples"], "payload.bootstrap.resamples", minimum=1)
    _integer(bootstrap["seed"], "payload.bootstrap.seed")
    confidence = _number(bootstrap["confidence"], "payload.bootstrap.confidence")
    if not math.isclose(confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
        _invalid("payload.bootstrap.confidence", "must equal 0.95 for this 95% evidence report")

    objective = _mapping(root["objective"], "payload.objective")
    _required(objective, ("name", "formula", "cost_unit"), "payload.objective")
    for field in ("name", "formula", "cost_unit"):
        _string(objective[field], f"payload.objective.{field}")

    data = _mapping(root["data"], "payload.data")
    _required(data, ("path", "sha256", "schema_version", "profile"), "payload.data")
    _string(data["path"], "payload.data.path")
    _sha256(data["sha256"], "payload.data.sha256")
    if data["schema_version"] is None:
        _invalid("payload.data.schema_version", "is required")
    profile = _mapping(data["profile"], "payload.data.profile")
    if not profile:
        _invalid("payload.data.profile", "must contain dataset evidence")
    if (
        "record_count" in profile
        and _integer(profile["record_count"], "payload.data.profile.record_count", minimum=1)
        != records
    ):
        _invalid("payload.data.profile.record_count", "must equal payload.records")

    policy = _mapping(root["policy"], "payload.policy")
    _required(
        policy,
        ("path", "sha256", "artifact_version", "feature_names", "training"),
        "payload.policy",
    )
    _string(policy["path"], "payload.policy.path")
    _sha256(policy["sha256"], "payload.policy.sha256")
    if policy["artifact_version"] is None:
        _invalid("payload.policy.artifact_version", "is required")
    features = _list(policy["feature_names"], "payload.policy.feature_names")
    if not features:
        _invalid("payload.policy.feature_names", "must contain at least one feature")
    for index, feature in enumerate(features):
        _string(feature, f"payload.policy.feature_names[{index}]")
    training = _mapping(policy["training"], "payload.policy.training")
    _required(training, ("algorithm",), "payload.policy.training")
    _string(training["algorithm"], "payload.policy.training.algorithm")

    frontier = _list(root["pareto_frontier"], "payload.pareto_frontier")
    if not frontier:
        _invalid("payload.pareto_frontier", "must contain at least one operating point")
    frontier_fields = (
        "policy",
        "family",
        "scoring_cost",
        "accuracy",
        "average_samples",
        "average_tokens",
        "p50_samples",
        "p90_samples",
        "utility",
    )
    for index, raw_point in enumerate(frontier):
        path = f"payload.pareto_frontier[{index}]"
        point = _mapping(raw_point, path)
        _required(point, frontier_fields, path)
        policy_name = _string(point["policy"], f"{path}.policy")
        family = _string(point["family"], f"{path}.family")
        cost = _number(point["scoring_cost"], f"{path}.scoring_cost")
        accuracy = _number(point["accuracy"], f"{path}.accuracy")
        samples = _number(point["average_samples"], f"{path}.average_samples")
        for field in ("average_tokens", "p50_samples", "p90_samples", "utility"):
            _number(point[field], f"{path}.{field}")
        matching = any(
            str(row["policy"]) == policy_name
            and str(row["family"]) == family
            and math.isclose(float(row["accuracy"]), accuracy, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(row["average_samples"]), samples, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(float(row["scoring_cost"]), cost, rel_tol=0.0, abs_tol=1e-12)
            for row in full_rows
        )
        if not matching:
            _invalid(path, "must match a measured row so its uncertainty is available")

    protocol = root.get("protocol")
    if protocol is not None:
        protocol_mapping = _mapping(protocol, "payload.protocol")
        _required(
            protocol_mapping,
            ("path", "sha256", "status", "evidence_tier"),
            "payload.protocol",
        )
        _string(protocol_mapping["path"], "payload.protocol.path")
        _sha256(protocol_mapping["sha256"], "payload.protocol.sha256")
        _string(protocol_mapping["status"], "payload.protocol.status")
        _string(protocol_mapping["evidence_tier"], "payload.protocol.evidence_tier")

    return root


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _family_label(family: str) -> str:
    metadata = _FAMILY_META.get(family)
    if metadata is not None:
        return metadata[0]
    return family.replace("-", " ").replace("_", " ").title()


def _family_color(family: str) -> str:
    metadata = _FAMILY_META.get(family)
    return metadata[1] if metadata is not None else _PALETTE["other"]


def _family_class(family: str) -> str:
    return family if family in _FAMILY_META else "other"


def _unique_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["family"]),
            str(row["policy"]),
            float(row["accuracy"]),
            float(row["average_samples"]),
        )
        points.setdefault(key, row)
    family_order = {"offline-rl": 0, "fixed": 1, "heuristic": 2}
    return sorted(
        points.values(),
        key=lambda row: (
            family_order.get(str(row["family"]), 3),
            float(row["average_samples"]),
            str(row["policy"]),
        ),
    )


def _frontier_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload["rows"]
    matched: list[dict[str, Any]] = []
    for point in payload["pareto_frontier"]:
        match = next(
            row
            for row in rows
            if str(row["policy"]) == str(point["policy"])
            and str(row["family"]) == str(point["family"])
            and math.isclose(
                float(row["scoring_cost"]),
                float(point["scoring_cost"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(row["accuracy"]),
                float(point["accuracy"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(row["average_samples"]),
                float(point["average_samples"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        matched.append(match)
    return sorted(matched, key=lambda row: (float(row["average_samples"]), -float(row["accuracy"])))


def _marker_svg(family: str, x: float, y: float, color: str, *, frontier: bool) -> str:
    stroke_width = 3 if frontier else 2
    radius = 7 if frontier else 5
    if family == "offline-rl":
        return (
            f'<path d="M {x:.1f} {y - radius:.1f} L {x + radius:.1f} {y:.1f} '
            f'L {x:.1f} {y + radius:.1f} L {x - radius:.1f} {y:.1f} Z" '
            f'fill="{color}" stroke="{_PALETTE["canvas"]}" stroke-width="{stroke_width}"/>'
        )
    if family == "fixed":
        return (
            f'<rect x="{x - radius:.1f}" y="{y - radius:.1f}" width="{radius * 2}" '
            f'height="{radius * 2}" rx="2" fill="{color}" stroke="{_PALETTE["canvas"]}" '
            f'stroke-width="{stroke_width}"/>'
        )
    if family == "heuristic":
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" '
            f'stroke="{_PALETTE["canvas"]}" stroke-width="{stroke_width}"/>'
        )
    return (
        f'<path d="M {x:.1f} {y - radius:.1f} L {x + radius:.1f} {y + radius:.1f} '
        f'L {x - radius:.1f} {y + radius:.1f} Z" fill="{color}" '
        f'stroke="{_PALETTE["canvas"]}" stroke-width="{stroke_width}"/>'
    )


def _render_svg(payload: dict[str, Any], title: str) -> str:
    points = _unique_points(payload["rows"])
    frontier = _frontier_rows(payload)
    frontier_keys = {
        (
            str(row["family"]),
            str(row["policy"]),
            float(row["accuracy"]),
            float(row["average_samples"]),
        )
        for row in frontier
    }
    width, height = 1200, 720
    left, right, top, bottom = 96, 54, 154, 84
    plot_width, plot_height = width - left - right, height - top - bottom

    x_bounds = [
        bound
        for point in points
        for bound in (
            float(point["average_samples_interval"]["lower"]),
            float(point["average_samples_interval"]["upper"]),
        )
    ]
    y_bounds = [
        bound
        for point in points
        for bound in (
            float(point["accuracy_interval"]["lower"]),
            float(point["accuracy_interval"]["upper"]),
        )
    ]
    x_min, x_max = min(x_bounds), max(x_bounds)
    x_span = max(0.5, x_max - x_min)
    x_floor = max(0.0, x_min - x_span * 0.06)
    x_ceiling = x_max + x_span * 0.06
    y_min, y_max = min(y_bounds), max(y_bounds)
    y_span = max(0.06, y_max - y_min)
    y_floor = max(0.0, y_min - y_span * 0.08)
    y_ceiling = min(1.0, y_max + y_span * 0.08)
    if x_ceiling - x_floor < 1e-9:
        x_ceiling = x_floor + 1.0
    if y_ceiling - y_floor < 1e-9:
        y_ceiling = min(1.0, y_floor + 0.1)
        y_floor = max(0.0, y_ceiling - 0.1)

    def x(value: float) -> float:
        return left + (value - x_floor) / (x_ceiling - x_floor) * plot_width

    def y(value: float) -> float:
        return top + (y_ceiling - value) / (y_ceiling - y_floor) * plot_height

    escaped_title = _escape(title)
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
            'role="img" aria-labelledby="branchpilot-chart-title branchpilot-chart-desc">'
        ),
        f'<title id="branchpilot-chart-title">{escaped_title}</title>',
        (
            '<desc id="branchpilot-chart-desc">'
            f"{_escape(len(points))} measured policy operating points from "
            f"{_escape(payload['records'])} held-out records. Horizontal and vertical bars "
            "show paired-bootstrap 95% confidence intervals for average samples and accuracy. "
            "A solid line joins the empirical Pareto frontier. Learned policies are diamonds, "
            "fixed-sample policies are squares, heuristics are circles, and unknown families "
            "are triangles.</desc>"
        ),
        "<defs>",
        (
            '<clipPath id="branchpilot-plot-clip">'
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}"/>'
            "</clipPath>"
        ),
        "<style>",
        'text{font-family:"Avenir Next",Avenir,"Segoe UI",sans-serif}',
        f'.title{{font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif;font-size:30px;font-weight:700;fill:{_PALETTE["text"]}}}',
        f".kicker{{font-size:11px;font-weight:700;letter-spacing:1.8px;fill:{_PALETTE['learned']}}}",
        f".subtitle,.legend,.axis{{fill:{_PALETTE['muted']}}}",
        ".subtitle{font-size:13px}.legend{font-size:12px;font-weight:600}.axis{font-size:13px;font-weight:600}",
        f'.tick,.note,.label{{font-family:"SFMono-Regular",Consolas,monospace;fill:{_PALETTE["subtle"]}}}',
        ".tick{font-size:11px}.note{font-size:11px}.label{font-size:10px;font-weight:700}",
        f".grid{{stroke:{_PALETTE['rule']};stroke-width:1}}",
        f".frontier{{fill:none;stroke:{_PALETTE['text']};stroke-width:2.5;stroke-opacity:.72;stroke-linecap:round;stroke-linejoin:round}}",
        ".uncertainty{fill:none;stroke-width:1.5;stroke-opacity:.58}",
        ".uncertainty-frontier{stroke-width:2;stroke-opacity:.9}",
        f".label{{paint-order:stroke;stroke:{_PALETTE['surface']};stroke-width:4px;stroke-linejoin:round}}",
        "</style></defs>",
        f'<rect width="{width}" height="{height}" fill="{_PALETTE["canvas"]}"/>',
        f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" fill="none" stroke="{_PALETTE["rule"]}"/>',
        f'<text x="{left}" y="35" class="kicker">HELD-OUT EVIDENCE · 95% BOOTSTRAP INTERVALS</text>',
        f'<text x="{left}" y="70" class="title">{escaped_title}</text>',
        (
            f'<text x="{left}" y="95" class="subtitle">{int(payload["records"]):,} records · '
            f"{len(payload['costs'])} evaluated costs · line marks the empirical frontier</text>"
        ),
        (
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" '
            f'fill="{_PALETTE["surface"]}" stroke="{_PALETTE["rule"]}"/>'
        ),
    ]

    legend_families = [
        family for family in _FAMILY_META if any(str(point["family"]) == family for point in points)
    ]
    if any(str(point["family"]) not in _FAMILY_META for point in points):
        legend_families.append("other")
    legend_x = left
    legend_y = 126
    for family in legend_families:
        label = "Other family" if family == "other" else _FAMILY_META[family][0]
        color = _PALETTE["other"] if family == "other" else _FAMILY_META[family][1]
        parts.append(_marker_svg(family, legend_x, legend_y, color, frontier=False))
        parts.append(
            f'<text x="{legend_x + 13}" y="{legend_y + 4}" class="legend">{_escape(label)}</text>'
        )
        legend_x += 142 if family != "offline-rl" else 154
    parts.extend(
        [
            f'<line x1="{legend_x + 3}" y1="{legend_y}" x2="{legend_x + 35}" y2="{legend_y}" class="frontier"/>',
            f'<text x="{legend_x + 43}" y="{legend_y + 4}" class="legend">Pareto frontier</text>',
            f'<line x1="{legend_x + 163}" y1="{legend_y}" x2="{legend_x + 195}" y2="{legend_y}" stroke="{_PALETTE["muted"]}" class="uncertainty"/>',
            f'<line x1="{legend_x + 163}" y1="{legend_y - 5}" x2="{legend_x + 163}" y2="{legend_y + 5}" stroke="{_PALETTE["muted"]}" class="uncertainty"/>',
            f'<line x1="{legend_x + 195}" y1="{legend_y - 5}" x2="{legend_x + 195}" y2="{legend_y + 5}" stroke="{_PALETTE["muted"]}" class="uncertainty"/>',
            f'<text x="{legend_x + 203}" y="{legend_y + 4}" class="legend">95% CI</text>',
        ]
    )

    for tick_index in range(6):
        value = y_floor + (y_ceiling - y_floor) * tick_index / 5
        py = y(value)
        parts.append(
            f'<line x1="{left}" y1="{py:.1f}" x2="{width - right}" y2="{py:.1f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 13}" y="{py + 4:.1f}" text-anchor="end" class="tick">{value:.1%}</text>'
        )
    for tick_index in range(6):
        value = x_floor + (x_ceiling - x_floor) * tick_index / 5
        px = x(value)
        parts.append(
            f'<line x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{height - bottom}" class="grid"/>'
        )
        parts.append(
            f'<text x="{px:.1f}" y="{height - bottom + 27}" text-anchor="middle" class="tick">{value:.1f}</text>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 25}" text-anchor="middle" class="axis">Average samples per record</text>',
            f'<text x="25" y="{top + plot_height / 2:.1f}" transform="rotate(-90 25 {top + plot_height / 2:.1f})" text-anchor="middle" class="axis">Exact-match accuracy</text>',
            f'<text x="{width - right - 14}" y="{top + 22}" text-anchor="end" class="note">higher accuracy ↑ · lower compute ←</text>',
        ]
    )

    if len(frontier) > 1:
        frontier_path = " ".join(
            f"{'M' if index == 0 else 'L'} {x(float(point['average_samples'])):.1f} {y(float(point['accuracy'])):.1f}"
            for index, point in enumerate(frontier)
        )
        parts.append(
            f'<path d="{frontier_path}" class="frontier" clip-path="url(#branchpilot-plot-clip)"/>'
        )

    parts.append('<g clip-path="url(#branchpilot-plot-clip)">')
    for point in points:
        family = str(point["family"])
        color = _family_color(family)
        px = x(float(point["average_samples"]))
        py = y(float(point["accuracy"]))
        x_lower = x(float(point["average_samples_interval"]["lower"]))
        x_upper = x(float(point["average_samples_interval"]["upper"]))
        y_lower = y(float(point["accuracy_interval"]["lower"]))
        y_upper = y(float(point["accuracy_interval"]["upper"]))
        key = (
            family,
            str(point["policy"]),
            float(point["accuracy"]),
            float(point["average_samples"]),
        )
        is_frontier = key in frontier_keys
        uncertainty_class = "uncertainty uncertainty-frontier" if is_frontier else "uncertainty"
        tooltip = (
            f"{point['policy']} ({_family_label(family)}): "
            f"{float(point['accuracy']):.1%} accuracy, 95% CI "
            f"[{float(point['accuracy_interval']['lower']):.1%}, "
            f"{float(point['accuracy_interval']['upper']):.1%}]; "
            f"{float(point['average_samples']):.2f} average samples, 95% CI "
            f"[{float(point['average_samples_interval']['lower']):.2f}, "
            f"{float(point['average_samples_interval']['upper']):.2f}]"
        )
        parts.append(f"<g><title>{_escape(tooltip)}</title>")
        parts.extend(
            [
                f'<line x1="{x_lower:.1f}" y1="{py:.1f}" x2="{x_upper:.1f}" y2="{py:.1f}" stroke="{color}" class="{uncertainty_class}"/>',
                f'<line x1="{x_lower:.1f}" y1="{py - 5:.1f}" x2="{x_lower:.1f}" y2="{py + 5:.1f}" stroke="{color}" class="{uncertainty_class}"/>',
                f'<line x1="{x_upper:.1f}" y1="{py - 5:.1f}" x2="{x_upper:.1f}" y2="{py + 5:.1f}" stroke="{color}" class="{uncertainty_class}"/>',
                f'<line x1="{px:.1f}" y1="{y_upper:.1f}" x2="{px:.1f}" y2="{y_lower:.1f}" stroke="{color}" class="{uncertainty_class}"/>',
                f'<line x1="{px - 5:.1f}" y1="{y_upper:.1f}" x2="{px + 5:.1f}" y2="{y_upper:.1f}" stroke="{color}" class="{uncertainty_class}"/>',
                f'<line x1="{px - 5:.1f}" y1="{y_lower:.1f}" x2="{px + 5:.1f}" y2="{y_lower:.1f}" stroke="{color}" class="{uncertainty_class}"/>',
                _marker_svg(family, px, py, color, frontier=is_frontier),
                "</g>",
            ]
        )
    parts.append("</g>")

    label_points = frontier
    label_positions: list[float] = []
    for index, point in enumerate(label_points):
        px = x(float(point["average_samples"]))
        py = y(float(point["accuracy"]))
        label_y = py - 14 if index % 2 == 0 else py + 24
        label_y = max(top + 15, min(height - bottom - 8, label_y))
        if label_positions and abs(label_y - label_positions[-1]) < 14:
            label_y = min(height - bottom - 8, label_positions[-1] + 14)
        label_positions.append(label_y)
        label = str(point["policy"])
        if str(point["family"]) == "offline-rl":
            label = label.replace("BranchPilot ", "")
        near_right = px > width - right - 70
        label_x = px - 11 if near_right else px + 11
        anchor = "end" if near_right else "start"
        parts.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" class="label">{_escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def render_svg(payload: dict[str, Any], title: str = "Accuracy × inference compute") -> str:
    validated = _validate_payload(payload)
    return _render_svg(validated, str(title))


def _fmt_cost(value: Any) -> str:
    return f"{float(value):.4g}"


def _fmt_point(value: Any, *, kind: str, signed: bool = False) -> str:
    number = float(value)
    if kind == "percent":
        return f"{number:+.1%}" if signed else f"{number:.1%}"
    if kind == "samples":
        return f"{number:+.2f}" if signed else f"{number:.2f}"
    if kind == "tokens":
        return f"{number:+,.0f}" if signed else f"{number:,.0f}"
    return f"{number:+.3f}" if signed else f"{number:.3f}"


def _fmt_interval(interval: dict[str, Any], *, kind: str, signed: bool = False) -> str:
    lower = _fmt_point(interval["lower"], kind=kind, signed=signed)
    upper = _fmt_point(interval["upper"], kind=kind, signed=signed)
    return f"[{lower}, {upper}]"


def _metric_cell(row: dict[str, Any], field: str, *, kind: str) -> str:
    return (
        '<td class="numeric metric-cell">'
        f"<span>{_fmt_point(row[field], kind=kind)}</span>"
        f"<small>95% CI {_fmt_interval(row[f'{field}_interval'], kind=kind)}</small>"
        "</td>"
    )


def _family_key(family: str) -> str:
    family_class = _family_class(family)
    return (
        f'<span class="family-key family-{family_class}">'
        f'<span aria-hidden="true"></span>{_escape(_family_label(family))}</span>'
    )


def _render_metric_rows(rows: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for row in rows:
        histogram = ", ".join(str(int(count)) for count in row["stop_histogram"])
        rendered.append(
            "<tr>"
            f"<td>{_family_key(str(row['family']))}</td>"
            f'<th scope="row">{_escape(row["policy"])}</th>'
            f'<td class="numeric">{_fmt_cost(row["scoring_cost"])}</td>'
            f"{_metric_cell(row, 'accuracy', kind='percent')}"
            f"{_metric_cell(row, 'average_samples', kind='samples')}"
            f"{_metric_cell(row, 'average_tokens', kind='tokens')}"
            f'<td class="numeric">{_fmt_point(row["p50_samples"], kind="samples")}</td>'
            f'<td class="numeric">{_fmt_point(row["p90_samples"], kind="samples")}</td>'
            f"{_metric_cell(row, 'utility', kind='utility')}"
            f'<td class="histogram-cell"><code>{_escape(histogram)}</code></td>'
            "</tr>"
        )
    return "".join(rendered)


def _delta_cell(comparison: dict[str, Any], field: str, *, kind: str) -> str:
    return (
        '<td class="numeric metric-cell">'
        f"<span>{_fmt_point(comparison[field], kind=kind, signed=True)}</span>"
        f"<small>95% CI {_fmt_interval(comparison[f'{field}_interval'], kind=kind, signed=True)}</small>"
        "</td>"
    )


def _render_comparison_rows(comparisons: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for comparison in comparisons:
        is_frozen = str(comparison["selection"]) == "validation-frozen"
        badge_class = "frozen" if is_frozen else "exploratory"
        rendered.append(
            "<tr>"
            f'<th scope="row" class="numeric">{_fmt_cost(comparison["scoring_cost"])}</th>'
            f"<td>{_escape(comparison['learned_policy'])}</td>"
            f"<td>{_escape(comparison['baseline_policy'])}</td>"
            f'<td><span class="status-badge status-{badge_class}">{_escape(comparison["selection"])}</span></td>'
            f"{_delta_cell(comparison, 'accuracy_delta', kind='percent')}"
            f"{_delta_cell(comparison, 'average_samples_delta', kind='samples')}"
            f"{_delta_cell(comparison, 'average_tokens_delta', kind='tokens')}"
            f"{_delta_cell(comparison, 'utility_delta', kind='utility')}"
            "</tr>"
        )
    return "".join(rendered)


def _comparison_evidence(
    learned: dict[str, Any], comparison: dict[str, Any]
) -> tuple[str, str, str]:
    selection = str(comparison["selection"])
    frozen = selection == "validation-frozen"
    lower = float(comparison["utility_delta_interval"]["lower"])
    upper = float(comparison["utility_delta_interval"]["upper"])
    cost = _fmt_cost(comparison["scoring_cost"])
    learned_name = str(comparison["learned_policy"])
    baseline_name = str(comparison["baseline_policy"])
    if frozen:
        status = "Validation-frozen paired comparison"
        tone = "positive" if lower > 0 else "negative" if upper < 0 else "neutral"
        if lower > 0:
            conclusion = (
                f"At λ={cost}, the validation-frozen paired 95% utility interval is strictly "
                f"above zero. {learned_name} beats {baseline_name} on the stated additional-sample "
                "utility objective at this evaluated cost."
            )
        elif upper < 0:
            conclusion = (
                f"At λ={cost}, the validation-frozen paired 95% utility interval is strictly "
                f"below zero. {learned_name} trails {baseline_name} on the stated utility objective "
                "at this evaluated cost."
            )
        else:
            conclusion = (
                f"At λ={cost}, the validation-frozen paired 95% utility interval includes zero. "
                f"These data do not resolve a utility difference between {learned_name} and "
                f"{baseline_name} at this evaluated cost."
            )
        return status, tone, conclusion

    status = "Exploratory comparator selection"
    tone = "exploratory"
    direction = "above zero" if lower > 0 else "below zero" if upper < 0 else "includes zero"
    conclusion = (
        f"At λ={cost}, the paired 95% utility interval {direction}. The comparator method is "
        f"{selection}, not validation-frozen; {baseline_name} was selected using this evaluation "
        "evidence. Treat the result as exploratory and hypothesis-generating, not confirmatory."
    )
    return status, tone, conclusion


def _interactive_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for cost in payload["costs"]:
        learned = next(
            row
            for row in payload["rows"]
            if float(row["scoring_cost"]) == float(cost) and str(row["family"]) == "offline-rl"
        )
        comparison = next(
            item for item in payload["comparisons"] if float(item["scoring_cost"]) == float(cost)
        )
        status, tone, conclusion = _comparison_evidence(learned, comparison)
        entries.append(
            {
                "cost": _fmt_cost(cost),
                "policy": str(learned["policy"]),
                "comparator": str(comparison["baseline_policy"]),
                "selection": str(comparison["selection"]),
                "status": status,
                "tone": tone,
                "conclusion": conclusion,
                "accuracy": {
                    "value": _fmt_point(learned["accuracy"], kind="percent"),
                    "interval": f"95% CI {_fmt_interval(learned['accuracy_interval'], kind='percent')}",
                },
                "samples": {
                    "value": _fmt_point(learned["average_samples"], kind="samples"),
                    "interval": f"95% CI {_fmt_interval(learned['average_samples_interval'], kind='samples')}",
                },
                "tokens": {
                    "value": _fmt_point(learned["average_tokens"], kind="tokens"),
                    "interval": f"95% CI {_fmt_interval(learned['average_tokens_interval'], kind='tokens')}",
                },
                "utility": {
                    "value": _fmt_point(comparison["utility_delta"], kind="utility", signed=True),
                    "interval": f"paired 95% CI {_fmt_interval(comparison['utility_delta_interval'], kind='utility', signed=True)}",
                },
                "histogram": [int(count) for count in learned["stop_histogram"]],
            }
        )
    return entries


def _script_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _histogram_rows(counts: list[int]) -> str:
    total = sum(counts)
    rendered: list[str] = []
    for sample_count, count in enumerate(counts, start=1):
        percentage = count / total * 100 if total else 0.0
        rendered.append(
            f'<li class="stop-row" aria-label="Stopped after {sample_count} samples: {count} records, {percentage:.1f} percent">'
            f'<span class="stop-bin">{sample_count}</span>'
            '<span class="stop-track" aria-hidden="true">'
            f'<span class="stop-fill" style="--share:{percentage:.4f}%"></span></span>'
            f'<span class="stop-count">{count:,}</span></li>'
        )
    return "".join(rendered)


def _pretty_json(value: Any) -> str:
    return _escape(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


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
  --other: {_PALETTE["other"]};
  --positive: {_PALETTE["positive"]};
  --negative: {_PALETTE["negative"]};
  --space-1: .25rem;
  --space-2: .5rem;
  --space-3: .75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-7: 3rem;
  --space-8: 4rem;
  --space-9: 6rem;
  --radius-sm: .25rem;
  --radius-md: .5rem;
  --shadow: 0 1.5rem 4rem rgb(3 5 9 / .36);
  font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif;
  background: var(--canvas);
  color: var(--text);
}}
* {{ box-sizing: border-box; }}
html {{ background: var(--canvas); scroll-behavior: smooth; }}
body {{
  margin: 0;
  min-width: 20rem;
  background: var(--canvas);
  color: var(--text);
  line-height: 1.55;
  text-rendering: optimizeLegibility;
}}
a {{ color: var(--learned-strong); text-underline-offset: .2em; }}
a:hover {{ color: var(--text); }}
a:focus-visible, input:focus-visible, summary:focus-visible {{
  outline: 2px solid var(--learned-strong);
  outline-offset: var(--space-1);
}}
.skip-link {{
  position: fixed;
  z-index: 20;
  top: var(--space-3);
  left: var(--space-3);
  padding: var(--space-2) var(--space-4);
  transform: translateY(-220%);
  background: var(--text);
  color: var(--canvas);
  font-weight: 800;
}}
.skip-link:focus {{ transform: translateY(0); }}
.page-shell {{ width: min(calc(100% - 2rem), 78rem); margin-inline: auto; }}
.hero {{ padding-block: var(--space-5) var(--space-8); border-bottom: 1px solid var(--rule); }}
.topline {{
  display: flex;
  justify-content: space-between;
  gap: var(--space-4);
  padding-bottom: var(--space-5);
  border-bottom: 1px solid var(--rule);
  color: var(--subtle);
  font: 700 .6875rem/1.4 "SFMono-Regular", Consolas, monospace;
  letter-spacing: .11em;
  text-transform: uppercase;
}}
.hero-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1.55fr) minmax(15rem, .45fr);
  gap: var(--space-8);
  align-items: end;
  padding-top: var(--space-8);
}}
.kicker, .section-index, .eyebrow {{
  color: var(--learned);
  font: 700 .6875rem/1.35 "SFMono-Regular", Consolas, monospace;
  letter-spacing: .15em;
  text-transform: uppercase;
}}
.kicker {{ margin: 0 0 var(--space-4); }}
h1, h2, h3, p {{ margin-top: 0; }}
h1, h2 {{ font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif; }}
h1 {{
  max-width: 56rem;
  margin-bottom: var(--space-5);
  font-size: clamp(2.75rem, 7vw, 5.75rem);
  font-weight: 700;
  letter-spacing: -.045em;
  line-height: .98;
}}
h1 span {{ color: var(--learned-strong); }}
.lede {{ max-width: 48rem; margin-bottom: 0; color: var(--muted); font-size: 1.0625rem; line-height: 1.72; }}
.evidence-stamp {{ border-left: 1px solid var(--rule-strong); padding-left: var(--space-5); }}
.evidence-stamp .eyebrow {{ display: block; margin-bottom: var(--space-3); }}
.evidence-stamp strong {{ display: block; margin-bottom: var(--space-3); font-size: 1.125rem; line-height: 1.35; }}
.evidence-stamp p {{ margin-bottom: 0; color: var(--subtle); font-size: .8125rem; }}
.report-section {{ padding-block: var(--space-9); border-bottom: 1px solid var(--rule); }}
.section-heading {{
  display: grid;
  grid-template-columns: minmax(15rem, .7fr) minmax(18rem, 1fr);
  gap: var(--space-7);
  align-items: end;
  margin-bottom: var(--space-7);
}}
.section-index {{ display: block; margin-bottom: var(--space-3); }}
h2 {{ margin-bottom: 0; font-size: clamp(2rem, 4.5vw, 3.25rem); letter-spacing: -.035em; line-height: 1.04; }}
.section-heading p {{ margin-bottom: 0; color: var(--muted); font-size: .9375rem; line-height: 1.72; }}
.console {{ border: 1px solid var(--rule-strong); background: var(--surface); box-shadow: var(--shadow); }}
.dial-panel {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(16rem, .52fr);
  gap: var(--space-7);
  padding: var(--space-6);
  border-bottom: 1px solid var(--rule);
}}
.dial-label {{ display: flex; justify-content: space-between; gap: var(--space-4); align-items: end; margin-bottom: var(--space-4); }}
.dial-label span {{ color: var(--muted); font-size: .8125rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }}
.dial-label output {{ color: var(--learned-strong); font: 700 1.75rem/1 "SFMono-Regular", Consolas, monospace; }}
input[type="range"] {{
  width: 100%;
  height: 2.25rem;
  margin: 0;
  accent-color: var(--learned);
  cursor: pointer;
}}
.dial-scale {{ display: flex; justify-content: space-between; gap: var(--space-2); color: var(--subtle); font: 600 .6875rem/1.3 "SFMono-Regular", Consolas, monospace; }}
.selection-meta {{ margin: 0; border-left: 1px solid var(--rule); }}
.selection-meta div {{ display: grid; grid-template-columns: 6.5rem minmax(0, 1fr); gap: var(--space-3); padding: var(--space-2) 0 var(--space-2) var(--space-5); }}
.selection-meta dt {{ color: var(--subtle); font-size: .75rem; }}
.selection-meta dd {{ margin: 0; overflow-wrap: anywhere; font-size: .8125rem; font-weight: 700; }}
.status-line {{
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-4) var(--space-6);
  border-bottom: 1px solid var(--rule);
  color: var(--muted);
  font-size: .8125rem;
  font-weight: 700;
}}
.status-line::before {{ content: ""; width: .625rem; height: .625rem; flex: 0 0 auto; border: 2px solid var(--subtle); }}
.status-line[data-tone="positive"]::before {{ border-color: var(--positive); background: var(--positive); }}
.status-line[data-tone="negative"]::before {{ border-color: var(--negative); background: var(--negative); }}
.status-line[data-tone="exploratory"]::before {{ border-color: var(--heuristic); }}
.metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.metric-card {{ min-width: 0; padding: var(--space-6); border-right: 1px solid var(--rule); }}
.metric-card:last-child {{ border-right: 0; }}
.metric-card h3 {{ margin-bottom: var(--space-4); color: var(--muted); font-size: .75rem; letter-spacing: .08em; text-transform: uppercase; }}
.metric-value {{ display: block; margin-bottom: var(--space-3); font: 700 clamp(1.75rem, 4vw, 2.75rem)/1 "SFMono-Regular", Consolas, monospace; letter-spacing: -.055em; }}
.metric-interval {{ display: block; color: var(--subtle); font: 600 .6875rem/1.55 "SFMono-Regular", Consolas, monospace; overflow-wrap: anywhere; }}
.interpretation {{ padding: var(--space-5) var(--space-6); border-block: 1px solid var(--rule); }}
.interpretation .eyebrow {{ display: block; margin-bottom: var(--space-3); }}
.interpretation p {{ max-width: 66rem; margin-bottom: 0; color: var(--muted); font-size: .9375rem; }}
.histogram {{ display: grid; grid-template-columns: minmax(15rem, .42fr) minmax(0, 1fr); gap: var(--space-7); padding: var(--space-6); }}
.histogram-copy h3 {{ margin-bottom: var(--space-3); font-size: 1rem; }}
.histogram-copy p {{ margin-bottom: 0; color: var(--subtle); font-size: .8125rem; }}
.stop-list {{ display: grid; gap: var(--space-2); margin: 0; padding: 0; list-style: none; }}
.stop-row {{ display: grid; grid-template-columns: 2rem minmax(0, 1fr) 4rem; gap: var(--space-3); align-items: center; }}
.stop-bin, .stop-count {{ font: 600 .75rem/1.3 "SFMono-Regular", Consolas, monospace; font-variant-numeric: tabular-nums; }}
.stop-bin {{ color: var(--muted); }}
.stop-count {{ color: var(--subtle); text-align: right; }}
.stop-track {{ height: .5rem; background: var(--rule); overflow: hidden; }}
.stop-fill {{ display: block; width: var(--share); height: 100%; background: var(--learned); }}
.no-script {{ margin: 0; padding: var(--space-4) var(--space-6); border-top: 1px solid var(--rule); color: var(--heuristic); font-size: .8125rem; }}
figure {{ margin: 0; }}
.figure-shell {{ overflow: hidden; border: 1px solid var(--rule); background: var(--surface); box-shadow: var(--shadow); }}
.chart-viewport {{ overflow-x: auto; }}
.chart-viewport svg {{ display: block; width: 100%; height: auto; min-width: 48rem; }}
figcaption {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: var(--space-6);
  padding: var(--space-5) var(--space-6);
  border-top: 1px solid var(--rule);
  color: var(--subtle);
  font-size: .75rem;
}}
figcaption strong {{ color: var(--muted); }}
.table-intro {{ margin: calc(-1 * var(--space-4)) 0 var(--space-5); color: var(--subtle); font-size: .8125rem; }}
.table-shell {{ overflow-x: auto; border-block: 1px solid var(--rule); }}
table {{ width: 100%; min-width: 74rem; border-collapse: collapse; font-size: .8125rem; }}
th, td {{ padding: var(--space-4) var(--space-3); border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; white-space: nowrap; }}
thead th {{ color: var(--subtle); font: 700 .6875rem/1.35 "SFMono-Regular", Consolas, monospace; letter-spacing: .07em; text-transform: uppercase; }}
tbody th {{ color: var(--text); font-weight: 700; }}
tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: 0; }}
tbody tr:hover {{ background: var(--surface-raised); }}
.numeric {{ text-align: right; font-family: "SFMono-Regular", Consolas, monospace; font-variant-numeric: tabular-nums; }}
.metric-cell span, .metric-cell small {{ display: block; }}
.metric-cell small {{ margin-top: var(--space-1); color: var(--subtle); font-size: .6875rem; }}
.family-key {{ display: inline-flex; align-items: center; gap: var(--space-2); color: var(--muted); }}
.family-key > span {{ width: .5rem; height: .5rem; flex: 0 0 auto; background: var(--other); }}
.family-offline-rl > span {{ background: var(--learned); transform: rotate(45deg); }}
.family-fixed > span {{ background: var(--fixed); border-radius: .0625rem; }}
.family-heuristic > span {{ background: var(--heuristic); border-radius: 50%; }}
.histogram-cell code {{ color: var(--subtle); }}
.status-badge {{ display: inline-block; padding: var(--space-1) var(--space-2); border: 1px solid var(--rule-strong); color: var(--muted); font: 700 .6875rem/1.4 "SFMono-Regular", Consolas, monospace; }}
.status-frozen {{ border-color: var(--positive); color: var(--positive); }}
.status-exploratory {{ border-color: var(--heuristic); color: var(--heuristic); }}
.subsection-title {{ margin: var(--space-8) 0 var(--space-4); font-family: "Iowan Old Style", "Palatino Linotype", Palatino, serif; font-size: 1.5rem; }}
details {{ margin-top: var(--space-6); border-block: 1px solid var(--rule); }}
summary {{ display: flex; justify-content: space-between; gap: var(--space-4); padding-block: var(--space-5); color: var(--muted); cursor: pointer; font-weight: 700; list-style: none; }}
summary::-webkit-details-marker {{ display: none; }}
summary::after {{ content: "+"; color: var(--learned); font: 700 1rem/1 "SFMono-Regular", Consolas, monospace; }}
details[open] summary::after {{ content: "−"; }}
details > .table-shell, details > pre {{ margin-bottom: var(--space-5); }}
.method-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-block: 1px solid var(--rule); }}
.method-block {{ min-width: 0; padding: var(--space-6); border-right: 1px solid var(--rule); }}
.method-block:first-child {{ padding-left: 0; }}
.method-block:last-child {{ padding-right: 0; border-right: 0; }}
.method-block h3 {{ margin-bottom: var(--space-4); font-size: 1rem; }}
.method-block p {{ color: var(--muted); font-size: .8125rem; }}
.method-block p:last-child {{ margin-bottom: 0; }}
.method-list, .provenance-list {{ margin: 0; }}
.method-list div, .provenance-list div {{ padding-block: var(--space-3); border-bottom: 1px solid var(--rule); }}
.method-list div:last-child, .provenance-list div:last-child {{ border-bottom: 0; }}
.method-list dt, .provenance-list dt {{ margin-bottom: var(--space-1); color: var(--subtle); font-size: .6875rem; letter-spacing: .06em; text-transform: uppercase; }}
.method-list dd, .provenance-list dd {{ margin: 0; color: var(--text); font-size: .8125rem; overflow-wrap: anywhere; }}
.formula {{ color: var(--learned-strong); font-family: "SFMono-Regular", Consolas, monospace; }}
.provenance-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-7); margin-top: var(--space-8); }}
.provenance-block {{ min-width: 0; border-top: 1px solid var(--rule-strong); padding-top: var(--space-5); }}
.provenance-block h3 {{ margin-bottom: var(--space-4); font-size: 1rem; }}
code, pre {{ font-family: "SFMono-Regular", Consolas, monospace; }}
.hash {{ display: block; color: var(--muted); overflow-wrap: anywhere; white-space: normal; user-select: text; }}
pre {{ max-width: 100%; margin: 0; padding: var(--space-4); overflow: auto; background: var(--surface); border: 1px solid var(--rule); color: var(--muted); font-size: .75rem; line-height: 1.65; }}
.report-footer {{ display: flex; justify-content: space-between; gap: var(--space-5); padding-block: var(--space-6) var(--space-8); color: var(--subtle); font: 600 .6875rem/1.6 "SFMono-Regular", Consolas, monospace; }}
@media (max-width: 64rem) {{
  .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .metric-card:nth-child(2) {{ border-right: 0; }}
  .metric-card:nth-child(-n+2) {{ border-bottom: 1px solid var(--rule); }}
  .method-grid {{ grid-template-columns: 1fr; }}
  .method-block, .method-block:first-child, .method-block:last-child {{ padding: var(--space-5) 0; border-right: 0; border-bottom: 1px solid var(--rule); }}
  .method-block:last-child {{ border-bottom: 0; }}
}}
@media (max-width: 48rem) {{
  .page-shell {{ width: min(calc(100% - 1.5rem), 78rem); }}
  .topline, .report-footer {{ flex-direction: column; }}
  .hero-grid, .section-heading, .dial-panel, .histogram, .provenance-grid {{ grid-template-columns: 1fr; }}
  .hero-grid {{ gap: var(--space-6); padding-top: var(--space-7); }}
  .evidence-stamp {{ border-left: 0; border-top: 1px solid var(--rule); padding: var(--space-5) 0 0; }}
  .report-section {{ padding-block: var(--space-8); }}
  .dial-panel {{ gap: var(--space-5); padding: var(--space-5); }}
  .selection-meta {{ border-left: 0; border-top: 1px solid var(--rule); padding-top: var(--space-3); }}
  .selection-meta div {{ padding-left: 0; }}
  .status-line, .metric-card, .interpretation, .histogram {{ padding-inline: var(--space-5); }}
  figcaption {{ grid-template-columns: 1fr; gap: var(--space-3); }}
}}
@media (max-width: 28rem) {{
  .metric-grid {{ grid-template-columns: 1fr; }}
  .metric-card, .metric-card:nth-child(2) {{ border-right: 0; border-bottom: 1px solid var(--rule); }}
  .metric-card:last-child {{ border-bottom: 0; }}
  .dial-scale span:nth-child(even) {{ display: none; }}
  .selection-meta div {{ grid-template-columns: 5.5rem minmax(0, 1fr); }}
  .stop-row {{ grid-template-columns: 1.5rem minmax(0, 1fr) 3.5rem; gap: var(--space-2); }}
}}
@media (prefers-reduced-motion: reduce) {{
  html {{ scroll-behavior: auto; }}
  *, *::before, *::after {{ scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; }}
}}
@media (prefers-reduced-motion: no-preference) {{
  .skip-link {{ transition: transform 160ms cubic-bezier(.16, 1, .3, 1); }}
  .stop-fill {{ transition: width 180ms cubic-bezier(.16, 1, .3, 1); }}
}}
"""


def _render_html(payload: dict[str, Any], svg: str) -> str:
    entries = _interactive_entries(payload)
    initial = entries[0]
    learned_rows = sorted(
        (row for row in payload["rows"] if str(row["family"]) == "offline-rl"),
        key=lambda row: float(row["scoring_cost"]),
    )
    all_rows = sorted(
        payload["rows"],
        key=lambda row: (
            float(row["scoring_cost"]),
            {"offline-rl": 0, "fixed": 1, "heuristic": 2}.get(str(row["family"]), 3),
            float(row["average_samples"]),
            str(row["policy"]),
        ),
    )
    comparisons = sorted(payload["comparisons"], key=lambda item: float(item["scoring_cost"]))
    selections = sorted({str(item["selection"]) for item in comparisons})
    all_frozen = all(selection == "validation-frozen" for selection in selections)
    any_frozen = any(selection == "validation-frozen" for selection in selections)
    if all_frozen:
        comparison_scope = "Validation-frozen paired comparisons"
        comparison_note = (
            "Every named comparator was fixed independently of these evaluation outcomes."
        )
    elif any_frozen:
        comparison_scope = "Mixed comparator-selection protocol"
        comparison_note = "Some costs use validation-frozen comparators; others remain exploratory. Read each cost status."
    else:
        comparison_scope = "Exploratory paired comparisons"
        comparison_note = "Comparators were selected from these evaluation results; findings are hypothesis-generating."

    protocol = payload.get("protocol")
    if protocol is None:
        protocol_stamp = "No separate protocol artifact supplied"
        protocol_note = comparison_note
        protocol_provenance = (
            "<div><dt>Protocol artifact</dt><dd>Not supplied</dd></div>"
            f"<div><dt>Comparator selection</dt><dd>{_escape(', '.join(selections))}</dd></div>"
        )
    else:
        protocol_stamp = f"{protocol['evidence_tier']} · {protocol['status']}"
        protocol_note = f"{comparison_scope}. Protocol artifact: {protocol['path']}."
        protocol_provenance = (
            f"<div><dt>Protocol path</dt><dd>{_escape(protocol['path'])}</dd></div>"
            f"<div><dt>Protocol status</dt><dd>{_escape(protocol['status'])}</dd></div>"
            f"<div><dt>Evidence tier</dt><dd>{_escape(protocol['evidence_tier'])}</dd></div>"
            f'<div><dt>Protocol SHA-256</dt><dd><code class="hash">{_escape(protocol["sha256"])}</code></dd></div>'
            f"<div><dt>Comparator selection</dt><dd>{_escape(', '.join(selections))}</dd></div>"
        )

    tick_labels = "".join(f"<span>{_fmt_cost(cost)}</span>" for cost in payload["costs"])
    datalist_options = "".join(
        f'<option value="{index}" label="{_fmt_cost(cost)}"></option>'
        for index, cost in enumerate(payload["costs"])
    )
    features = ", ".join(str(feature) for feature in payload["policy"]["feature_names"])
    objective = payload["objective"]
    bootstrap = payload["bootstrap"]
    data = payload["data"]
    policy = payload["policy"]
    interactive_json = _script_json(entries)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="description" content="Standalone BranchPilot held-out benchmark evidence with paired-bootstrap uncertainty.">
<title>BranchPilot · held-out policy evidence</title>
<style>{_report_styles()}</style>
</head>
<body>
<a class="skip-link" href="#report-content">Skip to evidence</a>
<div class="page-shell">
<header class="hero">
  <div class="topline">
    <span>BranchPilot / benchmark schema v2</span>
    <span>{int(payload["records"]):,} held-out records · {len(payload["costs"])} evaluated costs</span>
  </div>
  <div class="hero-grid">
    <div>
      <p class="kicker">Cost-conditioned stopping policy</p>
      <h1>Held-out policy evidence,<br><span>cost by cost.</span></h1>
      <p class="lede">A standalone account of learned, fixed-sample, and heuristic stopping policies. Point estimates are shown with 95% bootstrap intervals; paired deltas always name the comparator and its selection method.</p>
    </div>
    <aside class="evidence-stamp" aria-label="Evidence status">
      <span class="eyebrow">Evidence status</span>
      <strong>{_escape(comparison_scope)}</strong>
      <p>{_escape(protocol_stamp)}. {_escape(protocol_note)}</p>
    </aside>
  </div>
</header>
<main id="report-content">
  <section class="report-section" aria-labelledby="console-title">
    <div class="section-heading">
      <div>
        <span class="section-index">01 / EVIDENCE CONSOLE</span>
        <h2 id="console-title">One evaluated cost at a time</h2>
      </div>
      <p>The dial moves only across costs present in the benchmark. Use the arrow keys after focusing it. Every estimate, interval, comparator, interpretation, and stop-count bar updates from the same selected row.</p>
    </div>
    <div class="console">
      <div class="dial-panel">
        <div>
          <label class="dial-label" for="cost-dial"><span>Scoring cost λ</span><output id="cost-output" for="cost-dial">{_escape(initial["cost"])}</output></label>
          <input id="cost-dial" type="range" min="0" max="{len(entries) - 1}" step="1" value="0" list="cost-options" aria-describedby="dial-help" aria-valuetext="λ {_escape(initial["cost"])}">
          <datalist id="cost-options">{datalist_options}</datalist>
          <div class="dial-scale" aria-hidden="true">{tick_labels}</div>
          <p id="dial-help" class="table-intro">Only measured settings are selectable; no values are interpolated.</p>
        </div>
        <dl class="selection-meta">
          <div><dt>Learned policy</dt><dd id="selected-policy">{_escape(initial["policy"])}</dd></div>
          <div><dt>Comparator</dt><dd id="selected-comparator">{_escape(initial["comparator"])}</dd></div>
          <div><dt>Selection</dt><dd id="selected-method">{_escape(initial["selection"])}</dd></div>
        </dl>
      </div>
      <div class="status-line" id="selected-status" data-tone="{_escape(initial["tone"])}">{_escape(initial["status"])}</div>
      <div class="metric-grid">
        <article class="metric-card">
          <h3>Learned accuracy</h3>
          <strong class="metric-value" id="accuracy-value">{_escape(initial["accuracy"]["value"])}</strong>
          <span class="metric-interval" id="accuracy-interval">{_escape(initial["accuracy"]["interval"])}</span>
        </article>
        <article class="metric-card">
          <h3>Average samples</h3>
          <strong class="metric-value" id="samples-value">{_escape(initial["samples"]["value"])}</strong>
          <span class="metric-interval" id="samples-interval">{_escape(initial["samples"]["interval"])}</span>
        </article>
        <article class="metric-card">
          <h3>Completion tokens</h3>
          <strong class="metric-value" id="tokens-value">{_escape(initial["tokens"]["value"])}</strong>
          <span class="metric-interval" id="tokens-interval">{_escape(initial["tokens"]["interval"])}</span>
        </article>
        <article class="metric-card">
          <h3>Paired utility delta</h3>
          <strong class="metric-value" id="utility-value">{_escape(initial["utility"]["value"])}</strong>
          <span class="metric-interval" id="utility-interval">{_escape(initial["utility"]["interval"])}</span>
        </article>
      </div>
      <div class="interpretation">
        <span class="eyebrow">What this interval supports</span>
        <p id="evidence-conclusion" aria-live="polite">{_escape(initial["conclusion"])}</p>
      </div>
      <div class="histogram">
        <div class="histogram-copy">
          <h3 id="histogram-title">Selected-policy stop counts</h3>
          <p id="histogram-policy">{_escape(initial["policy"])} · count of records stopping after each sample number.</p>
        </div>
        <ol class="stop-list" id="stop-list" aria-labelledby="histogram-title">{_histogram_rows(initial["histogram"])}</ol>
      </div>
      <noscript><p class="no-script">JavaScript is disabled. The console shows λ={_escape(initial["cost"])}; the complete learned and paired-comparison tables below remain available for every evaluated cost.</p></noscript>
    </div>
  </section>

  <section class="report-section" aria-labelledby="frontier-title">
    <div class="section-heading">
      <div>
        <span class="section-index">02 / FRONTIER</span>
        <h2 id="frontier-title">Accuracy–compute frontier</h2>
      </div>
      <p>Every unique measured policy is shown with horizontal and vertical 95% intervals. The solid line connects only the supplied empirical Pareto frontier; it does not imply interpolation between policies.</p>
    </div>
    <figure>
      <div class="figure-shell">
        <div class="chart-viewport">{svg}</div>
        <figcaption>
          <span><strong>Figure 1.</strong> Exact-match accuracy against average samples per held-out record. Upper-left operating points use less inference compute at higher measured accuracy.</span>
          <span><strong>Uncertainty.</strong> Bars are marginal 95% paired-bootstrap intervals for each metric. Overlap is not a paired significance test; use the named paired utility delta for the direct comparison.</span>
        </figcaption>
      </div>
    </figure>
  </section>

  <section class="report-section" aria-labelledby="tables-title">
    <div class="section-heading">
      <div>
        <span class="section-index">03 / COMPLETE RESULTS</span>
        <h2 id="tables-title">Measured rows, without omissions</h2>
      </div>
      <p>Point estimates and 95% intervals are kept together. Wide tables scroll horizontally on small screens; stop histograms list bins in sample-count order from one through {int(payload["max_samples"])}.</p>
    </div>
    <h3 class="subsection-title">Learned policy sweep</h3>
    <p class="table-intro">One learned operating point per evaluated cost.</p>
    <div class="table-shell">
      <table aria-label="Complete learned policy results">
        <thead><tr><th>Family</th><th>Policy</th><th class="numeric">λ</th><th class="numeric">Accuracy</th><th class="numeric">Avg. samples</th><th class="numeric">Avg. tokens</th><th class="numeric">P50 samples</th><th class="numeric">P90 samples</th><th class="numeric">Utility</th><th>Stop counts 1…{int(payload["max_samples"])}</th></tr></thead>
        <tbody>{_render_metric_rows(learned_rows)}</tbody>
      </table>
    </div>

    <h3 class="subsection-title">Paired learned–comparator evidence</h3>
    <p class="table-intro">Deltas are learned minus the named comparator. Selection status is repeated on every row.</p>
    <div class="table-shell">
      <table aria-label="Complete paired comparison results">
        <thead><tr><th class="numeric">λ</th><th>Learned policy</th><th>Named comparator</th><th>Selection</th><th class="numeric">Accuracy Δ</th><th class="numeric">Avg. samples Δ</th><th class="numeric">Avg. tokens Δ</th><th class="numeric">Utility Δ</th></tr></thead>
        <tbody>{_render_comparison_rows(comparisons)}</tbody>
      </table>
    </div>

    <details>
      <summary><span>Full learned, fixed, and heuristic matrix</span><span>{len(all_rows)} scored rows</span></summary>
      <div class="table-shell">
        <table aria-label="Full benchmark matrix">
          <thead><tr><th>Family</th><th>Policy</th><th class="numeric">λ</th><th class="numeric">Accuracy</th><th class="numeric">Avg. samples</th><th class="numeric">Avg. tokens</th><th class="numeric">P50 samples</th><th class="numeric">P90 samples</th><th class="numeric">Utility</th><th>Stop counts 1…{int(payload["max_samples"])}</th></tr></thead>
          <tbody>{_render_metric_rows(all_rows)}</tbody>
        </table>
      </div>
    </details>
  </section>

  <section class="report-section" aria-labelledby="method-title">
    <div class="section-heading">
      <div>
        <span class="section-index">04 / METHOD & PROVENANCE</span>
        <h2 id="method-title">What produced this evidence</h2>
      </div>
      <p>Objective, uncertainty, comparator protocol, schemas, hashes, features, and training metadata are visible as selectable text. Nothing methodological is hidden behind a tooltip.</p>
    </div>
    <div class="method-grid">
      <article class="method-block">
        <h3>Objective</h3>
        <p>{_escape(objective["name"])}</p>
        <dl class="method-list">
          <div><dt>Formula</dt><dd><code class="formula">{_escape(objective["formula"])}</code></dd></div>
          <div><dt>Cost unit</dt><dd>{_escape(objective["cost_unit"])}</dd></div>
          <div><dt>Evaluated λ</dt><dd>{_escape(", ".join(_fmt_cost(cost) for cost in payload["costs"]))}</dd></div>
        </dl>
      </article>
      <article class="method-block">
        <h3>Bootstrap uncertainty</h3>
        <p>Intervals are generated from shared resample indices. Comparison intervals are paired learned-minus-comparator deltas on the same held-out records.</p>
        <dl class="method-list">
          <div><dt>Confidence</dt><dd>{float(bootstrap["confidence"]):.0%}</dd></div>
          <div><dt>Resamples</dt><dd>{int(bootstrap["resamples"]):,}</dd></div>
          <div><dt>Seed</dt><dd>{int(bootstrap["seed"])}</dd></div>
          <div><dt>Records</dt><dd>{int(payload["records"]):,}</dd></div>
        </dl>
      </article>
      <article class="method-block">
        <h3>Comparator protocol</h3>
        <p>{_escape(comparison_note)}</p>
        <dl class="method-list">
          <div><dt>Evidence status</dt><dd>{_escape(comparison_scope)}</dd></div>
          <div><dt>Selection method</dt><dd>{_escape(", ".join(selections))}</dd></div>
          <div><dt>Protocol status / tier</dt><dd>{_escape(protocol_stamp)}</dd></div>
        </dl>
      </article>
    </div>

    <div class="provenance-grid">
      <article class="provenance-block">
        <h3>Evaluation data</h3>
        <dl class="provenance-list">
          <div><dt>Path</dt><dd>{_escape(data["path"])}</dd></div>
          <div><dt>SHA-256</dt><dd><code class="hash">{_escape(data["sha256"])}</code></dd></div>
          <div><dt>Data schema</dt><dd>{_escape(data["schema_version"])}</dd></div>
          <div><dt>Benchmark schema</dt><dd>2</dd></div>
          <div><dt>Records</dt><dd>{int(payload["records"]):,}</dd></div>
          {protocol_provenance}
        </dl>
        <details>
          <summary><span>Dataset profile</span><span>copyable JSON</span></summary>
          <pre>{_pretty_json(data["profile"])}</pre>
        </details>
      </article>
      <article class="provenance-block">
        <h3>Learned policy artifact</h3>
        <dl class="provenance-list">
          <div><dt>Path</dt><dd>{_escape(policy["path"])}</dd></div>
          <div><dt>SHA-256</dt><dd><code class="hash">{_escape(policy["sha256"])}</code></dd></div>
          <div><dt>Policy artifact schema</dt><dd>{_escape(policy["artifact_version"])}</dd></div>
          <div><dt>Training algorithm</dt><dd>{_escape(policy["training"]["algorithm"])}</dd></div>
          <div><dt>Features</dt><dd>{_escape(features)}</dd></div>
        </dl>
        <details>
          <summary><span>Training metadata</span><span>copyable JSON</span></summary>
          <pre>{_pretty_json(policy["training"])}</pre>
        </details>
      </article>
    </div>
  </section>
</main>
<footer class="report-footer">
  <span>BranchPilot standalone evidence artifact · no external assets or requests</span>
  <a href="https://github.com/mottopanikeiku/branchpilot">BranchPilot repository</a>
</footer>
</div>
<script id="evidence-data" type="application/json">{interactive_json}</script>
<script>
(() => {{
  const dial = document.getElementById("cost-dial");
  const dataNode = document.getElementById("evidence-data");
  if (!dial || !dataNode) return;
  const evidence = JSON.parse(dataNode.textContent);
  const byId = (id) => document.getElementById(id);
  const put = (id, value) => {{ const node = byId(id); if (node) node.textContent = value; }};
  const stopList = byId("stop-list");

  function renderStops(counts) {{
    if (!stopList) return;
    const total = counts.reduce((sum, count) => sum + count, 0);
    stopList.replaceChildren(...counts.map((count, index) => {{
      const percentage = total ? count / total * 100 : 0;
      const row = document.createElement("li");
      row.className = "stop-row";
      row.setAttribute("aria-label", `Stopped after ${{index + 1}} samples: ${{count}} records, ${{percentage.toFixed(1)}} percent`);
      const bin = document.createElement("span");
      bin.className = "stop-bin";
      bin.textContent = String(index + 1);
      const track = document.createElement("span");
      track.className = "stop-track";
      track.setAttribute("aria-hidden", "true");
      const fill = document.createElement("span");
      fill.className = "stop-fill";
      fill.style.setProperty("--share", `${{percentage.toFixed(4)}}%`);
      track.append(fill);
      const value = document.createElement("span");
      value.className = "stop-count";
      value.textContent = count.toLocaleString("en-US");
      row.append(bin, track, value);
      return row;
    }}));
  }}

  function update() {{
    const selected = evidence[Number(dial.value)];
    if (!selected) return;
    put("cost-output", selected.cost);
    put("selected-policy", selected.policy);
    put("selected-comparator", selected.comparator);
    put("selected-method", selected.selection);
    put("selected-status", selected.status);
    put("accuracy-value", selected.accuracy.value);
    put("accuracy-interval", selected.accuracy.interval);
    put("samples-value", selected.samples.value);
    put("samples-interval", selected.samples.interval);
    put("tokens-value", selected.tokens.value);
    put("tokens-interval", selected.tokens.interval);
    put("utility-value", selected.utility.value);
    put("utility-interval", selected.utility.interval);
    put("evidence-conclusion", selected.conclusion);
    put("histogram-policy", `${{selected.policy}} · count of records stopping after each sample number.`);
    const status = byId("selected-status");
    if (status) status.dataset.tone = selected.tone;
    dial.setAttribute("aria-valuetext", `λ ${{selected.cost}}`);
    renderStops(selected.histogram);
  }}

  dial.addEventListener("input", update);
}})();
</script>
</body>
</html>
"""


def write_report(
    benchmark_path: str | Path,
    svg_path: str | Path | None = None,
    html_path: str | Path | None = None,
) -> None:
    payload = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    validated = _validate_payload(payload)
    svg = _render_svg(validated, "Accuracy × inference compute")
    if svg_path is not None:
        atomic_write_text(svg_path, svg)
    if html_path is not None:
        atomic_write_text(html_path, _render_html(validated, svg))
