"""Static SVG charts for the P3 report: no plotting dependency, every value from its caller."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence
from html import escape
from math import ceil, floor, log10

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3; every pair validates
SERIES_DARK = ("#3987e5", "#d95926", "#199e70")
NEUTRAL = "#8a8983"


class Series:
    def __init__(
        self,
        label: str,
        x: Sequence[float],
        y: Sequence[float | None],
        low: Sequence[float | None] | None = None,
        high: Sequence[float | None] | None = None,
        *,
        neutral: bool = False,
    ) -> None:
        self.label, self.x, self.y = label, list(x), list(y)
        self.low, self.high, self.neutral = low, high, neutral


WIDTH, HEIGHT = 760, 420
LEFT, RIGHT, TOP, BOTTOM = 72, 24, 112, 56
SUBTITLE_WIDTH = 100


def _frame(title: str, subtitle: str, x_label: str, y_label: str) -> list[str]:
    swatch_rules = "".join(f".s{i}{{stroke:{c};fill:{c}}}" for i, c in enumerate(SERIES))
    dark_rules = "".join(f".s{i}{{stroke:{c};fill:{c}}}" for i, c in enumerate(SERIES_DARK))
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-label="{escape(title)}">',
        "<style>"
        ".bg{fill:#fcfcfb}.t{fill:#0b0b0b}.m{fill:#52514e}.g{stroke:#e4e3de;stroke-width:1}"
        "text{font:12px system-ui,sans-serif}.h{font-size:15px;font-weight:600}"
        f"{swatch_rules}.n{{stroke:{NEUTRAL};fill:{NEUTRAL}}}"
        ".l{fill:none!important;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
        ".b{stroke:none!important;fill-opacity:.1}.r{stroke:#fcfcfb!important;stroke-width:2}"
        "@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}.t{fill:#f0efec}.m{fill:#c3c2b7}"
        f".g{{stroke:#3a3a37}}{dark_rules}.r{{stroke:#1a1a19!important}}}}"
        "</style>",
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}"/>',
        f'<text class="t h" x="{LEFT - 56}" y="26">{escape(title)}</text>',
        *[
            f'<text class="m" x="{LEFT - 56}" y="{46 + 16 * i}">{escape(line)}</text>'
            for i, line in enumerate(textwrap.wrap(subtitle, SUBTITLE_WIDTH))
        ],
        f'<text class="m" x="{(LEFT + WIDTH - RIGHT) / 2}" y="{HEIGHT - 12}" '
        f'text-anchor="middle">{escape(x_label)}</text>',
        f'<text class="m" transform="translate(16 {(TOP + HEIGHT - BOTTOM) / 2}) rotate(-90)" '
        f'text-anchor="middle">{escape(y_label)}</text>',
    ]


def _legend(labels: Sequence[tuple[str, str]]) -> list[str]:
    parts, x = [], LEFT
    for label, css in labels:
        parts.append(f'<rect class="{css}" x="{x}" y="84" width="14" height="4" rx="2"/>')
        parts.append(f'<text class="m" x="{x + 20}" y="90">{escape(label)}</text>')
        x += 34 + 7 * len(label)
    return parts


def _ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round ticks that enclose ``[low, high]``."""
    raw = (high - low or 1.0) / count
    magnitude = 10.0 ** floor(log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return [
        round(index * step, 10)
        for index in range(floor(low / step + 1e-9), ceil(high / step - 1e-9) + 1)
    ]


def _label(value: float) -> str:
    return f"{value:g}"


def lines(
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    *,
    log_x: bool = False,
    markers: bool = False,
) -> str:
    def finite(values: Sequence[float | None] | None) -> list[float]:
        return [v for v in (values or []) if v is not None]

    ys = [v for s in series for v in finite(s.y) + finite(s.low) + finite(s.high)] or [0.0, 1.0]
    xs = [float(v) for s in series for v in s.x]
    y_ticks = _ticks(min(*ys, 0.0), max(ys))
    y_low, y_high = y_ticks[0], y_ticks[-1]
    x_low, x_high = min(xs), max(xs)

    def px(value: float) -> float:
        if log_x:
            share = (log10(value) - log10(x_low)) / ((log10(x_high) - log10(x_low)) or 1)
        else:
            share = (value - x_low) / ((x_high - x_low) or 1)
        return round(LEFT + share * (WIDTH - LEFT - RIGHT), 2)

    def py(value: float) -> float:
        share = (value - y_low) / ((y_high - y_low) or 1)
        return round(HEIGHT - BOTTOM - share * (HEIGHT - TOP - BOTTOM), 2)

    parts = _frame(title, subtitle, x_label, y_label)
    for tick in y_ticks:
        parts.append(f'<line class="g" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{py(tick)}" '
                     f'y2="{py(tick)}"/>')  # fmt: skip
        parts.append(f'<text class="m" x="{LEFT - 8}" y="{py(tick) + 4}" '
                     f'text-anchor="end">{_label(tick)}</text>')  # fmt: skip
    x_ticks = [float(v) for v in (1, 10, 100, 1024)] if log_x else _ticks(x_low, x_high)
    x_ticks = [v for v in x_ticks if x_low <= v <= x_high]
    for tick in x_ticks:
        parts.append(f'<text class="m" x="{px(float(tick))}" y="{HEIGHT - BOTTOM + 18}" '
                     f'text-anchor="middle">{_label(float(tick))}</text>')  # fmt: skip
    legend = []
    slot = 0
    for line in series:
        css = "n" if line.neutral else f"s{slot}"
        slot += 0 if line.neutral else 1
        legend.append((line.label, css))
        rows = [(x, y) for x, y in zip(line.x, line.y, strict=True) if y is not None]
        if line.low is not None and line.high is not None:
            edge = [
                (x, lo, hi)
                for x, lo, hi in zip(line.x, line.low, line.high, strict=True)
                if lo is not None and hi is not None
            ]
            points = [f"{px(x)},{py(hi)}" for x, _, hi in edge]
            points += [f"{px(x)},{py(lo)}" for x, lo, _ in reversed(edge)]
            parts.append(f'<polygon class="{css} b" points="{" ".join(points)}"/>')
        path = " ".join(f"{px(x)},{py(y)}" for x, y in rows)
        parts.append(f'<polyline class="{css} l" points="{path}"/>')
        if markers:
            for x, y in rows:
                parts.append(
                    f'<circle class="{css}" cx="{px(x)}" cy="{py(y)}" r="4"><title>'
                    f"{escape(line.label)}: {_label(x)}, {y:.3f}</title></circle>"
                )
    if len(series) > 1:
        parts += _legend(legend)
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def bars(
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    categories: Sequence[str],
    values: dict[str, list[float | None]],
) -> str:
    known = [v for row in values.values() for v in row if v is not None]
    ticks = _ticks(0.0, max(known) if known else 1.0)
    high = ticks[-1]

    def py(value: float) -> float:
        share = value / (high or 1)
        return round(HEIGHT - BOTTOM - share * (HEIGHT - TOP - BOTTOM), 2)

    parts = _frame(title, subtitle, x_label, y_label)
    for tick in ticks:
        parts.append(f'<line class="g" x1="{LEFT}" x2="{WIDTH - RIGHT}" y1="{py(tick)}" '
                     f'y2="{py(tick)}"/>')  # fmt: skip
        parts.append(f'<text class="m" x="{LEFT - 8}" y="{py(tick) + 4}" '
                     f'text-anchor="end">{_label(tick)}</text>')  # fmt: skip
    band = (WIDTH - LEFT - RIGHT) / max(len(categories), 1)
    names = list(values)
    bar, gap = 24, 2
    for c, category in enumerate(categories):
        center = LEFT + band * (c + 0.5)
        parts.append(f'<text class="m" x="{round(center, 2)}" y="{HEIGHT - BOTTOM + 18}" '
                     f'text-anchor="middle">{escape(category)}</text>')  # fmt: skip
        start = center - (len(names) * bar + (len(names) - 1) * gap) / 2
        for s, name in enumerate(names):
            value = values[name][c]
            if value is None:
                continue
            x, top = round(start + s * (bar + gap), 2), py(value)
            base = py(0.0)
            radius = min(4.0, max(base - top, 0.0))
            path = (
                f"M{x} {base}V{top + radius}Q{x} {top} {x + radius} {top}"
                f"H{x + bar - radius}Q{x + bar} {top} {x + bar} {top + radius}V{base}Z"
            )
            parts.append(
                f'<path class="s{s}" d="{path}"><title>{escape(name)}, {escape(category)}: '
                f"{value:.3f}</title></path>"
            )
    if len(names) > 1:
        parts += _legend([(name, f"s{i}") for i, name in enumerate(names)])
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
