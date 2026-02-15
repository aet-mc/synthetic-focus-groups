from __future__ import annotations

import math
from html import escape


PALETTE = {
    "blue": "#2563EB",
    "green": "#10B981",
    "red": "#EF4444",
    "yellow": "#F59E0B",
    "gray": "#6B7280",
    "light": "#E5E7EB",
    "text": "#111827",
}


class ChartGenerator:
    """Generate inline SVG charts for the report. All methods return SVG strings."""

    @staticmethod
    def horizontal_bar_chart(
        labels: list[str],
        values: list[float],
        max_value: float = 1.0,
        title: str = "",
        width: int = 600,
        height: int | None = None,
        colors: list[str] | None = None,
        show_values: bool = True,
        value_format: str = "{:.0%}",
    ) -> str:
        labels = labels or []
        values = values or []
        n = min(len(labels), len(values))
        labels = labels[:n]
        values = values[:n]

        top = 40 if title else 18
        bar_h = 22
        bar_gap = 14
        inner_h = n * (bar_h + bar_gap) + 8
        chart_h = height if height is not None else top + inner_h + 16

        left_pad = 190
        right_pad = 80
        bar_x = left_pad
        bar_w = max(120, width - left_pad - right_pad)

        palette = colors or ["#93C5FD", "#60A5FA", "#3B82F6", "#2563EB", "#1D4ED8", "#1E40AF"]

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_h}" viewBox="0 0 {width} {chart_h}" role="img">',
            '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#111827}</style>',
        ]
        if title:
            parts.append(
                f'<text x="14" y="24" font-size="16" font-weight="700">{escape(title)}</text>'
            )

        for i, (label, raw_value) in enumerate(zip(labels, values)):
            value = max(0.0, min(float(raw_value), max_value if max_value > 0 else 1.0))
            y = top + i * (bar_h + bar_gap)
            fill = palette[i % len(palette)]
            filled_w = (value / max_value) * bar_w if max_value > 0 else 0

            parts.append(
                f'<text x="14" y="{y + 15}" font-size="12" font-weight="600">{escape(str(label))}</text>'
            )
            parts.append(
                f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="4" fill="{PALETTE["light"]}"/>'
            )
            parts.append(
                f'<rect x="{bar_x}" y="{y}" width="{filled_w:.2f}" height="{bar_h}" rx="4" fill="{fill}"/>'
            )
            if show_values:
                parts.append(
                    f'<text x="{bar_x + bar_w + 10}" y="{y + 15}" font-size="12" font-weight="700">{escape(value_format.format(value))}</text>'
                )

        if n == 0:
            parts.append('<text x="14" y="40" font-size="12" fill="#6B7280">No data available</text>')

        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def sentiment_line_chart(
        phases: list[str],
        values: list[float],
        title: str = "Sentiment by Phase",
        width: int = 600,
        height: int = 250,
    ) -> str:
        n = min(len(phases), len(values))
        phases = phases[:n]
        values = [max(-1.0, min(1.0, float(v))) for v in values[:n]]

        left, right, top, bottom = 52, 20, 38, 48
        plot_w = max(100, width - left - right)
        plot_h = max(80, height - top - bottom)

        def x_at(i: int) -> float:
            if n <= 1:
                return left + plot_w / 2
            return left + (plot_w * i / (n - 1))

        def y_at(v: float) -> float:
            return top + ((1 - v) / 2) * plot_h

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#111827}</style>',
            f'<text x="14" y="24" font-size="16" font-weight="700">{escape(title)}</text>',
        ]

        for tick in (-1.0, -0.5, 0.0, 0.5, 1.0):
            y = y_at(tick)
            stroke = PALETTE["gray"] if tick != 0 else PALETTE["yellow"]
            width_tick = 1 if tick != 0 else 2
            opacity = 0.2 if tick != 0 else 0.8
            parts.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="{stroke}" stroke-width="{width_tick}" opacity="{opacity}"/>'
            )
            parts.append(
                f'<text x="{left - 30}" y="{y + 4:.2f}" font-size="11" fill="#6B7280">{tick:.1f}</text>'
            )

        for i in range(max(0, n - 1)):
            x1, y1 = x_at(i), y_at(values[i])
            x2, y2 = x_at(i + 1), y_at(values[i + 1])
            avg = (values[i] + values[i + 1]) / 2
            color = PALETTE["green"] if avg >= 0 else PALETTE["red"]
            parts.append(
                f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
            )

        for i in range(n):
            x = x_at(i)
            y = y_at(values[i])
            color = PALETTE["green"] if values[i] >= 0 else PALETTE["red"]
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{color}"/>')
            parts.append(
                f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle" font-size="11">{escape(str(phases[i]).replace("_", " ").title())}</text>'
            )

        if n == 0:
            parts.append('<text x="14" y="52" font-size="12" fill="#6B7280">No sentiment data available</text>')

        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def donut_chart(
        labels: list[str],
        values: list[float],
        title: str = "",
        width: int = 250,
        height: int = 250,
        colors: list[str] | None = None,
    ) -> str:
        n = min(len(labels), len(values))
        labels = labels[:n]
        values = [max(0.0, float(v)) for v in values[:n]]
        total = sum(values)

        palette = colors or [PALETTE["blue"], PALETTE["green"], PALETTE["yellow"], PALETTE["red"], PALETTE["gray"]]
        cx = width / 2
        cy = (height / 2) - (12 if title else 0)
        radius = min(width, height - (24 if title else 0)) * 0.34
        inner = radius * 0.58

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#111827}</style>',
        ]
        if title:
            parts.append(
                f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-size="14" font-weight="700">{escape(title)}</text>'
            )

        if total <= 0:
            parts.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" fill="none" stroke="{PALETTE["light"]}" stroke-width="{radius - inner:.2f}"/>'
            )
            parts.append(f'<text x="{cx:.1f}" y="{cy + 5:.1f}" text-anchor="middle" font-size="11">No data</text>')
            parts.append("</svg>")
            return "".join(parts)

        start = -math.pi / 2
        ring = max(8.0, radius - inner)

        def pt(angle: float, r: float) -> tuple[float, float]:
            return cx + r * math.cos(angle), cy + r * math.sin(angle)

        for i, (label, value) in enumerate(zip(labels, values)):
            if value <= 0:
                continue
            sweep = (value / total) * (2 * math.pi)
            end = start + sweep
            x1, y1 = pt(start, (inner + radius) / 2)
            x2, y2 = pt(end, (inner + radius) / 2)
            large_arc = 1 if sweep > math.pi else 0
            color = palette[i % len(palette)]
            parts.append(
                f'<path d="M{x1:.2f},{y1:.2f} A{((inner + radius) / 2):.2f},{((inner + radius) / 2):.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f}" fill="none" stroke="{color}" stroke-width="{ring:.2f}" stroke-linecap="butt"/>'
            )
            start = end

        parts.append(f'<text x="{cx:.1f}" y="{cy - 2:.1f}" text-anchor="middle" font-size="16" font-weight="700">{total:.0f}</text>')
        parts.append(f'<text x="{cx:.1f}" y="{cy + 14:.1f}" text-anchor="middle" font-size="11" fill="#6B7280">Responses</text>')

        ly = height - (14 * len(labels)) - 8
        for i, (label, value) in enumerate(zip(labels, values)):
            y = ly + (i * 14)
            color = palette[i % len(palette)]
            pct = (value / total) if total else 0
            parts.append(f'<rect x="10" y="{y - 8}" width="9" height="9" rx="2" fill="{color}"/>')
            parts.append(
                f'<text x="24" y="{y}" font-size="10">{escape(str(label))} ({pct:.0%})</text>'
            )

        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def score_gauge(
        value: float,
        max_value: float = 1.0,
        label: str = "",
        width: int = 150,
        height: int = 100,
    ) -> str:
        safe_max = max(0.0001, float(max_value))
        ratio = max(0.0, min(1.0, float(value) / safe_max))

        if ratio > 0.65:
            value_color = PALETTE["green"]
        elif ratio >= 0.45:
            value_color = PALETTE["yellow"]
        else:
            value_color = PALETTE["red"]

        cx = width / 2
        cy = height - 12
        r = min(width * 0.42, height * 0.8)

        def arc_path(start_ratio: float, end_ratio: float) -> str:
            start_ang = math.pi - (math.pi * start_ratio)
            end_ang = math.pi - (math.pi * end_ratio)
            x1 = cx + r * math.cos(start_ang)
            y1 = cy - r * math.sin(start_ang)
            x2 = cx + r * math.cos(end_ang)
            y2 = cy - r * math.sin(end_ang)
            large = 1 if end_ratio - start_ratio > 0.5 else 0
            return f"M{x1:.2f},{y1:.2f} A{r:.2f},{r:.2f} 0 {large} 1 {x2:.2f},{y2:.2f}"

        pointer_ang = math.pi - (math.pi * ratio)
        px = cx + (r - 6) * math.cos(pointer_ang)
        py = cy - (r - 6) * math.sin(pointer_ang)

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#111827}</style>',
            f'<path d="{arc_path(0.00, 0.45)}" fill="none" stroke="{PALETTE["red"]}" stroke-width="11" stroke-linecap="round"/>',
            f'<path d="{arc_path(0.45, 0.65)}" fill="none" stroke="{PALETTE["yellow"]}" stroke-width="11" stroke-linecap="round"/>',
            f'<path d="{arc_path(0.65, 1.00)}" fill="none" stroke="{PALETTE["green"]}" stroke-width="11" stroke-linecap="round"/>',
            f'<line x1="{cx:.2f}" y1="{cy:.2f}" x2="{px:.2f}" y2="{py:.2f}" stroke="{value_color}" stroke-width="3"/>',
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" fill="{value_color}"/>',
            f'<text x="{cx:.1f}" y="{cy - 18:.1f}" text-anchor="middle" font-size="17" font-weight="700" fill="{value_color}">{ratio:.0%}</text>',
        ]

        if label:
            parts.append(
                f'<text x="{cx:.1f}" y="{height - 2}" text-anchor="middle" font-size="11">{escape(label)}</text>'
            )

        parts.append("</svg>")
        return "".join(parts)

    @staticmethod
    def participant_grid(personas: list, width: int = 700) -> str:
        personas = personas or []
        cols = 4
        card_w = max(130, width // cols)
        card_h = 120
        rows = max(1, math.ceil(len(personas) / cols))
        height = rows * card_h + 16

        def tone(valence: float | None) -> str:
            if valence is None:
                return PALETTE["gray"]
            if valence > 0.2:
                return PALETTE["green"]
            if valence < -0.2:
                return PALETTE["red"]
            return PALETTE["gray"]

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
            '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#111827}</style>',
        ]

        for i, persona in enumerate(personas):
            col = i % cols
            row = i // cols
            x = col * card_w + 10
            y = row * card_h + 10
            cx = x + (card_w / 2) - 10
            color = tone(getattr(persona, "opinion_valence", None))

            name = str(getattr(persona, "name", "Participant"))
            initials = "".join([chunk[:1].upper() for chunk in name.split()[:2]]) or "P"
            age = getattr(getattr(persona, "demographics", None), "age", "-")
            occupation = str(getattr(getattr(persona, "demographics", None), "occupation", ""))
            if len(occupation) > 24:
                occupation = occupation[:21] + "..."

            parts.append(f'<rect x="{x}" y="{y}" width="{card_w - 20}" height="{card_h - 16}" rx="10" fill="#F9FAFB" stroke="#E5E7EB"/>')
            parts.append(f'<circle cx="{cx:.1f}" cy="{y + 30}" r="18" fill="{color}"/>')
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 34}" text-anchor="middle" font-size="12" font-weight="700" fill="#FFFFFF">{escape(initials)}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 58}" text-anchor="middle" font-size="12" font-weight="700">{escape(name)}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 74}" text-anchor="middle" font-size="11" fill="#6B7280">Age {age}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 90}" text-anchor="middle" font-size="10" fill="#6B7280">{escape(occupation)}</text>'
            )

        if not personas:
            parts.append('<text x="14" y="26" font-size="12" fill="#6B7280">No participant profiles available</text>')

        parts.append("</svg>")
        return "".join(parts)
