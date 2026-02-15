from __future__ import annotations

from report.charts import ChartGenerator


def test_horizontal_bar_chart_returns_svg() -> None:
    svg = ChartGenerator.horizontal_bar_chart(
        labels=["Purchase Intent", "Appeal", "Relevance"],
        values=[0.62, 0.71, 0.54],
        title="Concept Scores",
    )

    assert "<svg" in svg
    assert "</svg>" in svg


def test_horizontal_bar_chart_contains_all_labels() -> None:
    labels = ["Purchase Intent", "Overall Appeal", "Uniqueness"]
    svg = ChartGenerator.horizontal_bar_chart(labels=labels, values=[0.5, 0.6, 0.4])

    for label in labels:
        assert label in svg


def test_sentiment_line_chart_has_expected_data_points() -> None:
    values = [0.1, 0.3, -0.2, 0.4, 0.2]
    svg = ChartGenerator.sentiment_line_chart(
        phases=["warmup", "exploration", "deep_dive", "reaction", "synthesis"],
        values=values,
    )

    assert svg.count("<circle") == len(values)


def test_donut_chart_renders() -> None:
    svg = ChartGenerator.donut_chart(labels=["GO", "ITERATE", "NO-GO"], values=[6, 1, 1])
    assert "<svg" in svg
    assert "</svg>" in svg


def test_score_gauge_color_thresholds() -> None:
    high_svg = ChartGenerator.score_gauge(value=0.8)
    mid_svg = ChartGenerator.score_gauge(value=0.5)
    low_svg = ChartGenerator.score_gauge(value=0.3)

    assert "#10B981" in high_svg
    assert "#F59E0B" in mid_svg
    assert "#EF4444" in low_svg
