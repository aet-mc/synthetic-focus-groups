from __future__ import annotations

from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from analysis.models import AnalysisReport
from discussion.models import DiscussionTranscript, MessageRole

from .charts import ChartGenerator, PALETTE


class ReportGenerator:
    def __init__(self):
        self.chart_gen = ChartGenerator()
        templates_dir = Path(__file__).resolve().parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate_html(
        self,
        report: AnalysisReport,
        transcript: DiscussionTranscript,
        personas: list,
        include_transcript: bool = True,
    ) -> str:
        context = self._prepare_context(report=report, transcript=transcript, personas=personas)
        context["include_transcript"] = include_transcript
        template = self.env.get_template("report.html")
        return template.render(**context)

    def save_html(
        self,
        report: AnalysisReport,
        transcript: DiscussionTranscript,
        personas: list,
        output_path: str,
        include_transcript: bool = True,
    ) -> str:
        html = self.generate_html(
            report=report,
            transcript=transcript,
            personas=personas,
            include_transcript=include_transcript,
        )
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding="utf-8")
        return str(target)

    def _prepare_context(
        self,
        report: AnalysisReport,
        transcript: DiscussionTranscript,
        personas: list,
    ) -> dict:
        participant_count = len(personas)
        phase_count = len(transcript.config.phases)
        concept = transcript.config.product_concept

        rec_label = self._recommendation_label(report.recommendation)
        rec_color = {
            "GO": PALETTE["green"],
            "ITERATE": PALETTE["yellow"],
            "NO-GO": PALETTE["red"],
        }.get(rec_label, PALETTE["gray"])

        metrics = [
            ("Purchase Intent", report.concept_scores.purchase_intent),
            ("Overall Appeal", report.concept_scores.overall_appeal),
            ("Uniqueness", report.concept_scores.uniqueness),
            ("Relevance", report.concept_scores.relevance),
            ("Believability", report.concept_scores.believability),
            ("Value Perception", report.concept_scores.value_perception),
        ]

        metric_rows = [
            {
                "name": name,
                "score": score,
                "formatted": f"{score:.0%}",
                "status": self._status_label(score),
                "status_class": self._status_css(score),
                "interpretation": self._metric_interpretation(name, score),
            }
            for name, score in metrics
        ]

        phases = list(report.sentiment_timeline.by_phase.keys())
        sentiment_values = list(report.sentiment_timeline.by_phase.values())
        sentiment_rows = [
            {
                "phase": phase.replace("_", " ").title(),
                "score": score,
                "label": self._sentiment_label(score),
                "summary": self._sentiment_summary(phase, score),
            }
            for phase, score in zip(phases, sentiment_values)
        ]

        quote_speaker_lookup = self._quote_speaker_map(transcript)
        theme_rows = []
        for theme in report.themes:
            prevalence_count = len(theme.participant_ids)
            quotes = [
                {
                    "text": quote,
                    "speaker": quote_speaker_lookup.get(quote, "Participant"),
                }
                for quote in theme.supporting_quotes[:3]
            ]
            theme_rows.append(
                {
                    "name": theme.name,
                    "description": theme.description,
                    "prevalence": theme.prevalence,
                    "prevalence_text": f"{prevalence_count} of {participant_count} participants",
                    "sentiment": self._sentiment_label(theme.sentiment),
                    "sentiment_class": self._sentiment_css(theme.sentiment),
                    "quotes": quotes,
                }
            )

        participant_rows = []
        final_valence_map = self._final_opinion_map(transcript)
        for persona in personas:
            income_value = getattr(persona.demographics, "income", 0)
            participant_rows.append(
                {
                    "name": persona.name,
                    "age": persona.demographics.age,
                    "gender": persona.demographics.gender.title(),
                    "income_range": self._income_range(income_value),
                    "education": persona.demographics.education,
                    "vals_type": persona.psychographics.vals_type,
                    "initial_opinion": self._opinion_text(persona.opinion_valence),
                    "final_opinion": self._opinion_text(final_valence_map.get(persona.id, persona.opinion_valence)),
                }
            )

        segment_rows = [
            {
                "name": segment.segment_name,
                "purchase_intent": f"{segment.purchase_intent:.0%}",
                "difference": segment.differs_from_overall or "Aligned with overall findings.",
                "quote": segment.distinguishing_quote,
            }
            for segment in report.segment_insights
        ]

        quote_sections = {
            "positive": report.quotes.positive[:3],
            "negative": report.quotes.negative[:3],
            "surprising": report.quotes.surprising[:3],
            "most_impactful": report.quotes.most_impactful[:5],
        }

        transcript_rows = []
        current_phase = None
        for message in transcript.messages:
            phase_label = message.phase.value.replace("_", " ").title()
            row = {
                "is_phase": current_phase != message.phase.value,
                "phase": phase_label,
                "speaker": message.speaker_name,
                "content": message.content,
                "role": message.role.value,
            }
            transcript_rows.append(row)
            current_phase = message.phase.value

        recommendation_chart = self.chart_gen.donut_chart(
            labels=[rec_label, "Remaining"],
            values=[report.concept_scores.excitement_score, 1 - report.concept_scores.excitement_score],
            title="Recommendation Signal",
            colors=[rec_color, "#E5E7EB"],
        )

        theme_prevalence_chart = self.chart_gen.horizontal_bar_chart(
            labels=[theme["name"] for theme in theme_rows[:7]],
            values=[theme["prevalence"] for theme in theme_rows[:7]],
            title="Theme Prevalence",
        )

        return {
            "product_concept": concept,
            "study_date": date.today().strftime("%B %d, %Y"),
            "study_type": "Synthetic Focus Group - AI-Powered Market Research",
            "executive_summary": report.executive_summary,
            "recommendation": report.recommendation,
            "recommendation_label": rec_label,
            "recommendation_color": rec_color,
            "recommendation_chart": recommendation_chart,
            "confidence_level": report.confidence_level.title(),
            "key_stats": [
                {"value": f"{report.concept_scores.purchase_intent:.0%}", "label": "Purchase Intent"},
                {"value": str(len(report.themes)), "label": "Key Themes"},
                {"value": str(participant_count), "label": "Participants"},
            ],
            "excitement_gauge": self.chart_gen.score_gauge(
                value=report.concept_scores.excitement_score,
                label="Excitement Score",
                width=180,
                height=120,
            ),
            "concept_chart": self.chart_gen.horizontal_bar_chart(
                labels=[name for name, _ in metrics],
                values=[score for _, score in metrics],
                title="Top-2-Box Concept Scores",
            ),
            "metric_rows": metric_rows,
            "theme_rows": theme_rows,
            "theme_prevalence_chart": theme_prevalence_chart,
            "sentiment_chart": self.chart_gen.sentiment_line_chart(
                phases=phases,
                values=sentiment_values,
            ),
            "sentiment_trend": report.sentiment_timeline.trend,
            "sentiment_rows": sentiment_rows,
            "participant_grid_chart": self.chart_gen.participant_grid(personas=personas),
            "participant_rows": participant_rows,
            "segment_rows": segment_rows,
            "concerns": report.top_concerns,
            "opportunities": report.top_opportunities,
            "improvements": report.suggested_improvements,
            "quote_sections": quote_sections,
            "phase_count": phase_count,
            "message_count": len(transcript.messages),
            "participant_count": participant_count,
            "transcript_rows": transcript_rows,
            "methodology_points": [
                "This study used AI-generated personas grounded in US Census demographic data.",
                "Personality profiles were based on the Five-Factor (OCEAN) model with age and gender-adjusted distributions.",
                "Discussion was facilitated by an AI moderator using a five-phase qualitative research methodology.",
                "Analysis included thematic coding (Braun and Clarke framework), concept scoring, and segment analysis.",
                "Synthetic research is designed to complement, not replace, traditional research methods.",
            ],
        }

    @staticmethod
    def _status_css(score: float) -> str:
        if score > 0.65:
            return "status-good"
        if score >= 0.45:
            return "status-mid"
        return "status-low"

    @staticmethod
    def _status_label(score: float) -> str:
        if score > 0.65:
            return "Strong"
        if score >= 0.45:
            return "Moderate"
        return "Weak"

    @staticmethod
    def _sentiment_css(score: float) -> str:
        if score > 0.2:
            return "sentiment-positive"
        if score < -0.2:
            return "sentiment-negative"
        return "sentiment-mixed"

    @staticmethod
    def _sentiment_label(score: float) -> str:
        if score > 0.2:
            return "Positive"
        if score < -0.2:
            return "Negative"
        return "Mixed"

    @staticmethod
    def _recommendation_label(recommendation: str) -> str:
        upper = recommendation.upper()
        if upper.startswith("GO"):
            return "GO"
        if upper.startswith("ITERATE"):
            return "ITERATE"
        if upper.startswith("NO-GO"):
            return "NO-GO"
        return "ITERATE"

    @staticmethod
    def _metric_interpretation(name: str, score: float) -> str:
        if score > 0.65:
            return f"{name} is a clear strength with strong participant support."
        if score >= 0.45:
            return f"{name} is promising but requires refinement before scaling."
        return f"{name} signals material risk and needs substantial improvement."

    @staticmethod
    def _sentiment_summary(phase: str, score: float) -> str:
        phase_label = phase.replace("_", " ")
        if score > 0.2:
            tone = "participants responded positively"
        elif score < -0.2:
            tone = "participants expressed concerns"
        else:
            tone = "participants were mixed"
        return f"In {phase_label}, {tone} (sentiment {score:+.2f})."

    @staticmethod
    def _quote_speaker_map(transcript: DiscussionTranscript) -> dict[str, str]:
        out: dict[str, str] = {}
        for message in transcript.messages:
            if message.role == MessageRole.PARTICIPANT:
                out[message.content] = message.speaker_name
        return out

    @staticmethod
    def _final_opinion_map(transcript: DiscussionTranscript) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for message in transcript.messages:
            if message.role == MessageRole.PARTICIPANT:
                out[message.speaker_id] = message.sentiment
        return out

    @staticmethod
    def _opinion_text(value: float | None) -> str:
        if value is None:
            return "Neutral"
        if value > 0.2:
            return "Positive"
        if value < -0.2:
            return "Negative"
        return "Neutral"

    @staticmethod
    def _income_range(income: int) -> str:
        bands = [
            (25_000, "<$25k"),
            (50_000, "$25k-$50k"),
            (75_000, "$50k-$75k"),
            (100_000, "$75k-$100k"),
            (150_000, "$100k-$150k"),
        ]
        for threshold, label in bands:
            if income < threshold:
                return label
        return "$150k+"
