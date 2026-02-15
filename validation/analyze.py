#!/usr/bin/env python3
"""
Validation Analysis: Compare synthetic focus group results vs real human responses.

Usage:
    python3 analyze.py --responses responses.json --synthetic-dir data/
    python3 analyze.py --sheets-json exported_sheet.json --synthetic-dir data/

Reads synthetic HTML reports from data/ and human responses from Google Sheets export.
Outputs correlation report + publishable summary.
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ─── Synthetic Data Extraction ───────────────────────────────────────────────

def extract_scores_from_html(html_path: Path) -> dict:
    """Extract concept scores from a synthetic focus group HTML report."""
    text = html_path.read_text()
    
    scores = {}
    # Look for score patterns in the HTML
    metrics = ["purchase_intent", "overall_appeal", "uniqueness", "relevance", "believability", "value_perception"]
    
    for metric in metrics:
        # Try multiple patterns
        patterns = [
            rf'{metric}["\s:]+(\d+(?:\.\d+)?)\s*%',
            rf'{metric.replace("_", " ").title()}[^0-9]*?(\d+(?:\.\d+)?)\s*%',
            rf"'{metric}':\s*(\d+(?:\.\d+)?)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                scores[metric] = float(m.group(1)) / 100.0
                break
    
    # Extract recommendation
    rec_match = re.search(r'Recommendation:\s*(GO|NO-GO|ITERATE)', text)
    if rec_match:
        scores['recommendation'] = rec_match.group(1)
    
    # Extract excitement score
    exc_match = re.search(r'Excitement Score:\s*(\d+(?:\.\d+)?)\s*%', text)
    if exc_match:
        scores['excitement_score'] = float(exc_match.group(1)) / 100.0
    
    # Extract theme count
    theme_match = re.search(r'(\d+)\s*themes', text)
    if theme_match:
        scores['theme_count'] = int(theme_match.group(1))
    
    # Extract message count
    msg_match = re.search(r'(\d+)\s*messages', text)
    if msg_match:
        scores['message_count'] = int(msg_match.group(1))
    
    return scores


def load_synthetic_data(data_dir: Path) -> dict:
    """Load all synthetic runs, grouped by concept."""
    concepts = {}
    for html_file in sorted(data_dir.glob("concept_*_seed*.html")):
        # Parse filename: concept_a_seed42.html
        parts = html_file.stem.split("_")
        concept_key = parts[1]  # a, b, c
        seed = parts[2].replace("seed", "")
        
        scores = extract_scores_from_html(html_file)
        scores['seed'] = seed
        scores['file'] = html_file.name
        
        concepts.setdefault(concept_key, []).append(scores)
    
    return concepts


def aggregate_synthetic(runs: list[dict]) -> dict:
    """Average scores across multiple runs of the same concept."""
    metrics = ["purchase_intent", "overall_appeal", "uniqueness", "relevance", "believability", "value_perception", "excitement_score"]
    agg = {}
    for metric in metrics:
        values = [r[metric] for r in runs if metric in r]
        if values:
            agg[metric] = np.mean(values)
            agg[f"{metric}_std"] = np.std(values)
    return agg


# ─── Human Data Processing ───────────────────────────────────────────────────

def parse_human_responses(responses: list[dict]) -> dict:
    """Parse human survey responses into per-concept scores."""
    # Filter out attention check failures
    valid = [r for r in responses if r.get("attention") == "agree"]
    failed = len(responses) - len(valid)
    
    concepts = {
        "a": {"purchase": [], "appeal": [], "unique": [], "relevant": [], "believe": [], "value": [], "likes": [], "concerns": []},
        "b": {"purchase": [], "appeal": [], "unique": [], "relevant": [], "believe": [], "value": [], "likes": [], "concerns": []},
        "c": {"purchase": [], "appeal": [], "unique": [], "relevant": [], "believe": [], "value": [], "likes": [], "concerns": []},
    }
    
    demographics = {"age": [], "gender": [], "income": [], "country": []}
    rankings = {"rank_1": [], "rank_2": [], "rank_3": [], "recommend": []}
    
    for r in valid:
        for concept in ["a", "b", "c"]:
            for metric in ["purchase", "appeal", "unique", "relevant", "believe", "value"]:
                key = f"{concept}_{metric}"
                val = r.get(key)
                if val is not None:
                    try:
                        concepts[concept][metric].append(int(val))
                    except (ValueError, TypeError):
                        pass
            
            concepts[concept]["likes"].append(r.get(f"{concept}_likes", ""))
            concepts[concept]["concerns"].append(r.get(f"{concept}_concerns", ""))
        
        for demo_key in demographics:
            demographics[demo_key].append(r.get(demo_key, ""))
        for rank_key in rankings:
            rankings[rank_key].append(r.get(rank_key, ""))
    
    return {
        "concepts": concepts,
        "demographics": demographics,
        "rankings": rankings,
        "total_responses": len(responses),
        "valid_responses": len(valid),
        "attention_failures": failed,
    }


def human_concept_scores(concept_data: dict) -> dict:
    """Convert raw human responses to comparable scores."""
    metrics_map = {
        "purchase": "purchase_intent",
        "appeal": "overall_appeal",
        "unique": "uniqueness",
        "relevant": "relevance",
        "believe": "believability",
        "value": "value_perception",
    }
    
    scores = {}
    for short_name, long_name in metrics_map.items():
        values = concept_data[short_name]
        if values:
            arr = np.array(values, dtype=float)
            scores[long_name] = np.mean(arr) / 5.0  # Normalize to 0-1
            scores[f"{long_name}_std"] = np.std(arr) / 5.0
            scores[f"{long_name}_n"] = len(values)
            # Top-2-box: % scoring 4 or 5
            scores[f"{long_name}_t2b"] = np.mean(arr >= 4)
    
    return scores


# ─── Comparison & Correlation ────────────────────────────────────────────────

def pearson_correlation(x: list[float], y: list[float]) -> tuple[float, str]:
    """Calculate Pearson r and interpret it."""
    if len(x) < 3:
        return float("nan"), "insufficient data (need 3+ concepts)"
    
    x_arr = np.array(x)
    y_arr = np.array(y)
    
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan"), "no variance"
    
    r = np.corrcoef(x_arr, y_arr)[0, 1]
    
    if r >= 0.90:
        interp = "exceptional"
    elif r >= 0.80:
        interp = "strong"
    elif r >= 0.70:
        interp = "decent"
    elif r >= 0.50:
        interp = "moderate"
    else:
        interp = "weak"
    
    return float(r), interp


def rank_agreement(synthetic_order: list[str], human_order: list[str]) -> tuple[float, str]:
    """Kendall's tau-like rank agreement for 3 items."""
    if synthetic_order == human_order:
        return 1.0, "perfect agreement"
    
    # Count concordant pairs
    concordant = 0
    discordant = 0
    n = len(synthetic_order)
    for i in range(n):
        for j in range(i + 1, n):
            s_i = synthetic_order.index(synthetic_order[i])
            s_j = synthetic_order.index(synthetic_order[j])
            h_i = human_order.index(synthetic_order[i])
            h_j = human_order.index(synthetic_order[j])
            if (s_i - s_j) * (h_i - h_j) > 0:
                concordant += 1
            else:
                discordant += 1
    
    tau = (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0
    
    if tau >= 0.8:
        interp = "strong agreement"
    elif tau >= 0.5:
        interp = "moderate agreement"
    else:
        interp = "weak agreement"
    
    return tau, interp


# ─── Report Generation ───────────────────────────────────────────────────────

def generate_report(synthetic: dict, human: dict, output_path: Path) -> None:
    """Generate the full validation comparison report."""
    lines = []
    lines.append("# Synthetic Focus Groups — Validation Report")
    lines.append(f"\n*Generated from {human['valid_responses']} valid human responses "
                 f"({human['attention_failures']} failed attention check)*\n")
    
    # Demographics summary
    lines.append("## Human Panel Demographics\n")
    for key in ["age", "gender", "country"]:
        counts = Counter(human["demographics"][key])
        total = sum(counts.values())
        dist = ", ".join(f"{k}: {v} ({v/total:.0%})" for k, v in counts.most_common(6))
        lines.append(f"**{key.title()}:** {dist}")
    lines.append("")
    
    # Per-concept comparison
    concept_names = {"a": "Wearable AI Pin", "b": "Premium Dog Meal Kit", "c": "Walk-to-Earn App"}
    metrics = ["purchase_intent", "overall_appeal", "uniqueness", "relevance", "believability", "value_perception"]
    
    all_synthetic_values = {m: [] for m in metrics}
    all_human_values = {m: [] for m in metrics}
    
    lines.append("## Per-Concept Comparison\n")
    
    for concept_key, concept_name in concept_names.items():
        lines.append(f"### {concept_name}\n")
        lines.append(f"| Metric | Synthetic (mean±std) | Human (mean±std) | Δ | Human T2B |")
        lines.append(f"|--------|---------------------|-----------------|---|-----------|")
        
        synth_runs = synthetic.get(concept_key, [])
        synth_agg = aggregate_synthetic(synth_runs) if synth_runs else {}
        human_scores = human_concept_scores(human["concepts"][concept_key])
        
        for metric in metrics:
            s_val = synth_agg.get(metric, float("nan"))
            s_std = synth_agg.get(f"{metric}_std", 0)
            h_val = human_scores.get(metric, float("nan"))
            h_std = human_scores.get(f"{metric}_std", 0)
            h_t2b = human_scores.get(f"{metric}_t2b", float("nan"))
            delta = abs(s_val - h_val) if not (np.isnan(s_val) or np.isnan(h_val)) else float("nan")
            
            all_synthetic_values[metric].append(s_val)
            all_human_values[metric].append(h_val)
            
            lines.append(
                f"| {metric.replace('_', ' ').title()} "
                f"| {s_val:.1%} ± {s_std:.1%} "
                f"| {h_val:.1%} ± {h_std:.1%} "
                f"| {delta:.1%} "
                f"| {h_t2b:.0%} |"
            )
        lines.append("")
    
    # Overall correlations
    lines.append("## Correlation Analysis\n")
    lines.append("| Metric | Pearson r | Interpretation |")
    lines.append("|--------|-----------|----------------|")
    
    correlations = []
    for metric in metrics:
        s_vals = all_synthetic_values[metric]
        h_vals = all_human_values[metric]
        r, interp = pearson_correlation(s_vals, h_vals)
        correlations.append(r)
        lines.append(f"| {metric.replace('_', ' ').title()} | {r:.3f} | {interp} |")
    
    valid_corrs = [c for c in correlations if not np.isnan(c)]
    avg_corr = np.mean(valid_corrs) if valid_corrs else float("nan")
    lines.append(f"\n**Average correlation across all metrics: {avg_corr:.3f}**\n")
    
    # Verdict
    lines.append("## Verdict\n")
    if avg_corr >= 0.90:
        lines.append("🟢 **EXCEPTIONAL** — Synthetic results match real humans with >90% correlation. Publishable. Lead with this everywhere.")
    elif avg_corr >= 0.80:
        lines.append("🟢 **STRONG** — Synthetic groups match real humans with >80% accuracy. Defensible for marketing and sales.")
    elif avg_corr >= 0.70:
        lines.append("🟡 **DECENT** — Directional accuracy for rapid screening. Needs caveats in positioning.")
    elif avg_corr >= 0.50:
        lines.append("🟠 **MODERATE** — Some signal but significant gaps. Improve engine before launching.")
    else:
        lines.append("🔴 **WEAK** — Synthetic results do not reliably match human responses. Do not publish numbers. Fix engine first.")
    
    # Ranking comparison
    lines.append("\n## Concept Ranking Comparison\n")
    
    # Human ranking from survey
    rank_counts = Counter(human["rankings"]["rank_1"])
    human_rank_1 = rank_counts.most_common(1)[0][0] if rank_counts else "unknown"
    
    recommend_counts = Counter(human["rankings"]["recommend"])
    human_recommend = recommend_counts.most_common(1)[0][0] if recommend_counts else "unknown"
    
    lines.append(f"**Human most appealing:** {human_rank_1}")
    lines.append(f"**Human most recommended:** {human_recommend}")
    
    # Synthetic ranking from excitement scores
    synth_excitement = {}
    for concept_key in ["a", "b", "c"]:
        runs = synthetic.get(concept_key, [])
        if runs:
            exc_values = [r.get("excitement_score", 0) for r in runs]
            synth_excitement[concept_key] = np.mean(exc_values)
    
    if synth_excitement:
        synth_order = sorted(synth_excitement, key=synth_excitement.get, reverse=True)
        concept_map = {"a": "ai-pin", "b": "dog-food", "c": "walk-earn"}
        lines.append(f"**Synthetic ranking (by excitement):** {' > '.join(concept_names[k] for k in synth_order)}")
    
    lines.append("")
    
    # Raw data tables
    lines.append("## Raw Synthetic Scores (per run)\n")
    for concept_key, concept_name in concept_names.items():
        lines.append(f"### {concept_name}")
        runs = synthetic.get(concept_key, [])
        for run in runs:
            seed = run.get("seed", "?")
            pi = run.get("purchase_intent", "?")
            exc = run.get("excitement_score", "?")
            rec = run.get("recommendation", "?")
            pi_str = f"{pi:.0%}" if isinstance(pi, float) else str(pi)
            exc_str = f"{exc:.0%}" if isinstance(exc, float) else str(exc)
            lines.append(f"- Seed {seed}: PI={pi_str}, Excitement={exc_str}, Rec={rec}")
        lines.append("")
    
    report = "\n".join(lines)
    output_path.write_text(report)
    print(f"\n{'='*60}")
    print(f"  Validation report written to: {output_path}")
    print(f"  Human responses: {human['valid_responses']}")
    print(f"  Average correlation: {avg_corr:.3f}")
    print(f"{'='*60}\n")
    print(report)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze synthetic vs human focus group results")
    parser.add_argument("--responses", type=Path, help="JSON file with human responses (array of objects)")
    parser.add_argument("--synthetic-dir", type=Path, default=Path("data"), help="Directory with synthetic HTML reports")
    parser.add_argument("--output", type=Path, default=Path("validation_report.md"), help="Output report path")
    parser.add_argument("--min-responses", type=int, default=20, help="Minimum responses before running analysis")
    args = parser.parse_args()
    
    # Load synthetic data
    print("Loading synthetic data...")
    synthetic = load_synthetic_data(args.synthetic_dir)
    for key, runs in synthetic.items():
        print(f"  Concept {key}: {len(runs)} runs")
    
    if not synthetic:
        print("ERROR: No synthetic data found in", args.synthetic_dir)
        sys.exit(1)
    
    # Load human data
    if args.responses and args.responses.exists():
        print(f"Loading human responses from {args.responses}...")
        raw = json.loads(args.responses.read_text())
        
        # Handle different formats
        if isinstance(raw, list):
            responses = raw
        elif isinstance(raw, dict) and "responses" in raw:
            responses = raw["responses"]
        else:
            # Might be Google Sheets format — each row has a "Response Data" field with JSON
            responses = []
            for row in (raw if isinstance(raw, list) else [raw]):
                rd = row.get("Response Data") or row.get("response_data") or row.get("Response data")
                if rd:
                    try:
                        responses.append(json.loads(rd))
                    except json.JSONDecodeError:
                        pass
        
        print(f"  Total: {len(responses)} responses")
        
        if len(responses) < args.min_responses:
            print(f"  ⚠️  Only {len(responses)} responses (minimum: {args.min_responses}). Waiting for more.")
            print(f"  Run with --min-responses {len(responses)} to force analysis.")
            sys.exit(0)
        
        human = parse_human_responses(responses)
    else:
        print("ERROR: No responses file provided or file not found.")
        print("Export your Google Sheet as JSON and pass with --responses")
        sys.exit(1)
    
    # Generate report
    print("Generating comparison report...")
    generate_report(synthetic, human, args.output)


if __name__ == "__main__":
    main()
