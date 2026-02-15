#!/bin/bash
# Run 9 synthetic focus groups: 3 concepts × 3 random seeds
set -e

cd /home/openclaw/.openclaw/workspace/synthetic-focus-groups/src
export PYTHONUNBUFFERED=1

OUTDIR="/home/openclaw/.openclaw/workspace/synthetic-focus-groups/validation/data"
mkdir -p "$OUTDIR"

CONCEPT_A="A small, screenless wearable device that clips to your clothing and acts as your personal AI assistant. It replaces your smartphone — no screen, fully voice-controlled, always listening for your commands. It can make calls, send messages, answer questions, give directions, and take photos using a tiny projector for visual output. The idea: free yourself from screen addiction while staying connected."

CONCEPT_B="A subscription meal service for dogs — fresh, human-grade ingredients, portioned and personalized to your dog's breed, age, weight, and health needs. Designed by veterinary nutritionists. Delivered weekly in eco-friendly packaging. No kibble, no preservatives. The idea: your dog eats as well as you do, with meals tailored to their specific needs."

CONCEPT_C="A mobile app that pays you to walk. It tracks your verified steps using your phone's sensors and GPS, then converts them into reward points redeemable for gift cards (Amazon, Starbucks, Target, etc.). No crypto or tokens — just straightforward points for steps. Walk 10,000 steps per day, earn roughly 1-3 dollars per day in rewards. The idea: get paid to be healthy."

run_one() {
  local concept="$1"
  local category="$2"
  local label="$3"
  local seed="$4"
  local outfile="$OUTDIR/${label}_seed${seed}.html"

  echo "=== Running: ${label} seed=${seed} ==="
  python3 demo.py "$concept" \
    --category "$category" \
    --provider google \
    --output "$outfile" \
    --seed "$seed" 2>&1 | tail -5
  echo "=== Done: ${label} seed=${seed} → ${outfile} ==="
  echo ""
}

# Concept A: AI Pin (3 seeds)
run_one "$CONCEPT_A" "consumer electronics" "concept_a" 42
run_one "$CONCEPT_A" "consumer electronics" "concept_a" 123
run_one "$CONCEPT_A" "consumer electronics" "concept_a" 777

# Concept B: Dog Meal Kit (3 seeds)
run_one "$CONCEPT_B" "pet food" "concept_b" 42
run_one "$CONCEPT_B" "pet food" "concept_b" 123
run_one "$CONCEPT_B" "pet food" "concept_b" 777

# Concept C: Walk-to-Earn (3 seeds)
run_one "$CONCEPT_C" "fitness app" "concept_c" 42
run_one "$CONCEPT_C" "fitness app" "concept_c" 123
run_one "$CONCEPT_C" "fitness app" "concept_c" 777

echo "ALL 9 RUNS COMPLETE"
ls -la "$OUTDIR"
