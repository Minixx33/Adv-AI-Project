#!/usr/bin/env bash
# run_mixed2_experiments.sh
#
# Runs the mixed-population experiment (2 suspicion agents, 8 players)
# for all 3 compatible agent types × 3 seeds.
#
# Total: 9 runs × 10,000 games = 90,000 games
# Estimated time: ~25-28 hrs on a standard laptop (runs sequentially)
#
# Usage (from WSL, anywhere):
#   bash /mnt/c/Users/yasmi/OneDrive/Desktop/Github/Adv-AI-Project/run_mixed2_experiments.sh
#
# Output lands in:
#   Model_V2/results/<agent>_mixed2_seed<N>_<timestamp>/

set -euo pipefail

PROJECT_DIR="/mnt/c/Users/yasmi/OneDrive/Desktop/Github/Adv-AI-Project"
MODEL_DIR="${PROJECT_DIR}/Model_V2"
PYTHON="python3"

SEEDS=(42 123 456)
AGENTS=("bayesian" "heuristic" "mcts")

cd "$MODEL_DIR"

echo "========================================"
echo "  Mixed-2 Experiment Suite"
echo "  Agents: ${AGENTS[*]}"
echo "  Seeds:  ${SEEDS[*]}"
echo "  Games:  10,000 per run"
echo "  Total runs: $((${#AGENTS[@]} * ${#SEEDS[@]}))"
echo "========================================"
echo ""

TOTAL_RUNS=$(( ${#AGENTS[@]} * ${#SEEDS[@]} ))
RUN_NUM=0

for AGENT in "${AGENTS[@]}"; do
    CONFIG="${PROJECT_DIR}/configs/mixed2_${AGENT}.json"

    for SEED in "${SEEDS[@]}"; do
        RUN_NUM=$(( RUN_NUM + 1 ))
        RUN_NAME="${AGENT}_mixed2_seed${SEED}"

        echo "----------------------------------------"
        echo "  Run ${RUN_NUM}/${TOTAL_RUNS}: ${RUN_NAME}"
        echo "  Config: ${CONFIG}  |  Seed: ${SEED}"
        echo "----------------------------------------"

        $PYTHON run_game.py \
            --config "$CONFIG" \
            --seed "$SEED" \
            --run-name "$RUN_NAME"

        echo "  ✓ Done: ${RUN_NAME}"
        echo ""
    done
done

echo "========================================"
echo "  All runs complete."
echo "  Results in: ${MODEL_DIR}/results/"
echo "========================================"
