#!/usr/bin/env bash
# run_fullsus_experiments.sh
#
# Runs the full-suspicion experiment (all 8 players use suspicion module).
# Upper-bound benchmark. Compatible agents: bayesian, heuristic, mcts.
#
# Total: 9 runs x 10,000 games = 90,000 games
#
# Usage (from WSL, anywhere):
#   bash /mnt/c/Users/yasmi/OneDrive/Desktop/Github/Adv-AI-Project/run_fullsus_experiments.sh

set -euo pipefail

PROJECT_DIR="/mnt/c/Users/yasmi/OneDrive/Desktop/Github/Adv-AI-Project"
MODEL_DIR="${PROJECT_DIR}/Model_V2"
PYTHON="python3"

SEEDS=(42 123 456)
AGENTS=("bayesian" "heuristic" "mcts")

cd "$MODEL_DIR"

echo "========================================"
echo "  Full-Suspicion Experiment Suite"
echo "  Agents: ${AGENTS[*]}"
echo "  Seeds:  ${SEEDS[*]}"
echo "  Games:  10,000 per run"
echo "  Total runs: $((${#AGENTS[@]} * ${#SEEDS[@]}))"
echo "========================================"
echo ""

TOTAL_RUNS=$(( ${#AGENTS[@]} * ${#SEEDS[@]} ))
RUN_NUM=0

for AGENT in "${AGENTS[@]}"; do
    CONFIG="${PROJECT_DIR}/configs/${AGENT}_fullsus.json"

    for SEED in "${SEEDS[@]}"; do
        RUN_NUM=$(( RUN_NUM + 1 ))
        RUN_NAME="${AGENT}_fullsus_seed${SEED}"

        echo "----------------------------------------"
        echo "  Run ${RUN_NUM}/${TOTAL_RUNS}: ${RUN_NAME}"
        echo "  Config: ${CONFIG}  |  Seed: ${SEED}"
        echo "----------------------------------------"

        $PYTHON run_game.py             --config "$CONFIG"             --seed "$SEED"             --run-name "$RUN_NAME"             --games 10000             --players 8

        echo "  Done: ${RUN_NAME}"
        echo ""
    done
done

echo "========================================"
echo "  All runs complete."
echo "  Results in: ${MODEL_DIR}/results/"
echo "========================================"
