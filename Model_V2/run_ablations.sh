#!/usr/bin/env bash
# =============================================================================
# run_ablations.sh
#
# Runs all 4 ablation experiments across 3 seeds.
#
# Ablations:
#   1. detector     -- leave-one-out (D1-D5)           6 conditions × 3 seeds
#   2. weights      -- w1/w2 decision-weight sweep      6 conditions × 3 seeds
#   3. adversarial  -- wolf strategy ablation (RQ3)     6 conditions × 3 seeds
#   4. learner      -- online-learner vs fixed weights  2 conditions × 3 seeds
#                      (runs sequentially; uses 2k games to keep runtime manageable)
#
# Configuration:
#   Agent   : bayesian
#   Games   : 10000  (2000 for learner — sequential run)
#   Players : 8 (hardcoded in ablation scripts)
#   Seeds   : 42  123  456
#
# Output:
#   All results land in ./results/ablations/
#   Each run gets its own timestamped subfolder.
#   A master log is written to ./results/ablations/ablations_master.log
#
# Usage:
#   cd Model_V2
#   bash run_ablations.sh
#
#   Override output dir:
#   bash run_ablations.sh --output /path/to/output
#
#   Dry run (print commands, don't execute):
#   bash run_ablations.sh --dry-run
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
AGENT="bayesian"
GAMES=10000
LEARNER_GAMES=2000       # learner runs sequentially; cap to keep runtime sane
SEEDS=(42 123 456)
OUTPUT="./results/ablations"
DRY_RUN=false

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)    OUTPUT="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=true; shift ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$OUTPUT"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/ablations_master.log"

_log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

_elapsed() {
    local secs=$1
    printf "%dh %02dm %02ds" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

_run() {
    local label="$1"; shift
    if $DRY_RUN; then
        echo "  [DRY-RUN] python $*"
        return 0
    fi
    local t_start=$SECONDS
    _log "START  $label"
    python "$@" 2>&1 | tee -a "$MASTER_LOG"
    local exit_code=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - t_start ))
    if [[ $exit_code -eq 0 ]]; then
        _log "DONE   $label  ($(_elapsed $elapsed))"
    else
        _log "FAILED $label  (exit=$exit_code, $(_elapsed $elapsed))"
        echo "ERROR: $label failed. Check $MASTER_LOG for details." >&2
        exit $exit_code
    fi
}

# ── Banner ────────────────────────────────────────────────────────────────────
_log "============================================================"
_log "  Werewolf Ablation Suite"
_log "  Agent   : $AGENT"
_log "  Games   : $GAMES  (learner: $LEARNER_GAMES)"
_log "  Seeds   : ${SEEDS[*]}"
_log "  Output  : $OUTPUT"
_log "  Dry-run : $DRY_RUN"
_log "============================================================"

TOTAL_RUNS=$(( ${#SEEDS[@]} * 4 ))
RUN_NUM=0
SUITE_START=$SECONDS

# ── 1. Detector leave-one-out ─────────────────────────────────────────────────
_log ""
_log "── ABLATION 1/4: Detector Leave-One-Out ─────────────────────"
_log "  6 conditions (all_detectors + no_d1..no_d5) × ${#SEEDS[@]} seeds"
_log "  Approx: 6 × $GAMES × ${#SEEDS[@]} = $(( 6 * GAMES * ${#SEEDS[@]} )) games total"

for SEED in "${SEEDS[@]}"; do
    RUN_NUM=$(( RUN_NUM + 1 ))
    _log ""
    _log "  Run $RUN_NUM/$TOTAL_RUNS — detector ablation, seed=$SEED"
    _run "detector_seed${SEED}" \
        "$SCRIPT_DIR/ablations/run_ablation_detector.py" \
        --experiment-name "detector" \
        --seed "$SEED" \
        --games "$GAMES" \
        --agent "$AGENT" \
        --output "$OUTPUT"
done

# ── 2. w1/w2 weight sweep ─────────────────────────────────────────────────────
_log ""
_log "── ABLATION 2/4: w1/w2 Decision-Weight Sweep ────────────────"
_log "  6 conditions (w1/w2 pairs) × ${#SEEDS[@]} seeds"
_log "  Approx: 6 × $GAMES × ${#SEEDS[@]} = $(( 6 * GAMES * ${#SEEDS[@]} )) games total"

for SEED in "${SEEDS[@]}"; do
    RUN_NUM=$(( RUN_NUM + 1 ))
    _log ""
    _log "  Run $RUN_NUM/$TOTAL_RUNS — weights ablation, seed=$SEED"
    _run "weights_seed${SEED}" \
        "$SCRIPT_DIR/ablations/run_ablation_weights.py" \
        --experiment-name "weights" \
        --seed "$SEED" \
        --games "$GAMES" \
        --agent "$AGENT" \
        --output "$OUTPUT"
done

# ── 3. Adversarial wolf strategies ───────────────────────────────────────────
_log ""
_log "── ABLATION 3/4: Adversarial Wolf Strategies (RQ3) ─────────"
_log "  6 conditions (3 strategies × susp on/off) × ${#SEEDS[@]} seeds"
_log "  Approx: 6 × $GAMES × ${#SEEDS[@]} = $(( 6 * GAMES * ${#SEEDS[@]} )) games total"

for SEED in "${SEEDS[@]}"; do
    RUN_NUM=$(( RUN_NUM + 1 ))
    _log ""
    _log "  Run $RUN_NUM/$TOTAL_RUNS — adversarial ablation, seed=$SEED"
    _run "adversarial_seed${SEED}" \
        "$SCRIPT_DIR/ablations/run_ablation_adversarial.py" \
        --experiment-name "adversarial" \
        --seed "$SEED" \
        --games "$GAMES" \
        --agent "$AGENT" \
        --output "$OUTPUT"
done

# ── 4. Online learner ─────────────────────────────────────────────────────────
_log ""
_log "── ABLATION 4/4: Online Learner vs Fixed Weights ────────────"
_log "  2 conditions (fixed, online) × ${#SEEDS[@]} seeds"
_log "  NOTE: online-learner condition runs sequentially (no parallel workers)."
_log "  Using $LEARNER_GAMES games (not $GAMES) to keep runtime manageable."
_log "  Approx: $(( (GAMES + LEARNER_GAMES) * ${#SEEDS[@]} )) games total"

for SEED in "${SEEDS[@]}"; do
    RUN_NUM=$(( RUN_NUM + 1 ))
    _log ""
    _log "  Run $RUN_NUM/$TOTAL_RUNS — learner ablation, seed=$SEED"
    _run "learner_seed${SEED}" \
        "$SCRIPT_DIR/ablations/run_ablation_learner.py" \
        --experiment-name "learner" \
        --seed "$SEED" \
        --games "$LEARNER_GAMES" \
        --agent "$AGENT" \
        --output "$OUTPUT"
done

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( SECONDS - SUITE_START ))
_log ""
_log "============================================================"
_log "  ALL ABLATIONS COMPLETE"
_log "  Total wall time : $(_elapsed $TOTAL_ELAPSED)"
_log "  Results in      : $OUTPUT"
_log "  Master log      : $MASTER_LOG"
_log "============================================================"

echo ""
echo "Results tree:"
find "$OUTPUT" -name "ablation_summary.csv" | sort | while read -r f; do
    echo "  $f"
done
