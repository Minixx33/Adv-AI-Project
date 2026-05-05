"""
ablations/run_ablation_weights.py

w1/w2 decision-weight sweep ablation for the Werewolf Suspicion module.

Sweeps the balance between base-agent belief (w1) and suspicion sigma (w2)
across 6 conditions, holding all detector weights fixed at their defaults.
This directly answers RQ4: what is the optimal w1/w2 balance?

Conditions (label: w1, w2):
  w1_1.0_w2_0.0   pure base-belief, suspicion ignored
  w1_0.8_w2_0.2
  w1_0.6_w2_0.4   default configuration
  w1_0.4_w2_0.6
  w1_0.2_w2_0.8
  w1_0.0_w2_1.0   pure suspicion sigma, base-belief ignored

Output structure:
  {output}/{experiment_name}_seed{seed}_{YYYY-MM-DD_HH-MM-SS}/
      w1_1.0_w2_0.0/
          summary.csv
          config_used.json
          run_*.log
      w1_0.8_w2_0.2/
          ...
      ...
      ablation_summary.csv    <- cross-condition comparison table

Usage:
  python ablations/run_ablation_weights.py \\
      --experiment-name yasmi \\
      --seed 42 \\
      --games 10000 \\
      --agent bayesian \\
      --output ./results/ablations
"""

import argparse
import csv
import datetime
import os
import sys

_MODEL_V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_V2 not in sys.path:
    sys.path.insert(0, _MODEL_V2)

from runner.game_runner import GameRunner

# ── Sweep grid ───────────────────────────────────────────────────────────────
# (label, w1, w2)
_SWEEP = [
    ("w1_1.0_w2_0.0", 1.0, 0.0),
    ("w1_0.8_w2_0.2", 0.8, 0.2),
    ("w1_0.6_w2_0.4", 0.6, 0.4),   # default
    ("w1_0.4_w2_0.6", 0.4, 0.6),
    ("w1_0.2_w2_0.8", 0.2, 0.8),
    ("w1_0.0_w2_1.0", 0.0, 1.0),
]

_DEFAULT_DET_WEIGHTS = {
    "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="w1/w2 decision-weight sweep ablation"
    )
    p.add_argument("--experiment-name", required=True, dest="experiment_name",
                   help="Label for output folder (e.g. 'yasmi')")
    p.add_argument("--seed", type=int, required=True,
                   help="Base random seed for reproducibility")
    p.add_argument("--games", type=int, default=10000,
                   help="Games per condition (default: 10000)")
    p.add_argument("--agent", default="bayesian",
                   choices=["bayesian", "heuristic", "mcts"],
                   help="Agent type for suspicion-enhanced players (default: bayesian)")
    p.add_argument("--output", default="./results/ablations",
                   help="Root output directory (default: ./results/ablations)")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers (default: min(cpu_count, 4))")
    return p.parse_args()


# ── Directory & config helpers ───────────────────────────────────────────────

def _make_top_dir(args) -> str:
    ts   = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{args.experiment_name}_seed{args.seed}_{ts}"
    path = os.path.join(args.output, name)
    os.makedirs(path, exist_ok=True)
    return path


def _base_config(args) -> dict:
    """
    Composition: 4 suspicion-enhanced + 4 base agents.
    Composition order matters: base agents are assigned PIDs 1-4, enhanced
    agents PIDs 5-8. The role_assigner then restricts wolves to PIDs 1-4
    (the base agent slots). This mirrors the main experiment setup and ensures
    the Seer (always in PIDs 5-8, enhanced) divines into the wolf-eligible
    range (PIDs 1-4) first, giving the village a realistic chance to find wolves.
    """
    agent     = args.agent
    n_workers = args.workers or min(os.cpu_count() or 4, 4)
    return {
        "players":              8,
        "seed":                 args.seed,
        "suspicion_agent_type": agent,
        "agent_composition": {
            agent:                     4,
            f"{agent}_with_suspicion": 4,
        },
        "wolf_strategies": {
            "bus_driver_probability":  0.2,
            "false_claimer_probability": 0.5,
        },
        "n_workers":            n_workers,
        "max_talk_rounds":      5,
        "mcts_simulations":     500,
        "mcts_depth":           5,
        "write_xlsx":           False,
    }


# ── Run helpers ──────────────────────────────────────────────────────────────

def _run_condition(label: str, cfg: dict, n_games: int, top_dir: str) -> list:
    cond_dir      = os.path.join(top_dir, label)
    cfg           = dict(cfg)
    cfg["output_dir"] = cond_dir
    runner        = GameRunner(cfg)
    _, rows       = runner.run_games(n_games)
    return rows


def _aggregate(rows: list, condition: str, w1: float, w2: float) -> dict:
    n = len(rows)
    if n == 0:
        return {"condition": condition, "n_games": 0}

    def _mean(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    village_win_rate = sum(1 for r in rows if r.get("village_win")) / n

    return {
        "condition":              condition,
        "w1":                     w1,
        "w2":                     w2,
        "n_games":                n,
        "village_win_rate":       round(village_win_rate, 4),
        "wolf_vote_precision":    _mean("wolf_vote_precision"),
        "wolf_vote_recall":       _mean("wolf_vote_recall"),
        "false_accusation_rate":  _mean("false_accusation_rate"),
        "sigma_gap":              _mean("sigma_gap"),
        "combined_gap":           _mean("combined_gap"),
        "avg_days":               _mean("n_days"),
    }


def _write_ablation_summary(agg_rows: list, top_dir: str) -> None:
    if not agg_rows:
        return
    path = os.path.join(top_dir, "ablation_summary.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"\n  Ablation summary -> {path}")


def _print_table(agg_rows: list) -> None:
    print()
    print("=" * 72)
    print("  w1/w2 WEIGHT SWEEP RESULTS")
    print(f"  {'Condition':<22} {'w1':>5} {'w2':>5} {'VillageWin':>11} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 72)
    for r in agg_rows:
        marker = " <-- default" if r["w1"] == 0.6 else ""
        print(
            f"  {r['condition']:<22} "
            f"{r['w1']:>5.1f} "
            f"{r['w2']:>5.1f} "
            f"{r['village_win_rate']:>11.4f} "
            f"{(r['wolf_vote_precision'] or 0):>10.4f} "
            f"{(r['wolf_vote_recall'] or 0):>8.4f}"
            f"{marker}"
        )
    print("=" * 72)

    best = max(agg_rows, key=lambda r: r["village_win_rate"])
    print(f"\n  Best village win rate: {best['condition']}  "
          f"(w1={best['w1']}, w2={best['w2']})  "
          f"→ {best['village_win_rate']:.4f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    top_dir = _make_top_dir(args)
    base_cfg = _base_config(args)

    print(f"\n{'='*60}")
    print(f"  w1/w2 Decision-Weight Sweep Ablation")
    print(f"  Experiment  : {args.experiment_name}")
    print(f"  Seed        : {args.seed}")
    print(f"  Games/cond  : {args.games}")
    print(f"  Agent       : {args.agent}")
    print(f"  Conditions  : {len(_SWEEP)}")
    print(f"  Output      : {top_dir}")
    print(f"{'='*60}\n")

    agg_rows = []
    for label, w1, w2 in _SWEEP:
        print(f"\n--- Condition: {label}  (w1={w1}, w2={w2}) ---")
        cfg = dict(base_cfg)
        cfg["suspicion_weights"] = {
            **_DEFAULT_DET_WEIGHTS,
            "w1": w1,
            "w2": w2,
        }
        rows = _run_condition(label, cfg, args.games, top_dir)
        agg  = _aggregate(rows, label, w1, w2)
        agg_rows.append(agg)

    _write_ablation_summary(agg_rows, top_dir)
    _print_table(agg_rows)
    print(f"Done. Results in: {top_dir}\n")


if __name__ == "__main__":
    main()
