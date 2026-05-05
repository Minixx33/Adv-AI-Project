"""
ablations/run_ablation_detector.py

Detector leave-one-out ablation for the Werewolf Suspicion module.

Runs 6 conditions using the chosen agent (default: bayesian):
  all_detectors   all 5 detectors active at default weights
  no_d1           Vote-Accusation Mismatch disabled (weight zeroed, rest renormalized)
  no_d2           Bandwagon Index disabled
  no_d3           Pressure-Triggered Accusations disabled
  no_d4           Protection Patterns disabled
  no_d5           Claim Consistency disabled

For each "no_dN" condition the zeroed weight is removed and the remaining
a1-a5 weights are renormalized to sum to 1.0, keeping E in [0,1] and
making the comparison fair across conditions.

Output structure:
  {output}/{experiment_name}_seed{seed}_{YYYY-MM-DD_HH-MM-SS}/
      all_detectors/
          summary.csv
          config_used.json
          run_*.log
      no_d1/
          ...
      ...
      ablation_summary.csv    <- cross-condition comparison table

Usage:
  python ablations/run_ablation_detector.py \\
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

# ── Default suspicion weights ────────────────────────────────────────────────
_DEFAULT_WEIGHTS = {
    "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
    "w1": 0.60, "w2": 0.40,
}
_DETECTOR_KEYS = ["a1", "a2", "a3", "a4", "a5"]
_DETECTOR_NAMES = {
    "a1": "Vote-Accusation Mismatch (D1)",
    "a2": "Bandwagon Index (D2)",
    "a3": "Pressure-Triggered Accusations (D3)",
    "a4": "Protection Patterns (D4)",
    "a5": "Claim Consistency (D5)",
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Detector leave-one-out ablation for the suspicion module"
    )
    p.add_argument("--experiment-name", required=True, dest="experiment_name",
                   help="Label for output folder (e.g. 'yasmi')")
    p.add_argument("--seed", type=int, required=True,
                   help="Base random seed for reproducibility")
    p.add_argument("--games", type=int, default=10000,
                   help="Games per condition (default: 10000)")
    p.add_argument("--agent", default="bayesian",
                   choices=["bayesian", "heuristic", "mcts"],
                   help="Agent type used for suspicion-enhanced players (default: bayesian)")
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
    Composition: 6 suspicion-enhanced + 2 base agents.
    With 8 players (2 wolves) the role_assigner allows at most 6 suspicion
    agents (6 non-wolf roles: 4 Villager, 1 Seer, 1 Possessed).
    """
    agent     = args.agent
    n_workers = args.workers or min(os.cpu_count() or 4, 4)
    return {
        "players":              8,
        "seed":                 args.seed,
        "suspicion_agent_type": agent,
        "agent_composition": {
            f"{agent}_with_suspicion": 6,
            agent:                     2,
        },
        "suspicion_weights":    dict(_DEFAULT_WEIGHTS),
        "n_workers":            n_workers,
        "max_talk_rounds":      5,
        "mcts_simulations":     500,
        "mcts_depth":           5,
        "write_xlsx":           False,
    }


def _renormalize_without(weights: dict, zeroed_key: str) -> dict:
    """
    Return a copy of weights with zeroed_key set to 0.0 and the remaining
    detector weights (a1-a5) rescaled so they sum to 1.0.
    Decision weights w1/w2 are passed through unchanged.
    """
    new_w = dict(weights)
    new_w[zeroed_key] = 0.0
    det_sum = sum(new_w[k] for k in _DETECTOR_KEYS)
    if det_sum > 0:
        scale = 1.0 / det_sum
        for k in _DETECTOR_KEYS:
            new_w[k] = round(new_w[k] * scale, 6)
    return new_w


# ── Run helpers ──────────────────────────────────────────────────────────────

def _run_condition(label: str, cfg: dict, n_games: int, top_dir: str) -> list:
    cond_dir      = os.path.join(top_dir, label)
    cfg           = dict(cfg)
    cfg["output_dir"] = cond_dir
    runner        = GameRunner(cfg)
    _, rows       = runner.run_games(n_games)
    return rows


def _aggregate(rows: list, condition: str, detector_zeroed: str = None) -> dict:
    n = len(rows)
    if n == 0:
        return {"condition": condition, "n_games": 0}

    def _mean(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    village_win_rate = sum(1 for r in rows if r.get("village_win")) / n

    return {
        "condition":              condition,
        "detector_zeroed":        detector_zeroed or "none",
        "n_games":                n,
        "village_win_rate":       round(village_win_rate, 4),
        "wolf_vote_precision":    _mean("wolf_vote_precision"),
        "wolf_vote_recall":       _mean("wolf_vote_recall"),
        "false_accusation_rate":  _mean("false_accusation_rate"),
        "sigma_gap":              _mean("sigma_gap"),
        "combined_gap":           _mean("combined_gap"),
        "mean_d1":                _mean("mean_d1"),
        "mean_d2":                _mean("mean_d2"),
        "mean_d3":                _mean("mean_d3"),
        "mean_d4":                _mean("mean_d4"),
        "mean_d5":                _mean("mean_d5"),
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
    print("=" * 76)
    print("  DETECTOR LEAVE-ONE-OUT ABLATION RESULTS")
    print(f"  {'Condition':<22} {'VillageWin':>11} {'Precision':>10} "
          f"{'Recall':>8} {'SigmaGap':>10}")
    print("-" * 76)
    for r in agg_rows:
        print(
            f"  {r['condition']:<22} "
            f"{r['village_win_rate']:>11.4f} "
            f"{(r['wolf_vote_precision'] or 0):>10.4f} "
            f"{(r['wolf_vote_recall'] or 0):>8.4f} "
            f"{(r['sigma_gap'] or 0):>10.4f}"
        )
    print("=" * 76)

    baseline = next((r for r in agg_rows if r["condition"] == "all_detectors"), None)
    if baseline:
        print("\n  Δ Village win rate vs. all_detectors  (≈ each detector's contribution):")
        for r in agg_rows:
            if r["condition"] == "all_detectors":
                continue
            delta = baseline["village_win_rate"] - r["village_win_rate"]
            key   = r["detector_zeroed"]
            name  = _DETECTOR_NAMES.get(key, key)
            sign  = "+" if delta >= 0 else ""
            print(f"    {key}  {name:<44}  {sign}{delta:.4f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    top_dir = _make_top_dir(args)
    base_cfg = _base_config(args)

    print(f"\n{'='*60}")
    print(f"  Detector Leave-One-Out Ablation")
    print(f"  Experiment  : {args.experiment_name}")
    print(f"  Seed        : {args.seed}")
    print(f"  Games/cond  : {args.games}")
    print(f"  Agent       : {args.agent}")
    print(f"  Output      : {top_dir}")
    print(f"{'='*60}\n")

    # Build conditions: (folder_label, detector_key_to_zero | None)
    conditions = [("all_detectors", None)]
    for key in _DETECTOR_KEYS:
        conditions.append((f"no_{key}", key))   # no_a1 → "no_d1" in spirit

    agg_rows = []
    for label, zeroed_key in conditions:
        print(f"\n--- Condition: {label} ---")
        cfg = dict(base_cfg)
        cfg["suspicion_weights"] = dict(base_cfg["suspicion_weights"])
        if zeroed_key is not None:
            cfg["suspicion_weights"] = _renormalize_without(
                cfg["suspicion_weights"], zeroed_key
            )
            w = cfg["suspicion_weights"]
            print(f"  Weights: a1={w['a1']} a2={w['a2']} a3={w['a3']} "
                  f"a4={w['a4']} a5={w['a5']}")
        rows = _run_condition(label, cfg, args.games, top_dir)
        agg  = _aggregate(rows, label, zeroed_key)
        agg_rows.append(agg)

    _write_ablation_summary(agg_rows, top_dir)
    _print_table(agg_rows)
    print(f"Done. Results in: {top_dir}\n")


if __name__ == "__main__":
    main()
