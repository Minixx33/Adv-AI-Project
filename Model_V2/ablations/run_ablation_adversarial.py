"""
ablations/run_ablation_adversarial.py

Adversarial wolf strategy ablation for the Werewolf Suspicion module.

Tests whether the suspicion module catches wolves using deceptive strategies,
directly answering RQ3. Runs 6 conditions (3 wolf strategies × 2 suspicion states):

  standard_no_susp       No wolf strategy,   no suspicion module
  standard_with_susp     No wolf strategy,   suspicion active
  bus_driver_no_susp     Bus-driver wolves,  no suspicion module
  bus_driver_with_susp   Bus-driver wolves,  suspicion active     <- tests D4
  false_claimer_no_susp  False-claimer wolves, no suspicion
  false_claimer_with_susp False-claimer wolves, suspicion active  <- tests D5

Bus-driver wolf:   votes against own wolf-partner, exploiting social trust (D4).
False-claimer wolf: claims Seer role to deflect suspicion (D5).

The key comparison is: for each adversarial strategy, how much does adding
suspicion improve village win rate? A large delta confirms the module catches
that behavior.

Output structure:
  {output}/{experiment_name}_seed{seed}_{YYYY-MM-DD_HH-MM-SS}/
      standard_no_susp/
          summary.csv
          config_used.json
          run_*.log
      standard_with_susp/
          ...
      bus_driver_no_susp/
          ...
      ...
      ablation_summary.csv

Usage:
  python ablations/run_ablation_adversarial.py \\
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

_DEFAULT_WEIGHTS = {
    "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
    "w1": 0.60, "w2": 0.40,
}

# (label, wolf_strategies dict, use_suspicion)
_CONDITIONS = [
    ("standard_no_susp",        {},                                                           False),
    ("standard_with_susp",      {},                                                           True),
    ("bus_driver_no_susp",      {"bus_driver_probability": 1.0, "false_claimer_probability": 0.0}, False),
    ("bus_driver_with_susp",    {"bus_driver_probability": 1.0, "false_claimer_probability": 0.0}, True),
    ("false_claimer_no_susp",   {"bus_driver_probability": 0.0, "false_claimer_probability": 1.0}, False),
    ("false_claimer_with_susp", {"bus_driver_probability": 0.0, "false_claimer_probability": 1.0}, True),
]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Adversarial wolf strategy ablation (RQ3)"
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


def _build_config(args, wolf_strategies: dict, use_suspicion: bool) -> dict:
    """
    Build config for one condition.

    With suspicion:    6 suspicion-enhanced + 2 base agents.
    Without suspicion: 8 base agents, suspicion_agent_type = "none".

    With 8 players (2 wolves) the role_assigner allows at most 6 suspicion
    agents (6 non-wolf roles: 4 Villager, 1 Seer, 1 Possessed).
    """
    agent     = args.agent
    n_workers = args.workers or min(os.cpu_count() or 4, 4)

    if use_suspicion:
        composition  = {f"{agent}_with_suspicion": 6, agent: 2}
        susp_type    = agent
        susp_weights = dict(_DEFAULT_WEIGHTS)
    else:
        composition  = {agent: 8}
        susp_type    = "none"
        susp_weights = {}

    cfg = {
        "players":              8,
        "seed":                 args.seed,
        "suspicion_agent_type": susp_type,
        "agent_composition":    composition,
        "n_workers":            n_workers,
        "max_talk_rounds":      5,
        "mcts_simulations":     500,
        "mcts_depth":           5,
        "write_xlsx":           False,
    }
    if susp_weights:
        cfg["suspicion_weights"] = susp_weights
    if wolf_strategies:
        cfg["wolf_strategies"] = wolf_strategies
    return cfg


# ── Run helpers ──────────────────────────────────────────────────────────────

def _run_condition(label: str, cfg: dict, n_games: int, top_dir: str) -> list:
    cond_dir      = os.path.join(top_dir, label)
    cfg           = dict(cfg)
    cfg["output_dir"] = cond_dir
    runner        = GameRunner(cfg)
    _, rows       = runner.run_games(n_games)
    return rows


def _aggregate(rows: list, condition: str,
               wolf_strategy: str, use_suspicion: bool) -> dict:
    n = len(rows)
    if n == 0:
        return {"condition": condition, "n_games": 0}

    def _mean(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    village_win_rate = sum(1 for r in rows if r.get("village_win")) / n

    return {
        "condition":              condition,
        "wolf_strategy":          wolf_strategy,
        "suspicion_active":       use_suspicion,
        "n_games":                n,
        "village_win_rate":       round(village_win_rate, 4),
        "wolf_vote_precision":    _mean("wolf_vote_precision"),
        "wolf_vote_recall":       _mean("wolf_vote_recall"),
        "false_accusation_rate":  _mean("false_accusation_rate"),
        "sigma_gap":              _mean("sigma_gap"),
        "combined_gap":           _mean("combined_gap"),
        # Detector averages — meaningful for suspicion conditions
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
    print("=" * 80)
    print("  ADVERSARIAL WOLF ABLATION RESULTS  (RQ3)")
    print(f"  {'Condition':<28} {'VillageWin':>11} {'D4':>7} {'D5':>7} {'SigmaGap':>10}")
    print("-" * 80)
    for r in agg_rows:
        print(
            f"  {r['condition']:<28} "
            f"{r['village_win_rate']:>11.4f} "
            f"{(r['mean_d4'] or 0):>7.4f} "
            f"{(r['mean_d5'] or 0):>7.4f} "
            f"{(r['sigma_gap'] or 0):>10.4f}"
        )
    print("=" * 80)

    # Show suspicion benefit (Δ) per wolf strategy
    print("\n  Suspicion benefit per wolf strategy (with_susp − no_susp):")
    strategies = ["standard", "bus_driver", "false_claimer"]
    for strat in strategies:
        no_s  = next((r for r in agg_rows
                      if r["wolf_strategy"] == strat and not r["suspicion_active"]), None)
        yes_s = next((r for r in agg_rows
                      if r["wolf_strategy"] == strat and r["suspicion_active"]), None)
        if no_s and yes_s:
            delta = yes_s["village_win_rate"] - no_s["village_win_rate"]
            sign  = "+" if delta >= 0 else ""
            d4_delta = (yes_s.get("mean_d4") or 0) - (no_s.get("mean_d4") or 0)
            d5_delta = (yes_s.get("mean_d5") or 0) - (no_s.get("mean_d5") or 0)
            print(f"    {strat:<16}  Δ village_win={sign}{delta:.4f}  "
                  f"Δ D4={d4_delta:+.4f}  Δ D5={d5_delta:+.4f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    top_dir = _make_top_dir(args)

    print(f"\n{'='*60}")
    print(f"  Adversarial Wolf Ablation  (RQ3)")
    print(f"  Experiment  : {args.experiment_name}")
    print(f"  Seed        : {args.seed}")
    print(f"  Games/cond  : {args.games}")
    print(f"  Agent       : {args.agent}")
    print(f"  Conditions  : {len(_CONDITIONS)}")
    print(f"  Output      : {top_dir}")
    print(f"{'='*60}\n")

    # Map label → wolf_strategy string for the aggregate row
    def _wolf_strategy(wolf_cfg: dict) -> str:
        if not wolf_cfg:
            return "standard"
        if wolf_cfg.get("bus_driver_probability", 0) > 0:
            return "bus_driver"
        if wolf_cfg.get("false_claimer_probability", 0) > 0:
            return "false_claimer"
        return "standard"

    agg_rows = []
    for label, wolf_cfg, use_susp in _CONDITIONS:
        print(f"\n--- Condition: {label} ---")
        cfg  = _build_config(args, wolf_cfg, use_susp)
        rows = _run_condition(label, cfg, args.games, top_dir)
        agg  = _aggregate(rows, label, _wolf_strategy(wolf_cfg), use_susp)
        agg_rows.append(agg)

    _write_ablation_summary(agg_rows, top_dir)
    _print_table(agg_rows)
    print(f"Done. Results in: {top_dir}\n")


if __name__ == "__main__":
    main()
