"""
ablations/run_ablation_learner.py

Online-learner ablation for the Werewolf Suspicion module.

Compares two conditions:
  fixed_weights    Detector weights (a1-a5) frozen at defaults for all N games.
                   Uses standard parallel GameRunner (fast).
  online_learner   Detector weights updated by OnlineLearner after every game
                   (gradient-descent on MSE: wolf=1, non-wolf=0).
                   Must run sequentially so weights propagate game-to-game.

NOTE: The standard parallel GameRunner does not thread the OnlineLearner
through worker processes, so the online-learner condition here uses a
sequential loop — it will be slower than the fixed-weights run.
Consider using fewer --games (e.g. 1000-2000) if time is limited;
the learning curve is usually visible within 1000 games.

Output structure:
  {output}/{experiment_name}_seed{seed}_{YYYY-MM-DD_HH-MM-SS}/
      fixed_weights/
          summary.csv
          config_used.json
          run_*.log
          weight_history.csv   (empty — weights never changed)
      online_learner/
          summary.csv
          config_used.json
          run_*.log
          weight_history.csv   <- per-game weight trajectory
      ablation_summary.csv

Usage:
  python ablations/run_ablation_learner.py \\
      --experiment-name yasmi \\
      --seed 42 \\
      --games 2000 \\
      --agent bayesian \\
      --output ./results/ablations
"""

import argparse
import csv
import datetime
import json
import os
import sys

_MODEL_V2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODEL_V2 not in sys.path:
    sys.path.insert(0, _MODEL_V2)

from runner.game_runner import GameRunner, _run_one_game
from suspicion.online_learner import OnlineLearner

_DEFAULT_WEIGHTS = {
    "a1": 0.30, "a2": 0.15, "a3": 0.20, "a4": 0.20, "a5": 0.15,
    "w1": 0.60, "w2": 0.40,
}


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Online-learner vs. fixed-weights ablation"
    )
    p.add_argument("--experiment-name", required=True, dest="experiment_name",
                   help="Label for output folder (e.g. 'yasmi')")
    p.add_argument("--seed", type=int, required=True,
                   help="Base random seed for reproducibility")
    p.add_argument("--games", type=int, default=2000,
                   help="Games per condition (default: 2000; online-learner runs sequentially)")
    p.add_argument("--agent", default="bayesian",
                   choices=["bayesian", "heuristic", "mcts"],
                   help="Agent type for suspicion-enhanced players (default: bayesian)")
    p.add_argument("--output", default="./results/ablations",
                   help="Root output directory (default: ./results/ablations)")
    p.add_argument("--workers", type=int, default=None,
                   help="Parallel workers for fixed-weights condition (default: min(cpu_count, 4))")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Online-learner learning rate (default: 0.01)")
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
        "suspicion_weights":    dict(_DEFAULT_WEIGHTS),
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


# ── Fixed-weights condition ──────────────────────────────────────────────────

def _run_fixed(cfg: dict, n_games: int, top_dir: str) -> list:
    """Standard parallel GameRunner — weights never change."""
    cond_dir      = os.path.join(top_dir, "fixed_weights")
    cfg           = dict(cfg)
    cfg["output_dir"] = cond_dir
    runner        = GameRunner(cfg)
    _, rows       = runner.run_games(n_games)

    # Write an empty weight history to keep output structure consistent
    hist_path = os.path.join(cond_dir, "weight_history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["game_id", "a1", "a2", "a3", "a4", "a5"])
        writer.writeheader()
    return rows


# ── Online-learner condition ─────────────────────────────────────────────────

def _run_online(cfg: dict, n_games: int, top_dir: str, lr: float) -> list:
    """
    Sequential loop: run each game, feed evolution rows to the OnlineLearner,
    update suspicion_weights in the config, repeat.

    Because _run_one_game is a multiprocessing-safe top-level function that
    reads suspicion_weights from the config dict passed to it, we can update
    the dict between calls to propagate the learned weights.
    """
    cond_dir = os.path.join(top_dir, "online_learner")
    os.makedirs(cond_dir, exist_ok=True)

    cfg = dict(cfg)
    cfg["output_dir"]  = cond_dir
    cfg["suspicion_weights"] = dict(_DEFAULT_WEIGHTS)

    learner = OnlineLearner(
        initial_weights=dict(cfg["suspicion_weights"]),
        lr=lr,
    )

    # We need a GameRunner instance only for _build_summary_row and _write_summary
    runner = GameRunner(cfg)

    summary_rows: list = []
    weight_history: list = []

    print_every = max(1, n_games // 20)
    print(f"  Running {n_games} games sequentially (online learner active)...")

    for gid in range(n_games):
        result = _run_one_game((cfg, gid))
        record, roles, all_evo_rows, first_susp, pid_to_type, susp_pids = result

        # Build summary row
        row = runner._build_summary_row(
            record, roles, first_susp, pid_to_type,
            all_evo_rows=all_evo_rows, susp_pids=susp_pids,
        )
        summary_rows.append(row)

        # Update learner and propagate new weights into config
        if all_evo_rows:
            new_weights = learner.update(gid, all_evo_rows)
            cfg["suspicion_weights"].update(new_weights)
            weight_history.append({
                "game_id": gid,
                **{k: round(new_weights.get(k, _DEFAULT_WEIGHTS[k]), 6)
                   for k in ["a1", "a2", "a3", "a4", "a5"]},
            })

        if (gid + 1) % print_every == 0 or gid + 1 == n_games:
            pct = (gid + 1) / n_games * 100
            print(f"    {gid+1}/{n_games} games  ({pct:.0f}%)", flush=True)

    # Write summary.csv
    runner._write_summary(summary_rows, cond_dir)
    runner._write_config_echo(cond_dir)

    # Write weight history
    hist_path = os.path.join(cond_dir, "weight_history.csv")
    if weight_history:
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(weight_history[0].keys()))
            writer.writeheader()
            writer.writerows(weight_history)

    # Print aggregate (reuse runner method)
    runner._print_aggregate(summary_rows, cond_dir)

    # Show final learned weights
    final_w = cfg["suspicion_weights"]
    print("  Final learned weights:")
    for k in ["a1", "a2", "a3", "a4", "a5"]:
        print(f"    {k} = {final_w.get(k, _DEFAULT_WEIGHTS[k]):.6f}  "
              f"(started at {_DEFAULT_WEIGHTS[k]})")

    return summary_rows


# ── Aggregate helpers ────────────────────────────────────────────────────────

def _aggregate(rows: list, condition: str) -> dict:
    n = len(rows)
    if n == 0:
        return {"condition": condition, "n_games": 0}

    def _mean(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    village_win_rate = sum(1 for r in rows if r.get("village_win")) / n

    return {
        "condition":              condition,
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
    print("  ONLINE-LEARNER ABLATION RESULTS")
    print(f"  {'Condition':<22} {'VillageWin':>11} {'Precision':>10} "
          f"{'Recall':>8} {'SigmaGap':>10}")
    print("-" * 72)
    for r in agg_rows:
        print(
            f"  {r['condition']:<22} "
            f"{r['village_win_rate']:>11.4f} "
            f"{(r['wolf_vote_precision'] or 0):>10.4f} "
            f"{(r['wolf_vote_recall'] or 0):>8.4f} "
            f"{(r['sigma_gap'] or 0):>10.4f}"
        )
    print("=" * 72)
    if len(agg_rows) == 2:
        delta = agg_rows[1]["village_win_rate"] - agg_rows[0]["village_win_rate"]
        sign  = "+" if delta >= 0 else ""
        print(f"\n  Online-learner Δ village win rate vs fixed: {sign}{delta:.4f}")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args    = _parse_args()
    top_dir = _make_top_dir(args)
    base_cfg = _base_config(args)

    print(f"\n{'='*60}")
    print(f"  Online-Learner Ablation")
    print(f"  Experiment  : {args.experiment_name}")
    print(f"  Seed        : {args.seed}")
    print(f"  Games/cond  : {args.games}")
    print(f"  Agent       : {args.agent}")
    print(f"  LR          : {args.lr}")
    print(f"  Output      : {top_dir}")
    print(f"  Note: online-learner condition runs sequentially (no parallel workers)")
    print(f"{'='*60}\n")

    agg_rows = []

    # ── Condition 1: fixed weights ──
    print("\n--- Condition: fixed_weights (parallel) ---")
    rows_fixed = _run_fixed(base_cfg, args.games, top_dir)
    agg_rows.append(_aggregate(rows_fixed, "fixed_weights"))

    # ── Condition 2: online learner ──
    print("\n--- Condition: online_learner (sequential) ---")
    rows_online = _run_online(base_cfg, args.games, top_dir, args.lr)
    agg_rows.append(_aggregate(rows_online, "online_learner"))

    _write_ablation_summary(agg_rows, top_dir)
    _print_table(agg_rows)
    print(f"Done. Results in: {top_dir}\n")
    print(f"  Tip: plot weight_history.csv from online_learner/ to show "
          f"how a1-a5 converge over {args.games} games.\n")


if __name__ == "__main__":
    main()
