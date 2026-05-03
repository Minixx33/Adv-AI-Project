"""
runner/grid_search.py

Two-stage hyperparameter grid search over suspicion module weights (spec §3.8).

Hierarchy / contents:
  GridSearch
    ├── run(output_dir)              — full two-stage search
    ├── _stage1_detector_weights()   — Latin-hypercube over a1..a5
    ├── _stage2_decision_weights()   — sweep w1 from 0..1, w2 = 1 - w1
    ├── _evaluate_config(weights, n_games)  → metrics dict
    └── _write_results(rows, path)

Stage 1 (spec §3.8):
  Vary each of a1..a5 in [0, 0.5] step 0.1.
  Constraint: must sum to 1.0 (normalize after sampling).
  Use Latin Hypercube Sampling to keep total combos ≈ 200.
  Run 2,000 games per configuration.

Stage 2 (spec §3.8):
  Fix best a1..a5 from Stage 1.
  Sweep w1 from 0.0 to 1.0 step 0.1 (w2 = 1 - w1).
  Run 2,000 games per configuration (11 total).

Output CSV columns:
  config_id, a1, a2, a3, a4, a5, w1, w2, games,
  village_winrate, wolf_id_precision, suspicion_calibration
"""

import csv
import os
import random
from itertools import product
from typing import List


class GridSearch:
    """
    Two-stage grid search for optimal suspicion weights.

    Parameters
    ----------
    base_config : dict   — GameRunner config (players, agents, etc.)
    n_games     : int    — games per configuration (spec default: 2000)
    n_samples   : int    — LHS samples for Stage 1 (spec default: 200)
    seed        : int | None
    """

    def __init__(
        self,
        base_config: dict,
        n_games:     int  = 2000,
        n_samples:   int  = 200,
        seed:        int  = None,
    ):
        self._base_config = dict(base_config)
        self._n_games     = n_games
        self._n_samples   = n_samples
        self._rng         = random.Random(seed)

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def run(self, output_dir: str) -> list:
        """
        Run the full two-stage search.
        Writes grid_search_results.csv and returns all result rows.
        """
        os.makedirs(output_dir, exist_ok=True)
        all_rows: list = []

        print("=== Grid Search: Stage 1 — Detector weights (a1..a5) ===")
        stage1_rows = self._stage1_detector_weights(output_dir)
        all_rows.extend(stage1_rows)

        # Best config from Stage 1
        best = max(stage1_rows, key=lambda r: r["village_winrate"])
        print(
            f"\nStage 1 best: a1={best['a1']:.2f} a2={best['a2']:.2f} "
            f"a3={best['a3']:.2f} a4={best['a4']:.2f} a5={best['a5']:.2f} "
            f"→ village_winrate={best['village_winrate']:.3f}"
        )

        print("\n=== Grid Search: Stage 2 — Decision weights (w1, w2) ===")
        stage2_rows = self._stage2_decision_weights(best, output_dir,
                                                     start_id=len(stage1_rows))
        all_rows.extend(stage2_rows)

        # Write combined results
        path = os.path.join(output_dir, "grid_search_results.csv")
        self._write_results(all_rows, path)

        # Print top 5
        top5 = sorted(all_rows, key=lambda r: r["village_winrate"], reverse=True)[:5]
        print("\n=== Top 5 configurations by village win rate ===")
        for i, r in enumerate(top5, 1):
            print(
                f"  {i}. config_id={r['config_id']}  "
                f"a=({r['a1']:.2f},{r['a2']:.2f},{r['a3']:.2f},"
                f"{r['a4']:.2f},{r['a5']:.2f})  "
                f"w=({r['w1']:.2f},{r['w2']:.2f})  "
                f"village_wr={r['village_winrate']:.3f}"
            )

        return all_rows

    # ------------------------------------------------------------------ #
    #  Stage 1                                                             #
    # ------------------------------------------------------------------ #

    def _stage1_detector_weights(self, output_dir: str) -> list:
        configs = self._lhs_detector_configs()
        rows: list = []
        for i, weights in enumerate(configs):
            print(f"  Stage 1 config {i+1}/{len(configs)}...", end="\r")
            metrics = self._evaluate_config(
                detector_weights=weights,
                decision_weights={"w1": 0.6, "w2": 0.4},
            )
            rows.append({
                "config_id":             i,
                "a1": round(weights["a1"], 4),
                "a2": round(weights["a2"], 4),
                "a3": round(weights["a3"], 4),
                "a4": round(weights["a4"], 4),
                "a5": round(weights["a5"], 4),
                "w1": 0.6, "w2": 0.4,
                "games":                 self._n_games,
                "village_winrate":       round(metrics["village_winrate"], 4),
                "wolf_id_precision":     round(metrics["wolf_id_precision"], 4),
                "suspicion_calibration": round(metrics["suspicion_calibration"], 4),
            })
        print()
        return rows

    def _lhs_detector_weights(self) -> list:
        """
        Latin Hypercube Sampling: generate n_samples detector weight vectors,
        each normalised to sum to 1.
        Each of a1..a5 drawn from [0, 0.5] stratum.
        """
        n = self._n_samples
        k = 5  # number of weights

        # Create k permutations of [0..n-1]
        perms = [list(range(n)) for _ in range(k)]
        for p in perms:
            self._rng.shuffle(p)

        samples = []
        for i in range(n):
            # Sample mid-point of each stratum
            raw = [(perms[j][i] + self._rng.random()) / n * 0.5 for j in range(k)]
            total = sum(raw)
            if total == 0:
                total = 1.0
            normalised = [v / total for v in raw]
            samples.append(normalised)
        return samples

    def _lhs_detector_configs(self) -> list:
        """Return list of {a1..a5} dicts from LHS sampling."""
        samples = self._lhs_detector_weights()
        keys = ["a1", "a2", "a3", "a4", "a5"]
        return [dict(zip(keys, s)) for s in samples]

    # ------------------------------------------------------------------ #
    #  Stage 2                                                             #
    # ------------------------------------------------------------------ #

    def _stage2_decision_weights(
        self,
        best_detector: dict,
        output_dir:    str,
        start_id:      int = 200,
    ) -> list:
        rows: list = []
        w1_values = [round(i * 0.1, 1) for i in range(11)]   # 0.0 .. 1.0
        for j, w1 in enumerate(w1_values):
            w2 = round(1.0 - w1, 1)
            print(f"  Stage 2 config {j+1}/11 (w1={w1:.1f})...", end="\r")
            metrics = self._evaluate_config(
                detector_weights={
                    k: best_detector[k] for k in ("a1","a2","a3","a4","a5")
                },
                decision_weights={"w1": w1, "w2": w2},
            )
            rows.append({
                "config_id":             start_id + j,
                "a1": best_detector["a1"],
                "a2": best_detector["a2"],
                "a3": best_detector["a3"],
                "a4": best_detector["a4"],
                "a5": best_detector["a5"],
                "w1": w1, "w2": w2,
                "games":                 self._n_games,
                "village_winrate":       round(metrics["village_winrate"], 4),
                "wolf_id_precision":     round(metrics["wolf_id_precision"], 4),
                "suspicion_calibration": round(metrics["suspicion_calibration"], 4),
            })
        print()
        return rows

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def _evaluate_config(
        self,
        detector_weights: dict,
        decision_weights: dict,
    ) -> dict:
        """
        Run self._n_games games with the given weights.
        Returns {village_winrate, wolf_id_precision, suspicion_calibration}.
        """
        from .game_runner import GameRunner

        config = dict(self._base_config)
        config["suspicion_weights"] = {**detector_weights, **decision_weights}

        # Use a small output dir to avoid filling disk during grid search
        config["output_dir"] = os.path.join(
            self._base_config.get("output_dir", "./results"),
            "_gs_tmp"
        )

        runner = GameRunner(config)
        _, summary_rows = runner.run_games(self._n_games)

        if not summary_rows:
            return {"village_winrate": 0.5, "wolf_id_precision": 0.0,
                    "suspicion_calibration": 0.0}

        village_wins = sum(1 for r in summary_rows if r["winner"] == "VILLAGE")
        total        = len(summary_rows)
        village_wr   = village_wins / total

        # Wolf identification precision: fraction of votes that targeted real wolves
        total_votes   = sum(r["total_votes"] for r in summary_rows)
        correct_votes = sum(r["correct_wolf_votes"] for r in summary_rows)
        wolf_precision = correct_votes / total_votes if total_votes > 0 else 0.0

        # Suspicion calibration: fraction of games where suspicion agent survived
        # and their team won (rough proxy for calibration quality)
        relevant = [r for r in summary_rows if r["suspicion_agent_id"]]
        if relevant:
            calibrated = sum(1 for r in relevant if r["suspicion_agent_won"])
            calib = calibrated / len(relevant)
        else:
            calib = 0.5

        return {
            "village_winrate":       village_wr,
            "wolf_id_precision":     wolf_precision,
            "suspicion_calibration": calib,
        }

    # ------------------------------------------------------------------ #
    #  Output                                                              #
    # ------------------------------------------------------------------ #

    def _write_results(self, rows: list, path: str) -> None:
        fields = [
            "config_id", "a1", "a2", "a3", "a4", "a5", "w1", "w2",
            "games", "village_winrate", "wolf_id_precision",
            "suspicion_calibration",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nGrid search results saved → {path}")
