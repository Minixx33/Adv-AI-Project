"""
runner/evolution_logger.py

Writes the per-round suspicion/trust evolution CSV specified in §3.6.

Hierarchy / contents:
  EvolutionLogger
    └── write(evolution_rows, game_id, output_dir)
          Writes <output_dir>/game_<id>/suspicion_evolution.csv

CSV columns (spec §3.6):
  day, round, target_player, target_role, sigma, trust, belief_wolf,
  combined_score, d1, d2, d3, d4, d5, E, delta_sigma
"""

import csv
import os


_CSV_FIELDS = [
    "day", "round", "target_player", "target_role",
    "sigma", "trust", "belief_wolf", "combined_score",
    "d1", "d2", "d3", "d4", "d5", "E", "delta_sigma",
]


class EvolutionLogger:
    """Writes suspicion_evolution.csv from accumulated evolution rows."""

    def write(
        self,
        evolution_rows: list,
        game_id:        int,
        output_dir:     str,
    ) -> str:
        """
        Save the evolution CSV.  Returns path to the written file.
        evolution_rows comes from EnhancedAgent.get_evolution_rows().
        """
        folder = os.path.join(output_dir, f"game_{game_id:04d}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "suspicion_evolution.csv")

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(evolution_rows)

        return path
