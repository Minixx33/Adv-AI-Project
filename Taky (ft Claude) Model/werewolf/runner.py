"""werewolf.runner — Experiment runner and CSV logger.

This module wires together the game engine and the four agent types to
produce statistically meaningful results across many games.

Design decisions
----------------
* Fresh agent instances are created for every game via ``make_agents()`` so
  that per-game state never bleeds across games.
* The 5-player lineup uses one instance of each of the four agent types plus
  a second ``RandomAgent`` as the fifth slot, balancing the team slightly
  against the wolf team.
* Role assignment is random every game (shuffled inside ``WerewolfGame``),
  so each agent type experiences every role proportionally over many runs.
* CSV rows are flushed every game (not buffered) so the file is usable even
  if a long run is interrupted.

CSV output schema
-----------------
+----------------------+-------+-----------------------------------------------+
| Column               | Type  | Description                                   |
+======================+=======+===============================================+
| game_id              | int   | Sequential game number (1-based).             |
+----------------------+-------+-----------------------------------------------+
| winner               | str   | 'village' or 'wolf'.                          |
+----------------------+-------+-----------------------------------------------+
| n_rounds             | int   | Number of day-phase rounds played.            |
+----------------------+-------+-----------------------------------------------+
| agent_name           | str   | Agent class name.                             |
+----------------------+-------+-----------------------------------------------+
| agent_idx            | int   | Player slot (1–5).                            |
+----------------------+-------+-----------------------------------------------+
| agent_role           | str   | Role assigned (e.g. 'WEREWOLF').              |
+----------------------+-------+-----------------------------------------------+
| survived             | 0/1   | 1 if alive at game end.                       |
+----------------------+-------+-----------------------------------------------+
| votes_cast           | int   | Total votes across all rounds.                |
+----------------------+-------+-----------------------------------------------+
| correct_wolf_votes   | int   | Votes that eliminated the Werewolf.           |
+----------------------+-------+-----------------------------------------------+
| false_accusations    | int   | Votes that eliminated a non-wolf player.      |
+----------------------+-------+-----------------------------------------------+
| final_belief_wolf    | float | Max wolf belief at game end (Bayesian /       |
|                      |       | Suspicion agents); empty for others.          |
+----------------------+-------+-----------------------------------------------+
| final_sigma_wolf     | float | Max sigma at game end (Suspicion agent only); |
|                      |       | empty for all others.                         |
+----------------------+-------+-----------------------------------------------+

Classes
-------
GameRunner : Orchestrates N-game experiments and produces the CSV output.
"""
from __future__ import annotations

import csv
import os
import random
import time
from collections import defaultdict
from typing import List, Optional

from werewolf.agent import AbstractAgent
from werewolf.agents.random_agent import RandomAgent
from werewolf.agents.heuristic_agent import HeuristicAgent
from werewolf.agents.bayesian_agent import BayesianAgent
from werewolf.agents.suspicion_agent import SuspicionAgent
from werewolf.game import WerewolfGame
from werewolf.const import Role, Status


_CSV_HEADER: List[str] = [
    "game_id",
    "winner",
    "n_rounds",
    "agent_name",
    "agent_idx",
    "agent_role",
    "survived",
    "votes_cast",
    "correct_wolf_votes",
    "false_accusations",
    "final_belief_wolf",
    "final_sigma_wolf",
]
"""Ordered list of column names for the output CSV."""


class GameRunner:
    """Orchestrates N-game experiments and writes results to a CSV file.

    Typical usage::

        runner = GameRunner(output_path="results/results.csv", n_games=10000)
        runner.run_experiments()
        GameRunner.summarise("results/results.csv")

    Attributes
    ----------
    _output_path : str
        Filesystem path for the CSV output file.
    _n_games     : int
        Total number of games to simulate.
    """

    def __init__(
        self,
        output_path: str = "results/results.csv",
        n_games: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the runner.

        Args:
            output_path: Path where the CSV will be written.  Parent
                         directories are created automatically.
            n_games:     Number of independent games to simulate.
            seed:        Optional integer seed for ``random.seed()``.
                         Set this for reproducible experiments.
        """
        self._output_path = output_path
        self._n_games     = n_games
        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Agent factory
    # ------------------------------------------------------------------

    @staticmethod
    def make_agents() -> List[AbstractAgent]:
        """Create one fresh agent instance per player slot.

        Returns a new list every call so agents never share state across games.

        Lineup (slot order — roles are shuffled randomly each game):

        +------+-------------------+
        | Slot | Agent type        |
        +======+===================+
        |  1   | RandomAgent       |
        +------+-------------------+
        |  2   | HeuristicAgent    |
        +------+-------------------+
        |  3   | BayesianAgent     |
        +------+-------------------+
        |  4   | SuspicionAgent    |
        +------+-------------------+
        |  5   | RandomAgent (2nd) |
        +------+-------------------+

        Returns:
            A list of 5 fresh ``AbstractAgent`` instances.
        """
        return [
            RandomAgent(),
            HeuristicAgent(),
            BayesianAgent(),
            SuspicionAgent(),
            RandomAgent(),
        ]

    # ------------------------------------------------------------------
    # Main experiment loop
    # ------------------------------------------------------------------

    def run_experiments(self) -> None:
        """Simulate ``n_games`` games and write one CSV row per agent per game.

        Progress is printed every 500 games.  The CSV file is opened in write
        mode, so any existing file at ``output_path`` is overwritten.

        Raises:
            OSError: If the output path is not writable.
        """
        os.makedirs(os.path.dirname(self._output_path) or ".", exist_ok=True)
        start = time.time()

        with open(self._output_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADER)
            writer.writeheader()

            for game_id in range(1, self._n_games + 1):
                agents = self.make_agents()
                game   = WerewolfGame(agents)
                winner = game.run()

                writer.writerows(self._build_rows(game_id, winner, game))

                if game_id % 500 == 0:
                    elapsed = time.time() - start
                    print(
                        f"  {game_id:>6}/{self._n_games}  "
                        f"elapsed={elapsed:.1f}s  "
                        f"last_winner={winner}"
                    )

        elapsed = time.time() - start
        print(f"\nDone. {self._n_games} games in {elapsed:.1f}s -> {self._output_path}")

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------

    def _build_rows(
        self, game_id: int, winner: str, game: WerewolfGame
    ) -> List[dict]:
        """Build one CSV row per agent from a completed game.

        Args:
            game_id: Sequential game identifier (1-based).
            winner:  The winning team string from ``WerewolfGame.run()``.
            game:    The completed ``WerewolfGame`` instance.

        Returns:
            A list of 5 dicts (one per player slot) ready for ``csv.DictWriter``.
        """
        rows      = []
        role_map  = game.role_map
        agent_map = game.agent_map

        for agent_obj, impl in agent_map.items():
            survived = game._status_map.get(agent_obj) == Status.ALIVE
            stats    = impl.get_stats()
            rows.append({
                "game_id":            game_id,
                "winner":             winner,
                "n_rounds":           game.round_count,
                "agent_name":         impl.name,
                "agent_idx":          agent_obj.agent_idx,
                "agent_role":         role_map[agent_obj].value,
                "survived":           int(survived),
                "votes_cast":         stats.get("votes_cast", 0),
                "correct_wolf_votes": stats.get("correct_wolf_votes", 0),
                "false_accusations":  stats.get("false_accusations", 0),
                "final_belief_wolf":  stats.get("final_belief_wolf"),
                "final_sigma_wolf":   stats.get("final_sigma_wolf"),
            })
        return rows

    # ------------------------------------------------------------------
    # Quick summary (no pandas required)
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(output_path: str = "results/results.csv") -> None:
        """Print a win-rate table grouped by agent type and role.

        Reads the CSV written by ``run_experiments()`` and computes the
        fraction of games won by each (agent_type, role) combination.

        A game is a win for an agent when:

        * the agent is on the village team and ``winner == 'village'``, or
        * the agent is on the wolf team (WEREWOLF or POSSESSED) and
          ``winner == 'wolf'``.

        Args:
            output_path: Path to the CSV file produced by ``run_experiments()``.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
        """
        wins:  dict = defaultdict(int)
        games: dict = defaultdict(int)

        with open(output_path, newline="") as fh:
            for row in csv.DictReader(fh):
                key    = (row["agent_name"], row["agent_role"])
                role   = row["agent_role"]
                winner = row["winner"]
                team   = "wolf" if role in ("WEREWOLF", "POSSESSED") else "village"
                games[key] += 1
                if winner == team:
                    wins[key] += 1

        print(f"\n{'Agent':<22} {'Role':<12} {'Wins':>6} {'Games':>6} {'WinRate':>8}")
        print("-" * 60)
        for key in sorted(games):
            agent_name, role = key
            g = games[key]
            w = wins[key]
            print(f"{agent_name:<22} {role:<12} {w:>6} {g:>6} {w/g:>8.3f}")
