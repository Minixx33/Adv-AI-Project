"""
runner/__init__.py

Game runner and logging sub-package.

Hierarchy:
  runner/
    game_runner.py          — GameRunner: runs N games, manages composition
    conversation_logger.py  — Formats game events → conversation.txt (spec §3.4)
    suspicion_tracer.py     — Formats trace snapshots → suspicion_trace.txt (§3.5)
    evolution_logger.py     — Writes suspicion_evolution.csv (spec §3.6)
    grid_search.py          — Two-stage grid search over suspicion weights (§3.8)
"""

from .game_runner import GameRunner
