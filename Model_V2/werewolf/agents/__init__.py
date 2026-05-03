"""
werewolf/agents/__init__.py

Agent subpackage.  Exports all concrete agent classes.

Hierarchy:
  AbstractAgent  (werewolf/agent.py)
    ├── RandomAgent       — floor baseline; uniform-random decisions
    ├── HeuristicAgent    — accusation-count heuristic; compatible with suspicion
    ├── BayesianAgent     — probabilistic role beliefs; compatible with suspicion
    ├── MCTSAgent         — Monte Carlo Tree Search; compatible with suspicion
    └── LogicAgent        — forward-chaining knowledge base; NOT compatible with suspicion
"""

from .random_agent   import RandomAgent
from .heuristic_agent import HeuristicAgent
from .bayesian_agent  import BayesianAgent
from .mcts_agent      import MCTSAgent
from .logic_agent     import LogicAgent

__all__ = [
    "RandomAgent",
    "HeuristicAgent",
    "BayesianAgent",
    "MCTSAgent",
    "LogicAgent",
]
