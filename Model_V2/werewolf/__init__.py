"""
werewolf/__init__.py

Package root for the in-process Werewolf game engine.

Module hierarchy:
  werewolf/
    const.py          — Role, Status, Species, Team enums; ROLE_CONFIG
    gameinfo.py       — Talk, Vote, Judge, GameInfo, GameSetting data classes
    agent.py          — AbstractAgent base class
    role_assigner.py  — Role assignment with suspicion-not-wolf constraint
    game.py           — WerewolfGame engine (runs one complete game)
    agents/
      random_agent.py     — RandomAgent (floor baseline)
      heuristic_agent.py  — HeuristicAgent (accusation tracking)
      bayesian_agent.py   — BayesianAgent (probabilistic beliefs)
      mcts_agent.py       — MCTSAgent (Monte Carlo Tree Search)
      logic_agent.py      — LogicAgent (forward-chaining KB)
"""

from .const import Role, Species, Status, Team, ROLE_CONFIG, ROLE_TO_SPECIES, ROLE_TO_TEAM
from .gameinfo import Talk, Vote, Judge, GameInfo, GameSetting
from .agent import AbstractAgent
from .game import WerewolfGame
