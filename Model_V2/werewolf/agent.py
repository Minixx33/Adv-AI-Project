"""
werewolf/agent.py

Abstract base class that every agent must subclass.

Hierarchy:
  AbstractAgent  (this file)
    ├── RandomAgent       — werewolf/agents/random_agent.py
    ├── HeuristicAgent    — werewolf/agents/heuristic_agent.py
    ├── BayesianAgent     — werewolf/agents/bayesian_agent.py
    ├── MCTSAgent         — werewolf/agents/mcts_agent.py
    └── LogicAgent        — werewolf/agents/logic_agent.py

  EnhancedAgent wraps any "compatible" AbstractAgent subclass together with
  a SuspicionModule (suspicion/enhanced_agent.py).

Method contract (mirrors AIWolf agent interface):
  initialize(game_info, game_setting)  — called once, before day 1
  update(game_info, game_setting)      — called after every observable event
  day_start()                          — called at top of each day phase
  talk()    → str                      — utterance for the talk phase
  vote()    → int                      — player ID to vote for elimination
  whisper() → str                      — wolf-only: coordination utterance
  attack()  → int                      — wolf-only: player ID to attack
  divine()  → int                      — seer-only: player ID to divine
  guard()   → int                      — bodyguard-only: player ID to protect
  finish()                             — called when the game ends

Agents that don't hold a particular role still need callable fallbacks for
the role-specific methods (divine/attack/guard) because any agent can be
assigned any role.  Default fallbacks return a random alive non-self player.
"""

import random
from abc import ABC, abstractmethod

from .gameinfo import GameInfo, GameSetting
from .const import Role, Status


class AbstractAgent(ABC):
    """
    Base class for all Werewolf agents.
    Subclasses must override vote().
    Role-specific methods (divine, attack, guard) have random-fallback defaults
    so that any agent can be assigned any role without crashing.
    """

    def __init__(self, agent_id: int, name: str = "", seed: int = None):
        self.agent_id:    int         = agent_id
        self.name:        str         = name or self.__class__.__name__
        self.game_info:   GameInfo    = None
        self.game_setting: GameSetting = None
        self._rng = random.Random(seed)
        self._talk_queue: list = []   # token buffer for multi-token talk sequences
        self._wolf_strategy: str = None  # "bus_driver" | "false_claimer" | None
        self._false_claim_done: bool = False  # state for false_claimer strategy

    # ------------------------------------------------------------------ #
    #  Lifecycle hooks                                                     #
    # ------------------------------------------------------------------ #

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        """Called once before the first day. Store state here."""
        self.game_info    = game_info
        self.game_setting = game_setting

    def update(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        """Called after every observable event so the agent can update beliefs."""
        self.game_info = game_info

    def day_start(self) -> None:
        """Called at the start of each day phase. Clears the talk queue."""
        self._talk_queue = []

    def finish(self) -> None:
        """Called when the game ends."""

    # ------------------------------------------------------------------ #
    #  Talk-phase action                                                   #
    # ------------------------------------------------------------------ #

    def talk(self) -> str:
        """
        Return the next utterance for the public talk phase.

        On each call the agent pops from its _talk_queue. When the queue is
        empty it calls _prepare_round_tokens() to generate the next sequence
        (up to 5 tokens including the terminal 'Over' or 'Skip').
        Wolf-strategy tokens take priority over the agent's normal sequence.
        """
        if not self._talk_queue:
            strategy = self._get_wolf_strategy_tokens()
            self._talk_queue = strategy if strategy is not None \
                else list(self._prepare_round_tokens())
        if not self._talk_queue:
            return "Skip"
        token = self._talk_queue.pop(0)
        if token.strip().lower() in ("skip", "over"):
            self._talk_queue = []
        return token

    def _prepare_round_tokens(self) -> list:
        """Generate the token sequence for one round. Override in subclasses."""
        return ["Skip"]

    def set_wolf_strategy(self, strategy: str) -> None:
        """Called by the runner to assign a wolf-only talk strategy."""
        self._wolf_strategy = strategy
        self._false_claim_done = False

    def _get_wolf_strategy_tokens(self):
        """Return wolf-strategy token list, or None to use normal strategy."""
        if self._wolf_strategy is None or self.game_info is None:
            return None
        if self.my_role() != Role.WEREWOLF:
            return None

        if self._wolf_strategy == "bus_driver":
            wolf_mates = [
                pid for pid, role in self.game_info.role_map.items()
                if role == Role.WEREWOLF and pid != self.agent_id
                and self.game_info.status_map.get(pid) == Status.ALIVE
            ]
            if wolf_mates and self._rng.random() < 0.20:
                # Throw a teammate under the bus
                target = self._rng.choice(wolf_mates)
                return [f"ATTACK P{target}", "Over"]
            # Normal accusation toward a non-wolf
            others = self.alive_others()
            non_wolves = [p for p in others
                          if self.game_info.role_map.get(p) != Role.WEREWOLF]
            pool = non_wolves or others
            if pool:
                return [f"ATTACK P{self._rng.choice(pool)}", "Over"]

        elif self._wolf_strategy == "false_claimer":
            if not self._false_claim_done:
                self._false_claim_done = True
                others = self.alive_others()
                non_wolves = [p for p in others
                              if self.game_info.role_map.get(p) != Role.WEREWOLF]
                pool = non_wolves or others
                if pool:
                    fake_target = self._rng.choice(pool)
                    return [
                        f"COMINGOUT P{self.agent_id} SEER",
                        f"DIVINED P{fake_target} WEREWOLF",
                        "Over",
                    ]
            else:
                # Defend the false seer claim in subsequent rounds
                return [f"NOT P{self.agent_id} WEREWOLF", "BECAUSE divine_result", "Over"]

        return None

    # ------------------------------------------------------------------ #
    #  Vote (required for all agents)                                      #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def vote(self) -> int:
        """Return the player ID this agent votes to eliminate."""

    # ------------------------------------------------------------------ #
    #  Wolf-only actions                                                   #
    # ------------------------------------------------------------------ #

    def whisper(self) -> str:
        """Wolf-only: coordination utterance. Default: skip."""
        return "Skip"

    def attack(self) -> int:
        """Wolf-only: return player ID to attack. Default: random non-wolf."""
        return self._random_non_wolf_target()

    # ------------------------------------------------------------------ #
    #  Role-specific actions (Seer, Bodyguard)                             #
    # ------------------------------------------------------------------ #

    def divine(self) -> int:
        """Seer-only: return player ID to divine. Default: random alive other."""
        return self._random_other()

    def guard(self) -> int:
        """Bodyguard-only: return player ID to protect. Default: random alive other."""
        return self._random_other()

    # ------------------------------------------------------------------ #
    #  Belief interface (optional — used by SuspicionModule)               #
    # ------------------------------------------------------------------ #

    def get_wolf_belief(self, player_id: int) -> float:
        """Return P(player_id is a werewolf) in [0, 1]. Default: prior."""
        if self.game_setting is None:
            return 0.5
        n = self.game_setting.player_num
        n_wolves = self.game_setting.role_num_map.get(Role.WEREWOLF, 0)
        return n_wolves / max(n - 1, 1)

    # ------------------------------------------------------------------ #
    #  Convenience helpers                                                 #
    # ------------------------------------------------------------------ #

    def alive_others(self) -> list:
        """Alive players excluding self."""
        if not self.game_info:
            return []
        return [p for p in self.game_info.alive_agents if p != self.agent_id]

    def my_role(self):
        """Own Role enum, or None if not yet initialized."""
        if self.game_info and self.agent_id in self.game_info.role_map:
            return self.game_info.role_map[self.agent_id]
        return None

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _random_other(self) -> int:
        targets = self.alive_others()
        return self._rng.choice(targets) if targets else self.agent_id

    def _random_non_wolf_target(self) -> int:
        """Return a random alive player not known to be a werewolf."""
        if not self.game_info:
            return self.agent_id
        targets = [
            p for p in self.game_info.alive_agents
            if p != self.agent_id
            and self.game_info.role_map.get(p) != Role.WEREWOLF
        ]
        return self._rng.choice(targets) if targets else self.agent_id

    def __repr__(self) -> str:
        return f"{self.name}(P{self.agent_id})"
