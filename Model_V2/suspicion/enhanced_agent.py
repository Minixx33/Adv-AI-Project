"""
suspicion/enhanced_agent.py

EnhancedAgent — wraps any compatible AbstractAgent with a SuspicionModule.

Hierarchy:
  AbstractAgent
    └── EnhancedAgent
          ├── _base_agent    : AbstractAgent subclass (Heuristic/Bayesian/MCTS)
          └── _suspicion     : SuspicionModule

Combined decision formula (spec §2):
  score(p) = w1 * base_belief(p) + w2 * σ(p)
  vote()   → argmax score(p) over alive others

The EnhancedAgent delegates ALL non-vote actions to the base agent and
overrides only vote() with the combined score logic.

Constraint (spec §3.9):
  The EnhancedAgent's role must never be WEREWOLF.
  This is enforced at role-assignment time (role_assigner.py), NOT here.

Trace logging:
  After each vote(), a snapshot is stored in _trace_snapshots.
  After each end-of-day update, evolution rows are appended.
  These are retrieved by the runner's logging components.

Compatible base agents: HeuristicAgent, BayesianAgent, MCTSAgent.
NOT compatible: RandomAgent (no beliefs), LogicAgent (certainty-based).
"""

import os as _os
import sys as _sys

_MODEL_V2 = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _MODEL_V2 not in _sys.path:
    _sys.path.insert(0, _MODEL_V2)

from typing import Optional

from werewolf.agent import AbstractAgent
from werewolf.gameinfo import GameInfo, GameSetting
from werewolf.const import Role
from .module import SuspicionModule


_DEFAULT_DECISION_WEIGHTS = {"w1": 0.6, "w2": 0.4}


class EnhancedAgent(AbstractAgent):
    """
    Suspicion-enhanced wrapper around any compatible base agent.

    Parameters
    ----------
    base_agent        : AbstractAgent — the underlying decision-making agent
    detector_weights  : dict | None   — a1..a5 (falls back to module defaults)
    decision_weights  : dict | None   — w1, w2 (defaults: 0.6, 0.4)
    """

    def __init__(
        self,
        base_agent:        AbstractAgent,
        detector_weights:  dict = None,
        decision_weights:  dict = None,
    ):
        super().__init__(
            agent_id=base_agent.agent_id,
            name=f"{base_agent.name}+Suspicion",
            seed=None,
        )
        self._base             = base_agent
        self._decision_weights = dict(_DEFAULT_DECISION_WEIGHTS)
        if decision_weights:
            self._decision_weights.update(decision_weights)
        self._detector_weights = detector_weights  # forwarded to SuspicionModule
        self._suspicion: Optional[SuspicionModule] = None

        # Accumulated outputs for loggers
        self._trace_snapshots: list = []
        self._last_vote_target: Optional[int] = None
        self._last_alive: list = []   # alive_others() at vote() time
        self._true_roles: dict = {}   # set after initialize

    # ------------------------------------------------------------------ #
    #  Lifecycle — delegate to base agent then update module               #
    # ------------------------------------------------------------------ #

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().initialize(game_info, game_setting)
        self._base.initialize(game_info, game_setting)

        n_wolves = game_setting.role_num_map.get(Role.WEREWOLF, 0)
        self._suspicion = SuspicionModule(
            player_ids=list(game_info.status_map.keys()),
            self_id=self.agent_id,
            detector_weights=self._detector_weights,
            n_wolves=n_wolves,
        )
        self._suspicion.observe_talks(game_info.day, game_info.talk_list)

    def update(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().update(game_info, game_setting)
        self._base.update(game_info, game_setting)

        # Feed new observable information to the suspicion module
        self._suspicion.observe_talks(game_info.day, game_info.talk_list)

        # If our divine result arrived (we are the Seer), record it
        if game_info.divine_result is not None:
            dr = game_info.divine_result
            self._suspicion.observe_actual_divine(dr.target, dr.result)

    def day_start(self) -> None:
        self._base.day_start()

    def finish(self) -> None:
        self._base.finish()

    # ------------------------------------------------------------------ #
    #  Talk — delegate to base agent                                       #
    # ------------------------------------------------------------------ #

    def talk(self) -> str:
        return self._base.talk()

    # ------------------------------------------------------------------ #
    #  Vote — combined score override                                      #
    # ------------------------------------------------------------------ #

    def vote(self) -> int:
        alive = self.alive_others()
        if not alive:
            return self.agent_id

        w1 = self._decision_weights["w1"]
        w2 = self._decision_weights["w2"]

        best_pid   = alive[0]
        best_score = -1.0
        for pid in alive:
            belief = self._base.get_wolf_belief(pid)
            sigma  = self._suspicion.get_sigma(pid)
            score  = w1 * belief + w2 * sigma
            if score > best_score:
                best_score = score
                best_pid   = pid

        self._last_vote_target = best_pid
        self._last_alive       = alive  # saved for notify_vote_phase_complete
        return best_pid

    def notify_vote_phase_complete(self, day: int, vote_map: dict) -> None:
        """
        Called by the game engine AFTER all votes are collected for this day.
        Feeds vote data to detectors, updates sigma, then captures the trace
        snapshot — ensuring D-values are non-zero at snapshot time.
        """
        alive = self._last_alive or []

        # 1. Feed complete vote map so D1/D2/D3/D4 counters are updated
        self.observe_votes(day, vote_map)

        # 2. Recompute sigma with the updated detector values
        self.end_of_day_update(
            day=day,
            rnd=0,
            true_roles=self._true_roles,
            alive_pids=alive,
        )

        # 3. Capture trace snapshot with fully updated state
        base_beliefs = {p: self._base.get_wolf_belief(p) for p in alive}
        snap = self._suspicion.snapshot(
            day=day,
            rnd=0,
            base_beliefs=base_beliefs,
            decision_weights=self._decision_weights,
            alive_pids=alive,
            decision_pid=self._last_vote_target,
        )
        self._trace_snapshots.append(snap)

    # ------------------------------------------------------------------ #
    #  Whisper / attack / divine / guard — delegate to base agent          #
    # ------------------------------------------------------------------ #

    def whisper(self) -> str:
        return self._base.whisper()

    def attack(self) -> int:
        return self._base.attack()

    def divine(self) -> int:
        return self._base.divine()

    def guard(self) -> int:
        return self._base.guard()

    # ------------------------------------------------------------------ #
    #  Suspicion module observation bridges                                #
    # ------------------------------------------------------------------ #

    def observe_votes(self, day: int, vote_map: dict) -> None:
        """Called by the runner after each voting phase."""
        self._suspicion.observe_votes(day, vote_map)

    def observe_execution(self, day: int, agent_id: int, revealed_role: str) -> None:
        """Called by the runner when a player is eliminated."""
        self._suspicion.observe_execution(day, agent_id, revealed_role)

    def end_of_day_update(
        self,
        day:           int,
        rnd:           int,
        true_roles:    dict,
        alive_pids:    list,
    ) -> None:
        """Called by the runner at end-of-day to update sigma values."""
        base_beliefs = {
            p: self._base.get_wolf_belief(p)
            for p in alive_pids if p != self.agent_id
        }
        self._suspicion.end_of_day_update(
            day=day,
            rnd=rnd,
            true_roles=true_roles,
            base_beliefs=base_beliefs,
            decision_weights=self._decision_weights,
            alive_pids=alive_pids,
        )

    # ------------------------------------------------------------------ #
    #  Belief interface                                                    #
    # ------------------------------------------------------------------ #

    def get_wolf_belief(self, player_id: int) -> float:
        return self._base.get_wolf_belief(player_id)

    # ------------------------------------------------------------------ #
    #  Logging accessors                                                   #
    # ------------------------------------------------------------------ #

    def get_trace_snapshots(self) -> list:
        return list(self._trace_snapshots)

    def get_evolution_rows(self) -> list:
        return self._suspicion.get_evolution_rows()

    def set_true_roles(self, roles: dict) -> None:
        """Called by runner at game start so logging can show true roles."""
        self._true_roles = roles

    def update_weights(self, detector_weights: dict = None,
                       decision_weights: dict = None) -> None:
        """Hot-reload weights (used during grid search)."""
        if detector_weights and self._suspicion:
            self._suspicion.update_weights(detector_weights)
        if decision_weights:
            self._decision_weights.update(decision_weights)
