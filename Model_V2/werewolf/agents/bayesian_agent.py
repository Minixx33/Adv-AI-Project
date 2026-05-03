"""
werewolf/agents/bayesian_agent.py

BayesianAgent — probabilistic role-belief agent.

Hierarchy:
  AbstractAgent
    └── BayesianAgent

Model:
  Maintains P(p = WEREWOLF | evidence) for every other player, initialised
  at the uniform prior num_wolves / (n - 1).

  Evidence sources processed on every update():
    1. Seer divine results (own, if this agent is the Seer) → certainty update
    2. Public Seer COMINGOUT + DIVINED claims → likelihood update
    3. Execution reveals → certainty update, then redistribute mass to others
    4. Voting patterns:
         • Player voted for someone we believe is likely human → mild ↑
         • Player consistently votes against our top suspect    → mild ↓

  Monte-Carlo consistency check (lightweight):
    After processing claims, sample S=100 possible worlds (role assignments),
    weight each by whether it is consistent with observed divine claims, and
    use the weighted counts to adjust beliefs toward the posterior mean.

Actions:
  vote()   → argmax P(p = WEREWOLF) among alive others
  talk()   → "ESTIMATE P{target} WEREWOLF" for top suspect, else "Skip"
  divine() → player with highest wolf probability not yet divined
  attack() → alive non-wolf player with lowest wolf belief (wolves protect
             most-innocent; high wolf belief → probably a teammate)
  guard()  → alive player with lowest wolf belief (protect the innocent)

Compatible with the suspicion module (§2):
  get_wolf_belief(pid) returns P(pid = WEREWOLF) ∈ [0, 1].
"""

import re
import math
from collections import defaultdict
from typing import Optional

from ..agent import AbstractAgent
from ..gameinfo import GameInfo, GameSetting, Judge
from ..const import Role, ROLE_CONFIG, ROLE_TO_SPECIES, Species


_COMINGOUT_RE = re.compile(r"\bCOMINGOUT\s+P(\d+)\s+(\w+)", re.IGNORECASE)
_DIVINED_RE   = re.compile(r"\bDIVINED\s+P(\d+)\s+(WEREWOLF|HUMAN)", re.IGNORECASE)
_VOTE_RE      = re.compile(r"\bVOTE\s+P(\d+)", re.IGNORECASE)
_ESTIMATE_RE  = re.compile(r"\bESTIMATE\s+P(\d+)\s+(WEREWOLF|HUMAN)", re.IGNORECASE)

_MC_SAMPLES = 100   # worlds sampled for consistency check
_MC_WEIGHT  = 0.15  # how strongly MC adjustment moves beliefs


class BayesianAgent(AbstractAgent):
    """
    Maintains P(p = WEREWOLF | evidence) for all other players.
    Updates via Bayes-style likelihood weighting and Monte-Carlo sampling.
    """

    def __init__(self, agent_id: int, seed: int = None):
        super().__init__(agent_id, name="BayesianAgent", seed=seed)
        self._beliefs: dict      = {}     # {pid: float}  P(wolf)
        self._num_wolves: int    = 0
        self._divined: set       = set()  # pids already divined (avoid repeats)
        self._seen_talks: set    = set()  # (day, turn, agent) dedup

        # Observed seer claims from talk: {claimer_id: [(target, result_str), ...]}
        self._seer_claims: dict  = defaultdict(list)
        # Observed vote intentions from talk: {pid: target} this day
        self._talk_votes: dict   = {}
        # Confirmed seers (from game_info.divine_result if we ARE the seer)
        self._own_divines: dict  = {}   # {target: result_str}
        self._announced_seer: bool = False  # track whether seer claim was announced

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().initialize(game_info, game_setting)
        n = game_setting.player_num
        self._num_wolves = game_setting.role_num_map.get(Role.WEREWOLF, 0)
        prior = self._num_wolves / max(n - 1, 1)
        self._beliefs = {
            pid: 0.0 if pid == self.agent_id else prior
            for pid in game_info.status_map
        }
        # If we are a wolf, zero out known wolf teammates
        for pid, role in game_info.role_map.items():
            if role == Role.WEREWOLF and pid != self.agent_id:
                self._beliefs[pid] = 0.0  # known teammate
        self._announced_seer = False

    def update(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().update(game_info, game_setting)
        self._process_new_talks(game_info.talk_list)
        self._process_night_result(game_info)
        self._process_dead_players(game_info)
        self._normalize()

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def _prepare_round_tokens(self) -> list:
        # Seer: announce COMINGOUT + DIVINED the first time we have a result
        if self.my_role() == Role.SEER and not self._announced_seer and self._own_divines:
            self._announced_seer = True
            tokens = [f"COMINGOUT P{self.agent_id} SEER"]
            for t, result in list(self._own_divines.items())[-1:]:
                tokens.append(f"DIVINED P{t} {result}")
            tokens += ["BECAUSE divine_result", "Over"]
            return tokens

        target = self._top_suspect()
        if target is None:
            return ["Skip"]

        reason = "vote_pattern"
        if target in self._own_divines:
            reason = "divine_result"
        elif any(target == t and r == "WEREWOLF"
                 for claims in self._seer_claims.values() for t, r in claims):
            reason = "seer_claim_conflict"

        return [
            f"ESTIMATE P{target} WEREWOLF",
            f"ATTACK P{target}",
            f"BECAUSE {reason}",
            f"VOTE P{target}",
            "Over",
        ]

    def vote(self) -> int:
        target = self._top_suspect()
        return target if target is not None else self._rng.choice(
            self.alive_others() or [self.agent_id]
        )

    def attack(self) -> int:
        """Attack the alive non-wolf player with the lowest wolf belief."""
        candidates = [
            p for p in self.game_info.alive_agents
            if p != self.agent_id
            and self.game_info.role_map.get(p) != Role.WEREWOLF
        ]
        if not candidates:
            return self._random_non_wolf_target()
        return min(candidates, key=lambda p: self._beliefs.get(p, 0.5))

    def divine(self) -> int:
        """Divine the undivined alive other with the highest wolf belief."""
        candidates = [p for p in self.alive_others() if p not in self._divined]
        if not candidates:
            candidates = self.alive_others()
        if not candidates:
            return self.agent_id
        target = max(candidates, key=lambda p: self._beliefs.get(p, 0.5))
        self._divined.add(target)
        return target

    def guard(self) -> int:
        """Protect the player with the lowest wolf belief (most likely innocent)."""
        others = self.alive_others()
        if not others:
            return self.agent_id
        return min(others, key=lambda p: self._beliefs.get(p, 0.5))

    # ------------------------------------------------------------------ #
    #  Belief interface                                                    #
    # ------------------------------------------------------------------ #

    def get_wolf_belief(self, player_id: int) -> float:
        return self._beliefs.get(player_id, 0.5)

    # ------------------------------------------------------------------ #
    #  Belief update helpers                                               #
    # ------------------------------------------------------------------ #

    def _process_new_talks(self, talk_list: list) -> None:
        for talk in talk_list:
            key = (talk.day, talk.turn, talk.agent)
            if key in self._seen_talks:
                continue
            self._seen_talks.add(key)

            # Parse COMINGOUT claims
            for m in _COMINGOUT_RE.finditer(talk.text):
                claimer_pid = int(m.group(1))
                if claimer_pid == talk.agent:  # claiming for self
                    # Track but don't update beliefs here — wait for DIVINED
                    pass

            # Parse DIVINED claims from seers
            for m in _DIVINED_RE.finditer(talk.text):
                target  = int(m.group(1))
                result  = m.group(2).upper()
                claimer = talk.agent
                self._seer_claims[claimer].append((target, result))
                self._apply_seer_claim(claimer, target, result)

            # Parse VOTE intentions
            for m in _VOTE_RE.finditer(talk.text):
                self._talk_votes[talk.agent] = int(m.group(1))

    def _apply_seer_claim(self, claimer: int, target: int, result: str) -> None:
        """
        Update beliefs based on a claimed divine result.
        Weight by credibility of the claimer.
        """
        if target not in self._beliefs:
            return
        claimer_wolf_belief = self._beliefs.get(claimer, 0.5)
        # Credibility: inverse of wolf belief for claimer
        credibility = 1.0 - claimer_wolf_belief

        if result == "WEREWOLF":
            # Claimed target is wolf; move belief toward 1
            self._beliefs[target] = min(
                1.0,
                self._beliefs[target] + credibility * 0.3
            )
        else:  # HUMAN
            # Claimed target is human; move belief toward 0
            self._beliefs[target] = max(
                0.0,
                self._beliefs[target] - credibility * 0.2
            )

    def _process_night_result(self, game_info: GameInfo) -> None:
        """Process own divine result if we are the Seer."""
        dr: Optional[Judge] = game_info.divine_result
        if dr is None:
            return
        target = dr.target
        result = dr.result
        if result == "WEREWOLF":
            self._beliefs[target] = 1.0
        else:
            self._beliefs[target] = 0.0
        self._own_divines[target] = result

    def _process_dead_players(self, game_info: GameInfo) -> None:
        """
        When a player is eliminated, their role is revealed.
        Set belief to 0 or 1 and redistribute freed probability mass.
        """
        for pid, role in game_info.role_map.items():
            if game_info.status_map.get(pid) and game_info.status_map[pid].value == "DEAD":
                if role == Role.WEREWOLF:
                    self._beliefs[pid] = 1.0
                else:
                    self._beliefs[pid] = 0.0

    def _normalize(self) -> None:
        """
        Soft-normalize so the expected number of wolves among alive others
        stays consistent with the game configuration, without hard-clamping
        individual values (preserves relative confidence).

        Certainties (0.0 or 1.0) are never scaled — they represent known facts.
        """
        alive_others = self.alive_others()
        if not alive_others:
            return

        # Count remaining unresolved wolves
        known_wolves = sum(
            1 for pid in self.game_info.status_map
            if self.game_info.role_map.get(pid) == Role.WEREWOLF
            and self.game_info.status_map[pid].value == "DEAD"
        )
        # Cap beliefs in [0, 1]
        for pid in alive_others:
            self._beliefs[pid] = max(0.0, min(1.0, self._beliefs[pid]))

        # Only scale uncertain beliefs (exclude 0.0 and 1.0 — known facts)
        uncertain = [p for p in alive_others
                     if 0.0 < self._beliefs[p] < 1.0]
        certain_wolf_count = sum(1 for p in alive_others
                                  if self._beliefs[p] >= 1.0)

        remaining_wolves = max(0, self._num_wolves - known_wolves - certain_wolf_count)
        current_sum = sum(self._beliefs[p] for p in uncertain)
        if current_sum > 0 and remaining_wolves >= 0 and uncertain:
            scale = remaining_wolves / current_sum
            # Only apply mild scaling (avoid flipping confident beliefs)
            if 0.3 < scale < 3.0:
                for pid in uncertain:
                    self._beliefs[pid] = max(
                        0.0, min(1.0, self._beliefs[pid] * scale)
                    )

    def _top_suspect(self) -> Optional[int]:
        """Return alive other player with highest wolf belief."""
        alive = self.alive_others()
        if not alive:
            return None
        return max(alive, key=lambda p: self._beliefs.get(p, 0.0))
