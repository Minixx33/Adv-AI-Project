"""
werewolf/agents/heuristic_agent.py

HeuristicAgent — simple accusation-count baseline.

Hierarchy:
  AbstractAgent
    └── HeuristicAgent

Behaviour:
  • Tracks how many times each player has been accused (received a VOTE or
    ESTIMATE … WEREWOLF in the talk phase).
  • vote()  → the most-accused alive player (random tiebreak).
  • talk()  → "VOTE P{target}" declaring the current most-accused target.
  • attack()→ targets the least-accused non-wolf alive player (wolves avoid
               suspicion by attacking players that draw little accusation).
  • divine()→ the most-accused alive player not yet divined.
  • guard() → the least-accused alive player (protect the innocent).
  • get_wolf_belief(pid) → normalised accusation score ∈ [0, 1]; used by
               the SuspicionModule as the base_belief component.

Compatible with the suspicion module (§ spec, §2).
"""

import re
from collections import defaultdict
from typing import Optional

from ..agent import AbstractAgent
from ..gameinfo import GameInfo, GameSetting
from ..const import Role


# Talk patterns that count as an accusation
_VOTE_RE     = re.compile(r"\bVOTE\s+P(\d+)", re.IGNORECASE)
_ESTIMATE_RE = re.compile(r"\bESTIMATE\s+P(\d+)\s+WEREWOLF", re.IGNORECASE)


class HeuristicAgent(AbstractAgent):
    """
    Votes for the player most frequently accused in public talk.
    Compatible with the suspicion module.
    """

    def __init__(self, agent_id: int, seed: int = None):
        super().__init__(agent_id, name="HeuristicAgent", seed=seed)
        self._accusation_count: dict = defaultdict(int)
        self._seen_talks: set = set()   # (day, turn, agent) already processed
        self._divined: set   = set()    # player IDs already divined (avoid repeats)

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().initialize(game_info, game_setting)
        self._accusation_count = defaultdict(int)
        self._seen_talks = set()
        self._divined    = set()
        self._process_talks(game_info.talk_list)

    def update(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        super().update(game_info, game_setting)
        self._process_talks(game_info.talk_list)

    def day_start(self) -> None:
        super().day_start()  # clears talk queue
        # accusation counts persist across days

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def _prepare_round_tokens(self) -> list:
        target = self._most_accused_alive()
        if target is None:
            return ["Skip"]
        tokens = [f"VOTE P{target}"]
        agreeable = self._find_accuser_of(target)
        if agreeable:
            tokens.append(f"AGREE P{agreeable}")
        tokens.append(f"ATTACK P{target}")
        tokens.append("Over")
        return tokens

    def _find_accuser_of(self, target: int) -> Optional[int]:
        """Return an alive player who recently ATTACK/VOTE'd the same target."""
        if not self.game_info:
            return None
        alive = set(self.alive_others())
        for talk in reversed(self.game_info.talk_list):
            if (talk.agent != self.agent_id
                    and talk.agent in alive
                    and (f"ATTACK P{target}" in talk.text
                         or f"VOTE P{target}" in talk.text)):
                return talk.agent
        return None

    def vote(self) -> int:
        target = self._most_accused_alive()
        return target if target is not None else self._rng.choice(
            self.alive_others() or [self.agent_id]
        )

    def attack(self) -> int:
        """Attack the least-accused non-wolf alive player."""
        candidates = [
            p for p in self.game_info.alive_agents
            if p != self.agent_id
            and self.game_info.role_map.get(p) != Role.WEREWOLF
        ]
        if not candidates:
            return self._random_non_wolf_target()
        return min(candidates, key=lambda p: self._accusation_count[p])

    def divine(self) -> int:
        """Divine the most-accused alive player not yet divined."""
        candidates = [
            p for p in self.alive_others()
            if p not in self._divined
        ]
        if not candidates:
            return self._random_other()
        target = max(candidates, key=lambda p: self._accusation_count[p])
        self._divined.add(target)
        return target

    def guard(self) -> int:
        """Protect the least-accused alive non-self player."""
        others = self.alive_others()
        if not others:
            return self.agent_id
        return min(others, key=lambda p: self._accusation_count[p])

    # ------------------------------------------------------------------ #
    #  Belief interface (used by SuspicionModule)                          #
    # ------------------------------------------------------------------ #

    def get_wolf_belief(self, player_id: int) -> float:
        """Normalised accusation count as proxy for wolf probability."""
        alive = self.alive_others()
        if not alive:
            return 0.5
        total = sum(self._accusation_count[p] for p in alive)
        if total == 0:
            return 1.0 / len(alive)
        return self._accusation_count[player_id] / total

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _process_talks(self, talk_list: list) -> None:
        for talk in talk_list:
            key = (talk.day, talk.turn, talk.agent)
            if key in self._seen_talks:
                continue
            self._seen_talks.add(key)
            if talk.agent == self.agent_id:
                continue  # don't count own talk
            # Parse accusation patterns
            for m in _VOTE_RE.finditer(talk.text):
                accused = int(m.group(1))
                self._accusation_count[accused] += 1
            for m in _ESTIMATE_RE.finditer(talk.text):
                accused = int(m.group(1))
                self._accusation_count[accused] += 1

    def _most_accused_alive(self):
        alive = self.alive_others()
        if not alive:
            return None
        best = max(alive, key=lambda p: self._accusation_count[p])
        # If nobody accused: return random to avoid all returning same player
        if self._accusation_count[best] == 0:
            return self._rng.choice(alive)
        return best
