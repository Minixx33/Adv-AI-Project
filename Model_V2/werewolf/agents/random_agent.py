"""
werewolf/agents/random_agent.py

RandomAgent — floor baseline.

Hierarchy:
  AbstractAgent
    └── RandomAgent

Behaviour:
  • talk()   → always "Skip"
  • vote()   → uniform random choice among alive other players
  • attack() → uniform random choice among alive non-wolf players
               (wolves are visible in role_map for this agent if it is a wolf)
  • divine() → uniform random choice among alive other players
  • guard()  → uniform random choice among alive other players
  • whisper()→ always "Skip"

Per spec: this agent cannot be enhanced with the suspicion module.
get_wolf_belief() is not overridden; falls back to the prior in AbstractAgent.
"""

from ..agent import AbstractAgent
from ..gameinfo import GameInfo, GameSetting
from ..const import Role


class RandomAgent(AbstractAgent):
    """
    Uniform-random decisions in every phase.
    No belief tracking.  Cannot interact with the suspicion module.
    """

    def __init__(self, agent_id: int, seed: int = None):
        super().__init__(agent_id, name="RandomAgent", seed=seed)

    def _prepare_round_tokens(self) -> list:
        if self._rng.random() < 0.40:
            return ["Skip"]
        others = self.alive_others()
        if not others:
            return ["Skip"]
        target = self._rng.choice(others)
        count = self._rng.randint(1, 2)
        return [f"VOTE P{target}"] * count + ["Over"]

    def vote(self) -> int:
        targets = self.alive_others()
        return self._rng.choice(targets) if targets else self.agent_id

    def attack(self) -> int:
        return self._random_non_wolf_target()

    def divine(self) -> int:
        return self._random_other()

    def guard(self) -> int:
        return self._random_other()
