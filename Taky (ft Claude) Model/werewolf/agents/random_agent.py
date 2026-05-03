"""werewolf.agents.random_agent — Baseline 1: purely random decisions.

This agent serves as the lower-bound baseline.  It makes no use of any
game information: votes are uniformly random, talk is always SKIP, and
night actions pick a random valid target.

Classes
-------
RandomAgent : AbstractAgent subclass with zero strategic reasoning.
"""
from __future__ import annotations

import random

from werewolf.agent import AbstractAgent
from werewolf.gameinfo import Agent, GameInfo, GameSetting, AGENT_NONE
from werewolf.const import Role


class RandomAgent(AbstractAgent):
    """Baseline 1 — purely random decisions, never produces meaningful talk.

    Strategy summary
    ----------------
    * **talk()** : always returns ``"SKIP"``; contributes nothing to the
      communal information pool.
    * **vote()** : uniform-random choice among alive opponents.
    * **divine()**: uniform-random choice among alive opponents (Seer role).
    * **attack()**: uniform-random choice among alive non-wolf players
      (Werewolf role).

    This agent provides a performance floor for comparison: any agent that
    cannot beat RandomAgent in win-rate or wolf-identification accuracy has
    failed to leverage the available information.

    Attributes
    ----------
    _game_info         : GameInfo  — Latest snapshot from ``update()``.
    _my_role           : Role      — This agent's role for the current game.
    _votes_cast        : int       — Running count of votes cast.
    _correct_wolf_votes: int       — Votes that happened to land on the wolf.
    _false_accusations : int       — Votes that landed on a non-wolf player.
    """

    def __init__(self) -> None:
        self._game_info: GameInfo = None          # type: ignore[assignment]
        self._my_role: Role = None                # type: ignore[assignment]
        self._votes_cast: int = 0
        self._correct_wolf_votes: int = 0
        self._false_accusations: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        """Reset all per-game state.

        Args:
            game_info:    Day-0 snapshot containing this agent's assigned role.
            game_setting: Static game configuration (unused by this agent).
        """
        self._game_info = game_info
        self._my_role   = game_info.my_role
        self._votes_cast         = 0
        self._correct_wolf_votes = 0
        self._false_accusations  = 0

    def update(self, game_info: GameInfo) -> None:
        """Store the latest snapshot (no analysis performed).

        Args:
            game_info: Current game state snapshot.
        """
        self._game_info = game_info

    def day_start(self) -> None:
        """No-op — this agent does not track day boundaries."""

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def talk(self) -> str:
        """Always skip; this agent contributes no information.

        Returns:
            The string ``"SKIP"``.
        """
        return "SKIP"

    def vote(self) -> Agent:
        """Vote for a uniformly random alive opponent.

        Returns:
            A randomly chosen agent from the alive-others list, or
            ``game_info.agent`` (self) if no other alive agents exist.
        """
        candidates = self._alive_others(self._game_info)
        if not candidates:
            return self._game_info.agent
        target = random.choice(candidates)
        self._votes_cast += 1
        return target

    def whisper(self) -> str:
        """No-op wolf whisper.

        Returns:
            The string ``"SKIP"``.
        """
        return "SKIP"

    def attack(self) -> Agent:
        """Kill a uniformly random non-wolf alive player.

        The wolf team is identified via ``game_info.role_map``, which exposes
        all wolf-team members to wolf-role agents.

        Returns:
            A randomly chosen non-wolf alive agent, falling back to any
            alive opponent if no non-wolf candidates remain.
        """
        gi = self._game_info
        wolf_team  = set(gi.role_map.keys())
        candidates = [a for a in gi.alive_agent_list if a not in wolf_team]
        if not candidates:
            candidates = self._alive_others(gi)
        return random.choice(candidates) if candidates else gi.agent

    def divine(self) -> Agent:
        """Investigate a uniformly random alive opponent.

        Returns:
            A randomly chosen agent from the alive-others list.
        """
        candidates = self._alive_others(self._game_info)
        return random.choice(candidates) if candidates else self._game_info.agent

    def finish(self) -> None:
        """No-op — this agent needs no end-of-game cleanup."""

    # ------------------------------------------------------------------
    # Stats hook
    # ------------------------------------------------------------------

    def record_vote_outcome(
        self, target: Agent, true_roles: dict[Agent, Role]
    ) -> None:
        """Update vote-accuracy counters after each elimination vote.

        Called by ``WerewolfGame`` once per game day with the executed agent
        and the true role map so each agent can self-score.

        Args:
            target:     The agent who was executed.
            true_roles: The ground-truth role assignment for all players.
        """
        if target == AGENT_NONE:
            return
        role = true_roles.get(target)
        if role == Role.WEREWOLF:
            self._correct_wolf_votes += 1
        elif role is not None:
            self._false_accusations += 1

    def get_stats(self) -> dict:
        """Return performance metrics for CSV logging.

        Returns:
            Dict with keys: ``votes_cast``, ``correct_wolf_votes``,
            ``false_accusations``, ``final_belief_wolf`` (None),
            ``final_sigma_wolf`` (None).
        """
        return {
            "votes_cast":           self._votes_cast,
            "correct_wolf_votes":   self._correct_wolf_votes,
            "false_accusations":    self._false_accusations,
            "final_belief_wolf":    None,
            "final_sigma_wolf":     None,
        }
