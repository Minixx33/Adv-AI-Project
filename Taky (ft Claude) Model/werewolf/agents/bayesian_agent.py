"""werewolf.agents.bayesian_agent — Baseline 3: single-layer Bayesian belief tracker.

This agent maintains a probability distribution ``belief[p] = P(p is Werewolf)``
over all other players and updates it after every observed talk.  All decisions
(vote, divine, attack) derive directly from this belief distribution.

Design notes
------------
* The prior is uniform: ``1/4`` for each of the four other players in a
  5-player game (one wolf among four).
* Beliefs are updated incrementally as talks arrive; each ``update()`` call
  only processes talks since the last ``_talk_head`` index.
* Beliefs are clamped to ``[0.0, 1.0]`` after each update rather than
  globally normalised, so individual values remain interpretable as rough
  probabilities rather than forcing them to sum to 1.
* Dead players have their belief zeroed so they never appear in ``argmax``.

Update rules
------------
+-------------------------------------+---------+------------------------------+
| Observation                         | Delta   | Rationale                    |
+=====================================+=========+==============================+
| Another player says ``VOTE p``      | +0.05   | Accusation raises suspicion. |
+-------------------------------------+---------+------------------------------+
| ``ESTIMATE p WEREWOLF``             | +0.10   | Stronger accusation signal.  |
+-------------------------------------+---------+------------------------------+
| ``ESTIMATE p VILLAGER/SEER``        | -0.04   | Explicit defence lowers it.  |
+-------------------------------------+---------+------------------------------+
| ``DIVINED p WEREWOLF``              | → 0.90  | Strong positive evidence.    |
+-------------------------------------+---------+------------------------------+
| ``DIVINED p HUMAN``                 | → 0.08  | Strong negative evidence.    |
+-------------------------------------+---------+------------------------------+
| Own divine result (Seer role only)  | → 0.90  | Certain positive evidence.   |
|                                     | / 0.08  | / certain negative evidence. |
+-------------------------------------+---------+------------------------------+

Module-level constants
----------------------
_VOTE_DELTA          : float = 0.05   — belief nudge for a VOTE talk.
_ESTIMATE_WOLF_DELTA : float = 0.10   — belief nudge for ESTIMATE WEREWOLF.
_DEFEND_DELTA        : float = 0.04   — belief reduction for a defence talk.
_DIVINE_WOLF         : float = 0.90   — belief set on DIVINED WEREWOLF.
_DIVINE_HUMAN        : float = 0.08   — belief set on DIVINED HUMAN.

Classes
-------
BayesianAgent : AbstractAgent with incremental Bayesian belief updates.
"""
from __future__ import annotations

from typing import Optional

from werewolf.agent import AbstractAgent
from werewolf.gameinfo import Agent, GameInfo, GameSetting, Talk, parse_talk, AGENT_NONE
from werewolf.const import Role, Species, Status


_VOTE_DELTA:          float = 0.05
_ESTIMATE_WOLF_DELTA: float = 0.10
_DEFEND_DELTA:        float = 0.04
_DIVINE_WOLF:         float = 0.90
_DIVINE_HUMAN:        float = 0.08


class BayesianAgent(AbstractAgent):
    """Baseline 3 — single-layer Bayesian wolf-belief tracker.

    Strategy summary
    ----------------
    * **talk()** : announces ``"VOTE <argmax belief>"`` to signal its
      current wolf suspect to the group.
    * **vote()** : votes for ``argmax(belief)`` — the player with the
      highest probability of being the Werewolf.
    * **divine()**: investigates ``argmax(belief)`` — confirms or refutes
      the top suspect.
    * **attack()**: kills ``argmin(belief)`` among non-wolf players — the
      most-trusted villager is the greatest threat to the wolf team.

    Attributes
    ----------
    _game_info          : GameInfo            — Latest snapshot from ``update()``.
    _my_role            : Role                — This agent's role.
    _belief             : dict[Agent, float]  — ``P(p is Werewolf)`` per player.
    _talk_head          : int                 — Index of next unprocessed talk.
    _agent_lookup       : dict[str, Agent]    — String → Agent for parsing.
    _votes_cast         : int                 — Cumulative vote count.
    _correct_wolf_votes : int                 — Votes that landed on the wolf.
    _false_accusations  : int                 — Votes on non-wolf players.
    """

    def __init__(self) -> None:
        self._game_info: GameInfo = None          # type: ignore[assignment]
        self._my_role: Role = None                # type: ignore[assignment]
        self._belief: dict[Agent, float]          = {}
        self._talk_head: int                      = 0
        self._agent_lookup: dict[str, Agent]      = {}
        self._votes_cast: int                     = 0
        self._correct_wolf_votes: int             = 0
        self._false_accusations: int              = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        """Initialise belief distribution with a uniform prior.

        Sets ``belief[p] = 1 / (n_players - 1)`` for all other players,
        representing one unknown wolf distributed uniformly across opponents.

        Args:
            game_info:    Day-0 snapshot; used to enumerate all players.
            game_setting: Static game configuration (unused by this agent).
        """
        self._game_info    = game_info
        self._my_role      = game_info.my_role
        self._agent_lookup = {str(a): a for a in game_info.status_map}
        others = [a for a in game_info.status_map if a != game_info.agent]
        prior  = 1.0 / max(len(others), 1)
        self._belief     = {a: prior for a in others}
        self._talk_head  = 0
        self._votes_cast         = 0
        self._correct_wolf_votes = 0
        self._false_accusations  = 0

    def update(self, game_info: GameInfo) -> None:
        """Store the latest snapshot and process new talks.

        Args:
            game_info: Current game state snapshot.
        """
        self._game_info = game_info
        self._process_new_talks()

    def day_start(self) -> None:
        """Apply the Seer's divine result and zero dead players' beliefs.

        Called at the start of each day phase.  If this agent is the Seer,
        ``game_info.divine_result`` contains last night's investigation
        outcome and is used to hard-set the target's belief.
        """
        self._talk_head = 0
        gi = self._game_info

        # Apply personal divine result (only non-None when we are the Seer)
        if gi.divine_result:
            dr = gi.divine_result
            if dr.target in self._belief:
                if dr.result == Species.WEREWOLF:
                    self._belief[dr.target] = _DIVINE_WOLF
                else:
                    self._belief[dr.target] = _DIVINE_HUMAN

        # Zero out dead players so they are excluded from argmax / argmin
        for agent, status in gi.status_map.items():
            if status == Status.DEAD and agent in self._belief:
                self._belief[agent] = 0.0

    # ------------------------------------------------------------------
    # Internal — belief updates
    # ------------------------------------------------------------------

    def _clamp(self, v: float) -> float:
        """Clamp *v* to the valid belief range [0.0, 1.0].

        Args:
            v: Raw updated belief value.

        Returns:
            ``v`` constrained to ``[0.0, 1.0]``.
        """
        return max(0.0, min(1.0, v))

    def _process_new_talks(self) -> None:
        """Update beliefs based on talks not yet seen.

        Iterates from ``_talk_head`` to the end of ``talk_list``, applying
        the update rules described in the module docstring.  Advances
        ``_talk_head`` to avoid double-processing.
        """
        gi = self._game_info
        for i in range(self._talk_head, len(gi.talk_list)):
            tk: Talk = gi.talk_list[i]
            parsed = parse_talk(tk.text, self._agent_lookup)
            topic  = parsed.get("topic")
            target: Optional[Agent] = parsed.get("target")

            if not target or target not in self._belief:
                continue

            if topic == "VOTE":
                self._belief[target] = self._clamp(
                    self._belief[target] + _VOTE_DELTA
                )
            elif topic == "ESTIMATE":
                role_str = parsed.get("role", "")
                if role_str == "WEREWOLF":
                    self._belief[target] = self._clamp(
                        self._belief[target] + _ESTIMATE_WOLF_DELTA
                    )
                elif role_str in ("VILLAGER", "SEER"):
                    self._belief[target] = self._clamp(
                        self._belief[target] - _DEFEND_DELTA
                    )
            elif topic == "DIVINED":
                result_str = parsed.get("result", "")
                if result_str == "WEREWOLF":
                    self._belief[target] = _DIVINE_WOLF
                elif result_str == "HUMAN":
                    self._belief[target] = _DIVINE_HUMAN

        self._talk_head = len(gi.talk_list)

    def _argmax_belief(self) -> Agent:
        """Return the alive opponent with the highest wolf belief.

        Returns:
            The alive non-self agent maximising ``belief``, or self if no
            alive opponents remain.
        """
        gi         = self._game_info
        candidates = self._alive_others(gi)
        if not candidates:
            return gi.agent
        return max(candidates, key=lambda a: self._belief.get(a, 0.0))

    def _argmin_belief(self) -> Agent:
        """Return the alive non-wolf opponent with the lowest wolf belief.

        Used by the Werewolf role: kills the player least suspected of being
        a wolf (i.e. most trusted by the village).

        Returns:
            The alive non-wolf-team agent minimising ``belief``.
        """
        gi        = self._game_info
        wolf_team = set(gi.role_map.keys())
        candidates = [a for a in gi.alive_agent_list if a not in wolf_team]
        if not candidates:
            candidates = self._alive_others(gi)
        if not candidates:
            return gi.agent
        return min(candidates, key=lambda a: self._belief.get(a, 1.0))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def talk(self) -> str:
        """Announce a vote intention for the current top suspect.

        Returns:
            ``"VOTE Agent[XX]"`` for ``argmax(belief)`` among alive opponents.
        """
        target = self._argmax_belief()
        return f"VOTE {target}"

    def vote(self) -> Agent:
        """Vote for the alive player most believed to be the Werewolf.

        Returns:
            ``argmax(belief)`` among alive opponents.
        """
        target = self._argmax_belief()
        self._votes_cast += 1
        return target

    def whisper(self) -> str:
        """No-op wolf whisper.

        Returns:
            The string ``"SKIP"``.
        """
        return "SKIP"

    def attack(self) -> Agent:
        """Kill the alive non-wolf player with the lowest wolf belief.

        Returns:
            ``argmin(belief)`` among alive non-wolf players.
        """
        return self._argmin_belief()

    def divine(self) -> Agent:
        """Investigate the current top wolf suspect.

        Returns:
            ``argmax(belief)`` among alive opponents.
        """
        return self._argmax_belief()

    def finish(self) -> None:
        """No-op — this agent needs no end-of-game cleanup."""

    # ------------------------------------------------------------------
    # Stats hook
    # ------------------------------------------------------------------

    def record_vote_outcome(
        self, target: Agent, true_roles: dict[Agent, Role]
    ) -> None:
        """Update vote-accuracy counters after each elimination vote.

        Args:
            target:     The agent who was executed.
            true_roles: Ground-truth role assignment for all players.
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

        The ``final_belief_wolf`` value is the maximum belief across alive
        opponents at game end — a proxy for whether the agent had correctly
        identified the wolf before the game concluded.

        Returns:
            Dict with keys: ``votes_cast``, ``correct_wolf_votes``,
            ``false_accusations``, ``final_belief_wolf`` (float or None),
            ``final_sigma_wolf`` (None).
        """
        gi         = self._game_info
        candidates = self._alive_others(gi) if gi else []
        final_belief = max(
            (self._belief.get(a, 0.0) for a in candidates), default=None
        )
        return {
            "votes_cast":           self._votes_cast,
            "correct_wolf_votes":   self._correct_wolf_votes,
            "false_accusations":    self._false_accusations,
            "final_belief_wolf":    round(final_belief, 4) if final_belief is not None else None,
            "final_sigma_wolf":     None,
        }
