"""werewolf.agents.heuristic_agent — Baseline 2: accusation-count heuristic.

This agent tracks how many times each player has been publicly accused
(via VOTE or ESTIMATE WEREWOLF talks) and always votes for the
most-accused player.  When playing as the Werewolf it targets the
*least*-accused player — the one the group trusts most.

Classes
-------
HeuristicAgent : AbstractAgent subclass using simple accusation counting.
"""
from __future__ import annotations

from collections import defaultdict

from werewolf.agent import AbstractAgent
from werewolf.gameinfo import Agent, GameInfo, GameSetting, Talk, parse_talk, AGENT_NONE
from werewolf.const import Role, Status


class HeuristicAgent(AbstractAgent):
    """Baseline 2 — accusation-count heuristic with no probabilistic belief.

    Strategy summary
    ----------------
    * **talk()** : broadcasts ``"VOTE <most_accused_alive>"`` each turn,
      signalling its intention to the group.
    * **vote()** : votes for the alive player with the highest accusation
      count (``argmax`` over alive opponents).
    * **divine()**: targets the most-accused alive player (same as vote).
    * **attack()**: targets the *least*-accused alive non-wolf player —
      the community's most-trusted player is the wolf's best strategic kill.

    Accusation sources
    ------------------
    A player's accusation count increases when another player says:

    * ``"VOTE <player>"``               — direct vote declaration
    * ``"ESTIMATE <player> WEREWOLF"``  — explicit wolf accusation

    Talk analysis uses a ``_talk_head`` pointer so each talk is processed
    exactly once across multiple ``update()`` calls.

    Attributes
    ----------
    _game_info         : GameInfo               — Latest snapshot.
    _my_role           : Role                   — This agent's role.
    _accusation_count  : dict[Agent, int]       — Accumulated accusation tally.
    _talk_head         : int                    — Index of next unprocessed talk.
    _agent_lookup      : dict[str, Agent]       — String-to-Agent map for parsing.
    _votes_cast        : int                    — Running vote count.
    _correct_wolf_votes: int                    — Votes that hit the wolf.
    _false_accusations : int                    — Votes that missed.
    """

    def __init__(self) -> None:
        self._game_info: GameInfo = None          # type: ignore[assignment]
        self._my_role: Role = None                # type: ignore[assignment]
        self._accusation_count: dict[Agent, int]  = defaultdict(int)
        self._talk_head: int                      = 0
        self._agent_lookup: dict[str, Agent]      = {}
        self._votes_cast: int                     = 0
        self._correct_wolf_votes: int             = 0
        self._false_accusations: int              = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        """Reset all per-game state and build the agent string-lookup table.

        Args:
            game_info:    Day-0 snapshot; used to enumerate all players.
            game_setting: Static game configuration (unused by this agent).
        """
        self._game_info  = game_info
        self._my_role    = game_info.my_role
        self._accusation_count.clear()
        self._talk_head  = 0
        self._agent_lookup = {str(a): a for a in game_info.status_map}
        self._votes_cast         = 0
        self._correct_wolf_votes = 0
        self._false_accusations  = 0

    def update(self, game_info: GameInfo) -> None:
        """Store the latest snapshot and parse any new talks.

        Args:
            game_info: Current game state snapshot.
        """
        self._game_info = game_info
        self._process_new_talks()

    def day_start(self) -> None:
        """Reset the talk-head to 0 at the start of each day's discussion."""
        self._talk_head = 0

    # ------------------------------------------------------------------
    # Internal — talk analysis
    # ------------------------------------------------------------------

    def _process_new_talks(self) -> None:
        """Parse all talks from ``_talk_head`` onward and update accusation counts.

        Increments ``_accusation_count[target]`` for each VOTE or
        ESTIMATE WEREWOLF talk made by another player.  Self-talks are ignored
        to prevent gaming the heuristic.
        """
        gi = self._game_info
        for i in range(self._talk_head, len(gi.talk_list)):
            tk: Talk = gi.talk_list[i]
            if tk.agent == gi.agent:
                continue
            parsed = parse_talk(tk.text, self._agent_lookup)
            topic  = parsed.get("topic")
            target = parsed.get("target")
            if topic == "VOTE" and target:
                self._accusation_count[target] += 1
            elif topic == "ESTIMATE" and target and parsed.get("role") == "WEREWOLF":
                self._accusation_count[target] += 1
        self._talk_head = len(gi.talk_list)

    def _most_accused_alive(self) -> Agent:
        """Return the alive opponent with the highest accusation count.

        Ties are broken by the natural ordering of ``max()`` (first
        encountered with the maximum value).

        Returns:
            The most-accused alive opponent, or ``game_info.agent`` (self)
            if no alive opponents exist.
        """
        gi           = self._game_info
        alive_others = self._alive_others(gi)
        if not alive_others:
            return gi.agent
        return max(alive_others, key=lambda a: self._accusation_count[a])

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def talk(self) -> str:
        """Declare a vote intention for the most-accused alive player.

        Returns:
            ``"VOTE Agent[XX]"`` for the current most-accused player, or
            ``"SKIP"`` if no valid target exists.
        """
        target = self._most_accused_alive()
        if target != AGENT_NONE:
            return f"VOTE {target}"
        return "SKIP"

    def vote(self) -> Agent:
        """Vote for the most-accused alive opponent.

        Returns:
            The alive agent with the highest accusation count.
        """
        target = self._most_accused_alive()
        self._votes_cast += 1
        return target

    def whisper(self) -> str:
        """No-op wolf whisper.

        Returns:
            The string ``"SKIP"``.
        """
        return "SKIP"

    def attack(self) -> Agent:
        """Kill the alive non-wolf player with the *lowest* accusation count.

        The least-accused player is the one the village trusts most, making
        them the most valuable target for the wolf team to eliminate.

        Returns:
            The alive non-wolf agent with the minimum accusation count,
            falling back to any alive opponent if no non-wolf candidates exist.
        """
        gi        = self._game_info
        wolf_team = set(gi.role_map.keys())
        candidates = [a for a in gi.alive_agent_list if a not in wolf_team]
        if not candidates:
            candidates = self._alive_others(gi)
        if not candidates:
            return gi.agent
        return min(candidates, key=lambda a: self._accusation_count[a])

    def divine(self) -> Agent:
        """Investigate the most-accused alive player.

        Returns:
            Same target as ``vote()`` — the most publicly suspected player.
        """
        return self._most_accused_alive()

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
