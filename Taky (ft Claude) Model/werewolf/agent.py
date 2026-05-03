"""werewolf.agent — Abstract base class that all agent implementations must subclass.

The interface defined here mirrors the AIWolf platform's ``AbstractPlayer``
API exactly, so an agent written for this engine could be ported to the
networked AIWolf server with minimal changes.

Call order enforced by ``WerewolfGame``
---------------------------------------
Day 0 (initialisation)::

    update(game_info)  →  initialize(game_info, game_setting)

Each day phase::

    update(game_info)  →  day_start()
    [for each talk slot]  update(game_info)  →  talk()
    [voting]              update(game_info)  →  vote()

Night phase (role-specific)::

    Seer:     update(game_info)  →  divine()
    Werewolf: update(game_info)  →  whisper()  →  update(game_info)  →  attack()

Game over::

    update(game_info)  →  finish()

Classes
-------
AbstractAgent : Base class; subclass and override the methods you need.
"""
from __future__ import annotations

from werewolf.gameinfo import Agent, GameInfo, GameSetting, AGENT_NONE
from werewolf.const import Status


class AbstractAgent:
    """Base class for all Werewolf agent implementations.

    Subclasses override whichever lifecycle methods are relevant to their
    role.  The default implementations are safe no-ops or sensible fallbacks,
    so a minimal agent only needs to override ``vote()`` and one or two
    role-specific methods.

    Attributes
    ----------
    name : str  (read-only property)
        The class name, used as the agent-type label in CSV output.
    """

    # ------------------------------------------------------------------
    # Lifecycle hooks — called by WerewolfGame in a fixed order
    # ------------------------------------------------------------------

    def initialize(self, game_info: GameInfo, game_setting: GameSetting) -> None:
        """Set up all data structures for a new game.

        Called exactly once per game, after the first ``update()``, on day 0.
        Agents should reset every piece of per-game state here so that
        re-using the same instance across games works correctly.

        Args:
            game_info:    Day-0 snapshot.  ``game_info.my_role`` reveals
                          this agent's assigned role.
            game_setting: Static configuration (player count, role counts,
                          talk-round limit).
        """

    def update(self, game_info: GameInfo) -> None:
        """Receive the latest game-state snapshot before any action.

        Called by the engine immediately before every other method.  Agents
        should store the snapshot and process any newly added talks here so
        that ``talk()``, ``vote()``, etc. work with current information.

        Args:
            game_info: The most recent filtered snapshot for this agent.
        """

    def day_start(self) -> None:
        """React to the start of a new day phase.

        Called once per day, after ``update()``, before the first ``talk()``
        of that day.  The Seer can read ``game_info.divine_result`` here to
        process the previous night's investigation result.
        """

    def talk(self) -> str:
        """Produce one utterance during the discussion phase.

        Called once per talk slot (5 rounds × alive agents per day).  The
        returned string is appended to ``game_info.talk_list`` and becomes
        visible to all agents on their next ``update()``.

        Returns:
            A talk string.  Recognised formats:
              * ``"VOTE Agent[XX]"``           — declare vote intention
              * ``"ESTIMATE Agent[XX] ROLE"``  — express role belief
              * ``"COMINGOUT Agent[XX] ROLE"`` — claim a role
              * ``"DIVINED Agent[XX] SPECIES"``— share a Seer result
              * ``"SKIP"``                     — pass this turn
              * ``"OVER"``                     — signal done for the day

            The default implementation always returns ``"SKIP"``.
        """
        return "SKIP"

    def vote(self) -> Agent:
        """Choose a player to eliminate during the day-phase vote.

        Called once per day after the discussion phase.  The engine resolves
        all votes by plurality; ties are broken randomly.

        Returns:
            The ``Agent`` this player votes to eliminate.  Returning
            ``AGENT_NONE`` causes the engine to substitute a random alive
            opponent as a fallback.
        """
        return AGENT_NONE

    def whisper(self) -> str:
        """Produce a private wolf-team whisper during the night phase.

        Only called for Werewolf-role agents (and wolf-team agents if the
        engine is extended).  Whispers are visible only to the wolf team via
        ``game_info.whisper_list``.

        Returns:
            A talk-formatted string, typically announcing an attack target
            (``"VOTE Agent[XX]"``).  Default returns ``"SKIP"``.
        """
        return "SKIP"

    def attack(self) -> Agent:
        """Choose which player the wolf kills tonight.

        Only called for the living Werewolf-role agent each night phase.  The
        engine guarantees the target is a living non-wolf-team player; if the
        returned agent is invalid or on the wolf team the engine picks randomly.

        Returns:
            The ``Agent`` to attack.  Returning ``AGENT_NONE`` triggers the
            random fallback.
        """
        return AGENT_NONE

    def divine(self) -> Agent:
        """Choose which player the Seer investigates tonight.

        Only called for the living Seer-role agent each night phase.  The
        engine records a ``Judge`` result containing the target's ``Species``
        (HUMAN or WEREWOLF) which becomes available via
        ``game_info.divine_result`` at the start of the next day.

        Returns:
            The ``Agent`` to investigate.  Returning ``AGENT_NONE`` triggers
            the random fallback.
        """
        return AGENT_NONE

    def finish(self) -> None:
        """React to the game ending.

        Called once, after the final ``update()``, when a winner has been
        determined.  Agents may log final state or compute post-game metrics
        here.  The default is a no-op.
        """

    # ------------------------------------------------------------------
    # Stats API — called by GameRunner after each game for CSV logging
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return a flat dict of performance metrics for this game.

        The runner writes these values directly to the CSV.  All keys listed
        below should be present; unknown keys are silently ignored.

        Returns:
            A dict with the following keys:

            +----------------------+-------+------------------------------------------+
            | Key                  | Type  | Description                              |
            +======================+=======+==========================================+
            | votes_cast           | int   | Total votes cast across all rounds.      |
            +----------------------+-------+------------------------------------------+
            | correct_wolf_votes   | int   | Votes that landed on the Werewolf.       |
            +----------------------+-------+------------------------------------------+
            | false_accusations    | int   | Votes that landed on a non-wolf player.  |
            +----------------------+-------+------------------------------------------+
            | final_belief_wolf    | float | Max wolf-belief at game end              |
            |                      | None  | (Bayesian/Suspicion agents only).        |
            +----------------------+-------+------------------------------------------+
            | final_sigma_wolf     | float | Max suspicion-sigma at game end          |
            |                      | None  | (Suspicion agent only).                  |
            +----------------------+-------+------------------------------------------+
        """
        return {}

    # ------------------------------------------------------------------
    # Shared helpers — available to all subclasses
    # ------------------------------------------------------------------

    def _is_alive(self, agent: Agent, game_info: GameInfo) -> bool:
        """Return True if *agent* is currently alive.

        Args:
            agent:     The agent to check.
            game_info: The current game snapshot.

        Returns:
            True if ``game_info.status_map[agent] == Status.ALIVE``.
        """
        return game_info.status_map.get(agent) == Status.ALIVE

    def _alive_others(self, game_info: GameInfo) -> list[Agent]:
        """Return the list of alive agents excluding this agent.

        Args:
            game_info: The current game snapshot.

        Returns:
            A list of ``Agent`` objects from ``alive_agent_list`` that are
            not equal to ``game_info.agent``.
        """
        return [a for a in game_info.alive_agent_list if a != game_info.agent]

    @property
    def name(self) -> str:
        """The class name used as the agent-type label in CSV output."""
        return self.__class__.__name__
