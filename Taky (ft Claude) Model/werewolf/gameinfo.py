"""werewolf.gameinfo — Core data structures that mirror the AIWolf platform API.

Every object in this module is intentionally kept as a pure data container
with no game logic, so agents can safely store and inspect snapshots without
side-effects on the engine.

Classes
-------
Agent       : Immutable, hashable identity token for a player slot.
Talk        : A single utterance recorded during the discussion phase.
Vote        : A single vote cast during the voting phase.
Judge       : The result of a Seer divine action.
GameInfo    : Per-agent snapshot of all visible game state.
GameSetting : Static configuration handed to agents at initialisation.

Constants
---------
AGENT_NONE : Sentinel agent (index 0) used as a null / "no target" value.

Functions
---------
parse_talk(text, agent_lookup) -> dict
    Tokenise a talk string into a structured intent dictionary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from werewolf.const import Role, Species, Status


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------

class Agent:
    """Immutable, hashable identity token for one player slot.

    Player slots are numbered 1–5; slot 0 is reserved for AGENT_NONE.
    Agents compare equal if and only if their indices are equal, so they
    are safe to use as dictionary keys or set members.

    Attributes
    ----------
    agent_idx : int
        1-based integer index (0 for the AGENT_NONE sentinel).

    Examples
    --------
    >>> a1, a2 = Agent(1), Agent(2)
    >>> str(a1)
    'Agent[01]'
    >>> a1 == Agent(1)
    True
    >>> sorted([a2, a1])
    [Agent[01], Agent[02]]
    """

    __slots__ = ("agent_idx",)

    def __init__(self, idx: int) -> None:
        """Create an Agent with the given integer index.

        Args:
            idx: Player slot number.  Use 1–5 for real players; 0 for the
                 AGENT_NONE sentinel.
        """
        object.__setattr__(self, "agent_idx", idx)

    def __setattr__(self, *_) -> None:
        raise AttributeError("Agent is immutable — do not set attributes directly.")

    def __repr__(self) -> str:
        return f"Agent[{self.agent_idx:02d}]"

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.agent_idx)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Agent) and self.agent_idx == other.agent_idx

    def __lt__(self, other: "Agent") -> bool:
        return self.agent_idx < other.agent_idx


# Sentinel equivalent to AIWolf's AGENT_NONE constant
AGENT_NONE: Agent = Agent(0)
"""Null agent sentinel.

Used as a default return value when an agent has no valid target (e.g.
``vote()`` when no other player is alive).  The game engine treats any
method return of AGENT_NONE as "no preference" and applies a random
fallback.
"""


# ---------------------------------------------------------------------------
# Immutable game-event records
# ---------------------------------------------------------------------------

@dataclass
class Talk:
    """A single utterance recorded during the day-phase discussion.

    Attributes
    ----------
    day   : int   — Game day on which this talk was made.
    agent : Agent — The player who spoke.
    text  : str   — Raw talk string (e.g. ``"VOTE Agent[03]"``).
    """

    day:   int
    agent: Agent
    text:  str


@dataclass
class Vote:
    """A single vote cast during the day-phase elimination vote.

    Attributes
    ----------
    day    : int   — Game day of the vote.
    agent  : Agent — The player who cast the vote.
    target : Agent — The player who received the vote.
    """

    day:    int
    agent:  Agent
    target: Agent


@dataclass
class Judge:
    """The result of one Seer divine action, or one wolf attack record.

    When used for Seer divinations, ``result`` is the Species the Seer
    observes — WEREWOLF only for the actual Werewolf role (the Possessed
    appears as HUMAN).

    Attributes
    ----------
    day    : int     — Game day the action was taken.
    agent  : Agent   — The Seer who performed the divination.
    target : Agent   — The player who was investigated.
    result : Species — HUMAN or WEREWOLF as seen by the Seer.
    """

    day:    int
    agent:  Agent
    target: Agent
    result: Species

    def __bool__(self) -> bool:
        """Return False only for the empty sentinel (agent == AGENT_NONE)."""
        return self.agent != AGENT_NONE


# ---------------------------------------------------------------------------
# Per-agent game snapshot
# ---------------------------------------------------------------------------

@dataclass
class GameInfo:
    """A filtered snapshot of game state from one agent's perspective.

    The engine constructs a separate ``GameInfo`` for every living agent
    before each action, applying information hiding:

    * ``role_map`` — village-team agents see only their own role; wolf-team
      agents see all wolf-team members and their roles.
    * ``whisper_list`` / ``attack_vote_list`` — wolf-team only.
    * ``divine_result`` — Seer only; contains the result from last night.

    Agents must **not** modify any list or dict in this object.  The engine
    passes copies, but mutation would still corrupt local state.

    Attributes
    ----------
    day                : int
        Current game day (0 = initialisation, 1+ = active rounds).
    agent              : Agent
        This agent's own identity token.
    role_map           : Dict[Agent, Role]
        Visible role assignments.  Village players see only ``{self: role}``.
        Wolf-team players see ``{wolf: Role.WEREWOLF, possessed: Role.POSSESSED}``.
    status_map         : Dict[Agent, Status]
        ALIVE / DEAD for every player in the game.
    talk_list          : List[Talk]
        All talks recorded so far during the current day phase.
    whisper_list       : List[Talk]
        Wolf-team whispers from the current night phase (wolf team only).
    vote_list          : List[Vote]
        Elimination votes cast during the current day phase.
    attack_vote_list   : List[Vote]
        Wolf attack votes from the current night phase (wolf team only).
    executed_agent     : Optional[Agent]
        The player eliminated by voting today (None on day 1 before voting).
    attacked_agent     : Optional[Agent]
        The player killed by the wolf last night (None on day 1).
    divine_result      : Optional[Judge]
        The Seer's divine result from last night (Seer role only; None otherwise).
    alive_agent_list   : List[Agent]
        All players currently alive, sorted by index.
    last_dead_agent_list : List[Agent]
        Players who died since the previous day start (attacked + executed).
    """

    day:                  int
    agent:                Agent
    role_map:             Dict[Agent, Role]
    status_map:           Dict[Agent, Status]
    talk_list:            List[Talk]
    whisper_list:         List[Talk]
    vote_list:            List[Vote]
    attack_vote_list:     List[Vote]
    executed_agent:       Optional[Agent]
    attacked_agent:       Optional[Agent]
    divine_result:        Optional[Judge]
    alive_agent_list:     List[Agent]
    last_dead_agent_list: List[Agent]

    # ------------------------------------------------------------------
    # Convenience aliases mirroring the real AIWolf AbstractPlayer API
    # ------------------------------------------------------------------

    @property
    def me(self) -> Agent:
        """Alias for ``agent`` — mirrors the AIWolf ``game_info.me`` API."""
        return self.agent

    @property
    def agent_list(self) -> List[Agent]:
        """All players (alive and dead), sorted by index.

        Derived from ``status_map`` so it is always complete regardless of
        which day or phase this snapshot was taken.
        """
        return sorted(self.status_map.keys())

    @property
    def my_role(self) -> Role:
        """This agent's own Role, read from the visible ``role_map``.

        Raises:
            KeyError: If the engine failed to include this agent's role
                      in the snapshot (should never happen in normal use).
        """
        return self.role_map[self.agent]


# ---------------------------------------------------------------------------
# Static game configuration
# ---------------------------------------------------------------------------

@dataclass
class GameSetting:
    """Immutable game configuration handed to agents during ``initialize()``.

    Attributes
    ----------
    player_num    : int
        Total number of players (always 5 in the standard setup).
    role_num_map  : Dict[Role, int]
        How many of each Role exist in the game.
        Default: {VILLAGER: 2, WEREWOLF: 1, SEER: 1, POSSESSED: 1}.
    max_talk      : int
        Number of talk rounds per day phase (each alive agent speaks once
        per round).  Default: 5.
    max_skip      : int
        Maximum number of SKIP utterances allowed per agent per round.
        Set to a large value to allow unlimited skips.  Default: 999.
    """

    player_num:   int = 5
    role_num_map: Dict[Role, int] = field(
        default_factory=lambda: {
            Role.VILLAGER:  2,
            Role.WEREWOLF:  1,
            Role.SEER:      1,
            Role.POSSESSED: 1,
        }
    )
    max_talk: int = 5
    max_skip: int = 999


# ---------------------------------------------------------------------------
# Talk parser
# ---------------------------------------------------------------------------

def parse_talk(text: str, agent_lookup: Dict[str, Agent]) -> dict:
    """Tokenise a talk string into a structured intent dictionary.

    Agents communicate via plain text strings during the discussion phase.
    This function parses those strings so that other agents can react to
    declared intentions.

    Recognised topic formats
    ------------------------
    +------------------------------------+-----------------------------------+
    | Input text                         | Returned dict                     |
    +====================================+===================================+
    | ``"VOTE Agent[02]"``               | ``{"topic":"VOTE",                |
    |                                    |    "target": Agent(2)}``          |
    +------------------------------------+-----------------------------------+
    | ``"ESTIMATE Agent[03] WEREWOLF"``  | ``{"topic":"ESTIMATE",            |
    |                                    |    "target": Agent(3),            |
    |                                    |    "role": "WEREWOLF"}``          |
    +------------------------------------+-----------------------------------+
    | ``"COMINGOUT Agent[01] SEER"``     | ``{"topic":"COMINGOUT",           |
    |                                    |    "target": Agent(1),            |
    |                                    |    "role": "SEER"}``              |
    +------------------------------------+-----------------------------------+
    | ``"DIVINED Agent[04] HUMAN"``      | ``{"topic":"DIVINED",             |
    |                                    |    "target": Agent(4),            |
    |                                    |    "result": "HUMAN"}``           |
    +------------------------------------+-----------------------------------+
    | ``"SKIP"`` / ``"OVER"``            | ``{"topic": "SKIP"}`` /           |
    |                                    | ``{"topic": "OVER"}``             |
    +------------------------------------+-----------------------------------+
    | Anything unrecognised              | ``{"topic": "SKIP"}``             |
    +------------------------------------+-----------------------------------+

    Args:
        text:         The raw talk string produced by an agent's ``talk()``
                      method.
        agent_lookup: A mapping from agent string representations (e.g.
                      ``"Agent[02]"``) to ``Agent`` objects.  Build this once
                      per game from ``{str(a): a for a in status_map}``.

    Returns:
        A dict with at minimum a ``"topic"`` key.  Additional keys depend on
        the topic:  ``"target"`` (Agent), ``"role"`` (str), ``"result"`` (str).
        Unknown or malformed strings fall back to ``{"topic": "SKIP"}``.
    """
    parts = text.strip().split()
    if not parts:
        return {"topic": "SKIP"}

    topic = parts[0].upper()

    if topic in ("SKIP", "OVER"):
        return {"topic": topic}

    if topic == "VOTE" and len(parts) >= 2:
        target = agent_lookup.get(parts[1])
        if target:
            return {"topic": "VOTE", "target": target}

    if topic == "ESTIMATE" and len(parts) >= 3:
        target = agent_lookup.get(parts[1])
        if target:
            return {"topic": "ESTIMATE", "target": target, "role": parts[2].upper()}

    if topic == "COMINGOUT" and len(parts) >= 3:
        target = agent_lookup.get(parts[1])
        if target:
            return {"topic": "COMINGOUT", "target": target, "role": parts[2].upper()}

    if topic == "DIVINED" and len(parts) >= 3:
        target = agent_lookup.get(parts[1])
        if target:
            return {"topic": "DIVINED", "target": target, "result": parts[2].upper()}

    return {"topic": "SKIP"}
