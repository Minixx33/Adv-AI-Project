"""werewolf.game — Self-contained 5-player Werewolf game engine.

This module is the heart of the project.  It manages all game state, enforces
the rules, drives the agent call-sequence, and produces per-agent ``GameInfo``
snapshots that enforce information hiding.

Game flow
---------
::

    Day 0   → _setup()  → _day_zero()   (roles assigned, agents initialised)
    Day N   → _day_phase()              (discuss → vote → execute)
            → _night_phase()            (seer divines → wolf attacks)
    Repeat until _check_winner() returns a non-None team string.

Information hiding in _build_game_info()
-----------------------------------------
+-----------------------+------------------+---------+-----------+-----------+
| Field                 | Villager / Seer  | Seer    | Werewolf  | Possessed |
+=======================+==================+=========+===========+===========+
| role_map              | {self: role}     | same    | all wolves| all wolves|
+-----------------------+------------------+---------+-----------+-----------+
| whisper_list          | []               | []      | full list | full list |
+-----------------------+------------------+---------+-----------+-----------+
| attack_vote_list      | []               | []      | full list | full list |
+-----------------------+------------------+---------+-----------+-----------+
| divine_result         | None             | result  | None      | None      |
+-----------------------+------------------+---------+-----------+-----------+

Win conditions
--------------
Village wins
    When no agent with ``Role.WEREWOLF`` is alive.  A lone Possessed cannot
    kill at night, so the village is safe.
Wolf wins
    When ``len(wolf_team_alive) >= len(village_team_alive)``.  The wolf team
    includes both Werewolf and Possessed.

Classes
-------
WerewolfGame : Complete, single-game engine.

Module constants
----------------
_DEFAULT_ROLES : List[Role]  — The 5 roles used in every game.
_GAME_SETTING  : GameSetting — Singleton settings object passed to agents.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

from werewolf.agent import AbstractAgent
from werewolf.gameinfo import (
    Agent, GameInfo, GameSetting, Talk, Vote, Judge, AGENT_NONE,
)
from werewolf.const import (
    Role, Species, Status, WOLF_ROLES, VILLAGE_ROLES, role_to_species,
)
from werewolf.agents.suspicion_agent import SuspicionAgent


_DEFAULT_ROLES: List[Role] = [
    Role.VILLAGER,
    Role.VILLAGER,
    Role.WEREWOLF,
    Role.SEER,
    Role.POSSESSED,
]
"""Standard 5-player role pool.  Shuffled anew every game in ``_setup()``."""

_GAME_SETTING: GameSetting = GameSetting()
"""Singleton ``GameSetting`` shared across all games."""


class WerewolfGame:
    """Self-contained 5-player Werewolf game engine.

    Instantiate with five ``AbstractAgent`` implementations, call ``run()``,
    and read back the winner string and per-agent statistics.

    Attributes
    ----------
    _agents       : List[AbstractAgent]
        The five agent implementations in slot order.
    _role_list    : List[Role]
        The role pool to shuffle and assign at game start.
    _agent_objs   : List[Agent]
        Identity tokens ``Agent(1)`` through ``Agent(5)``.
    _agent_map    : Dict[Agent, AbstractAgent]
        Maps each identity token to its implementation.
    _role_map     : Dict[Agent, Role]
        Ground-truth role assignment for the current game.
    _status_map   : Dict[Agent, Status]
        Current alive / dead status for every player.
    _day          : int
        Current game day (0 on initialisation, 1+ during active play).
    _talk_list    : List[Talk]
        Talks accumulated during the current day phase.
    _whisper_list : List[Talk]
        Wolf whispers from the current night phase.
    _vote_list    : List[Vote]
        Elimination votes cast during the current day phase.
    _attack_vote_list : List[Vote]
        Wolf attack votes from the current night phase.
    _executed_agent   : Optional[Agent]
        Agent eliminated by vote today; None until the vote resolves.
    _attacked_agent   : Optional[Agent]
        Agent killed by the wolf last night; None until night resolves.
    _last_dead_list   : List[Agent]
        Union of last night's attacked agent and today's executed agent.
    _last_divine_result : Optional[Judge]
        Seer's divine result from last night; None if seer has not yet acted.
    _winner       : Optional[str]
        'village', 'wolf', or None if game is still in progress.
    _round_count  : int
        Number of complete day-phase rounds played.
    _actual_votes_this_round : Dict[Agent, Agent]
        Maps each voter to their actual vote target (used by SuspicionAgent).
    """

    def __init__(
        self,
        agents: List[AbstractAgent],
        roles: Optional[List[Role]] = None,
    ) -> None:
        """Create a new game with the given agent implementations.

        Args:
            agents: Exactly 5 ``AbstractAgent`` instances.  Slot i receives
                    the role that ends up at position i after shuffling.
            roles:  Optional custom role pool (length 5).  Defaults to
                    ``_DEFAULT_ROLES`` (2 Villagers, 1 Werewolf, 1 Seer,
                    1 Possessed).

        Raises:
            AssertionError: If ``len(agents) != 5``.
        """
        assert len(agents) == 5, "Exactly 5 agents required"
        self._agents    = agents
        self._role_list = list(roles or _DEFAULT_ROLES)

        self._agent_objs: List[Agent]               = [Agent(i + 1) for i in range(5)]
        self._agent_map:  Dict[Agent, AbstractAgent] = {}
        self._role_map:   Dict[Agent, Role]          = {}
        self._status_map: Dict[Agent, Status]        = {}

        self._day:            int            = 0
        self._talk_list:      List[Talk]     = []
        self._whisper_list:   List[Talk]     = []
        self._vote_list:      List[Vote]     = []
        self._attack_vote_list: List[Vote]   = []
        self._executed_agent: Optional[Agent] = None
        self._attacked_agent: Optional[Agent] = None
        self._last_dead_list: List[Agent]    = []
        self._last_divine_result: Optional[Judge] = None

        self._winner:      Optional[str] = None
        self._round_count: int           = 0

        self._actual_votes_this_round: Dict[Agent, Agent] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> str:
        """Execute a complete game from role assignment to finish.

        Calls ``_setup()`` and ``_day_zero()`` to initialise, then alternates
        ``_day_phase()`` / ``_night_phase()`` until a winner is found.

        A hard cap of 29 active rounds prevents infinite loops in degenerate
        edge cases (should never trigger in normal 5-player play).

        Returns:
            ``'village'`` or ``'wolf'``.
        """
        self._setup()
        self._day_zero()

        for day in range(1, 30):
            self._day        = day
            self._round_count += 1

            winner = self._day_phase()
            if winner:
                self._winner = winner
                self._finish_all()
                return winner

            winner = self._night_phase()
            if winner:
                self._winner = winner
                self._finish_all()
                return winner

        self._winner = "wolf"
        self._finish_all()
        return "wolf"

    @property
    def role_map(self) -> Dict[Agent, Role]:
        """Read-only copy of the ground-truth role assignment."""
        return dict(self._role_map)

    @property
    def round_count(self) -> int:
        """Number of complete day-phase rounds played."""
        return self._round_count

    @property
    def agent_map(self) -> Dict[Agent, AbstractAgent]:
        """Read-only mapping from identity tokens to agent implementations."""
        return dict(self._agent_map)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Shuffle roles and bind each agent slot to a role and identity token."""
        roles = list(self._role_list)
        random.shuffle(roles)
        for agent_obj, role, impl in zip(self._agent_objs, roles, self._agents):
            self._role_map[agent_obj]   = role
            self._status_map[agent_obj] = Status.ALIVE
            self._agent_map[agent_obj]  = impl

    def _day_zero(self) -> None:
        """Call ``update`` + ``initialize`` on every agent with the day-0 snapshot."""
        self._day = 0
        for agent_obj, impl in self._agent_map.items():
            gi = self._build_game_info(agent_obj)
            impl.update(gi)
            impl.initialize(gi, _GAME_SETTING)

    # ------------------------------------------------------------------
    # Day phase
    # ------------------------------------------------------------------

    def _day_phase(self) -> Optional[str]:
        """Run one complete day phase: ``day_start`` → discussion → voting.

        Steps:
        1. Reset talk and vote lists.
        2. Deliver ``update`` + ``day_start`` to every alive agent.
        3. Run ``max_talk`` discussion rounds (each alive agent speaks once).
        4. Collect votes, resolve by plurality (random tiebreak), mark executed.
        5. Notify ``SuspicionAgent`` instances of the vote outcome.
        6. Notify all agents via ``record_vote_outcome`` for stats tracking.
        7. Check and return the winner (or None to continue).

        Returns:
            ``'village'``, ``'wolf'``, or ``None`` if the game continues.
        """
        self._talk_list               = []
        self._vote_list               = []
        self._actual_votes_this_round = {}

        def _gi(a: Agent) -> GameInfo:
            return self._build_game_info(a)

        # day_start for every living agent
        for agent_obj in self._alive():
            impl = self._agent_map[agent_obj]
            impl.update(_gi(agent_obj))
            impl.day_start()

        # Discussion: max_talk rounds × all alive agents
        for _ in range(_GAME_SETTING.max_talk):
            for agent_obj in self._alive():
                impl = self._agent_map[agent_obj]
                impl.update(_gi(agent_obj))
                text = impl.talk() or "SKIP"
                self._talk_list.append(Talk(self._day, agent_obj, text))

        # Voting
        for agent_obj in self._alive():
            impl   = self._agent_map[agent_obj]
            impl.update(_gi(agent_obj))
            target = impl.vote()
            if target == AGENT_NONE or target not in self._status_map:
                target = self._random_alive_other(agent_obj)
            self._vote_list.append(Vote(self._day, agent_obj, target))
            self._actual_votes_this_round[agent_obj] = target

        executed = self._resolve_votes(self._vote_list)
        self._executed_agent = executed
        if executed and executed != AGENT_NONE:
            self._status_map[executed] = Status.DEAD

        # Notify SuspicionAgent instances
        self._notify_suspicion_vote_round(executed)

        # Notify all agents for vote-accuracy stats
        if executed and executed != AGENT_NONE:
            for impl in self._agent_map.values():
                if hasattr(impl, "record_vote_outcome"):
                    impl.record_vote_outcome(executed, self._role_map)

        # Build last_dead_list: attacked last night + executed today
        self._last_dead_list = []
        if self._attacked_agent and self._attacked_agent != AGENT_NONE:
            self._last_dead_list.append(self._attacked_agent)
        if executed and executed != AGENT_NONE:
            self._last_dead_list.append(executed)

        return self._check_winner()

    # ------------------------------------------------------------------
    # Night phase
    # ------------------------------------------------------------------

    def _night_phase(self) -> Optional[str]:
        """Run one complete night phase: Seer divines, then Werewolf attacks.

        Night order (rules-mandated):
        1. **Seer** calls ``divine()``.  The result is stored in
           ``_last_divine_result`` and delivered only to the Seer at the
           next ``day_start()``.  The Seer's result is recorded even if
           the wolf attacks the Seer that same night.
        2. **Werewolf** calls ``whisper()`` then ``attack()``.  The
           attacked agent is marked DEAD immediately; ``_attacked_agent``
           carries this forward to the next day's ``GameInfo``.

        The wolf attack is guarded: if the chosen target is on the wolf
        team, the engine substitutes a random non-wolf target to prevent
        self-kills.

        Returns:
            ``'village'``, ``'wolf'``, or ``None`` if the game continues.
        """
        self._whisper_list      = []
        self._attack_vote_list  = []

        def _gi(a: Agent) -> GameInfo:
            return self._build_game_info(a)

        # Seer divines
        seer = self._find_alive_role(Role.SEER)
        self._last_divine_result = None
        if seer:
            impl   = self._agent_map[seer]
            impl.update(_gi(seer))
            target = impl.divine()
            if target == AGENT_NONE or target not in self._status_map:
                target = self._random_alive_other(seer)
            species = role_to_species(self._role_map[target])
            self._last_divine_result = Judge(self._day, seer, target, species)

        # Wolf whispers then attacks
        wolf = self._find_alive_role(Role.WEREWOLF)
        self._attacked_agent = None
        if wolf:
            impl   = self._agent_map[wolf]
            impl.update(_gi(wolf))
            text = impl.whisper() or "SKIP"
            self._whisper_list.append(Talk(self._day, wolf, text))

            impl.update(_gi(wolf))
            attack_target = impl.attack()
            if attack_target == AGENT_NONE or attack_target not in self._status_map:
                attack_target = self._random_alive_other(wolf)

            # Guard against friendly-fire
            while (
                attack_target in self._role_map
                and self._role_map[attack_target] in WOLF_ROLES
            ):
                candidates = [
                    a for a in self._alive()
                    if self._role_map[a] not in WOLF_ROLES
                ]
                if not candidates:
                    break
                attack_target = random.choice(candidates)

            self._attack_vote_list.append(Vote(self._day, wolf, attack_target))
            self._attacked_agent               = attack_target
            self._status_map[attack_target]    = Status.DEAD

        return self._check_winner()

    # ------------------------------------------------------------------
    # Win-condition check
    # ------------------------------------------------------------------

    def _check_winner(self) -> Optional[str]:
        """Evaluate the current board state for a win condition.

        Village wins when no ``Role.WEREWOLF`` agent is alive (a surviving
        Possessed alone cannot threaten the village).

        Wolf team wins when their alive count equals or exceeds the alive
        village count.

        Returns:
            ``'village'``, ``'wolf'``, or ``None`` if neither condition holds.
        """
        alive          = self._alive()
        wolves_alive   = [a for a in alive if self._role_map[a] == Role.WEREWOLF]
        village_alive  = [a for a in alive if self._role_map[a] in VILLAGE_ROLES]
        wolf_team_alive = [a for a in alive if self._role_map[a] in WOLF_ROLES]

        if not wolves_alive:
            return "village"
        if len(wolf_team_alive) >= len(village_alive):
            return "wolf"
        return None

    # ------------------------------------------------------------------
    # GameInfo construction
    # ------------------------------------------------------------------

    def _build_game_info(self, agent: Agent) -> GameInfo:
        """Construct a filtered ``GameInfo`` snapshot for *agent*.

        Applies the information-hiding rules described in the module docstring:
        role_map, whisper_list, attack_vote_list, and divine_result are each
        filtered based on the requesting agent's role.

        Args:
            agent: The agent for whom the snapshot is being built.

        Returns:
            A ``GameInfo`` instance reflecting only what *agent* is allowed
            to know at this point in the game.
        """
        role         = self._role_map[agent]
        is_wolf_team = role in WOLF_ROLES

        visible_role_map = (
            {a: r for a, r in self._role_map.items() if r in WOLF_ROLES}
            if is_wolf_team
            else {agent: role}
        )
        visible_whispers      = list(self._whisper_list)      if is_wolf_team else []
        visible_attack_votes  = list(self._attack_vote_list)  if is_wolf_team else []
        divine_result: Optional[Judge] = (
            self._last_divine_result if role == Role.SEER else None
        )

        return GameInfo(
            day=self._day,
            agent=agent,
            role_map=visible_role_map,
            status_map=dict(self._status_map),
            talk_list=list(self._talk_list),
            whisper_list=visible_whispers,
            vote_list=list(self._vote_list),
            attack_vote_list=visible_attack_votes,
            executed_agent=self._executed_agent,
            attacked_agent=self._attacked_agent,
            divine_result=divine_result,
            alive_agent_list=self._alive(),
            last_dead_agent_list=list(self._last_dead_list),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _alive(self) -> List[Agent]:
        """Return all currently alive agents, sorted by index.

        Returns:
            List of ``Agent`` objects whose ``status_map`` entry is ALIVE.
        """
        return sorted(
            [a for a, s in self._status_map.items() if s == Status.ALIVE]
        )

    def _find_alive_role(self, role: Role) -> Optional[Agent]:
        """Find the first living agent assigned *role*.

        Args:
            role: The role to search for.

        Returns:
            The matching ``Agent``, or ``None`` if no alive agent holds *role*.
        """
        for a in self._alive():
            if self._role_map[a] == role:
                return a
        return None

    def _random_alive_other(self, agent: Agent) -> Agent:
        """Return a random alive agent that is not *agent*.

        Used as a fallback when an agent returns ``AGENT_NONE`` from a
        decision method.

        Args:
            agent: The agent to exclude from selection.

        Returns:
            A random alive other agent, or *agent* itself if no others exist.
        """
        others = [a for a in self._alive() if a != agent]
        return random.choice(others) if others else agent

    def _resolve_votes(self, votes: List[Vote]) -> Optional[Agent]:
        """Determine the executed agent by plurality vote with random tiebreak.

        Args:
            votes: All ``Vote`` records cast this round.

        Returns:
            The ``Agent`` who received the most votes, chosen randomly among
            tied leaders, or ``None`` if no votes were cast.
        """
        tally: Dict[Agent, int] = {}
        for v in votes:
            tally[v.target] = tally.get(v.target, 0) + 1
        if not tally:
            return None
        max_votes = max(tally.values())
        leaders   = [a for a, cnt in tally.items() if cnt == max_votes]
        return random.choice(leaders)

    def _notify_suspicion_vote_round(self, executed: Optional[Agent]) -> None:
        """Deliver vote-round data to all living ``SuspicionAgent`` instances.

        ``SuspicionAgent.record_vote_round()`` is the hook that updates sigma
        from the five behavioural detectors after each voting round.

        Args:
            executed: The agent eliminated by the vote, or None.
        """
        for agent_obj, impl in self._agent_map.items():
            if (
                isinstance(impl, SuspicionAgent)
                and self._status_map.get(agent_obj) == Status.ALIVE
            ):
                impl.record_vote_round(
                    self._actual_votes_this_round,
                    executed,
                    self._role_map,
                )

    def _finish_all(self) -> None:
        """Deliver ``update`` + ``finish`` to every agent once the game ends."""
        for agent_obj, impl in self._agent_map.items():
            gi = self._build_game_info(agent_obj)
            impl.update(gi)
            impl.finish()
