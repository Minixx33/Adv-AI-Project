"""
werewolf/game.py

Self-contained in-process Werewolf game engine.

Hierarchy / contents:
  WerewolfGame   — runs one complete game given pre-built agents and roles
    _initialize()       — sends GameInfo/GameSetting to every agent
    _day_start()        — notifies agents of day start
    _talk_phase()       — runs max_talk_rounds of public talk
    _vote_phase()       — collects votes, breaks ties, eliminates player
    _divine_phase()     — Seer divines one player
    _guard_phase()      — Bodyguard protects one player (10-player only)
    _whisper_phase()    — Wolves coordinate via whisper
    _attack_phase()     — Wolves vote on attack target, resolved vs guard
    _eliminate()        — marks player DEAD, appends event, updates all agents
    _check_winner()     — Team.VILLAGE / Team.WOLF / None
    _build_game_info()  — constructs role-limited GameInfo per agent
    _update_all()       — broadcasts current GameInfo to every alive agent
    run()               — top-level entry; returns GameRecord dict

GameRecord (returned by run()):
  {
    "game_id":              int,
    "n_players":            int,
    "roles":                {player_id: Role},
    "agent_names":          {player_id: str},
    "winner":               Team,
    "n_days":               int,
    "events":               [event_dict, ...],
    "suspicion_agent_id":   int | None,
    "suspicion_agent_role": str | None,
  }

Event dict keys (varies by type):
  Common:      type, day
  "day_start"  —
  "talk"       — round, agent, text
  "vote"       — agent, target
  "execution"  — agent, role, cause="vote"
  "night_start"—
  "divine"     — seer, target, result ("HUMAN"/"WEREWOLF")
  "guard"      — bodyguard, target
  "whisper"    — agent, text
  "attack"     — target
  "guard_saved"— target
  "death"      — agent, role, cause="attack"
  "game_end"   — winner (Team.value), n_days
"""

import random
from collections import Counter
from typing import Optional

from .const import Role, Status, Team, WOLF_ROLES, ROLE_TO_SPECIES, ROLE_CONFIG
from .gameinfo import GameInfo, GameSetting, Talk, Vote, Judge
from .agent import AbstractAgent


class WerewolfGame:
    """
    Runs one complete Werewolf game in-process.

    Parameters
    ----------
    agents : dict[int, AbstractAgent]
        Player-ID → agent. Player IDs must be consecutive 1-indexed ints.
    roles  : dict[int, Role]
        Player-ID → Role. Must cover every key in `agents`.
    game_id : int
        Identifier echoed in the returned GameRecord.
    seed : int | None
        RNG seed for vote tie-breaking and other internal randomness.
    max_talk_rounds : int
        Talk rounds per day (default 5, mirrors AIWolf default).
    verbose : bool
        Print game progress to stdout when True.
    suspicion_agent_ids : list[int] | None
        Player IDs of all suspicion-enhanced agents.
        Also accepts the legacy keyword ``suspicion_agent_id`` (single int).
    """

    MAX_DAYS = 30  # safety valve against infinite loops

    def __init__(
        self,
        agents:     dict,
        roles:      dict,
        game_id:    int  = 0,
        seed:       Optional[int] = None,
        max_talk_rounds: int = 5,
        verbose:    bool = False,
        suspicion_agent_ids = None,
        suspicion_agent_id:  Optional[int] = None,  # legacy single-ID alias
    ):
        self.agents    = agents                   # {pid: agent}
        self.roles     = roles                    # {pid: Role}
        self.game_id   = game_id
        self.rng       = random.Random(seed)
        self.max_talk_rounds = max_talk_rounds
        self.verbose   = verbose
        # Merge legacy single-id and new list form
        ids = list(suspicion_agent_ids or [])
        if suspicion_agent_id is not None and suspicion_agent_id not in ids:
            ids.append(suspicion_agent_id)
        self.suspicion_agent_ids: list = ids

        self.player_ids: list = sorted(agents.keys())
        self.n_players:  int  = len(self.player_ids)
        self.status: dict     = {p: Status.ALIVE for p in self.player_ids}
        self.day:    int      = 1

        # Accumulate game events for logging
        self.events: list = []

        # Per-day state (reset each day)
        self.day_talks:   list = []
        self.day_votes:   list = []

        # Night state
        self.night_whispers: list = []
        self.divine_history: dict = {}   # {day: Judge}
        self.guard_history:  dict = {}   # {day: Judge}

        self._game_setting = self._make_setting()

    # ------------------------------------------------------------------ #
    #  Public entry point                                                  #
    # ------------------------------------------------------------------ #

    def run(self) -> dict:
        """Run a complete game and return the GameRecord dict."""
        self._initialize()

        for _ in range(self.MAX_DAYS):
            # ---- Day phase ------------------------------------------ #
            self._log({"type": "day_start", "day": self.day})
            self._day_start()
            self._talk_phase()
            executed = self._vote_phase()
            self._eliminate(executed, cause="vote")

            winner = self._check_winner()
            if winner:
                break

            # ---- Night phase ---------------------------------------- #
            self._log({"type": "night_start", "day": self.day})
            self._divine_phase()
            guard_target = self._guard_phase()
            self._whisper_phase()
            attack_target = self._attack_phase()
            self._resolve_attack(attack_target, guard_target)

            winner = self._check_winner()
            if winner:
                break

            # Advance day
            self.day += 1
            self.day_talks  = []
            self.day_votes  = []
            self.night_whispers = []
        else:
            # Fallback — should not happen with proper win conditions
            winner = Team.VILLAGE

        self._finish_all()
        self._log({"type": "game_end", "winner": winner.value,
                   "n_days": self.day})

        if self.verbose:
            print(f"Game {self.game_id}: {winner.value} wins in {self.day} days")

        return self._build_game_record(winner)

    # ------------------------------------------------------------------ #
    #  Phase implementations                                               #
    # ------------------------------------------------------------------ #

    def _initialize(self):
        setting = self._game_setting
        for pid, agent in self.agents.items():
            gi = self._build_game_info(pid)
            agent.initialize(gi, setting)

    def _day_start(self):
        for pid, agent in self._alive_agents():
            agent.day_start()

    _TOKEN_BUDGET = 5  # max utterances per agent per round

    def _talk_phase(self):
        for rnd in range(self.max_talk_rounds):
            # Randomise speaker order each round
            order = list(self._alive_agents())
            self.rng.shuffle(order)
            for pid, agent in order:
                tokens_used = 0
                while tokens_used < self._TOKEN_BUDGET:
                    text = agent.talk()
                    talk = Talk(day=self.day, turn=rnd, agent=pid, text=text)
                    self.day_talks.append(talk)
                    self._log({"type": "talk", "day": self.day, "round": rnd,
                               "agent": pid, "text": text})
                    self._update_all()
                    tokens_used += 1
                    if text.strip().lower() in ("skip", "over"):
                        break

    def _vote_phase(self) -> int:
        """Collect votes, resolve plurality (random tiebreak), return loser."""
        vote_map: dict = {}
        for pid, agent in self._alive_agents():
            target = agent.vote()
            # Guard: target must be alive and not self
            if target not in self.status or self.status[target] != Status.ALIVE:
                target = self.rng.choice(
                    [p for p in self.player_ids
                     if self.status[p] == Status.ALIVE and p != pid]
                    or [pid]
                )
            vote_map[pid] = target
            v = Vote(day=self.day, agent=pid, target=target)
            self.day_votes.append(v)
            self._log({"type": "vote", "day": self.day,
                       "agent": pid, "target": target})

        # Notify all suspicion agents with the complete vote map so detectors
        # can update before each trace snapshot is captured.
        for sid in self.suspicion_agent_ids:
            susp = self.agents.get(sid)
            if susp is not None and hasattr(susp, 'notify_vote_phase_complete'):
                susp.notify_vote_phase_complete(self.day, vote_map)

        executed = self._plurality(vote_map)
        return executed

    def _divine_phase(self):
        seer = self._alive_with_role(Role.SEER)
        if seer is None:
            return
        target = self.agents[seer].divine()
        if target not in self.status or self.status[target] != Status.ALIVE:
            candidates = [p for p in self.player_ids
                          if self.status[p] == Status.ALIVE and p != seer]
            target = self.rng.choice(candidates) if candidates else seer
        result = ROLE_TO_SPECIES[self.roles[target]].value
        judge  = Judge(day=self.day, agent=seer, target=target, result=result)
        self.divine_history[self.day] = judge
        self._log({"type": "divine", "day": self.day,
                   "seer": seer, "target": target, "result": result})
        # Give seer the result immediately
        gi = self._build_game_info(seer, divine_result=judge)
        self.agents[seer].update(gi, self._game_setting)

    def _guard_phase(self) -> Optional[int]:
        bodyguard = self._alive_with_role(Role.BODYGUARD)
        if bodyguard is None:
            return None
        target = self.agents[bodyguard].guard()
        if target not in self.status or self.status[target] != Status.ALIVE:
            candidates = [p for p in self.player_ids
                          if self.status[p] == Status.ALIVE and p != bodyguard]
            target = self.rng.choice(candidates) if candidates else bodyguard
        judge = Judge(day=self.day, agent=bodyguard, target=target, result="GUARD")
        self.guard_history[self.day] = judge
        self._log({"type": "guard", "day": self.day,
                   "bodyguard": bodyguard, "target": target})
        gi = self._build_game_info(bodyguard, guard_result=judge)
        self.agents[bodyguard].update(gi, self._game_setting)
        return target

    def _whisper_phase(self):
        wolves = [pid for pid, _ in self._alive_agents()
                  if self.roles[pid] == Role.WEREWOLF]
        for pid in wolves:
            text = self.agents[pid].whisper()
            self.night_whispers.append(
                Talk(day=self.day, turn=0, agent=pid, text=text)
            )
            self._log({"type": "whisper", "day": self.day,
                       "agent": pid, "text": text})
        # Broadcast whispers to all wolves
        for pid in wolves:
            gi = self._build_game_info(pid, whisper_list=list(self.night_whispers))
            self.agents[pid].update(gi, self._game_setting)

    def _attack_phase(self) -> Optional[int]:
        wolves = [pid for pid, _ in self._alive_agents()
                  if self.roles[pid] == Role.WEREWOLF]
        if not wolves:
            return None
        attack_votes: dict = {}
        for pid in wolves:
            target = self.agents[pid].attack()
            # Target must be alive and not a wolf
            valid = [p for p in self.player_ids
                     if self.status[p] == Status.ALIVE
                     and self.roles[p] != Role.WEREWOLF]
            if target not in valid:
                target = self.rng.choice(valid) if valid else None
            if target is not None:
                attack_votes[pid] = target
        if not attack_votes:
            return None
        self._log({"type": "attack", "day": self.day,
                   "votes": attack_votes})
        return self._plurality(attack_votes)

    def _resolve_attack(self, attack_target: Optional[int],
                        guard_target: Optional[int]):
        if attack_target is None:
            return
        if attack_target == guard_target:
            self._log({"type": "guard_saved", "day": self.day,
                       "target": attack_target})
            return
        self._eliminate(attack_target, cause="attack")

    def _eliminate(self, player_id: int, cause: str):
        self.status[player_id] = Status.DEAD
        role = self.roles[player_id]
        event = {"type": "execution" if cause == "vote" else "death",
                 "day": self.day, "agent": player_id,
                 "role": role.value, "cause": cause}
        self._log(event)
        if self.verbose:
            print(f"  Day {self.day}: P{player_id} ({role.value}) eliminated"
                  f" by {cause}")
        # Notify all suspicion agents of the revealed role (updates D5 claim checks)
        for sid in self.suspicion_agent_ids:
            susp = self.agents.get(sid)
            if susp is not None and hasattr(susp, 'observe_execution'):
                susp.observe_execution(self.day, player_id, role.value)
        self._update_all()

    def _finish_all(self):
        for _, agent in self.agents.items():
            agent.finish()

    # ------------------------------------------------------------------ #
    #  Win-condition check                                                 #
    # ------------------------------------------------------------------ #

    def _check_winner(self) -> Optional[Team]:
        alive = [p for p in self.player_ids if self.status[p] == Status.ALIVE]
        alive_wolves = [p for p in alive if self.roles[p] in WOLF_ROLES]
        alive_others = [p for p in alive if self.roles[p] not in WOLF_ROLES]

        if not alive_wolves:
            return Team.VILLAGE
        if len(alive_wolves) >= len(alive_others):
            return Team.WOLF
        return None

    # ------------------------------------------------------------------ #
    #  GameInfo construction                                               #
    # ------------------------------------------------------------------ #

    def _build_game_info(
        self,
        agent_id: int,
        *,
        whisper_list: list = None,
        divine_result: Judge = None,
        guard_result:  Judge = None,
        executed_agent: int = None,
        attacked_agent: int = None,
        last_dead_agent_list: list = None,
    ) -> GameInfo:
        """
        Build a role-filtered GameInfo snapshot for agent_id.
        Wolves see each other. Dead players' roles are public.
        Seer/Bodyguard night results are agent-specific.
        """
        role_map: dict = {}

        # Own role always visible
        role_map[agent_id] = self.roles[agent_id]

        # Wolves see each other's roles
        if self.roles[agent_id] == Role.WEREWOLF:
            for pid, role in self.roles.items():
                if role == Role.WEREWOLF:
                    role_map[pid] = role

        # Dead players' roles are publicly revealed
        for pid in self.player_ids:
            if self.status[pid] == Status.DEAD:
                role_map[pid] = self.roles[pid]

        # Night result defaults: look up stored history if not overridden
        if divine_result is None and self.roles.get(agent_id) == Role.SEER:
            prev_day = self.day - 1
            if prev_day in self.divine_history:
                divine_result = self.divine_history[prev_day]

        if guard_result is None and self.roles.get(agent_id) == Role.BODYGUARD:
            prev_day = self.day - 1
            if prev_day in self.guard_history:
                guard_result = self.guard_history[prev_day]

        w_list = []
        if self.roles.get(agent_id) == Role.WEREWOLF:
            w_list = list(whisper_list or self.night_whispers)

        return GameInfo(
            day=self.day,
            agent=agent_id,
            role_map=role_map,
            status_map=dict(self.status),
            talk_list=list(self.day_talks),
            whisper_list=w_list,
            vote_list=list(self.day_votes),
            divine_result=divine_result,
            guard_result=guard_result,
            executed_agent=executed_agent,
            attacked_agent=attacked_agent,
            last_dead_agent_list=list(last_dead_agent_list or []),
        )

    def _update_all(self):
        """Broadcast current state to every alive agent."""
        setting = self._game_setting
        for pid, agent in self.agents.items():
            gi = self._build_game_info(pid)
            agent.update(gi, setting)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def _alive_agents(self):
        """Yield (player_id, agent) for every alive player."""
        for pid in self.player_ids:
            if self.status[pid] == Status.ALIVE:
                yield pid, self.agents[pid]

    def _alive_with_role(self, role: Role) -> Optional[int]:
        """Return the player_id of the first alive player with the given role."""
        for pid in self.player_ids:
            if self.status[pid] == Status.ALIVE and self.roles[pid] == role:
                return pid
        return None

    def _plurality(self, vote_map: dict) -> int:
        """
        Return the player with the most votes.
        Breaks ties uniformly at random.
        """
        counts = Counter(vote_map.values())
        if not counts:
            return self.rng.choice(
                [p for p in self.player_ids if self.status[p] == Status.ALIVE]
            )
        max_votes = max(counts.values())
        tied = [p for p, c in counts.items() if c == max_votes]
        return self.rng.choice(tied)

    def _log(self, event: dict):
        self.events.append(event)

    def _make_setting(self) -> GameSetting:
        from .const import ROLE_CONFIG
        role_dist = ROLE_CONFIG.get(self.n_players, {})
        return GameSetting(
            player_num=self.n_players,
            role_num_map=dict(role_dist),
            max_talk=self.max_talk_rounds,
        )

    def _build_game_record(self, winner: Team) -> dict:
        first_sid = self.suspicion_agent_ids[0] if self.suspicion_agent_ids else None
        susp_role = self.roles[first_sid].value if first_sid and first_sid in self.roles else None
        return {
            "game_id":              self.game_id,
            "n_players":            self.n_players,
            "roles":                {p: r.value for p, r in self.roles.items()},
            "agent_names":          {p: a.name   for p, a in self.agents.items()},
            "status":               {p: s.value  for p, s in self.status.items()},
            "winner":               winner,
            "n_days":               self.day,
            "events":               list(self.events),
            "suspicion_agent_ids":  list(self.suspicion_agent_ids),
            "suspicion_agent_id":   first_sid,   # backward-compat alias
            "suspicion_agent_role": susp_role,
        }
