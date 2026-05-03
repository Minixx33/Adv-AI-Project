"""
werewolf/gameinfo.py

Immutable data containers passed to agents each turn — mirrors the AIWolf
platform's GameInfo / GameSetting interface so agents stay portable.

Hierarchy / contents:
  Talk        — one utterance (day, turn, agent_id, text)
  Vote        — one vote declaration (day, voter_id, target_id)
  Judge       — result of a Seer divine or Bodyguard guard action
  GameInfo    — full snapshot of *visible* game state for one specific agent
                  • role_map contains only roles that agent can legitimately see
                  • alive_agents property derived from status_map
  GameSetting — static game parameters broadcast to all agents at initialize()
"""

from dataclasses import dataclass, field
from typing import Optional

from .const import Role, Status


# ── Talk-token vocabulary (all are valid text values in Talk.text) ──────────
# Existing tokens (unchanged):
#   VOTE Pn              — intend to vote for Pn
#   ESTIMATE Pn WEREWOLF — probabilistic accusation
#   COMINGOUT Pn ROLE    — role claim (usually for self: COMINGOUT P3 SEER)
#   DIVINED Pn RESULT    — seer claim (RESULT = WEREWOLF | HUMAN)
#   Skip                 — pass (no utterance this call)
#   Over                 — done talking for this round
#
# New tokens added in v2 update:
#   AGREE Pn             — agree with Pn's last statement
#   DISAGREE Pn          — disagree with Pn's last statement
#   REQUEST Pn action    — ask Pn to do something (e.g. REQUEST P3 reveal_role)
#   INQUIRE Pn           — question Pn's behaviour or claim
#   BECAUSE reason       — justify previous statement (see BECAUSE_REASONS)
#   ATTACK Pn            — strong public accusation targeting Pn
#   DEFEND Pn            — defend Pn against accusations
#   NOT Pn ROLE          — assert Pn is NOT a particular role
#   IDENTIFIED Pn ROLE   — medium identifies a dead player's role
#   GUARDED Pn           — bodyguard claims to have protected Pn
#
# BECAUSE reason codes (standardised):
BECAUSE_REASONS = frozenset({
    "divine_result", "seer_claim_conflict", "vote_pattern", "bandwagon",
    "defended_wolf", "claim_contradiction", "silent_player",
    "aggressive_accuser", "protect_seer", "last_resort",
})


@dataclass
class Talk:
    day:   int
    turn:  int   # round within the day (0-indexed)
    agent: int   # 1-indexed player ID who spoke
    text:  str


@dataclass
class Vote:
    day:    int
    agent:  int  # voter, 1-indexed
    target: int  # voted-for player, 1-indexed


@dataclass
class Judge:
    """Result of a Seer divine or Bodyguard guard action."""
    day:    int
    agent:  int   # who performed the action (1-indexed)
    target: int   # who was targeted (1-indexed)
    result: str   # "HUMAN" or "WEREWOLF"  (Species.value)


@dataclass
class GameInfo:
    """
    Snapshot of game state visible to one specific agent for the current moment.

    Construction contract (enforced by WerewolfGame._build_game_info):
      • role_map always contains the agent's own role.
      • Werewolves additionally see all other live werewolves in role_map.
      • Dead players' roles are always in role_map (publicly revealed).
      • divine_result is set only for the Seer, and only from day 2 onward.
      • guard_result  is set only for the Bodyguard, and only from day 2 onward.
      • whisper_list  is set only for Werewolves.
    """
    day:          int
    agent:        int                    # this agent's player ID (1-indexed)
    role_map:     dict                   # {player_id: Role}  — visible roles only
    status_map:   dict                   # {player_id: Status} — all players

    talk_list:        list = field(default_factory=list)  # Talk objects, current day
    whisper_list:     list = field(default_factory=list)  # Talk objects, wolves only
    vote_list:        list = field(default_factory=list)  # Vote objects, current day
    attack_vote_list: list = field(default_factory=list)  # Vote objects, wolves only

    divine_result:       Optional[Judge] = None  # Seer: last night's result
    guard_result:        Optional[Judge] = None  # Bodyguard: last night's result
    executed_agent:      Optional[int]  = None   # eliminated by vote today
    attacked_agent:      Optional[int]  = None   # killed by wolves last night
    last_dead_agent_list: list = field(default_factory=list)  # all who died overnight

    @property
    def alive_agents(self) -> list:
        return [p for p, s in self.status_map.items() if s == Status.ALIVE]


@dataclass
class GameSetting:
    """
    Static game parameters, broadcast to every agent at initialize().
    Mirrors AIWolf's GameSetting for interface compatibility.
    """
    player_num:       int
    role_num_map:     dict   # {Role: count}
    max_talk:         int = 5    # talk rounds per day
    max_talk_turn:    int = 10   # max utterances per player per day (unused in engine)
    talk_on_first_day: bool = False
    enable_no_attack:  bool = False
    enable_role_request: bool = False
