"""
werewolf/const.py

Enumerated constants and static role-configuration tables for the engine.

Hierarchy / contents:
  Role          — player roles: VILLAGER, WEREWOLF, SEER, POSSESSED, BODYGUARD
  Species       — what the Seer sees when divining: HUMAN, WEREWOLF
  Status        — player liveness: ALIVE, DEAD
  Team          — winning-team alignment: VILLAGE, WOLF

  ROLE_TO_SPECIES — maps each Role to the Species the Seer would see
  ROLE_TO_TEAM    — maps each Role to the winning-team it belongs to
  ROLE_CONFIG     — maps player count {5, 8, 10} to {Role: count} dict
  WOLF_ROLES      — set of roles that count as "werewolf" for win condition
"""

from enum import Enum


class Role(Enum):
    VILLAGER  = "VILLAGER"
    WEREWOLF  = "WEREWOLF"
    SEER      = "SEER"
    POSSESSED = "POSSESSED"
    BODYGUARD = "BODYGUARD"


class Species(Enum):
    """What the Seer perceives when they divine a player."""
    HUMAN     = "HUMAN"
    WEREWOLF  = "WEREWOLF"


class Status(Enum):
    ALIVE = "ALIVE"
    DEAD  = "DEAD"


class Team(Enum):
    """The team that wins when this player's side achieves victory."""
    VILLAGE = "VILLAGE"
    WOLF    = "WOLF"


# ---- lookup tables ------------------------------------------------

ROLE_TO_SPECIES: dict = {
    Role.VILLAGER:  Species.HUMAN,
    Role.WEREWOLF:  Species.WEREWOLF,
    Role.SEER:      Species.HUMAN,
    Role.POSSESSED: Species.HUMAN,   # Possessed reads as HUMAN to the Seer
    Role.BODYGUARD: Species.HUMAN,
}

ROLE_TO_TEAM: dict = {
    Role.VILLAGER:  Team.VILLAGE,
    Role.WEREWOLF:  Team.WOLF,
    Role.SEER:      Team.VILLAGE,
    Role.POSSESSED: Team.WOLF,       # Possessed wins with the wolf team
    Role.BODYGUARD: Team.VILLAGE,
}

# Roles that count as "literal werewolf" for the wolf-win condition check.
# Possessed is wolf-team but NOT in this set — wolves need to out-number
# non-werewolves (including Possessed) to trigger their win.
WOLF_ROLES: frozenset = frozenset({Role.WEREWOLF})

# Roles that can see each other during the whisper/attack night phase.
WOLF_TEAM_ROLES: frozenset = frozenset({Role.WEREWOLF, Role.POSSESSED})

# ---- player-count role distributions --------------------------------

ROLE_CONFIG: dict = {
    5: {
        Role.VILLAGER:  2,
        Role.WEREWOLF:  1,
        Role.SEER:      1,
        Role.POSSESSED: 1,
    },
    8: {
        Role.VILLAGER:  4,
        Role.WEREWOLF:  2,
        Role.SEER:      1,
        Role.POSSESSED: 1,
    },
    10: {
        Role.VILLAGER:  5,
        Role.WEREWOLF:  2,
        Role.SEER:      1,
        Role.POSSESSED: 1,
        Role.BODYGUARD: 1,
    },
}

VALID_PLAYER_COUNTS: tuple = (5, 8, 10)
