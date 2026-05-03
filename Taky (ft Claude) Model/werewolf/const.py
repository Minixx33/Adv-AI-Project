"""werewolf.const — Game-wide enumerations and role-classification helpers.

This module is imported by every other module in the package.  Keep it
free of imports from within the package to avoid circular dependencies.

Enumerations
------------
Role    : The four roles a player can hold in a 5-player game.
Status  : Whether a player is currently alive or dead.
Species : The canonical two-faction classification used by the Seer.

Constants
---------
WOLF_ROLES    : frozenset of Role values that belong to the wolf team.
VILLAGE_ROLES : frozenset of Role values that belong to the village team.

Functions
---------
role_to_species(role) -> Species
    Map a Role to the Species the Seer would see when divining that player.
role_to_team(role) -> str
    Map a Role to a plain-string team identifier ('wolf' or 'village').
"""
from __future__ import annotations
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Role(Enum):
    """The role assigned to a player at the start of the game.

    Values
    ------
    VILLAGER  : Standard village-team member with no special ability.
    WEREWOLF  : Wolf-team killer; attacks one player each night.
    SEER      : Village-team investigator; learns one player's Species per night.
    POSSESSED : Wolf-team ally who appears as HUMAN to the Seer.
    """

    VILLAGER  = "VILLAGER"
    WEREWOLF  = "WEREWOLF"
    SEER      = "SEER"
    POSSESSED = "POSSESSED"


class Status(Enum):
    """Whether a player is currently in the game.

    Values
    ------
    ALIVE : The player is still participating.
    DEAD  : The player has been eliminated (executed or attacked).
    """

    ALIVE = "ALIVE"
    DEAD  = "DEAD"


class Species(Enum):
    """The two-faction classification revealed by a Seer divine action.

    The Seer never learns the exact Role of a target — only whether that
    target belongs to the human (village-aligned) or werewolf faction.

    Values
    ------
    HUMAN     : The target is village-aligned (Villager, Seer, or Possessed).
    WEREWOLF  : The target is the actual Werewolf role.
    """

    HUMAN    = "HUMAN"
    WEREWOLF = "WEREWOLF"


# ---------------------------------------------------------------------------
# Team membership sets
# ---------------------------------------------------------------------------

WOLF_ROLES: frozenset[Role] = frozenset({Role.WEREWOLF, Role.POSSESSED})
"""Roles whose victory condition is the wolf team winning."""

VILLAGE_ROLES: frozenset[Role] = frozenset({Role.VILLAGER, Role.SEER})
"""Roles whose victory condition is the village team winning."""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def role_to_species(role: Role) -> Species:
    """Return the Species that the Seer would observe when divining *role*.

    The Possessed player appears as HUMAN to the Seer, which is the key
    asymmetry that makes the Possessed role dangerous for the village team.

    Args:
        role: The true Role of the player being divined.

    Returns:
        Species.WEREWOLF if role is Role.WEREWOLF, otherwise Species.HUMAN.
    """
    return Species.WEREWOLF if role == Role.WEREWOLF else Species.HUMAN


def role_to_team(role: Role) -> str:
    """Return a plain-string team label for *role*.

    Used by GameRunner when deciding whether a given role won a game.

    Args:
        role: Any Role enum value.

    Returns:
        'wolf' for WOLF_ROLES members, 'village' for VILLAGE_ROLES members.
    """
    return "wolf" if role in WOLF_ROLES else "village"
