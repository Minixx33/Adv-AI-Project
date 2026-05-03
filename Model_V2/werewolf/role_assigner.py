"""
werewolf/role_assigner.py

Assigns roles to player slots while enforcing the hard constraint from §3.9:
the suspicion-enhanced agent must never receive the WEREWOLF role.

Hierarchy / contents:
  assign_roles(player_ids, n_players, rng, suspicion_player_id=None)
    → dict[int, Role]

  _shuffle_excluding(pool, exclude_idx, rng)
    → list[Role]  — helper: Fisher-Yates with a forced non-wolf first slot

Algorithm (§3.9 spec):
  1. Build the full role list from ROLE_CONFIG[n_players].
  2. If suspicion_player_id is given, pick a non-WEREWOLF role for it first,
     then shuffle the remaining roles among the remaining players.
  3. Otherwise, shuffle roles uniformly at random.
  4. Bodyguard-special: D4 detector is deactivated for 5-player games (only
     one wolf so protection pattern cannot exist); this is handled at detector
     level, not here.

The suspicion agent CAN be assigned POSSESSED (§3.9 last bullet).
"""

import random
from typing import Optional

from .const import Role, ROLE_CONFIG, WOLF_ROLES


def assign_roles(
    player_ids:           list,
    n_players:            int,
    rng:                  random.Random,
    suspicion_player_id:  Optional[int]  = None,   # legacy single-ID form
    suspicion_player_ids: Optional[list] = None,   # new list form
) -> dict:
    """
    Return {player_id: Role} with optional no-wolf constraint on one or more slots.

    Parameters
    ----------
    player_ids : list[int]
        Ordered list of 1-indexed player IDs (length must equal n_players).
    n_players : int
        Must be in {5, 8, 10}.
    rng : random.Random
        Seeded RNG for reproducibility.
    suspicion_player_id : int | None
        Legacy single-ID form.  Merged into suspicion_player_ids.
    suspicion_player_ids : list[int] | None
        All suspicion-enhanced players; each is guaranteed NOT to receive
        Role.WEREWOLF.

    Raises
    ------
    ValueError
        If n_players is not supported or a suspicion ID is not in player_ids.
    """
    if n_players not in ROLE_CONFIG:
        raise ValueError(
            f"Unsupported player count {n_players}. Must be one of "
            f"{sorted(ROLE_CONFIG.keys())}."
        )
    if len(player_ids) != n_players:
        raise ValueError(
            f"player_ids length ({len(player_ids)}) must equal "
            f"n_players ({n_players})."
        )

    # Merge legacy single-ID and new list form
    susp_ids: list = list(suspicion_player_ids or [])
    if suspicion_player_id is not None and suspicion_player_id not in susp_ids:
        susp_ids.append(suspicion_player_id)

    for sid in susp_ids:
        if sid not in player_ids:
            raise ValueError(f"suspicion player {sid} not in player_ids.")

    # Build flat role list
    role_dist = ROLE_CONFIG[n_players]
    roles_pool: list = []
    for role, count in role_dist.items():
        roles_pool.extend([role] * count)

    if not susp_ids:
        rng.shuffle(roles_pool)
        return dict(zip(player_ids, roles_pool))

    # Constraint: every suspicion player must not be WEREWOLF
    wolf_roles_in_pool = [r for r in roles_pool if r in WOLF_ROLES]
    non_wolf_roles     = [r for r in roles_pool if r not in WOLF_ROLES]

    if len(non_wolf_roles) < len(susp_ids):
        raise ValueError(
            f"Not enough non-wolf roles ({len(non_wolf_roles)}) for "
            f"{len(susp_ids)} suspicion agents."
        )

    # Pick one non-wolf role per suspicion agent
    assignment: dict = {}
    for sid in susp_ids:
        idx = rng.randrange(len(non_wolf_roles))
        assignment[sid] = non_wolf_roles.pop(idx)

    # Assign remaining roles randomly to non-suspicion players
    remaining_roles = wolf_roles_in_pool + non_wolf_roles
    rng.shuffle(remaining_roles)
    remaining_ids = [pid for pid in player_ids if pid not in susp_ids]
    for pid, role in zip(remaining_ids, remaining_roles):
        assignment[pid] = role

    return assignment


def validate_composition(composition: dict, n_players: int) -> None:
    """
    Validate that the agent composition sums to n_players.
    Raises ValueError with a clear message on mismatch (spec §3.3).
    """
    total = sum(composition.values())
    if total != n_players:
        raise ValueError(
            f"Agent composition sums to {total}, but n_players={n_players}. "
            f"Counts: {composition}"
        )
