"""
suspicion/detectors.py

Five behavioural detectors (D1-D5) for the SuspicionModule (v2 update).

Hierarchy / contents:
  DetectorState      — per-player running statistics for all five detectors
  compute_d1(state)  — Vote-Accusation Mismatch  (now ATTACK-token based)
  compute_d2(state)  — Bandwagon Index            (now AGREE-token based)
  compute_d3(state)  — Pressure-Triggered Accusations (INQUIRE+ATTACK pressure)
  compute_d4(state)  — Protection Patterns        (DEFEND/DISAGREE/NOT tokens)
  compute_d5(state)  — Claim Consistency          (COMINGOUT + DIVINED tokens)
  compute_E(d_vals, weights) → float  — composite evidence score

All detector outputs are in [0, 1].
DetectorState is updated by SuspicionModule.observe_* methods.

Detector formulae (v2 update):
  D1(p) = mismatches(p) / max(1, attack_rounds(p))
           where mismatch = ATTACK target ≠ actual vote target
  D2(p) = agree_majority(p) / max(1, total_agrees(p))
           where agree_majority = AGREE emitted when agreed-with statement
           targeted a player already with majority accusation
  D3(p) = max(0, pearson_correlation(pressure_series, aggression_series))
           pressure(p, r)    = count of (INQUIRE p) + (ATTACK p) by others in round r
           aggression(p, r)  = count of (ATTACK Pn) emitted by p in round r
  D4(p) = max over q of (defenses(p,q) / max(1, rounds))
           defenses = DEFEND q + DISAGREE (when agreeing attacker on q) + NOT q WEREWOLF
           disabled for 5-player games (single wolf)
  D5(p) = contradicted(p) / max(1, total_claims(p))
           claims include COMINGOUT and DIVINED tokens

  E(p)  = a1*D1 + a2*D2 + a3*D3 + a4*D4 + a5*D5

Default weights (configurable, never hardcoded in logic):
  a1=0.30, a2=0.15, a3=0.20, a4=0.20, a5=0.15
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DetectorState:
    """
    Accumulates statistics for ONE player across all observed events.
    Updated by SuspicionModule.observe_*() methods.
    Evidence lists hold (day, description) tuples for trace logging.
    """
    # D1: Vote-Accusation Mismatch (ATTACK token vs actual vote)
    d1_rounds:     int = 0   # rounds where the player emitted an ATTACK token
    d1_mismatches: int = 0   # rounds where ATTACK target != actual vote target
    d1_evidence:   list = field(default_factory=list)  # (day, "ATTACK Px but voted Py")

    # D2: Bandwagon Index (AGREE-token based)
    d2_applicable: int = 0   # total AGREE tokens emitted
    d2_followed:   int = 0   # AGREE tokens that joined an existing majority accusation
    d2_evidence:   list = field(default_factory=list)  # (day, "AGREE Px — was majority")

    # D3: Pressure-Triggered Accusations
    # per-round series: pressure_on_p, aggression_by_p
    d3_suspicion_series: List[float] = field(default_factory=list)
    d3_accused_series:   List[float] = field(default_factory=list)
    d3_evidence:         list        = field(default_factory=list)

    # D4: Protection Patterns (DEFEND + DISAGREE-on-attack + NOT p WEREWOLF)
    d4_defends:  Dict[int, int] = field(default_factory=dict)  # {defended_pid: count}
    d4_rounds:   int = 0
    d4_evidence: list = field(default_factory=list)  # (day, token_text, defended_pid)

    # D5: Claim Consistency (COMINGOUT + DIVINED)
    d5_total_claims:        int = 0
    d5_contradicted_claims: int = 0
    d5_evidence:            list = field(default_factory=list)  # (claim_text, verdict)

    # Scratch: intended ATTACK target this round (cleared after vote observed)
    intended_attack: Optional[int] = None
    # Legacy alias kept for any external code that reads it
    intended_vote:   Optional[int] = None

    # Per-round ATTACK counts by this player (day -> count), for D3 aggression
    d3_round_attacks: Dict[int, int] = field(default_factory=dict)  # day -> count


def compute_d1(state: DetectorState) -> float:
    """D1: fraction of rounds where ATTACK target != actual vote."""
    if state.d1_rounds == 0:
        return 0.0
    return state.d1_mismatches / state.d1_rounds


def compute_d2(state: DetectorState) -> float:
    """D2: fraction of AGREE tokens that joined an existing majority accusation."""
    if state.d2_applicable == 0:
        return 0.0
    return state.d2_followed / state.d2_applicable


def compute_d3(state: DetectorState) -> float:
    """
    D3: pressure-triggered aggression score.

    n == 0  → 0.0 (no data)
    n == 1  → geometric mean of (pressure, aggression); 0 if either is zero.
    n == 2  → Pearson correlation (gives ±1; clamped to [0, 1])
    n >= 3  → Pearson correlation (full reliability)
    """
    xs = state.d3_suspicion_series
    ys = state.d3_accused_series
    n  = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    if n == 1:
        # Single round: use raw aggression (ATTACK count) as the signal.
        # Pressure correlation needs multiple rounds; for Day-1 the best
        # discriminator is simply whether the player is heavily attacking.
        return min(1.0, ys[0])
    # n >= 2: Pearson correlation
    mx = sum(xs[:n]) / n
    my = sum(ys[:n]) / n
    num   = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom = math.sqrt(
        sum((xs[i] - mx) ** 2 for i in range(n)) *
        sum((ys[i] - my) ** 2 for i in range(n))
    )
    if denom == 0:
        return 0.0
    return max(0.0, min(1.0, num / denom))


def compute_d4(state: DetectorState, single_wolf_game: bool = False) -> float:
    """
    D4: maximum consistent protection rate across all defended players.
    Disabled (returns 0) for single-wolf games (5-player).
    """
    if single_wolf_game or not state.d4_defends or state.d4_rounds == 0:
        return 0.0
    max_rate = max(state.d4_defends.values()) / state.d4_rounds
    return min(1.0, max_rate)


def compute_d5(state: DetectorState) -> float:
    """D5: fraction of COMINGOUT/DIVINED claims contradicted by revealed outcomes."""
    if state.d5_total_claims == 0:
        return 0.0
    return state.d5_contradicted_claims / state.d5_total_claims


def compute_E(
    d_vals: tuple,
    weights: dict,
) -> float:
    """
    Composite evidence score.
    d_vals = (d1, d2, d3, d4, d5)
    weights must contain keys 'a1'..'a5'.
    """
    d1, d2, d3, d4, d5 = d_vals
    return (
        weights["a1"] * d1 +
        weights["a2"] * d2 +
        weights["a3"] * max(d3, 0.0) +
        weights["a4"] * d4 +
        weights["a5"] * d5
    )
